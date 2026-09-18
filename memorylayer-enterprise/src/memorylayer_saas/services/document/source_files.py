# SPDX-License-Identifier: AGPL-3.0-only
"""Short-lived capabilities for canonical page bytes on the HTTP data plane.

The minting endpoint must authorize the owning workspace. Consumers revalidate
live task authority before exposing downloaded/cached files. No source copies
are written to another storage domain; canonical deletion/re-ingestion applies.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import re
import secrets
import time
from datetime import UTC, datetime
from urllib.parse import urlsplit

from fastapi import HTTPException
from PIL import Image

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_TEXT_BYTES = 20 * 1024 * 1024
CAP_TTL = 120
_PROCESS_KEY = secrets.token_bytes(32)


def _key(v):
    configured = v.environ("MEMORYLAYER_SOURCE_FILES_SIGNING_KEY", default="")
    if not configured:
        # A restart invalidates outstanding tickets. Replicas may share an
        # operator-provided key; never persist or log the process fallback.
        return _PROCESS_KEY
    if not isinstance(configured, str) or len(configured.encode()) < 32:
        raise HTTPException(503, "Source file signing key must contain at least 32 bytes")
    return configured.encode()


def _tenant(v):
    tenant = v.environ("MEMORYLAYER_TENANT_ID", default="")
    if not isinstance(tenant, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", tenant):
        raise HTTPException(503, "Source file tenant is not configured correctly")
    return tenant


def _fetch_base(v):
    value = str(v.environ("MEMORYLAYER_SOURCE_FILES_FETCH_URL", default="") or "").rstrip("/")
    url = urlsplit(value)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise HTTPException(503, "Source file downloads are not configured correctly")
    return value


async def _page_data(page, blob_service, kind):
    if kind == "image":
        if not page.image_storage_path:
            raise HTTPException(404, "Page image unavailable")
        data = await blob_service.retrieve_file(page.image_storage_path)
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(422, "Page image exceeds the supported size")
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in ("PNG", "JPEG") or image.width * image.height > 25_000_000:
                    raise ValueError("unsupported image")
                mime = Image.MIME[image.format]
                image.verify()
        except Exception as exc:
            raise HTTPException(422, "Invalid page image") from exc
    elif kind == "transcript":
        data = (page.transcript or "").encode("utf-8")
        if len(data) > MAX_TEXT_BYTES:
            raise HTTPException(422, "Page transcript exceeds the supported size")
        mime = "text/plain; charset=utf-8"
    else:
        raise HTTPException(422, "Unsupported source file kind")
    return data, mime


async def export_page(v, ctx, page, blob_service, kind: str) -> dict:
    fetch_url = _fetch_base(v)
    if ctx.tenant_id != _tenant(v) or not page.workspace_id:
        raise HTTPException(403, "Source file tenant/workspace scope required")
    data, mime = await _page_data(page, blob_service, kind)
    digest = hashlib.sha256(data).hexdigest()
    expires = int(time.time()) + CAP_TTL
    claims = {
        "v": 1,
        "tenant": ctx.tenant_id,
        "workspace": page.workspace_id,
        "document": page.document_id,
        "page": page.id,
        "kind": kind,
        "sha256": digest,
        "size": len(data),
        "expires": expires,
    }
    payload = base64.urlsafe_b64encode(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode()).rstrip(b"=")
    token = payload.decode() + "." + hmac.new(_key(v), payload, hashlib.sha256).hexdigest()
    return {
        "version": 1,
        "document_id": page.document_id,
        "page_id": page.id,
        "kind": kind,
        "mime": mime,
        "size_bytes": len(data),
        "sha256": digest,
        "url": fetch_url + "/blob/source-pages/" + ctx.tenant_id + "/" + token,
        "expires_at": datetime.fromtimestamp(expires, UTC).isoformat(),
    }


async def redeem_page(v, token: str, storage, blob_service):
    """Serve exactly the current canonical page named by an unexpired ticket."""
    try:
        if not isinstance(token, str) or len(token) > 2048:
            raise ValueError("invalid ticket size")
        payload, signature = token.split(".")
        if not hmac.compare_digest(signature, hmac.new(_key(v), payload.encode(), hashlib.sha256).hexdigest()):
            raise ValueError("invalid signature")
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if claims["v"] != 1 or claims["tenant"] != _tenant(v) or not time.time() < claims["expires"] <= time.time() + CAP_TTL:
            raise ValueError("expired ticket")
        page = await storage.get_page(claims["page"])
        if not page or page.document_id != claims["document"] or page.workspace_id != claims["workspace"]:
            raise ValueError("page no longer available")
        data, mime = await _page_data(page, blob_service, claims["kind"])
        if len(data) != claims["size"] or hashlib.sha256(data).hexdigest() != claims["sha256"]:
            raise ValueError("page changed")
    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError):
        raise HTTPException(404, "Source file unavailable") from None
    return data, mime
