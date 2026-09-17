# SPDX-License-Identifier: AGPL-3.0-only
"""Export canonical page files to the existing HTTP storage data plane.

Callers must authorize the owning workspace before calling this module. Returned
capabilities are short lived, secret transport metadata, never model text. The
export is a derived document file (not a VFS entry or a new ingestion job).
"""

from __future__ import annotations

import hashlib
import io
from urllib.parse import quote, urlsplit

import httpx
from fastapi import HTTPException
from PIL import Image

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_TEXT_BYTES = 20 * 1024 * 1024
CAP_TTL = 120


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def document_prefix(workspace: str, document: str) -> str:
    return f"source-files/{_digest(workspace)}/{_digest(document)}/"


def settings(v):
    keys = ("MEMORYLAYER_SOURCE_FILES_BLOB_URL", "MEMORYLAYER_SOURCE_FILES_EDGE_URL", "MEMORYLAYER_SOURCE_FILES_FETCH_URL")
    values = tuple(str(v.environ(k, default="") or "").rstrip("/") for k in keys)
    if not all(values):
        raise HTTPException(503, "Source file downloads are not configured")
    for value in values:
        url = urlsplit(value)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise HTTPException(503, "Invalid source file download configuration")
    return values


async def export_page(v, ctx, page, blob_service, kind: str) -> dict:
    blob_url, edge_url, fetch_url = settings(v)
    if not ctx.tenant_id or not page.workspace_id:
        raise HTTPException(403, "Source file scope required")
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
    digest = hashlib.sha256(data).hexdigest()
    # One stable ref per document/page/kind. Re-ingestion replaces the ref;
    # consumers still reject any bytes that do not match their snapshot/hash.
    ref = document_prefix(page.workspace_id, page.document_id) + _digest(page.id) + "/" + kind
    async with httpx.AsyncClient(timeout=30, follow_redirects=False, trust_env=False) as client:
        response = await client.put(
            blob_url + "/v1/objects/" + quote(ref, safe="/"), headers={"X-Blobgw-Domain": ctx.tenant_id, "Content-Type": mime}, content=data
        )
        response.raise_for_status()
        response = await client.post(
            edge_url + "/capabilities",
            headers={"X-Auth-Tenant-ID": ctx.tenant_id, "X-Scitrera-User": "memorylayer"},
            json={"op": "GET", "ref": ref, "ttl_seconds": CAP_TTL, "require_auth": False},
        )
        response.raise_for_status()
        cap = response.json()
    parsed = urlsplit(cap["capability_url"])
    # Retain only the storage server's path/query, never a supplied origin.
    if not parsed.path.startswith("/") or parsed.fragment:
        raise HTTPException(502, "Invalid storage capability")
    return {
        "version": 1,
        "document_id": page.document_id,
        "page_id": page.id,
        "kind": kind,
        "mime": mime,
        "size_bytes": len(data),
        "sha256": digest,
        "url": fetch_url + parsed.path + "?" + parsed.query,
        "expires_at": cap["expires_at"],
    }


async def delete_document_files(v, tenant: str, workspace: str, document: str) -> None:
    """Remove derived HTTP exports before deleting the canonical document."""
    configured = v.environ("MEMORYLAYER_SOURCE_FILES_BLOB_URL", default="")
    if not isinstance(configured, str) or not configured:
        return
    blob_url, _, _ = settings(v)
    if not tenant or not workspace or not document:
        raise ValueError("Source export deletion requires tenant/workspace/document")
    async with httpx.AsyncClient(timeout=30, follow_redirects=False, trust_env=False) as client:
        response = await client.delete(
            blob_url + "/v1/objects", headers={"X-Blobgw-Domain": tenant}, params={"prefix": document_prefix(workspace, document)}
        )
        response.raise_for_status()
