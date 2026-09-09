# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""blobgw-backed blob storage service.

A drop-in alternative to the default fsspec-backed :class:`BlobStorageService`
that routes I/O through the **blobgw** object gateway (Layer 1 of the casstore
storage platform), gaining content-addressed dedup + compression across the
whole enterprise instance.

The path-convention methods are inherited unchanged from
:class:`BlobStorageService` — a deterministic storage path becomes a gateway
**ref** verbatim. Only the I/O methods change backend.

Select it by setting ``MEMORYLAYER_BLOB_STORAGE_SERVICE=blobgw``. The gateway
URL comes from ``MEMORYLAYER_BLOBGW_URL`` (default ``http://localhost:8080``).

This module never imports ``blobgw_client`` at top level: the client is
lazily imported inside the plugin's :meth:`initialize` so recursive plugin
discovery never fails when the (optional) client isn't installed.
"""
from __future__ import annotations

import asyncio
import mimetypes
import os
from datetime import datetime
from logging import Logger

from scitrera_app_framework import Variables

from . import BlobStoragePluginBase
from .blob_storage import BlobStorageService
from ...config import (
    MEMORYLAYER_BLOB_STORAGE_BASE_PATH,
    DEFAULT_MEMORYLAYER_BLOB_STORAGE_BASE_PATH,
)

# Local config key for the gateway URL (kept here to avoid touching the shared
# config module for an optional provider).
MEMORYLAYER_BLOBGW_URL = "MEMORYLAYER_BLOBGW_URL"
DEFAULT_MEMORYLAYER_BLOBGW_URL = "http://localhost:8080"

# Extensions whose content type we want to preserve for nicer downloads; the
# dedup/compression behavior is decided server-side by content class, not by
# this hint.
_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
}


def _content_type_for(path: str) -> str:
    _, ext = os.path.splitext(path)
    if ext in _CONTENT_TYPES:
        return _CONTENT_TYPES[ext]
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def _parse_rfc3339(value: str) -> float | None:
    """Parse an RFC3339 timestamp (incl. 9-digit nanoseconds + ``Z``) to epoch
    seconds, or None if unparseable. Python's fromisoformat rejects >6 fraction
    digits, so we truncate to microseconds first."""
    if not value:
        return None
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    # Truncate fractional seconds to 6 digits if present.
    if "." in v:
        head, frac = v.split(".", 1)
        # frac may carry a timezone suffix after the digits.
        digits = ""
        rest = ""
        for i, ch in enumerate(frac):
            if ch.isdigit():
                digits += ch
            else:
                rest = frac[i:]
                break
        v = f"{head}.{digits[:6]}{rest}"
    try:
        return datetime.fromisoformat(v).timestamp()
    except ValueError:
        return None


class BlobGWBlobStorageService(BlobStorageService):
    """:class:`BlobStorageService` whose I/O is backed by the blobgw gateway.

    Inherits every path-convention method; overrides the I/O surface to call
    the gateway client. Deterministic paths are used directly as refs (with any
    leading ``/`` stripped, since refs are not absolute).
    """

    def __init__(self, client, not_found_exc: type[Exception], base_path: str, logger: Logger):
        # Intentionally does not call super().__init__ (which requires an
        # fsspec filesystem). Only self._base_path / self.logger are needed by
        # the inherited path-convention methods.
        self._client = client
        self._not_found = not_found_exc
        self._base_path = base_path.rstrip("/")
        self.logger = logger

    @staticmethod
    def _ref(path: str) -> str:
        """A storage path used directly as a gateway ref (refs aren't absolute)."""
        return path.lstrip("/")

    def _dir_prefix(self, prefix: str) -> str:
        """Normalize a directory prefix to a ref prefix ending in '/'."""
        return self._ref(prefix).rstrip("/") + "/"

    # === Async I/O Operations (override fsspec backend) ===

    async def store_file(self, path: str, data: bytes) -> str:
        self.logger.debug("blobgw: storing %d bytes at %s", len(data), path)
        ref = self._ref(path)
        await asyncio.to_thread(self._client.put, ref, data, _content_type_for(path))
        return path

    async def retrieve_file(self, path: str) -> bytes:
        self.logger.debug("blobgw: retrieving %s", path)
        try:
            return await asyncio.to_thread(self._client.get, self._ref(path))
        except self._not_found:
            raise FileNotFoundError(path) from None

    async def delete_tree(self, prefix: str) -> None:
        self.logger.debug("blobgw: deleting tree at %s", prefix)

        def _delete() -> None:
            ref = self._ref(prefix)
            dir_prefix = ref.rstrip("/") + "/"
            # Everything under the directory, plus an exact object at the prefix.
            refs = {o.ref for o in self._client.list(dir_prefix)}
            refs.update(o.ref for o in self._client.list(ref) if o.ref == ref)
            for r in refs:
                self._client.delete(r)

        await asyncio.to_thread(_delete)

    async def exists(self, path: str) -> bool:
        return await asyncio.to_thread(self._client.exists, self._ref(path))

    async def list_dir(self, path: str) -> list[str]:
        """Immediate children of a directory, as full paths.

        blobgw lists by prefix (all descendants); we collapse to the immediate
        child segment to match the fsspec ``ls`` semantics callers expect.
        """

        def _ls() -> list[str]:
            prefix = self._dir_prefix(path)
            children: set[str] = set()
            for obj in self._client.list(prefix):
                remainder = obj.ref[len(prefix):]
                if not remainder:
                    continue
                first = remainder.split("/", 1)[0]
                # Re-attach the original (possibly absolute) path style.
                children.add(path.rstrip("/") + "/" + first)
            return sorted(children)

        return await asyncio.to_thread(_ls)

    async def iter_document_dirs(self) -> list[tuple[str, str, str]]:
        """Enumerate ``{base}/*/documents/*`` document directories from refs."""

        def _walk() -> list[tuple[str, str, str]]:
            base_ref = self._ref(self._base_path).rstrip("/")
            prefix = base_ref + "/" if base_ref else ""
            seen: set[tuple[str, str]] = set()
            results: list[tuple[str, str, str]] = []
            for obj in self._client.list(prefix):
                rel = obj.ref[len(prefix):]
                parts = rel.split("/")
                # Expect <workspace>/documents/<doc>/...
                if len(parts) < 3 or parts[1] != "documents":
                    continue
                workspace_id, doc_id = parts[0], parts[2]
                if (workspace_id, doc_id) in seen:
                    continue
                seen.add((workspace_id, doc_id))
                full_path = f"{self._base_path}/{workspace_id}/documents/{doc_id}"
                results.append((workspace_id, doc_id, full_path))
            return results

        return await asyncio.to_thread(_walk)

    async def newest_mtime(self, path: str) -> float | None:
        """Newest object creation time under a prefix, in epoch seconds."""

        def _newest() -> float | None:
            newest: float | None = None
            for obj in self._client.list(self._dir_prefix(path)):
                ts = _parse_rfc3339(obj.created_at)
                if ts is None:
                    continue
                if newest is None or ts > newest:
                    newest = ts
            return newest

        return await asyncio.to_thread(_newest)


class BlobGWBlobStoragePlugin(BlobStoragePluginBase):
    """Plugin exposing the blobgw-backed blob storage service.

    Auto-discovered by recursive plugin registration; activated when
    ``MEMORYLAYER_BLOB_STORAGE_SERVICE=blobgw``.
    """

    PROVIDER_NAME = "blobgw"

    def initialize(self, v: Variables, logger: Logger) -> BlobGWBlobStorageService:
        # Lazy import so plugin discovery never depends on blobgw_client being
        # installed unless this provider is actually selected.
        from blobgw_client import BlobGWClient, NotFound

        base_url = v.environ(MEMORYLAYER_BLOBGW_URL, default=DEFAULT_MEMORYLAYER_BLOBGW_URL)
        base_path = v.environ(
            MEMORYLAYER_BLOB_STORAGE_BASE_PATH,
            default=DEFAULT_MEMORYLAYER_BLOB_STORAGE_BASE_PATH,
        )
        client = BlobGWClient(base_url)
        logger.info("Initialized blobgw blob storage: url=%s, base_path=%s", base_url, base_path)
        return BlobGWBlobStorageService(
            client=client, not_found_exc=NotFound, base_path=base_path, logger=logger,
        )

    def on_registration(self, v: Variables) -> None:
        super().on_registration(v)
        v.set_default_value(MEMORYLAYER_BLOBGW_URL, DEFAULT_MEMORYLAYER_BLOBGW_URL)
