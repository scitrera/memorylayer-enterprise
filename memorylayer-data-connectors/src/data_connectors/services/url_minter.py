# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""URL minting service.

Generates presigned URLs for upload, download, and JIT fetch operations.
Per the "always upstream URLs" decision, fetch URLs are minted directly
from the blob store — no proxying through data-connectors at fetch time.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import uuid4

from data_connectors.vfs.blob_store import BlobStore
from data_connectors.vfs.catalog import VfsCatalog, VfsEntry

logger = logging.getLogger(__name__)

# Default TTLs
_UPLOAD_TTL_S = 3600
_DOWNLOAD_TTL_S = 3600
_FETCH_TTL_S = 900  # Shorter for JIT fetch (worker uses immediately)


class UrlMinter:
    """Mints presigned URLs backed by the blob store.

    Upload URLs come with a pre-allocated blob key. Download and fetch URLs
    resolve the blob key from the VFS catalog and return upstream presigned
    URLs directly (no proxy).
    """

    def __init__(self, blob_store: BlobStore, catalog: VfsCatalog) -> None:
        self._blob_store = blob_store
        self._catalog = catalog

    async def mint_upload_url(
        self,
        workspace_id: str,
        filename: str,
        content_type: Optional[str] = None,
        connector_id: str = "manual_upload",
        method: str = "POST",
        source_path: Optional[str] = None,
        size_bytes: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Mint a presigned upload URL.

        Allocates a blob key, **pre-registers a placeholder VFS entry**, and
        returns either a POST-policy form or a PUT URL per the ``method``
        argument. POST is the default since the browser client uses multipart
        FormData.

        The pre-registration step matters: callers (frontend chat composer,
        agent attachments) need a stable ``vfs_ref`` they can include in
        downstream messages BEFORE the upload completes. Without it, the
        cowork agent's metadata resolution and ``vfs_fetch`` calls fall back
        to the raw blob_key, which the catalog has no record of and causes
        ``get_vfs_entry`` to hang/404.

        ``content_hash`` is empty at registration time; the post-upload
        ``finalize`` endpoint fills it in (and derives ``size_bytes`` when not
        supplied here) from the actual S3 object metadata.

        ``source_path`` defaults to ``filename``. Pass it explicitly to record a
        grouping path (e.g. '/Bids/acme/report.pdf') that ``list_entries``
        can filter on via ``source_path_prefix`` — filename stays the display
        name, which flows downstream into the ingested document.

        ``size_bytes`` and ``metadata`` were previously accepted by the request
        model and then silently discarded here, so a caller's metadata never
        reached the entry and had to be re-sent at finalize. They are now
        persisted at registration.

        Returns:
            Dict from BlobStore.generate_upload_url augmented with ``blob_key``
            and ``vfs_ref``. Keys: ``method``, ``url``, ``fields``, ``headers``,
            ``expires_at``, ``blob_key``, ``vfs_ref``.
        """
        blob_key = f"{workspace_id}/{connector_id}/{uuid4().hex[:12]}/{filename}"
        result = await self._blob_store.generate_upload_url(
            key=blob_key,
            content_type=content_type,
            ttl_seconds=_UPLOAD_TTL_S,
            method=method,
        )
        result["blob_key"] = blob_key

        # Pre-register the VFS entry so the caller has a real vfs_ref.
        entry = await self._catalog.register(
            workspace_id=workspace_id,
            connector_id=connector_id,
            source_path=source_path or filename,
            content_hash="",  # filled in by the post-upload finalize
            content_type=content_type,
            size_bytes=size_bytes,
            blob_key=blob_key,
            metadata=metadata or {},
        )
        result["vfs_ref"] = entry.vfs_ref

        logger.debug(
            "Minted upload URL for blob_key=%s vfs_ref=%s (workspace=%s, method=%s)",
            blob_key, entry.vfs_ref, workspace_id, result["method"],
        )
        return result

    async def mint_download_url(
        self, vfs_ref: str, workspace_id: str, subject: Optional[str] = None
    ) -> tuple[str, dict[str, str], datetime]:
        """Mint a presigned download URL for a VFS entry.

        ``subject`` is the end-user identity the download token should bind to
        (the value auth-go stamps as ``X-Scitrera-User``). It is forwarded to the
        blob store so an auth-bound token's subject equals that user; absent it,
        the mint falls back to the dc service id (back-compat).

        Returns:
            Tuple of (url, headers, expires_at).

        Raises:
            ValueError: If the VFS entry is not found or has no blob key.
        """
        entry = await self._catalog.get(vfs_ref)
        if entry is None:
            raise ValueError(f"VFS entry not found: {vfs_ref}")
        # TODO: this workspace check is not real security; it "hinted" at security but since workspace_id is
        #       an input, then it's just a check that the caller knows the workspace_id. Real security should
        #       come with OBO and authz check against memorylayer. (Also, cross-workspace content is OK if subject has rights.)
        # if entry.workspace_id != workspace_id:
        #     raise ValueError(f"VFS entry {vfs_ref} does not belong to workspace {workspace_id}")
        if not entry.blob_key:
            raise ValueError(f"VFS entry {vfs_ref} has no blob key")

        url, headers, expires_at = await self._blob_store.generate_download_url(
            key=entry.blob_key,
            ttl_seconds=_DOWNLOAD_TTL_S,
            subject=subject,
        )
        return url, headers, expires_at

    async def mint_fetch_url(self, vfs_ref: str, workspace_id: str) -> tuple[str, dict[str, str], datetime]:
        """Mint a JIT fetch URL for a VFS entry (used by IN-CLUSTER server-side
        fetchers: MemoryLayer workers and the sahara harness).

        Unlike :meth:`mint_download_url` (browser-facing: external/public,
        auth-bound), this returns an INTERNAL edge URL (``blobgw-edge.storage.svc``)
        with a **bearer** capability and a shorter TTL (the caller fetches
        immediately). The external ingress that fronts the public download URL
        strips inbound identity and re-stamps from a session; a server-side fetcher
        has none, so it must use the internal edge, which is network-isolated.

        Returns:
            Tuple of (url, headers, expires_at).

        Raises:
            ValueError: If the VFS entry is not found or has no blob key.
        """
        entry = await self._catalog.get(vfs_ref)
        if entry is None:
            raise ValueError(f"VFS entry not found: {vfs_ref}")
        if entry.workspace_id != workspace_id:
            raise ValueError(f"VFS entry {vfs_ref} does not belong to workspace {workspace_id}")
        if not entry.blob_key:
            raise ValueError(f"VFS entry {vfs_ref} has no blob key")

        url, headers, expires_at = await self._blob_store.generate_fetch_url(
            key=entry.blob_key,
            ttl_seconds=_FETCH_TTL_S,
        )
        return url, headers, expires_at
