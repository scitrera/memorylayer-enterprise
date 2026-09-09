"""Manual upload connector.

Handles user-initiated file uploads via presigned URLs.  This connector
does not poll — entries are registered when the upload-complete callback
fires (triggered by the frontend after a successful PUT to the presigned URL).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from data_connectors.connectors import ConnectorRegistry
from data_connectors.vfs.blob_store import BlobStore
from data_connectors.vfs.catalog import VfsCatalog

logger = logging.getLogger(__name__)

CONNECTOR_TYPE = "manual_upload"


class ManualUploadConnector:
    """Connector for user-initiated file uploads.

    Unlike polling connectors, this one is event-driven: the frontend
    uploads to a presigned URL, then calls back to register the entry.
    ``poll()`` always returns an empty list.
    """

    def __init__(self, blob_store: BlobStore, catalog: VfsCatalog) -> None:
        self._blob_store = blob_store
        self._catalog = catalog

    async def load(self) -> None:
        """No-op for manual uploads — no credentials to validate."""
        logger.debug("ManualUploadConnector loaded")

    async def poll(self) -> list[dict[str, Any]]:
        """Manual uploads are event-driven; poll always returns empty."""
        return []

    async def get_content_url(self, vfs_ref: str) -> Optional[str]:
        """Return a download URL for an uploaded file."""
        entry = await self._catalog.get(vfs_ref)
        if entry is None or not entry.blob_key:
            return None
        url, _, _ = await self._blob_store.generate_download_url(entry.blob_key)
        return url

    async def get_metadata(self, vfs_ref: str) -> dict[str, Any]:
        """Return metadata for an uploaded file."""
        entry = await self._catalog.get(vfs_ref)
        if entry is None:
            return {}
        return {
            "source_path": entry.source_path,
            "content_type": entry.content_type,
            "size_bytes": entry.size_bytes,
            "connector_type": CONNECTOR_TYPE,
            **entry.metadata,
        }

    async def register_upload_complete(
        self,
        workspace_id: str,
        blob_key: str,
        filename: str,
        content_hash: str,
        content_type: Optional[str] = None,
        size_bytes: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Register a completed upload as a VFS entry.

        Called after the frontend successfully PUTs to the presigned URL.

        Returns:
            The vfs_ref of the registered entry.
        """
        entry = await self._catalog.register(
            workspace_id=workspace_id,
            connector_id=CONNECTOR_TYPE,
            source_path=filename,
            content_hash=content_hash,
            content_type=content_type,
            size_bytes=size_bytes,
            blob_key=blob_key,
            metadata=metadata,
        )
        logger.info("Registered manual upload: vfs_ref=%s, filename=%s", entry.vfs_ref, filename)
        return entry.vfs_ref


ConnectorRegistry.register(CONNECTOR_TYPE, ManualUploadConnector)
