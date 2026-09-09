# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Local filesystem connector -- syncs files from a local directory.

Walks a base directory recursively, computes SHA256 content hashes,
uploads file bytes to the data-connectors blob store, and returns
presigned blob URLs for content fetch.  Mirrors the materialize-then-
presign pattern from ``slack.py`` / ``discord.py``.

Ported from the legacy ``LocalFileSystemProvider`` at
``backend/scitrera_app_api/data_providers/local.py``.
"""
from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Optional

from data_connectors.connectors import ConnectorRegistry
from data_connectors.vfs.blob_store import BlobStore
from data_connectors.vfs.catalog import VfsCatalog

logger = logging.getLogger(__name__)

CONNECTOR_TYPE = "local_fs"

# Read files in 256 KiB chunks for SHA256 hashing (matches legacy provider)
_HASH_CHUNK_SIZE = 262144


def _compute_sha256(file_path: Path) -> str:
    """Compute the SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def _guess_content_type(file_path: Path) -> str:
    """Best-effort MIME type from file extension."""
    mime, _ = mimetypes.guess_type(str(file_path))
    return mime or "application/octet-stream"


class LocalFsConnector:
    """Connector that syncs files from a local directory.

    Args:
        base_directory: Root directory to scan.
        catalog: VFS catalog for resolving vfs_ref entries.
        blob_store: Blob store for materializing file content.
    """

    def __init__(
        self,
        base_directory: str,
        catalog: VfsCatalog | None = None,
        blob_store: BlobStore | None = None,
    ) -> None:
        self._base_directory = Path(base_directory).resolve()
        self._catalog = catalog
        self._blob_store = blob_store

    async def load(self) -> None:
        """Validate that the base directory exists and is a directory."""
        if not self._base_directory.exists():
            raise FileNotFoundError(
                f"Base directory does not exist: {self._base_directory}"
            )
        if not self._base_directory.is_dir():
            raise NotADirectoryError(
                f"Base path is not a directory: {self._base_directory}"
            )
        logger.info("LocalFsConnector loaded: base_directory=%s", self._base_directory)

    async def poll(self) -> list[dict[str, Any]]:
        """Walk the base directory recursively and return entry dicts.

        For each file, computes SHA256 and uploads bytes to the blob store
        (materialize-then-presign pattern).  Each entry includes source_path,
        content_hash, content_type, size_bytes, and blob_key.
        """
        entries: list[dict[str, Any]] = []

        for root, _dirs, files in os.walk(self._base_directory):
            for filename in files:
                file_path = Path(root) / filename
                if not file_path.is_file():
                    continue

                # Relative path from the base directory
                rel_path = str(file_path.relative_to(self._base_directory))
                content_hash = _compute_sha256(file_path)
                content_type = _guess_content_type(file_path)
                size_bytes = file_path.stat().st_size

                # Upload to blob store if available (materialize-then-presign)
                blob_key: str | None = None
                if self._blob_store is not None:
                    blob_key = f"local_fs/{content_hash[:8]}/{filename}"
                    file_bytes = file_path.read_bytes()
                    await self._blob_store.put_object(
                        blob_key, file_bytes, content_type=content_type
                    )

                entries.append({
                    "source_path": rel_path,
                    "content_hash": content_hash,
                    "content_type": content_type,
                    "size_bytes": size_bytes,
                    "blob_key": blob_key,
                    "metadata": {
                        "local_fs_base": str(self._base_directory),
                        "local_fs_abs_path": str(file_path),
                    },
                })

        logger.info(
            "LocalFsConnector polled %d files from %s",
            len(entries), self._base_directory,
        )
        return entries

    async def get_content_url(self, vfs_ref: str) -> Optional[str]:
        """Return a presigned download URL from the blob store.

        Local files are materialized into the blob store during poll(),
        so content fetch always goes through the blob store presigned URL.
        """
        if self._catalog is None or self._blob_store is None:
            return None
        entry = await self._catalog.get(vfs_ref)
        if entry is None or not entry.blob_key:
            return None
        url, _, _ = await self._blob_store.generate_download_url(entry.blob_key)
        return url

    async def get_metadata(self, vfs_ref: str) -> dict[str, Any]:
        """Return metadata for a local filesystem entry."""
        if self._catalog is None:
            return {"connector_type": CONNECTOR_TYPE}
        entry = await self._catalog.get(vfs_ref)
        if entry is None:
            return {"connector_type": CONNECTOR_TYPE}
        return {
            "connector_type": CONNECTOR_TYPE,
            "source_path": entry.source_path,
            "content_type": entry.content_type,
            "size_bytes": entry.size_bytes,
            **entry.metadata,
        }


ConnectorRegistry.register(CONNECTOR_TYPE, LocalFsConnector)
