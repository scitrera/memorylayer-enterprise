# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Dropbox connector — syncs files from Dropbox.

Discovers files via the Dropbox Python SDK, computes content hashes, and
returns upstream temporary-link URLs for content fetch.

``get_content_url`` returns an upstream Dropbox temporary link — no blob
materialization needed.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from data_connectors.connectors import ConnectorRegistry
from data_connectors.vfs.catalog import VfsCatalog

logger = logging.getLogger(__name__)

CONNECTOR_TYPE = "dropbox"


def _build_client(credentials: dict):
    """Create a Dropbox client from connector credentials.

    Returns a ``dropbox.Dropbox`` instance.
    """
    # Delayed import: dropbox SDK is only needed when this connector runs
    import dropbox
    access_token = credentials.get("access_token")
    if not access_token:
        raise ValueError("dropbox credentials must include 'access_token'")
    return dropbox.Dropbox(access_token)


def _is_indexable_file(name: str) -> bool:
    """Heuristic check for file extensions worth indexing."""
    indexable_exts = {
        ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm",
        ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log", ".py",
        ".js", ".ts", ".java", ".c", ".cpp", ".h", ".rs", ".go",
        ".rb", ".sh", ".bat", ".ps1", ".sql", ".r", ".tex",
        ".pdf", ".docx", ".xlsx", ".pptx", ".doc", ".xls",
    }
    import os
    _, ext = os.path.splitext(name.lower())
    return ext in indexable_exts


class DropboxConnector:
    """Connector for Dropbox files.

    Args:
        credentials: Dict with ``access_token``.
        settings: Dict with optional key ``path`` (folder path to scan).
        catalog: VFS catalog for resolving vfs_ref entries.
    """

    def __init__(
        self,
        credentials: dict,
        settings: dict | None = None,
        catalog: VfsCatalog | None = None,
    ) -> None:
        self._credentials = credentials
        self._settings = settings or {}
        self._catalog = catalog

    async def load(self) -> None:
        """Validate credentials by fetching current account info."""
        import asyncio

        dbx = _build_client(self._credentials)
        await asyncio.get_running_loop().run_in_executor(None, dbx.users_get_current_account)
        logger.info("DropboxConnector loaded (path=%s)", self._settings.get("path", "/"))

    async def poll(self) -> list[dict[str, Any]]:
        """Recursively list files in the configured Dropbox path.

        Each entry includes source_path, content_hash, content_type,
        and size_bytes.  ``content_hash`` is derived from the Dropbox
        ``content_hash`` field (available on FileMetadata).
        """
        import asyncio
        # Delayed import: dropbox types only needed at runtime
        import dropbox as dbx_module

        dbx = _build_client(self._credentials)
        path = self._settings.get("path", "")
        entries: list[dict[str, Any]] = []

        def _list(cursor=None):
            if cursor:
                return dbx.files_list_folder_continue(cursor)
            return dbx.files_list_folder(path, recursive=True)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _list)

        while True:
            for entry in result.entries:
                if isinstance(entry, dbx_module.files.FileMetadata):
                    if not _is_indexable_file(entry.name):
                        continue

                    # Dropbox content_hash is a reliable change-detection hash
                    dbx_hash = entry.content_hash or ""
                    content_hash = hashlib.sha256(
                        (dbx_hash or f"{entry.id}:{entry.server_modified}").encode()
                    ).hexdigest()

                    modified = entry.server_modified.timestamp() if entry.server_modified else 0

                    entries.append({
                        "source_path": entry.path_display or entry.name,
                        "content_hash": content_hash,
                        "content_type": None,  # Dropbox doesn't provide MIME in list
                        "size_bytes": entry.size,
                        "metadata": {
                            "dropbox_id": entry.id,
                            "dropbox_path": entry.path_lower or "",
                            "dropbox_content_hash": dbx_hash,
                            "dropbox_modified": modified,
                        },
                    })

            if not result.has_more:
                break
            result = await loop.run_in_executor(None, _list, result.cursor)

        logger.info("DropboxConnector polled %d files from %s", len(entries), path or "/")
        return entries

    async def get_content_url(self, vfs_ref: str) -> Optional[str]:
        """Return an upstream Dropbox temporary link for the file.

        Dropbox ``get_temporary_link`` returns a short-lived direct download
        URL — no blob materialization needed.  Per the "always upstream URLs"
        decision.
        """
        import asyncio

        if self._catalog is None:
            return None
        entry = await self._catalog.get(vfs_ref)
        if entry is None:
            return None

        dropbox_path = entry.metadata.get("dropbox_path")
        if not dropbox_path:
            return None

        dbx = _build_client(self._credentials)

        def _get_link():
            result = dbx.files_get_temporary_link(dropbox_path)
            return result.link

        return await asyncio.get_running_loop().run_in_executor(None, _get_link)

    async def get_metadata(self, vfs_ref: str) -> dict[str, Any]:
        """Return metadata for a Dropbox file."""
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


ConnectorRegistry.register(CONNECTOR_TYPE, DropboxConnector)
