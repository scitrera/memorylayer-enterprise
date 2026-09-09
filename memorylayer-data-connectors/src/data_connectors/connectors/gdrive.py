"""Google Drive connector — syncs files from Google Drive.

Discovers documents via the Drive v3 API, computes content hashes from
file metadata, and returns upstream OAuth-bearer URLs for content fetch.
Supports both service-account and user-OAuth authentication.

``get_content_url`` returns an upstream Google Drive download URL with
the user's OAuth token — no blob materialization needed.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from data_connectors.connectors import ConnectorRegistry
from data_connectors.vfs.catalog import VfsCatalog

logger = logging.getLogger(__name__)

CONNECTOR_TYPE = "gdrive"

# MIME types that Google can export (native Docs/Sheets/Slides)
_EXPORT_MIME_MAP: dict[str, str] = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}


def _build_service(credentials: dict):
    """Build a Google Drive API v3 service resource.

    Returns a ``googleapiclient.discovery.Resource`` for Drive v3.
    """
    # Delayed import: google-api-python-client and google-auth are heavy
    # dependencies only needed when this connector is actually used.
    from google.oauth2.credentials import Credentials as OAuthCredentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from googleapiclient.discovery import build

    if "service_account_json" in credentials:
        import json
        info = (
            json.loads(credentials["service_account_json"])
            if isinstance(credentials["service_account_json"], str)
            else credentials["service_account_json"]
        )
        creds = ServiceAccountCredentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
    elif "oauth_token" in credentials:
        creds = OAuthCredentials(token=credentials["oauth_token"])
    else:
        raise ValueError("gdrive credentials must include 'service_account_json' or 'oauth_token'")

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _file_query(settings: dict, time_filter: str | None = None) -> str:
    """Build a Drive files.list ``q`` parameter."""
    clauses: list[str] = ["trashed = false"]
    if "folder_id" in settings:
        clauses.append(f"'{settings['folder_id']}' in parents")
    if time_filter:
        clauses.append(time_filter)
    mime_types: list[str] | None = settings.get("mime_types")
    if mime_types:
        mime_clause = " or ".join(f"mimeType = '{m}'" for m in mime_types)
        clauses.append(f"({mime_clause})")
    return " and ".join(clauses)


def _parse_rfc3339(s: str) -> float:
    """Parse an RFC-3339 timestamp to epoch seconds, returning 0 on failure."""
    if not s:
        return 0
    try:
        from datetime import datetime
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0


class GoogleDriveConnector:
    """Connector for Google Drive files.

    Args:
        credentials: Dict with ``service_account_json`` (dict or JSON string)
            or ``oauth_token`` (string).
        settings: Dict with optional keys ``folder_id``, ``mime_types``.
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
        """Validate credentials by performing a lightweight files.list call."""
        import asyncio

        service = _build_service(self._credentials)

        def _check():
            service.files().list(
                q="trashed = false",
                fields="files(id)",
                pageSize=1,
            ).execute()

        await asyncio.get_running_loop().run_in_executor(None, _check)
        logger.info("GoogleDriveConnector loaded (folder_id=%s)", self._settings.get("folder_id"))

    async def poll(self) -> list[dict[str, Any]]:
        """List files in the configured Drive folder and return entry dicts.

        Each entry includes source_path, content_hash, content_type, and
        size_bytes.  ``content_hash`` is derived from the Drive file's
        ``md5Checksum`` (for binary files) or ``modifiedTime`` (for
        Google-native docs which have no md5).
        """
        import asyncio

        service = _build_service(self._credentials)
        q = _file_query(self._settings)
        entries: list[dict[str, Any]] = []
        page_token: str | None = None

        while True:
            def _list_page(pt=page_token):
                return service.files().list(
                    q=q,
                    fields=(
                        "nextPageToken, files(id, name, mimeType, modifiedTime, "
                        "md5Checksum, size, webViewLink, owners(displayName,emailAddress,permissionId))"
                    ),
                    pageSize=100,
                    pageToken=pt,
                ).execute()

            result = await asyncio.get_running_loop().run_in_executor(None, _list_page)

            for f in result.get("files", []):
                file_id = f["id"]
                md5 = f.get("md5Checksum", "")
                if md5:
                    content_hash = hashlib.sha256(md5.encode()).hexdigest()
                else:
                    # Google-native docs have no md5; use modifiedTime as change signal
                    content_hash = hashlib.sha256(
                        f"{file_id}:{f.get('modifiedTime', '')}".encode()
                    ).hexdigest()

                entries.append({
                    "source_path": f.get("name", file_id),
                    "content_hash": content_hash,
                    "content_type": f.get("mimeType"),
                    "size_bytes": int(f["size"]) if f.get("size") else None,
                    "metadata": {
                        "gdrive_file_id": file_id,
                        "gdrive_mime_type": f.get("mimeType", ""),
                        "gdrive_web_view_link": f.get("webViewLink", ""),
                        "gdrive_modified_time": f.get("modifiedTime", ""),
                        "owners": f.get("owners") or [],
                    },
                })

            page_token = result.get("nextPageToken")
            if not page_token:
                break

        logger.info("GoogleDriveConnector polled %d files", len(entries))
        return entries

    async def get_content_url(self, vfs_ref: str) -> Optional[str]:
        """Return an upstream Google Drive download URL with OAuth bearer token.

        For binary files, returns the Drive ``/files/{id}?alt=media`` URL.
        For Google-native docs, returns the export URL with the appropriate
        MIME type.  The caller must include the ``Authorization: Bearer ...``
        header from ``get_metadata`` when fetching.

        Per the "always upstream URLs" decision, no blob materialization occurs.
        """
        if self._catalog is None:
            return None
        entry = await self._catalog.get(vfs_ref)
        if entry is None:
            return None

        file_id = entry.metadata.get("gdrive_file_id")
        if not file_id:
            return None

        mime = entry.metadata.get("gdrive_mime_type", "")
        if mime in _EXPORT_MIME_MAP:
            export_mime = _EXPORT_MIME_MAP[mime]
            return f"https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType={export_mime}"
        return f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"

    async def get_metadata(self, vfs_ref: str) -> dict[str, Any]:
        """Return metadata for a Google Drive file.

        Includes ``authorization_header`` with the Bearer token so callers
        can authenticate when fetching the upstream URL.
        """
        if self._catalog is None:
            return {"connector_type": CONNECTOR_TYPE}
        entry = await self._catalog.get(vfs_ref)
        if entry is None:
            return {"connector_type": CONNECTOR_TYPE}

        token = self._credentials.get("oauth_token", "")
        meta: dict[str, Any] = {
            "connector_type": CONNECTOR_TYPE,
            "source_path": entry.source_path,
            "content_type": entry.content_type,
            "size_bytes": entry.size_bytes,
            **entry.metadata,
        }
        if token:
            meta["authorization_header"] = f"Bearer {token}"
        return meta


ConnectorRegistry.register(CONNECTOR_TYPE, GoogleDriveConnector)
