# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Microsoft Teams connector — syncs channel messages via Microsoft Graph API.

Discovers messages via Azure AD client-credentials OAuth, materializes
text content into the data-connectors blob store, and returns presigned
blob URLs for content fetch (Teams has no native presigned-URL mechanism
for message text).
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from data_connectors.connectors import ConnectorRegistry
from data_connectors.vfs.blob_store import BlobStore
from data_connectors.vfs.catalog import VfsCatalog

logger = logging.getLogger(__name__)

CONNECTOR_TYPE = "teams"

_GRAPH_BASE = "https://graph.microsoft.com/v1.0"


async def _get_access_token(credentials: dict) -> str:
    """Obtain an Azure AD access token via client credentials flow.

    Requires ``client_id``, ``client_secret``, and ``tenant_id`` in *credentials*.
    """
    # Delayed import: httpx is only needed when this connector is actually used
    import httpx

    client_id = credentials.get("client_id")
    client_secret = credentials.get("client_secret")
    tenant_id = credentials.get("tenant_id")
    if not all([client_id, client_secret, tenant_id]):
        raise ValueError(
            "teams credentials must include 'client_id', 'client_secret', and 'tenant_id'"
        )

    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://graph.microsoft.com/.default",
        })
        resp.raise_for_status()
        return resp.json()["access_token"]


async def _graph_get(http, path: str, params: dict | None = None) -> dict:
    """Execute a GET request against the Microsoft Graph API."""
    resp = await http.get(f"{_GRAPH_BASE}{path}", params=params)
    resp.raise_for_status()
    return resp.json()


def _parse_iso(s: str) -> float:
    """Parse an ISO-8601 timestamp to epoch seconds."""
    if not s:
        return 0
    try:
        from datetime import datetime
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0


class TeamsConnector:
    """Connector for Microsoft Teams channel messages.

    Args:
        credentials: Dict with ``client_id``, ``client_secret``, ``tenant_id``.
        settings: Dict with ``team_id`` and optional ``channel_ids`` (list).
        blob_store: Blob store for materializing text content.
        catalog: VFS catalog for resolving vfs_ref entries.
    """

    def __init__(
        self,
        credentials: dict,
        settings: dict | None = None,
        blob_store: BlobStore | None = None,
        catalog: VfsCatalog | None = None,
    ) -> None:
        self._credentials = credentials
        self._settings = settings or {}
        self._blob_store = blob_store
        self._catalog = catalog

    async def load(self) -> None:
        """Validate credentials by fetching a token and listing team channels."""
        # Delayed import: httpx is only needed at runtime
        import httpx

        token = await _get_access_token(self._credentials)
        team_id = self._settings.get("team_id")
        if not team_id:
            raise ValueError("teams settings must include 'team_id'")

        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        ) as http:
            await _graph_get(http, f"/teams/{team_id}/channels")
        logger.info("TeamsConnector loaded (team_id=%s)", team_id)

    async def poll(self) -> list[dict[str, Any]]:
        """Fetch messages from configured team channels and return entry dicts.

        Each entry includes source_path, content_hash, content_type, and
        size_bytes.  Message text is included in metadata for later
        materialization into the blob store.
        """
        # Delayed import: httpx is only needed at runtime
        import httpx

        token = await _get_access_token(self._credentials)
        team_id = self._settings.get("team_id")
        if not team_id:
            raise ValueError("teams settings must include 'team_id'")

        channel_ids = self._settings.get("channel_ids")
        entries: list[dict[str, Any]] = []

        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        ) as http:
            if not channel_ids:
                channel_ids = await self._list_channel_ids(http, team_id)

            for channel_id in channel_ids:
                channel_entries = await self._fetch_channel_messages(http, team_id, channel_id)
                entries.extend(channel_entries)

        logger.info("TeamsConnector polled %d messages from team %s", len(entries), team_id)
        return entries

    async def get_content_url(self, vfs_ref: str) -> Optional[str]:
        """Materialize message text into blob store and return presigned URL.

        Teams messages are text/HTML with no native presigned URL mechanism,
        so we write the content to the blob store and presign from there.
        """
        if self._catalog is None or self._blob_store is None:
            return None
        entry = await self._catalog.get(vfs_ref)
        if entry is None:
            return None

        # If already materialized (has a blob_key), just presign
        if entry.blob_key:
            url, _, _ = await self._blob_store.generate_download_url(entry.blob_key)
            return url

        # Materialize from stored metadata
        content = entry.metadata.get("teams_content", "")
        if not content:
            return None

        blob_key = f"{entry.workspace_id}/{CONNECTOR_TYPE}/{vfs_ref}/content.txt"
        await self._blob_store.put_object(blob_key, content.encode(), content_type="text/plain")
        await self._catalog.update(vfs_ref, blob_key=blob_key)

        url, _, _ = await self._blob_store.generate_download_url(blob_key)
        return url

    async def get_metadata(self, vfs_ref: str) -> dict[str, Any]:
        """Return metadata for a Teams message."""
        if self._catalog is None:
            return {"connector_type": CONNECTOR_TYPE}
        entry = await self._catalog.get(vfs_ref)
        if entry is None:
            return {"connector_type": CONNECTOR_TYPE}
        # Exclude the raw content from metadata to keep responses lean
        meta = {k: v for k, v in entry.metadata.items() if k != "teams_content"}
        return {
            "connector_type": CONNECTOR_TYPE,
            "source_path": entry.source_path,
            "content_type": entry.content_type,
            "size_bytes": entry.size_bytes,
            **meta,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _list_channel_ids(self, http, team_id: str) -> list[str]:
        """List all channel IDs for a team."""
        try:
            data = await _graph_get(http, f"/teams/{team_id}/channels")
            return [ch["id"] for ch in data.get("value", [])]
        except Exception:
            logger.error("Failed to list channels for team %s", team_id, exc_info=True)
            return []

    async def _fetch_channel_messages(
        self, http, team_id: str, channel_id: str,
    ) -> list[dict[str, Any]]:
        """Paginate through channel messages and return entry dicts."""
        entries: list[dict[str, Any]] = []
        path = f"/teams/{team_id}/channels/{channel_id}/messages"
        next_link: str | None = None

        while True:
            try:
                if next_link:
                    resp = await http.get(next_link)
                    resp.raise_for_status()
                    data = resp.json()
                else:
                    data = await _graph_get(http, path, params={"$top": "50"})
            except Exception:
                logger.error(
                    "Error fetching Teams messages for channel %s", channel_id, exc_info=True,
                )
                break

            for msg in data.get("value", []):
                created = msg.get("createdDateTime", "")
                body = msg.get("body", {})
                content = body.get("content", "")
                if not content:
                    continue

                sender = msg.get("from", {}).get("user", {}).get("displayName", "unknown")
                msg_id = msg.get("id", "")
                content_hash = hashlib.sha256(content.encode()).hexdigest()

                entries.append({
                    "source_path": f"teams/{team_id}/{channel_id}/{msg_id}",
                    "content_hash": content_hash,
                    "content_type": "text/plain",
                    "size_bytes": len(content.encode()),
                    "metadata": {
                        "teams_team_id": team_id,
                        "teams_channel_id": channel_id,
                        "teams_message_id": msg_id,
                        "teams_sender": sender,
                        "teams_created": created,
                        "teams_body_type": body.get("contentType", "text"),
                        "teams_content": content,
                    },
                })

            next_link = data.get("@odata.nextLink")
            if not next_link:
                break

        return entries


ConnectorRegistry.register(CONNECTOR_TYPE, TeamsConnector)
