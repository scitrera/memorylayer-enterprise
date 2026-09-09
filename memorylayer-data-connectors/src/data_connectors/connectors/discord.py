# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Discord connector — syncs channel messages via the Discord REST API.

Discovers messages via ``httpx`` (no ``discord.py`` bot framework needed),
materializes text content into the data-connectors blob store, and returns
presigned blob URLs for content fetch (Discord has no native presigned-URL
mechanism for message text).
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from data_connectors.connectors import ConnectorRegistry
from data_connectors.vfs.blob_store import BlobStore
from data_connectors.vfs.catalog import VfsCatalog

logger = logging.getLogger(__name__)

CONNECTOR_TYPE = "discord"

_DISCORD_API = "https://discord.com/api/v10"


def _epoch_to_snowflake(epoch_seconds: float) -> str:
    """Convert epoch seconds to a Discord snowflake ID (lower bound)."""
    # Discord epoch is 2015-01-01T00:00:00Z = 1420070400
    discord_epoch_ms = int((epoch_seconds - 1420070400) * 1000)
    if discord_epoch_ms < 0:
        discord_epoch_ms = 0
    return str(discord_epoch_ms << 22)


def _snowflake_to_epoch(snowflake: str) -> float:
    """Convert a Discord snowflake ID to epoch seconds."""
    discord_epoch_ms = int(snowflake) >> 22
    return (discord_epoch_ms / 1000) + 1420070400


class DiscordConnector:
    """Connector for Discord channel messages.

    Args:
        credentials: Dict with ``bot_token``.
        settings: Dict with ``channel_ids`` (list) and/or ``guild_id``.
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
        """Validate credentials by fetching the bot's own user info."""
        # Delayed import: httpx is only needed at runtime
        import httpx

        bot_token = self._credentials.get("bot_token")
        if not bot_token:
            raise ValueError("discord credentials must include 'bot_token'")

        async with httpx.AsyncClient(
            headers={"Authorization": f"Bot {bot_token}"},
            timeout=30.0,
        ) as http:
            resp = await http.get(f"{_DISCORD_API}/users/@me")
            resp.raise_for_status()
        logger.info("DiscordConnector loaded (guild_id=%s)", self._settings.get("guild_id"))

    async def poll(self) -> list[dict[str, Any]]:
        """Fetch messages from configured channels and return entry dicts.

        Each entry includes source_path, content_hash, content_type, and
        size_bytes.  Message text is included in metadata for later
        materialization into the blob store.
        """
        # Delayed import: httpx is only needed at runtime
        import httpx

        bot_token = self._credentials.get("bot_token")
        if not bot_token:
            raise ValueError("discord credentials must include 'bot_token'")

        channel_ids: list[str] = self._settings.get("channel_ids", [])
        guild_id: str = self._settings.get("guild_id", "")
        entries: list[dict[str, Any]] = []

        async with httpx.AsyncClient(
            headers={"Authorization": f"Bot {bot_token}"},
            timeout=30.0,
        ) as http:
            if not channel_ids and guild_id:
                channel_ids = await self._list_text_channels(http, guild_id)

            for channel_id in channel_ids:
                channel_entries = await self._fetch_messages(http, channel_id, guild_id)
                entries.extend(channel_entries)

        logger.info("DiscordConnector polled %d messages from %d channels", len(entries), len(channel_ids))
        return entries

    async def get_content_url(self, vfs_ref: str) -> Optional[str]:
        """Materialize message text into blob store and return presigned URL.

        Discord messages are text-only with no native presigned URL mechanism,
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
        content = entry.metadata.get("discord_content", "")
        if not content:
            return None

        blob_key = f"{entry.workspace_id}/{CONNECTOR_TYPE}/{vfs_ref}/content.txt"
        await self._blob_store.put_object(blob_key, content.encode(), content_type="text/plain")
        await self._catalog.update(vfs_ref, blob_key=blob_key)

        url, _, _ = await self._blob_store.generate_download_url(blob_key)
        return url

    async def get_metadata(self, vfs_ref: str) -> dict[str, Any]:
        """Return metadata for a Discord message."""
        if self._catalog is None:
            return {"connector_type": CONNECTOR_TYPE}
        entry = await self._catalog.get(vfs_ref)
        if entry is None:
            return {"connector_type": CONNECTOR_TYPE}
        # Exclude the raw content from metadata to keep responses lean
        meta = {k: v for k, v in entry.metadata.items() if k != "discord_content"}
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

    async def _list_text_channels(self, http, guild_id: str) -> list[str]:
        """List text channel IDs for a guild."""
        url = f"{_DISCORD_API}/guilds/{guild_id}/channels"
        try:
            resp = await http.get(url)
            resp.raise_for_status()
            channels = resp.json()
            # type 0 = text channel
            return [ch["id"] for ch in channels if ch.get("type") == 0]
        except Exception:
            logger.error("Failed to list Discord channels for guild %s", guild_id, exc_info=True)
            return []

    async def _fetch_messages(
        self, http, channel_id: str, guild_id: str,
    ) -> list[dict[str, Any]]:
        """Paginate through channel messages and return entry dicts."""
        url = f"{_DISCORD_API}/channels/{channel_id}/messages"
        entries: list[dict[str, Any]] = []
        last_id: str | None = None

        while True:
            params: dict[str, str] = {"limit": "100"}
            if last_id:
                params["after"] = last_id

            try:
                resp = await http.get(url, params=params)
                if resp.status_code == 403:
                    logger.warning("No access to Discord channel %s", channel_id)
                    break
                resp.raise_for_status()
                messages = resp.json()
            except Exception:
                logger.error("Error fetching Discord messages for channel %s", channel_id, exc_info=True)
                break

            if not messages:
                break

            # Discord returns newest first when using 'after'; sort ascending
            messages.sort(key=lambda m: m["id"])

            for msg in messages:
                content = msg.get("content", "")
                if not content:
                    continue

                author = msg.get("author", {}).get("username", "unknown")
                msg_id = msg["id"]
                ts = _snowflake_to_epoch(msg_id)
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                source_url = f"https://discord.com/channels/{guild_id}/{channel_id}/{msg_id}"

                entries.append({
                    "source_path": f"discord/{guild_id}/{channel_id}/{msg_id}",
                    "content_hash": content_hash,
                    "content_type": "text/plain",
                    "size_bytes": len(content.encode()),
                    "metadata": {
                        "discord_guild_id": guild_id,
                        "discord_channel_id": channel_id,
                        "discord_message_id": msg_id,
                        "discord_author": author,
                        "discord_timestamp": ts,
                        "discord_source_url": source_url,
                        "discord_content": content,
                    },
                })

            last_id = messages[-1]["id"]

            if len(messages) < 100:
                break

        return entries


ConnectorRegistry.register(CONNECTOR_TYPE, DiscordConnector)
