"""Slack connector — syncs channel messages from Slack.

Discovers messages via the ``slack_sdk`` async web client, materializes
text content into the data-connectors blob store, and returns presigned
blob URLs for content fetch (Slack has no native presigned-URL mechanism
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

CONNECTOR_TYPE = "slack"


def _build_client(credentials: dict):
    """Create an async Slack web client.

    Returns a ``slack_sdk.web.async_client.AsyncWebClient``.
    """
    # Delayed import: slack_sdk is a heavy optional dependency
    from slack_sdk.web.async_client import AsyncWebClient
    bot_token = credentials.get("bot_token")
    if not bot_token:
        raise ValueError("slack credentials must include 'bot_token'")
    return AsyncWebClient(token=bot_token)


class SlackConnector:
    """Connector for Slack channel messages.

    Args:
        credentials: Dict with ``bot_token``.
        settings: Dict with ``channels`` (list of channel names or IDs)
            and optional ``include_threads`` (bool).
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
        """Validate credentials by calling auth.test."""
        client = _build_client(self._credentials)
        resp = await client.auth_test()
        if not resp.get("ok"):
            raise ValueError(f"Slack auth.test failed: {resp.get('error', 'unknown')}")
        logger.info("SlackConnector loaded (team=%s)", resp.get("team"))

    async def poll(self) -> list[dict[str, Any]]:
        """Fetch messages from configured channels and return entry dicts.

        Each entry includes source_path, content_hash, content_type, and
        size_bytes.  Message text is included in metadata for later
        materialization into the blob store.
        """
        client = _build_client(self._credentials)
        channels = self._settings.get("channels", [])
        include_threads = self._settings.get("include_threads", False)
        entries: list[dict[str, Any]] = []

        for channel in channels:
            channel_id = await self._resolve_channel(client, channel)
            if not channel_id:
                logger.warning("Could not resolve Slack channel: %s", channel)
                continue

            cursor = None
            while True:
                try:
                    kwargs: dict = {"channel": channel_id, "limit": 200}
                    if cursor:
                        kwargs["cursor"] = cursor
                    resp = await client.conversations_history(**kwargs)
                except Exception:
                    logger.error("Error fetching Slack history for %s", channel_id, exc_info=True)
                    break

                for msg in resp.get("messages", []):
                    text = msg.get("text", "")
                    if not text:
                        continue
                    ts = msg.get("ts", "0")
                    user = msg.get("user", "unknown")

                    thread_text = ""
                    if include_threads and msg.get("reply_count", 0) > 0:
                        thread_text = await self._fetch_thread(client, channel_id, ts)

                    full_content = text
                    if thread_text:
                        full_content = f"{text}\n\n--- Thread Replies ---\n{thread_text}"

                    content_hash = hashlib.sha256(full_content.encode()).hexdigest()
                    source_url = f"https://slack.com/archives/{channel_id}/p{ts.replace('.', '')}"

                    entries.append({
                        "source_path": f"slack/{channel}/{ts}",
                        "content_hash": content_hash,
                        "content_type": "text/plain",
                        "size_bytes": len(full_content.encode()),
                        "metadata": {
                            "slack_channel": channel,
                            "slack_channel_id": channel_id,
                            "slack_ts": ts,
                            "slack_user": user,
                            "slack_source_url": source_url,
                            "slack_content": full_content,
                        },
                    })

                cursor = resp.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

        logger.info("SlackConnector polled %d messages from %d channels", len(entries), len(channels))
        return entries

    async def get_content_url(self, vfs_ref: str) -> Optional[str]:
        """Materialize message text into blob store and return presigned URL.

        Slack messages are text-only with no native presigned URL mechanism,
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
        content = entry.metadata.get("slack_content", "")
        if not content:
            return None

        blob_key = f"{entry.workspace_id}/{CONNECTOR_TYPE}/{vfs_ref}/content.txt"
        await self._blob_store.put_object(blob_key, content.encode(), content_type="text/plain")
        await self._catalog.update(vfs_ref, blob_key=blob_key)

        url, _, _ = await self._blob_store.generate_download_url(blob_key)
        return url

    async def get_metadata(self, vfs_ref: str) -> dict[str, Any]:
        """Return metadata for a Slack message."""
        if self._catalog is None:
            return {"connector_type": CONNECTOR_TYPE}
        entry = await self._catalog.get(vfs_ref)
        if entry is None:
            return {"connector_type": CONNECTOR_TYPE}
        # Exclude the raw content from metadata to keep responses lean
        meta = {k: v for k, v in entry.metadata.items() if k != "slack_content"}
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

    async def _resolve_channel(self, client, channel: str) -> str | None:
        """Resolve a channel name or ID to a channel ID."""
        if channel.startswith("C") or channel.startswith("G"):
            return channel
        try:
            cursor = None
            while True:
                kwargs: dict = {"limit": 200, "types": "public_channel,private_channel"}
                if cursor:
                    kwargs["cursor"] = cursor
                resp = await client.conversations_list(**kwargs)
                for ch in resp.get("channels", []):
                    if ch.get("name") == channel or ch.get("name") == channel.lstrip("#"):
                        return ch["id"]
                cursor = resp.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break
        except Exception:
            logger.error("Error resolving Slack channel %s", channel, exc_info=True)
        return None

    async def _fetch_thread(self, client, channel_id: str, thread_ts: str) -> str:
        """Fetch all replies in a thread and return as combined text."""
        parts: list[str] = []
        cursor = None
        while True:
            try:
                kwargs: dict = {"channel": channel_id, "ts": thread_ts, "limit": 200}
                if cursor:
                    kwargs["cursor"] = cursor
                resp = await client.conversations_replies(**kwargs)
            except Exception:
                logger.error("Error fetching Slack thread %s in %s", thread_ts, channel_id, exc_info=True)
                break

            for msg in resp.get("messages", []):
                if msg.get("ts") == thread_ts:
                    continue
                text = msg.get("text", "")
                if text:
                    user = msg.get("user", "unknown")
                    parts.append(f"{user}: {text}")

            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return "\n".join(parts)


ConnectorRegistry.register(CONNECTOR_TYPE, SlackConnector)
