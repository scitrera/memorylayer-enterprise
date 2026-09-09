# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Sync engine — polls connectors and emits Aether POOL tasks.

When a connector discovers new or changed content, the sync engine:
1. Registers a VFS entry in the catalog.
2. Calls ``task_client.create_task(task_type="memorylayer-task.doc_added",
   workspace=ws, target_implementation="memorylayer",
   assignment_mode=POOL, payload=msgpack(...))`` per new entry.

Mirrors the pattern in ``memorylayer-enterprise/.../services/tasks/aether_task_service.py``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)

# Task type prefix matching MemoryLayer's convention
_TASK_PREFIX = "memorylayer-task"
_DOC_ADDED_TASK_TYPE = f"{_TASK_PREFIX}.doc_added"
_TARGET_IMPLEMENTATION = "memorylayer"


# Aether TaskClass values (mirrors aether.proto TaskClass enum). The worker
# also reads a string label out of task metadata in a later phase, so emitters
# stamp both the native field and a recoverable metadata label.
TASK_CLASS_BACKGROUND = 2  # TASK_CLASS_BACKGROUND: scheduled/connector-driven ingest
TASK_CLASS_BATCH = 3  # TASK_CLASS_BATCH: user-initiated ingest (e.g. file upload)
TASK_CLASS_LABELS = {
    TASK_CLASS_BACKGROUND: "background",
    TASK_CLASS_BATCH: "batch",
}


class TaskClient(Protocol):
    """Protocol matching the Aether client's create_task signature."""
    async def create_task(
        self,
        *,
        task_type: str,
        workspace: str,
        metadata: dict[str, str],
        payload: bytes,
        target_implementation: str,
        assignment_mode: Any,
        task_class: int = 0,
    ) -> None: ...


class SyncEngine:
    """Sync engine that polls connectors and emits doc_added tasks.

    Args:
        task_client: Aether client (or mock) implementing ``create_task``.
        catalog: VFS catalog for entry registration.
    """

    def __init__(self, task_client: Optional[TaskClient], catalog: Any) -> None:
        self._task_client = task_client
        self._catalog = catalog

    async def emit_doc_added(
        self,
        workspace_id: str,
        vfs_ref: str,
        content_hash: str,
        connector_id: str,
        filename_hint: str,
        task_class: int = 0,
        initiated_by: Optional[str] = None,
        visibility: str = "workspace",
    ) -> Optional[str]:
        """Emit a doc_added POOL task for a newly registered VFS entry.

        Uses msgpack serialization for the payload, matching MemoryLayer's
        existing task service pattern.

        Args:
            workspace_id: Workspace scope.
            vfs_ref: VFS reference for the new entry.
            content_hash: Content hash for dedup.
            connector_id: Source connector ID.
            filename_hint: Original filename for display.
            task_class: Aether ``TaskClass`` value (UI/lifecycle hint). Connector-
                driven ingest passes ``TASK_CLASS_BACKGROUND``; user-initiated
                ingest passes ``TASK_CLASS_BATCH``.
            initiated_by: Originating party for user-initiated ingest (e.g.
                ``"us::{user}"``). Omitted from metadata when None.
            visibility: Task-level visibility, ``"workspace"`` (default) or
                ``"private"``. Stamped into metadata.

        Returns:
            Task ID (``dctask_<hex>``) or None if no task client is available.
        """
        if self._task_client is None:
            logger.warning("No task client available; skipping doc_added emission for vfs_ref=%s", vfs_ref)
            return None

        # Delayed import: msgpack is cheap but we follow the same pattern
        # as MemoryLayer's aether_task_service for consistency.
        from scitrera_rt_data.serialization.msgpack import msgpack_serialize
        from scitrera_aether_client import POOL

        task_id = f"dctask_{uuid4().hex[:12]}"
        payload_dict = {
            "vfs_ref": vfs_ref,
            "content_hash": content_hash,
            "connector_id": connector_id,
            "filename_hint": filename_hint,
            "workspace_id": workspace_id,
        }

        # Aether metadata is a map<string, string>; keep every value a string.
        # The worker reads the Background Tasks fields out of metadata in a
        # later phase, so task_class is stamped here as a recoverable label in
        # addition to the native create_task field.
        metadata: dict[str, str] = {
            "task_id": task_id,
            "connector_id": connector_id,
            "title": filename_hint,
            "bg_kind": "ingest",
            "visibility": visibility,
            "task_class": TASK_CLASS_LABELS.get(task_class, str(task_class)),
        }
        if initiated_by is not None:
            metadata["initiated_by"] = initiated_by

        try:
            await self._task_client.create_task(
                task_type=_DOC_ADDED_TASK_TYPE,
                workspace=workspace_id,
                metadata=metadata,
                payload=msgpack_serialize(payload_dict),
                target_implementation=_TARGET_IMPLEMENTATION,
                assignment_mode=POOL,
                task_class=task_class,
            )
            logger.info(
                "Emitted doc_added task %s (vfs_ref=%s, workspace=%s)",
                task_id, vfs_ref, workspace_id,
            )
            return task_id
        except Exception:
            logger.error(
                "Failed to emit doc_added task for vfs_ref=%s",
                vfs_ref,
                exc_info=True,
            )
            return None

    async def sync_provider(
        self,
        provider_id: str,
        workspace_id: str,
        connector: Any,
        full_sync: bool = False,
    ) -> dict:
        """Run a sync cycle for a single provider.

        Calls the connector's ``poll()`` method, registers discovered entries,
        and emits ``doc_added`` tasks for each new entry.

        Args:
            provider_id: Provider ID.
            workspace_id: Workspace scope.
            connector: Connector instance with ``poll()`` and ``load()`` methods.
            full_sync: If True, ignore checkpoint and do a full resync.

        Returns:
            Summary dict with discovered/synced counts.
        """
        logger.info("Starting sync for provider %s (workspace=%s, full=%s)", provider_id, workspace_id, full_sync)

        discovered = 0
        synced = 0

        try:
            entries = await connector.poll()
            discovered = len(entries)
            connector_type = getattr(type(connector), "connector_type", None)
            if not isinstance(connector_type, str) or not connector_type:
                connector_type = type(connector).__module__.rsplit(".", 1)[-1]

            for entry_info in entries:
                source_path = entry_info.get("source_path", "")
                content_hash = entry_info.get("content_hash", "")
                filename = source_path.split("/")[-1] if source_path else "unknown"

                # Check for dedup
                existing = await self._catalog.find_by_content_hash(workspace_id, content_hash)
                if existing and not full_sync:
                    logger.debug("Skipping duplicate entry: %s (hash=%s)", source_path, content_hash)
                    continue

                # Register VFS entry
                entry_metadata = dict(entry_info.get("metadata", {}))
                entry_metadata.setdefault("connector_type", connector_type)
                vfs_entry = await self._catalog.register(
                    workspace_id=workspace_id,
                    connector_id=provider_id,
                    source_path=source_path,
                    content_hash=content_hash,
                    content_type=entry_info.get("content_type"),
                    size_bytes=entry_info.get("size_bytes"),
                    blob_key=entry_info.get("blob_key"),
                    metadata=entry_metadata,
                )

                # Emit doc_added task. Connector-driven sync is scheduled with
                # no user in the loop, so it is classed as BACKGROUND with no
                # initiated_by.
                await self.emit_doc_added(
                    workspace_id=workspace_id,
                    vfs_ref=vfs_entry.vfs_ref,
                    content_hash=content_hash,
                    connector_id=provider_id,
                    filename_hint=filename,
                    task_class=TASK_CLASS_BACKGROUND,
                )
                synced += 1

        except Exception:
            logger.error("Sync failed for provider %s", provider_id, exc_info=True)

        logger.info("Sync complete for provider %s: discovered=%d, synced=%d", provider_id, discovered, synced)
        return {"discovered": discovered, "synced": synced}
