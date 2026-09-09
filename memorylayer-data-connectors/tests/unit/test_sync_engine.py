# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the sync engine — create_task payload shape assertions.

Mirrors the pattern from memorylayer-enterprise's test_aether_task_service.py.
All Aether SDK interactions are mocked.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from data_connectors.vfs.catalog import VfsCatalog
from data_connectors.services.sync_engine import (
    SyncEngine,
    _DOC_ADDED_TASK_TYPE,
    _TARGET_IMPLEMENTATION,
    TASK_CLASS_BACKGROUND,
    TASK_CLASS_BATCH,
)


@pytest.fixture
def catalog():
    return VfsCatalog()


@pytest.fixture
def mock_task_client():
    client = AsyncMock()
    client.create_task = AsyncMock()
    return client


@pytest.fixture
def engine(mock_task_client, catalog):
    return SyncEngine(task_client=mock_task_client, catalog=catalog)


class TestEmitDocAdded:
    """Verify create_task payload shape for doc_added tasks."""

    @pytest.mark.asyncio
    async def test_emits_correct_task_type(self, engine, mock_task_client):
        task_id = await engine.emit_doc_added(
            workspace_id="ws-1",
            vfs_ref="vfs_abc123",
            content_hash="sha256_xyz",
            connector_id="manual_upload",
            filename_hint="test.pdf",
        )

        assert task_id is not None
        assert task_id.startswith("dctask_")
        mock_task_client.create_task.assert_awaited_once()

        call_kwargs = mock_task_client.create_task.call_args.kwargs
        assert call_kwargs["task_type"] == _DOC_ADDED_TASK_TYPE
        assert call_kwargs["task_type"] == "memorylayer-task.doc_added"

    @pytest.mark.asyncio
    async def test_uses_pool_assignment_mode(self, engine, mock_task_client):
        await engine.emit_doc_added(
            workspace_id="ws-1", vfs_ref="vfs_1",
            content_hash="h1", connector_id="c1", filename_hint="f.txt",
        )

        call_kwargs = mock_task_client.create_task.call_args.kwargs
        assert "assignment_mode" in call_kwargs
        # POOL is imported at call time; verify it was passed
        from scitrera_aether_client import POOL
        assert call_kwargs["assignment_mode"] == POOL

    @pytest.mark.asyncio
    async def test_targets_memorylayer_pool(self, engine, mock_task_client):
        await engine.emit_doc_added(
            workspace_id="ws-1", vfs_ref="vfs_1",
            content_hash="h1", connector_id="c1", filename_hint="f.txt",
        )

        call_kwargs = mock_task_client.create_task.call_args.kwargs
        assert call_kwargs["target_implementation"] == _TARGET_IMPLEMENTATION
        assert call_kwargs["target_implementation"] == "memorylayer"

    @pytest.mark.asyncio
    async def test_workspace_passed_correctly(self, engine, mock_task_client):
        await engine.emit_doc_added(
            workspace_id="prod-workspace", vfs_ref="vfs_1",
            content_hash="h1", connector_id="c1", filename_hint="f.txt",
        )

        call_kwargs = mock_task_client.create_task.call_args.kwargs
        assert call_kwargs["workspace"] == "prod-workspace"

    @pytest.mark.asyncio
    async def test_payload_is_msgpack_bytes(self, engine, mock_task_client):
        await engine.emit_doc_added(
            workspace_id="ws-1", vfs_ref="vfs_abc",
            content_hash="hash123", connector_id="manual_upload",
            filename_hint="report.pdf",
        )

        call_kwargs = mock_task_client.create_task.call_args.kwargs
        assert isinstance(call_kwargs["payload"], bytes)

        from scitrera_rt_data.serialization.msgpack import msgpack_deserialize
        payload = msgpack_deserialize(call_kwargs["payload"])
        assert payload["vfs_ref"] == "vfs_abc"
        assert payload["content_hash"] == "hash123"
        assert payload["connector_id"] == "manual_upload"
        assert payload["filename_hint"] == "report.pdf"
        assert payload["workspace_id"] == "ws-1"

    @pytest.mark.asyncio
    async def test_metadata_contains_task_id(self, engine, mock_task_client):
        task_id = await engine.emit_doc_added(
            workspace_id="ws-1", vfs_ref="vfs_1",
            content_hash="h1", connector_id="c1", filename_hint="f.txt",
        )

        call_kwargs = mock_task_client.create_task.call_args.kwargs
        assert "metadata" in call_kwargs
        assert call_kwargs["metadata"]["task_id"] == task_id
        assert call_kwargs["metadata"]["connector_id"] == "c1"

    @pytest.mark.asyncio
    async def test_connector_path_background_no_initiated_by(self, engine, mock_task_client):
        """Connector-driven ingest: BACKGROUND task_class, no initiated_by, title/bg_kind stamped."""
        await engine.emit_doc_added(
            workspace_id="ws-1", vfs_ref="vfs_1",
            content_hash="h1", connector_id="local_fs", filename_hint="report.pdf",
            task_class=TASK_CLASS_BACKGROUND,
        )

        call_kwargs = mock_task_client.create_task.call_args.kwargs
        # Native field
        assert call_kwargs["task_class"] == TASK_CLASS_BACKGROUND
        # Recoverable metadata
        md = call_kwargs["metadata"]
        assert md["task_class"] == "background"
        assert md["bg_kind"] == "ingest"
        assert md["title"] == "report.pdf"
        assert md["visibility"] == "workspace"  # default
        assert "initiated_by" not in md
        assert md["connector_id"] == "local_fs"

    @pytest.mark.asyncio
    async def test_user_path_batch_with_initiated_by_and_visibility(self, engine, mock_task_client):
        """User-initiated ingest: BATCH task_class, initiated_by + visibility stamped."""
        await engine.emit_doc_added(
            workspace_id="ws-1", vfs_ref="vfs_1",
            content_hash="h1", connector_id="manual_upload", filename_hint="upload.docx",
            task_class=TASK_CLASS_BATCH,
            initiated_by="us::alice",
            visibility="private",
        )

        call_kwargs = mock_task_client.create_task.call_args.kwargs
        # Native field
        assert call_kwargs["task_class"] == TASK_CLASS_BATCH
        # Recoverable metadata
        md = call_kwargs["metadata"]
        assert md["task_class"] == "batch"
        assert md["bg_kind"] == "ingest"
        assert md["title"] == "upload.docx"
        assert md["initiated_by"] == "us::alice"
        assert md["visibility"] == "private"

    @pytest.mark.asyncio
    async def test_returns_none_without_client(self, catalog):
        engine = SyncEngine(task_client=None, catalog=catalog)
        task_id = await engine.emit_doc_added(
            workspace_id="ws-1", vfs_ref="vfs_1",
            content_hash="h1", connector_id="c1", filename_hint="f.txt",
        )
        assert task_id is None

    @pytest.mark.asyncio
    async def test_handles_create_task_failure(self, engine, mock_task_client):
        mock_task_client.create_task.side_effect = RuntimeError("gRPC unavailable")
        task_id = await engine.emit_doc_added(
            workspace_id="ws-1", vfs_ref="vfs_1",
            content_hash="h1", connector_id="c1", filename_hint="f.txt",
        )
        assert task_id is None


class TestSyncProvider:
    """Verify sync_provider orchestrates connector polling and task emission."""

    @pytest.mark.asyncio
    async def test_sync_discovers_and_emits(self, engine, mock_task_client, catalog):
        connector = AsyncMock()
        connector.poll = AsyncMock(return_value=[
            {"source_path": "docs/a.pdf", "content_hash": "h1", "content_type": "application/pdf"},
            {"source_path": "docs/b.pdf", "content_hash": "h2", "content_type": "application/pdf"},
        ])

        result = await engine.sync_provider(
            provider_id="dp_test",
            workspace_id="ws-1",
            connector=connector,
        )

        assert result["discovered"] == 2
        assert result["synced"] == 2
        assert mock_task_client.create_task.await_count == 2
        entries, _ = await catalog.list_entries("ws-1")
        assert all(entry.metadata["connector_type"] == "mock" for entry in entries)
        # Connector-driven sync emits BACKGROUND tasks with no initiated_by.
        for call in mock_task_client.create_task.call_args_list:
            assert call.kwargs["task_class"] == TASK_CLASS_BACKGROUND
            assert "initiated_by" not in call.kwargs["metadata"]
            assert call.kwargs["metadata"]["bg_kind"] == "ingest"

    @pytest.mark.asyncio
    async def test_sync_skips_duplicates(self, engine, mock_task_client, catalog):
        # Pre-register an entry with the same hash
        await catalog.register(
            workspace_id="ws-1", connector_id="dp_test",
            source_path="existing.pdf", content_hash="h1",
        )

        connector = AsyncMock()
        connector.poll = AsyncMock(return_value=[
            {"source_path": "docs/a.pdf", "content_hash": "h1"},
            {"source_path": "docs/b.pdf", "content_hash": "h2"},
        ])

        result = await engine.sync_provider(
            provider_id="dp_test",
            workspace_id="ws-1",
            connector=connector,
        )

        assert result["discovered"] == 2
        assert result["synced"] == 1  # h1 was deduped
