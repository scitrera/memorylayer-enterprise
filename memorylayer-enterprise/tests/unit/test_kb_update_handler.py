"""Unit tests for the thin kb_update task handler.

The coalesce join now owns debounce, so the handler is a thin
regenerate-and-announce:
- workspace_id present -> generate(workspace_id) once (positional, NO regenerate
  kwarg) -> kb_updated event emitted -> completed progress
- generation failure -> failed progress, no re-raise
- a ``degraded`` payload flag is tolerated (join on_timeout path)
- malformed payload returns cleanly
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memorylayer_saas.tasks.kb_update import KBUpdateTaskHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KB_UPDATE_PAYLOAD = {"workspace_id": "ws_test"}

AETHER_TASK_ID = "atask_kb123"

TASK_METADATA = {
    "task_id": "dctask_kb",
    "title": "Knowledgebase update",
    "bg_kind": "kb",
    "visibility": "workspace",
    "task_class": "background",
}


def _payload_with_progress(task_metadata=None, **extra):
    return {
        **KB_UPDATE_PAYLOAD,
        "_aether_task_id": AETHER_TASK_ID,
        "_task_metadata": task_metadata if task_metadata is not None else TASK_METADATA,
        **extra,
    }


@pytest.fixture()
def mock_variables():
    v = MagicMock()
    v.environ = MagicMock(side_effect=lambda key, default, **kw: default)
    return v


def _make_mock_agent_service():
    """Build a mock agent service whose client emits events / progress."""
    agent_svc = MagicMock()
    client = AsyncMock()
    client.send_event = AsyncMock()
    client.report_progress = AsyncMock()
    if hasattr(client, "update_task"):
        del client.update_task
    agent_svc.client = client
    return agent_svc


def _make_mock_kb_service():
    svc = MagicMock()
    svc.generate = AsyncMock()
    return svc


@contextmanager
def patch_kb_update(agent_service=None, kb_service=None):
    """Patch all dependencies for KBUpdateTaskHandler."""
    agent_service = agent_service or _make_mock_agent_service()
    kb_service = kb_service or _make_mock_kb_service()

    from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION

    def ext_side_effect(ext_name, v=None):
        return {EXT_AETHER_SERVICE_CONNECTION: agent_service}.get(ext_name)

    with patch("memorylayer_saas.tasks.kb_update.get_knowledgebase_service",
               return_value=kb_service), \
         patch("memorylayer_saas.tasks.kb_update.get_logger", return_value=MagicMock()), \
         patch("memorylayer_saas.tasks.doc_added.get_extension", side_effect=ext_side_effect):
        yield agent_service, kb_service


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKBUpdateHandler:
    """Tests for the thin KBUpdateTaskHandler."""

    def test_get_task_type(self):
        handler = KBUpdateTaskHandler()
        assert handler.get_task_type() == "kb_update"

    def test_get_schedule_is_none(self):
        handler = KBUpdateTaskHandler()
        assert handler.get_schedule(MagicMock()) is None

    def test_is_task_handler_plugin(self):
        from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
        handler = KBUpdateTaskHandler()
        assert isinstance(handler, TaskHandlerPlugin)

    @pytest.mark.asyncio
    async def test_malformed_payload_returns_cleanly(self, mock_variables):
        """Missing workspace_id returns without raising or generating."""
        with patch_kb_update() as (_, kb_service):
            handler = KBUpdateTaskHandler()
            await handler.handle(mock_variables, {"junk": "value"})

        kb_service.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_regenerates_once_and_emits(self, mock_variables):
        """Happy path: generate(workspace_id) once (positional) -> kb_updated event."""
        agent_svc = _make_mock_agent_service()
        kb_service = _make_mock_kb_service()

        with patch_kb_update(agent_service=agent_svc, kb_service=kb_service):
            handler = KBUpdateTaskHandler()
            await handler.handle(mock_variables, _payload_with_progress())

        # Exactly one generate, called positionally with the workspace id and
        # NO regenerate kwarg (generate builds KBGenerateOptions internally).
        kb_service.generate.assert_called_once()
        assert kb_service.generate.call_args.args[0] == "ws_test"
        assert "regenerate" not in kb_service.generate.call_args.kwargs

        # kb_updated event emitted with the JSON EventPayload shape.
        assert agent_svc.client.send_event.called
        raw = agent_svc.client.send_event.call_args.args[0]
        envelope = json.loads(raw.decode("utf-8"))
        assert envelope["source_agent"] == "memorylayer"
        assert envelope["event_names"] == ["memorylayer.kb_updated"]
        assert envelope["data"]["workspace_id"] == "ws_test"
        assert "generated_at" in envelope["data"]

    @pytest.mark.asyncio
    async def test_degraded_payload_is_tolerated(self, mock_variables):
        """A degraded flag (join on_timeout path) still regenerates + emits."""
        agent_svc = _make_mock_agent_service()
        kb_service = _make_mock_kb_service()

        with patch_kb_update(agent_service=agent_svc, kb_service=kb_service):
            handler = KBUpdateTaskHandler()
            await handler.handle(mock_variables, _payload_with_progress(degraded=True))

        kb_service.generate.assert_called_once()
        assert kb_service.generate.call_args.args[0] == "ws_test"
        assert agent_svc.client.send_event.called

    @pytest.mark.asyncio
    async def test_generation_failure_emits_failed_no_reraise(self, mock_variables):
        """A generation failure emits failed progress and does NOT raise."""
        agent_svc = _make_mock_agent_service()
        kb_service = _make_mock_kb_service()
        kb_service.generate.side_effect = ValueError("graph analysis exploded")

        with patch_kb_update(agent_service=agent_svc, kb_service=kb_service):
            handler = KBUpdateTaskHandler()
            # Must NOT raise.
            await handler.handle(mock_variables, _payload_with_progress())

        kb_service.generate.assert_called_once()

        # A terminal "failed" progress report was emitted.
        failed = [
            c for c in agent_svc.client.report_progress.call_args_list
            if c.kwargs.get("state") == "failed"
        ]
        assert len(failed) == 1
        assert "graph analysis exploded" in failed[0].kwargs["summary"]

    @pytest.mark.asyncio
    async def test_generation_failure_emits_kb_update_failed_event(self, mock_variables):
        """On generate() failure the UI signal kb_update_failed is emitted.

        Mirrors the kb_updated event shape but on the failure event name, with
        the owning workspace id and an error summary so the UI can react.
        """
        agent_svc = _make_mock_agent_service()
        kb_service = _make_mock_kb_service()
        kb_service.generate.side_effect = ValueError("graph analysis exploded")

        with patch_kb_update(agent_service=agent_svc, kb_service=kb_service):
            handler = KBUpdateTaskHandler()
            await handler.handle(mock_variables, _payload_with_progress())

        # Exactly one event, and it is the failure equivalent (NOT kb_updated).
        assert agent_svc.client.send_event.call_count == 1
        raw = agent_svc.client.send_event.call_args.args[0]
        envelope = json.loads(raw.decode("utf-8"))
        assert envelope["source_agent"] == "memorylayer"
        assert envelope["event_names"] == ["memorylayer.kb_update_failed"]
        assert envelope["data"]["workspace_id"] == "ws_test"
        assert "graph analysis exploded" in envelope["data"]["error"]
        assert "generated_at" in envelope["data"]

    @pytest.mark.asyncio
    async def test_terminal_success_emits_completed(self, mock_variables):
        """Happy path ends with state=completed, completion=1.0."""
        agent_svc = _make_mock_agent_service()

        with patch_kb_update(agent_service=agent_svc):
            handler = KBUpdateTaskHandler()
            await handler.handle(mock_variables, _payload_with_progress())

        last = agent_svc.client.report_progress.call_args_list[-1]
        assert last.kwargs["state"] == "completed"
        assert last.kwargs["completion"] == 1.0
