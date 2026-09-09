"""Unit tests for the memorylayer-worker distributed task runner.

Tests:
- Handler discovery and registration
- Native task-assignment dispatch (POOL) and Aether task_id threading
- Completion/failure reporting back to Aether (retry/DLQ safety)
- CLI argument parsing and defaults
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from memorylayer_saas.worker.runner import WorkerRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handler_plugin(task_type: str, handler: AsyncMock | None = None) -> MagicMock:
    """Create a fake TaskHandlerPlugin with ``get_task_type()`` and ``handle()``."""
    plugin = MagicMock()
    plugin.get_task_type.return_value = task_type
    plugin.handle = handler or AsyncMock()
    return plugin


# ---------------------------------------------------------------------------
# Handler discovery
# ---------------------------------------------------------------------------

class TestHandlerDiscovery:
    """Tests for WorkerRunner.discover_handlers and initialize."""

    def test_discovers_all_handlers(self) -> None:
        """All TaskHandlerPlugin instances are registered when no filter is set."""
        v = MagicMock()
        runner = WorkerRunner(v=v, task_type_filter=None)

        handler_a = _make_handler_plugin("decay_memories")
        handler_b = _make_handler_plugin("detect_contradictions")
        extensions = {"decay": handler_a, "contradictions": handler_b}

        with (
            patch("memorylayer_saas.worker.runner._ext_multi_task_handlers", return_value="test-ext-key"),
            patch("memorylayer_saas.worker.runner.get_extensions", return_value=extensions),
        ):
            handlers = runner.discover_handlers()

        assert set(handlers.keys()) == {"decay_memories", "detect_contradictions"}
        assert handlers["decay_memories"] is handler_a.handle
        assert handlers["detect_contradictions"] is handler_b.handle

    def test_filters_by_task_type(self) -> None:
        """Only task types in the filter set are registered."""
        v = MagicMock()
        runner = WorkerRunner(v=v, task_type_filter={"decay_memories"})

        handler_a = _make_handler_plugin("decay_memories")
        handler_b = _make_handler_plugin("detect_contradictions")
        extensions = {"decay": handler_a, "contradictions": handler_b}

        with (
            patch("memorylayer_saas.worker.runner._ext_multi_task_handlers", return_value="test-ext-key"),
            patch("memorylayer_saas.worker.runner.get_extensions", return_value=extensions),
        ):
            handlers = runner.discover_handlers()

        assert set(handlers.keys()) == {"decay_memories"}

    def test_empty_extensions(self) -> None:
        """No handlers registered when no plugins are found."""
        v = MagicMock()
        runner = WorkerRunner(v=v)

        with (
            patch("memorylayer_saas.worker.runner._ext_multi_task_handlers", return_value="test-ext-key"),
            patch("memorylayer_saas.worker.runner.get_extensions", return_value={}),
        ):
            handlers = runner.discover_handlers()

        assert handlers == {}

    def test_initialize_populates_handlers(self) -> None:
        """initialize() stores discovered handlers internally."""
        v = MagicMock()
        runner = WorkerRunner(v=v)

        handler_a = _make_handler_plugin("session_cleanup")
        extensions = {"cleanup": handler_a}

        with (
            patch("memorylayer_saas.worker.runner._ext_multi_task_handlers", return_value="test-ext-key"),
            patch("memorylayer_saas.worker.runner.get_extensions", return_value=extensions),
        ):
            runner.initialize()

        assert "session_cleanup" in runner._handlers

    def test_filter_with_multiple_overlapping_types(self) -> None:
        """Filter with multiple types selects exactly those types."""
        v = MagicMock()
        runner = WorkerRunner(v=v, task_type_filter={"decay_memories", "session_cleanup"})

        handler_a = _make_handler_plugin("decay_memories")
        handler_b = _make_handler_plugin("detect_contradictions")
        handler_c = _make_handler_plugin("session_cleanup")
        extensions = {"a": handler_a, "b": handler_b, "c": handler_c}

        with (
            patch("memorylayer_saas.worker.runner._ext_multi_task_handlers", return_value="test-ext-key"),
            patch("memorylayer_saas.worker.runner.get_extensions", return_value=extensions),
        ):
            handlers = runner.discover_handlers()

        assert set(handlers.keys()) == {"decay_memories", "session_cleanup"}


# ---------------------------------------------------------------------------
# Native task assignment (POOL) — Aether task_id threading
#
# Tasks are delivered exclusively via Aether's native task lifecycle
# (``create_task`` / POOL assignment) -> ``_handle_task_assignment``.  The
# legacy topic-publish ``_handle_message`` dispatch path was removed (it never
# reported completion/failure, so failures were silently dropped); the native
# path below reports complete/fail to Aether for retry/DLQ safety.
# ---------------------------------------------------------------------------

def _make_task_assignment(
    task_type: str = "memorylayer-task.doc_added",
    task_id: str = "atask_aether123",
    metadata: dict | None = None,
    payload: bytes = b"",
) -> SimpleNamespace:
    """Create a fake TaskAssignment-like object."""
    return SimpleNamespace(
        task_type=task_type,
        task_id=task_id,
        metadata=metadata if metadata is not None else {},
        payload=payload,
    )


class TestNativeTaskAssignment:
    """Tests for WorkerRunner._handle_task_assignment Aether task_id threading."""

    @pytest.mark.asyncio
    async def test_injects_aether_task_id_and_metadata(self) -> None:
        """The handler receives the Aether task_id + task metadata via reserved keys."""
        v = MagicMock()
        runner = WorkerRunner(v=v)
        handler = AsyncMock()
        runner._handlers = {"doc_added": handler}
        runner._client = AsyncMock()

        metadata = {
            "task_id": "dctask_deadbeef",  # dctask id — NOT the correlation key
            "title": "report.pdf",
            "visibility": "workspace",
            "task_class": "background",
        }
        # msgpack-serialized payload so deserialization picks the real branch.
        from scitrera_rt_data.serialization.msgpack import msgpack_serialize
        payload_bytes = msgpack_serialize({"workspace_id": "ws-1", "vfs_ref": "vfs_x"})

        assignment = _make_task_assignment(
            task_id="atask_aether123", metadata=metadata, payload=payload_bytes,
        )

        await runner._handle_task_assignment(assignment)

        handler.assert_awaited_once()
        passed_payload = handler.await_args.args[1]
        assert passed_payload["_aether_task_id"] == "atask_aether123"
        assert passed_payload["_task_metadata"] == metadata
        # The dctask id is preserved inside metadata, never promoted to the key.
        assert passed_payload["_aether_task_id"] != metadata["task_id"]
        # Completion is reported against the Aether task_id.
        runner._client.complete_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handler_failure_reports_fail_to_aether(self) -> None:
        """A failing handler reports fail_task to Aether (retry/DLQ-safe).

        This is the failure-handling guarantee that the removed legacy
        ``on_message`` path lacked: handler exceptions must not be silently
        dropped — they are reported back to Aether against the task_id.
        """
        from scitrera_rt_data.serialization.msgpack import msgpack_serialize

        v = MagicMock()
        runner = WorkerRunner(v=v)
        runner._handlers = {"doc_added": AsyncMock(side_effect=RuntimeError("boom"))}
        runner._client = AsyncMock()

        assignment = _make_task_assignment(
            task_id="atask_fail",
            metadata={"task_id": "dctask_fail"},
            payload=msgpack_serialize({"workspace_id": "ws-1"}),
        )

        # Must not raise — the exception is caught and reported.
        await runner._handle_task_assignment(assignment)

        runner._client.complete_task.assert_not_awaited()
        runner._client.fail_task.assert_awaited_once()
        # Failure is reported against the Aether task_id with the error reason.
        call = runner._client.fail_task.await_args
        assert call.args[0] == "atask_fail"
        assert "boom" in call.kwargs.get("reason", "")

    @pytest.mark.asyncio
    async def test_unknown_task_type_reports_fail_to_aether(self) -> None:
        """An unknown task type is reported as failed (not silently dropped)."""
        v = MagicMock()
        runner = WorkerRunner(v=v)
        runner._handlers = {}
        runner._client = AsyncMock()

        assignment = _make_task_assignment(
            task_type="memorylayer-task.nonexistent",
            task_id="atask_unknown",
            metadata={"task_id": "dctask_unknown"},
            payload=b"",
        )

        await runner._handle_task_assignment(assignment)

        runner._client.fail_task.assert_awaited_once()
        assert runner._client.fail_task.await_args.args[0] == "atask_unknown"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestCLI:
    """Tests for the Click CLI argument parsing."""

    def test_start_help(self) -> None:
        """``memorylayer-worker start --help`` exits cleanly with usage info."""
        from memorylayer_saas.worker.cli import cli

        result = CliRunner().invoke(cli, ["start", "--help"])
        assert result.exit_code == 0
        assert "Start the MemoryLayer distributed worker" in result.output

    def test_cli_help(self) -> None:
        """``memorylayer-worker --help`` shows the top-level group help."""
        from memorylayer_saas.worker.cli import cli

        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "distributed task execution" in result.output

    def test_start_with_options(self) -> None:
        """CLI options are passed through to the runner."""
        from memorylayer_saas.worker.cli import cli

        with patch("memorylayer_saas.worker.cli.asyncio") as mock_asyncio:
            # Prevent actual asyncio.run from executing
            mock_asyncio.run = MagicMock()

            result = CliRunner().invoke(cli, [
                "start",
                "--aether-addr", "myhost:9999",
                "--workspace", "test-ws",
                "--specifier", "test-spec",
                "--task-types", "decay,enrich",
            ])

        # The command should have printed configuration and attempted to run
        assert "myhost:9999" in result.output
        assert "test-ws" in result.output
        assert "test-spec" in result.output

    def test_start_task_types_parsing(self) -> None:
        """Comma-separated task types are parsed into a set."""
        from memorylayer_saas.worker.cli import cli

        with patch("memorylayer_saas.worker.cli.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock()

            result = CliRunner().invoke(cli, [
                "start",
                "--task-types", "decay_memories, enrich_memory ,session_cleanup",
            ])

        assert result.exit_code == 0
        # All three types should appear in output
        assert "decay_memories" in result.output
        assert "enrich_memory" in result.output
        assert "session_cleanup" in result.output

    def test_start_no_task_types_means_all(self) -> None:
        """When --task-types is not provided, 'all' is shown."""
        from memorylayer_saas.worker.cli import cli

        with patch("memorylayer_saas.worker.cli.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock()

            result = CliRunner().invoke(cli, ["start"])

        assert result.exit_code == 0
        assert "all" in result.output
