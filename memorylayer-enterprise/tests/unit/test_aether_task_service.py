# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the Aether-backed distributed task service.

All Aether SDK interactions are mocked so no live gateway is required.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.services.tasks.base import TaskStatus, EXT_STORAGE_BACKEND

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_variables():
    """Minimal ``Variables`` stub that supports ``environ`` and logging."""
    v = MagicMock()
    v.environ = MagicMock(side_effect=lambda key, default, **kwargs: default)
    return v


@pytest.fixture()
def mock_logger():
    return MagicMock()


def _make_service(mock_variables, **overrides):
    """Construct an ``AetherTaskService`` with sensible test defaults."""
    from memorylayer_saas.services.tasks import AetherTaskService

    kwargs = dict(
        tasks_enabled=True,
    )
    kwargs.update(overrides)
    return AetherTaskService(mock_variables, **kwargs)


def _attach_mock_client(service, workspace="test-workspace"):
    """Attach a mock Aether client to the service so it appears connected."""
    mock_client = AsyncMock()
    mock_client.create_task = AsyncMock()
    mock_client.close = AsyncMock()
    # Workflow schedule methods return a successful response by default
    mock_response = MagicMock()
    mock_response.success = True
    mock_response.error = ""
    mock_client.upsert_schedule = AsyncMock(return_value=mock_response)
    mock_client.delete_schedule = AsyncMock(return_value=mock_response)
    mock_client.list_schedules = AsyncMock(return_value=mock_response)
    service._client = mock_client
    service._workspace = workspace
    return mock_client


def _make_mock_agent_service(mock_client=None, workspace="test-workspace"):
    """Create a mock AetherAgentService for bind_client testing."""
    agent_svc = MagicMock()
    agent_svc.client = mock_client
    agent_svc.workspace = workspace
    agent_svc.set_task_assignment_handler = MagicMock()
    return agent_svc


# ---------------------------------------------------------------------------
# Test: schedule_task sends correct message to correct topic
# ---------------------------------------------------------------------------

class TestScheduleTask:
    """Verify ``schedule_task`` creates native Aether tasks correctly."""

    @pytest.mark.asyncio
    async def test_creates_native_task_with_correct_type(self, mock_variables):
        service = _make_service(mock_variables)
        mock_client = _attach_mock_client(service, workspace="prod")

        task_id = await service.schedule_task(
            task_type="decay_memories",
            payload={"workspace_id": "ws-1"},
        )

        assert task_id.startswith("atask_")
        mock_client.create_task.assert_awaited_once()

        call_args = mock_client.create_task.call_args
        assert call_args.kwargs["task_type"] == "memorylayer-task.decay_memories"
        assert call_args.kwargs["workspace"] == "prod"

    @pytest.mark.asyncio
    async def test_metadata_contains_task_id(self, mock_variables):
        service = _make_service(mock_variables)
        mock_client = _attach_mock_client(service)

        await service.schedule_task(
            task_type="detect_contradictions",
            payload={"workspace_id": "ws-1"},
        )

        call_args = mock_client.create_task.call_args
        metadata = call_args.kwargs["metadata"]

        assert "task_id" in metadata
        assert metadata["task_id"].startswith("atask_")

    @pytest.mark.asyncio
    async def test_payload_sent_as_binary(self, mock_variables):
        """Payload is msgpack-serialized and sent as binary, not in metadata."""
        service = _make_service(mock_variables)
        mock_client = _attach_mock_client(service)

        await service.schedule_task(
            task_type="detect_contradictions",
            payload={"workspace_id": "ws-1"},
        )

        call_args = mock_client.create_task.call_args
        assert "payload" in call_args.kwargs
        assert isinstance(call_args.kwargs["payload"], bytes)

    @pytest.mark.asyncio
    async def test_task_status_transitions_to_running(self, mock_variables):
        service = _make_service(mock_variables)
        _attach_mock_client(service)

        task_id = await service.schedule_task(
            task_type="decay_memories",
            payload={},
        )
        status = await service.get_task_status(task_id)
        assert status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_disabled(self, mock_variables):
        service = _make_service(mock_variables, tasks_enabled=False)

        task_id = await service.schedule_task(
            task_type="decay_memories",
            payload={},
        )
        assert task_id == ""


# ---------------------------------------------------------------------------
# Test: schedule_recurring starts loop and sends messages
# ---------------------------------------------------------------------------

class TestScheduleRecurring:
    """Verify ``schedule_recurring`` registers Aether workflow schedules."""

    @pytest.mark.asyncio
    async def test_returns_deterministic_schedule_id(self, mock_variables):
        service = _make_service(mock_variables)
        _attach_mock_client(service)

        schedule_id = await service.schedule_recurring(
            task_type="decay_memories",
            interval_seconds=60,
            payload={"workspace_id": "ws-1"},
        )

        assert schedule_id == "ml-decay_memories"

    @pytest.mark.asyncio
    async def test_calls_upsert_schedule(self, mock_variables):
        """Verify upsert_schedule is called with correct parameters."""
        service = _make_service(mock_variables)
        mock_client = _attach_mock_client(service)

        await service.schedule_recurring(
            task_type="session_cleanup",
            interval_seconds=3600,
            payload={"mode": "full"},
        )

        mock_client.upsert_schedule.assert_awaited_once()
        call_kwargs = mock_client.upsert_schedule.call_args.kwargs

        assert call_kwargs["schedule_id"] == "ml-session_cleanup"
        assert call_kwargs["name"] == "memorylayer:session_cleanup"
        assert call_kwargs["schedule_type"] == "interval"
        assert call_kwargs["schedule_expr"] == "3600s"
        assert call_kwargs["miss_policy"] == "fire_once"
        assert call_kwargs["max_concurrent"] == 1

        action = call_kwargs["action"]
        assert action["type"] == "create_task"
        assert action["task_type"] == "memorylayer-task.session_cleanup"
        assert action["payload"] == {"mode": "full"}

    @pytest.mark.asyncio
    async def test_idempotent_across_calls(self, mock_variables):
        """Multiple calls with same task_type produce the same schedule ID."""
        service = _make_service(mock_variables)
        _attach_mock_client(service)

        id1 = await service.schedule_recurring("decay_memories", 60, {})
        id2 = await service.schedule_recurring("decay_memories", 60, {})

        assert id1 == id2 == "ml-decay_memories"

    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self, mock_variables):
        service = _make_service(mock_variables, tasks_enabled=False)
        schedule_id = await service.schedule_recurring(
            task_type="decay_memories",
            interval_seconds=60,
            payload={},
        )
        assert schedule_id == ""

    @pytest.mark.asyncio
    async def test_returns_empty_when_not_connected(self, mock_variables):
        service = _make_service(mock_variables)
        # No client attached
        schedule_id = await service.schedule_recurring(
            task_type="decay_memories",
            interval_seconds=60,
            payload={},
        )
        assert schedule_id == ""


# ---------------------------------------------------------------------------
# Test: cancel_task
# ---------------------------------------------------------------------------

class TestCancelTask:
    """Verify cancellation of both recurring and one-shot tasks."""

    @pytest.mark.asyncio
    async def test_cancel_aether_schedule(self, mock_variables):
        """Cancelling an ml-* schedule calls delete_schedule on Aether."""
        service = _make_service(mock_variables)
        mock_client = _attach_mock_client(service)

        result = await service.cancel_task("ml-decay_memories")
        assert result is True
        mock_client.delete_schedule.assert_awaited_once_with("ml-decay_memories")

    @pytest.mark.asyncio
    async def test_cancel_aether_schedule_failure(self, mock_variables):
        """If delete_schedule fails, cancel returns False."""
        service = _make_service(mock_variables)
        mock_client = _attach_mock_client(service)
        failed_resp = MagicMock()
        failed_resp.success = False
        failed_resp.error = "not found"
        mock_client.delete_schedule = AsyncMock(return_value=failed_resp)

        result = await service.cancel_task("ml-decay_memories")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_pending_one_shot(self, mock_variables):
        """A task that has not been published yet (delayed) can be cancelled."""
        service = _make_service(mock_variables)
        _attach_mock_client(service)

        # Manually insert a pending status to simulate a delayed task
        service._task_status["atask_fakeid"] = TaskStatus.PENDING

        result = await service.cancel_task("atask_fakeid")
        assert result is True
        assert service._task_status["atask_fakeid"] == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_returns_false(self, mock_variables):
        service = _make_service(mock_variables)
        result = await service.cancel_task("atask_doesnotexist")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_already_running_returns_false(self, mock_variables):
        """One-shot tasks already in RUNNING state cannot be cancelled."""
        service = _make_service(mock_variables)
        service._task_status["atask_running"] = TaskStatus.RUNNING

        result = await service.cancel_task("atask_running")
        assert result is False


# ---------------------------------------------------------------------------
# Test: task payload serialization format
# ---------------------------------------------------------------------------

class TestPayloadSerialization:
    """Verify the payload format passed to native task creation."""

    @pytest.mark.asyncio
    async def test_payload_is_msgpack_bytes(self, mock_variables):
        """Payload is sent as msgpack-serialized bytes."""
        service = _make_service(mock_variables)
        mock_client = _attach_mock_client(service)

        await service.schedule_task(
            task_type="enrich_memories",
            payload={"memory_ids": ["m1", "m2"]},
        )

        call_args = mock_client.create_task.call_args
        assert isinstance(call_args.kwargs["payload"], bytes)

        from scitrera_rt_data.serialization.msgpack import msgpack_deserialize
        deserialized = msgpack_deserialize(call_args.kwargs["payload"])
        assert deserialized == {"memory_ids": ["m1", "m2"]}

    @pytest.mark.asyncio
    async def test_uses_pool_assignment_mode(self, mock_variables):
        """Native tasks must use POOL assignment for competing consumers."""
        service = _make_service(mock_variables)
        mock_client = _attach_mock_client(service)

        await service.schedule_task(
            task_type="decay_memories",
            payload={},
        )

        call_args = mock_client.create_task.call_args
        assert "assignment_mode" in call_args.kwargs


# ---------------------------------------------------------------------------
# Test: auth context propagation in payload
# ---------------------------------------------------------------------------

class TestAuthContextPropagation:
    """Verify auth_context flows through in the msgpack payload."""

    @pytest.mark.asyncio
    async def test_auth_context_included_in_payload(self, mock_variables):
        service = _make_service(mock_variables)
        mock_client = _attach_mock_client(service)

        auth = {
            "tenant_id": "tenant-abc",
            "workspace_id": "workspace-123",
            "user_id": "user-456",
            "access_level": 20,
        }

        await service.schedule_task(
            task_type="decay_memories",
            payload={"workspace_id": "ws-1", "auth_context": auth},
        )

        call_args = mock_client.create_task.call_args
        from scitrera_rt_data.serialization.msgpack import msgpack_deserialize
        payload = msgpack_deserialize(call_args.kwargs["payload"])

        assert payload["auth_context"] == auth

    @pytest.mark.asyncio
    async def test_payload_without_auth_context(self, mock_variables):
        service = _make_service(mock_variables)
        mock_client = _attach_mock_client(service)

        await service.schedule_task(
            task_type="decay_memories",
            payload={"workspace_id": "ws-1"},
        )

        call_args = mock_client.create_task.call_args
        from scitrera_rt_data.serialization.msgpack import msgpack_deserialize
        payload = msgpack_deserialize(call_args.kwargs["payload"])

        assert "auth_context" not in payload


# ---------------------------------------------------------------------------
# Test: graceful handling when Aether connection is not available
# ---------------------------------------------------------------------------

class TestGracefulDisconnect:
    """Verify the service degrades gracefully without a live Aether gateway."""

    @pytest.mark.asyncio
    async def test_publish_without_connection_marks_failed(self, mock_variables):
        """Publishing without a connected client should mark the task FAILED."""
        service = _make_service(mock_variables)
        # Deliberately do NOT attach a client

        task_id = await service.schedule_task(
            task_type="decay_memories",
            payload={},
        )

        status = await service.get_task_status(task_id)
        assert status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_create_task_failure_marks_task_failed(self, mock_variables):
        """If the Aether client raises, the task should be marked FAILED."""
        service = _make_service(mock_variables)
        mock_client = _attach_mock_client(service)
        mock_client.create_task.side_effect = RuntimeError("gRPC unavailable")

        task_id = await service.schedule_task(
            task_type="decay_memories",
            payload={},
        )

        status = await service.get_task_status(task_id)
        assert status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_disconnect_is_idempotent(self, mock_variables):
        """Calling disconnect when not connected should not raise."""
        service = _make_service(mock_variables)
        await service.disconnect()  # no client attached, should be safe

    @pytest.mark.asyncio
    async def test_disconnect_releases_client_without_closing(self, mock_variables):
        """Disconnect releases the client reference but does NOT close it."""
        service = _make_service(mock_variables)
        mock_client = _attach_mock_client(service)

        await service.disconnect()
        # The shared client should NOT be closed (not our lifecycle)
        mock_client.close.assert_not_awaited()
        assert service._client is None

    @pytest.mark.asyncio
    async def test_disconnect_preserves_aether_schedules(self, mock_variables):
        """Aether-managed schedules persist across disconnect (no delete_schedule call)."""
        service = _make_service(mock_variables)
        mock_client = _attach_mock_client(service)

        await service.schedule_recurring(
            task_type="cleanup",
            interval_seconds=300,
            payload={},
        )

        await service.disconnect()

        # Aether schedules are NOT cancelled on disconnect - they persist in the DB
        mock_client.delete_schedule.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test: bind_client
# ---------------------------------------------------------------------------

class TestBindClient:
    """Verify bind_client obtains the shared client and registers handler."""

    def test_bind_client_sets_client_and_workspace(self, mock_variables):
        service = _make_service(mock_variables)
        mock_client = AsyncMock()
        agent_svc = _make_mock_agent_service(mock_client, workspace="prod")

        service.bind_client(agent_svc)

        assert service._client is mock_client
        assert service._workspace == "prod"
        agent_svc.set_task_assignment_handler.assert_called_once()

    def test_bind_client_registers_task_handler(self, mock_variables):
        service = _make_service(mock_variables)
        agent_svc = _make_mock_agent_service(AsyncMock())

        service.bind_client(agent_svc)

        # The handler registered should be the service's _handle_task_assignment method
        handler = agent_svc.set_task_assignment_handler.call_args.args[0]
        assert handler.__func__ is type(service)._handle_task_assignment
        assert handler.__self__ is service


# ---------------------------------------------------------------------------
# Test: register_handler
# ---------------------------------------------------------------------------

class TestRegisterHandler:
    """Verify handler registration stores the callable."""

    def test_registers_handler(self, mock_variables):
        service = _make_service(mock_variables)

        async def my_handler(v, payload):
            pass

        service.register_handler("test_task", my_handler)
        assert "test_task" in service._handlers
        assert service._handlers["test_task"] is my_handler


# ---------------------------------------------------------------------------
# Test: plugin lifecycle
# ---------------------------------------------------------------------------

class TestAetherTaskServicePlugin:
    """Verify the plugin follows the expected lifecycle pattern."""

    def test_provider_name(self):
        from memorylayer_saas.services.tasks import AetherTaskServicePlugin

        plugin = AetherTaskServicePlugin()
        assert plugin.PROVIDER_NAME == "aether"

    def test_initialize_returns_service(self, mock_variables, mock_logger):
        from memorylayer_saas.services.tasks import AetherTaskServicePlugin, AetherTaskService

        plugin = AetherTaskServicePlugin()
        service = plugin.initialize(mock_variables, mock_logger)

        assert isinstance(service, AetherTaskService)

    def test_plugin_name_includes_provider(self):
        from memorylayer_saas.services.tasks import AetherTaskServicePlugin

        plugin = AetherTaskServicePlugin()
        assert "aether" in plugin.name()

    def test_dependencies_include_storage_and_agent_service(self, mock_variables):
        from memorylayer_saas.services.tasks import AetherTaskServicePlugin
        from memorylayer_saas.services.aether_agent import EXT_AETHER_AGENT_SERVICE

        plugin = AetherTaskServicePlugin()
        deps = plugin.get_dependencies(mock_variables)
        assert EXT_STORAGE_BACKEND in deps
        assert EXT_AETHER_AGENT_SERVICE in deps


# ---------------------------------------------------------------------------
# Test: get_task_status
# ---------------------------------------------------------------------------

class TestGetTaskStatus:
    """Verify status lookup for various task states."""

    @pytest.mark.asyncio
    async def test_not_found(self, mock_variables):
        service = _make_service(mock_variables)
        status = await service.get_task_status("atask_nonexistent")
        assert status == TaskStatus.NOT_FOUND

    @pytest.mark.asyncio
    async def test_aether_schedule_status_not_tracked_locally(self, mock_variables):
        """Aether-managed schedules (ml-*) don't store local status."""
        service = _make_service(mock_variables)
        _attach_mock_client(service)

        schedule_id = await service.schedule_recurring(
            task_type="test",
            interval_seconds=300,
            payload={},
        )
        assert schedule_id == "ml-test"

        # ml-* schedules are managed by Aether, not tracked in local status
        status = await service.get_task_status(schedule_id)
        assert status == TaskStatus.NOT_FOUND



# ---------------------------------------------------------------------------
# Test: _build_task_envelope helper (still used internally for delayed tasks)
# ---------------------------------------------------------------------------

class TestBuildTaskEnvelope:
    """Verify envelope structure matches the documented schema."""

    def test_envelope_structure(self):
        from memorylayer_server.services.tasks.aether import _build_task_envelope

        envelope = _build_task_envelope(
            task_id="atask_abc123",
            task_type="decay_memories",
            payload={"foo": "bar"},
            priority=7,
            auth_context={"tenant_id": "t1"},
        )

        assert envelope["task_id"] == "atask_abc123"
        assert envelope["task_type"] == "decay_memories"
        assert envelope["payload"] == {"foo": "bar"}
        assert envelope["priority"] == 7
        assert envelope["auth_context"] == {"tenant_id": "t1"}
        assert "scheduled_at" in envelope

    def test_envelope_defaults(self):
        from memorylayer_server.services.tasks.aether import _build_task_envelope

        envelope = _build_task_envelope(
            task_id="atask_xyz",
            task_type="test",
            payload={},
        )

        assert envelope["priority"] == 5
        assert envelope["auth_context"] == {}
