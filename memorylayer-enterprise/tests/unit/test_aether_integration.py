# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Integration tests for Aether-backed enterprise services.

All Aether SDK interactions are mocked — no live gateway is required.
Covers:
- AetherKVRateLimitService     (kv_increment path via shared client)
- AetherTaskService             (native create_task path via shared client)
- AetherServiceConnection       (unified client: task-assignment dispatch,
                                 client/workspace properties)

Phase 1 (Aether convergence) note: the legacy ``on_message`` dispatch
path that handled ``recall``/``remember``/``search`` actions over Aether
agent messages has been removed in favour of REST-over-Aether (Phase 2).
The corresponding tests were retired with that handler — REST endpoints
are exercised by the FastAPI-level test suites instead.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_variables():
    """Minimal Variables stub that satisfies ``get_logger`` and ``environ``."""
    v = MagicMock()
    v.environ = MagicMock(side_effect=lambda key, default, **kwargs: default)
    return v


@pytest.fixture()
def mock_aether_client():
    """Fully-mocked AsyncAgentClient with the methods used by the services."""
    client = AsyncMock()
    client.kv_increment = AsyncMock()
    client.kv_get = AsyncMock(return_value=None)
    client.create_task = AsyncMock()
    client.send_message_to_agent = AsyncMock()
    client.send_message_to_user_session = AsyncMock()
    client.close = AsyncMock()
    client.complete_task = AsyncMock()
    client.fail_task = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kv_response(counter_value: int):
    """Return a minimal KVResponse-like object with ``counter_value``."""
    return SimpleNamespace(counter_value=counter_value)


# ===========================================================================
# AetherKVRateLimitService
# ===========================================================================


def _make_rate_limit_service(mock_variables, **overrides):
    """Construct AetherKVRateLimitService with test defaults."""
    from memorylayer_saas.services.rate_limit.aether_kv import AetherKVRateLimitService

    kwargs = dict(
        default_limit=100,
        default_window_seconds=60,
    )
    kwargs.update(overrides)
    return AetherKVRateLimitService(mock_variables, **kwargs)


class TestAetherKVRateLimitService:
    """Tests for AetherKVRateLimitService.check_rate_limit."""

    @pytest.mark.asyncio
    async def test_rate_limit_allows_under_limit(self, mock_variables, mock_aether_client):
        """Returns allowed=True and correct remaining when counter is below the limit."""
        mock_aether_client.kv_increment.return_value = _kv_response(counter_value=5)

        service = _make_rate_limit_service(mock_variables, default_limit=100)
        service._client = mock_aether_client

        result = await service.check_rate_limit("user:alice")

        assert result.allowed is True
        assert result.remaining == 95
        assert result.limit == 100

    @pytest.mark.asyncio
    async def test_rate_limit_denies_over_limit(self, mock_variables, mock_aether_client):
        """Returns allowed=False and remaining=0 when counter exceeds the limit."""
        mock_aether_client.kv_increment.return_value = _kv_response(counter_value=101)

        service = _make_rate_limit_service(mock_variables, default_limit=100)
        service._client = mock_aether_client

        result = await service.check_rate_limit("user:bob")

        assert result.allowed is False
        assert result.remaining == 0

    @pytest.mark.asyncio
    async def test_rate_limit_allows_at_exact_limit(self, mock_variables, mock_aether_client):
        """Counter equal to limit is still allowed (boundary: count <= limit)."""
        mock_aether_client.kv_increment.return_value = _kv_response(counter_value=100)

        service = _make_rate_limit_service(mock_variables, default_limit=100)
        service._client = mock_aether_client

        result = await service.check_rate_limit("user:carol")

        assert result.allowed is True
        assert result.remaining == 0

    @pytest.mark.asyncio
    async def test_rate_limit_fails_open_on_timeout(self, mock_variables, mock_aether_client):
        """When kv_increment returns None (timeout), the request is allowed (fail-open)."""
        mock_aether_client.kv_increment.return_value = None

        service = _make_rate_limit_service(mock_variables, default_limit=100)
        service._client = mock_aether_client

        result = await service.check_rate_limit("user:dave")

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_rate_limit_fails_open_on_exception(self, mock_variables, mock_aether_client):
        """When kv_increment raises, the request is allowed (fail-open)."""
        mock_aether_client.kv_increment.side_effect = RuntimeError("gRPC error")

        service = _make_rate_limit_service(mock_variables, default_limit=100)
        service._client = mock_aether_client

        result = await service.check_rate_limit("user:eve")

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_rate_limit_fails_open_when_not_connected(self, mock_variables):
        """When the client is not connected (None), the request is allowed (fail-open)."""
        service = _make_rate_limit_service(mock_variables, default_limit=10)
        # Deliberately leave _client as None

        result = await service.check_rate_limit("user:frank")

        assert result.allowed is True
        assert result.remaining == 10

    @pytest.mark.asyncio
    async def test_rate_limit_key_format_contains_identifier(self, mock_variables, mock_aether_client):
        """The key passed to kv_increment contains the rate-limit identifier."""
        mock_aether_client.kv_increment.return_value = _kv_response(counter_value=1)

        service = _make_rate_limit_service(mock_variables)
        service._client = mock_aether_client

        await service.check_rate_limit("tenant:acme")

        call_kwargs = mock_aether_client.kv_increment.call_args.kwargs
        assert "tenant:acme" in call_kwargs["key"]

    @pytest.mark.asyncio
    async def test_rate_limit_ttl_matches_window_seconds(self, mock_variables, mock_aether_client):
        """The ttl parameter passed to kv_increment matches the configured window."""
        mock_aether_client.kv_increment.return_value = _kv_response(counter_value=1)

        service = _make_rate_limit_service(mock_variables, default_window_seconds=120)
        service._client = mock_aether_client

        await service.check_rate_limit("user:grace")

        call_kwargs = mock_aether_client.kv_increment.call_args.kwargs
        assert call_kwargs["ttl"] == 120

    @pytest.mark.asyncio
    async def test_rate_limit_explicit_limit_overrides_default(self, mock_variables, mock_aether_client):
        """An explicit limit argument overrides the service default."""
        mock_aether_client.kv_increment.return_value = _kv_response(counter_value=3)

        service = _make_rate_limit_service(mock_variables, default_limit=100)
        service._client = mock_aether_client

        result = await service.check_rate_limit("user:henry", limit=5)

        assert result.limit == 5
        assert result.remaining == 2

    @pytest.mark.asyncio
    async def test_rate_limit_explicit_window_overrides_default(self, mock_variables, mock_aether_client):
        """An explicit window_seconds argument is forwarded to kv_increment as ttl."""
        mock_aether_client.kv_increment.return_value = _kv_response(counter_value=1)

        service = _make_rate_limit_service(mock_variables, default_window_seconds=60)
        service._client = mock_aether_client

        await service.check_rate_limit("user:iris", window_seconds=300)

        call_kwargs = mock_aether_client.kv_increment.call_args.kwargs
        assert call_kwargs["ttl"] == 300

    @pytest.mark.asyncio
    async def test_bind_client_sets_client_and_workspace(self, mock_variables, mock_aether_client):
        """bind_client obtains the shared client from an agent service."""
        service = _make_rate_limit_service(mock_variables)

        agent_svc = MagicMock()
        agent_svc.client = mock_aether_client
        agent_svc.workspace = "prod-workspace"

        service.bind_client(agent_svc)

        assert service._client is mock_aether_client
        assert service._workspace == "prod-workspace"


# ===========================================================================
# AetherTaskService — native create_task path
# ===========================================================================


def _make_task_service(mock_variables, **overrides):
    """Construct AetherTaskService with test defaults."""
    from memorylayer_saas.services.tasks import AetherTaskService

    kwargs = dict(
        tasks_enabled=True,
    )
    kwargs.update(overrides)
    return AetherTaskService(mock_variables, **kwargs)


class TestAetherTaskServiceNativePath:
    """Tests for the Aether native create_task path via shared client."""

    @pytest.mark.asyncio
    async def test_schedule_task_calls_create_task(self, mock_variables, mock_aether_client):
        """schedule_task calls create_task on the shared Aether client."""
        service = _make_task_service(mock_variables)
        service._client = mock_aether_client

        task_id = await service.schedule_task(
            task_type="decay_memories",
            payload={"workspace_id": "ws-1"},
        )

        assert task_id.startswith("atask_")
        mock_aether_client.create_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_task_type_is_prefixed_with_memorylayer_task(self, mock_variables, mock_aether_client):
        """The task_type passed to create_task is prefixed with 'memorylayer-task.'."""
        service = _make_task_service(mock_variables)
        service._client = mock_aether_client

        await service.schedule_task(task_type="enrich_memories", payload={})

        call_kwargs = mock_aether_client.create_task.call_args.kwargs
        assert call_kwargs["task_type"] == "memorylayer-task.enrich_memories"

    @pytest.mark.asyncio
    async def test_task_uses_configured_workspace(self, mock_variables, mock_aether_client):
        """The workspace passed to create_task matches the service's configured workspace."""
        service = _make_task_service(mock_variables)
        service._client = mock_aether_client
        service._workspace = "prod-workspace"

        await service.schedule_task(task_type="cleanup", payload={})

        call_kwargs = mock_aether_client.create_task.call_args.kwargs
        assert call_kwargs["workspace"] == "prod-workspace"

    @pytest.mark.asyncio
    async def test_task_metadata_includes_task_id(self, mock_variables, mock_aether_client):
        """The metadata dict passed to create_task contains the task_id key."""
        service = _make_task_service(mock_variables)
        service._client = mock_aether_client

        task_id = await service.schedule_task(task_type="test_task", payload={"foo": "bar"})

        call_kwargs = mock_aether_client.create_task.call_args.kwargs
        metadata = call_kwargs["metadata"]
        assert metadata["task_id"] == task_id

    @pytest.mark.asyncio
    async def test_task_payload_is_msgpack_binary(self, mock_variables, mock_aether_client):
        """The payload is sent as msgpack-serialized bytes."""
        service = _make_task_service(mock_variables)
        service._client = mock_aether_client

        await service.schedule_task(task_type="test_task", payload={"key": "value", "count": 7})

        call_kwargs = mock_aether_client.create_task.call_args.kwargs
        assert isinstance(call_kwargs["payload"], bytes)

        from scitrera_rt_data.serialization.msgpack import msgpack_deserialize
        deserialized = msgpack_deserialize(call_kwargs["payload"])
        assert deserialized["key"] == "value"
        assert deserialized["count"] == 7

    @pytest.mark.asyncio
    async def test_task_status_transitions_to_running(self, mock_variables, mock_aether_client):
        """After a successful create_task call, the task status is RUNNING."""
        from memorylayer_server.services.tasks.base import TaskStatus

        service = _make_task_service(mock_variables)
        service._client = mock_aether_client

        task_id = await service.schedule_task(task_type="decay_memories", payload={})
        status = await service.get_task_status(task_id)

        assert status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_task_failure_marks_task_failed(self, mock_variables, mock_aether_client):
        """When create_task raises, the task status is FAILED."""
        from memorylayer_server.services.tasks.base import TaskStatus

        mock_aether_client.create_task.side_effect = RuntimeError("Aether unavailable")

        service = _make_task_service(mock_variables)
        service._client = mock_aether_client

        task_id = await service.schedule_task(task_type="decay_memories", payload={})
        status = await service.get_task_status(task_id)

        assert status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_task_without_connection_marks_failed(self, mock_variables):
        """When not connected, create_task path marks the task FAILED without raising."""
        from memorylayer_server.services.tasks.base import TaskStatus

        service = _make_task_service(mock_variables)
        # Deliberately leave _client as None

        task_id = await service.schedule_task(task_type="decay_memories", payload={})
        status = await service.get_task_status(task_id)

        assert status == TaskStatus.FAILED


# ===========================================================================
# AetherServiceConnection — unified client (task-assignment dispatch +
# client/workspace properties)
#
# Phase 1 (Aether convergence): the legacy on_message dispatch path that
# handled ``recall``/``remember``/``search`` actions over Aether agent
# messages was removed.  Those operations are exposed exclusively via REST
# now (Phase 2 covers REST-over-Aether), and the removed handler is no
# longer tested here — the FastAPI test suites cover the canonical path.
# ===========================================================================


def _make_service_connection(mock_variables, **overrides):
    """Construct AetherServiceConnection with test defaults."""
    # Prefer the canonical OSS import path so the tests exercise the new
    # module directly.  The enterprise shim still works (it re-exports
    # ``AetherAgentService`` as an alias of ``AetherServiceConnection``)
    # but tests should pin to the new identifiers.
    from memorylayer_server.services.aether_service import AetherServiceConnection

    kwargs = dict(
        gateway_addr="localhost:50051",
        workspace="test-workspace",
        specifier="main",
    )
    kwargs.update(overrides)
    return AetherServiceConnection(mock_variables, **kwargs)


class TestAetherServiceConnectionUnifiedClient:
    """Tests for the unified client features of AetherServiceConnection."""

    def test_client_property_returns_none_when_not_connected(self, mock_variables):
        service = _make_service_connection(mock_variables)
        assert service.client is None

    def test_client_property_returns_client_when_connected(
        self, mock_variables, mock_aether_client
    ):
        service = _make_service_connection(mock_variables)
        service._client = mock_aether_client
        assert service.client is mock_aether_client

    def test_workspace_property(self, mock_variables):
        service = _make_service_connection(mock_variables, workspace="prod")
        assert service.workspace == "prod"

    def test_set_task_assignment_handler(self, mock_variables):
        service = _make_service_connection(mock_variables)

        async def my_handler(assignment):
            pass

        service.set_task_assignment_handler(my_handler)
        assert service._task_assignment_handler is my_handler

    @pytest.mark.asyncio
    async def test_task_assignment_dispatches_to_handler(self, mock_variables):
        service = _make_service_connection(mock_variables)
        handler = AsyncMock()
        service.set_task_assignment_handler(handler)

        fake_assignment = MagicMock()
        await service._on_task_assignment(fake_assignment)
        # Handler is dispatched via asyncio.create_task; yield to let it run
        await asyncio.sleep(0)

        handler.assert_awaited_once_with(fake_assignment)

    @pytest.mark.asyncio
    async def test_task_assignment_without_handler_logs_warning(self, mock_variables):
        """When no handler is registered, task assignments are logged but not processed."""
        service = _make_service_connection(mock_variables)

        fake_assignment = MagicMock()
        fake_assignment.task_type = "test_task"

        # Should not raise
        await service._on_task_assignment(fake_assignment)
