"""Unit tests for DefaultContextService.delete_context wiring.

Verifies that the service:
- delegates to the storage backend's delete_context and returns its result
- refuses to delete the _default context
- surfaces a NotImplementedError (rather than silently returning False) when the
  storage backend does not implement delete_context
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.models.workspace import Context

from memorylayer_saas.services.context_kv.default import DefaultContextService
from memorylayer_saas.services.context_kv.base import DEFAULT_CONTEXT_NAME


def _make_context(workspace_id: str = "ws", context_id: str = "ctx_1", name: str = "project-x") -> Context:
    return Context(
        id=context_id,
        workspace_id=workspace_id,
        name=name,
        description=None,
        settings={},
    )


def _make_service(storage) -> DefaultContextService:
    # Bypass __init__ logging/v wiring; set fields directly.
    svc = DefaultContextService.__new__(DefaultContextService)
    svc._storage = storage
    svc.logger = MagicMock()
    return svc


class TestServiceDeleteContext:
    @pytest.mark.asyncio
    async def test_delegates_to_storage_and_returns_true(self):
        storage = AsyncMock()
        storage.get_context = AsyncMock(return_value=_make_context())
        storage.delete_context = AsyncMock(return_value=True)
        svc = _make_service(storage)

        result = await svc.delete_context("ws", "ctx_1")

        assert result is True
        storage.delete_context.assert_awaited_once_with("ws", "ctx_1")

    @pytest.mark.asyncio
    async def test_returns_false_when_storage_reports_not_found(self):
        storage = AsyncMock()
        storage.get_context = AsyncMock(return_value=None)
        storage.delete_context = AsyncMock(return_value=False)
        svc = _make_service(storage)

        result = await svc.delete_context("ws", "ctx_missing")

        assert result is False
        storage.delete_context.assert_awaited_once_with("ws", "ctx_missing")

    @pytest.mark.asyncio
    async def test_refuses_to_delete_default_context(self):
        storage = AsyncMock()
        storage.get_context = AsyncMock(
            return_value=_make_context(name=DEFAULT_CONTEXT_NAME)
        )
        storage.delete_context = AsyncMock(return_value=True)
        svc = _make_service(storage)

        with pytest.raises(ValueError):
            await svc.delete_context("ws", "ctx_default")
        storage.delete_context.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_backend_lacks_delete_context(self):
        # A backend object without a delete_context attribute.
        class _NoDelete:
            async def get_context(self, workspace_id, context_id):
                return _make_context()

        svc = _make_service(_NoDelete())

        with pytest.raises(NotImplementedError):
            await svc.delete_context("ws", "ctx_1")
