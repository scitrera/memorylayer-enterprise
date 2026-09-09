# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""
Unit tests for tiering API authentication and authorization enforcement.

Verifies that each endpoint:
- Calls build_context to extract the caller's identity
- Calls require_authorization with resource="admin" and the correct action
- Propagates ctx.workspace_id to downstream service calls
- Returns 401/403 on auth failures
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from memorylayer_server.models.auth import RequestContext
from memorylayer_server.services.authentication import AuthenticationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(workspace_id: str = "ws-admin-test") -> RequestContext:
    """Build a minimal RequestContext for testing."""
    return RequestContext(
        tenant_id="tenant-1",
        workspace_id=workspace_id,
        user_id="admin-user",
        metadata={},
    )


def _make_auth_service(ctx: RequestContext | None = None) -> AsyncMock:
    """Create a mock AuthenticationService that returns the given context."""
    svc = AsyncMock()
    svc.build_context = AsyncMock(return_value=ctx or _make_ctx())
    return svc


def _make_authz_service(deny: bool = False) -> AsyncMock:
    """Create a mock AuthorizationService that allows or denies."""
    svc = AsyncMock()
    if deny:
        svc.require_authorization = AsyncMock(
            side_effect=HTTPException(status_code=403, detail="Access denied to admin")
        )
    else:
        svc.require_authorization = AsyncMock(return_value=None)
    return svc


def _make_tiering_stats():
    """Build a minimal mock TieringStats."""
    stats = MagicMock()
    stats.hot_memory_count = 10
    stats.cold_memory_count = 5
    stats.hot_storage_bytes = 1024
    stats.cold_storage_bytes = 512
    stats.compression_ratio = 2.0
    stats.estimated_savings_bytes = 256
    stats.archival_candidates_count = 3
    return stats


def _make_archival_result():
    """Build a minimal mock ArchivalResult."""
    result = MagicMock()
    result.archived_count = 2
    result.failed_count = 0
    result.archived_memory_ids = ["mem-1", "mem-2"]
    result.failed_memory_ids = []
    return result


def _make_restore_result():
    """Build a minimal mock RestoreResult."""
    result = MagicMock()
    result.restored_count = 1
    result.failed_count = 0
    result.restored_memory_ids = ["mem-1"]
    result.failed_memory_ids = []
    return result


def _make_workspace_storage(settings: dict | None = None) -> AsyncMock:
    """Create a mock storage backend exposing get_workspace/update_workspace.

    The workspace's settings are mutated in place by update_workspace so the
    mock provides read-after-write behaviour for tiering-config tests.
    """
    storage = AsyncMock()
    state = {"settings": dict(settings or {})}

    def _ws():
        ws = MagicMock()
        ws.settings = dict(state["settings"])
        return ws

    async def _get_workspace(workspace_id):
        return _ws()

    async def _update_workspace(workspace_id, **updates):
        if "settings" in updates:
            state["settings"] = dict(updates["settings"])
        return _ws()

    storage.get_workspace = AsyncMock(side_effect=_get_workspace)
    storage.update_workspace = AsyncMock(side_effect=_update_workspace)
    return storage


def _make_tiering_service() -> AsyncMock:
    """Create a minimal mock TieringService."""
    svc = AsyncMock()
    svc.get_tiering_stats = AsyncMock(return_value=_make_tiering_stats())
    svc.archive_memories = AsyncMock(return_value=_make_archival_result())
    svc.restore_memories = AsyncMock(return_value=_make_restore_result())
    svc.run_warmup_cycle = AsyncMock(return_value=_make_restore_result())
    return svc


# ---------------------------------------------------------------------------
# Tests for tiering stats endpoint
# ---------------------------------------------------------------------------

class TestTieringStatsAuth:
    """Tests for GET /v1/tiering/stats auth wiring."""

    @pytest.mark.asyncio
    async def test_get_stats_calls_build_context_and_require_authorization(self):
        """GET /v1/tiering/stats must call build_context then require_authorization."""
        from memorylayer_saas.api.v1.tiering import get_tiering_stats

        ctx = _make_ctx("ws-stats")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        tiering_svc = _make_tiering_service()
        request = MagicMock()
        logger = MagicMock()

        await get_tiering_stats(
            http_request=request,
            include_candidates=True,
            auth_service=auth_svc,
            authz_service=authz_svc,
            tiering_service=tiering_svc,
            logger=logger,
        )

        auth_svc.build_context.assert_awaited_once_with(request)
        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "admin", "read", workspace_id="ws-stats"
        )

    @pytest.mark.asyncio
    async def test_get_stats_uses_ctx_workspace_id(self):
        """get_tiering_stats must pass ctx.workspace_id to service."""
        from memorylayer_saas.api.v1.tiering import get_tiering_stats

        ctx = _make_ctx("ws-stats-propagated")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        tiering_svc = _make_tiering_service()
        logger = MagicMock()

        await get_tiering_stats(
            http_request=MagicMock(),
            include_candidates=True,
            auth_service=auth_svc,
            authz_service=authz_svc,
            tiering_service=tiering_svc,
            logger=logger,
        )

        tiering_svc.get_tiering_stats.assert_awaited_once()
        call_kwargs = tiering_svc.get_tiering_stats.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-stats-propagated"

    @pytest.mark.asyncio
    async def test_get_stats_auth_error_raises_401(self):
        """Authentication failure on stats must surface as HTTP 401."""
        from memorylayer_saas.api.v1.tiering import get_tiering_stats

        auth_svc = _make_auth_service()
        auth_svc.build_context = AsyncMock(
            side_effect=AuthenticationError("invalid token", status_code=401)
        )
        authz_svc = _make_authz_service()
        tiering_svc = _make_tiering_service()
        logger = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await get_tiering_stats(
                http_request=MagicMock(),
                include_candidates=True,
                auth_service=auth_svc,
                authz_service=authz_svc,
                tiering_service=tiering_svc,
                logger=logger,
            )
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_stats_authz_denied_raises_403(self):
        """Authorization denial on stats must raise 403."""
        from memorylayer_saas.api.v1.tiering import get_tiering_stats

        ctx = _make_ctx()
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service(deny=True)
        tiering_svc = _make_tiering_service()
        logger = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await get_tiering_stats(
                http_request=MagicMock(),
                include_candidates=True,
                auth_service=auth_svc,
                authz_service=authz_svc,
                tiering_service=tiering_svc,
                logger=logger,
            )
        assert exc_info.value.status_code == 403
        tiering_svc.get_tiering_stats.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests for archive memories endpoint
# ---------------------------------------------------------------------------

class TestTieringArchiveAuth:
    """Tests for POST /v1/tiering/archive auth wiring."""

    @pytest.mark.asyncio
    async def test_archive_calls_auth_write(self):
        """POST /v1/tiering/archive must require admin/write."""
        from memorylayer_saas.api.v1.tiering import archive_memories

        ctx = _make_ctx("ws-archive")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        tiering_svc = _make_tiering_service()
        request = MagicMock()
        logger = MagicMock()

        archive_req = MagicMock()
        archive_req.memory_ids = ["mem-1"]
        archive_req.auto_detect = False
        archive_req.max_importance = None
        archive_req.max_access_count = None
        archive_req.older_than_days = None
        archive_req.batch_size = 100

        await archive_memories(
            http_request=request,
            request=archive_req,
            auth_service=auth_svc,
            authz_service=authz_svc,
            tiering_service=tiering_svc,
            logger=logger,
        )

        auth_svc.build_context.assert_awaited_once_with(request)
        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "admin", "write", workspace_id="ws-archive"
        )

    @pytest.mark.asyncio
    async def test_archive_uses_ctx_workspace_id(self):
        """archive_memories must pass ctx.workspace_id to service."""
        from memorylayer_saas.api.v1.tiering import archive_memories

        ctx = _make_ctx("ws-archive-propagated")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        tiering_svc = _make_tiering_service()
        logger = MagicMock()

        archive_req = MagicMock()
        archive_req.memory_ids = ["mem-1"]
        archive_req.auto_detect = False
        archive_req.max_importance = None
        archive_req.max_access_count = None
        archive_req.older_than_days = None
        archive_req.batch_size = 100

        await archive_memories(
            http_request=MagicMock(),
            request=archive_req,
            auth_service=auth_svc,
            authz_service=authz_svc,
            tiering_service=tiering_svc,
            logger=logger,
        )

        tiering_svc.archive_memories.assert_awaited_once()
        call_kwargs = tiering_svc.archive_memories.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-archive-propagated"

    @pytest.mark.asyncio
    async def test_archive_authz_denied_skips_service(self):
        """Authorization denied on archive must not call tiering service."""
        from memorylayer_saas.api.v1.tiering import archive_memories

        ctx = _make_ctx()
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service(deny=True)
        tiering_svc = _make_tiering_service()
        logger = MagicMock()

        archive_req = MagicMock()
        archive_req.memory_ids = ["mem-1"]
        archive_req.auto_detect = False

        with pytest.raises(HTTPException) as exc_info:
            await archive_memories(
                http_request=MagicMock(),
                request=archive_req,
                auth_service=auth_svc,
                authz_service=authz_svc,
                tiering_service=tiering_svc,
                logger=logger,
            )
        assert exc_info.value.status_code == 403
        tiering_svc.archive_memories.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests for restore memories endpoint
# ---------------------------------------------------------------------------

class TestTieringRestoreAuth:
    """Tests for POST /v1/tiering/restore auth wiring."""

    @pytest.mark.asyncio
    async def test_restore_calls_auth_write(self):
        """POST /v1/tiering/restore must require admin/write."""
        from memorylayer_saas.api.v1.tiering import restore_memories

        ctx = _make_ctx("ws-restore")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        tiering_svc = _make_tiering_service()
        request = MagicMock()
        logger = MagicMock()

        restore_req = MagicMock()
        restore_req.memory_ids = ["mem-1"]
        restore_req.regenerate_embeddings = False

        await restore_memories(
            http_request=request,
            request=restore_req,
            auth_service=auth_svc,
            authz_service=authz_svc,
            tiering_service=tiering_svc,
            logger=logger,
        )

        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "admin", "write", workspace_id="ws-restore"
        )

    @pytest.mark.asyncio
    async def test_restore_uses_ctx_workspace_id(self):
        """restore_memories must pass ctx.workspace_id to service."""
        from memorylayer_saas.api.v1.tiering import restore_memories

        ctx = _make_ctx("ws-restore-propagated")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        tiering_svc = _make_tiering_service()
        logger = MagicMock()

        restore_req = MagicMock()
        restore_req.memory_ids = ["mem-1"]
        restore_req.regenerate_embeddings = False

        await restore_memories(
            http_request=MagicMock(),
            request=restore_req,
            auth_service=auth_svc,
            authz_service=authz_svc,
            tiering_service=tiering_svc,
            logger=logger,
        )

        call_kwargs = tiering_svc.restore_memories.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-restore-propagated"


# ---------------------------------------------------------------------------
# Tests for config endpoints
# ---------------------------------------------------------------------------

class TestTieringConfigAuth:
    """Tests for GET/PUT /v1/tiering/config auth wiring."""

    @pytest.mark.asyncio
    async def test_get_config_calls_auth_read(self):
        """GET /v1/tiering/config must require admin/read."""
        from unittest.mock import patch
        from memorylayer_saas.api.v1.tiering import get_tiering_config_endpoint

        ctx = _make_ctx("ws-config")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        logger = MagicMock()
        request = MagicMock()

        mock_tiering_service = MagicMock()
        mock_tiering_service.DEFAULT_OLDER_THAN_DAYS = 30
        mock_tiering_service.DEFAULT_MAX_IMPORTANCE = 0.3
        mock_tiering_service.DEFAULT_MAX_ACCESS_COUNT = 5
        mock_tiering_service.DEFAULT_WARMUP_ACCESS_THRESHOLD = 3

        mock_audit_service = AsyncMock()
        storage = _make_workspace_storage()

        with patch("memorylayer_saas.api.v1.tiering.get_extension", return_value=storage):
            await get_tiering_config_endpoint(
                http_request=request,
                auth_service=auth_svc,
                authz_service=authz_svc,
                tiering_service=mock_tiering_service,
                audit_service=mock_audit_service,
                v=MagicMock(),
                logger=logger,
            )

        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "admin", "read", workspace_id="ws-config"
        )

    @pytest.mark.asyncio
    async def test_update_config_calls_auth_write(self):
        """PUT /v1/tiering/config must require admin/write."""
        from unittest.mock import patch
        from memorylayer_saas.api.v1.tiering import update_tiering_config

        ctx = _make_ctx("ws-config-update")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        logger = MagicMock()
        request = MagicMock()

        config_req = MagicMock()
        config_req.model_dump.return_value = {}

        mock_tiering_service = MagicMock()
        mock_tiering_service.DEFAULT_OLDER_THAN_DAYS = 30
        mock_tiering_service.DEFAULT_MAX_IMPORTANCE = 0.3
        mock_tiering_service.DEFAULT_MAX_ACCESS_COUNT = 5
        mock_tiering_service.DEFAULT_WARMUP_ACCESS_THRESHOLD = 3

        mock_audit_service = AsyncMock()
        storage = _make_workspace_storage()

        with patch("memorylayer_saas.api.v1.tiering.get_extension", return_value=storage):
            await update_tiering_config(
                http_request=request,
                request=config_req,
                auth_service=auth_svc,
                authz_service=authz_svc,
                tiering_service=mock_tiering_service,
                audit_service=mock_audit_service,
                v=MagicMock(),
                logger=logger,
            )

        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "admin", "write", workspace_id="ws-config-update"
        )


# ---------------------------------------------------------------------------
# Tests for warmup endpoint
# ---------------------------------------------------------------------------

class TestTieringWarmupAuth:
    """Tests for POST /v1/tiering/warmup auth wiring."""

    @pytest.mark.asyncio
    async def test_warmup_calls_auth_write(self):
        """POST /v1/tiering/warmup must require admin/write."""
        from memorylayer_saas.api.v1.tiering import run_warmup_cycle

        ctx = _make_ctx("ws-warmup")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        tiering_svc = _make_tiering_service()
        request = MagicMock()
        logger = MagicMock()

        await run_warmup_cycle(
            http_request=request,
            access_threshold=None,
            limit=100,
            auth_service=auth_svc,
            authz_service=authz_svc,
            tiering_service=tiering_svc,
            logger=logger,
        )

        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "admin", "write", workspace_id="ws-warmup"
        )

    @pytest.mark.asyncio
    async def test_warmup_uses_ctx_workspace_id(self):
        """run_warmup_cycle must pass ctx.workspace_id to service."""
        from memorylayer_saas.api.v1.tiering import run_warmup_cycle

        ctx = _make_ctx("ws-warmup-propagated")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        tiering_svc = _make_tiering_service()
        logger = MagicMock()

        await run_warmup_cycle(
            http_request=MagicMock(),
            access_threshold=None,
            limit=100,
            auth_service=auth_svc,
            authz_service=authz_svc,
            tiering_service=tiering_svc,
            logger=logger,
        )

        tiering_svc.run_warmup_cycle.assert_awaited_once()
        call_kwargs = tiering_svc.run_warmup_cycle.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-warmup-propagated"

    @pytest.mark.asyncio
    async def test_warmup_authz_denied_skips_service(self):
        """Authorization denied on warmup must not call tiering service."""
        from memorylayer_saas.api.v1.tiering import run_warmup_cycle

        ctx = _make_ctx()
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service(deny=True)
        tiering_svc = _make_tiering_service()
        logger = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await run_warmup_cycle(
                http_request=MagicMock(),
                access_threshold=None,
                limit=100,
                auth_service=auth_svc,
                authz_service=authz_svc,
                tiering_service=tiering_svc,
                logger=logger,
            )
        assert exc_info.value.status_code == 403
        tiering_svc.run_warmup_cycle.assert_not_awaited()
