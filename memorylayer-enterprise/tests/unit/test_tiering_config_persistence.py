"""
Unit tests for tiering config persistence (read-after-write).

Verifies that PUT /v1/tiering/config persists per-workspace overrides to
``workspace.settings["tiering"]`` and that GET /v1/tiering/config reads them
back, so the two endpoints are consistent. Also verifies that the audit
"update" event is only recorded when something actually changed.

These tests use mock services (no live framework / Postgres) consistent with
test_tiering_api_auth.py.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memorylayer_server.models.auth import RequestContext

from memorylayer_saas.models.tiering import TieringConfigUpdateRequest


def _make_ctx(workspace_id: str = "ws-persist") -> RequestContext:
    return RequestContext(
        tenant_id="tenant-1",
        workspace_id=workspace_id,
        user_id="admin-user",
        metadata={},
    )


def _make_auth_service(ctx: RequestContext) -> AsyncMock:
    svc = AsyncMock()
    svc.build_context = AsyncMock(return_value=ctx)
    return svc


def _make_authz_service() -> AsyncMock:
    svc = AsyncMock()
    svc.require_authorization = AsyncMock(return_value=None)
    return svc


def _make_tiering_service() -> MagicMock:
    svc = MagicMock()
    svc.DEFAULT_OLDER_THAN_DAYS = 90
    svc.DEFAULT_MAX_IMPORTANCE = 0.3
    svc.DEFAULT_MAX_ACCESS_COUNT = 5
    svc.DEFAULT_WARMUP_ACCESS_THRESHOLD = 10
    return svc


def _make_workspace_storage(settings: dict | None = None) -> AsyncMock:
    """Mock storage with in-memory read-after-write workspace settings."""
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
    storage._state = state  # expose for assertions
    return storage


class TestTieringConfigPersistence:
    """PUT persists overrides; GET reads them back (read-after-write)."""

    @pytest.mark.asyncio
    async def test_put_persists_and_get_reflects_change(self):
        from memorylayer_saas.api.v1.tiering import (
            update_tiering_config,
            get_tiering_config_endpoint,
        )

        ctx = _make_ctx("ws-rw")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        tiering_svc = _make_tiering_service()
        audit_svc = AsyncMock()
        logger = MagicMock()
        storage = _make_workspace_storage()

        update = TieringConfigUpdateRequest(
            cold_tier_enabled=False,
            archival_age_days=30,
            warmup_access_threshold=7,
        )

        with patch("memorylayer_saas.api.v1.tiering.get_extension", return_value=storage):
            put_resp = await update_tiering_config(
                http_request=MagicMock(),
                request=update,
                auth_service=auth_svc,
                authz_service=authz_svc,
                tiering_service=tiering_svc,
                audit_service=audit_svc,
                v=MagicMock(),
                logger=logger,
            )

            # PUT response reflects the change immediately
            assert put_resp.cold_tier_enabled is False
            assert put_resp.archival_age_days == 30
            assert put_resp.warmup_access_threshold == 7
            # untouched field keeps the service default
            assert put_resp.min_importance_threshold == 0.3

            # Persisted under workspace.settings["tiering"]
            assert storage._state["settings"]["tiering"]["cold_tier_enabled"] is False
            assert storage._state["settings"]["tiering"]["archival_age_days"] == 30

            # GET reads back the same persisted values
            get_resp = await get_tiering_config_endpoint(
                http_request=MagicMock(),
                auth_service=_make_auth_service(ctx),
                authz_service=_make_authz_service(),
                tiering_service=tiering_svc,
                audit_service=AsyncMock(),
                v=MagicMock(),
                logger=logger,
            )

        assert get_resp.cold_tier_enabled is False
        assert get_resp.archival_age_days == 30
        assert get_resp.warmup_access_threshold == 7
        assert get_resp.min_importance_threshold == 0.3

    @pytest.mark.asyncio
    async def test_put_records_audit_on_real_change(self):
        from memorylayer_saas.api.v1.tiering import update_tiering_config

        ctx = _make_ctx("ws-audit-change")
        audit_svc = AsyncMock()
        storage = _make_workspace_storage()

        update = TieringConfigUpdateRequest(archival_age_days=45)

        with patch("memorylayer_saas.api.v1.tiering.get_extension", return_value=storage):
            await update_tiering_config(
                http_request=MagicMock(),
                request=update,
                auth_service=_make_auth_service(ctx),
                authz_service=_make_authz_service(),
                tiering_service=_make_tiering_service(),
                audit_service=audit_svc,
                v=MagicMock(),
                logger=MagicMock(),
            )

        audit_svc.record.assert_awaited_once()
        storage.update_workspace.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_put_empty_request_does_not_record_audit_or_persist(self):
        from memorylayer_saas.api.v1.tiering import update_tiering_config

        ctx = _make_ctx("ws-noop")
        audit_svc = AsyncMock()
        storage = _make_workspace_storage()

        update = TieringConfigUpdateRequest()  # all None -> nothing to change

        with patch("memorylayer_saas.api.v1.tiering.get_extension", return_value=storage):
            await update_tiering_config(
                http_request=MagicMock(),
                request=update,
                auth_service=_make_auth_service(ctx),
                authz_service=_make_authz_service(),
                tiering_service=_make_tiering_service(),
                audit_service=audit_svc,
                v=MagicMock(),
                logger=MagicMock(),
            )

        audit_svc.record.assert_not_awaited()
        storage.update_workspace.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_put_same_values_is_noop(self):
        """Writing the already-persisted values must not re-persist or audit."""
        from memorylayer_saas.api.v1.tiering import update_tiering_config

        ctx = _make_ctx("ws-same")
        audit_svc = AsyncMock()
        storage = _make_workspace_storage(
            settings={"tiering": {"archival_age_days": 60}}
        )

        update = TieringConfigUpdateRequest(archival_age_days=60)

        with patch("memorylayer_saas.api.v1.tiering.get_extension", return_value=storage):
            await update_tiering_config(
                http_request=MagicMock(),
                request=update,
                auth_service=_make_auth_service(ctx),
                authz_service=_make_authz_service(),
                tiering_service=_make_tiering_service(),
                audit_service=audit_svc,
                v=MagicMock(),
                logger=MagicMock(),
            )

        audit_svc.record.assert_not_awaited()
        storage.update_workspace.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_put_merges_with_existing_overrides(self):
        """A second PUT must merge, not clobber, prior overrides."""
        from memorylayer_saas.api.v1.tiering import update_tiering_config

        ctx = _make_ctx("ws-merge")
        storage = _make_workspace_storage(
            settings={"tiering": {"cold_tier_enabled": False}}
        )

        update = TieringConfigUpdateRequest(archival_age_days=15)

        with patch("memorylayer_saas.api.v1.tiering.get_extension", return_value=storage):
            resp = await update_tiering_config(
                http_request=MagicMock(),
                request=update,
                auth_service=_make_auth_service(ctx),
                authz_service=_make_authz_service(),
                tiering_service=_make_tiering_service(),
                audit_service=AsyncMock(),
                v=MagicMock(),
                logger=MagicMock(),
            )

        # both old and new override values present
        assert resp.cold_tier_enabled is False
        assert resp.archival_age_days == 15
        persisted = storage._state["settings"]["tiering"]
        assert persisted["cold_tier_enabled"] is False
        assert persisted["archival_age_days"] == 15

    @pytest.mark.asyncio
    async def test_get_defaults_match_recall_defaults(self):
        """GET with no persisted override must report cold_tier OFF (matching recall's absent-key default).

        EnterpriseMemoryService.recall reads workspace.settings["tiering"].get(
        "cold_tier_enabled", False) and .get("cold_tier_search_enabled", False).
        A fresh workspace with no override must report False from GET so operators
        see the same effective state that recall uses.
        """
        from memorylayer_saas.api.v1.tiering import get_tiering_config_endpoint

        ctx = _make_ctx("ws-defaults")
        storage = _make_workspace_storage()  # no tiering key persisted

        with patch("memorylayer_saas.api.v1.tiering.get_extension", return_value=storage):
            resp = await get_tiering_config_endpoint(
                http_request=MagicMock(),
                auth_service=_make_auth_service(ctx),
                authz_service=_make_authz_service(),
                tiering_service=_make_tiering_service(),
                audit_service=AsyncMock(),
                v=MagicMock(),
                logger=MagicMock(),
            )

        assert resp.cold_tier_enabled is False, (
            "cold_tier_enabled default must be False to match recall's absent-key default"
        )
        assert resp.cold_tier_search_enabled is False, (
            "cold_tier_search_enabled default must be False to match recall's absent-key default"
        )
