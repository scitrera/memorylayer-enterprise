# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""
Unit tests for Aether Authentication and Authorization services.

Tests cover:
- AetherAuthenticationService header extraction and fallback behavior
- AetherAuthorizationService access level permission mapping
- ALLOW/DENY decisions at each access level boundary
- Integration between authentication metadata and authorization decisions
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from memorylayer_server.models.authz import AuthorizationDecision, AuthorizationContext
from memorylayer_server.config import DEFAULT_TENANT_ID
from memorylayer_server.services.authentication import AuthenticationError

from memorylayer_saas.services.authentication import (
    AetherAuthenticationService,
    HEADER_AUTH_TENANT_ID,
    HEADER_AUTH_USER_ID,
    HEADER_AUTH_API_KEY_ID,
    HEADER_AUTH_WORKSPACE_ACCESS,
    HEADER_AUTH_PRINCIPAL_TYPE,
    HEADER_AUTH_SCOPES,
    META_ACCESS_LEVEL,
    META_PRINCIPAL_TYPE,
    META_SCOPES,
    DEFAULT_ACCESS_LEVEL,
)
from memorylayer_saas.services.authorization import (
    AetherAuthorizationService,
    ACCESS_NONE,
    ACCESS_READ,
    ACCESS_READWRITE,
    ACCESS_MANAGE,
    ACCESS_ADMIN,
    ACCESS_SUPERADMIN,
    get_required_access_level,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_request(headers: dict[str, str] | None = None) -> MagicMock:
    """Create a mock FastAPI Request with the given headers."""
    request = MagicMock()
    all_headers = headers or {}
    request.headers = MagicMock()
    request.headers.get = lambda key, default=None: all_headers.get(key, default)
    # build_context consults request.query_params.get("workspace_id"); without an
    # explicit stub the bare MagicMock returns a MagicMock that fails RequestContext
    # validation. Default to no query param so workspace resolution falls through.
    request.query_params = MagicMock()
    request.query_params.get = lambda key, default=None: default
    return request


@pytest.fixture
def mock_session_service():
    """Create a mock SessionService."""
    svc = MagicMock()
    svc.get = AsyncMock(return_value=None)
    svc.create_session = AsyncMock()
    return svc


@pytest.fixture
def mock_workspace_service():
    """Create a mock WorkspaceService."""
    svc = MagicMock()
    svc.ensure_workspace = AsyncMock()
    return svc


@pytest.fixture
def auth_service(mock_session_service, mock_workspace_service):
    """Create an AetherAuthenticationService backed by mocks."""
    return AetherAuthenticationService(
        session_service=mock_session_service,
        workspace_service=mock_workspace_service,
        implicit_session_create=True,
        logger=MagicMock(),
    )


@pytest.fixture
def dev_fallback_auth_service(mock_session_service, mock_workspace_service):
    """AetherAuthenticationService with the legacy default-tenant dev fallback ON."""
    return AetherAuthenticationService(
        session_service=mock_session_service,
        workspace_service=mock_workspace_service,
        implicit_session_create=True,
        allow_default_tenant_fallback=True,
        logger=MagicMock(),
    )


@pytest.fixture
def authz_service():
    """Create an AetherAuthorizationService."""
    return AetherAuthorizationService()


# ===========================================================================
# AetherAuthenticationService Tests
# ===========================================================================

class TestAetherAuthenticationBuildContext:
    """Tests for AetherAuthenticationService.build_context()."""

    @pytest.mark.asyncio
    async def test_extracts_all_gateway_headers(self, auth_service):
        """Test that all X-Auth-* headers are correctly extracted."""
        request = _make_request({
            HEADER_AUTH_TENANT_ID: "tenant-abc",
            HEADER_AUTH_USER_ID: "user-123",
            HEADER_AUTH_API_KEY_ID: "key-456",
            HEADER_AUTH_WORKSPACE_ACCESS: "30",
            HEADER_AUTH_PRINCIPAL_TYPE: "user",
            HEADER_AUTH_SCOPES: "read,write,admin",
        })

        ctx = await auth_service.build_context(request)

        assert ctx.tenant_id == "tenant-abc"
        assert ctx.user_id == "user-123"
        assert ctx.metadata[META_ACCESS_LEVEL] == 30
        assert ctx.metadata[META_PRINCIPAL_TYPE] == "user"
        assert ctx.metadata[META_SCOPES] == ["read", "write", "admin"]

    @pytest.mark.asyncio
    async def test_fail_closed_when_no_headers(self, auth_service):
        """Test fail-closed: absent tenant header raises 401 (no default-tenant fallback)."""
        request = _make_request({})

        with pytest.raises(AuthenticationError) as exc_info:
            await auth_service.build_context(request)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_dev_fallback_to_default_tenant_when_no_headers(self, dev_fallback_auth_service):
        """Test legacy default-tenant fallback only when the dev opt-in is enabled."""
        request = _make_request({})

        ctx = await dev_fallback_auth_service.build_context(request)

        assert ctx.tenant_id == DEFAULT_TENANT_ID
        assert ctx.user_id is None
        assert ctx.metadata[META_ACCESS_LEVEL] == DEFAULT_ACCESS_LEVEL
        assert ctx.metadata[META_PRINCIPAL_TYPE] is None
        assert ctx.metadata[META_SCOPES] == []

    @pytest.mark.asyncio
    async def test_partial_headers_tenant_only(self, auth_service):
        """Test with only tenant ID header present."""
        request = _make_request({
            HEADER_AUTH_TENANT_ID: "tenant-xyz",
        })

        ctx = await auth_service.build_context(request)

        assert ctx.tenant_id == "tenant-xyz"
        assert ctx.user_id is None
        assert ctx.metadata[META_ACCESS_LEVEL] == DEFAULT_ACCESS_LEVEL

    @pytest.mark.asyncio
    async def test_malformed_access_level_defaults_to_zero(self, auth_service):
        """Test that non-numeric access level falls back to 0."""
        request = _make_request({
            HEADER_AUTH_TENANT_ID: "tenant-abc",
            HEADER_AUTH_WORKSPACE_ACCESS: "not-a-number",
        })

        ctx = await auth_service.build_context(request)

        assert ctx.metadata[META_ACCESS_LEVEL] == DEFAULT_ACCESS_LEVEL

    @pytest.mark.asyncio
    async def test_workspace_resolution_from_header(self, auth_service):
        """Test workspace resolved from X-Workspace-ID header."""
        request = _make_request({
            HEADER_AUTH_TENANT_ID: "tenant-abc",
            "X-Workspace-ID": "ws-custom",
        })

        ctx = await auth_service.build_context(request)

        assert ctx.workspace_id == "ws-custom"

    @pytest.mark.asyncio
    async def test_workspace_defaults_when_not_provided(self, auth_service):
        """Test workspace defaults to _default when not specified."""
        request = _make_request({
            HEADER_AUTH_TENANT_ID: "tenant-abc",
        })

        ctx = await auth_service.build_context(request)

        assert ctx.workspace_id == "_default"

    @pytest.mark.asyncio
    async def test_session_resolution(self, auth_service, mock_session_service):
        """Test that X-Session-ID header triggers session lookup."""
        mock_session = MagicMock()
        mock_session.id = "sess-123"
        mock_session.workspace_id = "ws-from-session"
        mock_session_service.get = AsyncMock(return_value=mock_session)

        request = _make_request({
            HEADER_AUTH_TENANT_ID: "tenant-abc",
            "X-Session-ID": "sess-123",
        })

        ctx = await auth_service.build_context(request)

        assert ctx.session is mock_session
        mock_session_service.get.assert_awaited_once_with("sess-123")

    @pytest.mark.asyncio
    async def test_scopes_empty_string(self, auth_service):
        """Test that empty scopes header produces empty list."""
        request = _make_request({
            HEADER_AUTH_TENANT_ID: "tenant-abc",
            HEADER_AUTH_SCOPES: "",
        })

        ctx = await auth_service.build_context(request)

        assert ctx.metadata[META_SCOPES] == []

    @pytest.mark.asyncio
    async def test_scopes_with_whitespace(self, auth_service):
        """Test that scopes with whitespace are properly trimmed."""
        request = _make_request({
            HEADER_AUTH_TENANT_ID: "tenant-abc",
            HEADER_AUTH_SCOPES: " read , write , ",
        })

        ctx = await auth_service.build_context(request)

        assert ctx.metadata[META_SCOPES] == ["read", "write"]


class TestAetherAuthenticationIdentityExtraction:
    """Tests for _extract_identity_from_headers."""

    def test_full_identity(self, auth_service):
        """Test extraction with all identity headers present."""
        request = _make_request({
            HEADER_AUTH_TENANT_ID: "t-1",
            HEADER_AUTH_USER_ID: "u-1",
            HEADER_AUTH_API_KEY_ID: "k-1",
        })

        identity = auth_service._extract_identity_from_headers(request)

        assert identity.tenant_id == "t-1"
        assert identity.user_id == "u-1"
        assert identity.api_key_id == "k-1"

    def test_empty_user_id_becomes_none(self, auth_service):
        """Test that empty string user_id is normalized to None."""
        request = _make_request({
            HEADER_AUTH_TENANT_ID: "t-1",
            HEADER_AUTH_USER_ID: "",
        })

        identity = auth_service._extract_identity_from_headers(request)

        assert identity.user_id is None

    def test_missing_tenant_fails_closed(self, auth_service):
        """Test that missing tenant header fails closed with a 401."""
        request = _make_request({})

        with pytest.raises(AuthenticationError) as exc_info:
            auth_service._extract_identity_from_headers(request)

        assert exc_info.value.status_code == 401

    def test_missing_tenant_dev_fallback(self, dev_fallback_auth_service):
        """Test that the dev opt-in restores the default-tenant fallback."""
        request = _make_request({})

        identity = dev_fallback_auth_service._extract_identity_from_headers(request)

        assert identity.tenant_id == DEFAULT_TENANT_ID


class TestAetherAuthenticationAccessLevelExtraction:
    """Tests for _extract_access_level."""

    @pytest.mark.parametrize("header_value,expected", [
        ("0", 0),
        ("10", 10),
        ("20", 20),
        ("30", 30),
        ("40", 40),
        ("50", 50),
    ])
    def test_valid_access_levels(self, header_value, expected):
        """Test parsing of valid numeric access levels."""
        request = _make_request({HEADER_AUTH_WORKSPACE_ACCESS: header_value})
        assert AetherAuthenticationService._extract_access_level(request) == expected

    def test_missing_header(self):
        """Test that missing header returns default access level."""
        request = _make_request({})
        assert AetherAuthenticationService._extract_access_level(request) == DEFAULT_ACCESS_LEVEL

    def test_non_numeric_header(self):
        """Test that non-numeric header returns default access level."""
        request = _make_request({HEADER_AUTH_WORKSPACE_ACCESS: "invalid"})
        assert AetherAuthenticationService._extract_access_level(request) == DEFAULT_ACCESS_LEVEL

    def test_empty_header(self):
        """Test that empty header returns default access level."""
        request = _make_request({HEADER_AUTH_WORKSPACE_ACCESS: ""})
        assert AetherAuthenticationService._extract_access_level(request) == DEFAULT_ACCESS_LEVEL


class TestAetherAuthenticationVerifyApiKey:
    """Tests for verify_api_key (no-op in Aether mode)."""

    @pytest.mark.asyncio
    async def test_verify_api_key_returns_default(self, auth_service):
        """Test that verify_api_key returns default identity."""
        identity = await auth_service.verify_api_key("some-key")

        assert identity.tenant_id == DEFAULT_TENANT_ID
        assert identity.user_id is None

    @pytest.mark.asyncio
    async def test_verify_api_key_with_none(self, auth_service):
        """Test that verify_api_key handles None gracefully."""
        identity = await auth_service.verify_api_key(None)

        assert identity.tenant_id == DEFAULT_TENANT_ID


# ===========================================================================
# AetherAuthorizationService Tests
# ===========================================================================

class TestAetherAuthorizationAuthorize:
    """Tests for AetherAuthorizationService.authorize()."""

    @pytest.mark.asyncio
    async def test_read_allowed_at_read_level(self, authz_service):
        """Test that READ level allows reading memories."""
        ctx = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource="memories", action="read",
            metadata={META_ACCESS_LEVEL: ACCESS_READ},
        )
        assert await authz_service.authorize(ctx) == AuthorizationDecision.ALLOW

    @pytest.mark.asyncio
    async def test_write_denied_at_read_level(self, authz_service):
        """Test that READ level denies writing memories."""
        ctx = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource="memories", action="write",
            metadata={META_ACCESS_LEVEL: ACCESS_READ},
        )
        assert await authz_service.authorize(ctx) == AuthorizationDecision.DENY

    @pytest.mark.asyncio
    async def test_write_allowed_at_readwrite_level(self, authz_service):
        """Test that READWRITE level allows writing memories."""
        ctx = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource="memories", action="write",
            metadata={META_ACCESS_LEVEL: ACCESS_READWRITE},
        )
        assert await authz_service.authorize(ctx) == AuthorizationDecision.ALLOW

    @pytest.mark.asyncio
    async def test_delete_denied_at_readwrite_level(self, authz_service):
        """Test that READWRITE level denies deleting memories."""
        ctx = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource="memories", action="delete",
            metadata={META_ACCESS_LEVEL: ACCESS_READWRITE},
        )
        assert await authz_service.authorize(ctx) == AuthorizationDecision.DENY

    @pytest.mark.asyncio
    async def test_delete_allowed_at_manage_level(self, authz_service):
        """Test that MANAGE level allows deleting memories."""
        ctx = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource="memories", action="delete",
            metadata={META_ACCESS_LEVEL: ACCESS_MANAGE},
        )
        assert await authz_service.authorize(ctx) == AuthorizationDecision.ALLOW

    @pytest.mark.asyncio
    async def test_admin_denied_at_manage_level(self, authz_service):
        """Test that MANAGE level denies admin operations."""
        ctx = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource="admin", action="write",
            metadata={META_ACCESS_LEVEL: ACCESS_MANAGE},
        )
        assert await authz_service.authorize(ctx) == AuthorizationDecision.DENY

    @pytest.mark.asyncio
    async def test_admin_allowed_at_admin_level(self, authz_service):
        """Test that ADMIN level allows admin operations."""
        ctx = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource="admin", action="write",
            metadata={META_ACCESS_LEVEL: ACCESS_ADMIN},
        )
        assert await authz_service.authorize(ctx) == AuthorizationDecision.ALLOW

    @pytest.mark.asyncio
    async def test_no_access_level_denies_everything(self, authz_service):
        """Test that missing access level results in DENY."""
        ctx = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource="memories", action="read",
            metadata={},
        )
        assert await authz_service.authorize(ctx) == AuthorizationDecision.DENY

    @pytest.mark.asyncio
    async def test_zero_access_level_denies_everything(self, authz_service):
        """Test that ACCESS_NONE (0) denies all operations."""
        ctx = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource="memories", action="read",
            metadata={META_ACCESS_LEVEL: ACCESS_NONE},
        )
        assert await authz_service.authorize(ctx) == AuthorizationDecision.DENY

    @pytest.mark.asyncio
    async def test_superadmin_allows_everything(self, authz_service):
        """Test that SUPERADMIN level allows all operations."""
        ctx = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource="admin", action="write",
            metadata={META_ACCESS_LEVEL: ACCESS_SUPERADMIN},
        )
        assert await authz_service.authorize(ctx) == AuthorizationDecision.ALLOW

    @pytest.mark.asyncio
    async def test_wildcard_action_requires_admin(self, authz_service):
        """Test that action='*' requires ADMIN level."""
        ctx_manage = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource="admin", action="*",
            metadata={META_ACCESS_LEVEL: ACCESS_MANAGE},
        )
        assert await authz_service.authorize(ctx_manage) == AuthorizationDecision.DENY

        ctx_admin = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource="admin", action="*",
            metadata={META_ACCESS_LEVEL: ACCESS_ADMIN},
        )
        assert await authz_service.authorize(ctx_admin) == AuthorizationDecision.ALLOW


class TestAetherAuthorizationResourceActions:
    """Tests for specific resource/action permission mappings."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("resource,action,min_level", [
        ("memories", "read", ACCESS_READ),
        ("memories", "write", ACCESS_READWRITE),
        ("memories", "delete", ACCESS_MANAGE),
        ("sessions", "read", ACCESS_READ),
        ("sessions", "write", ACCESS_READWRITE),
        ("sessions", "delete", ACCESS_MANAGE),
        ("workspaces", "read", ACCESS_READ),
        ("workspaces", "write", ACCESS_READWRITE),
        ("workspaces", "delete", ACCESS_MANAGE),
        ("documents", "read", ACCESS_READ),
        # write/delete relaxed MANAGE->READWRITE: a workspace writer can manage
        # that workspace's documents (matches every other content resource and
        # the files-library flow where the bridge already gates on the user's
        # can_write_to_workspace before deleting).
        ("documents", "write", ACCESS_READWRITE),
        ("documents", "delete", ACCESS_READWRITE),
        ("threads", "read", ACCESS_READ),
        ("threads", "write", ACCESS_READWRITE),
        ("threads", "delete", ACCESS_MANAGE),
        ("entities", "read", ACCESS_READ),
        ("entities", "write", ACCESS_READWRITE),
        ("context", "read", ACCESS_READ),
        ("context", "write", ACCESS_READWRITE),
        ("admin", "read", ACCESS_ADMIN),
        ("admin", "write", ACCESS_ADMIN),
        ("admin", "delete", ACCESS_ADMIN),
    ])
    async def test_minimum_level_boundary(self, authz_service, resource, action, min_level):
        """Test that exactly the minimum level is required."""
        # One level below should DENY
        if min_level > ACCESS_NONE:
            ctx_below = AuthorizationContext(
                tenant_id="t", workspace_id="w", user_id="u",
                resource=resource, action=action,
                metadata={META_ACCESS_LEVEL: min_level - 10},
            )
            decision = await authz_service.authorize(ctx_below)
            assert decision == AuthorizationDecision.DENY, (
                f"Expected DENY for {resource}/{action} at level {min_level - 10}"
            )

        # Exactly at minimum level should ALLOW
        ctx_at = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource=resource, action=action,
            metadata={META_ACCESS_LEVEL: min_level},
        )
        decision = await authz_service.authorize(ctx_at)
        assert decision == AuthorizationDecision.ALLOW, (
            f"Expected ALLOW for {resource}/{action} at level {min_level}"
        )

    @pytest.mark.asyncio
    async def test_unknown_resource_defaults_to_manage(self, authz_service):
        """Test that unmapped resource/action pairs require MANAGE."""
        ctx_readwrite = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource="unknown_resource", action="unknown_action",
            metadata={META_ACCESS_LEVEL: ACCESS_READWRITE},
        )
        assert await authz_service.authorize(ctx_readwrite) == AuthorizationDecision.DENY

        ctx_manage = AuthorizationContext(
            tenant_id="t", workspace_id="w", user_id="u",
            resource="unknown_resource", action="unknown_action",
            metadata={META_ACCESS_LEVEL: ACCESS_MANAGE},
        )
        assert await authz_service.authorize(ctx_manage) == AuthorizationDecision.ALLOW


class TestAetherAuthorizationWorkspacesAndRoles:
    """Tests for get_allowed_workspaces and get_user_role."""

    @pytest.mark.asyncio
    async def test_get_allowed_workspaces_returns_wildcard(self, authz_service):
        """Test that get_allowed_workspaces returns wildcard (gateway-scoped)."""
        result = await authz_service.get_allowed_workspaces("t", "u")
        assert result == ["*"]

    @pytest.mark.asyncio
    async def test_get_user_role_returns_none(self, authz_service):
        """Test that get_user_role returns None (role is request-scoped)."""
        result = await authz_service.get_user_role("t", "w", "u")
        assert result is None


class TestGetRequiredAccessLevel:
    """Tests for the get_required_access_level helper."""

    def test_known_resource_action(self):
        """Test lookup of known resource/action pair."""
        assert get_required_access_level("memories", "read") == ACCESS_READ
        assert get_required_access_level("memories", "write") == ACCESS_READWRITE
        assert get_required_access_level("memories", "delete") == ACCESS_MANAGE
        assert get_required_access_level("admin", "write") == ACCESS_ADMIN

    def test_unknown_resource_action(self):
        """Test that unknown pairs return MANAGE (conservative default)."""
        assert get_required_access_level("unknown", "unknown") == ACCESS_MANAGE

    def test_wildcard_action(self):
        """Test that wildcard action returns ADMIN."""
        assert get_required_access_level("anything", "*") == ACCESS_ADMIN


# ===========================================================================
# Integration Tests: Auth -> Authz flow
# ===========================================================================

class TestAuthAuthzIntegration:
    """Tests verifying the metadata flows correctly from auth to authz."""

    @pytest.mark.asyncio
    async def test_metadata_flows_from_auth_to_authz(
        self, auth_service, authz_service,
    ):
        """Test that access level from auth headers reaches authz decisions."""
        # Simulate a request with READ access
        request = _make_request({
            HEADER_AUTH_TENANT_ID: "tenant-1",
            HEADER_AUTH_USER_ID: "user-1",
            HEADER_AUTH_WORKSPACE_ACCESS: "10",
        })

        ctx = await auth_service.build_context(request)

        # Build authorization context with the metadata
        authz_ctx = AuthorizationContext(
            tenant_id=ctx.tenant_id,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            resource="memories",
            action="read",
            metadata=ctx.metadata,
        )

        # READ should be allowed
        assert await authz_service.authorize(authz_ctx) == AuthorizationDecision.ALLOW

        # WRITE should be denied
        authz_ctx_write = AuthorizationContext(
            tenant_id=ctx.tenant_id,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            resource="memories",
            action="write",
            metadata=ctx.metadata,
        )
        assert await authz_service.authorize(authz_ctx_write) == AuthorizationDecision.DENY

    @pytest.mark.asyncio
    async def test_admin_access_allows_everything(
        self, auth_service, authz_service,
    ):
        """Test that ADMIN access level from gateway allows admin ops."""
        request = _make_request({
            HEADER_AUTH_TENANT_ID: "tenant-1",
            HEADER_AUTH_USER_ID: "admin-user",
            HEADER_AUTH_WORKSPACE_ACCESS: "40",
        })

        ctx = await auth_service.build_context(request)

        authz_ctx = AuthorizationContext(
            tenant_id=ctx.tenant_id,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            resource="admin",
            action="write",
            metadata=ctx.metadata,
        )
        assert await authz_service.authorize(authz_ctx) == AuthorizationDecision.ALLOW

    @pytest.mark.asyncio
    async def test_no_gateway_headers_fail_closed(
        self, auth_service, authz_service,
    ):
        """Test that requests without gateway headers fail closed at authentication."""
        request = _make_request({})

        with pytest.raises(AuthenticationError) as exc_info:
            await auth_service.build_context(request)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_dev_fallback_headers_denies_all(
        self, dev_fallback_auth_service, authz_service,
    ):
        """Test that even with the dev fallback, no access level means authz denies all."""
        request = _make_request({})

        ctx = await dev_fallback_auth_service.build_context(request)

        authz_ctx = AuthorizationContext(
            tenant_id=ctx.tenant_id,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            resource="memories",
            action="read",
            metadata=ctx.metadata,
        )
        assert await authz_service.authorize(authz_ctx) == AuthorizationDecision.DENY
