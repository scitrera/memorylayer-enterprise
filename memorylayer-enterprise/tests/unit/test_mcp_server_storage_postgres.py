# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for McpServer storage methods on PostgreSQLBackend.

NOTE: These tests use an in-memory SQLite database via SQLAlchemy's async
engine instead of a live PostgreSQL instance. This validates the ORM layer,
helper converters, and all 7 mcp_server methods end-to-end at the SQLAlchemy
abstraction level.

PostgreSQL-specific features tested here that require a live DB:
- Partial unique indexes (idx_mcp_servers_workspace_name_global / idx_mcp_servers_workspace_user_name)
- ARRAY type for args (falls back to JSON in the SQLite path)

All other logic (CRUD, filters, converters) is fully covered.
"""
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import DeclarativeBase

from memorylayer_server.models.mcp_server import McpServer
from memorylayer_saas.storage.models import McpServerModel


class _McpBase(DeclarativeBase):
    """Isolated declarative base for SQLite-compatible mcp_server-only schema."""
    pass


# Re-declare minimal SQLite-compatible table definition using JSON (not JSONB/ARRAY)
import sqlalchemy as _sa
from sqlalchemy.orm import Mapped, mapped_column


class _McpServerModelSQLite(_McpBase):
    __tablename__ = "mcp_servers"
    id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="_default")
    workspace_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    user_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    name: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    description: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    transport: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    command: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    args: Mapped[Any] = mapped_column("args", _sa.JSON, nullable=False, server_default="[]")
    env: Mapped[Any] = mapped_column("env", _sa.JSON, nullable=False, server_default="{}")
    url: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    headers: Mapped[Any] = mapped_column("headers", _sa.JSON, nullable=False, server_default="{}")
    meta: Mapped[Any] = mapped_column("metadata", _sa.JSON, nullable=False, server_default="{}")
    source_mode: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="server")
    manifest_hash: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="")
    enabled: Mapped[bool] = mapped_column(_sa.Boolean, nullable=False, server_default="1")
    created_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mcp_id() -> str:
    return f"mcp_{uuid.uuid4().hex[:12]}"


def _make_stdio_server(
    workspace_id: str = "ws_test",
    name: str = "my-server",
    user_id: str | None = None,
    tenant_id: str = "tenant_a",
    **kwargs: Any,
) -> McpServer:
    return McpServer(
        id=_mcp_id(),
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        user_id=user_id,
        name=name,
        description="A test MCP server",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem"],
        env={"API_KEY": "sk-test"},
        metadata={"vendor": "test"},
        source_mode="server",
        manifest_hash="abc123",
        **{"enabled": True, **kwargs},
    )


def _make_http_server(
    workspace_id: str = "ws_test",
    name: str = "my-http-server",
    user_id: str | None = None,
    **kwargs: Any,
) -> McpServer:
    return McpServer(
        id=_mcp_id(),
        tenant_id="tenant_a",
        workspace_id=workspace_id,
        user_id=user_id,
        name=name,
        description="An HTTP MCP server",
        transport="http",
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer token123"},
        metadata={},
        source_mode="server",
        manifest_hash="def456",
        **{"enabled": True, **kwargs},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="module")
async def engine():
    """In-memory SQLite async engine using SQLite-compatible mcp_server table."""
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(eng.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    async with eng.begin() as conn:
        await conn.run_sync(_McpBase.metadata.create_all)

    yield eng

    async with eng.begin() as conn:
        await conn.run_sync(_McpBase.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def backend(session_factory):
    """Minimal PostgreSQLBackend stand-in with mcp_server methods wired to SQLite session."""
    import memorylayer_saas.storage.postgresql as pg_module
    from memorylayer_saas.storage.postgresql import PostgreSQLBackend

    with patch.object(PostgreSQLBackend, "__init__", lambda self, *a, **kw: None):
        b = PostgreSQLBackend()

    b._session_factory = session_factory
    b.logger = MagicMock()
    b.logger.debug = MagicMock()

    with patch.object(pg_module, "McpServerModel", _McpServerModelSQLite):
        yield b


# ---------------------------------------------------------------------------
# Tests: McpServer CRUD
# ---------------------------------------------------------------------------

class TestMcpServerCRUD:

    @pytest.mark.asyncio
    async def test_create_and_get_stdio_server(self, backend):
        server = _make_stdio_server(name="create-get-server")
        created = await backend.create_mcp_server(server)

        assert created.id == server.id
        assert created.name == "create-get-server"
        assert created.transport == "stdio"
        assert created.command == "npx"
        assert created.args == ["-y", "@modelcontextprotocol/server-filesystem"]
        assert created.env == {"API_KEY": "sk-test"}
        assert created.workspace_id == "ws_test"
        assert created.enabled is True

        fetched = await backend.get_mcp_server("ws_test", server.id)
        assert fetched is not None
        assert fetched.id == server.id
        assert fetched.transport == "stdio"

    @pytest.mark.asyncio
    async def test_create_and_get_http_server(self, backend):
        server = _make_http_server(name="http-server")
        created = await backend.create_mcp_server(server)

        assert created.transport == "http"
        assert created.url == "https://example.com/mcp"
        assert created.headers == {"Authorization": "Bearer token123"}
        assert created.command is None

        fetched = await backend.get_mcp_server("ws_test", server.id)
        assert fetched is not None
        assert fetched.url == "https://example.com/mcp"

    @pytest.mark.asyncio
    async def test_get_wrong_workspace_returns_none(self, backend):
        server = _make_stdio_server(name="ws-check-server")
        await backend.create_mcp_server(server)

        result = await backend.get_mcp_server("wrong_workspace", server.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_name_workspace_scope(self, backend):
        server = _make_stdio_server(workspace_id="ws_byname", name="named-server", user_id=None)
        await backend.create_mcp_server(server)

        found = await backend.get_mcp_server_by_name("ws_byname", "named-server")
        assert found is not None
        assert found.id == server.id
        assert found.user_id is None

    @pytest.mark.asyncio
    async def test_get_by_name_user_scope(self, backend):
        server = _make_stdio_server(workspace_id="ws_user_mcp", name="user-server", user_id="usr_001")
        await backend.create_mcp_server(server)

        found = await backend.get_mcp_server_by_name("ws_user_mcp", "user-server", user_id="usr_001")
        assert found is not None
        assert found.user_id == "usr_001"

        # Workspace-scope lookup should NOT return user-scoped server
        not_found = await backend.get_mcp_server_by_name("ws_user_mcp", "user-server")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_list_basic(self, backend):
        ws = "ws_list_mcp"
        s1 = _make_stdio_server(workspace_id=ws, name="server-alpha")
        s2 = _make_http_server(workspace_id=ws, name="server-beta")
        await backend.create_mcp_server(s1)
        await backend.create_mcp_server(s2)

        results = await backend.list_mcp_servers(ws)
        ids = {r.id for r in results}
        assert s1.id in ids
        assert s2.id in ids

    @pytest.mark.asyncio
    async def test_list_filter_by_transport(self, backend):
        ws = "ws_list_transport"
        s_stdio = _make_stdio_server(workspace_id=ws, name="stdio-server")
        s_http = _make_http_server(workspace_id=ws, name="http-server-t")
        await backend.create_mcp_server(s_stdio)
        await backend.create_mcp_server(s_http)

        stdio_results = await backend.list_mcp_servers(ws, transport="stdio")
        assert s_stdio.id in {r.id for r in stdio_results}
        assert s_http.id not in {r.id for r in stdio_results}

        http_results = await backend.list_mcp_servers(ws, transport="http")
        assert s_http.id in {r.id for r in http_results}
        assert s_stdio.id not in {r.id for r in http_results}

    @pytest.mark.asyncio
    async def test_list_filter_by_enabled(self, backend):
        ws = "ws_list_enabled_mcp"
        active = _make_stdio_server(workspace_id=ws, name="active-server", enabled=True)
        inactive = _make_stdio_server(workspace_id=ws, name="inactive-server", enabled=False)
        await backend.create_mcp_server(active)
        await backend.create_mcp_server(inactive)

        active_results = await backend.list_mcp_servers(ws, enabled=True)
        assert active.id in {r.id for r in active_results}
        assert inactive.id not in {r.id for r in active_results}

    @pytest.mark.asyncio
    async def test_list_filter_by_user(self, backend):
        ws = "ws_list_user_mcp"
        global_server = _make_stdio_server(workspace_id=ws, name="global-server", user_id=None)
        user_server = _make_stdio_server(workspace_id=ws, name="user-server-l", user_id="usr_xyz")
        await backend.create_mcp_server(global_server)
        await backend.create_mcp_server(user_server)

        user_results = await backend.list_mcp_servers(ws, user_id="usr_xyz")
        assert user_server.id in {r.id for r in user_results}
        assert global_server.id not in {r.id for r in user_results}

    @pytest.mark.asyncio
    async def test_list_limit_offset(self, backend):
        ws = "ws_list_page_mcp"
        servers = [_make_stdio_server(workspace_id=ws, name=f"paged-{i}") for i in range(5)]
        for s in servers:
            await backend.create_mcp_server(s)

        page1 = await backend.list_mcp_servers(ws, limit=3, offset=0)
        page2 = await backend.list_mcp_servers(ws, limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 2
        assert {r.id for r in page1}.isdisjoint({r.id for r in page2})

    @pytest.mark.asyncio
    async def test_update_server(self, backend):
        server = _make_stdio_server(workspace_id="ws_update_mcp", name="update-me-mcp")
        await backend.create_mcp_server(server)

        updated = await backend.update_mcp_server(
            "ws_update_mcp", server.id,
            {"description": "Updated description", "enabled": False, "manifest_hash": "newhash"}
        )
        assert updated is not None
        assert updated.description == "Updated description"
        assert updated.enabled is False
        assert updated.manifest_hash == "newhash"

    @pytest.mark.asyncio
    async def test_update_metadata_key(self, backend):
        server = _make_stdio_server(workspace_id="ws_meta_mcp", name="meta-server")
        await backend.create_mcp_server(server)

        updated = await backend.update_mcp_server(
            "ws_meta_mcp", server.id,
            {"metadata": {"new_key": "new_value"}}
        )
        assert updated is not None
        assert updated.metadata == {"new_key": "new_value"}

    @pytest.mark.asyncio
    async def test_update_args_and_env(self, backend):
        server = _make_stdio_server(workspace_id="ws_args_mcp", name="args-server")
        await backend.create_mcp_server(server)

        updated = await backend.update_mcp_server(
            "ws_args_mcp", server.id,
            {"args": ["--new-arg"], "env": {"NEW_KEY": "val"}}
        )
        assert updated is not None
        assert updated.args == ["--new-arg"]
        assert updated.env == {"NEW_KEY": "val"}

    @pytest.mark.asyncio
    async def test_update_not_found_returns_none(self, backend):
        result = await backend.update_mcp_server("ws_test", "mcp_nonexistent", {"enabled": False})
        assert result is None

    @pytest.mark.asyncio
    async def test_update_no_updates_returns_current(self, backend):
        server = _make_stdio_server(workspace_id="ws_noop_mcp", name="noop-server")
        await backend.create_mcp_server(server)

        same = await backend.update_mcp_server("ws_noop_mcp", server.id, {})
        assert same is not None
        assert same.id == server.id

    @pytest.mark.asyncio
    async def test_delete_server(self, backend):
        server = _make_stdio_server(workspace_id="ws_delete_mcp", name="delete-me-mcp")
        await backend.create_mcp_server(server)

        deleted = await backend.delete_mcp_server("ws_delete_mcp", server.id)
        assert deleted is True

        fetched = await backend.get_mcp_server("ws_delete_mcp", server.id)
        assert fetched is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, backend):
        result = await backend.delete_mcp_server("ws_test", "mcp_ghost")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_logs_debug(self, backend):
        server = _make_stdio_server(workspace_id="ws_log_mcp", name="log-server")
        await backend.create_mcp_server(server)
        await backend.delete_mcp_server("ws_log_mcp", server.id)
        backend.logger.debug.assert_called_with("Deleted MCP server: %s", server.id)


class TestFindMcpServersByName:

    @pytest.mark.asyncio
    async def test_find_across_scopes(self, backend):
        ws_a = "ws_find_a"
        ws_b = "ws_find_b"
        s1 = _make_stdio_server(workspace_id=ws_a, name="shared-server")
        s2 = _make_stdio_server(workspace_id=ws_b, name="shared-server")
        await backend.create_mcp_server(s1)
        await backend.create_mcp_server(s2)

        results = await backend.find_mcp_servers_by_name(
            "shared-server",
            [{"workspace_id": ws_a}, {"workspace_id": ws_b}],
        )
        ids = {r.id for r in results}
        assert s1.id in ids
        assert s2.id in ids

    @pytest.mark.asyncio
    async def test_find_empty_scope_filters_returns_empty(self, backend):
        results = await backend.find_mcp_servers_by_name("any-server", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_find_user_scope_filter(self, backend):
        ws = "ws_find_user"
        global_s = _make_stdio_server(workspace_id=ws, name="find-user-server", user_id=None)
        user_s = _make_stdio_server(workspace_id=ws, name="find-user-server", user_id="usr_find")
        await backend.create_mcp_server(global_s)
        await backend.create_mcp_server(user_s)

        results = await backend.find_mcp_servers_by_name(
            "find-user-server",
            [{"workspace_id": ws, "user_id": "usr_find"}],
        )
        ids = {r.id for r in results}
        assert user_s.id in ids
        assert global_s.id not in ids
