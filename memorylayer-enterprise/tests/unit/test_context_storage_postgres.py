# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for Context storage methods on PostgreSQLBackend.

Focus: delete_context (newly added). Validates the ORM/SQLAlchemy layer using an
in-memory SQLite async engine instead of a live PostgreSQL instance (same pattern
as test_skill_storage_postgres.py), so create -> delete -> get/list -> re-delete
round-trips without needing a real DB / MLFS_TEST_DATABASE_URL.
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
import sqlalchemy as _sa
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from memorylayer_server.models.workspace import Context


class _ContextBase(DeclarativeBase):
    """Isolated declarative base for a SQLite-compatible contexts schema."""
    pass


class _ContextModelSQLite(_ContextBase):
    __tablename__ = "contexts"
    id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    name: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    description: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    settings: Mapped[dict] = mapped_column(_sa.JSON, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)
    __table_args__ = (
        _sa.UniqueConstraint("workspace_id", "name", name="uq_workspace_context_name"),
    )


def _ctx_id() -> str:
    return f"ctx_{uuid.uuid4().hex[:12]}"


def _make_context(workspace_id: str = "ws_ctx", name: str = "project-alpha") -> Context:
    return Context(
        id=_ctx_id(),
        workspace_id=workspace_id,
        name=name,
        description="A test context",
        settings={"k": "v"},
        created_at=datetime.now(timezone.utc),
    )


@pytest_asyncio.fixture(scope="module")
async def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(eng.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    async with eng.begin() as conn:
        await conn.run_sync(_ContextBase.metadata.create_all)

    yield eng

    async with eng.begin() as conn:
        await conn.run_sync(_ContextBase.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def backend(session_factory):
    """PostgreSQLBackend stand-in wired to the SQLite session with SQLite-compatible ContextModel."""
    import memorylayer_saas.storage.postgresql as pg_module
    from memorylayer_saas.storage.postgresql import PostgreSQLBackend

    with patch.object(PostgreSQLBackend, "__init__", lambda self, *a, **kw: None):
        b = PostgreSQLBackend()

    b._session_factory = session_factory
    b.logger = MagicMock()

    with patch.object(pg_module, "ContextModel", _ContextModelSQLite):
        yield b


class TestContextDelete:
    """create -> delete -> get/list no longer return it -> second delete False."""

    @pytest.mark.asyncio
    async def test_delete_context_removes_row(self, backend):
        ctx = _make_context(workspace_id="ws_del", name="to-delete")
        await backend.create_context("ws_del", ctx)

        # Present before delete
        assert await backend.get_context("ws_del", ctx.id) is not None
        assert any(c.id == ctx.id for c in await backend.list_contexts("ws_del"))

        deleted = await backend.delete_context("ws_del", ctx.id)
        assert deleted is True

        # Absent after delete
        assert await backend.get_context("ws_del", ctx.id) is None
        assert all(c.id != ctx.id for c in await backend.list_contexts("ws_del"))

    @pytest.mark.asyncio
    async def test_delete_context_not_found_returns_false(self, backend):
        result = await backend.delete_context("ws_del", "ctx_does_not_exist")
        assert result is False

    @pytest.mark.asyncio
    async def test_second_delete_returns_false(self, backend):
        ctx = _make_context(workspace_id="ws_twice", name="twice")
        await backend.create_context("ws_twice", ctx)

        assert await backend.delete_context("ws_twice", ctx.id) is True
        # Second delete of the same context returns False
        assert await backend.delete_context("ws_twice", ctx.id) is False

    @pytest.mark.asyncio
    async def test_delete_context_wrong_workspace_returns_false(self, backend):
        """Deleting with a mismatched workspace must not remove the row."""
        ctx = _make_context(workspace_id="ws_owner", name="scoped")
        await backend.create_context("ws_owner", ctx)

        # Wrong workspace -> not deleted
        assert await backend.delete_context("ws_other", ctx.id) is False
        # Still present in the owning workspace
        assert await backend.get_context("ws_owner", ctx.id) is not None
