# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the claim-race CAS primitive on PostgreSQLBackend (Phase 2.3).

Covers ``try_claim_document`` — the atomic conditional UPDATE that closes the
read-then-write window the freshness check leaves open. Exactly one of two
concurrent workers wins the claim.

NOTE: These tests use an in-memory SQLite database via SQLAlchemy's async engine
instead of a live PostgreSQL instance — the SAME shim pattern as
``test_memory_source_page_ids_postgres.py``. ``DocumentModel`` carries JSONB
columns SQLite cannot create, so we declare a SQLite-compatible stand-in covering
the columns the CAS touches and patch ``pg_module.DocumentModel`` to it. The CAS
is plain conditional-UPDATE SQL (no PG-only constructs), so this validates the
exact query construction (id+workspace scope, status / NULL / staleness OR-group,
rowcount==1 win semantics) end-to-end at the SQLAlchemy abstraction level.
"""
from datetime import UTC, datetime, timedelta
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

TTL = 900


class _DocBase(DeclarativeBase):
    """Isolated declarative base for the SQLite-compatible documents schema."""
    pass


class _DocumentModelSQLite(_DocBase):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    tenant_id: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="_default")
    filename: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    document_type: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(_sa.Integer, nullable=False)
    status: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="pending")
    page_count: Mapped[int] = mapped_column(_sa.Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)
    processing_started_at: Mapped[datetime | None] = mapped_column(_sa.DateTime(timezone=True), nullable=True)
    processing_completed_at: Mapped[datetime | None] = mapped_column(_sa.DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
        await conn.run_sync(_DocBase.metadata.create_all)

    yield eng

    async with eng.begin() as conn:
        await conn.run_sync(_DocBase.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def backend(session_factory):
    import memorylayer_saas.storage.postgresql as pg_module
    from memorylayer_saas.storage.postgresql import PostgreSQLBackend

    with patch.object(PostgreSQLBackend, "__init__", lambda self, *a, **kw: None):
        b = PostgreSQLBackend()

    b._session_factory = session_factory
    b.logger = MagicMock()

    with patch.object(pg_module, "DocumentModel", _DocumentModelSQLite):
        yield b


async def _add_doc(session_factory, *, doc_id, workspace_id, status, started_at=None):
    now = datetime.now(UTC)
    async with session_factory() as session:
        session.add(_DocumentModelSQLite(
            id=doc_id,
            workspace_id=workspace_id,
            tenant_id="_default",
            filename="report.pdf",
            document_type="pdf",
            content_hash="h_%s" % doc_id,
            size_bytes=1,
            status=status,
            page_count=0,
            created_at=now,
            processing_started_at=started_at,
        ))
        await session.commit()


async def _status_and_started(session_factory, doc_id):
    async with session_factory() as session:
        row = await session.execute(
            _sa.select(
                _DocumentModelSQLite.status,
                _DocumentModelSQLite.processing_started_at,
            ).where(_DocumentModelSQLite.id == doc_id)
        )
        return row.first()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTryClaimDocument:

    @pytest.mark.asyncio
    async def test_claims_non_processing_doc(self, backend, session_factory):
        """A FAILED (non-processing) doc is claimable; CAS flips it to processing."""
        await _add_doc(session_factory, doc_id="d1", workspace_id="ws", status="failed")

        won = await backend.try_claim_document("d1", "ws", TTL)
        assert won is True

        status, started = await _status_and_started(session_factory, "d1")
        assert status == "processing"
        assert started is not None

    @pytest.mark.asyncio
    async def test_claims_stale_processing_doc(self, backend, session_factory):
        """A PROCESSING doc older than the TTL is reclaimable (orphan)."""
        stale = datetime.now(UTC) - timedelta(seconds=TTL + 60)
        await _add_doc(
            session_factory, doc_id="d2", workspace_id="ws", status="processing",
            started_at=stale,
        )

        won = await backend.try_claim_document("d2", "ws", TTL)
        assert won is True

        _, started = await _status_and_started(session_factory, "d2")
        # re-stamped to a fresh time, newer than the stale value.
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        assert started > stale

    @pytest.mark.asyncio
    async def test_does_not_claim_fresh_processing_doc(self, backend, session_factory):
        """A fresh PROCESSING doc is owned by another worker — CAS loses (False)."""
        fresh = datetime.now(UTC)
        await _add_doc(
            session_factory, doc_id="d3", workspace_id="ws", status="processing",
            started_at=fresh,
        )

        won = await backend.try_claim_document("d3", "ws", TTL)
        assert won is False

        status, _ = await _status_and_started(session_factory, "d3")
        assert status == "processing"

    @pytest.mark.asyncio
    async def test_claims_processing_with_null_started_at(self, backend, session_factory):
        """A PROCESSING doc with no processing_started_at is claimable (orphan)."""
        await _add_doc(
            session_factory, doc_id="d4", workspace_id="ws", status="processing",
            started_at=None,
        )

        won = await backend.try_claim_document("d4", "ws", TTL)
        assert won is True

    @pytest.mark.asyncio
    async def test_claim_is_workspace_scoped(self, backend, session_factory):
        """A claim against the wrong workspace never matches the row."""
        await _add_doc(session_factory, doc_id="d5", workspace_id="ws", status="failed")

        won = await backend.try_claim_document("d5", "ws_other", TTL)
        assert won is False

        status, _ = await _status_and_started(session_factory, "d5")
        assert status == "failed"

    @pytest.mark.asyncio
    async def test_only_one_of_two_claims_wins(self, backend, session_factory):
        """Two sequential claims on a fresh doc: first wins, second loses (the CAS)."""
        await _add_doc(session_factory, doc_id="d6", workspace_id="ws", status="failed")

        first = await backend.try_claim_document("d6", "ws", TTL)
        second = await backend.try_claim_document("d6", "ws", TTL)
        assert first is True
        # Second sees a fresh PROCESSING row -> loses.
        assert second is False

    @pytest.mark.asyncio
    async def test_missing_doc_returns_false(self, backend):
        """A claim on a non-existent doc matches nothing (False)."""
        won = await backend.try_claim_document("nope", "ws", TTL)
        assert won is False
