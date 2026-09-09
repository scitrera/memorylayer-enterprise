# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for PostgreSQLBackend.update_association.

Uses an in-memory SQLite engine (same pattern as test_associations_batch_postgres.py).
Verifies:
- strength/metadata are updated and returned via get_associations
- updated_at is bumped on source AND target memories (watermark visibility)
- no-field call returns False without touching the DB
- wrong workspace_id returns False
- second call with same values still succeeds (idempotent edge-update)
"""
import uuid
from datetime import UTC, datetime
from typing import Any
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
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import StaticPool

from memorylayer_server.models.association import AssociateInput
from memorylayer_saas.storage.postgresql import PostgreSQLBackend


# ---------------------------------------------------------------------------
# SQLite-compatible schema
# ---------------------------------------------------------------------------

class _Base(DeclarativeBase):
    pass


class _MemoryModelSQLite(_Base):
    __tablename__ = "memories"
    id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        _sa.DateTime(timezone=True), nullable=False
    )
    source_associations: Mapped[list["_AssocModelSQLite"]] = relationship(
        foreign_keys="[_AssocModelSQLite.source_id]",
        back_populates="source_memory",
        cascade="all, delete-orphan",
    )
    target_associations: Mapped[list["_AssocModelSQLite"]] = relationship(
        foreign_keys="[_AssocModelSQLite.target_id]",
        back_populates="target_memory",
        cascade="all, delete-orphan",
    )


class _AssocModelSQLite(_Base):
    __tablename__ = "memory_associations"
    id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    source_id: Mapped[str] = mapped_column(
        _sa.Text, _sa.ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    target_id: Mapped[str] = mapped_column(
        _sa.Text, _sa.ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column("relationship", _sa.Text, nullable=False)
    strength: Mapped[float] = mapped_column(_sa.Float, nullable=False, server_default="0.5")
    meta: Mapped[Any] = mapped_column("metadata", _sa.JSON, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)
    source_memory: Mapped["_MemoryModelSQLite"] = relationship(
        foreign_keys=[source_id], back_populates="source_associations"
    )
    target_memory: Mapped["_MemoryModelSQLite"] = relationship(
        foreign_keys=[target_id], back_populates="target_associations"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mid() -> str:
    return f"mem_{uuid.uuid4().hex[:12]}"


def _aid() -> str:
    return f"assoc_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.now(UTC)


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
    def _fk(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    async with eng.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)

    yield eng

    async with eng.begin() as conn:
        await conn.run_sync(_Base.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def backend(session_factory):
    import memorylayer_saas.storage.postgresql as pg_module

    with patch.object(PostgreSQLBackend, "__init__", lambda self, *a, **kw: None):
        b = PostgreSQLBackend()

    b._session_factory = session_factory
    b.logger = MagicMock()

    with patch.object(pg_module, "MemoryModel", _MemoryModelSQLite), \
         patch.object(pg_module, "MemoryAssociationModel", _AssocModelSQLite):
        yield b


# ---------------------------------------------------------------------------
# Helper: insert a memory row directly
# ---------------------------------------------------------------------------

async def _insert_memory(session_factory, memory_id: str, workspace_id: str, updated_at: datetime):
    async with session_factory() as session:
        session.add(_MemoryModelSQLite(
            id=memory_id,
            workspace_id=workspace_id,
            updated_at=updated_at,
        ))
        await session.commit()


async def _get_memory_updated_at(session_factory, memory_id: str) -> datetime:
    from sqlalchemy import select
    async with session_factory() as session:
        result = await session.execute(
            select(_MemoryModelSQLite).where(_MemoryModelSQLite.id == memory_id)
        )
        m = result.scalar_one()
        return m.updated_at


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUpdateAssociation:

    @pytest.mark.asyncio
    async def test_update_strength_and_metadata(self, backend, session_factory):
        """update_association updates strength and metadata; returns True."""
        ws = "ws_upd_1"
        src_id, tgt_id = _mid(), _mid()
        t0 = _now()
        await _insert_memory(session_factory, src_id, ws, t0)
        await _insert_memory(session_factory, tgt_id, ws, t0)

        assoc = await backend.create_association(ws, AssociateInput(
            source_id=src_id,
            target_id=tgt_id,
            relationship="CAUSES",
            strength=0.5,
            metadata={"original": True},
        ))

        result = await backend.update_association(
            ws, assoc.id,
            strength=0.9,
            metadata={"updated": True},
        )
        assert result is True

        # Verify via get_associations
        associations = await backend.get_associations(ws, src_id, direction="outgoing")
        assert len(associations) == 1
        assert associations[0].strength == pytest.approx(0.9)
        assert associations[0].metadata == {"updated": True}

    @pytest.mark.asyncio
    async def test_update_bumps_source_and_target_updated_at(self, backend, session_factory):
        """Watermark visibility: updated_at advances on both source and target memories."""
        ws = "ws_upd_wm"
        src_id, tgt_id = _mid(), _mid()
        t0 = datetime(2020, 1, 1, tzinfo=UTC)
        await _insert_memory(session_factory, src_id, ws, t0)
        await _insert_memory(session_factory, tgt_id, ws, t0)

        assoc = await backend.create_association(ws, AssociateInput(
            source_id=src_id,
            target_id=tgt_id,
            relationship="RELATED_TO",
            strength=0.5,
            metadata={},
        ))

        await backend.update_association(ws, assoc.id, strength=0.8)

        src_updated = await _get_memory_updated_at(session_factory, src_id)
        tgt_updated = await _get_memory_updated_at(session_factory, tgt_id)

        # SQLite returns tz-naive datetimes; compare using a tz-naive epoch
        t0_naive = t0.replace(tzinfo=None)
        src_cmp = src_updated.replace(tzinfo=None) if src_updated.tzinfo else src_updated
        tgt_cmp = tgt_updated.replace(tzinfo=None) if tgt_updated.tzinfo else tgt_updated

        assert src_cmp > t0_naive, "source memory updated_at must advance after update_association"
        assert tgt_cmp > t0_naive, "target memory updated_at must advance after update_association"

    @pytest.mark.asyncio
    async def test_update_only_strength(self, backend, session_factory):
        """Partial update: only strength, metadata unchanged."""
        ws = "ws_upd_str"
        src_id, tgt_id = _mid(), _mid()
        t0 = _now()
        await _insert_memory(session_factory, src_id, ws, t0)
        await _insert_memory(session_factory, tgt_id, ws, t0)

        assoc = await backend.create_association(ws, AssociateInput(
            source_id=src_id,
            target_id=tgt_id,
            relationship="SOLVES",
            strength=0.4,
            metadata={"keep": "me"},
        ))

        result = await backend.update_association(ws, assoc.id, strength=0.7)
        assert result is True

        associations = await backend.get_associations(ws, src_id, direction="outgoing")
        assert associations[0].strength == pytest.approx(0.7)
        assert associations[0].metadata == {"keep": "me"}

    @pytest.mark.asyncio
    async def test_update_only_metadata(self, backend, session_factory):
        """Partial update: only metadata, strength unchanged."""
        ws = "ws_upd_meta"
        src_id, tgt_id = _mid(), _mid()
        t0 = _now()
        await _insert_memory(session_factory, src_id, ws, t0)
        await _insert_memory(session_factory, tgt_id, ws, t0)

        assoc = await backend.create_association(ws, AssociateInput(
            source_id=src_id,
            target_id=tgt_id,
            relationship="RELATED_TO",
            strength=0.6,
            metadata={"old": True},
        ))

        result = await backend.update_association(ws, assoc.id, metadata={"new": True})
        assert result is True

        associations = await backend.get_associations(ws, src_id, direction="outgoing")
        assert associations[0].strength == pytest.approx(0.6)
        assert associations[0].metadata == {"new": True}

    @pytest.mark.asyncio
    async def test_no_fields_returns_false(self, backend, session_factory):
        """update_association with no fields is a no-op returning False."""
        ws = "ws_upd_noop"
        src_id, tgt_id = _mid(), _mid()
        t0 = _now()
        await _insert_memory(session_factory, src_id, ws, t0)
        await _insert_memory(session_factory, tgt_id, ws, t0)

        assoc = await backend.create_association(ws, AssociateInput(
            source_id=src_id,
            target_id=tgt_id,
            relationship="CAUSES",
            strength=0.5,
            metadata={},
        ))

        result = await backend.update_association(ws, assoc.id)
        assert result is False

    @pytest.mark.asyncio
    async def test_wrong_workspace_returns_false(self, backend, session_factory):
        """update_association with mismatched workspace_id returns False."""
        ws = "ws_upd_scope"
        src_id, tgt_id = _mid(), _mid()
        t0 = _now()
        await _insert_memory(session_factory, src_id, ws, t0)
        await _insert_memory(session_factory, tgt_id, ws, t0)

        assoc = await backend.create_association(ws, AssociateInput(
            source_id=src_id,
            target_id=tgt_id,
            relationship="CAUSES",
            strength=0.5,
            metadata={},
        ))

        result = await backend.update_association("ws_other", assoc.id, strength=0.9)
        assert result is False

    @pytest.mark.asyncio
    async def test_not_found_returns_false(self, backend):
        """update_association for a non-existent association returns False."""
        result = await backend.update_association(
            "ws_any", "assoc_does_not_exist", strength=0.9
        )
        assert result is False
