# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for PostgreSQLBackend.get_associations_batch.

Uses an in-memory SQLite database (same pattern as other enterprise unit tests).
Verifies that get_associations_batch returns exactly the union of per-id
get_associations results, for all direction modes and a relationships filter,
and that deduplication is correct for direction="both".

Also includes a >1000-id case to confirm no parameter-limit issues.
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

from memorylayer_server.models.association import AssociateInput, Association
from memorylayer_saas.storage.postgresql import PostgreSQLBackend


# ---------------------------------------------------------------------------
# SQLite-compatible schema (mirrors MemoryAssociationModel without JSONB/PG types)
# ---------------------------------------------------------------------------

class _AssocBase(DeclarativeBase):
    """Isolated declarative base for association-only SQLite schema."""
    pass


class _MemoryModelSQLite(_AssocBase):
    __tablename__ = "memories"
    id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
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


class _AssocModelSQLite(_AssocBase):
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
    def _set_pragma(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    async with eng.begin() as conn:
        await conn.run_sync(_AssocBase.metadata.create_all)

    yield eng

    async with eng.begin() as conn:
        await conn.run_sync(_AssocBase.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def backend(session_factory):
    """Minimal PostgreSQLBackend wired to SQLite session with association models patched."""
    import memorylayer_saas.storage.postgresql as pg_module
    from memorylayer_saas.storage.models import MemoryAssociationModel

    with patch.object(PostgreSQLBackend, "__init__", lambda self, *a, **kw: None):
        b = PostgreSQLBackend()

    b._session_factory = session_factory
    b.logger = MagicMock()
    b.logger.debug = MagicMock()

    # Patch MemoryAssociationModel used inside get_associations / get_associations_batch
    with patch.object(pg_module, "MemoryAssociationModel", _AssocModelSQLite):
        yield b


# ---------------------------------------------------------------------------
# Low-level insert helpers (bypass ORM associations to avoid FK issues)
# ---------------------------------------------------------------------------

async def _insert_memory(session_factory, workspace_id: str, memory_id: str) -> None:
    async with session_factory() as session:
        session.add(_MemoryModelSQLite(id=memory_id, workspace_id=workspace_id))
        await session.commit()


async def _insert_assoc(
        session_factory,
        workspace_id: str,
        source_id: str,
        target_id: str,
        relation_type: str = "RELATED_TO",
        strength: float = 0.5,
) -> str:
    aid = _aid()
    async with session_factory() as session:
        session.add(_AssocModelSQLite(
            id=aid,
            workspace_id=workspace_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength,
            meta={},
            created_at=_now(),
        ))
        await session.commit()
    return aid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetAssociationsBatch:

    @pytest.mark.asyncio
    async def test_outgoing_matches_per_id_loop(self, backend, session_factory):
        """Batch outgoing returns same set as per-id loop."""
        ws = f"ws_{uuid.uuid4().hex[:8]}"
        m1, m2, m3, m4 = _mid(), _mid(), _mid(), _mid()
        for m in (m1, m2, m3, m4):
            await _insert_memory(session_factory, ws, m)

        a1 = await _insert_assoc(session_factory, ws, m1, m3)
        a2 = await _insert_assoc(session_factory, ws, m2, m4)
        a3 = await _insert_assoc(session_factory, ws, m1, m4)  # second edge from m1

        import memorylayer_saas.storage.postgresql as pg_module
        with patch.object(pg_module, "MemoryAssociationModel", _AssocModelSQLite):
            # Per-id loop reference
            loop_results: list[Association] = []
            seen: set[str] = set()
            for mid in (m1, m2):
                for a in await backend.get_associations(ws, mid, direction="outgoing"):
                    if a.id not in seen:
                        seen.add(a.id)
                        loop_results.append(a)

            batch_results = await backend.get_associations_batch(ws, [m1, m2], direction="outgoing")

        assert {a.id for a in batch_results} == {a.id for a in loop_results}
        assert {a1, a2, a3} == {a.id for a in batch_results}

    @pytest.mark.asyncio
    async def test_incoming_matches_per_id_loop(self, backend, session_factory):
        """Batch incoming returns same set as per-id loop."""
        ws = f"ws_{uuid.uuid4().hex[:8]}"
        m1, m2, m3 = _mid(), _mid(), _mid()
        for m in (m1, m2, m3):
            await _insert_memory(session_factory, ws, m)

        a1 = await _insert_assoc(session_factory, ws, m1, m2)
        a2 = await _insert_assoc(session_factory, ws, m3, m2)
        await _insert_assoc(session_factory, ws, m1, m3)  # not targeting m2

        import memorylayer_saas.storage.postgresql as pg_module
        with patch.object(pg_module, "MemoryAssociationModel", _AssocModelSQLite):
            loop_results: list[Association] = []
            seen: set[str] = set()
            for mid in (m2,):
                for a in await backend.get_associations(ws, mid, direction="incoming"):
                    if a.id not in seen:
                        seen.add(a.id)
                        loop_results.append(a)

            batch_results = await backend.get_associations_batch(ws, [m2], direction="incoming")

        assert {a.id for a in batch_results} == {a.id for a in loop_results}
        assert {a1, a2} == {a.id for a in batch_results}

    @pytest.mark.asyncio
    async def test_both_direction_deduplicates(self, backend, session_factory):
        """direction='both' does not double-count edges where the same memory is source and target."""
        ws = f"ws_{uuid.uuid4().hex[:8]}"
        m1, m2 = _mid(), _mid()
        for m in (m1, m2):
            await _insert_memory(session_factory, ws, m)

        # Edge m1->m2: m1 is source AND m2 is target — both in our query set
        a1 = await _insert_assoc(session_factory, ws, m1, m2)

        import memorylayer_saas.storage.postgresql as pg_module
        with patch.object(pg_module, "MemoryAssociationModel", _AssocModelSQLite):
            batch_results = await backend.get_associations_batch(ws, [m1, m2], direction="both")

        # Must appear exactly once despite matching both source_id and target_id conditions
        ids = [a.id for a in batch_results]
        assert ids.count(a1) == 1
        assert len(batch_results) == 1

    @pytest.mark.asyncio
    async def test_both_direction_matches_per_id_loop(self, backend, session_factory):
        """Batch direction='both' returns same set as per-id loop."""
        ws = f"ws_{uuid.uuid4().hex[:8]}"
        m1, m2, m3 = _mid(), _mid(), _mid()
        for m in (m1, m2, m3):
            await _insert_memory(session_factory, ws, m)

        a1 = await _insert_assoc(session_factory, ws, m1, m2)
        a2 = await _insert_assoc(session_factory, ws, m3, m1)  # m1 as target
        a3 = await _insert_assoc(session_factory, ws, m2, m3)  # neither m1; tests scoping

        import memorylayer_saas.storage.postgresql as pg_module
        with patch.object(pg_module, "MemoryAssociationModel", _AssocModelSQLite):
            loop_results: list[Association] = []
            seen: set[str] = set()
            for mid in (m1,):
                for a in await backend.get_associations(ws, mid, direction="both"):
                    if a.id not in seen:
                        seen.add(a.id)
                        loop_results.append(a)

            batch_results = await backend.get_associations_batch(ws, [m1], direction="both")

        assert {a.id for a in batch_results} == {a.id for a in loop_results}
        assert {a1, a2} == {a.id for a in batch_results}
        assert a3 not in {a.id for a in batch_results}

    @pytest.mark.asyncio
    async def test_relationships_filter(self, backend, session_factory):
        """relationships filter narrows results, same as per-id loop."""
        ws = f"ws_{uuid.uuid4().hex[:8]}"
        m1, m2, m3 = _mid(), _mid(), _mid()
        for m in (m1, m2, m3):
            await _insert_memory(session_factory, ws, m)

        a_causes = await _insert_assoc(session_factory, ws, m1, m2, relation_type="CAUSES")
        a_solves = await _insert_assoc(session_factory, ws, m1, m3, relation_type="SOLVES")
        a_related = await _insert_assoc(session_factory, ws, m2, m3, relation_type="RELATED_TO")

        import memorylayer_saas.storage.postgresql as pg_module
        with patch.object(pg_module, "MemoryAssociationModel", _AssocModelSQLite):
            loop_results: list[Association] = []
            seen: set[str] = set()
            for mid in (m1, m2):
                for a in await backend.get_associations(ws, mid, direction="outgoing", relationships=["CAUSES"]):
                    if a.id not in seen:
                        seen.add(a.id)
                        loop_results.append(a)

            batch_results = await backend.get_associations_batch(
                ws, [m1, m2], direction="outgoing", relationships=["CAUSES"]
            )

        assert {a.id for a in batch_results} == {a.id for a in loop_results}
        assert {a_causes} == {a.id for a in batch_results}
        assert a_solves not in {a.id for a in batch_results}
        assert a_related not in {a.id for a in batch_results}

    @pytest.mark.asyncio
    async def test_empty_memory_ids_returns_empty(self, backend, session_factory):
        """Empty memory_ids returns empty list without hitting the DB."""
        import memorylayer_saas.storage.postgresql as pg_module
        with patch.object(pg_module, "MemoryAssociationModel", _AssocModelSQLite):
            result = await backend.get_associations_batch("ws_any", [], direction="outgoing")
        assert result == []

    @pytest.mark.asyncio
    async def test_workspace_isolation(self, backend, session_factory):
        """Associations from another workspace are not returned."""
        ws_a = f"ws_{uuid.uuid4().hex[:8]}"
        ws_b = f"ws_{uuid.uuid4().hex[:8]}"
        # Use distinct memory IDs per workspace (memories table has a global PK)
        m_a1, m_a2 = _mid(), _mid()
        m_b1, m_b2 = _mid(), _mid()
        await _insert_memory(session_factory, ws_a, m_a1)
        await _insert_memory(session_factory, ws_a, m_a2)
        await _insert_memory(session_factory, ws_b, m_b1)
        await _insert_memory(session_factory, ws_b, m_b2)

        a_a = await _insert_assoc(session_factory, ws_a, m_a1, m_a2)
        a_b = await _insert_assoc(session_factory, ws_b, m_b1, m_b2)

        import memorylayer_saas.storage.postgresql as pg_module
        with patch.object(pg_module, "MemoryAssociationModel", _AssocModelSQLite):
            results = await backend.get_associations_batch(ws_a, [m_a1, m_b1], direction="outgoing")

        result_ids = {a.id for a in results}
        # Only the ws_a association is returned; ws_b edge is excluded by workspace filter
        assert a_a in result_ids
        assert a_b not in result_ids

    @pytest.mark.asyncio
    async def test_large_id_list_no_param_limit(self, backend, session_factory):
        """1001 memory ids does not blow up parameter limits."""
        ws = f"ws_{uuid.uuid4().hex[:8]}"

        # Create 5 real memories with associations; the other 996 ids are phantoms
        real_ids = [_mid() for _ in range(5)]
        for m in real_ids:
            await _insert_memory(session_factory, ws, m)

        target = _mid()
        await _insert_memory(session_factory, ws, target)

        expected_aids = set()
        for src in real_ids:
            aid = await _insert_assoc(session_factory, ws, src, target)
            expected_aids.add(aid)

        # Build a list of 1001 ids: 5 real + 996 phantom ids
        all_ids = real_ids + [_mid() for _ in range(996)]
        assert len(all_ids) == 1001

        import memorylayer_saas.storage.postgresql as pg_module
        with patch.object(pg_module, "MemoryAssociationModel", _AssocModelSQLite):
            results = await backend.get_associations_batch(ws, all_ids, direction="outgoing")

        assert {a.id for a in results} == expected_aids
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_domain_object_fields(self, backend, session_factory):
        """Returned Association objects have correct field values."""
        ws = f"ws_{uuid.uuid4().hex[:8]}"
        m1, m2 = _mid(), _mid()
        for m in (m1, m2):
            await _insert_memory(session_factory, ws, m)

        aid = await _insert_assoc(session_factory, ws, m1, m2, relation_type="CAUSES", strength=0.9)

        import memorylayer_saas.storage.postgresql as pg_module
        with patch.object(pg_module, "MemoryAssociationModel", _AssocModelSQLite):
            results = await backend.get_associations_batch(ws, [m1], direction="outgoing")

        assert len(results) == 1
        a = results[0]
        assert a.id == aid
        assert a.workspace_id == ws
        assert a.source_id == m1
        assert a.target_id == m2
        assert a.relationship == "CAUSES"
        assert abs(a.strength - 0.9) < 1e-6
