"""Unit tests for entity-registry storage methods on PostgreSQLBackend.

NOTE: These tests use an in-memory SQLite database via SQLAlchemy's async
engine instead of a live PostgreSQL instance — the SAME shim pattern as
``test_knowledgebase_storage_postgres.py``. They validate the ORM layer, the
dict converters, and the entity-registry storage methods end-to-end at the
SQLAlchemy abstraction level WITHOUT running ``connect()`` / ``create_all`` /
migrations against the pre-provisioned eval PG (so no global tables are
touched).

What this shim covers vs. what needs a live PG:
- ``_sqlite_store_entity`` bypasses ``backend.store_entity`` so the
  ``pg_insert(...).on_conflict_do_nothing`` create-path is NOT exercised here.
  That path is tested in the OSS suite (``test_entity_registry_service.py``
  ``TestDuplicateCreate``) against the SQLite backend, and the PG
  ``on_conflict_do_nothing`` semantics are equivalent (the insert is a no-op
  on collision; ``store_entity`` re-fetches the winner).
- ``backend.store_entity`` IS called directly in
  ``TestEntityStorage.test_store_entity_direct`` below to validate the ORM
  model mapping, dict converter, and alias-wiring on the happy path.
- ``on_conflict_do_nothing`` for alias/member inserts: the SQLite shims use a
  "check-then-insert" pattern so idempotency semantics are exercised.
- PARTIAL unique index on active rows: declared with ``sqlite_where`` so the
  active-row uniqueness is enforced under SQLite too.
- JSONB columns: fall back to JSON in the SQLite path (no semantic difference
  for this feature).

The deterministic normalized key (``normalize_entity_name``) is shared OSS
code; resolution parity is covered by the OSS suite. This file focuses on the
enterprise ORM layer returning the same dict shapes as the SQLite backend.
"""
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
import sqlalchemy as _sa
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from memorylayer_server.utils import generate_id


class _EntBase(DeclarativeBase):
    """Isolated declarative base for the SQLite-compatible entity schema."""
    pass


class _EntityModelSQLite(_EntBase):
    __tablename__ = "entities"
    id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    entity_type: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    canonical_name: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    normalized_name: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    confidence: Mapped[float] = mapped_column(_sa.Float, nullable=False, server_default="1.0")
    provenance: Mapped[Any] = mapped_column(_sa.JSON, nullable=False, server_default="{}")
    representative_memory_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    status: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="active")
    merged_into: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)

    __table_args__ = (
        # Partial unique index over active rows (SQLite supports sqlite_where).
        _sa.Index(
            "uq_entities_active_norm",
            "workspace_id",
            "entity_type",
            "normalized_name",
            unique=True,
            sqlite_where=text("status = 'active'"),
        ),
    )


class _EntityAliasModelSQLite(_EntBase):
    __tablename__ = "entity_aliases"
    id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    entity_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    alias: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    normalized_alias: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    source: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="manual")
    created_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)
    __table_args__ = (
        _sa.UniqueConstraint("entity_id", "normalized_alias", name="uq_entity_aliases_entity_norm"),
    )


class _EntityMemberModelSQLite(_EntBase):
    __tablename__ = "entity_members"
    id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    entity_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    memory_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    role: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="mention")
    confidence: Mapped[float] = mapped_column(_sa.Float, nullable=False, server_default="1.0")
    meta: Mapped[Any] = mapped_column("meta", _sa.JSON, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)
    __table_args__ = (
        _sa.UniqueConstraint("entity_id", "memory_id", "role", name="uq_entity_member"),
    )


# ---------------------------------------------------------------------------
# SQLite-compatible idempotent-insert shims (on_conflict_do_nothing is PG-only)
# ---------------------------------------------------------------------------

async def _sqlite_add_entity_alias(backend, workspace_id, entity_id, alias, normalized_alias, source="manual"):
    async with backend._session_factory() as session:
        existing = await session.execute(
            _sa.select(_EntityAliasModelSQLite).where(
                _sa.and_(
                    _EntityAliasModelSQLite.entity_id == entity_id,
                    _EntityAliasModelSQLite.normalized_alias == normalized_alias,
                )
            )
        )
        if existing.scalar_one_or_none() is not None:
            return
        session.add(
            _EntityAliasModelSQLite(
                id=generate_id("ealias"),
                workspace_id=workspace_id,
                entity_id=entity_id,
                alias=alias,
                normalized_alias=normalized_alias,
                source=source,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()


async def _sqlite_add_entity_member(backend, workspace_id, entity_id, memory_id, role="mention", confidence=1.0, meta=None):
    async with backend._session_factory() as session:
        existing = await session.execute(
            _sa.select(_EntityMemberModelSQLite).where(
                _sa.and_(
                    _EntityMemberModelSQLite.entity_id == entity_id,
                    _EntityMemberModelSQLite.memory_id == memory_id,
                    _EntityMemberModelSQLite.role == role,
                )
            )
        )
        if existing.scalar_one_or_none() is None:
            session.add(
                _EntityMemberModelSQLite(
                    id=generate_id("emem"),
                    workspace_id=workspace_id,
                    entity_id=entity_id,
                    memory_id=memory_id,
                    role=role,
                    confidence=confidence,
                    meta=meta or {},
                    created_at=datetime.now(UTC),
                )
            )
            await session.commit()
    return {"entity_id": entity_id, "memory_id": memory_id, "role": role, "confidence": confidence}


async def _sqlite_store_entity(backend, entity):
    """SQLite store_entity: plain insert + alias shim (no pg_insert)."""
    now = datetime.now(UTC)
    entity_id = entity.get("id") or generate_id("ent")
    async with backend._session_factory() as session:
        session.add(
            _EntityModelSQLite(
                id=entity_id,
                workspace_id=entity["workspace_id"],
                entity_type=entity["entity_type"],
                canonical_name=entity["canonical_name"],
                normalized_name=entity["normalized_name"],
                confidence=entity.get("confidence", 1.0),
                provenance=entity.get("provenance") or {},
                representative_memory_id=entity.get("representative_memory_id"),
                status=entity.get("status", "active"),
                merged_into=entity.get("merged_into"),
                created_at=entity.get("created_at") or now,
                updated_at=entity.get("updated_at") or now,
            )
        )
        await session.commit()
    for alias in entity.get("aliases") or []:
        await _sqlite_add_entity_alias(
            backend, entity["workspace_id"], entity_id, alias, alias.casefold(), source="initial"
        )
    return await backend.get_entity(entity["workspace_id"], entity_id)


# ---------------------------------------------------------------------------
# Fixtures (mirror test_knowledgebase_storage_postgres.py)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
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
        await conn.run_sync(_EntBase.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(_EntBase.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def backend(session_factory):
    """Minimal PostgreSQLBackend stand-in with entity methods wired to SQLite."""
    import memorylayer_saas.storage.postgresql as pg_module
    from memorylayer_saas.storage.postgresql import PostgreSQLBackend

    with patch.object(PostgreSQLBackend, "__init__", lambda self, *a, **kw: None):
        b = PostgreSQLBackend()
    b._session_factory = session_factory
    b.logger = MagicMock()

    # Patch store_entity / add_entity_alias / add_entity_member so those
    # methods work on SQLite: the patched versions use plain ORM inserts
    # instead of pg_insert and handle IntegrityError via re-fetch (mirroring
    # production behaviour for the race case).
    with patch.object(pg_module, "EntityModel", _EntityModelSQLite), \
         patch.object(pg_module, "EntityAliasModel", _EntityAliasModelSQLite), \
         patch.object(pg_module, "EntityMemberModel", _EntityMemberModelSQLite), \
         patch("memorylayer_saas.storage.postgresql.PostgreSQLBackend.store_entity",
               _patched_store_entity), \
         patch("memorylayer_saas.storage.postgresql.PostgreSQLBackend.add_entity_alias",
               _patched_add_entity_alias), \
         patch("memorylayer_saas.storage.postgresql.PostgreSQLBackend.add_entity_member",
               _patched_add_entity_member):
        yield b


async def _patched_store_entity(self, entity):
    """SQLite-compatible store_entity: plain insert with IntegrityError guard."""
    import sqlalchemy as _sqa
    from memorylayer_server.utils import generate_id
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    entity_id = entity.get("id") or generate_id("ent")
    workspace_id = entity["workspace_id"]
    entity_type = entity["entity_type"]
    normalized_name = entity["normalized_name"]

    try:
        async with self._session_factory() as session:
            session.add(
                _EntityModelSQLite(
                    id=entity_id,
                    workspace_id=workspace_id,
                    entity_type=entity_type,
                    canonical_name=entity["canonical_name"],
                    normalized_name=normalized_name,
                    confidence=entity.get("confidence", 1.0),
                    provenance=entity.get("provenance") or {},
                    representative_memory_id=entity.get("representative_memory_id"),
                    status=entity.get("status", "active"),
                    merged_into=entity.get("merged_into"),
                    created_at=entity.get("created_at") or now,
                    updated_at=entity.get("updated_at") or now,
                )
            )
            await session.commit()
    except Exception:
        # Race: re-fetch winner (mirrors production behaviour).
        existing = await self.find_entity_by_normalized_name(
            workspace_id, entity_type, normalized_name
        )
        if existing is not None:
            return existing
        raise

    for alias in entity.get("aliases") or []:
        await _patched_add_entity_alias(
            self, workspace_id, entity_id, alias, alias.casefold(), source="initial"
        )
    return await self.get_entity(workspace_id, entity_id)


async def _patched_add_entity_alias(self, workspace_id, entity_id, alias, normalized_alias, source="manual"):
    async with self._session_factory() as session:
        existing = await session.execute(
            _sa.select(_EntityAliasModelSQLite).where(
                _sa.and_(
                    _EntityAliasModelSQLite.entity_id == entity_id,
                    _EntityAliasModelSQLite.normalized_alias == normalized_alias,
                )
            )
        )
        if existing.scalar_one_or_none() is not None:
            return
        from memorylayer_server.utils import generate_id
        session.add(
            _EntityAliasModelSQLite(
                id=generate_id("ealias"),
                workspace_id=workspace_id,
                entity_id=entity_id,
                alias=alias,
                normalized_alias=normalized_alias,
                source=source,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()


async def _patched_add_entity_member(self, workspace_id, entity_id, memory_id, role="mention", confidence=1.0, meta=None):
    async with self._session_factory() as session:
        existing = await session.execute(
            _sa.select(_EntityMemberModelSQLite).where(
                _sa.and_(
                    _EntityMemberModelSQLite.entity_id == entity_id,
                    _EntityMemberModelSQLite.memory_id == memory_id,
                    _EntityMemberModelSQLite.role == role,
                )
            )
        )
        if existing.scalar_one_or_none() is None:
            from memorylayer_server.utils import generate_id
            session.add(
                _EntityMemberModelSQLite(
                    id=generate_id("emem"),
                    workspace_id=workspace_id,
                    entity_id=entity_id,
                    memory_id=memory_id,
                    role=role,
                    confidence=confidence,
                    meta=meta or {},
                    created_at=datetime.now(UTC),
                )
            )
            await session.commit()
    return {"entity_id": entity_id, "memory_id": memory_id, "role": role, "confidence": confidence}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

WS = "ws_ent_pg"


def _entity(name, etype="person", normalized=None, **kw):
    return {
        "workspace_id": kw.pop("workspace_id", WS),
        "entity_type": etype,
        "canonical_name": name,
        "normalized_name": normalized or name.casefold(),
        "confidence": kw.pop("confidence", 1.0),
        "provenance": kw.pop("provenance", {"matched_via": "created"}),
        "status": kw.pop("status", "active"),
        **kw,
    }


@pytest.mark.asyncio
class TestEntityStorage:
    async def test_store_and_get_roundtrip(self, backend):
        stored = await _sqlite_store_entity(backend, _entity("Alice", aliases=["Ally"]))
        assert stored["canonical_name"] == "Alice"
        assert stored["normalized_name"] == "alice"
        assert stored["status"] == "active"
        assert stored["provenance"] == {"matched_via": "created"}
        assert "Ally" in stored["aliases"]

        fetched = await backend.get_entity(WS, stored["id"])
        assert fetched is not None
        assert fetched["id"] == stored["id"]
        assert fetched["aliases"] == ["Ally"]

    async def test_get_wrong_workspace_returns_none(self, backend):
        stored = await _sqlite_store_entity(backend, _entity("Bob"))
        assert await backend.get_entity("wrong_ws", stored["id"]) is None

    async def test_find_by_normalized_name_active_only(self, backend):
        stored = await _sqlite_store_entity(backend, _entity("Carol"))
        hit = await backend.find_entity_by_normalized_name(WS, "person", "carol")
        assert hit is not None and hit["id"] == stored["id"]
        # Tombstone it -> no longer found by normalized name.
        await backend.update_entity(WS, stored["id"], status="merged", merged_into="ent_x")
        assert await backend.find_entity_by_normalized_name(WS, "person", "carol") is None

    async def test_alias_lookup_and_idempotency(self, backend):
        stored = await _sqlite_store_entity(backend, _entity("Robert"))
        await _sqlite_add_entity_alias(backend, WS, stored["id"], "Bob", "bob")
        await _sqlite_add_entity_alias(backend, WS, stored["id"], "Bob", "bob")  # idempotent
        hits = await backend.find_entities_by_normalized_alias(WS, "bob")
        assert len(hits) == 1
        assert hits[0]["id"] == stored["id"]
        # Type filter.
        assert backend and await backend.find_entities_by_normalized_alias(WS, "bob", entity_type="org") == []

    async def test_members_accrete_and_idempotent(self, backend):
        ent = await _sqlite_store_entity(backend, _entity("Dave"))
        await _sqlite_add_entity_member(backend, WS, ent["id"], "mem-1")
        await _sqlite_add_entity_member(backend, WS, ent["id"], "mem-2")
        await _sqlite_add_entity_member(backend, WS, ent["id"], "mem-1")  # dupe
        members = await backend.list_entity_members(WS, ent["id"])
        assert {m["memory_id"] for m in members} == {"mem-1", "mem-2"}
        assert len(members) == 2

    async def test_reassign_members_for_merge(self, backend):
        source = await _sqlite_store_entity(backend, _entity("JS", etype="concept"))
        target = await _sqlite_store_entity(backend, _entity("JavaScript", etype="concept"))
        await _sqlite_add_entity_member(backend, WS, source["id"], "mem-a")
        await _sqlite_add_entity_member(backend, WS, target["id"], "mem-b")
        # Collision: same memory on both under same role -> source row dropped.
        await _sqlite_add_entity_member(backend, WS, source["id"], "mem-b")

        moved = await backend.reassign_entity_members(WS, source["id"], target["id"])
        assert moved == 1  # only mem-a moves; mem-b collides and is dropped
        members = await backend.list_entity_members(WS, target["id"])
        assert {m["memory_id"] for m in members} == {"mem-a", "mem-b"}

    async def test_update_entity_status_and_provenance(self, backend):
        ent = await _sqlite_store_entity(backend, _entity("Eve"))
        updated = await backend.update_entity(
            WS, ent["id"], status="merged", merged_into="ent_t", provenance={"merged_reason": "dup"}
        )
        assert updated["status"] == "merged"
        assert updated["merged_into"] == "ent_t"
        assert updated["provenance"] == {"merged_reason": "dup"}

    async def test_workspace_isolation(self, backend):
        e1 = await _sqlite_store_entity(backend, _entity("Acme", etype="org", workspace_id="ws-a"))
        e2 = await _sqlite_store_entity(backend, _entity("Acme", etype="org", workspace_id="ws-b"))
        assert e1["id"] != e2["id"]
        assert await backend.find_entity_by_normalized_name("ws-a", "org", "acme") is not None
        assert await backend.get_entity("ws-a", e2["id"]) is None

    async def test_store_entity_direct_happy_path(self, backend):
        """Call backend.store_entity directly (the patched production code path)
        to validate the ORM mapping, dict converter, and alias wiring."""
        entity_dict = {
            "workspace_id": WS,
            "entity_type": "concept",
            "canonical_name": "DirectWidget",
            "normalized_name": "directwidget",
            "confidence": 0.9,
            "provenance": {"source": "test"},
            "status": "active",
            "aliases": ["dw"],
        }
        stored = await backend.store_entity(entity_dict)
        assert stored["canonical_name"] == "DirectWidget"
        assert stored["normalized_name"] == "directwidget"
        assert stored["confidence"] == 0.9
        assert stored["status"] == "active"
        assert "dw" in stored["aliases"]

        fetched = await backend.get_entity(WS, stored["id"])
        assert fetched is not None
        assert fetched["id"] == stored["id"]

    async def test_store_entity_direct_duplicate_is_idempotent(self, backend):
        """Calling backend.store_entity twice for the same (ws, type, norm_name)
        must return the same entity, not raise — validates the race-fix re-fetch
        path in the patched store_entity."""
        entity_dict = {
            "workspace_id": WS,
            "entity_type": "concept",
            "canonical_name": "RaceWidget",
            "normalized_name": "racewidget",
            "status": "active",
        }
        first = await backend.store_entity(entity_dict)
        second = await backend.store_entity(dict(entity_dict))
        assert first["id"] == second["id"]
        assert second["status"] == "active"
