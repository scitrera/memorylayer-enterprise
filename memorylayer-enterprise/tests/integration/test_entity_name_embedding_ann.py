"""Integration test for the enterprise name-embedding ANN storage method.

pgvector's cosine operator (``<=>``) needs a real PostgreSQL, so this test is
GATED on ``ML_AGE_TEST_DATABASE_URL`` and SKIPPED otherwise:

    ML_AGE_TEST_DATABASE_URL=postgresql://memorylayer:memorylayer_dev@localhost:55432/memorylayer \\
        .venv/bin/python -m pytest tests/integration/test_entity_name_embedding_ann.py

Isolation + cleanup: a unique workspace id per run; teardown deletes ONLY that
workspace's entity/alias/member rows + the workspace itself. NEVER a global
truncate. Embeddings are produced by the deterministic ``hash`` provider so the
ranking is reproducible without an embed server.
"""
import os
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.embedding.hash import HashEmbeddingProvider
from memorylayer_server.services.entity_registry._normalize import normalize_entity_name
from memorylayer_server.utils import generate_id

_DB_URL = os.getenv("ML_AGE_TEST_DATABASE_URL", "").strip()

pytestmark = pytest.mark.skipif(
    not _DB_URL,
    reason="Set ML_AGE_TEST_DATABASE_URL to a PostgreSQL+pgvector database to run the entity name-embedding ANN test.",
)

# Hash provider dim must match the deployed entities.name_embedding column dim
# (MEMORYLAYER_EMBEDDING_DIMENSIONS, default 1536). The eval PG was migrated with
# that default, so embed at that width here.
_DIM = int(os.getenv("MEMORYLAYER_EMBEDDING_DIMENSIONS", "1536"))


def _asyncpg_url(url: str) -> str:
    return (
        url.replace("postgres://", "postgresql://")
        .replace("postgresql+asyncpg://", "postgresql://")
        .replace("postgresql://", "postgresql+asyncpg://")
    )


@pytest_asyncio.fixture
async def backend():
    from memorylayer_saas.storage.postgresql import PostgreSQLBackend

    from scitrera_app_framework import Variables

    from memorylayer_saas.storage.models import (
        Base,
        EntityAliasModel,
        EntityMemberModel,
        EntityModel,
    )

    v = Variables()
    b = PostgreSQLBackend(v=v, connection_string=_asyncpg_url(_DB_URL))
    engine = create_async_engine(_asyncpg_url(_DB_URL), pool_pre_ping=True)
    b._engine = engine
    b._session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Provision the entity-registry tables (with name_embedding + HNSW index)
    # if they don't already exist. create_all is idempotent (checkfirst=True)
    # and scoped to just these three tables, so it never touches other schema.
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[
                EntityModel.__table__,
                EntityAliasModel.__table__,
                EntityMemberModel.__table__,
            ],
        )

    created: list[str] = []
    try:
        yield b, created
    finally:
        async with engine.connect() as conn:
            raw = (await conn.get_raw_connection()).driver_connection
            for ws in created:
                await raw.execute("DELETE FROM entity_members WHERE workspace_id = $1", ws)
                await raw.execute("DELETE FROM entity_aliases WHERE workspace_id = $1", ws)
                await raw.execute("DELETE FROM entities WHERE workspace_id = $1", ws)
                await raw.execute("DELETE FROM workspaces WHERE id = $1", ws)
        await engine.dispose()


async def _insert_entity(storage, ws_id, name, *, entity_type="person", name_embedding=None):
    """Insert an entity row directly via the ORM.

    This bypasses ``store_entity``'s ``on_conflict_do_nothing`` create-path on
    purpose (the same scoping documented in the SQLite-shim storage test): this
    file tests the ANN read method, not the conflict-tolerant insert path. The
    happy-path insert + dict-converter is covered by the SQLite-shim test.
    """
    from memorylayer_saas.storage.models import EntityModel

    async with storage._session_factory() as session:
        session.add(
            EntityModel(
                id=generate_id("ent"),
                workspace_id=ws_id,
                entity_type=entity_type,
                canonical_name=name,
                normalized_name=normalize_entity_name(name),
                status="active",
                name_embedding=name_embedding,
            )
        )
        await session.commit()
    return await storage.find_entity_by_normalized_name(
        ws_id, entity_type, normalize_entity_name(name)
    )


async def _ensure_ws(storage, ws_id: str) -> None:
    if not await storage.get_workspace(ws_id):
        await storage.create_workspace(
            Workspace(
                id=ws_id,
                tenant_id="default_tenant",
                name="entity ann %s" % ws_id,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )


@pytest.mark.asyncio
async def test_find_entities_by_name_embedding_ranks_by_cosine(backend):
    storage, created = backend
    ws = "ws-ann-%s" % generate_id("t")
    created.append(ws)
    await _ensure_ws(storage, ws)

    provider = HashEmbeddingProvider(v=None, dimensions=_DIM)

    async def _store(name: str):
        return await _insert_entity(
            storage, ws, name, name_embedding=await provider.embed(normalize_entity_name(name))
        )

    target = await _store("Caroline Chen")
    await _store("Microsoft")
    # An entity with NO embedding must be excluded from ANN results.
    await _insert_entity(storage, ws, "No Vector", name_embedding=None)

    query = await provider.embed(normalize_entity_name("Chen Caroline"))
    hits = await storage.find_entities_by_name_embedding(
        ws, "person", query, limit=5, min_score=0.5
    )

    assert hits, "expected at least the high-similarity match"
    # Best hit is Caroline Chen (cosine ~1.0 with reordered tokens).
    assert hits[0]["id"] == target["id"]
    assert hits[0]["score"] >= 0.92
    # Microsoft (cosine 0.0) is filtered by min_score; No-Vector entity excluded.
    names = {h["canonical_name"] for h in hits}
    assert "Microsoft" not in names
    assert "No Vector" not in names


@pytest.mark.asyncio
async def test_store_entity_conflict_returns_existing_no_duplicate(backend):
    """Real-PG store_entity called twice with the same active key must:
      - not raise (on_conflict_do_nothing with index_elements + index_where)
      - return the original row (same entity id) on the second call
      - leave exactly one active row in the table

    This is the test the prior slice lacked — the SQLite-shim bypasses the
    production on_conflict path, so the constraint= bug was invisible. This
    test exercises the real PostgreSQLBackend.store_entity on real PG.
    """
    storage, created = backend
    ws = "ws-conflict-%s" % generate_id("t")
    created.append(ws)
    await _ensure_ws(storage, ws)

    entity_data = {
        "workspace_id": ws,
        "entity_type": "person",
        "canonical_name": "Conflict Test Entity",
        "normalized_name": "conflict test entity",
        "status": "active",
        "confidence": 1.0,
        "provenance": {"matched_via": "created"},
    }

    # First call: fresh insert, should succeed and return the new row.
    first = await storage.store_entity(dict(entity_data))
    assert first is not None
    assert first["canonical_name"] == "Conflict Test Entity"
    assert first["status"] == "active"

    # Second call: same active (workspace, entity_type, normalized_name) key.
    # Must not raise UndefinedObjectError; must return the first entity.
    second = await storage.store_entity(dict(entity_data))
    assert second is not None
    assert second["id"] == first["id"], (
        "store_entity collision must return the winning row's id, not a new one"
    )

    # Exactly one active row for this workspace+type+name.
    async with storage._session_factory() as session:
        from sqlalchemy import func, select
        from memorylayer_saas.storage.models import EntityModel

        count = await session.scalar(
            select(func.count()).where(
                EntityModel.workspace_id == ws,
                EntityModel.entity_type == "person",
                EntityModel.normalized_name == "conflict test entity",
                EntityModel.status == "active",
            )
        )
    assert count == 1, "expected exactly one active entity row after duplicate store"


@pytest.mark.asyncio
async def test_ann_respects_type_and_workspace_isolation(backend):
    storage, created = backend
    ws_a = "ws-ann-a-%s" % generate_id("t")
    ws_b = "ws-ann-b-%s" % generate_id("t")
    created.extend([ws_a, ws_b])
    await _ensure_ws(storage, ws_a)
    await _ensure_ws(storage, ws_b)

    provider = HashEmbeddingProvider(v=None, dimensions=_DIM)
    emb = await provider.embed("caroline chen")

    # Same name in ws_b and as a different type in ws_a — neither should leak.
    await _insert_entity(storage, ws_b, "Caroline Chen", entity_type="person", name_embedding=emb)
    await _insert_entity(storage, ws_a, "Caroline Chen", entity_type="org", name_embedding=emb)

    # Query ws_a / person: no person rows there -> empty.
    hits = await storage.find_entities_by_name_embedding(ws_a, "person", emb, min_score=0.5)
    assert hits == []
