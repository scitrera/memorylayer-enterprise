"""Graph-moat C1/C2 integration tests: Entity materialization + entity_neighborhood.

C1 (Entity materialization into AGE):
  * materialize a workspace with registry entities -> assert Entity vertices +
    MENTIONS edges exist (cypher count);
  * idempotent re-materialize (delete-then-MERGE -> same counts);
  * watermark skip on an unchanged registry (no re-materialize) and a forced
    re-materialize when the entity watermark advances.

C2 (entity_neighborhood — the consumer that makes C1 non-speculative):
  * the enterprise AGE neighborhood (reading the C1 Entity/MENTIONS vertices+edges)
    returns the SAME normalized set as the OSS relational fallback on the same data;
  * an empty/disabled registry -> empty neighborhood, no error.

Injection conformance: a malicious workspace_id / entity_id is rejected with
``ValueError`` BEFORE reaching cypher (mirrors the existing graph-query
injection conformance test).

Requires a PostgreSQL-with-AGE database. SKIPPED unless ``ML_AGE_TEST_DATABASE_URL``
is set, e.g.::

    ML_AGE_TEST_DATABASE_URL=postgresql://memorylayer:memorylayer_dev@localhost:55432/memorylayer \\
        pytest tests/integration/test_age_entity_materialize.py -v

Isolation + cleanup: every fixture runs under a unique workspace id
(prefix ``ent_mat_<uuid>``). Teardown deletes ONLY its own relational rows
(entities/members/memories) and its own AGE subgraph (Entity + Memory layers) —
never truncates shared tables.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from scitrera_app_framework import Variables
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from memorylayer_server.models.memory import RememberInput
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.graph_query.default import RelationalGraphQueryService
from memorylayer_saas.services.graph_analysis import _cypher
from memorylayer_saas.services.graph_analysis._cypher import (
    delete_workspace_edges_sql,
    delete_workspace_entities_sql,
    delete_workspace_entity_mentions_sql,
    delete_workspace_vertices_sql,
)
from memorylayer_saas.services.graph_analysis._materialize import (
    compute_entity_watermark,
    materialize_workspace_entities_gated,
    materialize_workspace_subgraph_gated,
)
from memorylayer_saas.services.graph_analysis.bootstrap import (
    AGE_SEARCH_PATH,
    bootstrap_age,
    ensure_age_session,
)
from memorylayer_saas.services.graph_query.age import AgeGraphQueryService

# ---------------------------------------------------------------------------
# Skip gate
# ---------------------------------------------------------------------------

_AGE_DB_URL = os.getenv("ML_AGE_TEST_DATABASE_URL", "").strip()

pytestmark = pytest.mark.skipif(
    not _AGE_DB_URL,
    reason="Set ML_AGE_TEST_DATABASE_URL to a PostgreSQL-with-AGE database to run AGE entity-materialize tests.",
)


# ---------------------------------------------------------------------------
# Storage backend fixture (cloned from test_age_graph_query_conformance.py)
# ---------------------------------------------------------------------------


def _asyncpg_url(url: str) -> str:
    return (
        url.replace("postgres://", "postgresql://")
        .replace("postgresql+asyncpg://", "postgresql://")
        .replace("postgresql://", "postgresql+asyncpg://")
    )


def _make_age_engine(url: str):
    return create_async_engine(
        _asyncpg_url(url),
        pool_pre_ping=True,
        connect_args={"server_settings": {"search_path": AGE_SEARCH_PATH}},
    )


@pytest_asyncio.fixture
async def age_storage():
    from memorylayer_saas.storage.postgresql import PostgreSQLBackend

    v = Variables()
    backend = PostgreSQLBackend(v=v, connection_string=_asyncpg_url(_AGE_DB_URL))
    engine = _make_age_engine(_AGE_DB_URL)
    backend._engine = engine
    backend._session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    # Bootstrap AGE + the memorylayer graph once for the fixture.
    async with engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await bootstrap_age(raw)

    created_workspaces: list[str] = []
    try:
        yield backend, created_workspaces
    finally:
        await _cleanup(backend, created_workspaces)
        await engine.dispose()


async def _cleanup(backend, workspace_ids: list[str]) -> None:
    if not workspace_ids:
        return
    engine = backend._engine
    async with engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)
        for ws in workspace_ids:
            ws_param = json.dumps({"ws": ws})
            # Entity layer first, then Memory layer (no FK between AGE layers,
            # but keep a stable order).
            await raw.execute(delete_workspace_entity_mentions_sql(), ws_param)
            await raw.execute(delete_workspace_entities_sql(), ws_param)
            await raw.execute(delete_workspace_edges_sql(), ws_param)
            await raw.execute(delete_workspace_vertices_sql(), ws_param)
            await raw.execute("DELETE FROM entity_members WHERE workspace_id = $1", ws)
            await raw.execute("DELETE FROM entity_aliases WHERE workspace_id = $1", ws)
            await raw.execute("DELETE FROM entities WHERE workspace_id = $1", ws)
            await raw.execute("DELETE FROM memory_associations WHERE workspace_id = $1", ws)
            await raw.execute("DELETE FROM memories WHERE workspace_id = $1", ws)
            await raw.execute("DELETE FROM age_entity_materialization_watermark WHERE workspace_id = $1", ws)
            await raw.execute("DELETE FROM age_materialization_watermark WHERE workspace_id = $1", ws)
            await raw.execute("DELETE FROM workspaces WHERE id = $1", ws)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


async def _ensure_ws(storage, ws_id: str) -> None:
    if not await storage.get_workspace(ws_id):
        await storage.create_workspace(
            Workspace(
                id=ws_id,
                tenant_id="default_tenant",
                name="Entity-mat %s" % ws_id,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )


async def _add_mem(storage, ws_id: str, content: str) -> str:
    mem = await storage.create_memory(ws_id, RememberInput(content=content))
    return mem.id


async def _build_registry(storage, ws_id: str) -> dict[str, str]:
    """alice(self->m1, mention->m2), bob(mention->m2), proj(mention->m1,m3)."""
    await _ensure_ws(storage, ws_id)
    mems = {f"m{i}": await _add_mem(storage, ws_id, "ent node %d" % i) for i in range(4)}
    alice = await storage.store_entity(
        {"workspace_id": ws_id, "entity_type": "person", "canonical_name": "Alice", "normalized_name": "alice"}
    )
    bob = await storage.store_entity(
        {"workspace_id": ws_id, "entity_type": "person", "canonical_name": "Bob", "normalized_name": "bob"}
    )
    proj = await storage.store_entity(
        {"workspace_id": ws_id, "entity_type": "concept", "canonical_name": "Proj", "normalized_name": "proj"}
    )
    await storage.add_entity_member(ws_id, alice["id"], mems["m1"], role="self")
    await storage.add_entity_member(ws_id, alice["id"], mems["m2"], role="mention")
    await storage.add_entity_member(ws_id, bob["id"], mems["m2"], role="mention")
    await storage.add_entity_member(ws_id, proj["id"], mems["m1"], role="mention")
    await storage.add_entity_member(ws_id, proj["id"], mems["m3"], role="mention")
    return {"alice": alice["id"], "bob": bob["id"], "proj": proj["id"], **mems}


async def _materialize_full(storage, ws_id: str) -> None:
    """Materialize the Memory subgraph then the Entity layer (what C2 does)."""
    engine = storage._engine
    memories = await storage.search_memories_by_filter(ws_id, status="active", limit=10000)
    node_attrs = {
        m.id: {"memory_type": getattr(m, "memory_type", None), "memory_subtype": getattr(m, "subtype", None)}
        for m in memories
    }
    await materialize_workspace_subgraph_gated(
        engine, storage=storage, workspace_id=ws_id, node_attrs=node_attrs, associations=[]
    )
    entities = await storage.list_workspace_entities(ws_id)
    members = await storage.list_workspace_entity_members(ws_id)
    return await materialize_workspace_entities_gated(
        engine, workspace_id=ws_id, entities=entities, members=members
    )


async def _count_entity_vertices(storage, ws_id: str) -> int:
    engine = storage._engine
    sql = (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (e:Entity {workspace_id: $ws}) RETURN count(e)"
        "$$, $1) AS (cnt agtype)" % _cypher.GRAPH_NAME
    )
    async with engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)
        rows = await raw.fetch(sql, json.dumps({"ws": ws_id}))
    return int(json.loads(rows[0]["cnt"]))


async def _count_mentions_edges(storage, ws_id: str) -> int:
    engine = storage._engine
    sql = (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (e:Entity {workspace_id: $ws})-[r:MENTIONS]->(m:Memory {workspace_id: $ws})"
        " RETURN count(r)"
        "$$, $1) AS (cnt agtype)" % _cypher.GRAPH_NAME
    )
    async with engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)
        rows = await raw.fetch(sql, json.dumps({"ws": ws_id}))
    return int(json.loads(rows[0]["cnt"]))


# ---------------------------------------------------------------------------
# C1 — materialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_c1_materialize_creates_entity_vertices_and_mentions(age_storage):
    storage, created = age_storage
    ws = "ent_mat_create_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _build_registry(storage, ws)

    ran = await _materialize_full(storage, ws)
    assert ran is True
    # 3 entities, 5 member edges (all memories exist as Memory vertices).
    assert await _count_entity_vertices(storage, ws) == 3
    assert await _count_mentions_edges(storage, ws) == 5


@pytest.mark.asyncio
async def test_c1_idempotent_re_materialize_same_counts(age_storage):
    storage, created = age_storage
    ws = "ent_mat_idem_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _build_registry(storage, ws)

    await _materialize_full(storage, ws)
    v1 = await _count_entity_vertices(storage, ws)
    e1 = await _count_mentions_edges(storage, ws)

    # Force a full re-materialize (bypass the gate) to prove delete-then-MERGE
    # is idempotent: counts must not double.
    entities = await storage.list_workspace_entities(ws)
    members = await storage.list_workspace_entity_members(ws)
    from memorylayer_saas.services.graph_analysis._materialize import materialize_workspace_entities

    await materialize_workspace_entities(
        storage._engine, workspace_id=ws, entities=entities, members=members
    )
    assert await _count_entity_vertices(storage, ws) == v1 == 3
    assert await _count_mentions_edges(storage, ws) == e1 == 5


@pytest.mark.asyncio
async def test_c1_watermark_skips_unchanged_registry(age_storage):
    storage, created = age_storage
    ws = "ent_mat_wm_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _build_registry(storage, ws)

    # First materialize runs the work.
    assert await _materialize_full(storage, ws) is True
    # Second materialize on an UNCHANGED registry is skipped by the watermark.
    assert await _materialize_full(storage, ws) is False
    # Counts unchanged.
    assert await _count_entity_vertices(storage, ws) == 3

    # Adding a member advances the entity watermark -> re-materialize runs.
    extra_mem = await _add_mem(storage, ws, "extra")
    entities = await storage.list_workspace_entities(ws)
    alice_id = next(e["id"] for e in entities if e["canonical_name"] == "Alice")
    await storage.add_entity_member(ws, alice_id, extra_mem, role="mention")
    assert await _materialize_full(storage, ws) is True
    assert await _count_mentions_edges(storage, ws) == 6


@pytest.mark.asyncio
async def test_c1_empty_registry_is_clean_noop(age_storage):
    storage, created = age_storage
    ws = "ent_mat_empty_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _ensure_ws(storage, ws)

    # No entities/members: materialize is a clean no-op, zero vertices.
    await _materialize_full(storage, ws)
    assert await _count_entity_vertices(storage, ws) == 0
    assert await _count_mentions_edges(storage, ws) == 0


def test_c1_entity_watermark_token_shape():
    # Pure-unit check (no DB): token is (entity_count, max_updated, member_count).
    wm = compute_entity_watermark(
        [{"id": "e1", "updated_at": "2026-01-02T00:00:00"}, {"id": "e2", "updated_at": "2026-01-03T00:00:00"}],
        [{"entity_id": "e1", "memory_id": "m1", "role": "self"}],
    )
    # Token is now 4 elements: [entity_count, max_updated, member_count, digest].
    assert len(wm) == 4
    assert wm[0] == 2
    assert wm[1] == "2026-01-03T00:00:00"
    assert wm[2] == 1
    assert isinstance(wm[3], str) and len(wm[3]) == 16
    # Empty registry token.
    empty_wm = compute_entity_watermark([], [])
    assert empty_wm[0] == 0 and empty_wm[1] == "" and empty_wm[2] == 0


# ---------------------------------------------------------------------------
# C2 — entity_neighborhood (AGE reads C1's output; parity vs OSS relational)
# ---------------------------------------------------------------------------


def _norm_neighborhood(res) -> dict:
    return {
        "entity_id": res.entity_id,
        "memories": sorted((m.memory_id, m.role) for m in res.memories_for_entity),
        "co": sorted((c.entity_id, c.shared_memory_count) for c in res.entities_co_mentioned),
        "truncated": res.truncated,
    }


@pytest.mark.asyncio
async def test_c2_age_reads_c1_output(age_storage, monkeypatch):
    storage, created = age_storage
    ws = "ent_mat_c2_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    r = await _build_registry(storage, ws)

    v = Variables()
    # The AGE backend gates entity materialization behind the flag; enable it so
    # entity_neighborhood materializes (C1) and reads it back (C2).
    monkeypatch.setenv("MEMORYLAYER_GRAPH_ENTITY_MATERIALIZE_ENABLED", "true")
    age_svc = AgeGraphQueryService(storage=storage, v=v, age_engine=storage._engine)

    res = await age_svc.entity_neighborhood(ws, r["alice"])
    norm = _norm_neighborhood(res)
    assert norm["memories"] == sorted([(r["m1"], "self"), (r["m2"], "mention")])
    # bob shares m2, proj shares m1 — each count 1.
    assert norm["co"] == sorted([(r["bob"], 1), (r["proj"], 1)])
    assert norm["truncated"] is False


@pytest.mark.asyncio
async def test_c2_age_matches_oss_relational(age_storage, monkeypatch):
    storage, created = age_storage
    ws = "ent_mat_c2par_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    r = await _build_registry(storage, ws)

    v = Variables()
    monkeypatch.setenv("MEMORYLAYER_GRAPH_ENTITY_MATERIALIZE_ENABLED", "true")
    age_svc = AgeGraphQueryService(storage=storage, v=v, age_engine=storage._engine)
    rel_svc = RelationalGraphQueryService(storage=storage, v=v)

    golden = _norm_neighborhood(await rel_svc.entity_neighborhood(ws, r["alice"]))
    cand = _norm_neighborhood(await age_svc.entity_neighborhood(ws, r["alice"]))
    assert cand == golden, "entity_neighborhood diverged: rel=%s age=%s" % (golden, cand)


@pytest.mark.asyncio
async def test_c2_age_empty_registry_is_empty(age_storage, monkeypatch):
    storage, created = age_storage
    ws = "ent_mat_c2empty_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _ensure_ws(storage, ws)

    v = Variables()
    monkeypatch.setenv("MEMORYLAYER_GRAPH_ENTITY_MATERIALIZE_ENABLED", "true")
    age_svc = AgeGraphQueryService(storage=storage, v=v, age_engine=storage._engine)
    res = await age_svc.entity_neighborhood(ws, "ent_nope")
    assert res.memories_for_entity == []
    assert res.entities_co_mentioned == []


@pytest.mark.asyncio
async def test_c2_age_flag_off_is_empty(age_storage, monkeypatch):
    """Flag OFF: no Entity layer is materialized, so the neighborhood is empty."""
    storage, created = age_storage
    ws = "ent_mat_c2off_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    r = await _build_registry(storage, ws)

    v = Variables()
    monkeypatch.setenv("MEMORYLAYER_GRAPH_ENTITY_MATERIALIZE_ENABLED", "false")
    age_svc = AgeGraphQueryService(storage=storage, v=v, age_engine=storage._engine)
    res = await age_svc.entity_neighborhood(ws, r["alice"])
    assert res.memories_for_entity == []
    assert res.entities_co_mentioned == []


# ---------------------------------------------------------------------------
# MAJOR-1: multi-role parity — one memory mentioned under two roles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_c2_multi_role_parity_age_matches_oss(age_storage, monkeypatch):
    """MAJOR-1 regression: when a memory is mentioned under two roles by the
    same entity, both AGE and OSS must agree on which role wins (the
    lexicographically-first role, matching the ORDER BY m.id, r.role sort).
    """
    storage, created = age_storage
    ws = "ent_mat_multirole_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _ensure_ws(storage, ws)

    mem = await _add_mem(storage, ws, "multi-role memory")
    alice = await storage.store_entity(
        {"workspace_id": ws, "entity_type": "person",
         "canonical_name": "Alice", "normalized_name": "alice"}
    )
    # Add both roles for the same (entity, memory) pair. The unique constraint
    # is on (entity_id, memory_id, role), so both rows are valid.
    await storage.add_entity_member(ws, alice["id"], mem, role="mention")
    await storage.add_entity_member(ws, alice["id"], mem, role="self")

    v = Variables()
    monkeypatch.setenv("MEMORYLAYER_GRAPH_ENTITY_MATERIALIZE_ENABLED", "true")
    age_svc = AgeGraphQueryService(storage=storage, v=v, age_engine=storage._engine)
    rel_svc = RelationalGraphQueryService(storage=storage, v=v)

    golden = _norm_neighborhood(await rel_svc.entity_neighborhood(ws, alice["id"]))
    cand = _norm_neighborhood(await age_svc.entity_neighborhood(ws, alice["id"]))

    # Both must agree: one memory, one winning role (lexicographically first
    # = "mention" < "self"), and matching co-mention set (empty here).
    assert cand == golden, "multi-role parity diverged: rel=%s age=%s" % (golden, cand)
    # Sanity: only one entry in memories (dedup is correct).
    assert len(cand["memories"]) == 1
    assert cand["memories"][0][1] == "mention"  # "mention" < "self"


# ---------------------------------------------------------------------------
# MAJOR-2: watermark detects count-preserving role swap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_c1_watermark_detects_role_swap(age_storage):
    """MAJOR-2: a role mutation that preserves member count must still trigger
    a re-materialize because the member digest changes.
    """
    storage, created = age_storage
    ws = "ent_mat_roleswap_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _ensure_ws(storage, ws)

    mem = await _add_mem(storage, ws, "swap memory")
    alice = await storage.store_entity(
        {"workspace_id": ws, "entity_type": "person",
         "canonical_name": "Alice", "normalized_name": "alice"}
    )
    await storage.add_entity_member(ws, alice["id"], mem, role="mention")

    # First materialize.
    assert await _materialize_full(storage, ws) is True
    # Watermark skip on unchanged registry.
    assert await _materialize_full(storage, ws) is False

    # Simulate a count-preserving role swap: remove the "mention" row and add
    # "self" instead — same count, different digest.
    engine = storage._engine
    async with engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await raw.execute(
            "DELETE FROM entity_members WHERE workspace_id=$1 AND entity_id=$2 AND memory_id=$3 AND role=$4",
            ws, alice["id"], mem, "mention",
        )
        await raw.execute(
            "INSERT INTO entity_members (id, workspace_id, entity_id, memory_id, role, confidence, meta, created_at)"
            " VALUES (gen_random_uuid()::text, $1, $2, $3, 'self', 1.0, '{}', now())",
            ws, alice["id"], mem,
        )

    # Watermark must detect the change and re-materialize.
    assert await _materialize_full(storage, ws) is True


# ---------------------------------------------------------------------------
# Injection conformance
# ---------------------------------------------------------------------------


class TestEntityNeighborhoodInjection:
    _EVIL = [
        "$$) AS (x agtype); DROP TABLE entities; --",
        "ent$evil",
        "ent'quote",
        'ent"dquote',
        "ent;semicolon",
        "$$",
        "$",
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("evil", _EVIL)
    async def test_rejects_evil_entity_id(self, age_storage, evil):
        storage, _ = age_storage
        age_svc = AgeGraphQueryService(storage=storage, v=Variables(), age_engine=storage._engine)
        with pytest.raises(ValueError):
            await age_svc.entity_neighborhood("safe_ws", evil)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("evil", _EVIL)
    async def test_rejects_evil_workspace_id(self, age_storage, evil):
        storage, _ = age_storage
        age_svc = AgeGraphQueryService(storage=storage, v=Variables(), age_engine=storage._engine)
        with pytest.raises(ValueError):
            await age_svc.entity_neighborhood(evil, "safe_entity")
