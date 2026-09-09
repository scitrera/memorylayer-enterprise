"""Graph-moat P4.5 integration tests: Fragment materialization + fragments_for_memory.

P4.5 (Fragment materialization into AGE):
  * materialize a workspace with decomposed-fact memories -> assert Fragment
    vertices + DERIVED_FROM edges exist (cypher count);
  * idempotent re-materialize (delete-then-MERGE -> same counts);
  * watermark skip on an unchanged fact set (no re-materialize) and a forced
    re-materialize when the fragment watermark advances on a COUNT-PRESERVING
    source_id repoint (mirrors the C1 role-swap regression).

fragments_for_memory (the consumer that makes the Fragment layer non-speculative):
  * the enterprise AGE traversal (reading the P4.5 Fragment/DERIVED_FROM
    vertices+edges) returns the SAME normalized set as the OSS relational
    fallback on the same data;
  * an empty fact set / flag-off -> empty result, no error.

Injection conformance: a malicious workspace_id / memory_id is rejected with
``ValueError`` BEFORE reaching cypher (mirrors the entity-materialize injection
conformance test).

DARK + MEASURABLE: fragments_for_memory is NOT wired into recall. This is a
gated, measurable capability whose eval (does fragment-traversal beat the fact
channel E on LoCoMo) is a later measurement, not part of this slice.

Requires a PostgreSQL-with-AGE database. SKIPPED unless ``ML_AGE_TEST_DATABASE_URL``
is set, e.g.::

    ML_AGE_TEST_DATABASE_URL=postgresql://memorylayer:memorylayer_dev@localhost:55432/memorylayer \\
        pytest tests/integration/test_age_fragment_materialize.py -v

Isolation + cleanup: every fixture runs under a unique workspace id
(prefix ``frag_mat_<uuid>``). Teardown deletes ONLY its own relational rows
(memories) and its own AGE subgraph (Fragment + Memory layers) — never
truncates shared tables.
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

from memorylayer_server.models.memory import MemorySubtype, MemoryType, RememberInput
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.graph_query.default import RelationalGraphQueryService
from memorylayer_saas.services.graph_analysis import _cypher
from memorylayer_saas.services.graph_analysis._cypher import (
    delete_workspace_derived_from_sql,
    delete_workspace_edges_sql,
    delete_workspace_fragments_sql,
    delete_workspace_vertices_sql,
)
from memorylayer_saas.services.graph_analysis._materialize import (
    compute_fragment_watermark,
    materialize_workspace_fragments,
    materialize_workspace_fragments_gated,
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
    reason="Set ML_AGE_TEST_DATABASE_URL to a PostgreSQL-with-AGE database to run AGE fragment-materialize tests.",
)


# ---------------------------------------------------------------------------
# Storage backend fixture (cloned from test_age_entity_materialize.py)
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
            # Fragment layer first, then Memory layer.
            await raw.execute(delete_workspace_derived_from_sql(), ws_param)
            await raw.execute(delete_workspace_fragments_sql(), ws_param)
            await raw.execute(delete_workspace_edges_sql(), ws_param)
            await raw.execute(delete_workspace_vertices_sql(), ws_param)
            await raw.execute("DELETE FROM memory_associations WHERE workspace_id = $1", ws)
            await raw.execute("DELETE FROM memories WHERE workspace_id = $1", ws)
            await raw.execute("DELETE FROM age_fragment_materialization_watermark WHERE workspace_id = $1", ws)
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
                name="Frag-mat %s" % ws_id,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )


async def _add_mem(storage, ws_id: str, content: str) -> str:
    mem = await storage.create_memory(ws_id, RememberInput(content=content))
    return mem.id


async def _add_fact(storage, ws_id: str, content: str, source_id: str) -> str:
    """Store a decomposed-fact memory (subtype="fact", metadata source_id)."""
    mem = await storage.create_memory(
        ws_id,
        RememberInput(
            content=content,
            type=MemoryType.SEMANTIC,
            subtype=MemorySubtype.FACT.value,
            metadata={"kind": "fact", "source_id": source_id},
        ),
    )
    return mem.id


async def _build_facts(storage, ws_id: str) -> dict[str, str]:
    """src1 has facts f1,f2; src2 has fact f3. A non-fact memory exists too."""
    await _ensure_ws(storage, ws_id)
    src1 = await _add_mem(storage, ws_id, "source memory one")
    src2 = await _add_mem(storage, ws_id, "source memory two")
    f1 = await _add_fact(storage, ws_id, "fact one of src1", src1)
    f2 = await _add_fact(storage, ws_id, "fact two of src1", src1)
    f3 = await _add_fact(storage, ws_id, "fact one of src2", src2)
    return {"src1": src1, "src2": src2, "f1": f1, "f2": f2, "f3": f3}


async def _materialize_full(storage, ws_id: str) -> bool:
    """Materialize the Memory subgraph then the Fragment layer (what the query does)."""
    engine = storage._engine
    memories = await storage.search_memories_by_filter(ws_id, status="active", limit=10000)
    node_attrs = {
        m.id: {"memory_type": getattr(m, "memory_type", None), "memory_subtype": getattr(m, "subtype", None)}
        for m in memories
    }
    await materialize_workspace_subgraph_gated(
        engine, storage=storage, workspace_id=ws_id, node_attrs=node_attrs, associations=[]
    )
    facts = await storage.search_memories_by_filter(ws_id, subtypes=["fact"], status="active", limit=10000)
    fragments = [
        {
            "id": m.id,
            "content": getattr(m, "content", None),
            "source_id": (getattr(m, "metadata", None) or {}).get("source_id"),
            "updated_at": getattr(m, "updated_at", None),
        }
        for m in facts
    ]
    return await materialize_workspace_fragments_gated(
        engine, workspace_id=ws_id, fragments=fragments
    )


async def _count_fragment_vertices(storage, ws_id: str) -> int:
    engine = storage._engine
    sql = (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (f:Fragment {workspace_id: $ws}) RETURN count(f)"
        "$$, $1) AS (cnt agtype)" % _cypher.GRAPH_NAME
    )
    async with engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)
        rows = await raw.fetch(sql, json.dumps({"ws": ws_id}))
    return int(json.loads(rows[0]["cnt"]))


async def _count_derived_from_edges(storage, ws_id: str) -> int:
    engine = storage._engine
    sql = (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (f:Fragment {workspace_id: $ws})-[d:DERIVED_FROM]->(m:Memory {workspace_id: $ws})"
        " RETURN count(d)"
        "$$, $1) AS (cnt agtype)" % _cypher.GRAPH_NAME
    )
    async with engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)
        rows = await raw.fetch(sql, json.dumps({"ws": ws_id}))
    return int(json.loads(rows[0]["cnt"]))


# ---------------------------------------------------------------------------
# P4.5 — materialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p45_materialize_creates_fragment_vertices_and_edges(age_storage):
    storage, created = age_storage
    ws = "frag_mat_create_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _build_facts(storage, ws)

    ran = await _materialize_full(storage, ws)
    assert ran is True
    # 3 fact memories -> 3 Fragment vertices; 3 DERIVED_FROM edges (each fact has
    # a source_id that exists as a Memory vertex).
    assert await _count_fragment_vertices(storage, ws) == 3
    assert await _count_derived_from_edges(storage, ws) == 3


@pytest.mark.asyncio
async def test_p45_idempotent_re_materialize_same_counts(age_storage):
    storage, created = age_storage
    ws = "frag_mat_idem_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _build_facts(storage, ws)

    await _materialize_full(storage, ws)
    v1 = await _count_fragment_vertices(storage, ws)
    e1 = await _count_derived_from_edges(storage, ws)

    # Force a full re-materialize (bypass the gate) to prove delete-then-MERGE is
    # idempotent: counts must not double.
    facts = await storage.search_memories_by_filter(ws, subtypes=["fact"], status="active", limit=10000)
    fragments = [
        {"id": m.id, "content": getattr(m, "content", None),
         "source_id": (getattr(m, "metadata", None) or {}).get("source_id"),
         "updated_at": getattr(m, "updated_at", None)}
        for m in facts
    ]
    await materialize_workspace_fragments(storage._engine, workspace_id=ws, fragments=fragments)
    assert await _count_fragment_vertices(storage, ws) == v1 == 3
    assert await _count_derived_from_edges(storage, ws) == e1 == 3


@pytest.mark.asyncio
async def test_p45_watermark_skips_unchanged_facts(age_storage):
    storage, created = age_storage
    ws = "frag_mat_wm_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _build_facts(storage, ws)

    assert await _materialize_full(storage, ws) is True
    # Second materialize on an UNCHANGED fact set is skipped by the watermark.
    assert await _materialize_full(storage, ws) is False
    assert await _count_fragment_vertices(storage, ws) == 3

    # Adding a fact advances the fragment watermark -> re-materialize runs.
    src3 = await _add_mem(storage, ws, "source memory three")
    await _add_fact(storage, ws, "fact one of src3", src3)
    assert await _materialize_full(storage, ws) is True
    assert await _count_fragment_vertices(storage, ws) == 4


@pytest.mark.asyncio
async def test_p45_empty_facts_is_clean_noop(age_storage):
    storage, created = age_storage
    ws = "frag_mat_empty_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _ensure_ws(storage, ws)
    # A non-fact memory exists but no fact memories.
    await _add_mem(storage, ws, "just a regular memory")

    await _materialize_full(storage, ws)
    assert await _count_fragment_vertices(storage, ws) == 0
    assert await _count_derived_from_edges(storage, ws) == 0


def test_p45_fragment_watermark_token_shape():
    # Pure-unit check (no DB): token is (fragment_count, max_updated, digest).
    wm = compute_fragment_watermark(
        [{"id": "f1", "source_id": "m1", "updated_at": "2026-01-02T00:00:00"},
         {"id": "f2", "source_id": "m2", "updated_at": "2026-01-03T00:00:00"}],
    )
    assert len(wm) == 3
    assert wm[0] == 2
    assert wm[1] == "2026-01-03T00:00:00"
    assert isinstance(wm[2], str) and len(wm[2]) == 16
    # Empty fact set token.
    empty = compute_fragment_watermark([])
    assert empty[0] == 0 and empty[1] == ""


@pytest.mark.asyncio
async def test_p45_watermark_detects_source_repoint(age_storage):
    """A count-preserving source_id repoint must still trigger a re-materialize
    because the fragment digest changes (mirrors the C1 role-swap regression).
    """
    storage, created = age_storage
    ws = "frag_mat_repoint_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _ensure_ws(storage, ws)
    src1 = await _add_mem(storage, ws, "source one")
    src2 = await _add_mem(storage, ws, "source two")
    f1 = await _add_fact(storage, ws, "a fact", src1)

    assert await _materialize_full(storage, ws) is True
    assert await _materialize_full(storage, ws) is False

    # Repoint the fact's source_id from src1 -> src2 (count preserved).
    engine = storage._engine
    fact_mem = await storage.get_memory(ws, f1, track_access=False)
    new_meta = {**(fact_mem.metadata or {}), "source_id": src2}
    await storage.update_memory(ws, f1, metadata=new_meta)

    # The digest must detect the repoint and re-materialize.
    assert await _materialize_full(storage, ws) is True
    # The single DERIVED_FROM edge now points to src2.
    assert await _count_derived_from_edges(storage, ws) == 1


# ---------------------------------------------------------------------------
# fragments_for_memory — AGE reads P4.5's output; parity vs OSS relational
# ---------------------------------------------------------------------------


def _norm_fragments(res) -> dict:
    return {
        "source_id": res.source_id,
        "fragments": sorted((f.fragment_id, f.source_id) for f in res.fragments),
        "truncated": res.truncated,
    }


@pytest.mark.asyncio
async def test_traversal_age_reads_p45_output(age_storage, monkeypatch):
    storage, created = age_storage
    ws = "frag_mat_q_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    r = await _build_facts(storage, ws)

    v = Variables()
    monkeypatch.setenv("MEMORYLAYER_GRAPH_FRAGMENT_MATERIALIZE_ENABLED", "true")
    age_svc = AgeGraphQueryService(storage=storage, v=v, age_engine=storage._engine)

    res = await age_svc.fragments_for_memory(ws, r["src1"])
    norm = _norm_fragments(res)
    # src1 has f1, f2 derived from it.
    assert norm["fragments"] == sorted([(r["f1"], r["src1"]), (r["f2"], r["src1"])])
    assert norm["truncated"] is False


@pytest.mark.asyncio
async def test_traversal_age_matches_oss_relational(age_storage, monkeypatch):
    storage, created = age_storage
    ws = "frag_mat_qpar_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    r = await _build_facts(storage, ws)

    v = Variables()
    monkeypatch.setenv("MEMORYLAYER_GRAPH_FRAGMENT_MATERIALIZE_ENABLED", "true")
    age_svc = AgeGraphQueryService(storage=storage, v=v, age_engine=storage._engine)
    rel_svc = RelationalGraphQueryService(storage=storage, v=v)

    for src_key in ("src1", "src2"):
        golden = _norm_fragments(await rel_svc.fragments_for_memory(ws, r[src_key]))
        cand = _norm_fragments(await age_svc.fragments_for_memory(ws, r[src_key]))
        assert cand == golden, "fragments_for_memory diverged for %s: rel=%s age=%s" % (src_key, golden, cand)


@pytest.mark.asyncio
async def test_traversal_age_empty_facts_is_empty(age_storage, monkeypatch):
    storage, created = age_storage
    ws = "frag_mat_qempty_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _ensure_ws(storage, ws)
    src = await _add_mem(storage, ws, "no facts here")

    v = Variables()
    monkeypatch.setenv("MEMORYLAYER_GRAPH_FRAGMENT_MATERIALIZE_ENABLED", "true")
    age_svc = AgeGraphQueryService(storage=storage, v=v, age_engine=storage._engine)
    res = await age_svc.fragments_for_memory(ws, src)
    assert res.fragments == []
    assert res.truncated is False


@pytest.mark.asyncio
async def test_traversal_age_flag_off_is_empty(age_storage, monkeypatch):
    """Flag OFF: no Fragment layer is materialized, so the traversal is empty."""
    storage, created = age_storage
    ws = "frag_mat_qoff_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    r = await _build_facts(storage, ws)

    v = Variables()
    monkeypatch.setenv("MEMORYLAYER_GRAPH_FRAGMENT_MATERIALIZE_ENABLED", "false")
    age_svc = AgeGraphQueryService(storage=storage, v=v, age_engine=storage._engine)
    res = await age_svc.fragments_for_memory(ws, r["src1"])
    assert res.fragments == []


# ---------------------------------------------------------------------------
# Injection conformance
# ---------------------------------------------------------------------------


class TestFragmentsForMemoryInjection:
    _EVIL = [
        "$$) AS (x agtype); DROP TABLE memories; --",
        "frag$evil",
        "frag'quote",
        'frag"dquote',
        "frag;semicolon",
        "$$",
        "$",
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("evil", _EVIL)
    async def test_rejects_evil_memory_id(self, age_storage, evil):
        storage, _ = age_storage
        age_svc = AgeGraphQueryService(storage=storage, v=Variables(), age_engine=storage._engine)
        with pytest.raises(ValueError):
            await age_svc.fragments_for_memory("safe_ws", evil)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("evil", _EVIL)
    async def test_rejects_evil_workspace_id(self, age_storage, evil):
        storage, _ = age_storage
        age_svc = AgeGraphQueryService(storage=storage, v=Variables(), age_engine=storage._engine)
        with pytest.raises(ValueError):
            await age_svc.fragments_for_memory(evil, "safe_memory")
