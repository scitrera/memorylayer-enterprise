# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Conformance + security tests: AGE graph-QUERY backend vs relational golden (P2 Track A).

Proves that ``AgeGraphQueryService`` returns results that — after UUID-stripping
normalization — are IDENTICAL to the OSS ``RelationalGraphQueryService`` (the
golden) run on the SAME relational data, for EVERY method:

    neighbors (depth 1/2, direction, rel-filter), k_hop_subgraph,
    shortest_path, typed_pattern, relationship_rollup.

Normalization (``_norm_*`` helpers) strips DB-assigned UUIDs and compares only
graph-invariant quantities: node counts, edge ``(relationship, strength)``
multisets, ``truncated`` flags, path hop counts, and rollup counts.

Also tests the security invariant: injection payloads in memory_id /
relationship (``$$``, quotes, ``;``) are rejected with ``ValueError`` before
reaching cypher.

Requires a PostgreSQL-with-AGE database. SKIPPED unless ``ML_AGE_TEST_DATABASE_URL``
is set, e.g.::

    ML_AGE_TEST_DATABASE_URL=postgresql://memorylayer:memorylayer_dev@localhost:55432/memorylayer \\
        pytest tests/integration/test_age_graph_query_conformance.py -v

Isolation + cleanup: every fixture runs under a unique workspace id
(prefix ``gq_conf_<uuid>``). Teardown deletes ONLY its own relational rows and
its own AGE subgraph data — NEVER truncates shared tables.
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

from memorylayer_server.models.association import AssociateInput
from memorylayer_server.models.memory import RememberInput
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.graph_query.default import RelationalGraphQueryService
from memorylayer_saas.services.graph_analysis._cypher import (
    delete_workspace_edges_sql,
    delete_workspace_vertices_sql,
)
from memorylayer_saas.services.graph_analysis.bootstrap import AGE_SEARCH_PATH, ensure_age_session
from memorylayer_saas.services.graph_query.age import AgeGraphQueryService

# ---------------------------------------------------------------------------
# Skip gate
# ---------------------------------------------------------------------------

_AGE_DB_URL = os.getenv("ML_AGE_TEST_DATABASE_URL", "").strip()

pytestmark = pytest.mark.skipif(
    not _AGE_DB_URL,
    reason="Set ML_AGE_TEST_DATABASE_URL to a PostgreSQL-with-AGE database to run AGE graph-query conformance.",
)


# ---------------------------------------------------------------------------
# Storage backend fixture (cloned from test_age_graph_conformance.py)
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
    """Live PostgreSQLBackend wired to the AGE test database (function-scoped)."""
    from memorylayer_saas.storage.postgresql import PostgreSQLBackend

    v = Variables()
    backend = PostgreSQLBackend(v=v, connection_string=_asyncpg_url(_AGE_DB_URL))
    engine = _make_age_engine(_AGE_DB_URL)
    backend._engine = engine
    backend._session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

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
            await raw.execute(delete_workspace_edges_sql(), ws_param)
            await raw.execute(delete_workspace_vertices_sql(), ws_param)
            await raw.execute("DELETE FROM memory_associations WHERE workspace_id = $1", ws)
            await raw.execute("DELETE FROM memories WHERE workspace_id = $1", ws)
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
                name="GQ conformance %s" % ws_id,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )


async def _add_mem(storage, ws_id: str, content: str) -> str:
    mem = await storage.create_memory(ws_id, RememberInput(content=content))
    return mem.id


async def _add_assoc(storage, ws_id, src, tgt, rel="related_to", strength=0.7):
    await storage.create_association(
        ws_id, AssociateInput(source_id=src, target_id=tgt, relationship=rel, strength=strength)
    )


async def _build_chain(storage, ws_id: str) -> list[str]:
    """m0 -related_to-> m1 -leads_to-> m2 -leads_to-> m3, plus m1 -similar_to-> m4."""
    await _ensure_ws(storage, ws_id)
    ids = [await _add_mem(storage, ws_id, "gq node %d" % i) for i in range(5)]
    await _add_assoc(storage, ws_id, ids[0], ids[1], "related_to", 0.9)
    await _add_assoc(storage, ws_id, ids[1], ids[2], "leads_to", 0.8)
    await _add_assoc(storage, ws_id, ids[2], ids[3], "leads_to", 0.7)
    await _add_assoc(storage, ws_id, ids[1], ids[4], "similar_to", 0.5)
    return ids


async def _build_two_cluster(storage, ws_id: str) -> list[str]:
    """Two triangles bridged by a single solves edge — parallel + cross edges."""
    await _ensure_ws(storage, ws_id)
    ids = [await _add_mem(storage, ws_id, "gq cl node %d" % i) for i in range(6)]
    # Cluster A triangle: m0-m1-m2
    await _add_assoc(storage, ws_id, ids[0], ids[1], "similar_to", 0.8)
    await _add_assoc(storage, ws_id, ids[1], ids[2], "similar_to", 0.8)
    await _add_assoc(storage, ws_id, ids[0], ids[2], "causes", 0.6)
    # Cluster B triangle: m3-m4-m5
    await _add_assoc(storage, ws_id, ids[3], ids[4], "related_to", 0.7)
    await _add_assoc(storage, ws_id, ids[4], ids[5], "related_to", 0.7)
    await _add_assoc(storage, ws_id, ids[3], ids[5], "related_to", 0.7)
    # Bridge
    await _add_assoc(storage, ws_id, ids[2], ids[3], "solves", 0.4)
    return ids


def _make_services(storage):
    v = Variables()
    rel_svc = RelationalGraphQueryService(storage=storage, v=v)
    age_svc = AgeGraphQueryService(storage=storage, v=v, age_engine=storage._engine)
    return rel_svc, age_svc


# ---------------------------------------------------------------------------
# Normalization (strip UUIDs)
# ---------------------------------------------------------------------------


def _edge_multiset(edges) -> list[tuple]:
    return sorted((e.relationship, round(e.strength, 6)) for e in edges)


def _norm_neighbor_or_subgraph(res) -> dict:
    return {
        "node_count": len(res.nodes),
        "edges": _edge_multiset(res.edges),
        "truncated": res.truncated,
    }


def _norm_path(res) -> dict:
    return {
        "found": res.found,
        "hops": res.hops,
        "node_count": len(res.nodes),
        "edges": _edge_multiset(res.edges),
    }


def _norm_pattern(matches) -> dict:
    return {
        "count": len(matches),
        "edges": sorted((m.relationship, round(m.strength, 6)) for m in matches),
    }


def _norm_rollup(rollups) -> dict:
    return {r.relationship: r.count for r in rollups}


# ---------------------------------------------------------------------------
# Per-method parity tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [1, 2])
async def test_neighbors_parity(age_storage, depth):
    storage, created = age_storage
    ws = "gq_conf_neigh_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    ids = await _build_chain(storage, ws)
    rel_svc, age_svc = _make_services(storage)

    golden = _norm_neighbor_or_subgraph(await rel_svc.neighbors(ws, ids[1], depth=depth))
    cand = _norm_neighbor_or_subgraph(await age_svc.neighbors(ws, ids[1], depth=depth))
    assert cand == golden, "neighbors(depth=%d) diverged: rel=%s age=%s" % (depth, golden, cand)


@pytest.mark.asyncio
async def test_neighbors_relationship_filter_parity(age_storage):
    storage, created = age_storage
    ws = "gq_conf_relf_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    ids = await _build_chain(storage, ws)
    rel_svc, age_svc = _make_services(storage)

    golden = _norm_neighbor_or_subgraph(
        await rel_svc.neighbors(ws, ids[1], depth=1, relationship_types=["similar_to"])
    )
    cand = _norm_neighbor_or_subgraph(
        await age_svc.neighbors(ws, ids[1], depth=1, relationship_types=["similar_to"])
    )
    assert cand == golden, "neighbors(rel-filter) diverged: rel=%s age=%s" % (golden, cand)


@pytest.mark.asyncio
async def test_neighbors_truncation_parity(age_storage):
    storage, created = age_storage
    ws = "gq_conf_trunc_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    ids = await _build_chain(storage, ws)
    rel_svc, age_svc = _make_services(storage)

    golden = _norm_neighbor_or_subgraph(await rel_svc.neighbors(ws, ids[1], depth=1, limit=2))
    cand = _norm_neighbor_or_subgraph(await age_svc.neighbors(ws, ids[1], depth=1, limit=2))
    assert cand["truncated"] is True and golden["truncated"] is True
    assert cand["node_count"] == golden["node_count"] == 2


@pytest.mark.asyncio
async def test_k_hop_parity(age_storage):
    storage, created = age_storage
    ws = "gq_conf_khop_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    ids = await _build_two_cluster(storage, ws)
    rel_svc, age_svc = _make_services(storage)

    golden = _norm_neighbor_or_subgraph(await rel_svc.k_hop_subgraph(ws, [ids[0], ids[5]], depth=2))
    cand = _norm_neighbor_or_subgraph(await age_svc.k_hop_subgraph(ws, [ids[0], ids[5]], depth=2))
    assert cand == golden, "k_hop diverged: rel=%s age=%s" % (golden, cand)


@pytest.mark.asyncio
async def test_shortest_path_parity(age_storage):
    storage, created = age_storage
    ws = "gq_conf_path_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    ids = await _build_chain(storage, ws)
    rel_svc, age_svc = _make_services(storage)

    golden = _norm_path(await rel_svc.shortest_path(ws, ids[0], ids[3], max_hops=5))
    cand = _norm_path(await age_svc.shortest_path(ws, ids[0], ids[3], max_hops=5))
    assert cand["found"] == golden["found"] is True
    assert cand["hops"] == golden["hops"] == 3
    assert cand["node_count"] == golden["node_count"] == 4
    assert cand["edges"] == golden["edges"]


@pytest.mark.asyncio
async def test_shortest_path_unreachable_parity(age_storage):
    storage, created = age_storage
    ws = "gq_conf_nopath_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _ensure_ws(storage, ws)
    a = await _add_mem(storage, ws, "iso a")
    b = await _add_mem(storage, ws, "iso b")
    rel_svc, age_svc = _make_services(storage)

    golden = _norm_path(await rel_svc.shortest_path(ws, a, b, max_hops=3))
    cand = _norm_path(await age_svc.shortest_path(ws, a, b, max_hops=3))
    assert cand == golden
    assert cand["found"] is False


@pytest.mark.asyncio
async def test_typed_pattern_parity(age_storage):
    storage, created = age_storage
    ws = "gq_conf_typed_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _build_chain(storage, ws)
    rel_svc, age_svc = _make_services(storage)

    golden = _norm_pattern(await rel_svc.typed_pattern(ws, relationship="leads_to"))
    cand = _norm_pattern(await age_svc.typed_pattern(ws, relationship="leads_to"))
    assert cand == golden, "typed_pattern diverged: rel=%s age=%s" % (golden, cand)
    assert golden["count"] == 2


@pytest.mark.asyncio
async def test_relationship_rollup_parity(age_storage):
    storage, created = age_storage
    ws = "gq_conf_rollup_%s" % uuid.uuid4().hex[:8]
    created.append(ws)
    await _build_two_cluster(storage, ws)
    rel_svc, age_svc = _make_services(storage)

    golden = _norm_rollup(await rel_svc.relationship_rollup(ws))
    cand = _norm_rollup(await age_svc.relationship_rollup(ws))
    assert cand == golden, "rollup diverged: rel=%s age=%s" % (golden, cand)
    # similar_to x2, causes x1, related_to x3, solves x1
    assert golden == {"similar_to": 2, "causes": 1, "related_to": 3, "solves": 1}


# ---------------------------------------------------------------------------
# Injection rejection
# ---------------------------------------------------------------------------


class TestInjectionRejection:
    """Injection payloads in memory_id / relationship must raise ValueError."""

    _EVIL = [
        "$$) AS (x agtype); DROP TABLE memories; --",
        "ws$evil",
        "ws'quote",
        'ws"dquote',
        "ws;semicolon",
        "$$",
        "$",
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("evil", _EVIL)
    async def test_neighbors_rejects_evil_memory_id(self, age_storage, evil):
        storage, _ = age_storage
        _, age_svc = _make_services(storage)
        with pytest.raises(ValueError):
            await age_svc.neighbors("safe_ws", evil)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("evil", _EVIL)
    async def test_neighbors_rejects_evil_workspace_id(self, age_storage, evil):
        storage, _ = age_storage
        _, age_svc = _make_services(storage)
        with pytest.raises(ValueError):
            await age_svc.neighbors(evil, "safe_mem")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("evil", ["rel$bad", "rel'bad", "rel;bad", "$$"])
    async def test_typed_pattern_rejects_evil_relationship(self, age_storage, evil):
        storage, _ = age_storage
        _, age_svc = _make_services(storage)
        with pytest.raises(ValueError):
            await age_svc.typed_pattern("safe_ws", relationship=evil)

    @pytest.mark.asyncio
    async def test_shortest_path_rejects_evil_ids(self, age_storage):
        storage, _ = age_storage
        _, age_svc = _make_services(storage)
        with pytest.raises(ValueError):
            await age_svc.shortest_path("safe_ws", "$$evil", "safe_dst")
        with pytest.raises(ValueError):
            await age_svc.shortest_path("safe_ws", "safe_src", "evil;drop")

    @pytest.mark.asyncio
    async def test_neighbors_rejects_out_of_range_depth(self, age_storage):
        storage, _ = age_storage
        _, age_svc = _make_services(storage)
        with pytest.raises(ValueError):
            await age_svc.neighbors("safe_ws", "safe_mem", depth=99)
