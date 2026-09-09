# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Integration tests for P2 strategy-C watermark-gated AGE materialize (Gate B).

The AGE materialize-on-read normally does a full scope-DELETE-then-MERGE on EVERY call.
Gate B skips that work when the workspace's change-watermark is unchanged since the last
successful materialize (the materialized subgraph persists in AGE across calls). It falls
back to a full delete-then-MERGE when the watermark advanced or is unavailable (fail-safe).

These tests assert, against the eval Postgres-with-AGE database:
  1. Two analyze() runs on an unchanged workspace materialize ONCE (the 2nd is skipped),
     and the analysis is still correct from the persisted graph.
  2. Mutating the workspace re-materializes.
  3. The P2.1 staleness invariant still holds: deleting an association re-materializes and
     the next analysis shows NO phantom edge (this is the regression Gate B must not break,
     and the association-delete is the case timestamps alone cannot detect).

Requires ``ML_AGE_TEST_DATABASE_URL``. SKIPPED otherwise. Isolation: a unique workspace id
per test (prefix ``age_wm_<uuid>``); teardown deletes ONLY its own relational rows, its own
AGE subgraph, and its own materialization-watermark marker row.
"""

from __future__ import annotations

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
from memorylayer_server.services.graph_analysis.default import NetworkXGraphAnalysisService
from memorylayer_saas.services.graph_analysis import _materialize
from memorylayer_saas.services.graph_analysis._cypher import (
    delete_workspace_edges_sql,
    delete_workspace_vertices_sql,
)
from memorylayer_saas.services.graph_analysis.age import AgeGraphAnalysisService
from memorylayer_saas.services.graph_analysis.bootstrap import AGE_SEARCH_PATH, ensure_age_session

_AGE_DB_URL = os.getenv("ML_AGE_TEST_DATABASE_URL", "").strip()

pytestmark = pytest.mark.skipif(
    not _AGE_DB_URL,
    reason="Set ML_AGE_TEST_DATABASE_URL to a PostgreSQL-with-AGE database to run AGE watermark tests.",
)


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

    created_workspaces: list[str] = []
    try:
        yield backend, created_workspaces, engine
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
        import json

        for ws in workspace_ids:
            ws_param = json.dumps({"ws": ws})
            await raw.execute(delete_workspace_edges_sql(), ws_param)
            await raw.execute(delete_workspace_vertices_sql(), ws_param)
            await raw.execute("DELETE FROM memory_associations WHERE workspace_id = $1", ws)
            await raw.execute("DELETE FROM memories WHERE workspace_id = $1", ws)
            await raw.execute("DELETE FROM workspaces WHERE id = $1", ws)
            # Clean up this test's materialization-watermark marker row.
            try:
                await raw.execute(
                    "DELETE FROM age_materialization_watermark WHERE workspace_id = $1", ws
                )
            except Exception:
                pass  # table may not exist if no materialize ever ran


async def _ensure_workspace(storage, ws_id: str) -> None:
    existing = await storage.get_workspace(ws_id)
    if not existing:
        await storage.create_workspace(
            Workspace(
                id=ws_id,
                tenant_id="default_tenant",
                name="AGE Watermark Test %s" % ws_id,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )


async def _add_memory(storage, ws_id: str, content: str) -> str:
    mem = await storage.create_memory(ws_id, RememberInput(content=content))
    return mem.id


async def _add_assoc(storage, ws_id, src, tgt, relationship="related_to", strength=0.7):
    await storage.create_association(
        ws_id,
        AssociateInput(source_id=src, target_id=tgt, relationship=relationship, strength=strength),
    )


def _spy_materialize(monkeypatch) -> dict:
    """Wrap the SHARED materialize helper to count delete-then-MERGE executions.

    Both AGE backends call ``materialize_workspace_subgraph`` (the action) only when
    the gated wrapper decides to materialize; counting it counts the real work.
    """
    counter = {"calls": 0}
    real = _materialize.materialize_workspace_subgraph

    async def counting(*args, **kwargs):
        counter["calls"] += 1
        return await real(*args, **kwargs)

    monkeypatch.setattr(_materialize, "materialize_workspace_subgraph", counting)
    return counter


@pytest.mark.asyncio
async def test_unchanged_workspace_skips_materialize(age_storage, monkeypatch):
    """Two analyze() runs on an unchanged workspace materialize ONCE; the analysis is
    still correct on the skipped second run (served from the persisted AGE subgraph)."""
    storage, created, engine = age_storage
    ws = f"age_wm_{uuid.uuid4().hex[:8]}"
    created.append(ws)
    await _ensure_workspace(storage, ws)
    a = await _add_memory(storage, ws, "alice node")
    b = await _add_memory(storage, ws, "bob node")
    await _add_assoc(storage, ws, a, b)

    counter = _spy_materialize(monkeypatch)
    v = Variables()
    age_svc = AgeGraphAnalysisService(storage=storage, v=v, age_engine=engine)

    out1 = await age_svc.analyze(ws)
    assert counter["calls"] == 1, "first analyze must materialize"
    assert out1.stats.node_count == 2
    assert out1.stats.edge_count == 1

    out2 = await age_svc.analyze(ws)
    assert counter["calls"] == 1, "unchanged workspace must SKIP the delete-then-MERGE"
    # Analysis is still correct from the persisted graph.
    assert out2.stats.node_count == 2
    assert out2.stats.edge_count == 1


@pytest.mark.asyncio
async def test_mutation_rematerializes(age_storage, monkeypatch):
    """Adding a memory + association advances the watermark -> the next analyze re-materializes."""
    storage, created, engine = age_storage
    ws = f"age_wm_{uuid.uuid4().hex[:8]}"
    created.append(ws)
    await _ensure_workspace(storage, ws)
    a = await _add_memory(storage, ws, "alice node")
    b = await _add_memory(storage, ws, "bob node")
    await _add_assoc(storage, ws, a, b)

    counter = _spy_materialize(monkeypatch)
    v = Variables()
    age_svc = AgeGraphAnalysisService(storage=storage, v=v, age_engine=engine)

    await age_svc.analyze(ws)
    assert counter["calls"] == 1
    await age_svc.analyze(ws)
    assert counter["calls"] == 1, "idle re-analyze is skipped"

    # Mutate: a new node + a new edge.
    c = await _add_memory(storage, ws, "carol node")
    await _add_assoc(storage, ws, b, c)

    out = await age_svc.analyze(ws)
    assert counter["calls"] == 2, "mutated workspace must re-materialize"
    assert out.stats.node_count == 3
    assert out.stats.edge_count == 2


@pytest.mark.asyncio
async def test_association_delete_no_phantom_edge(age_storage, monkeypatch):
    """P2.1 STALENESS INVARIANT preserved: deleting an association re-materializes (the
    association_count drop is detected -- timestamps alone could not) and the next analysis
    shows NO phantom edge. This is the exact staleness the P2.1 review fixed; Gate B must
    not reintroduce it by skipping when an edge was removed."""
    storage, created, engine = age_storage
    ws = f"age_wm_{uuid.uuid4().hex[:8]}"
    created.append(ws)
    await _ensure_workspace(storage, ws)
    a = await _add_memory(storage, ws, "alice node")
    b = await _add_memory(storage, ws, "bob node")
    await _add_assoc(storage, ws, a, b)

    counter = _spy_materialize(monkeypatch)
    v = Variables()
    age_svc = AgeGraphAnalysisService(storage=storage, v=v, age_engine=engine)

    out1 = await age_svc.analyze(ws)
    assert counter["calls"] == 1
    assert out1.stats.edge_count == 1

    # Delete the single association via the storage backend (no timestamp advances;
    # only association_count drops 1 -> 0).
    assocs = await storage.get_associations(ws, a, direction="both")
    assert len(assocs) == 1
    deleted = await storage.delete_association(ws, assocs[0].id)
    assert deleted

    out2 = await age_svc.analyze(ws)
    assert counter["calls"] == 2, "association delete must force a re-materialize (staleness guard)"
    assert out2.stats.edge_count == 0, "no phantom edge may survive the association delete"


@pytest.mark.asyncio
async def test_watermark_unavailable_always_materializes(age_storage, monkeypatch):
    """If the watermark cannot be computed (returns None), every call materializes -- fail-safe."""
    storage, created, engine = age_storage
    ws = f"age_wm_{uuid.uuid4().hex[:8]}"
    created.append(ws)
    await _ensure_workspace(storage, ws)
    a = await _add_memory(storage, ws, "alice node")
    b = await _add_memory(storage, ws, "bob node")
    await _add_assoc(storage, ws, a, b)

    counter = _spy_materialize(monkeypatch)

    async def _no_watermark(workspace_id):
        return None

    monkeypatch.setattr(storage, "get_workspace_change_watermark", _no_watermark)

    v = Variables()
    age_svc = AgeGraphAnalysisService(storage=storage, v=v, age_engine=engine)

    await age_svc.analyze(ws)
    await age_svc.analyze(ws)
    assert counter["calls"] == 2, "watermark-unavailable must materialize on every call"


@pytest.mark.asyncio
async def test_subgraph_dropped_forces_rematerialize(age_storage, monkeypatch):
    """MINOR regression guard: if the AGE subgraph is dropped out-of-band while the
    materialization-watermark marker row survives, the existence probe detects the
    empty subgraph and falls through to a full re-materialize (defense-in-depth).

    Mechanism: after a successful materialize, surgically delete the AGE vertices/edges
    for this workspace WITHOUT clearing the marker table row. The watermark is unchanged
    and the marker row still holds the prior watermark, so a naive equality-only check
    would skip the MERGE and return an empty analysis. The probe (LIMIT-1 cypher query)
    detects absence and forces re-materialize -> analysis is correct.
    """
    import json as _json

    storage, created, engine = age_storage
    ws = f"age_wm_{uuid.uuid4().hex[:8]}"
    created.append(ws)
    await _ensure_workspace(storage, ws)
    a = await _add_memory(storage, ws, "alice node")
    b = await _add_memory(storage, ws, "bob node")
    await _add_assoc(storage, ws, a, b)

    counter = _spy_materialize(monkeypatch)
    v = Variables()
    age_svc = AgeGraphAnalysisService(storage=storage, v=v, age_engine=engine)

    # First analyze: materializes and records the marker row.
    out1 = await age_svc.analyze(ws)
    assert counter["calls"] == 1
    assert out1.stats.node_count == 2

    # Second analyze: watermark match + subgraph present -> skipped.
    out2 = await age_svc.analyze(ws)
    assert counter["calls"] == 1, "idle re-analyze must be skipped"
    assert out2.stats.node_count == 2

    # Drop the AGE subgraph out-of-band WITHOUT touching the marker table.
    ws_param = _json.dumps({"ws": ws})
    async with engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)
        await raw.execute(delete_workspace_edges_sql(), ws_param)
        await raw.execute(delete_workspace_vertices_sql(), ws_param)

    # Third analyze: watermark unchanged, marker row still present -- but the probe
    # detects the empty subgraph -> falls through to full re-materialize.
    out3 = await age_svc.analyze(ws)
    assert counter["calls"] == 2, (
        "dropped subgraph must be detected by the existence probe and force re-materialize"
    )
    assert out3.stats.node_count == 2, "re-materialized graph must be correct"
    assert out3.stats.edge_count == 1
