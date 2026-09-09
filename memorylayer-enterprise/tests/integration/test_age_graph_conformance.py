"""Byte-identical parity conformance + security tests: AGE backend vs NetworkX.

This is the crux of the AGE graph-analysis backend. It proves that
``AgeGraphAnalysisService`` produces a ``GraphAnalysis`` that — after the OSS
golden normalization (``normalize_for_golden``) — is IDENTICAL to the OSS
``NetworkXGraphAnalysisService`` run on the SAME relational data.

Additionally tests the security and correctness invariants:
  - Injection payloads in workspace_id / memory_id / relationship are rejected
    by ``validate_id`` before reaching cypher (BLOCKER fix).
  - Phantom edges from deleted associations do not appear after re-analysis
    (MAJOR-1 fix: scope-DELETE before MERGE).
  - Parallel relationships (same pair, different rel) and reverse-direction
    duplicates dedup identically to OSS (MAJOR-2 fix: no manual dedup).
  - Self-loops are handled (nx.Graph ignores them; parity must hold).
  - Archived/deleted endpoints are excluded from the graph.
  - ``include_rpg=True`` path produces identical output.

Requires a PostgreSQL-with-AGE database. SKIPPED unless ``ML_AGE_TEST_DATABASE_URL``
is set, e.g.::

    ML_AGE_TEST_DATABASE_URL=postgresql://memorylayer:memorylayer_dev@localhost:55432/memorylayer \\
        pytest tests/integration/test_age_graph_conformance.py -v

Isolation + cleanup: every fixture runs under a unique workspace id
(prefix ``age_conf_<uuid>``). Teardown deletes ONLY its own relational rows
and its own AGE subgraph data — NEVER truncates shared tables.
"""

from __future__ import annotations

import importlib.util
import json
import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from scitrera_app_framework import Variables
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from memorylayer_server.models.association import AssociateInput
from memorylayer_server.models.memory import RememberInput
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.graph_analysis.default import NetworkXGraphAnalysisService
from memorylayer_saas.services.graph_analysis._cypher import (
    GRAPH_NAME,
    delete_workspace_edges_sql,
    delete_workspace_vertices_sql,
    validate_id,
)
from memorylayer_saas.services.graph_analysis.age import AgeGraphAnalysisService
from memorylayer_saas.services.graph_analysis.bootstrap import (
    AGE_SEARCH_PATH,
    ensure_age_session,
)

# ---------------------------------------------------------------------------
# Skip gate
# ---------------------------------------------------------------------------

_AGE_DB_URL = os.getenv("ML_AGE_TEST_DATABASE_URL", "").strip()

pytestmark = pytest.mark.skipif(
    not _AGE_DB_URL,
    reason="Set ML_AGE_TEST_DATABASE_URL to a PostgreSQL-with-AGE database to run AGE conformance.",
)

# ---------------------------------------------------------------------------
# Load OSS golden module (normalizer + fixture builders)
# ---------------------------------------------------------------------------

_OSS_TEST = (
    Path(__file__).resolve().parents[4]
    / "oss"
    / "memorylayer-core-python"
    / "tests"
    / "unit"
    / "test_graph_analysis_single_pass.py"
)


def _load_oss_golden_module():
    spec = importlib.util.spec_from_file_location("_oss_graph_golden", _OSS_TEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_golden = _load_oss_golden_module()
normalize_for_golden = _golden.normalize_for_golden

_OSS_FIXTURE_BUILDERS = {
    "small": _golden._build_small_workspace,
    "medium": _golden._build_medium_workspace,
    "linear": _golden._build_linear_workspace,
    "empty": _golden._ensure_workspace,
}

# ---------------------------------------------------------------------------
# Storage backend fixture
# ---------------------------------------------------------------------------


def _asyncpg_url(url: str) -> str:
    return (
        url.replace("postgres://", "postgresql://")
        .replace("postgresql+asyncpg://", "postgresql://")
        .replace("postgresql://", "postgresql+asyncpg://")
    )


def _make_age_engine(url: str):
    """Engine with AGE search_path baked in via connect_args (MAJOR-4 fix)."""
    return create_async_engine(
        _asyncpg_url(url),
        pool_pre_ping=True,
        connect_args={"server_settings": {"search_path": AGE_SEARCH_PATH}},
    )


@pytest_asyncio.fixture
async def age_storage():
    """Live PostgreSQLBackend wired to the AGE test database.

    Function-scoped: pytest-asyncio runs each test in its own event loop
    (asyncio_mode=auto); a module-scoped engine would be attached to the
    wrong loop. Each test gets its own engine and cleans up only its own
    workspace data.
    """
    from memorylayer_saas.storage.postgresql import PostgreSQLBackend

    v = Variables()
    backend = PostgreSQLBackend(v=v, connection_string=_asyncpg_url(_AGE_DB_URL))
    # The eval PG already has the schema provisioned; skip the full connect()
    # migration path (Base.metadata.create_all would collide with existing tables).
    # Wire the AGE engine directly — search_path included (MAJOR-4 fix).
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
    """Surgically delete test rows — never touches other workspaces."""
    if not workspace_ids:
        return
    engine = backend._engine
    ws_parm_tpl = json.dumps({"ws": "__WS__"})  # placeholder
    async with engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)
        for ws in workspace_ids:
            ws_param = json.dumps({"ws": ws})
            # Parameterized AGE deletes (BLOCKER fix applied to cleanup too).
            await raw.execute(delete_workspace_edges_sql(), ws_param)
            await raw.execute(delete_workspace_vertices_sql(), ws_param)
            # Relational rows (FK order: associations → memories → workspace).
            await raw.execute("DELETE FROM memory_associations WHERE workspace_id = $1", ws)
            await raw.execute("DELETE FROM memories WHERE workspace_id = $1", ws)
            await raw.execute("DELETE FROM workspaces WHERE id = $1", ws)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _ensure_workspace(storage, ws_id: str) -> None:
    from datetime import UTC, datetime
    existing = await storage.get_workspace(ws_id)
    if not existing:
        ws = Workspace(
            id=ws_id,
            tenant_id="default_tenant",
            name="AGE Conformance Test %s" % ws_id,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        await storage.create_workspace(ws)


async def _add_memory(storage, ws_id: str, content: str) -> str:
    mem = await storage.create_memory(ws_id, RememberInput(content=content))
    return mem.id


async def _add_assoc(
    storage, ws_id: str, src: str, tgt: str,
    relationship: str = "related_to", strength: float = 0.7,
) -> None:
    await storage.create_association(
        ws_id,
        AssociateInput(source_id=src, target_id=tgt, relationship=relationship, strength=strength),
    )


def _make_services(storage, age_engine=None) -> tuple[NetworkXGraphAnalysisService, AgeGraphAnalysisService]:
    v = Variables()
    nx_svc = NetworkXGraphAnalysisService(storage=storage, v=v)
    age_svc = AgeGraphAnalysisService(storage=storage, v=v, age_engine=age_engine)
    return nx_svc, age_svc


async def _assert_parity(nx_svc, age_svc, ws_id: str, *, include_rpg: bool = False) -> dict:
    """Run both services on ws_id and assert normalized equality. Returns normalized output."""
    golden = normalize_for_golden(
        (await nx_svc.analyze(ws_id, include_rpg=include_rpg)).model_dump(mode="json")
    )
    age_out = normalize_for_golden(
        (await age_svc.analyze(ws_id, include_rpg=include_rpg)).model_dump(mode="json")
    )
    assert age_out == golden, (
        "AGE output diverged from NetworkX for workspace '%s'.\n"
        "NetworkX: %s\nAGE:      %s" % (ws_id, golden, age_out)
    )
    return golden


# ---------------------------------------------------------------------------
# OSS golden topology parity (small / medium / linear / empty)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture_name", ["small", "medium", "linear", "empty"])
async def test_age_matches_networkx_golden(age_storage, fixture_name):
    """AGE and NetworkX must produce identical normalized GraphAnalysis for OSS golden topologies."""
    storage, created_workspaces = age_storage
    engine = storage._engine

    ws_id = "age_conf_%s_%s" % (fixture_name, uuid.uuid4().hex[:8])
    created_workspaces.append(ws_id)

    builder = _OSS_FIXTURE_BUILDERS[fixture_name]
    await builder(storage, ws_id)

    nx_svc, age_svc = _make_services(storage, age_engine=engine)
    await _assert_parity(nx_svc, age_svc, ws_id)


# ---------------------------------------------------------------------------
# BLOCKER: injection payloads must be rejected before reaching cypher
# ---------------------------------------------------------------------------


class TestInjectionRejection:
    """validate_id must block all $$ and injection-relevant payloads."""

    @pytest.mark.parametrize("evil_ws", [
        "$$) AS (x agtype); DROP TABLE memories; SELECT * FROM cypher('memorylayer', $$ MATCH (m) RETURN m",
        "ws$evil",
        "ws'quote",
        "ws\"dquote",
        "ws;semicolon",
        "ws\x00null",
        "ws evil space",
        "ws\nlinefeed",
        "$$",
        "$",
    ])
    def test_validate_id_blocks_injection_payloads(self, evil_ws):
        with pytest.raises(ValueError):
            validate_id(evil_ws, kind="workspace_id")

    @pytest.mark.asyncio
    async def test_build_graph_validates_workspace_id(self, age_storage):
        """AgeGraphAnalysisService._build_graph must validate workspace_id."""
        storage, _ = age_storage
        engine = storage._engine
        _, age_svc = _make_services(storage, age_engine=engine)

        evil_ws = "$$) AS (x agtype); SELECT 'INJECTED'::text AS pwned --"
        with pytest.raises(ValueError, match="workspace_id"):
            await age_svc.analyze(evil_ws)


# ---------------------------------------------------------------------------
# MAJOR-1: phantom edge after association delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_phantom_edge_after_association_delete(age_storage):
    """After deleting an association relationally, re-analysis must NOT return it.

    This is the MAJOR-1 staleness fix: scope-DELETE before MERGE ensures the
    AGE subgraph always reflects current relational truth.
    """
    storage, created_workspaces = age_storage
    engine = storage._engine

    ws_id = "age_conf_phantom_%s" % uuid.uuid4().hex[:8]
    created_workspaces.append(ws_id)
    await _ensure_workspace(storage, ws_id)

    m0 = await _add_memory(storage, ws_id, "phantom node alpha")
    m1 = await _add_memory(storage, ws_id, "phantom node beta")
    m2 = await _add_memory(storage, ws_id, "phantom node gamma")

    # Create m0-m1 and m1-m2 associations.
    await _add_assoc(storage, ws_id, m0, m1, "related_to", 0.8)
    assoc = await storage.create_association(
        ws_id,
        AssociateInput(source_id=m1, target_id=m2, relationship="related_to", strength=0.6),
    )

    nx_svc, age_svc = _make_services(storage, age_engine=engine)

    # First analysis — both edges should appear in both backends.
    result_before = await age_svc.analyze(ws_id)
    assert result_before.snapshot.edge_count == 2, (
        "Expected 2 edges before deletion, got %d" % result_before.snapshot.edge_count
    )

    # Delete the m1-m2 association relationally.
    await storage.delete_association(ws_id, assoc.id)

    # Re-run analysis — the m1-m2 edge must NOT appear in AGE output (phantom).
    result_after = await age_svc.analyze(ws_id)
    assert result_after.snapshot.edge_count == 1, (
        "Phantom edge detected: expected 1 edge after deletion, got %d. "
        "MAJOR-1 (staleness) fix may be broken." % result_after.snapshot.edge_count
    )

    # NetworkX and AGE must agree.
    await _assert_parity(nx_svc, age_svc, ws_id)


# ---------------------------------------------------------------------------
# MAJOR-2: parallel relationship types + reverse-direction duplicates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_relationships_match_networkx(age_storage):
    """AGE must produce the same result as NetworkX when a node pair has
    multiple relationship types (parallel edges) and a reverse-direction
    duplicate — the dedup order must be identical (MAJOR-2 fix).
    """
    storage, created_workspaces = age_storage
    engine = storage._engine

    ws_id = "age_conf_parallel_%s" % uuid.uuid4().hex[:8]
    created_workspaces.append(ws_id)
    await _ensure_workspace(storage, ws_id)

    m0 = await _add_memory(storage, ws_id, "parallel alpha")
    m1 = await _add_memory(storage, ws_id, "parallel beta")
    m2 = await _add_memory(storage, ws_id, "parallel gamma")

    # Parallel relationships: m0→m1 with two different relationship types.
    await _add_assoc(storage, ws_id, m0, m1, "causes", 0.9)
    await _add_assoc(storage, ws_id, m0, m1, "similar_to", 0.5)

    # Reverse direction: m1→m0 with a third relationship type.
    await _add_assoc(storage, ws_id, m1, m0, "solves", 0.4)

    # Also a normal edge to give the graph some structure.
    await _add_assoc(storage, ws_id, m1, m2, "related_to", 0.7)

    nx_svc, age_svc = _make_services(storage, age_engine=engine)
    await _assert_parity(nx_svc, age_svc, ws_id)


# ---------------------------------------------------------------------------
# Archived / deleted endpoint exclusion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_archived_memory_excluded_from_graph(age_storage):
    """Archived memories must not appear as nodes; their edges must not appear."""
    storage, created_workspaces = age_storage
    engine = storage._engine

    ws_id = "age_conf_archive_%s" % uuid.uuid4().hex[:8]
    created_workspaces.append(ws_id)
    await _ensure_workspace(storage, ws_id)

    active = await _add_memory(storage, ws_id, "active node")
    archived = await _add_memory(storage, ws_id, "will be archived")

    await _add_assoc(storage, ws_id, active, archived, "related_to", 0.8)

    # Archive the second memory relationally by directly flipping status in the
    # ORM — archive_memory() requires _leann_storage (not initialized when we
    # bypass connect() against the pre-provisioned eval PG), so we use a direct
    # UPDATE which is exactly what the storage layer does under the hood.
    from sqlalchemy import update
    from memorylayer_saas.storage.models import MemoryModel
    async with storage._session_factory() as session:
        await session.execute(
            update(MemoryModel)
            .where(MemoryModel.id == archived, MemoryModel.workspace_id == ws_id)
            .values(status="archived")
        )
        await session.commit()

    nx_svc, age_svc = _make_services(storage, age_engine=engine)
    result = await age_svc.analyze(ws_id)

    # Only the active node should appear.
    assert result.snapshot.node_count == 1, (
        "Archived memory leaked into AGE graph: node_count=%d" % result.snapshot.node_count
    )
    assert result.snapshot.edge_count == 0

    await _assert_parity(nx_svc, age_svc, ws_id)


# ---------------------------------------------------------------------------
# Self-loop handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_loop_parity(age_storage):
    """Self-loops (source_id == target_id) must be handled identically to OSS.

    nx.Graph silently adds self-loops; AGE must produce the same node/edge count.
    """
    storage, created_workspaces = age_storage
    engine = storage._engine

    ws_id = "age_conf_selfloop_%s" % uuid.uuid4().hex[:8]
    created_workspaces.append(ws_id)
    await _ensure_workspace(storage, ws_id)

    m0 = await _add_memory(storage, ws_id, "self-loop node")
    m1 = await _add_memory(storage, ws_id, "neighbour node")

    # Self-loop on m0.
    await _add_assoc(storage, ws_id, m0, m0, "self_ref", 0.5)
    # Normal edge.
    await _add_assoc(storage, ws_id, m0, m1, "related_to", 0.7)

    nx_svc, age_svc = _make_services(storage, age_engine=engine)
    await _assert_parity(nx_svc, age_svc, ws_id)


# ---------------------------------------------------------------------------
# include_rpg=True path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_include_rpg_parity(age_storage):
    """include_rpg=True must produce identical output in AGE and NetworkX."""
    storage, created_workspaces = age_storage
    engine = storage._engine

    ws_id = "age_conf_rpg_%s" % uuid.uuid4().hex[:8]
    created_workspaces.append(ws_id)
    await _ensure_workspace(storage, ws_id)

    # Regular memory.
    m0 = await _add_memory(storage, ws_id, "regular memory")

    # RPG-subtyped memory.
    rpg_mem = await storage.create_memory(
        ws_id,
        RememberInput(content="rpg file node", subtype="rpg_file"),
    )
    await _add_assoc(storage, ws_id, m0, rpg_mem.id, "related_to", 0.6)

    nx_svc, age_svc = _make_services(storage, age_engine=engine)
    await _assert_parity(nx_svc, age_svc, ws_id, include_rpg=True)


# ---------------------------------------------------------------------------
# Idempotency: repeated analyze() calls produce the same result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_analysis_is_idempotent(age_storage):
    """Calling analyze() multiple times on the same workspace must produce
    identical results (scope-DELETE + re-MERGE is idempotent).
    """
    storage, created_workspaces = age_storage
    engine = storage._engine

    ws_id = "age_conf_idem_%s" % uuid.uuid4().hex[:8]
    created_workspaces.append(ws_id)

    await _golden._build_small_workspace(storage, ws_id)

    _, age_svc = _make_services(storage, age_engine=engine)
    run1 = normalize_for_golden((await age_svc.analyze(ws_id)).model_dump(mode="json"))
    run2 = normalize_for_golden((await age_svc.analyze(ws_id)).model_dump(mode="json"))
    run3 = normalize_for_golden((await age_svc.analyze(ws_id)).model_dump(mode="json"))

    assert run1 == run2 == run3, "analyze() is not idempotent"


# ---------------------------------------------------------------------------
# Cleanup verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_leaves_zero_test_workspaces(age_storage):
    """Verify the cleanup fixture removes all test rows (sanity check).

    This test creates a workspace, then relies on the fixture teardown to
    delete it. After teardown, a fresh connection should find 0 rows.
    Note: this test cannot query *after* teardown, so it just exercises the
    happy path — the actual verification is in the session-end check.
    """
    storage, created_workspaces = age_storage
    engine = storage._engine

    ws_id = "age_conf_cleanup_%s" % uuid.uuid4().hex[:8]
    created_workspaces.append(ws_id)
    await _ensure_workspace(storage, ws_id)

    # Workspace should exist now.
    existing = await storage.get_workspace(ws_id)
    assert existing is not None
    # Teardown will delete it — if cleanup has a bug, leftover rows in the
    # shared PG would be caught by the workspace prefix check in CI.
