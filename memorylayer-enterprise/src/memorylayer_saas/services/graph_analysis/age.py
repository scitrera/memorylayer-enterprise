"""Apache-AGE-backed graph-analysis service (ENTERPRISE-only).

Parity guarantee
----------------
Subclasses OSS ``NetworkXGraphAnalysisService`` and overrides ONLY
``_build_graph``. Every downstream computation — ``_analyze_core`` (Louvain
seed=42, betweenness, bridges, stats) and all six public ABC methods — is
inherited verbatim. As long as ``_build_graph`` produces a byte-identical
``nx.Graph`` (same node set, same node attrs, same edge set + attrs), the
emitted ``GraphAnalysis`` is byte-identical to the OSS NetworkX tier.

Security: parameterized cypher only
------------------------------------
All caller-supplied values (workspace_id, memory_id, assoc_id, relationship,
strength) are passed as a JSON object bound to the ``$1`` SQL parameter of
``cypher(graph, $$..$$, $1)`` — they are NEVER inlined into SQL or cypher
text. A strict charset allowlist in ``_cypher.validate_id`` provides a
second layer of defence. See ``_cypher.py`` for the full security model.

MAJOR-1 fix (staleness): scope-DELETE before MERGE
---------------------------------------------------
Before materialising from relational truth, ``_materialize_and_extract``
deletes this workspace's AGE subgraph (edges first, then vertices). This
matches OSS "fresh-per-call" semantics so phantom edges from deleted
associations and stale status on archived memories cannot accumulate.

MAJOR-2 fix (dedup divergence): no manual dedup
-------------------------------------------------
Edge rows are sorted by ``(src, tgt, relationship)`` for determinism, then
each row is passed directly to ``nx.Graph.add_edge``. ``nx.Graph`` naturally
collapses (A,B) and (B,A) to the same undirected edge with last-write-wins
on attrs — identical to how the OSS ``get_associations_batch`` loop behaves.

MAJOR-3 fix (bootstrap race): asyncio.Lock + DB advisory lock
--------------------------------------------------------------
``_ensure_bootstrap`` acquires a module-level ``asyncio.Lock`` (serialises
concurrent callers in the same process), then sets a process-level flag.
The DB-level ``pg_advisory_xact_lock(42)`` inside ``bootstrap_age`` handles
multi-process races.

MAJOR-4 fix (ORM search_path): engine created with search_path connect_arg
--------------------------------------------------------------------------
``AgeGraphAnalysisService`` builds its own engine from the storage backend's
connection string, with ``connect_args={"server_settings":{"search_path":...}}``
so every session-factory connection resolves ``ag_catalog``-resident tables
without a per-connection ``SET search_path`` call.

Selection: ``MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER=age``. Default stays
``"default"`` (NetworkX) so OSS and un-opted-in enterprise are untouched.
The plugin refuses to enable on a non-PostgreSQL backend.
"""

from __future__ import annotations

import json
import logging

import networkx as nx
from memorylayer_server.services._constants import EXT_STORAGE_BACKEND
from memorylayer_server.services.graph_analysis import GraphAnalysisServicePluginBase
from memorylayer_server.services.graph_analysis.default import NetworkXGraphAnalysisService
from memorylayer_server_rpg.services.ontology_contributor import RPG_SUBTYPES as RPG_SUBTYPE_GROUPS
from scitrera_app_framework import Variables, get_extension, get_logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ...storage.models import MemoryAssociationModel
from . import _cypher
from ._materialize import materialize_workspace_subgraph_gated
from .bootstrap import (
    _get_bootstrap_lock,
    bootstrap_age,
    ensure_age_session,
    make_age_engine_kwargs,
)

# Same hard cap the OSS _build_graph uses.
_MEMORY_LIMIT = 10000

# Keep AGE materialization aligned with the RPG plugin's ontology contributor.
_RPG_SUBTYPES = sorted(
    subtype for subtypes in RPG_SUBTYPE_GROUPS.values() for subtype in subtypes
)


class AgeGraphAnalysisService(NetworkXGraphAnalysisService):
    """Graph analysis over Apache-AGE, byte-identical output to NetworkX.

    Overrides ONLY ``_build_graph``; all other methods are inherited.
    """

    def __init__(self, storage, v: Variables, *, age_engine=None):
        super().__init__(storage=storage, v=v)
        self.logger = get_logger(v, name="AgeGraphAnalysisService")
        self._age_bootstrapped = False
        # Dedicated engine with the AGE search_path baked into connect_args.
        # This engine is separate from the storage backend's own engine so we
        # do not mutate shared state. It uses the same connection string.
        if age_engine is not None:
            self._age_engine = age_engine
        else:
            conn_str = storage.connection_string
            self._age_engine = create_async_engine(
                conn_str,
                pool_pre_ping=True,
                **make_age_engine_kwargs(),
            )
        # Session factory on the AGE engine — ORM reads resolve ag_catalog.
        self._age_session_factory = async_sessionmaker(
            self._age_engine, class_=AsyncSession, expire_on_commit=False
        )

    # --- bootstrap -------------------------------------------------------

    async def _ensure_bootstrap(self) -> None:
        """Provision AGE + the ``memorylayer`` graph, thread-safe and idempotent.

        Uses a module-level asyncio.Lock (process-level Python serialisation)
        plus a DB-level pg_advisory_xact_lock (cross-process serialisation in
        the same Postgres instance). Sets a per-instance flag so the happy
        path after first init is a single boolean check.
        """
        if self._age_bootstrapped:
            return
        lock = _get_bootstrap_lock()
        async with lock:
            if self._age_bootstrapped:  # double-checked inside lock
                return
            async with self._age_engine.connect() as conn:
                raw = (await conn.get_raw_connection()).driver_connection
                await bootstrap_age(raw)
            self._age_bootstrapped = True

    # --- the single overridden method ------------------------------------

    async def _build_graph(
        self,
        workspace_id: str,
        context_id: str | None = None,
        include_rpg: bool = False,
    ) -> nx.Graph:
        """Build the workspace ``nx.Graph`` via AGE, with a NetworkX fallback.

        AGE is the enterprise default, but it must never fail the KB pipeline
        closed. Any AGE/cypher/bootstrap fault is caught, loudly logged
        (WARN + error + workspace_id, so the fault is detectable), and the
        inherited OSS NetworkX ``_build_graph`` is used instead. The fallback
        returns the same ``nx.Graph`` shape every caller (``_analyze_core``
        and the six public methods) already expects, so the KB still updates.
        """
        try:
            return await self._build_graph_age(
                workspace_id,
                context_id=context_id,
                include_rpg=include_rpg,
            )
        except Exception as e:  # noqa: BLE001 — fail-open to NetworkX, never closed.
            # Loudly surface the AGE fault so it is detected (NOT swallowed),
            # then fall back to the inherited NetworkX path so KB still updates.
            self.logger.warning(
                "AGE graph build FAILED for workspace %s (%s: %s); falling back "
                "to NetworkX graph analysis. AGE fault requires investigation.",
                workspace_id,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return await super()._build_graph(
                workspace_id,
                context_id=context_id,
                include_rpg=include_rpg,
            )

    async def _build_graph_age(
        self,
        workspace_id: str,
        context_id: str | None = None,
        include_rpg: bool = False,
    ) -> nx.Graph:
        """Build the workspace ``nx.Graph`` via AGE, byte-identical to OSS.

        Steps:
          a. validate workspace_id (injection defence layer 2);
          b. ensure AGE + graph exist (once per instance, lock-protected);
          c. read the active-memory node set (and optional RPG nodes) and
             the associations from the RELATIONAL tables — identical source
             rows to the OSS impl;
          d. scope-DELETE then MERGE+extract via parameterized cypher
             (MAJOR-1: no phantom edges; BLOCKER: no inlining);
          e. assemble the same ``nx.Graph`` with deterministic edge ordering
             and no manual dedup (MAJOR-2 fix).
        """
        # (a) Validate workspace_id strictly — reject anything with $, $$,
        # quotes, or other chars that could confuse cypher even via $1 params.
        _cypher.validate_id(workspace_id, kind="workspace_id")

        await self._ensure_bootstrap()

        # (c) Relational reads — same node set as NetworkXGraphAnalysisService.
        memories = await self._storage.search_memories_by_filter(
            workspace_id,
            status="active",
            context_id=context_id,
            limit=_MEMORY_LIMIT,
        )
        node_attrs: dict[str, dict] = {}
        for mem in memories:
            node_attrs[mem.id] = {
                "memory_type": getattr(mem, "memory_type", None),
                "memory_subtype": getattr(mem, "subtype", None),
            }

        if include_rpg:
            try:
                rpg_memories = await self._storage.search_memories_by_filter(
                    workspace_id,
                    subtypes=_RPG_SUBTYPES,
                    status="active",
                    context_id=context_id,
                    limit=_MEMORY_LIMIT,
                )
                for mem in rpg_memories:
                    if mem.id not in node_attrs:
                        node_attrs[mem.id] = {
                            "memory_type": getattr(mem, "memory_type", None),
                            "memory_subtype": getattr(mem, "subtype", None),
                        }
            except Exception as e:  # noqa: BLE001
                self.logger.debug("RPG node loading skipped: %s", e)

        # One set-based SELECT replaces the inherited N+1 get_associations loop.
        associations = await self._read_associations(workspace_id)

        # (d) Scope-DELETE, MERGE, then extract — all parameterized.
        nodes_rows, edge_rows = await self._materialize_and_extract(
            workspace_id=workspace_id,
            node_attrs=node_attrs,
            associations=associations,
        )

        # (e) Assemble the nx.Graph.
        g = nx.Graph()
        for node_id in nodes_rows:
            attrs = node_attrs.get(node_id, {"memory_type": None, "memory_subtype": None})
            g.add_node(node_id, **attrs)

        # Sort edge rows by (src, tgt, relationship) for determinism, then call
        # add_edge for every row without manual dedup. nx.Graph collapses (A,B)
        # and (B,A) to one undirected edge — last write wins for attrs, matching
        # the OSS get_associations_batch iteration semantics (MAJOR-2 fix).
        edge_rows_sorted = sorted(edge_rows, key=lambda r: (r[0], r[1], r[2]))
        for src, tgt, rel, strength in edge_rows_sorted:
            if src not in g or tgt not in g:
                continue
            g.add_edge(src, tgt, relationship_type=rel, strength=strength)

        self.logger.debug(
            "Built AGE graph for workspace %s: %d nodes, %d edges",
            workspace_id,
            g.number_of_nodes(),
            g.number_of_edges(),
        )
        return g

    # --- helpers ---------------------------------------------------------

    async def _read_associations(self, workspace_id: str) -> list:
        """Read all associations for the workspace in ONE set-based query.

        Uses the AGE-engine session factory so the connection's search_path
        already resolves ``ag_catalog``-resident tables (MAJOR-4 fix).
        Returns rows of (id, source_id, target_id, relation_type, strength).
        """
        async with self._age_session_factory() as session:
            result = await session.execute(
                select(
                    MemoryAssociationModel.id,
                    MemoryAssociationModel.source_id,
                    MemoryAssociationModel.target_id,
                    # ORM attr is relation_type; DB column is "relationship".
                    MemoryAssociationModel.relation_type,
                    MemoryAssociationModel.strength,
                ).where(MemoryAssociationModel.workspace_id == workspace_id)
            )
            return list(result.all())

    async def _materialize_and_extract(
        self,
        *,
        workspace_id: str,
        node_attrs: dict[str, dict],
        associations: list,
    ) -> tuple[list[str], list[tuple]]:
        """Scope-DELETE, MERGE vertices+edges, then extract node/edge sets.

        All cypher is parameterized via ``$1`` JSON bind arg — no user values
        in SQL text. Scope-DELETE runs first (MAJOR-1: fresh-per-call, no
        phantom edges from deleted associations or stale status).

        Returns ``(node_ids, edge_rows)`` where each edge row is
        ``(source_id, target_id, relationship, strength)``. Both directions
        of each undirected edge are returned; the caller sorts and calls
        ``nx.Graph.add_edge`` for each (no manual dedup — MAJOR-2 fix).
        """
        # The delete-then-MERGE half is the shared dependency reused by the
        # Track-A graph-QUERY backend; it lives in ``_materialize`` so a single
        # fix applies to both backends and they cannot drift. The GATED wrapper
        # (P2 strategy-C, Gate B) skips the full delete-then-MERGE when the
        # workspace change-watermark is unchanged since the last materialize (the
        # AGE subgraph persists across calls); it falls back to the full
        # materialize when the watermark advanced or is unavailable (fail-safe).
        await materialize_workspace_subgraph_gated(
            self._age_engine,
            storage=self._storage,
            workspace_id=workspace_id,
            node_attrs=node_attrs,
            associations=associations,
        )

        ws_param = json.dumps({"ws": workspace_id})

        async with self._age_engine.connect() as conn:
            raw = (await conn.get_raw_connection()).driver_connection
            # LOAD + search_path are session-scoped; re-apply on each pooled conn.
            await ensure_age_session(raw)

            # Extract scoped node set.
            node_records = await raw.fetch(_cypher.extract_nodes_sql(), ws_param)
            node_ids = [json.loads(r["id"]) for r in node_records]

            # Extract scoped undirected edge list (both directions).
            edge_records = await raw.fetch(_cypher.extract_edges_sql(), ws_param)
            edge_rows: list[tuple] = []
            for r in edge_records:
                edge_rows.append((
                    json.loads(r["src"]),
                    json.loads(r["tgt"]),
                    json.loads(r["relationship"]),
                    float(json.loads(r["strength"])),
                ))

        return node_ids, edge_rows


class AgeGraphAnalysisServicePlugin(GraphAnalysisServicePluginBase):
    """Plugin enabling the AGE backend when ``MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER=age``.

    Auto-discovered by the enterprise ``register_package_plugins(services...,
    recursive=True)`` scan — no edit to ``plugins.py`` needed. Refuses to
    enable on a backend without ``session_factory`` (e.g. SQLite).
    """

    PROVIDER_NAME = "age"

    def initialize(self, v: Variables, logger: logging.Logger) -> AgeGraphAnalysisService:
        storage = get_extension(EXT_STORAGE_BACKEND, v)
        session_factory = getattr(storage, "session_factory", None)
        if session_factory is None:
            raise RuntimeError(
                "AgeGraphAnalysisService requires a PostgreSQL storage backend "
                f"with a 'session_factory' (got {type(storage).__name__}). "
                "Set MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER=default for the "
                "SQLite/OSS tier."
            )
        return AgeGraphAnalysisService(storage=storage, v=v)
