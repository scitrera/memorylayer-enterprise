# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Apache-AGE-backed graph-QUERY service (ENTERPRISE-only; P2 Track A).

``AgeGraphQueryService`` is the AGE implementation of the recall/RAG-facing
``GraphQueryService`` read seam. It is conformance-tested for byte-identical
(normalized) parity against the OSS relational ``RelationalGraphQueryService``:
both backends return the SAME ``models.graph_query`` DTOs for the same
relational data.

Self-sufficient materialize-on-read
-----------------------------------
Track A must NOT depend on the graph-ANALYSIS ``analyze()`` having run, so each
query first MATERIALIZES the workspace subgraph from relational truth via the
SHARED ``materialize_workspace_subgraph`` helper (the same scope-DELETE-then-
MERGE flow the P2.1 analysis backend uses), then runs a read-only parameterized
cypher query.

Watermark-gated materialize (P2 strategy-C, Gate B): each read calls the SHARED
``materialize_workspace_subgraph_gated`` helper, which skips the delete-then-MERGE
when the workspace's change-watermark is unchanged since the last successful
materialize (the AGE subgraph persists across calls) and falls back to the full
delete-then-MERGE when the watermark advanced or is unavailable (fail-safe). The
watermark is ``(max memory updated_at, max association created_at, memory_count,
association_count)`` — the two counts are what make an association DELETE (which
advances no timestamp) detectable, preserving the P2.1 no-phantom-edge invariant.

Security: parameterized + validated cypher only
------------------------------------------------
Every query validates its caller values FIRST (``validate_id`` for
workspace_id / memory ids, ``validate_relationship`` for relationship types,
``validate_hops`` for depth/max_hops) and then passes them through the ``$1``
JSON bind arg of ``cypher(graph, $$..$$, $1)`` — values are NEVER inlined. The
only literal placed in a cypher body is the range-checked integer hop bound
(AGE var-length ranges require literal bounds). See ``_cypher.py``.

Selection: ``MEMORYLAYER_GRAPH_QUERY_PROVIDER=age``. The plugin refuses to
enable on a non-PostgreSQL backend (no ``session_factory``).
"""

from __future__ import annotations

import json
import logging

from scitrera_app_framework import Variables, ext_parse_bool, get_extension, get_logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from memorylayer_server.config import (
    DEFAULT_MEMORYLAYER_GRAPH_ENTITY_MATERIALIZE_ENABLED,
    DEFAULT_MEMORYLAYER_GRAPH_FRAGMENT_MATERIALIZE_ENABLED,
    MEMORYLAYER_GRAPH_ENTITY_MATERIALIZE_ENABLED,
    MEMORYLAYER_GRAPH_FRAGMENT_MATERIALIZE_ENABLED,
)
from memorylayer_server.models.graph_query import (
    CoMentionedEntity,
    DerivedFragment,
    EntityMentionedMemory,
    EntityNeighborhood,
    FragmentResult,
    GraphEdge,
    GraphNode,
    NeighborResult,
    PathResult,
    PatternMatch,
    RelationshipRollup,
    SubgraphResult,
)
from memorylayer_server.services._constants import EXT_STORAGE_BACKEND
from memorylayer_server.services.graph_query import GraphQueryServicePluginBase
from memorylayer_server.services.graph_query.base import GraphQueryService

from ...storage.models import MemoryAssociationModel
from ..graph_analysis import _cypher
from ..graph_analysis._materialize import (
    materialize_workspace_entities_gated,
    materialize_workspace_fragments_gated,
    materialize_workspace_subgraph_gated,
)
from ..graph_analysis.bootstrap import (
    bootstrap_age,
    ensure_age_session,
    make_age_engine_kwargs,
    _get_bootstrap_lock,
)

# Same hard cap the graph-analysis AGE backend uses when bulk-loading memories.
_MEMORY_LIMIT = 10000


class AgeGraphQueryService(GraphQueryService):
    """Graph query over Apache-AGE, normalized-identical to the relational tier.

    Reuses the dedicated-AGE-engine + ``_ensure_bootstrap`` + ``age_engine=``
    injection pattern of ``AgeGraphAnalysisService``.
    """

    PROVIDER_NAME = "age"

    def __init__(self, storage, v: Variables, *, age_engine=None):
        self._storage = storage
        self.logger = get_logger(v, name="AgeGraphQueryService")
        self._age_bootstrapped = False
        # Graph-moat C1 gate: materialize the entity registry into AGE as
        # Entity vertices + MENTIONS edges. DARK by default; even when ON it is a
        # clean no-op if the registry is empty/unavailable. Read once at init.
        self._entity_materialize_enabled = v.environ(
            MEMORYLAYER_GRAPH_ENTITY_MATERIALIZE_ENABLED,
            default=DEFAULT_MEMORYLAYER_GRAPH_ENTITY_MATERIALIZE_ENABLED,
            type_fn=ext_parse_bool,
        )
        # Graph-moat P4.5 gate: materialize decomposed-fact memories into AGE as
        # Fragment vertices + DERIVED_FROM edges (Fragment->source Memory). DARK by
        # default; even when ON it is a clean no-op if there are no fact memories.
        # NOT wired into recall — a dark, MEASURABLE capability. Read once at init.
        self._fragment_materialize_enabled = v.environ(
            MEMORYLAYER_GRAPH_FRAGMENT_MATERIALIZE_ENABLED,
            default=DEFAULT_MEMORYLAYER_GRAPH_FRAGMENT_MATERIALIZE_ENABLED,
            type_fn=ext_parse_bool,
        )
        # Dedicated engine with the AGE search_path baked into connect_args —
        # separate from the storage backend's own engine so we never mutate
        # shared state. Same connection string. (Mirrors AgeGraphAnalysisService.)
        if age_engine is not None:
            self._age_engine = age_engine
        else:
            conn_str = storage.connection_string
            self._age_engine = create_async_engine(
                conn_str,
                pool_pre_ping=True,
                **make_age_engine_kwargs(),
            )
        self._age_session_factory = async_sessionmaker(
            self._age_engine, class_=AsyncSession, expire_on_commit=False
        )

    # --- bootstrap (identical pattern to AgeGraphAnalysisService) ---------

    async def _ensure_bootstrap(self) -> None:
        """Provision AGE + the ``memorylayer`` graph, thread-safe and idempotent."""
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

    # --- relational reads + materialize-on-read --------------------------

    async def _node_attr_map(self, workspace_id: str) -> dict[str, dict]:
        """Bulk-load active workspace memories into ``{id: {type, subtype}}``.

        Same source rows as ``RelationalGraphQueryService._node_attr_map`` and
        the graph-analysis node set.
        """
        memories = await self._storage.search_memories_by_filter(
            workspace_id,
            status="active",
            limit=_MEMORY_LIMIT,
        )
        return {
            mem.id: {
                "memory_type": getattr(mem, "memory_type", None),
                "memory_subtype": getattr(mem, "subtype", None),
            }
            for mem in memories
        }

    async def _read_associations(self, workspace_id: str) -> list:
        """Read all associations for the workspace in ONE set-based query.

        Uses the AGE-engine session factory so the connection search_path
        resolves ``ag_catalog``-resident tables. Returns rows of
        ``(id, source_id, target_id, relation_type, strength)``.
        """
        async with self._age_session_factory() as session:
            result = await session.execute(
                select(
                    MemoryAssociationModel.id,
                    MemoryAssociationModel.source_id,
                    MemoryAssociationModel.target_id,
                    MemoryAssociationModel.relation_type,
                    MemoryAssociationModel.strength,
                ).where(MemoryAssociationModel.workspace_id == workspace_id)
            )
            return list(result.all())

    async def _materialize(self, workspace_id: str, node_attrs: dict[str, dict]) -> None:
        """Materialize-on-read: scope-DELETE + MERGE the workspace subgraph.

        P2 strategy-C (Gate B): the GATED wrapper skips the full delete-then-MERGE
        when the workspace change-watermark is unchanged since the last materialize
        (the AGE subgraph persists across calls), and falls back to the full
        delete-then-MERGE when the watermark advanced or is unavailable (fail-safe).
        Read-only queries then run against the persisted subgraph either way.
        """
        associations = await self._read_associations(workspace_id)
        await materialize_workspace_subgraph_gated(
            self._age_engine,
            storage=self._storage,
            workspace_id=workspace_id,
            node_attrs=node_attrs,
            associations=associations,
        )

    def _hydrate_node(self, memory_id: str, attr_map: dict[str, dict]) -> GraphNode:
        attrs = attr_map.get(memory_id, {"memory_type": None, "memory_subtype": None})
        return GraphNode(
            memory_id=memory_id,
            memory_type=attrs.get("memory_type"),
            memory_subtype=attrs.get("memory_subtype"),
        )

    @staticmethod
    def _validate_rels(relationship_types: list[str] | None) -> list[str] | None:
        """Validate each relationship type strictly; return the list or None."""
        if not relationship_types:
            return None
        return [_cypher.validate_relationship(r) for r in relationship_types]

    async def _fetch_induced_edges(
        self,
        workspace_id: str,
        node_ids: list[str],
        rels: list[str] | None,
    ) -> list[GraphEdge]:
        """Run the induced-edge cypher over a node-id set, return sorted DTOs."""
        if not node_ids:
            return []
        param: dict = {"ws": workspace_id, "ids": node_ids}
        if rels is not None:
            param["rels"] = rels
        sql = _cypher.induced_edges_sql(with_rel_filter=rels is not None)
        async with self._age_engine.connect() as conn:
            raw = (await conn.get_raw_connection()).driver_connection
            await ensure_age_session(raw)
            records = await raw.fetch(sql, json.dumps(param))
        edges = [
            GraphEdge(
                source_id=json.loads(r["src"]),
                target_id=json.loads(r["tgt"]),
                relationship=json.loads(r["relationship"]),
                strength=float(json.loads(r["strength"])),
            )
            for r in records
        ]
        edges.sort(key=lambda e: (e.source_id, e.target_id, e.relationship))
        return edges

    async def _reachable_via_paths(self, sql: str, param: dict, rels: list[str] | None) -> set[str]:
        """Run a path-returning traversal query, return the reachable node-id set.

        AGE 1.7 cannot apply a relationship-list filter inside a var-length
        pattern (no ALL() predicate), so the cypher returns whole paths
        (``nodes(p)`` / ``relationships(p)``) and we apply the filter HERE:
        a path contributes its endpoint to the reachable set only if EVERY edge
        on the path is in the relationship allowlist — exactly the relational
        per-hop filter semantics, so the two backends agree.
        """
        rel_set = set(rels) if rels is not None else None
        async with self._age_engine.connect() as conn:
            raw = (await conn.get_raw_connection()).driver_connection
            await ensure_age_session(raw)
            records = await raw.fetch(sql, json.dumps(param))

        reachable: set[str] = set()
        for r in records:
            edges = _cypher.parse_path_edges(r["edges"])
            if rel_set is not None and not all(e["relationship"] in rel_set for e in edges):
                continue
            verts = _cypher.parse_path_vertices(r["vertices"])
            if verts:
                # The path endpoint (last vertex) is the newly reached node;
                # intermediate vertices are reached by shorter prefixes and are
                # picked up by their own (shorter) path rows.
                reachable.add(verts[-1].get("id"))
        return reachable

    # --- public API -------------------------------------------------------

    async def neighbors(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        direction: str = "both",
        limit: int = 50,
    ) -> NeighborResult:
        _cypher.validate_id(workspace_id, kind="workspace_id")
        _cypher.validate_id(memory_id, kind="memory_id")
        rels = self._validate_rels(relationship_types)
        d = _cypher.validate_hops(depth, kind="depth")

        await self._ensure_bootstrap()
        attr_map = await self._node_attr_map(workspace_id)
        await self._materialize(workspace_id, attr_map)

        # Phase 1: reachable node set within depth hops (paths + Python filter).
        reach_sql = _cypher.neighbors_paths_sql(d, direction)
        reachable = await self._reachable_via_paths(
            reach_sql, {"ws": workspace_id, "root": memory_id}, rels
        )
        visited = {memory_id} | reachable

        # Canonical truncation: roots first (sorted), then non-roots (sorted).
        # Matches the OSS _bfs post-expansion cap so both backends retain the
        # identical deterministic node set regardless of edge-batch ordering.
        truncated = False
        if len(visited) > limit:
            truncated = True
            sorted_roots = [memory_id]
            sorted_non_roots = sorted(visited - {memory_id})
            ordered = sorted_roots + sorted_non_roots
            visited = set(ordered[:limit])

        node_ids = sorted(visited)
        edges = await self._fetch_induced_edges(workspace_id, node_ids, rels)
        nodes = [self._hydrate_node(nid, attr_map) for nid in node_ids]
        return NeighborResult(
            root_id=memory_id,
            nodes=nodes,
            edges=edges,
            truncated=truncated,
        )

    async def k_hop_subgraph(
        self,
        workspace_id: str,
        memory_ids: list[str],
        *,
        depth: int = 2,
        relationship_types: list[str] | None = None,
        direction: str = "both",
        node_limit: int = 200,
    ) -> SubgraphResult:
        _cypher.validate_id(workspace_id, kind="workspace_id")
        for mid in memory_ids:
            _cypher.validate_id(mid, kind="memory_id")
        rels = self._validate_rels(relationship_types)
        d = _cypher.validate_hops(depth, kind="depth")

        await self._ensure_bootstrap()
        attr_map = await self._node_attr_map(workspace_id)
        await self._materialize(workspace_id, attr_map)

        roots = sorted(set(memory_ids))
        reach_sql = _cypher.k_hop_paths_sql(d, direction)
        reachable = await self._reachable_via_paths(
            reach_sql, {"ws": workspace_id, "roots": roots}, rels
        )
        visited = set(roots) | reachable

        # Canonical truncation: sorted roots first, then sorted non-roots.
        # Matches the OSS _bfs post-expansion cap ordering exactly.
        truncated = False
        if len(visited) > node_limit:
            truncated = True
            sorted_non_roots = sorted(visited - set(roots))
            ordered = roots + sorted_non_roots
            visited = set(ordered[:node_limit])

        node_ids = sorted(visited)
        edges = await self._fetch_induced_edges(workspace_id, node_ids, rels)
        nodes = [self._hydrate_node(nid, attr_map) for nid in node_ids]
        return SubgraphResult(
            nodes=nodes,
            edges=edges,
            root_ids=roots,
            truncated=truncated,
        )

    async def shortest_path(
        self,
        workspace_id: str,
        src_id: str,
        dst_id: str,
        *,
        max_hops: int = 5,
        relationship_types: list[str] | None = None,
    ) -> PathResult:
        _cypher.validate_id(workspace_id, kind="workspace_id")
        _cypher.validate_id(src_id, kind="memory_id")
        _cypher.validate_id(dst_id, kind="memory_id")
        rels = self._validate_rels(relationship_types)
        rel_set = set(rels) if rels is not None else None
        h = _cypher.validate_hops(max_hops, kind="max_hops")

        await self._ensure_bootstrap()
        attr_map = await self._node_attr_map(workspace_id)
        await self._materialize(workspace_id, attr_map)

        # Trivial path: src == dst.
        if src_id == dst_id:
            return PathResult(found=True, nodes=[self._hydrate_node(src_id, attr_map)], edges=[], hops=0)

        # AGE 1.7 has no shortestPath() and no path-level WHERE filter, so we
        # enumerate bounded paths (ORDER BY hops ASC LIMIT 1000) and filter in
        # Python. Among paths at the minimum hop count we pick the one whose
        # node-id sequence is lexicographically smallest — the same tie-break
        # the OSS BFS produces by expanding sorted frontiers.
        #
        # LIMIT boundary: if a relationship filter is active and more than 1000
        # equal-length non-matching paths precede the first matching one, this
        # will return found=False. Documented in ``_cypher.shortest_path_sql``.
        sql = _cypher.shortest_path_sql(h)
        param = {"ws": workspace_id, "src": src_id, "dst": dst_id}
        async with self._age_engine.connect() as conn:
            raw = (await conn.get_raw_connection()).driver_connection
            await ensure_age_session(raw)
            records = await raw.fetch(sql, json.dumps(param))

        # Phase 1: parse all rows, apply relationship filter, group by hops.
        # Records are already ordered hops ASC from the SQL; we stop once we
        # step past the minimum hop count at which a matching path was found.
        best_hops: int | None = None
        candidates: list[tuple[list[str], list[GraphEdge]]] = []

        for row in records:
            verts = _cypher.parse_path_vertices(row["vertices"])
            edges_raw = _cypher.parse_path_edges(row["edges"])
            internal_ids = _cypher.parse_path_vertex_internal_ids(row["vertices"])
            hops_val = int(json.loads(row["hops"]))

            # Once we've collected at least one match and moved past its hop
            # count, no shorter path can appear (results are ORDER BY hops ASC).
            if best_hops is not None and hops_val > best_hops:
                break

            if rel_set is not None and not all(e["relationship"] in rel_set for e in edges_raw):
                continue

            node_ids = [v.get("id") for v in verts]
            # Orient each edge by AGE internal start/end ids so source_id/target_id
            # reflect the STORED association direction.
            id_by_internal = dict(zip(internal_ids, node_ids))
            edge_dtos: list[GraphEdge] = []
            for e in edges_raw:
                src = id_by_internal.get(e["start_id"])
                tgt = id_by_internal.get(e["end_id"])
                edge_dtos.append(
                    GraphEdge(
                        source_id=src,
                        target_id=tgt,
                        relationship=e["relationship"],
                        strength=float(e["strength"]),
                    )
                )
            best_hops = hops_val
            candidates.append((node_ids, edge_dtos))

        if not candidates:
            return PathResult(found=False, nodes=[], edges=[], hops=0)

        # Phase 2: tie-break — lexicographically smallest node-id sequence.
        # This matches the OSS BFS which naturally picks the lex-smallest path
        # by expanding sorted frontiers and a sorted batch.
        candidates.sort(key=lambda c: c[0])
        chosen_node_ids, chosen_edges = candidates[0]
        nodes = [self._hydrate_node(nid, attr_map) for nid in chosen_node_ids]
        return PathResult(found=True, nodes=nodes, edges=chosen_edges, hops=len(chosen_edges))

    async def typed_pattern(
        self,
        workspace_id: str,
        *,
        relationship: str,
        limit: int = 100,
    ) -> list[PatternMatch]:
        _cypher.validate_id(workspace_id, kind="workspace_id")
        _cypher.validate_relationship(relationship)

        await self._ensure_bootstrap()
        attr_map = await self._node_attr_map(workspace_id)
        await self._materialize(workspace_id, attr_map)

        sql = _cypher.typed_pattern_sql()
        param = {"ws": workspace_id, "rel": relationship}
        async with self._age_engine.connect() as conn:
            raw = (await conn.get_raw_connection()).driver_connection
            await ensure_age_session(raw)
            records = await raw.fetch(sql, json.dumps(param))

        matches = [
            PatternMatch(
                source_id=json.loads(r["src"]),
                target_id=json.loads(r["tgt"]),
                relationship=json.loads(r["relationship"]),
                strength=float(json.loads(r["strength"])),
            )
            for r in records
        ]
        matches.sort(key=lambda m: (m.source_id, m.target_id, m.relationship))
        return matches[:limit]

    async def relationship_rollup(
        self,
        workspace_id: str,
    ) -> list[RelationshipRollup]:
        _cypher.validate_id(workspace_id, kind="workspace_id")

        await self._ensure_bootstrap()
        attr_map = await self._node_attr_map(workspace_id)
        await self._materialize(workspace_id, attr_map)

        sql = _cypher.relationship_rollup_sql()
        param = {"ws": workspace_id}
        async with self._age_engine.connect() as conn:
            raw = (await conn.get_raw_connection()).driver_connection
            await ensure_age_session(raw)
            records = await raw.fetch(sql, json.dumps(param))

        rollups = [
            RelationshipRollup(
                relationship=json.loads(r["relationship"]),
                count=int(json.loads(r["cnt"])),
            )
            for r in records
        ]
        rollups.sort(key=lambda r: (-r.count, r.relationship))
        return rollups

    # --- entity-neighborhood (graph-moat C2; reads the C1 Entity layer) ---

    async def _materialize_entities(self, workspace_id: str) -> bool:
        """C1: materialize the workspace's Entity vertices + MENTIONS edges.

        Gated behind the entity-materialize flag. Loads the ACTIVE entities and
        ALL member edges from the registry (the new workspace-scoped listing
        primitives) and delegates to the watermark-gated entity materialize.
        Returns ``True`` if the registry layer was materialized (or skipped on an
        unchanged watermark), ``False`` if the flag is off or the registry is
        unavailable (clean no-op — Entity materialization never errors out a
        Memory-layer query).
        """
        if not self._entity_materialize_enabled:
            return False
        try:
            entities = await self._storage.list_workspace_entities(workspace_id)
            members = await self._storage.list_workspace_entity_members(workspace_id)
        except NotImplementedError:
            # Storage backend has no entity registry — clean no-op.
            return False
        except Exception:  # noqa: BLE001 - entity layer is additive; never break the query
            self.logger.exception(
                "Failed to load registry for entity materialization in workspace %s; "
                "skipping entity layer", workspace_id,
            )
            return False

        # Watermark-gated delete-then-MERGE of the entity layer (separate from the
        # Memory subgraph watermark). An empty registry records a {0,"",0}
        # watermark and is a clean no-op.
        await materialize_workspace_entities_gated(
            self._age_engine,
            workspace_id=workspace_id,
            entities=entities,
            members=members,
        )
        return True

    async def entity_neighborhood(
        self,
        workspace_id: str,
        entity_id: str,
        *,
        hops: int = 1,
        memory_limit: int = 100,
        entity_limit: int = 50,
    ) -> EntityNeighborhood:
        """Enterprise AGE entity-neighborhood over the C1 Entity/MENTIONS layer.

        This is the consumer that makes C1 non-speculative: it materializes the
        Memory subgraph AND the Entity layer (C1), then runs read-only
        parameterized cypher over EXACTLY the vertices/edges C1 wrote.

        When the entity-materialize flag is OFF (or the registry is empty), the
        Entity layer is absent and the read cypher returns nothing — an empty
        neighborhood, never an error.
        """
        _cypher.validate_id(workspace_id, kind="workspace_id")
        _cypher.validate_id(entity_id, kind="entity_id")

        await self._ensure_bootstrap()
        # Materialize the Memory subgraph first so MENTIONS endpoints exist, then
        # the Entity layer (C1). Both are watermark-gated and idempotent.
        attr_map = await self._node_attr_map(workspace_id)
        await self._materialize(workspace_id, attr_map)
        await self._materialize_entities(workspace_id)

        # memories_for_entity: the memories this entity MENTIONS (1 hop).
        mem_param = json.dumps({"ws": workspace_id, "eid": entity_id})
        async with self._age_engine.connect() as conn:
            raw = (await conn.get_raw_connection()).driver_connection
            await ensure_age_session(raw)
            mem_records = await raw.fetch(_cypher.entity_mentioned_memories_sql(), mem_param)
            co_records = await raw.fetch(_cypher.entity_co_mentioned_entities_sql(), mem_param)

        # Dedup mentioned memories by id: first role wins on (memory_id, role)
        # ascending order — matching the OSS list_workspace_entity_members sort
        # (entity_id, memory_id, role) which, for one entity, is (memory_id, role).
        # The cypher already emits ORDER BY m.id, r.role; we re-sort here as a
        # defence-in-depth guard in case the fetch layer reorders rows.
        own_roles: dict[str, str] = {}
        for r in sorted(mem_records, key=lambda x: (
            json.loads(x["memory_id"]),
            json.loads(x["role"]) if x["role"] is not None else "mention",
        )):
            mid = json.loads(r["memory_id"])
            role = json.loads(r["role"]) if r["role"] is not None else "mention"
            own_roles.setdefault(mid, role)

        mentioned = [
            EntityMentionedMemory(
                memory_id=mid,
                role=own_roles[mid],
                memory_type=(attr_map.get(mid) or {}).get("memory_type"),
                memory_subtype=(attr_map.get(mid) or {}).get("memory_subtype"),
            )
            for mid in sorted(own_roles)
        ]

        # entities_co_mentioned: aggregate (co-entity, shared-memory) pairs into a
        # distinct shared-memory count per co-entity.
        co_shared: dict[str, set[str]] = {}
        co_meta: dict[str, dict] = {}
        for r in co_records:
            eid = json.loads(r["entity_id"])
            shared_mid = json.loads(r["shared_memory_id"])
            co_shared.setdefault(eid, set()).add(shared_mid)
            if eid not in co_meta:
                co_meta[eid] = {
                    "label": json.loads(r["label"]) if r["label"] is not None else None,
                    "entity_type": json.loads(r["entity_type"]) if r["entity_type"] is not None else None,
                }

        co_entities = [
            CoMentionedEntity(
                entity_id=eid,
                label=co_meta[eid]["label"],
                entity_type=co_meta[eid]["entity_type"],
                shared_memory_count=len(shared),
            )
            for eid, shared in co_shared.items()
        ]
        # Same ranking as the OSS backend: shared count desc, then id asc.
        co_entities.sort(key=lambda c: (-c.shared_memory_count, c.entity_id))

        truncated = False
        if len(mentioned) > memory_limit:
            truncated = True
            mentioned = mentioned[:memory_limit]
        if len(co_entities) > entity_limit:
            truncated = True
            co_entities = co_entities[:entity_limit]

        return EntityNeighborhood(
            entity_id=entity_id,
            memories_for_entity=mentioned,
            entities_co_mentioned=co_entities,
            truncated=truncated,
        )

    # --- fragment-traversal (graph-moat P4.5; reads the Fragment layer) ---

    async def _load_fragments(self, workspace_id: str) -> list[dict]:
        """Load the workspace's decomposed-fact memories as fragment dicts.

        The fact channel stores facts as ``subtype="fact"`` memories carrying
        ``metadata["source_id"]=<parent>``. Returns fragment dicts (``id``,
        ``content``, ``source_id``, ``updated_at``) — the input to both the
        fragment watermark and the materialize.
        """
        facts = await self._storage.search_memories_by_filter(
            workspace_id,
            subtypes=["fact"],
            status="active",
            limit=_MEMORY_LIMIT,
        )
        out: list[dict] = []
        for mem in facts:
            meta = getattr(mem, "metadata", None) or {}
            out.append({
                "id": mem.id,
                "content": getattr(mem, "content", None),
                "source_id": meta.get("source_id"),
                "updated_at": getattr(mem, "updated_at", None),
            })
        return out

    async def _materialize_fragments(self, workspace_id: str) -> bool:
        """P4.5: materialize the workspace's Fragment vertices + DERIVED_FROM edges.

        Gated behind the fragment-materialize flag. Loads the active fact
        memories and delegates to the watermark-gated fragment materialize.
        Returns ``True`` if the fragment layer was materialized (or skipped on an
        unchanged watermark), ``False`` if the flag is off or the load fails
        (clean no-op — fragment materialization never errors out the query).
        """
        if not self._fragment_materialize_enabled:
            return False
        try:
            fragments = await self._load_fragments(workspace_id)
        except NotImplementedError:
            return False
        except Exception:  # noqa: BLE001 - fragment layer is additive; never break the query
            self.logger.exception(
                "Failed to load fact memories for fragment materialization in workspace %s; "
                "skipping fragment layer", workspace_id,
            )
            return False

        # Watermark-gated delete-then-MERGE of the fragment layer (separate from
        # the Memory subgraph + Entity watermarks). An empty fact set records a
        # {0,"",<digest>} watermark and is a clean no-op.
        await materialize_workspace_fragments_gated(
            self._age_engine,
            workspace_id=workspace_id,
            fragments=fragments,
        )
        return True

    async def fragments_for_memory(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        limit: int = 100,
    ) -> FragmentResult:
        """Enterprise AGE fragment-traversal over the P4.5 Fragment/DERIVED_FROM layer.

        This is the consumer that makes the materialized Fragment/DERIVED_FROM
        non-speculative: it materializes the Memory subgraph AND the Fragment
        layer (P4.5), then runs read-only parameterized cypher over EXACTLY the
        vertices/edges P4.5 wrote (``Fragment -> DERIVED_FROM -> Memory``).

        When the fragment-materialize flag is OFF (or there are no fact
        memories), the Fragment layer is absent and the read cypher returns
        nothing — an empty result, never an error.

        DARK + MEASURABLE — NOT wired into recall.
        """
        _cypher.validate_id(workspace_id, kind="workspace_id")
        _cypher.validate_id(memory_id, kind="memory_id")

        await self._ensure_bootstrap()
        # Materialize the Memory subgraph first so DERIVED_FROM endpoints exist,
        # then the Fragment layer (P4.5). Both are watermark-gated and idempotent.
        attr_map = await self._node_attr_map(workspace_id)
        await self._materialize(workspace_id, attr_map)
        await self._materialize_fragments(workspace_id)

        param = json.dumps({"ws": workspace_id, "mid": memory_id})
        async with self._age_engine.connect() as conn:
            raw = (await conn.get_raw_connection()).driver_connection
            await ensure_age_session(raw)
            records = await raw.fetch(_cypher.fragments_for_memory_sql(), param)

        fragments = [
            DerivedFragment(
                fragment_id=json.loads(r["fragment_id"]),
                content=json.loads(r["content"]) if r["content"] is not None else None,
                source_id=memory_id,
            )
            for r in records
        ]
        # Deterministic ordering by fragment_id asc (the cypher already emits
        # ORDER BY f.id; re-sort as a defence-in-depth guard so the truncation
        # boundary matches the OSS relational fallback regardless of row order).
        fragments.sort(key=lambda d: d.fragment_id)

        truncated = False
        if len(fragments) > limit:
            truncated = True
            fragments = fragments[:limit]

        return FragmentResult(
            source_id=memory_id,
            fragments=fragments,
            truncated=truncated,
        )


class AgeGraphQueryServicePlugin(GraphQueryServicePluginBase):
    """Plugin enabling the AGE graph-query backend when
    ``MEMORYLAYER_GRAPH_QUERY_PROVIDER=age``.

    Auto-discovered by the enterprise ``register_package_plugins`` scan. Refuses
    to enable on a backend without ``session_factory`` (e.g. SQLite).
    """

    PROVIDER_NAME = "age"

    def initialize(self, v: Variables, logger: logging.Logger) -> AgeGraphQueryService:
        storage = get_extension(EXT_STORAGE_BACKEND, v)
        session_factory = getattr(storage, "session_factory", None)
        if session_factory is None:
            raise RuntimeError(
                "AgeGraphQueryService requires a PostgreSQL storage backend "
                "with a 'session_factory' (got %s). "
                "Set MEMORYLAYER_GRAPH_QUERY_PROVIDER=default for the "
                "SQLite/OSS tier." % type(storage).__name__
            )
        return AgeGraphQueryService(storage=storage, v=v)
