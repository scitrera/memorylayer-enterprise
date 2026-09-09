# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""openCypher query builders for the Apache-AGE graph-analysis backend.

All queries use AGE 1.5+ parameterized cypher: the third argument to
``cypher(graph, $$ body $$, $1)`` is a SQL bind parameter containing a JSON
object whose keys are referenced as ``$key`` inside the cypher body. This
eliminates all string inlining of caller-controlled values and is the primary
injection defence. No value is ever interpolated into the query text.

Graph schema (per ``DESIGN_age_graph_analysis_backend.md`` §2):
  * single graph ``memorylayer`` for the whole database
  * vertex label ``Memory`` with properties ``{id, workspace_id, memory_type,
    memory_subtype, status}``
  * edge label ``ASSOC`` with properties ``{assoc_id, workspace_id,
    relationship, strength}``; stored directed (source→target), queried
    undirected (``MATCH (m)-[e]-(n)``) for NetworkX ``nx.Graph`` parity.

Public API
----------
``GRAPH_NAME`` : str
    Fixed graph name — the only string interpolated into SQL, and it is a
    module constant, never caller-supplied.
``validate_id(value)`` : raises ``ValueError`` for unsafe values (second
    layer of defence; all callers should also use parameterized queries).
``batch_merge_vertices_sql()`` / ``batch_merge_edges_sql()`` :
    Return the full SQL string (with ``$1`` placeholder) for UNWIND-MERGE.
``delete_workspace_subgraph_sql()`` : parameterized scope-DELETE queries.
``extract_nodes_sql()`` / ``extract_edges_sql()`` :
    Parameterized extraction queries for the scoped node/edge sets.
``wrap_cypher(body, columns)`` :
    Used ONLY for the fixed-constant bootstrap queries (no user input). Do
    NOT use for any query that touches caller-supplied values.

Callers pass the parameter object as ``json.dumps({...})`` to asyncpg's
``execute``/``fetch`` as the ``$1`` positional argument.
"""

from __future__ import annotations

import json
import math
import re

GRAPH_NAME = "memorylayer"

# Label / property names — single source of truth for the AGE schema.
_MEMORY_LABEL = "Memory"
_ASSOC_LABEL = "ASSOC"

# Entity-layer labels/edges (graph-moat C1 — ENTERPRISE rich tier). The entity
# registry materializes onto the SAME ``memorylayer`` graph so an Entity vertex
# and the Memory vertices it MENTIONS share one graph and the SAME vertex ids:
# Entity↔Memory↔Memory(assoc) traversals compose.
#
# HAS_ALIAS shape decision (FLAGGED): aliases are stored as a LIST PROPERTY on
# the Entity vertex (``aliases``), NOT as separate alias vertices joined by a
# ``HAS_ALIAS`` edge. Rationale: an alias is a surface-form attribute of one
# entity, not a first-class graph node that participates in traversals. Modeling
# each alias as its own degree-1 vertex would inject thousands of leaf nodes that
# (a) pollute Memory↔Memory community/centrality analytics on the shared graph and
# (b) are never a traversal target. As a vertex property the alias set is still
# fully queryable (C2 surfaces it) and the graph stays clean. The ``HAS_ALIAS``
# name is retained ONLY as the conceptual relationship documented here; no edge
# label is emitted.
_ENTITY_LABEL = "Entity"
_MENTIONS_LABEL = "MENTIONS"

# Fragment-layer labels/edges (graph-moat P4.5 — ENTERPRISE rich tier). The fact
# channel stores decomposed facts as memories with ``subtype="fact"`` +
# ``metadata["source_id"]=<parent memory id>``. P4.5 materializes each such fact
# memory as a ``Fragment`` vertex on the SAME ``memorylayer`` graph and a
# ``DERIVED_FROM`` edge Fragment->source ``Memory`` (keyed on the parent
# memory_id), so Memory↔Fragment↔Entity traversals compose (a Fragment shares the
# shared graph with the Memory it was derived from and the Entity that MENTIONS
# that Memory). Fragment ``content`` is carried as a property VALUE (never cypher
# syntax), exactly like the Memory ``memory_type`` / Entity ``label`` properties.
_FRAGMENT_LABEL = "Fragment"
_DERIVED_FROM_LABEL = "DERIVED_FROM"

# Upper bound on variable-length traversal depth / shortest-path hops. AGE
# requires LITERAL integer bounds inside a ``[:ASSOC*1..N]`` range (they cannot
# be passed as a ``$param``), so callers validate the requested depth as a
# bounded int via ``validate_hops`` and the CHECKED int literal is inlined into
# the cypher body. This is the ONLY non-``$1`` value ever placed in a query
# body, and it is a range-checked integer, never a string. Keeping the cap
# small also bounds AGE's path-enumeration cost.
MAX_TRAVERSAL_HOPS = 6

# Strict allowlist for ids that flow into cypher (workspace_id, memory_id,
# assoc_id). Allows UUID hex, slug chars, and the test-fixture prefixes.
# Relationship types follow the same pattern (snake_case identifiers).
_ID_RE = re.compile(r'^[A-Za-z0-9_\-]{1,256}$')
_REL_RE = re.compile(r'^[A-Za-z0-9_\-]{1,128}$')


def validate_id(value: str, *, kind: str = "id") -> str:
    """Validate and return ``value`` if it matches the strict id charset.

    Raises ``ValueError`` for any value that does not match, preventing
    attacker-controlled strings from reaching cypher even if the parameterized
    path were somehow bypassed. Call on workspace_id / memory_id / assoc_id
    before constructing any query object.
    """
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValueError(
            "Invalid %s %r — must match [A-Za-z0-9_-]{1,256}. "
            "Refusing to issue cypher with this value." % (kind, value)
        )
    return value


def validate_relationship(value: str) -> str:
    """Validate a relationship type string."""
    if not isinstance(value, str) or not _REL_RE.match(value):
        raise ValueError(
            "Invalid relationship type %r — must match [A-Za-z0-9_-]{1,128}." % value
        )
    return value


def validate_hops(value: int, *, kind: str = "depth") -> int:
    """Validate a traversal depth / hop count as a bounded int.

    Returns the int if ``1 <= value <= MAX_TRAVERSAL_HOPS``; raises
    ``ValueError`` otherwise. The validated int is the ONLY value inlined into a
    cypher body (AGE var-length ranges require literal bounds and reject
    ``$param`` bounds). Because it is a range-checked ``int`` (not a string),
    inlining it cannot inject cypher.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Invalid %s %r — must be an int." % (kind, value))
    if value < 1 or value > MAX_TRAVERSAL_HOPS:
        raise ValueError(
            "Invalid %s %r — must be 1..%d." % (kind, value, MAX_TRAVERSAL_HOPS)
        )
    return value


def _safe_strength(value: float | int) -> float:
    """Return ``value`` as float, rejecting NaN/Inf which are invalid in cypher."""
    f = float(value)
    if not math.isfinite(f):
        raise ValueError("strength must be finite, got %r" % value)
    return f


# ---------------------------------------------------------------------------
# Parameterized SQL builders — NO user values in the SQL text
# ---------------------------------------------------------------------------

def batch_merge_vertices_sql() -> str:
    """SQL for UNWIND-MERGE of a batch of Memory vertices.

    Caller passes ``$1 = json.dumps({"nodes": [{"id":..., "ws":...,
    "status":..., "mt":..., "ms":...}, ...]})`` as the asyncpg bind arg.

    AGE 1.7 supports the 3-arg ``cypher(graph, $$..$$, $1)`` parameterized
    form where ``$1`` is a SQL bind parameter holding an agtype-compatible
    JSON object. Keys are referenced as ``$key`` inside the cypher body.

    The MERGE is keyed on ``{id, workspace_id}`` (relational identity).
    SET refreshes all mutable properties so re-runs after archival converge.
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " UNWIND $nodes AS row"
        " MERGE (m:%s {id: row.id, workspace_id: row.ws})"
        " SET m.status = row.status,"
        "     m.memory_type = row.mt,"
        "     m.memory_subtype = row.ms"
        " RETURN m.id"
        "$$, $1) AS (id agtype)" % (GRAPH_NAME, _MEMORY_LABEL)
    )


def batch_merge_edges_sql() -> str:
    """SQL for UNWIND-MERGE of a batch of ASSOC edges.

    Caller passes ``$1 = json.dumps({"edges": [{"aid":..., "src":...,
    "tgt":..., "ws":..., "rel":..., "strength":...}, ...]})`` as the
    asyncpg bind arg. Endpoints must already exist as Memory vertices.
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " UNWIND $edges AS row"
        " MATCH (s:%s {id: row.src, workspace_id: row.ws}),"
        "       (t:%s {id: row.tgt, workspace_id: row.ws})"
        " MERGE (s)-[e:%s {assoc_id: row.aid, workspace_id: row.ws}]->(t)"
        " SET e.relationship = row.rel, e.strength = row.strength"
        " RETURN e.assoc_id"
        "$$, $1) AS (assoc_id agtype)"
        % (GRAPH_NAME, _MEMORY_LABEL, _MEMORY_LABEL, _ASSOC_LABEL)
    )


def delete_workspace_edges_sql() -> str:
    """SQL to delete all ASSOC edges for a workspace (parameterized).

    Caller passes ``$1 = json.dumps({"ws": workspace_id})``.
    Run this BEFORE ``delete_workspace_vertices_sql`` (FK order).
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (m:%s {workspace_id: $ws})-[e:%s]->(n)"
        " DELETE e"
        "$$, $1) AS (x agtype)" % (GRAPH_NAME, _MEMORY_LABEL, _ASSOC_LABEL)
    )


def delete_workspace_vertices_sql() -> str:
    """SQL to DETACH DELETE all Memory vertices for a workspace (parameterized).

    Caller passes ``$1 = json.dumps({"ws": workspace_id})``.
    DETACH DELETE removes any remaining edges too (safety net).
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (m:%s {workspace_id: $ws})"
        " DETACH DELETE m"
        "$$, $1) AS (x agtype)" % (GRAPH_NAME, _MEMORY_LABEL)
    )


def probe_workspace_subgraph_sql() -> str:
    """SQL to cheaply check whether any Memory vertex exists for a workspace.

    Used by ``materialize_workspace_subgraph_gated`` as a defense-in-depth guard:
    before skipping the delete-then-MERGE on a watermark match, confirm the AGE
    subgraph is actually present (the marker table and the AGE graph are separate
    stores; an out-of-band drop/restore of the AGE graph would leave the marker row
    intact but the subgraph empty, causing incorrect empty results if skipped).

    Caller passes ``$1 = json.dumps({"ws": workspace_id})``.
    Returns at most one row ``(id,)``; any non-empty result confirms presence.
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (m:%s {workspace_id: $ws})"
        " RETURN m.id LIMIT 1"
        "$$, $1) AS (id agtype)"
        % (GRAPH_NAME, _MEMORY_LABEL)
    )


def extract_nodes_sql() -> str:
    """SQL to extract the active node set for a workspace (parameterized).

    Caller passes ``$1 = json.dumps({"ws": workspace_id})``.
    Returns columns ``(id, memory_type, memory_subtype)`` as agtype strings.
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (m:%s {workspace_id: $ws})"
        " WHERE m.status = 'active'"
        " RETURN m.id, m.memory_type, m.memory_subtype"
        "$$, $1) AS (id agtype, memory_type agtype, memory_subtype agtype)"
        % (GRAPH_NAME, _MEMORY_LABEL)
    )


def extract_edges_sql() -> str:
    """SQL to extract the undirected active edge list for a workspace (parameterized).

    Caller passes ``$1 = json.dumps({"ws": workspace_id})``.
    Returns ``(src, tgt, relationship, strength)`` as agtype strings.

    AGE returns BOTH directions for the undirected ``-[e]-`` match. The
    caller sorts rows by ``(src, tgt, relationship)`` and calls
    ``nx.Graph.add_edge`` for every row — ``nx.Graph`` collapses the two
    directions naturally (last write wins for attrs), matching the OSS
    ``get_associations_batch`` semantics.
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (m:%s {workspace_id: $ws})-[e:%s]-(n:%s {workspace_id: $ws})"
        " WHERE m.status = 'active' AND n.status = 'active'"
        " RETURN m.id, n.id, e.relationship, e.strength"
        "$$, $1) AS (src agtype, tgt agtype, relationship agtype, strength agtype)"
        % (GRAPH_NAME, _MEMORY_LABEL, _ASSOC_LABEL, _MEMORY_LABEL)
    )


# ---------------------------------------------------------------------------
# Entity-layer write builders (graph-moat C1 — ENTERPRISE rich tier)
# ---------------------------------------------------------------------------
#
# Same parameterized + validated discipline as the Memory/ASSOC builders above:
# EVERY caller value (workspace_id, entity_id, memory_id, normalized_name,
# entity_type, label, aliases, role) travels in the ``$1`` JSON bind arg
# (referenced as ``$key`` inside the cypher body) — NEVER inlined. The strict
# ``validate_id`` allowlist is the second layer of defence for the ids; the
# free-text label/normalized_name/aliases are safe because they are passed
# ONLY as agtype bind-param VALUES (never as cypher syntax), exactly like the
# Memory ``memory_type``/``memory_subtype`` properties.


def batch_merge_entities_sql() -> str:
    """SQL for UNWIND-MERGE of a batch of Entity vertices.

    Caller passes ``$1 = json.dumps({"entities": [{"id":..., "ws":...,
    "etype":..., "norm":..., "label":..., "aliases":[...]}, ...]})``.

    MERGE keyed on ``{id, workspace_id}`` (relational identity). SET refreshes
    all mutable properties so re-runs converge. ``aliases`` is stored as a LIST
    PROPERTY on the vertex (the HAS_ALIAS shape decision — see module header),
    not as separate alias vertices/edges.
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " UNWIND $entities AS row"
        " MERGE (e:%s {id: row.id, workspace_id: row.ws})"
        " SET e.entity_type = row.etype,"
        "     e.normalized_name = row.norm,"
        "     e.label = row.label,"
        "     e.aliases = row.aliases"
        " RETURN e.id"
        "$$, $1) AS (id agtype)" % (GRAPH_NAME, _ENTITY_LABEL)
    )


def batch_merge_mentions_sql() -> str:
    """SQL for UNWIND-MERGE of a batch of MENTIONS edges (Entity -> Memory).

    Caller passes ``$1 = json.dumps({"mentions": [{"eid":..., "mid":...,
    "ws":..., "role":...}, ...]})``. Both endpoints must already exist (Entity
    vertices merged first, Memory vertices materialized by the subgraph pass).
    Edges whose Memory endpoint is absent are simply not matched (no-op),
    mirroring the "both endpoints in graph" guard on ASSOC edges.

    The MERGE is keyed on ``{entity_id, memory_id, role, workspace_id}`` so a
    self-mention and a third-party mention of the same memory are distinct
    edges; ``role`` is also carried as a property for direct read.
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " UNWIND $mentions AS row"
        " MATCH (e:%s {id: row.eid, workspace_id: row.ws}),"
        "       (m:%s {id: row.mid, workspace_id: row.ws})"
        " MERGE (e)-[r:%s {entity_id: row.eid, memory_id: row.mid,"
        "                  role: row.role, workspace_id: row.ws}]->(m)"
        " RETURN r.entity_id"
        "$$, $1) AS (entity_id agtype)"
        % (GRAPH_NAME, _ENTITY_LABEL, _MEMORY_LABEL, _MENTIONS_LABEL)
    )


def delete_workspace_entity_mentions_sql() -> str:
    """SQL to delete all MENTIONS edges for a workspace (parameterized).

    Caller passes ``$1 = json.dumps({"ws": workspace_id})``.
    Run BEFORE ``delete_workspace_entities_sql`` (FK-like order).
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (e:%s {workspace_id: $ws})-[r:%s]->(m)"
        " DELETE r"
        "$$, $1) AS (x agtype)" % (GRAPH_NAME, _ENTITY_LABEL, _MENTIONS_LABEL)
    )


def delete_workspace_entities_sql() -> str:
    """SQL to DETACH DELETE all Entity vertices for a workspace (parameterized).

    Caller passes ``$1 = json.dumps({"ws": workspace_id})``. DETACH DELETE
    removes any remaining MENTIONS edges too (safety net). Memory vertices are
    untouched — only the Entity layer is scope-deleted, so an entity-only
    re-materialize never disturbs the Memory/ASSOC subgraph.
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (e:%s {workspace_id: $ws})"
        " DETACH DELETE e"
        "$$, $1) AS (x agtype)" % (GRAPH_NAME, _ENTITY_LABEL)
    )


def probe_workspace_entities_sql() -> str:
    """SQL to cheaply check whether any Entity vertex exists for a workspace.

    Caller passes ``$1 = json.dumps({"ws": workspace_id})``. Returns at most one
    row; any non-empty result confirms presence. Used by the watermark-gated
    entity materialize as a defense-in-depth presence probe (same role as
    ``probe_workspace_subgraph_sql`` for the Memory layer).
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (e:%s {workspace_id: $ws})"
        " RETURN e.id LIMIT 1"
        "$$, $1) AS (id agtype)" % (GRAPH_NAME, _ENTITY_LABEL)
    )


# ---------------------------------------------------------------------------
# Entity-layer read builders (graph-moat C2 — ENTERPRISE rich tier)
# ---------------------------------------------------------------------------
#
# Read-only MATCH/RETURN over the C1 Entity/MENTIONS vertices+edges. Same
# parameterization discipline: caller values travel in ``$1`` ($ws, $eid).


def entity_mentioned_memories_sql() -> str:
    """SQL returning the memories an Entity MENTIONS (Entity -> Memory).

    Caller passes ``$1 = json.dumps({"ws":..., "eid":...})``. Returns
    ``(memory_id, role)`` for each active memory the entity mentions (1 hop).

    ORDER BY m.id, r.role guarantees a deterministic row order that matches
    the OSS relational ``list_workspace_entity_members`` sort
    ``(entity_id, memory_id, role)`` — the two backends must agree on which
    role wins when the same memory is mentioned under multiple roles (the
    ``setdefault`` first-wins dedup in the callers is order-sensitive).
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (e:%s {id: $eid, workspace_id: $ws})-[r:%s]->(m:%s {workspace_id: $ws})"
        " WHERE m.status = 'active'"
        " RETURN m.id, r.role"
        " ORDER BY m.id, r.role"
        "$$, $1) AS (memory_id agtype, role agtype)"
        % (GRAPH_NAME, _ENTITY_LABEL, _MENTIONS_LABEL, _MEMORY_LABEL)
    )


def entity_co_mentioned_entities_sql() -> str:
    """SQL returning entities co-mentioned with a root entity (1 shared memory).

    Caller passes ``$1 = json.dumps({"ws":..., "eid":...})``. Traverses
    ``Entity -> MENTIONS -> Memory <- MENTIONS <- Entity`` and returns every
    OTHER entity that shares a mentioned (active) memory, with the shared memory
    id (so the caller can rank by co-mention count). The root entity itself is
    excluded via ``co.id <> $eid``.

    Returns ``(entity_id, label, entity_type, shared_memory_id)`` per
    (co-entity, shared-memory) pair; the caller aggregates by entity id.
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (e:%s {id: $eid, workspace_id: $ws})-[:%s]->(m:%s {workspace_id: $ws})"
        "<-[:%s]-(co:%s {workspace_id: $ws})"
        " WHERE m.status = 'active' AND co.id <> $eid"
        " RETURN co.id, co.label, co.entity_type, m.id"
        "$$, $1) AS (entity_id agtype, label agtype, entity_type agtype, shared_memory_id agtype)"
        % (GRAPH_NAME, _ENTITY_LABEL, _MENTIONS_LABEL, _MEMORY_LABEL, _MENTIONS_LABEL, _ENTITY_LABEL)
    )


# ---------------------------------------------------------------------------
# Fragment-layer write builders (graph-moat P4.5 — ENTERPRISE rich tier)
# ---------------------------------------------------------------------------
#
# Same parameterized + validated discipline as the Memory/ASSOC/Entity builders
# above: EVERY caller value (workspace_id, fragment_id, source_id, content)
# travels in the ``$1`` JSON bind arg (referenced as ``$key`` inside the cypher
# body) — NEVER inlined. The strict ``validate_id`` allowlist is the second layer
# of defence for the ids; the free-text ``content`` is safe because it is passed
# ONLY as an agtype bind-param VALUE (never as cypher syntax), exactly like the
# Memory ``memory_type`` / Entity ``label`` properties.


def batch_merge_fragments_sql() -> str:
    """SQL for UNWIND-MERGE of a batch of Fragment vertices.

    Caller passes ``$1 = json.dumps({"fragments": [{"id":..., "ws":...,
    "content":..., "source_id":...}, ...]})``.

    MERGE keyed on ``{id, workspace_id}`` (relational identity — the fact
    memory's id). SET refreshes all mutable properties so re-runs converge.
    ``source_id`` is carried as a property for direct read (it also keys the
    DERIVED_FROM edge below).
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " UNWIND $fragments AS row"
        " MERGE (f:%s {id: row.id, workspace_id: row.ws})"
        " SET f.content = row.content,"
        "     f.source_id = row.source_id"
        " RETURN f.id"
        "$$, $1) AS (id agtype)" % (GRAPH_NAME, _FRAGMENT_LABEL)
    )


def batch_merge_derived_from_sql() -> str:
    """SQL for UNWIND-MERGE of a batch of DERIVED_FROM edges (Fragment -> Memory).

    Caller passes ``$1 = json.dumps({"edges": [{"fid":..., "sid":...,
    "ws":...}, ...]})``. Both endpoints must already exist (Fragment vertices
    merged first; the source Memory vertex materialized by the subgraph pass).
    A DERIVED_FROM edge whose source Memory endpoint is absent is simply not
    matched (no-op), mirroring the "both endpoints in graph" guard on ASSOC /
    MENTIONS edges.

    The MERGE is keyed on ``{fragment_id, source_id, workspace_id}`` so re-runs
    converge to one edge per (fragment, source) pair.
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " UNWIND $edges AS row"
        " MATCH (f:%s {id: row.fid, workspace_id: row.ws}),"
        "       (m:%s {id: row.sid, workspace_id: row.ws})"
        " MERGE (f)-[d:%s {fragment_id: row.fid, source_id: row.sid,"
        "                  workspace_id: row.ws}]->(m)"
        " RETURN d.fragment_id"
        "$$, $1) AS (fragment_id agtype)"
        % (GRAPH_NAME, _FRAGMENT_LABEL, _MEMORY_LABEL, _DERIVED_FROM_LABEL)
    )


def delete_workspace_derived_from_sql() -> str:
    """SQL to delete all DERIVED_FROM edges for a workspace (parameterized).

    Caller passes ``$1 = json.dumps({"ws": workspace_id})``.
    Run BEFORE ``delete_workspace_fragments_sql`` (FK-like order).
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (f:%s {workspace_id: $ws})-[d:%s]->(m)"
        " DELETE d"
        "$$, $1) AS (x agtype)" % (GRAPH_NAME, _FRAGMENT_LABEL, _DERIVED_FROM_LABEL)
    )


def delete_workspace_fragments_sql() -> str:
    """SQL to DETACH DELETE all Fragment vertices for a workspace (parameterized).

    Caller passes ``$1 = json.dumps({"ws": workspace_id})``. DETACH DELETE
    removes any remaining DERIVED_FROM edges too (safety net). Memory and Entity
    vertices are untouched — only the Fragment layer is scope-deleted, so a
    fragment-only re-materialize never disturbs the Memory/ASSOC or Entity
    subgraphs.
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (f:%s {workspace_id: $ws})"
        " DETACH DELETE f"
        "$$, $1) AS (x agtype)" % (GRAPH_NAME, _FRAGMENT_LABEL)
    )


def probe_workspace_fragments_sql() -> str:
    """SQL to cheaply check whether any Fragment vertex exists for a workspace.

    Caller passes ``$1 = json.dumps({"ws": workspace_id})``. Returns at most one
    row; any non-empty result confirms presence. Used by the watermark-gated
    fragment materialize as a defense-in-depth presence probe (same role as
    ``probe_workspace_entities_sql`` for the Entity layer).
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (f:%s {workspace_id: $ws})"
        " RETURN f.id LIMIT 1"
        "$$, $1) AS (id agtype)" % (GRAPH_NAME, _FRAGMENT_LABEL)
    )


# ---------------------------------------------------------------------------
# Fragment-layer read builder (graph-moat P4.5 — ENTERPRISE rich tier)
# ---------------------------------------------------------------------------
#
# Read-only MATCH/RETURN over the P4.5 Fragment/DERIVED_FROM vertices+edges. Same
# parameterization discipline: caller values travel in ``$1`` ($ws, $mid).


def fragments_for_memory_sql() -> str:
    """SQL returning the Fragments DERIVED_FROM a source Memory (1 hop).

    Caller passes ``$1 = json.dumps({"ws":..., "mid":...})``. Traverses
    ``Fragment -> DERIVED_FROM -> Memory`` and returns each fragment derived from
    ``$mid``. The source Memory must be active (status guard mirrors the other
    read builders).

    ORDER BY f.id guarantees a deterministic row order that matches the OSS
    relational fallback's ``sorted(..., key=fragment_id)`` so the two backends
    agree on the truncation boundary.

    Returns ``(fragment_id, content)`` per derived fragment.
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (f:%s {workspace_id: $ws})-[:%s]->(m:%s {id: $mid, workspace_id: $ws})"
        " WHERE m.status = 'active'"
        " RETURN f.id, f.content"
        " ORDER BY f.id"
        "$$, $1) AS (fragment_id agtype, content agtype)"
        % (GRAPH_NAME, _FRAGMENT_LABEL, _DERIVED_FROM_LABEL, _MEMORY_LABEL)
    )


# ---------------------------------------------------------------------------
# Read-only parameterized builders for the graph-QUERY backend (P2 Track A)
# ---------------------------------------------------------------------------
#
# All of these take the SAME parameterized + validated approach as the
# materialize builders above: every caller value travels in the ``$1`` JSON
# bind arg (referenced as ``$key`` inside the cypher body) — NEVER inlined.
# The ONLY exception is the variable-length range bound (``*1..N``), which AGE
# requires to be a LITERAL int; callers pass it through ``validate_hops`` (a
# bounded int, never a string) and the builders inline that checked int.
#
# These are READ-ONLY: they MATCH/RETURN only and never mutate the graph. They
# assume the workspace subgraph was just materialized (materialize-on-read) so
# they see current relational truth.
#
# AGE-1.7 limitation handling (the reason traversal builders return PATHS):
#   * AGE 1.7 has NO ``shortestPath()`` function.
#   * AGE 1.7 does NOT support list comprehensions (``[x IN ... | x.id]``) or
#     the ``ALL(... WHERE ...)`` path predicate (both error out).
#   * It DOES support ``nodes(p)``, ``relationships(p)``, ``length(p)``, bounded
#     var-length ``[:ASSOC*1..N]``, and ``ORDER BY length(p)``.
# So the traversal builders return the raw ``nodes(p)`` / ``relationships(p)``
# agtype path elements; the relationship-type FILTER is applied in PYTHON over
# the parsed path edges (admit a path only if every edge matches the allowlist)
# — this is exactly the relational per-hop filter semantics, so the backends
# agree. See ``parse_path_vertices`` / ``parse_path_edges`` and
# ``AgeGraphQueryService``.


def neighbors_paths_sql(depth: int, direction: str) -> str:
    """SQL returning bounded var-length PATHS from a single root.

    Caller passes ``$1 = json.dumps({"ws":..., "root":...})``. ``depth`` is a
    validated int literal (``validate_hops``); ``direction`` selects the arrow.

    Returns ``(vertices agtype, edges agtype)`` per matched path: the
    ``nodes(p)`` and ``relationships(p)`` element lists. The caller parses
    these, applies the optional relationship filter per-path in Python, and
    unions the reachable node set. (List comprehensions / ALL() are unsupported
    in AGE 1.7, hence returning whole path elements.)
    """
    arrow_l, arrow_r = _direction_arrows(direction)
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH p = (m:%s {id: $root, workspace_id: $ws})"
        "%s[:%s*1..%d]%s(n:%s {workspace_id: $ws})"
        " WHERE m.status = 'active' AND n.status = 'active'"
        " RETURN nodes(p), relationships(p)"
        "$$, $1) AS (vertices agtype, edges agtype)"
        % (GRAPH_NAME, _MEMORY_LABEL, arrow_l, _ASSOC_LABEL, depth, arrow_r, _MEMORY_LABEL)
    )


def k_hop_paths_sql(depth: int, direction: str) -> str:
    """SQL returning bounded var-length PATHS from ANY of a set of roots.

    Caller passes ``$1 = json.dumps({"ws":..., "roots":[...]})``. Same shape as
    ``neighbors_paths_sql`` but seeds from ``UNWIND $roots``. Returns
    ``(vertices agtype, edges agtype)`` per path.
    """
    arrow_l, arrow_r = _direction_arrows(direction)
    return (
        "SELECT * FROM cypher('%s', $$"
        " UNWIND $roots AS root_id"
        " MATCH p = (m:%s {id: root_id, workspace_id: $ws})"
        "%s[:%s*1..%d]%s(n:%s {workspace_id: $ws})"
        " WHERE m.status = 'active' AND n.status = 'active'"
        " RETURN nodes(p), relationships(p)"
        "$$, $1) AS (vertices agtype, edges agtype)"
        % (GRAPH_NAME, _MEMORY_LABEL, arrow_l, _ASSOC_LABEL, depth, arrow_r, _MEMORY_LABEL)
    )


def induced_edges_sql(*, with_rel_filter: bool) -> str:
    """SQL returning all edges whose BOTH endpoints are in a given id set.

    Caller passes ``$1 = json.dumps({"ws":..., "ids":[...], "rels":[...]})``.
    Mirrors the relational "induced subgraph over the visited node set" phase.
    Returns ``(src, tgt, relationship, strength)`` for every ASSOC edge between
    two ids in ``$ids`` (both active). Direction is irrelevant here: we return
    the stored orientation and the caller dedups by assoc identity.

    The relationship filter here IS a single-edge ``WHERE e.relationship IN
    $rels`` (a plain bind list on a single ``MATCH`` edge — fully supported by
    AGE 1.7, unlike the path-level ALL() predicate).
    """
    rel_clause = _rel_where_clause(with_rel_filter, var="e", on_path=False)
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (m:%s {workspace_id: $ws})-[e:%s]->(n:%s {workspace_id: $ws})"
        " WHERE m.status = 'active' AND n.status = 'active'"
        "   AND m.id IN $ids AND n.id IN $ids"
        "%s"
        " RETURN m.id, n.id, e.relationship, e.strength"
        "$$, $1) AS (src agtype, tgt agtype, relationship agtype, strength agtype)"
        % (GRAPH_NAME, _MEMORY_LABEL, _ASSOC_LABEL, _MEMORY_LABEL, rel_clause)
    )


def shortest_path_sql(max_hops: int) -> str:
    """SQL returning bounded var-length PATHS between src and dst (undirected).

    Caller passes ``$1 = json.dumps({"ws":..., "src":..., "dst":...})``.
    ``max_hops`` is a validated int literal.

    AGE 1.7 has NO built-in ``shortestPath()``, list comprehensions, or the
    ``ALL()`` path predicate, so we cannot push a relationship-list filter into
    the cypher. Instead we enumerate bounded var-length paths up to
    ``[:ASSOC*1..max_hops]`` ordered by length ASC, return up to LIMIT=1000
    candidates, and the caller applies the optional relationship filter in Python
    over the parsed path edges. LIMIT=1000 bounds server-side enumeration cost;
    in pathological cases with a relationship filter active across more than 1000
    equal-length non-matching paths, the caller may not find a matching path even
    though one exists — this boundary is documented in
    ``AgeGraphQueryService.shortest_path``.

    Tie-break: for equal-length paths the caller selects the one whose
    node-id sequence is lexicographically smallest (same tie-break as the OSS
    BFS which expands sorted frontiers).

    Returns ``(vertices agtype, edges agtype, hops agtype)`` per path.
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH p = (m:%s {id: $src, workspace_id: $ws})"
        "-[:%s*1..%d]-(n:%s {id: $dst, workspace_id: $ws})"
        " WHERE m.status = 'active' AND n.status = 'active'"
        " RETURN nodes(p), relationships(p), length(p) AS hops"
        " ORDER BY hops ASC"
        " LIMIT 1000"
        "$$, $1) AS (vertices agtype, edges agtype, hops agtype)"
        % (GRAPH_NAME, _MEMORY_LABEL, _ASSOC_LABEL, max_hops, _MEMORY_LABEL)
    )


def typed_pattern_sql() -> str:
    """SQL returning all edges of a single relationship type (parameterized).

    Caller passes ``$1 = json.dumps({"ws":..., "rel":...})``. Returns
    ``(src, tgt, relationship, strength)`` for ASSOC edges whose
    ``relationship == $rel`` between two active nodes. The caller applies the
    ``limit`` after a deterministic sort (parity with the relational impl).
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (m:%s {workspace_id: $ws})-[e:%s {relationship: $rel}]->(n:%s {workspace_id: $ws})"
        " WHERE m.status = 'active' AND n.status = 'active'"
        " RETURN m.id, n.id, e.relationship, e.strength"
        "$$, $1) AS (src agtype, tgt agtype, relationship agtype, strength agtype)"
        % (GRAPH_NAME, _MEMORY_LABEL, _ASSOC_LABEL, _MEMORY_LABEL)
    )


def relationship_rollup_sql() -> str:
    """SQL returning per-relationship edge counts for a workspace (parameterized).

    Caller passes ``$1 = json.dumps({"ws": workspace_id})``. Returns
    ``(relationship, count)`` grouped by relationship over the STORED-direction
    edges (each association counted once), matching the relational
    ``GROUP BY relationship`` aggregate.
    """
    return (
        "SELECT * FROM cypher('%s', $$"
        " MATCH (m:%s {workspace_id: $ws})-[e:%s]->(n:%s {workspace_id: $ws})"
        " WHERE m.status = 'active' AND n.status = 'active'"
        " RETURN e.relationship, count(e)"
        "$$, $1) AS (relationship agtype, cnt agtype)"
        % (GRAPH_NAME, _MEMORY_LABEL, _ASSOC_LABEL, _MEMORY_LABEL)
    )


def _direction_arrows(direction: str) -> tuple[str, str]:
    """Return the (left, right) cypher arrow fragments for a direction.

    ``both`` -> ``-`` / ``-`` (undirected); ``outgoing`` -> ``-`` / ``->``;
    ``incoming`` -> ``<-`` / ``-``. Rejects unknown directions.
    """
    if direction == "both":
        return "-", "-"
    if direction == "outgoing":
        return "-", "->"
    if direction == "incoming":
        return "<-", "-"
    raise ValueError("Invalid direction %r — must be both/outgoing/incoming." % direction)


def _rel_where_clause(with_rel_filter: bool, *, var: str, on_path: bool = False) -> str:
    """Build the optional single-edge relationship-filter WHERE fragment.

    Returns ``""`` when no filter, else ``AND <var>.relationship IN $rels`` (a
    plain bind-list membership test on a single ``MATCH`` edge, fully supported
    by AGE 1.7). Path-level (``on_path``) filtering is NOT emitted here — AGE
    1.7 lacks the ``ALL()`` predicate, so traversal builders return whole paths
    and the caller filters edges in Python (see module header).
    """
    if not with_rel_filter or on_path:
        return ""
    return " AND %s.relationship IN $rels" % var


# ---------------------------------------------------------------------------
# agtype path-element parsing (AGE 1.7 returns ``::vertex`` / ``::edge`` arrays)
# ---------------------------------------------------------------------------
#
# ``nodes(p)`` / ``relationships(p)`` come back as an agtype JSON array whose
# elements carry a trailing ``::vertex`` / ``::edge`` type annotation, e.g.::
#
#   [{"id": 844..., "label": "Memory", "properties": {"id": "a", ...}}::vertex]
#
# These helpers strip the annotations and return the relational ``properties``
# we care about (the memory ``id`` for vertices; ``relationship`` + ``strength``
# + ``start_id``/``end_id`` for edges).

_AGTYPE_ANNOTATION_RE = re.compile(r"::(?:vertex|edge|path)\b")


def _strip_agtype_annotations(agtype_text: str) -> str:
    """Remove ``::vertex`` / ``::edge`` / ``::path`` annotations so the value is plain JSON."""
    return _AGTYPE_ANNOTATION_RE.sub("", agtype_text)


def parse_path_vertices(agtype_text: str) -> list[dict]:
    """Parse a ``nodes(p)`` agtype array into a list of vertex ``properties`` dicts."""
    raw = json.loads(_strip_agtype_annotations(agtype_text))
    return [elem.get("properties", {}) for elem in raw]


def parse_path_edges(agtype_text: str) -> list[dict]:
    """Parse a ``relationships(p)`` agtype array into edge descriptors.

    Each returned dict has ``start_id`` / ``end_id`` (AGE internal vertex ids,
    used only to orient an edge within a path), plus the relational
    ``properties`` (``relationship``, ``strength``, ...).
    """
    raw = json.loads(_strip_agtype_annotations(agtype_text))
    out = []
    for elem in raw:
        props = elem.get("properties", {})
        out.append({
            "start_id": elem.get("start_id"),
            "end_id": elem.get("end_id"),
            "relationship": props.get("relationship"),
            "strength": props.get("strength"),
        })
    return out


def parse_path_vertex_internal_ids(agtype_text: str) -> list[int]:
    """Parse a ``nodes(p)`` agtype array into the AGE internal vertex ids (ordered)."""
    raw = json.loads(_strip_agtype_annotations(agtype_text))
    return [elem.get("id") for elem in raw]


def wrap_cypher(cypher_body: str, *, columns: str) -> str:
    """Wrap a CONSTANT cypher body in the AGE SQL form.

    FOR BOOTSTRAP/INTERNAL USE ONLY — the body must not contain any
    caller-supplied values. All user-data queries use the parameterized
    ``$1`` builders above. The graph name is the fixed ``GRAPH_NAME``
    constant, never caller-supplied.
    """
    return "SELECT * FROM cypher('%s', $$ %s $$) AS (%s)" % (
        GRAPH_NAME, cypher_body, columns
    )
