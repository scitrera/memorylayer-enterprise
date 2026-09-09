"""Shared Apache-AGE workspace-subgraph materialization (ENTERPRISE-only).

This module holds the delete-then-MERGE half of the materialize step that was
originally inlined in ``age.AgeGraphAnalysisService._materialize_and_extract``.
It is the shared dependency of BOTH the P2.1 graph-ANALYSIS backend (which
materializes, then extracts the full node/edge set for NetworkX) and the P2
Track-A graph-QUERY backend (which materializes-on-read, then runs a read-only
cypher query against the freshly materialized subgraph).

Why a shared helper
-------------------
The materialize logic is the security- and correctness-critical part: it must
stay byte-for-byte the same scope-DELETE-then-MERGE flow that the P2.1
conformance suite locks in. Both backends call the SAME function so a single
fix (e.g. the ``$$``-injection BLOCKER fix) applies everywhere and the two
backends cannot drift.

Security: parameterized cypher only
------------------------------------
Every caller-supplied value (workspace_id, memory_id, assoc_id, relationship,
strength) flows through the ``$1`` JSON bind parameter of
``cypher(graph, $$..$$, $1)`` — NEVER inlined into SQL/cypher text. A strict
charset allowlist in ``_cypher.validate_id`` is the second line of defence.
``materialize_workspace_subgraph`` assumes the caller has already validated
``workspace_id`` (the analysis/query services validate before calling).

MAJOR-1 (staleness): scope-DELETE before MERGE
----------------------------------------------
The workspace's AGE subgraph (edges first, then vertices) is deleted before
MERGE so the materialized view always reflects current relational truth — no
phantom edges from deleted associations, no stale status from archived
memories. This matches OSS "fresh-per-call" semantics.
"""

from __future__ import annotations

import hashlib
import json
import logging

from . import _cypher

_log = logging.getLogger(__name__)
from ._cypher import (
    probe_workspace_entities_sql,
    probe_workspace_fragments_sql,
    probe_workspace_subgraph_sql,
)
from .bootstrap import ensure_age_session

# Per-workspace last-materialized watermark marker (P2 strategy-C, Gate B). A
# tiny relational table in the app schema (resolved by the AGE search_path's
# ``public`` entry) — NOT in the AGE graph itself. One row per workspace holds
# the change-watermark recorded at the last successful materialize, so a repeat
# read whose watermark is unchanged can skip the delete-then-MERGE entirely (the
# materialized subgraph persists in AGE across calls).
_WATERMARK_TABLE = "age_materialization_watermark"

# Separate per-workspace watermark marker for the ENTITY layer (graph-moat C1).
# The Entity/MENTIONS materialize gates INDEPENDENTLY of the Memory subgraph:
# entity-registry edits advance no memory/association timestamp, so the Memory
# watermark would miss them. This marker records the entity-layer watermark
# token (entity count + max entity updated_at + member count) at the last
# successful Entity materialize so an entity-only change still triggers a
# re-materialize while an unchanged registry is skipped.
_ENTITY_WATERMARK_TABLE = "age_entity_materialization_watermark"

_CREATE_WATERMARK_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS %s ("
    "workspace_id text PRIMARY KEY, "
    "watermark jsonb NOT NULL, "
    "materialized_at timestamptz NOT NULL DEFAULT now()"
    ")" % _WATERMARK_TABLE
)

_SELECT_WATERMARK_SQL = (
    "SELECT watermark FROM %s WHERE workspace_id = $1" % _WATERMARK_TABLE
)

_UPSERT_WATERMARK_SQL = (
    "INSERT INTO %s (workspace_id, watermark, materialized_at) "
    "VALUES ($1, $2::jsonb, now()) "
    "ON CONFLICT (workspace_id) DO UPDATE SET "
    "watermark = EXCLUDED.watermark, materialized_at = now()" % _WATERMARK_TABLE
)

# Separate per-workspace watermark marker for the FRAGMENT layer (graph-moat
# P4.5). The Fragment/DERIVED_FROM materialize gates INDEPENDENTLY of the Memory
# subgraph AND the Entity layer: a fact memory is itself a memory, so the Memory
# watermark would partially move when facts are created, but a count-preserving
# REPOINT of a fact's ``source_id`` (the parent it was derived from) advances no
# memory/association timestamp the Memory watermark tracks. This marker records
# the fragment-layer watermark token (fact-memory count + max fact updated_at +
# order-insensitive digest of (fragment_id, source_id)) at the last successful
# Fragment materialize so a count-preserving source_id repoint still triggers a
# re-materialize while an unchanged fragment set is skipped.
_FRAGMENT_WATERMARK_TABLE = "age_fragment_materialization_watermark"

_CREATE_ENTITY_WATERMARK_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS %s ("
    "workspace_id text PRIMARY KEY, "
    "watermark jsonb NOT NULL, "
    "materialized_at timestamptz NOT NULL DEFAULT now()"
    ")" % _ENTITY_WATERMARK_TABLE
)

_SELECT_ENTITY_WATERMARK_SQL = (
    "SELECT watermark FROM %s WHERE workspace_id = $1" % _ENTITY_WATERMARK_TABLE
)

_UPSERT_ENTITY_WATERMARK_SQL = (
    "INSERT INTO %s (workspace_id, watermark, materialized_at) "
    "VALUES ($1, $2::jsonb, now()) "
    "ON CONFLICT (workspace_id) DO UPDATE SET "
    "watermark = EXCLUDED.watermark, materialized_at = now()" % _ENTITY_WATERMARK_TABLE
)

_CREATE_FRAGMENT_WATERMARK_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS %s ("
    "workspace_id text PRIMARY KEY, "
    "watermark jsonb NOT NULL, "
    "materialized_at timestamptz NOT NULL DEFAULT now()"
    ")" % _FRAGMENT_WATERMARK_TABLE
)

_SELECT_FRAGMENT_WATERMARK_SQL = (
    "SELECT watermark FROM %s WHERE workspace_id = $1" % _FRAGMENT_WATERMARK_TABLE
)

_UPSERT_FRAGMENT_WATERMARK_SQL = (
    "INSERT INTO %s (workspace_id, watermark, materialized_at) "
    "VALUES ($1, $2::jsonb, now()) "
    "ON CONFLICT (workspace_id) DO UPDATE SET "
    "watermark = EXCLUDED.watermark, materialized_at = now()" % _FRAGMENT_WATERMARK_TABLE
)


async def materialize_workspace_subgraph_gated(
    age_engine,
    *,
    storage,
    workspace_id: str,
    node_attrs: dict[str, dict],
    associations: list,
) -> bool:
    """Watermark-gated materialize (P2 strategy-C, Gate B).

    Skips the full delete-then-MERGE when the workspace's current change-watermark
    EXACTLY matches the watermark recorded at its last successful materialize AND a
    prior materialization marker exists (the AGE subgraph persists across calls, so
    re-running the identical MERGE is redundant). Otherwise performs the SAME full
    delete-then-MERGE via ``materialize_workspace_subgraph`` (NOT a delta-MERGE —
    that would reintroduce the P2.1 phantom-row staleness) and records the new
    watermark.

    FAIL-SAFE: any ambiguity — watermark unavailable, no prior marker, mismatch, or
    any error reading/comparing — falls through to a full re-materialize. We never
    skip and risk serving a stale subgraph.

    Args:
        storage: the storage backend exposing ``get_workspace_change_watermark``.

    Returns:
        ``True`` if a full materialize ran, ``False`` if it was safely skipped.
    """
    current_watermark = None
    try:
        wm = await storage.get_workspace_change_watermark(workspace_id)
        current_watermark = list(wm) if wm is not None else None
    except NotImplementedError:
        current_watermark = None
    except Exception:
        current_watermark = None

    # If we cannot compute a watermark, always do the work (fail-safe).
    if current_watermark is None:
        await materialize_workspace_subgraph(
            age_engine,
            workspace_id=workspace_id,
            node_attrs=node_attrs,
            associations=associations,
        )
        return True

    async with age_engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)

        # Marker table is created idempotently on first use.
        await raw.execute(_CREATE_WATERMARK_TABLE_SQL)

        prior_row = None
        try:
            prior_row = await raw.fetchval(_SELECT_WATERMARK_SQL, workspace_id)
        except Exception:
            prior_row = None

        prior_watermark = None
        if prior_row is not None:
            try:
                prior_watermark = json.loads(prior_row) if isinstance(prior_row, str) else prior_row
            except (ValueError, TypeError):
                prior_watermark = None

        # SKIP only on an exact match against a recorded prior watermark AND a
        # confirmed non-empty subgraph (defense-in-depth: the marker table and the
        # AGE graph are separate stores; an out-of-band graph drop/restore while the
        # marker row survives would cause empty results if we skipped unconditionally).
        # A cheap LIMIT-1 probe confirms presence; absence falls through to full
        # re-materialize. Any probe error is treated as "absent" (fail-safe).
        if prior_watermark is not None and list(prior_watermark) == current_watermark:
            ws_param = json.dumps({"ws": workspace_id})
            subgraph_present = False
            try:
                probe_rows = await raw.fetch(probe_workspace_subgraph_sql(), ws_param)
                subgraph_present = len(probe_rows) > 0
            except Exception:
                subgraph_present = False  # fail-safe: treat as absent -> re-materialize
            if subgraph_present:
                return False

    # Watermark advanced (or no prior marker / mismatch): full delete-then-MERGE.
    await materialize_workspace_subgraph(
        age_engine,
        workspace_id=workspace_id,
        node_attrs=node_attrs,
        associations=associations,
    )

    # Record the new watermark AFTER a successful materialize.
    async with age_engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)
        await raw.execute(_CREATE_WATERMARK_TABLE_SQL)
        await raw.execute(_UPSERT_WATERMARK_SQL, workspace_id, json.dumps(current_watermark))

    return True


async def materialize_workspace_subgraph(
    age_engine,
    *,
    workspace_id: str,
    node_attrs: dict[str, dict],
    associations: list,
) -> None:
    """Scope-DELETE then MERGE this workspace's vertices + edges into AGE.

    This is the delete-then-MERGE flow extracted verbatim from the P2.1
    ``_materialize_and_extract`` (it does NOT extract — callers extract the
    node/edge set themselves with their own queries).

    Args:
        age_engine: a SQLAlchemy AsyncEngine built with the AGE search_path
            baked into ``connect_args`` (see ``make_age_engine_kwargs``). A
            fresh raw asyncpg connection is checked out and ``ensure_age_session``
            is re-applied (LOAD + search_path are session-scoped).
        workspace_id: the workspace to materialize. MUST already be validated
            by the caller via ``_cypher.validate_id``.
        node_attrs: ``{memory_id: {"memory_type": ..., "memory_subtype": ...}}``
            for every active node that should exist as a ``Memory`` vertex.
        associations: rows of ``(assoc_id, source_id, target_id, relationship,
            strength)``. Edges whose endpoints are not both in ``node_attrs`` are
            skipped (mirrors the OSS "both endpoints in graph" guard).

    All cypher is parameterized via the ``$1`` JSON bind arg — no user values
    in SQL text (BLOCKER fix).
    """
    # Build the validated node and edge payloads (identical to the prior
    # inline logic in age.py).
    nodes_payload = []
    for node_id, attrs in node_attrs.items():
        nodes_payload.append({
            "id": node_id,
            "ws": workspace_id,
            "status": "active",
            "mt": attrs.get("memory_type"),
            "ms": attrs.get("memory_subtype"),
        })

    edges_payload = []
    for assoc_id, source_id, target_id, relationship, strength in associations:
        if source_id not in node_attrs or target_id not in node_attrs:
            continue
        edges_payload.append({
            "aid": assoc_id,
            "src": source_id,
            "tgt": target_id,
            "ws": workspace_id,
            "rel": relationship,
            "strength": _cypher._safe_strength(strength),
        })

    ws_param = json.dumps({"ws": workspace_id})

    # NOTE — non-atomic materialize→extract boundary: this function closes its
    # connection after the MERGE. The caller (AgeGraphQueryService) opens a
    # second connection to run its read-only cypher. A concurrent writer that
    # commits between these two connections could let the extract see slightly
    # newer relational data than what was materialized here. This is benign for
    # Track-A read-only queries (the extract just sees a slightly fresher graph
    # than the materialize built) and is acceptable for Phase-1 semantics.
    # Phase-2 watermark optimization will fuse materialize+extract into one
    # connection when the relational subgraph has not changed.
    async with age_engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        # LOAD + search_path are session-scoped; re-apply on each pooled conn.
        await ensure_age_session(raw)

        # Wrap delete→MERGE in a single transaction so a mid-run failure cannot
        # leave the Memory subgraph half-wiped (e.g. vertices gone but edges
        # still present from a prior run, or vice-versa). Matches the parity
        # established by the Entity layer (~line 550) and Fragment layer (~line
        # 776) which both wrap DELETE→MERGE in `async with raw.transaction()`.
        # Same best-effort atomicity boundary (within one connection): a
        # concurrent reader on a DIFFERENT pooled connection may briefly see an
        # empty Memory subgraph during the window — the accepted
        # eventual-consistency window already documented in the non-atomic
        # materialize→extract boundary comment above.
        async with raw.transaction():
            # MAJOR-1: scope-DELETE this workspace's AGE subgraph first so the
            # materialized view always reflects current relational truth (no
            # phantom edges or stale status from prior runs).
            await raw.execute(_cypher.delete_workspace_edges_sql(), ws_param)
            await raw.execute(_cypher.delete_workspace_vertices_sql(), ws_param)

            # MERGE vertices (batch UNWIND — single parameterized statement).
            if nodes_payload:
                await raw.fetch(
                    _cypher.batch_merge_vertices_sql(),
                    json.dumps({"nodes": nodes_payload}),
                )

            # MERGE edges (batch UNWIND — single parameterized statement).
            if edges_payload:
                await raw.fetch(
                    _cypher.batch_merge_edges_sql(),
                    json.dumps({"edges": edges_payload}),
                )


# ---------------------------------------------------------------------------
# Entity-layer materialization (graph-moat C1 — ENTERPRISE rich tier)
# ---------------------------------------------------------------------------


def compute_entity_watermark(entities: list[dict], members: list[dict]) -> list:
    """Deterministic dirty-watermark for the Entity layer of a workspace.

    Token: ``[entity_count, max_entity_updated_at_iso, member_count,
    member_digest]``.

    ``entity_count`` and ``member_count`` make a DELETE detectable (a removal
    drops the count even when no timestamp advances). ``max_entity_updated_at``
    catches entity-level mutations (type, name, confidence, ...). These three
    values match the three-scalar token used for the Memory subgraph watermark.

    ``member_digest`` closes the remaining hole: an in-place member edit that
    preserves the count (e.g. role ``"mention"`` → ``"self"`` on one row while
    adding a new row and removing another) is not captured by counts or entity
    timestamps because ``entity_members`` has no ``updated_at`` column. The
    digest is a short SHA-256 hex prefix over the SORTED set of
    ``(entity_id, memory_id, role)`` tuples, making any membership change
    (insert, delete, role mutation) detectable with negligible collision risk.

    Precondition: ``entity.updated_at`` values must be ISO-8601 strings (or
    ``datetime`` objects whose ``str()`` is ISO-8601) so that lexical ``max``
    is equivalent to chronological ``max``. The PostgreSQL storage backend
    returns ``datetime.isoformat()`` strings satisfying this precondition.

    Callers treat the token as OPAQUE and test EQUALITY only. Returned as a
    ``list`` so it round-trips through JSON (jsonb) and compares cleanly
    against a deserialized prior token.
    """
    max_updated = ""
    for e in entities:
        u = e.get("updated_at")
        if u is not None:
            u = str(u)
            if u > max_updated:
                max_updated = u

    # Order-insensitive digest over the membership set. Sorting the tuples first
    # makes the hash independent of the order ``list_workspace_entity_members``
    # returns rows (which may vary across backends or after a host restart).
    tuples = sorted(
        (m["entity_id"], m["memory_id"], m["role"])
        for m in members
    )
    raw = json.dumps(tuples, separators=(",", ":"))
    member_digest = hashlib.sha256(raw.encode()).hexdigest()[:16]

    return [len(entities), max_updated, len(members), member_digest]


async def materialize_workspace_entities_gated(
    age_engine,
    *,
    workspace_id: str,
    entities: list[dict],
    members: list[dict],
) -> bool:
    """Watermark-gated Entity/MENTIONS materialize (graph-moat C1, Gate B).

    Mirrors ``materialize_workspace_subgraph_gated`` for the entity layer but
    gates on a SEPARATE entity-layer watermark (``compute_entity_watermark``) and
    a SEPARATE marker table, because entity-registry edits do not move the Memory
    watermark. Skips the full delete-then-MERGE only on an exact watermark match
    against a recorded prior token AND a confirmed-present Entity subgraph
    (defense-in-depth probe). Any ambiguity falls through to a full
    re-materialize (fail-safe — never serve a stale entity layer).

    Returns ``True`` if a full materialize ran, ``False`` if safely skipped.
    """
    current_watermark = compute_entity_watermark(entities, members)

    async with age_engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)
        await raw.execute(_CREATE_ENTITY_WATERMARK_TABLE_SQL)

        prior_row = None
        try:
            prior_row = await raw.fetchval(_SELECT_ENTITY_WATERMARK_SQL, workspace_id)
        except Exception:
            _log.warning(
                "Failed to read entity watermark for workspace %s; falling through to "
                "full re-materialize (fail-safe).", workspace_id, exc_info=True,
            )
            prior_row = None

        prior_watermark = None
        if prior_row is not None:
            try:
                prior_watermark = json.loads(prior_row) if isinstance(prior_row, str) else prior_row
            except (ValueError, TypeError):
                prior_watermark = None

        if prior_watermark is not None and list(prior_watermark) == current_watermark:
            ws_param = json.dumps({"ws": workspace_id})
            # An EMPTY registry has a [0, "", 0, <digest>] watermark; once
            # recorded a subsequent empty run matches it and there is nothing to
            # probe for (no Entity vertices ever existed). Treat entity_count==0
            # as a legitimate skip without requiring a presence probe.
            if current_watermark[0] == 0:
                return False
            entities_present = False
            try:
                probe_rows = await raw.fetch(probe_workspace_entities_sql(), ws_param)
                entities_present = len(probe_rows) > 0
            except Exception:
                _log.warning(
                    "Entity presence probe failed for workspace %s; falling through to "
                    "full re-materialize (fail-safe).", workspace_id, exc_info=True,
                )
                entities_present = False  # fail-safe: treat as absent -> re-materialize
            if entities_present:
                return False

    # Watermark advanced (or no prior marker / mismatch / absent subgraph):
    # full delete-then-MERGE of the entity layer.
    await materialize_workspace_entities(
        age_engine,
        workspace_id=workspace_id,
        entities=entities,
        members=members,
    )

    # Record the new watermark AFTER a successful materialize.
    async with age_engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)
        await raw.execute(_CREATE_ENTITY_WATERMARK_TABLE_SQL)
        await raw.execute(
            _UPSERT_ENTITY_WATERMARK_SQL, workspace_id, json.dumps(current_watermark)
        )

    return True


async def materialize_workspace_entities(
    age_engine,
    *,
    workspace_id: str,
    entities: list[dict],
    members: list[dict],
) -> None:
    """Scope-DELETE then MERGE this workspace's Entity vertices + MENTIONS edges.

    Mirrors ``materialize_workspace_subgraph`` for the entity layer:
      1. scope-DELETE the workspace's MENTIONS edges, then Entity vertices
         (delete-then-MERGE idempotency; Memory vertices are NOT touched);
      2. MERGE Entity vertices (aliases as a list property — the HAS_ALIAS shape
         decision in ``_cypher``);
      3. MERGE MENTIONS edges Entity->Memory carrying ``role``. A MENTIONS edge
         whose Memory endpoint is absent from the Memory subgraph is simply not
         matched (no-op), mirroring the ASSOC "both endpoints in graph" guard.

    Args:
        workspace_id: MUST already be validated by the caller via
            ``_cypher.validate_id``.
        entities: entity dicts (id, entity_type, canonical_name, normalized_name,
            aliases, ...) as returned by ``storage.list_workspace_entities``.
            Only ACTIVE entities should be passed; merged tombstones are excluded
            upstream so they never appear as vertices.
        members: member dicts (entity_id, memory_id, role, ...) as returned by
            ``storage.list_workspace_entity_members``.

    All cypher is parameterized via the ``$1`` JSON bind arg — no user values in
    SQL text. Entity ids and memory ids are validated below as a second layer of
    defence (the strict allowlist) before they reach the bind payload.
    """
    entities_payload = []
    valid_entity_ids: set[str] = set()
    for ent in entities:
        ent_id = _cypher.validate_id(ent["id"], kind="entity_id")
        valid_entity_ids.add(ent_id)
        entities_payload.append({
            "id": ent_id,
            "ws": workspace_id,
            "etype": ent.get("entity_type"),
            "norm": ent.get("normalized_name"),
            "label": ent.get("canonical_name"),
            # aliases stored as a list PROPERTY (HAS_ALIAS shape decision).
            "aliases": list(ent.get("aliases") or []),
        })

    mentions_payload = []
    for mem in members:
        entity_id = mem.get("entity_id")
        memory_id = mem.get("memory_id")
        # Skip members whose entity is not in the active set (e.g. a member of a
        # merged/tombstoned entity that was excluded upstream) — no dangling edge.
        if entity_id not in valid_entity_ids:
            continue
        # Validate the memory_id as a second defence layer before binding.
        _cypher.validate_id(memory_id, kind="memory_id")
        mentions_payload.append({
            "eid": entity_id,
            "mid": memory_id,
            "ws": workspace_id,
            "role": mem.get("role", "mention"),
        })

    ws_param = json.dumps({"ws": workspace_id})

    async with age_engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)

        # Wrap delete→MERGE in a single transaction so a mid-run failure cannot
        # leave the entity layer half-deleted (e.g. Entity vertices gone but
        # MENTIONS edges still present from a prior run, or vice-versa).
        # asyncpg raw connections expose transaction() as a context manager.
        # Note: this is a best-effort atomicity boundary within one connection.
        # A concurrent reader that checks out a DIFFERENT pooled connection
        # between the DELETE and MERGE commits may briefly see an empty entity
        # layer — the same accepted eventual-consistency window that the Memory
        # subgraph materialize documents in its non-atomic materialize→extract
        # boundary comment.
        async with raw.transaction():
            # Scope-DELETE the entity layer first (edges then vertices) so the
            # materialized view always reflects current registry truth.
            await raw.execute(_cypher.delete_workspace_entity_mentions_sql(), ws_param)
            await raw.execute(_cypher.delete_workspace_entities_sql(), ws_param)

            # MERGE Entity vertices (batch UNWIND).
            if entities_payload:
                await raw.fetch(
                    _cypher.batch_merge_entities_sql(),
                    json.dumps({"entities": entities_payload}),
                )

            # MERGE MENTIONS edges (batch UNWIND). Endpoints whose Memory vertex
            # is absent are simply not matched.
            if mentions_payload:
                await raw.fetch(
                    _cypher.batch_merge_mentions_sql(),
                    json.dumps({"mentions": mentions_payload}),
                )


# ---------------------------------------------------------------------------
# Fragment-layer materialization (graph-moat P4.5 — ENTERPRISE rich tier)
# ---------------------------------------------------------------------------


def compute_fragment_watermark(fragments: list[dict]) -> list:
    """Deterministic dirty-watermark for the Fragment layer of a workspace.

    Token: ``[fragment_count, max_fragment_updated_at_iso, fragment_digest]``.

    ``fragment_count`` makes a DELETE detectable (a removed fact memory drops the
    count even when no timestamp advances). ``max_fragment_updated_at`` catches
    content/source mutations on existing fact memories. These two values mirror
    the count + max-updated scalars of the Memory/Entity watermarks.

    ``fragment_digest`` closes the remaining hole that motivates a SEPARATE
    fragment watermark: a COUNT-PRESERVING repoint of a fact's ``source_id`` (the
    parent memory it was derived from) changes the DERIVED_FROM topology without
    necessarily advancing any timestamp the count + max-updated pair would catch.
    The digest is a short SHA-256 hex prefix over the SORTED set of
    ``(fragment_id, source_id)`` tuples, making any (fragment_id, source_id)
    change — insert, delete, or source repoint — detectable with negligible
    collision risk. This is the SAME shape as the Entity layer's member digest.

    Precondition: ``fragment.updated_at`` values must be ISO-8601 strings (or
    ``datetime`` objects whose ``str()`` is ISO-8601) so that lexical ``max`` is
    equivalent to chronological ``max``. The storage backend returns
    ``datetime.isoformat()`` strings satisfying this precondition.

    Callers treat the token as OPAQUE and test EQUALITY only. Returned as a
    ``list`` so it round-trips through JSON (jsonb) and compares cleanly against
    a deserialized prior token.
    """
    max_updated = ""
    for f in fragments:
        u = f.get("updated_at")
        if u is not None:
            u = str(u)
            if u > max_updated:
                max_updated = u

    # Order-insensitive digest over the (fragment_id, source_id) set. Sorting the
    # tuples first makes the hash independent of the order the fact memories are
    # returned (which may vary across backends or after a host restart). A
    # count-preserving source_id repoint changes a tuple and thus the digest.
    tuples = sorted(
        (f["id"], f.get("source_id") or "")
        for f in fragments
    )
    raw = json.dumps(tuples, separators=(",", ":"))
    fragment_digest = hashlib.sha256(raw.encode()).hexdigest()[:16]

    return [len(fragments), max_updated, fragment_digest]


async def materialize_workspace_fragments_gated(
    age_engine,
    *,
    workspace_id: str,
    fragments: list[dict],
) -> bool:
    """Watermark-gated Fragment/DERIVED_FROM materialize (graph-moat P4.5, Gate B).

    Mirrors ``materialize_workspace_entities_gated`` for the fragment layer but
    gates on a SEPARATE fragment-layer watermark (``compute_fragment_watermark``)
    and a SEPARATE marker table, because a count-preserving source_id repoint does
    not move the Memory or Entity watermark. Skips the full delete-then-MERGE only
    on an exact watermark match against a recorded prior token AND a
    confirmed-present Fragment subgraph (defense-in-depth probe). Any ambiguity
    falls through to a full re-materialize (fail-safe — never serve a stale
    fragment layer).

    Returns ``True`` if a full materialize ran, ``False`` if safely skipped.
    """
    current_watermark = compute_fragment_watermark(fragments)

    async with age_engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)
        await raw.execute(_CREATE_FRAGMENT_WATERMARK_TABLE_SQL)

        prior_row = None
        try:
            prior_row = await raw.fetchval(_SELECT_FRAGMENT_WATERMARK_SQL, workspace_id)
        except Exception:
            _log.warning(
                "Failed to read fragment watermark for workspace %s; falling through to "
                "full re-materialize (fail-safe).", workspace_id, exc_info=True,
            )
            prior_row = None

        prior_watermark = None
        if prior_row is not None:
            try:
                prior_watermark = json.loads(prior_row) if isinstance(prior_row, str) else prior_row
            except (ValueError, TypeError):
                prior_watermark = None

        if prior_watermark is not None and list(prior_watermark) == current_watermark:
            ws_param = json.dumps({"ws": workspace_id})
            # An EMPTY fragment set has a [0, "", <digest>] watermark; once
            # recorded a subsequent empty run matches it and there is nothing to
            # probe for (no Fragment vertices ever existed). Treat
            # fragment_count==0 as a legitimate skip without requiring a probe.
            if current_watermark[0] == 0:
                return False
            fragments_present = False
            try:
                probe_rows = await raw.fetch(probe_workspace_fragments_sql(), ws_param)
                fragments_present = len(probe_rows) > 0
            except Exception:
                _log.warning(
                    "Fragment presence probe failed for workspace %s; falling through to "
                    "full re-materialize (fail-safe).", workspace_id, exc_info=True,
                )
                fragments_present = False  # fail-safe: treat as absent -> re-materialize
            if fragments_present:
                return False

    # Watermark advanced (or no prior marker / mismatch / absent subgraph):
    # full delete-then-MERGE of the fragment layer.
    await materialize_workspace_fragments(
        age_engine,
        workspace_id=workspace_id,
        fragments=fragments,
    )

    # Record the new watermark AFTER a successful materialize.
    async with age_engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)
        await raw.execute(_CREATE_FRAGMENT_WATERMARK_TABLE_SQL)
        await raw.execute(
            _UPSERT_FRAGMENT_WATERMARK_SQL, workspace_id, json.dumps(current_watermark)
        )

    return True


async def materialize_workspace_fragments(
    age_engine,
    *,
    workspace_id: str,
    fragments: list[dict],
) -> None:
    """Scope-DELETE then MERGE this workspace's Fragment vertices + DERIVED_FROM edges.

    Mirrors ``materialize_workspace_entities`` for the fragment layer:
      1. scope-DELETE the workspace's DERIVED_FROM edges, then Fragment vertices
         (delete-then-MERGE idempotency; Memory + Entity vertices are NOT touched);
      2. MERGE Fragment vertices (one per fact memory: id, content, source_id);
      3. MERGE DERIVED_FROM edges Fragment->source Memory keyed on
         ``metadata["source_id"]``. A DERIVED_FROM edge whose source Memory
         endpoint is absent from the Memory subgraph is simply not matched
         (no-op), mirroring the ASSOC / MENTIONS "both endpoints in graph" guard.
         A fact memory with no ``source_id`` (or an empty one) materializes as a
         Fragment vertex but contributes no edge.

    Args:
        workspace_id: MUST already be validated by the caller via
            ``_cypher.validate_id``.
        fragments: fragment dicts (``id``, ``content``, ``source_id``,
            ``updated_at``) derived from the workspace's ``subtype="fact"``
            memories (``metadata["source_id"]`` lifted to ``source_id``). Only
            ACTIVE fact memories should be passed.

    All cypher is parameterized via the ``$1`` JSON bind arg — no user values in
    SQL text. Fragment ids and source ids are validated below as a second layer
    of defence (the strict allowlist) before they reach the bind payload.
    """
    fragments_payload = []
    edges_payload = []
    for frag in fragments:
        frag_id = _cypher.validate_id(frag["id"], kind="fragment_id")
        source_id = frag.get("source_id")
        fragments_payload.append({
            "id": frag_id,
            "ws": workspace_id,
            # content stored as a property VALUE (never cypher syntax).
            "content": frag.get("content"),
            "source_id": source_id,
        })
        # A DERIVED_FROM edge is only emitted when the fragment carries a
        # source_id (the parent memory it was decomposed from). Validate it as a
        # second defence layer before binding.
        if source_id:
            _cypher.validate_id(source_id, kind="source_id")
            edges_payload.append({
                "fid": frag_id,
                "sid": source_id,
                "ws": workspace_id,
            })

    ws_param = json.dumps({"ws": workspace_id})

    async with age_engine.connect() as conn:
        raw = (await conn.get_raw_connection()).driver_connection
        await ensure_age_session(raw)

        # Wrap delete->MERGE in a single transaction so a mid-run failure cannot
        # leave the fragment layer half-deleted. Same best-effort atomicity
        # boundary (within one connection) the entity materialize documents; a
        # concurrent reader on a DIFFERENT pooled connection may briefly see an
        # empty fragment layer (accepted eventual-consistency window).
        async with raw.transaction():
            # Scope-DELETE the fragment layer first (edges then vertices) so the
            # materialized view always reflects current fact-memory truth.
            await raw.execute(_cypher.delete_workspace_derived_from_sql(), ws_param)
            await raw.execute(_cypher.delete_workspace_fragments_sql(), ws_param)

            # MERGE Fragment vertices (batch UNWIND).
            if fragments_payload:
                await raw.fetch(
                    _cypher.batch_merge_fragments_sql(),
                    json.dumps({"fragments": fragments_payload}),
                )

            # MERGE DERIVED_FROM edges (batch UNWIND). Endpoints whose source
            # Memory vertex is absent are simply not matched.
            if edges_payload:
                await raw.fetch(
                    _cypher.batch_merge_derived_from_sql(),
                    json.dumps({"edges": edges_payload}),
                )
