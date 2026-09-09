"""Unit tests for the read-only graph-QUERY cypher builders (no database; P2 Track A).

Tests the pure-string + validation contract of the read-only builders added to
``services/graph_analysis/_cypher.py`` for the Track-A graph-query backend:
  - Every builder produces a ``$1``-parameterized query (no caller value inlined).
  - Caller values are referenced ONLY as ``$key`` cypher params ($ws, $root,
    $roots, $ids, $src, $dst, $rel, $rels) — never string-inlined.
  - The ONLY non-$1 value in a body is the validated integer hop bound in the
    var-length range ``*1..N`` (a checked int via validate_hops).
  - validate_hops enforces 1..MAX_TRAVERSAL_HOPS and rejects non-ints/bools.
  - Direction arrows are correct for both/outgoing/incoming.
  - Relationship-filter clauses reference $rels (bind list), not inlined values.

DB round-trip parity (and injection-payload round-trips) is covered by the
graph-query conformance test.
"""

import pytest

from memorylayer_saas.services.graph_analysis import _cypher as c


# ---------------------------------------------------------------------------
# validate_hops
# ---------------------------------------------------------------------------


class TestValidateHops:
    def test_accepts_in_range(self):
        assert c.validate_hops(1) == 1
        assert c.validate_hops(c.MAX_TRAVERSAL_HOPS) == c.MAX_TRAVERSAL_HOPS

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            c.validate_hops(0)

    def test_rejects_over_max(self):
        with pytest.raises(ValueError):
            c.validate_hops(c.MAX_TRAVERSAL_HOPS + 1)

    def test_rejects_bool(self):
        # bool is an int subclass — must be rejected so True/False can't slip in.
        with pytest.raises(ValueError):
            c.validate_hops(True)

    def test_rejects_non_int(self):
        with pytest.raises(ValueError):
            c.validate_hops("3")  # type: ignore[arg-type]

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            c.validate_hops(-2)


# ---------------------------------------------------------------------------
# _direction_arrows
# ---------------------------------------------------------------------------


class TestDirectionArrows:
    def test_both(self):
        assert c._direction_arrows("both") == ("-", "-")

    def test_outgoing(self):
        assert c._direction_arrows("outgoing") == ("-", "->")

    def test_incoming(self):
        assert c._direction_arrows("incoming") == ("<-", "-")

    def test_rejects_unknown(self):
        with pytest.raises(ValueError):
            c._direction_arrows("sideways")


# ---------------------------------------------------------------------------
# neighbors_paths_sql (returns whole paths; rel-filter applied in Python)
# ---------------------------------------------------------------------------


class TestNeighborsPathsSql:
    def test_parameterized(self):
        sql = c.neighbors_paths_sql(3, "both")
        assert "$1)" in sql
        assert "$root" in sql and "$ws" in sql

    def test_inlines_only_validated_int_bound(self):
        sql = c.neighbors_paths_sql(4, "both")
        # The validated int appears as the var-length upper bound literal.
        assert "*1..4]" in sql

    def test_direction_outgoing_arrow(self):
        sql = c.neighbors_paths_sql(2, "outgoing")
        assert "]->(" in sql

    def test_direction_incoming_arrow(self):
        sql = c.neighbors_paths_sql(2, "incoming")
        assert ")<-[" in sql

    def test_returns_path_elements(self):
        sql = c.neighbors_paths_sql(2, "both")
        assert "RETURN nodes(p), relationships(p)" in sql

    def test_no_unsupported_all_predicate(self):
        # AGE 1.7 lacks ALL(); the builder must never emit it.
        sql = c.neighbors_paths_sql(2, "both")
        assert "ALL(" not in sql


# ---------------------------------------------------------------------------
# k_hop_paths_sql
# ---------------------------------------------------------------------------


class TestKHopPathsSql:
    def test_parameterized_and_unwinds_roots(self):
        sql = c.k_hop_paths_sql(3, "both")
        assert "$1)" in sql
        assert "UNWIND $roots AS root_id" in sql

    def test_inlines_validated_int_bound(self):
        sql = c.k_hop_paths_sql(5, "both")
        assert "*1..5]" in sql

    def test_returns_path_elements(self):
        sql = c.k_hop_paths_sql(2, "both")
        assert "RETURN nodes(p), relationships(p)" in sql


# ---------------------------------------------------------------------------
# induced_edges_sql
# ---------------------------------------------------------------------------


class TestInducedEdgesSql:
    def test_parameterized(self):
        sql = c.induced_edges_sql(with_rel_filter=False)
        assert "$1)" in sql
        assert "$ids" in sql and "$ws" in sql

    def test_both_endpoints_in_ids(self):
        sql = c.induced_edges_sql(with_rel_filter=False)
        assert "m.id IN $ids AND n.id IN $ids" in sql

    def test_returns_edge_columns(self):
        sql = c.induced_edges_sql(with_rel_filter=False)
        assert "RETURN m.id, n.id, e.relationship, e.strength" in sql

    def test_rel_filter_single_edge(self):
        sql = c.induced_edges_sql(with_rel_filter=True)
        assert "e.relationship IN $rels" in sql


# ---------------------------------------------------------------------------
# shortest_path_sql
# ---------------------------------------------------------------------------


class TestShortestPathSql:
    def test_parameterized(self):
        sql = c.shortest_path_sql(5)
        assert "$1)" in sql
        assert "$src" in sql and "$dst" in sql and "$ws" in sql

    def test_inlines_validated_int_bound(self):
        sql = c.shortest_path_sql(6)
        assert "*1..6]" in sql

    def test_orders_by_hops_ascending(self):
        sql = c.shortest_path_sql(3)
        assert "ORDER BY hops ASC" in sql

    def test_returns_path_elements(self):
        sql = c.shortest_path_sql(3)
        assert "nodes(p)" in sql and "relationships(p)" in sql and "length(p)" in sql

    def test_no_unsupported_features(self):
        # AGE 1.7 lacks list comprehensions and ALL(); neither may appear.
        sql = c.shortest_path_sql(3)
        assert "ALL(" not in sql
        assert "| x.id" not in sql


# ---------------------------------------------------------------------------
# typed_pattern_sql / relationship_rollup_sql
# ---------------------------------------------------------------------------


class TestTypedPatternSql:
    def test_parameterized_with_rel_bind(self):
        sql = c.typed_pattern_sql()
        assert "$1)" in sql
        assert "relationship: $rel" in sql

    def test_returns_edge_columns(self):
        sql = c.typed_pattern_sql()
        assert "RETURN m.id, n.id, e.relationship, e.strength" in sql


class TestRelationshipRollupSql:
    def test_parameterized(self):
        sql = c.relationship_rollup_sql()
        assert "$1)" in sql
        assert "$ws" in sql

    def test_groups_count_by_relationship(self):
        sql = c.relationship_rollup_sql()
        assert "count(e)" in sql
        assert "RETURN e.relationship, count(e)" in sql


# ---------------------------------------------------------------------------
# No-inlining invariant across every read-only builder
# ---------------------------------------------------------------------------


class TestNoValueInlining:
    """No caller value may appear as a literal; only $key refs + validated int."""

    def test_all_builders_end_with_dollar_one(self):
        builders = [
            c.neighbors_paths_sql(3, "both"),
            c.k_hop_paths_sql(3, "both"),
            c.induced_edges_sql(with_rel_filter=True),
            c.shortest_path_sql(3),
            c.typed_pattern_sql(),
            c.relationship_rollup_sql(),
        ]
        for sql in builders:
            assert "$1)" in sql, sql

    def test_a_malicious_relationship_never_reaches_a_builder(self):
        # The builders never take the relationship value itself — it travels in
        # $rels / $rel. Construction is value-free; validation happens upstream.
        sql = c.typed_pattern_sql()
        assert "DROP TABLE" not in sql


# ---------------------------------------------------------------------------
# Entity-layer builders (graph-moat C1/C2 — ENTERPRISE rich tier)
# ---------------------------------------------------------------------------


class TestEntityWriteBuilders:
    """Entity vertex + MENTIONS edge MERGE/DELETE/probe builders (C1)."""

    def test_merge_entities_parameterized(self):
        sql = c.batch_merge_entities_sql()
        assert "$1)" in sql
        assert "UNWIND $entities AS row" in sql
        # Identity MERGE keyed on id+workspace; aliases as a list property.
        assert "MERGE (e:Entity {id: row.id, workspace_id: row.ws})" in sql
        assert "e.aliases = row.aliases" in sql
        assert "e.label = row.label" in sql

    def test_merge_entities_no_value_inlined(self):
        sql = c.batch_merge_entities_sql()
        # Only $key refs and the Entity label constant — no caller value literal.
        assert "DROP TABLE" not in sql
        assert "row.id" in sql and "row.ws" in sql and "row.etype" in sql

    def test_merge_mentions_parameterized(self):
        sql = c.batch_merge_mentions_sql()
        assert "$1)" in sql
        assert "UNWIND $mentions AS row" in sql
        # Edge keyed on entity_id+memory_id+role+workspace; role carried as prop.
        assert "MERGE (e)-[r:MENTIONS {entity_id: row.eid, memory_id: row.mid," in sql
        assert "role: row.role" in sql

    def test_merge_mentions_matches_both_endpoints(self):
        sql = c.batch_merge_mentions_sql()
        assert "MATCH (e:Entity {id: row.eid, workspace_id: row.ws})" in sql
        assert "(m:Memory {id: row.mid, workspace_id: row.ws})" in sql

    def test_delete_mentions_parameterized(self):
        sql = c.delete_workspace_entity_mentions_sql()
        assert "$1)" in sql and "$ws" in sql
        assert "DELETE r" in sql
        assert "[r:MENTIONS]" in sql

    def test_delete_entities_detach(self):
        sql = c.delete_workspace_entities_sql()
        assert "$1)" in sql and "$ws" in sql
        assert "DETACH DELETE e" in sql
        # Only touches Entity vertices — never DETACH DELETEs Memory.
        assert "(e:Entity {workspace_id: $ws})" in sql

    def test_probe_entities_limit_one(self):
        sql = c.probe_workspace_entities_sql()
        assert "$1)" in sql and "$ws" in sql
        assert "RETURN e.id LIMIT 1" in sql


class TestEntityReadBuilders:
    """Entity-neighborhood read builders (C2)."""

    def test_mentioned_memories_parameterized(self):
        sql = c.entity_mentioned_memories_sql()
        assert "$1)" in sql
        assert "$ws" in sql and "$eid" in sql
        assert "(e:Entity {id: $eid, workspace_id: $ws})-[r:MENTIONS]->(m:Memory" in sql
        assert "RETURN m.id, r.role" in sql
        # MAJOR-1 fix: deterministic ORDER BY so first-role-wins dedup is
        # identical to the OSS list_workspace_entity_members sort.
        assert "ORDER BY m.id, r.role" in sql
        # Active-memory filter (parity with Memory-layer reads).
        assert "m.status = 'active'" in sql

    def test_co_mentioned_entities_parameterized(self):
        sql = c.entity_co_mentioned_entities_sql()
        assert "$1)" in sql
        assert "$ws" in sql and "$eid" in sql
        # Entity -> MENTIONS -> Memory <- MENTIONS <- Entity traversal.
        assert "-[:MENTIONS]->(m:Memory {workspace_id: $ws})" in sql
        assert "<-[:MENTIONS]-(co:Entity {workspace_id: $ws})" in sql
        # Root entity excluded; returns the shared memory for count aggregation.
        assert "co.id <> $eid" in sql
        assert "RETURN co.id, co.label, co.entity_type, m.id" in sql

    def test_co_mentioned_no_value_inlined(self):
        sql = c.entity_co_mentioned_entities_sql()
        assert "DROP TABLE" not in sql


class TestEntityIdValidationConformance:
    """validate_id is the second-layer allowlist for entity ids reaching cypher.

    Mirrors the injection-conformance contract: a malicious entity_id /
    workspace_id / memory_id never breaks out of the bind param because it is
    rejected by validate_id BEFORE any payload is built (and would only ever be
    a $key bind value otherwise — never inlined).
    """

    _EVIL = [
        "$$) AS (x agtype); DROP TABLE entities; --",
        "ent$evil",
        "ent'quote",
        'ent"dquote',
        "ent;semicolon",
        "$$",
        "$",
    ]

    @pytest.mark.parametrize("evil", _EVIL)
    def test_validate_id_rejects_evil_entity_id(self, evil):
        with pytest.raises(ValueError):
            c.validate_id(evil, kind="entity_id")

    @pytest.mark.parametrize("evil", _EVIL)
    def test_validate_id_rejects_evil_memory_id(self, evil):
        with pytest.raises(ValueError):
            c.validate_id(evil, kind="memory_id")

    def test_validate_id_accepts_normal_entity_id(self):
        assert c.validate_id("ent_abc123-DEF", kind="entity_id") == "ent_abc123-DEF"


# ---------------------------------------------------------------------------
# compute_entity_watermark (MAJOR-2 — member digest)
# ---------------------------------------------------------------------------


class TestComputeEntityWatermark:
    """Unit tests for the entity-layer watermark token (no DB)."""

    def test_token_shape_four_elements(self):
        from memorylayer_saas.services.graph_analysis._materialize import compute_entity_watermark

        wm = compute_entity_watermark(
            [{"id": "e1", "updated_at": "2026-01-02T00:00:00"},
             {"id": "e2", "updated_at": "2026-01-03T00:00:00"}],
            [{"entity_id": "e1", "memory_id": "m1", "role": "self"}],
        )
        assert len(wm) == 4
        assert wm[0] == 2           # entity_count
        assert wm[1] == "2026-01-03T00:00:00"  # max updated_at
        assert wm[2] == 1           # member_count
        assert isinstance(wm[3], str) and len(wm[3]) == 16  # SHA-256 hex prefix

    def test_empty_registry_token(self):
        from memorylayer_saas.services.graph_analysis._materialize import compute_entity_watermark

        wm = compute_entity_watermark([], [])
        assert wm[0] == 0 and wm[1] == "" and wm[2] == 0
        assert isinstance(wm[3], str)

    def test_role_swap_changes_digest_not_counts(self):
        """MAJOR-2: a count-preserving role mutation must change the digest."""
        from memorylayer_saas.services.graph_analysis._materialize import compute_entity_watermark

        entities = [{"id": "e1", "updated_at": "2026-01-01T00:00:00"}]
        members_before = [{"entity_id": "e1", "memory_id": "m1", "role": "mention"}]
        members_after  = [{"entity_id": "e1", "memory_id": "m1", "role": "self"}]

        wm_before = compute_entity_watermark(entities, members_before)
        wm_after  = compute_entity_watermark(entities, members_after)

        # Counts are identical — only the digest changes.
        assert wm_before[0] == wm_after[0]  # entity_count unchanged
        assert wm_before[2] == wm_after[2]  # member_count unchanged
        assert wm_before[3] != wm_after[3]  # digest MUST differ

    def test_digest_is_order_insensitive(self):
        """Digest should be the same regardless of member list order."""
        from memorylayer_saas.services.graph_analysis._materialize import compute_entity_watermark

        entities = [{"id": "e1", "updated_at": "2026-01-01T00:00:00"}]
        m1 = {"entity_id": "e1", "memory_id": "m1", "role": "mention"}
        m2 = {"entity_id": "e1", "memory_id": "m2", "role": "self"}

        wm_ab = compute_entity_watermark(entities, [m1, m2])
        wm_ba = compute_entity_watermark(entities, [m2, m1])
        assert wm_ab == wm_ba
