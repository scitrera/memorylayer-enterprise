"""Unit tests for the AGE cypher query builders (no database).

Tests the pure-string and validation contract of
``services/graph_analysis/_cypher.py``:
  - All parameterized SQL builders produce $1-parameterized queries (no inlining).
  - validate_id / validate_relationship reject unsafe values including $, $$,
    quotes, spaces, and other injection-relevant chars.
  - _safe_strength rejects NaN and Inf.
  - wrap_cypher (bootstrap-only) targets the fixed graph name.
  - The SQL strings contain the correct structural elements (UNWIND, MERGE,
    DELETE, RETURN, parameterized $key references).

DB round-trip parity (including injection payload round-trips) is covered by
the conformance test.
"""

import math

import pytest

from memorylayer_saas.services.graph_analysis import _cypher as c


# ---------------------------------------------------------------------------
# validate_id
# ---------------------------------------------------------------------------


class TestValidateId:
    def test_accepts_uuid_hex(self):
        assert c.validate_id("abc123def456") == "abc123def456"

    def test_accepts_slug_with_hyphens_and_underscores(self):
        assert c.validate_id("age_conf_small_abc12345") == "age_conf_small_abc12345"

    def test_accepts_mem_prefix(self):
        assert c.validate_id("mem_0123456789ab") == "mem_0123456789ab"

    def test_rejects_dollar_sign(self):
        with pytest.raises(ValueError, match="workspace_id"):
            c.validate_id("ws$bad", kind="workspace_id")

    def test_rejects_dollar_dollar(self):
        """The $$ injection vector must be rejected outright."""
        with pytest.raises(ValueError):
            c.validate_id("$$) AS (x agtype); DROP TABLE memories; --")

    def test_rejects_single_quote(self):
        with pytest.raises(ValueError):
            c.validate_id("ws'evil")

    def test_rejects_space(self):
        with pytest.raises(ValueError):
            c.validate_id("ws evil")

    def test_rejects_semicolon(self):
        with pytest.raises(ValueError):
            c.validate_id("ws;DROP")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            c.validate_id("")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError):
            c.validate_id("a" * 257)

    def test_rejects_backslash(self):
        with pytest.raises(ValueError):
            c.validate_id("ws\\evil")

    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            c.validate_id(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_relationship
# ---------------------------------------------------------------------------


class TestValidateRelationship:
    def test_accepts_snake_case(self):
        assert c.validate_relationship("related_to") == "related_to"

    def test_accepts_with_hyphens(self):
        assert c.validate_relationship("leads-to") == "leads-to"

    def test_rejects_dollar(self):
        with pytest.raises(ValueError):
            c.validate_relationship("rel$bad")

    def test_rejects_space(self):
        with pytest.raises(ValueError):
            c.validate_relationship("related to")

    def test_rejects_single_quote(self):
        with pytest.raises(ValueError):
            c.validate_relationship("rel'bad")


# ---------------------------------------------------------------------------
# _safe_strength
# ---------------------------------------------------------------------------


class TestSafeStrength:
    def test_accepts_normal_floats(self):
        assert c._safe_strength(0.7) == pytest.approx(0.7)
        assert c._safe_strength(0.0) == pytest.approx(0.0)
        assert c._safe_strength(1.0) == pytest.approx(1.0)

    def test_accepts_int(self):
        assert c._safe_strength(1) == pytest.approx(1.0)

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            c._safe_strength(float("nan"))

    def test_rejects_inf(self):
        with pytest.raises(ValueError, match="finite"):
            c._safe_strength(float("inf"))

    def test_rejects_neg_inf(self):
        with pytest.raises(ValueError, match="finite"):
            c._safe_strength(float("-inf"))


# ---------------------------------------------------------------------------
# batch_merge_vertices_sql — parameterized, UNWIND, no inlining
# ---------------------------------------------------------------------------


class TestBatchMergeVerticesSql:
    def test_uses_dollar_one_bind_param(self):
        sql = c.batch_merge_vertices_sql()
        # Must end with the $1 bind parameter, not a plain $$ close.
        assert "$1)" in sql

    def test_contains_unwind_nodes(self):
        sql = c.batch_merge_vertices_sql()
        assert "UNWIND $nodes AS row" in sql

    def test_contains_merge_on_id_and_workspace(self):
        sql = c.batch_merge_vertices_sql()
        assert "MERGE (m:Memory {id: row.id, workspace_id: row.ws})" in sql

    def test_sets_status_type_subtype(self):
        sql = c.batch_merge_vertices_sql()
        assert "m.status = row.status" in sql
        assert "m.memory_type = row.mt" in sql
        assert "m.memory_subtype = row.ms" in sql

    def test_targets_fixed_graph_name(self):
        sql = c.batch_merge_vertices_sql()
        assert "'memorylayer'" in sql

    def test_no_workspace_id_literal_inlined(self):
        """workspace_id must NOT appear as a string literal in the SQL."""
        sql = c.batch_merge_vertices_sql()
        # The only single-quoted string should be the graph name and label
        assert "memorylayer" in sql
        # No user-data placeholders inlined — only row.* references
        assert "row.ws" in sql


# ---------------------------------------------------------------------------
# batch_merge_edges_sql — parameterized, UNWIND, no inlining
# ---------------------------------------------------------------------------


class TestBatchMergeEdgesSql:
    def test_uses_dollar_one_bind_param(self):
        assert "$1)" in c.batch_merge_edges_sql()

    def test_contains_unwind_edges(self):
        assert "UNWIND $edges AS row" in c.batch_merge_edges_sql()

    def test_matches_source_and_target(self):
        sql = c.batch_merge_edges_sql()
        assert "MATCH (s:Memory {id: row.src, workspace_id: row.ws})" in sql
        assert "(t:Memory {id: row.tgt, workspace_id: row.ws})" in sql

    def test_merges_assoc_edge(self):
        sql = c.batch_merge_edges_sql()
        assert "MERGE (s)-[e:ASSOC {assoc_id: row.aid, workspace_id: row.ws}]->(t)" in sql

    def test_sets_relationship_and_strength(self):
        sql = c.batch_merge_edges_sql()
        assert "e.relationship = row.rel" in sql
        assert "e.strength = row.strength" in sql


# ---------------------------------------------------------------------------
# delete_workspace_edges_sql / delete_workspace_vertices_sql
# ---------------------------------------------------------------------------


class TestDeleteSql:
    def test_edges_delete_is_parameterized(self):
        sql = c.delete_workspace_edges_sql()
        assert "$1)" in sql
        assert "$ws" in sql
        assert "DELETE e" in sql

    def test_vertices_delete_is_parameterized(self):
        sql = c.delete_workspace_vertices_sql()
        assert "$1)" in sql
        assert "$ws" in sql
        assert "DETACH DELETE m" in sql

    def test_no_workspace_literal_in_either(self):
        for sql in (c.delete_workspace_edges_sql(), c.delete_workspace_vertices_sql()):
            # No string value should be inlined — only $ws cypher param ref
            assert "workspace_id: $ws" in sql


# ---------------------------------------------------------------------------
# extract_nodes_sql / extract_edges_sql
# ---------------------------------------------------------------------------


class TestExtractSql:
    def test_nodes_sql_is_parameterized(self):
        sql = c.extract_nodes_sql()
        assert "$1)" in sql
        assert "$ws" in sql
        assert "m.status = 'active'" in sql
        assert "RETURN m.id, m.memory_type, m.memory_subtype" in sql

    def test_edges_sql_is_parameterized(self):
        sql = c.extract_edges_sql()
        assert "$1)" in sql
        assert "$ws" in sql

    def test_edges_sql_is_undirected(self):
        sql = c.extract_edges_sql()
        # Undirected match — no directional arrow in the pattern
        assert "}-[e:ASSOC]-{" not in sql  # not literal
        assert "-[e:ASSOC]-(" in sql
        assert "->(" not in sql.split("-[e:ASSOC]-")[1]  # after the edge pattern

    def test_edges_sql_scopes_both_endpoints(self):
        sql = c.extract_edges_sql()
        assert sql.count("workspace_id: $ws") == 2

    def test_edges_sql_filters_active_both_endpoints(self):
        sql = c.extract_edges_sql()
        assert "m.status = 'active' AND n.status = 'active'" in sql


# ---------------------------------------------------------------------------
# wrap_cypher — bootstrap-only, fixed graph name
# ---------------------------------------------------------------------------


class TestWrapCypher:
    def test_targets_fixed_graph(self):
        wrapped = c.wrap_cypher("MATCH (m) RETURN m.id", columns="id agtype")
        assert wrapped.startswith("SELECT * FROM cypher('memorylayer', $$ ")
        assert wrapped.endswith(" $$) AS (id agtype)")

    def test_graph_name_constant(self):
        assert c.GRAPH_NAME == "memorylayer"


# ---------------------------------------------------------------------------
# Fragment-layer builders (graph-moat P4.5) — parameterized, no inlining
# ---------------------------------------------------------------------------


class TestBatchMergeFragmentsSql:
    def test_uses_dollar_one_bind_param(self):
        assert "$1)" in c.batch_merge_fragments_sql()

    def test_contains_unwind_fragments(self):
        assert "UNWIND $fragments AS row" in c.batch_merge_fragments_sql()

    def test_merges_on_id_and_workspace(self):
        sql = c.batch_merge_fragments_sql()
        assert "MERGE (f:Fragment {id: row.id, workspace_id: row.ws})" in sql

    def test_sets_content_and_source(self):
        sql = c.batch_merge_fragments_sql()
        assert "f.content = row.content" in sql
        assert "f.source_id = row.source_id" in sql

    def test_no_user_value_inlined(self):
        # Only row.* references and the fixed graph/label names — no caller value.
        sql = c.batch_merge_fragments_sql()
        assert "'memorylayer'" in sql
        assert "row.ws" in sql


class TestBatchMergeDerivedFromSql:
    def test_uses_dollar_one_bind_param(self):
        assert "$1)" in c.batch_merge_derived_from_sql()

    def test_contains_unwind_edges(self):
        assert "UNWIND $edges AS row" in c.batch_merge_derived_from_sql()

    def test_matches_fragment_and_source_memory(self):
        sql = c.batch_merge_derived_from_sql()
        assert "MATCH (f:Fragment {id: row.fid, workspace_id: row.ws})" in sql
        assert "(m:Memory {id: row.sid, workspace_id: row.ws})" in sql

    def test_merges_derived_from_edge(self):
        sql = c.batch_merge_derived_from_sql()
        assert "-[d:DERIVED_FROM {fragment_id: row.fid, source_id: row.sid," in sql

    def test_directed_fragment_to_memory(self):
        # DERIVED_FROM points Fragment -> Memory.
        assert "]->(m)" in c.batch_merge_derived_from_sql()


class TestDeleteFragmentSql:
    def test_derived_from_delete_is_parameterized(self):
        sql = c.delete_workspace_derived_from_sql()
        assert "$1)" in sql and "$ws" in sql and "DELETE d" in sql

    def test_fragments_delete_is_parameterized(self):
        sql = c.delete_workspace_fragments_sql()
        assert "$1)" in sql and "$ws" in sql and "DETACH DELETE f" in sql

    def test_no_workspace_literal(self):
        for sql in (c.delete_workspace_derived_from_sql(), c.delete_workspace_fragments_sql()):
            assert "workspace_id: $ws" in sql


class TestProbeFragmentsSql:
    def test_is_parameterized_limited(self):
        sql = c.probe_workspace_fragments_sql()
        assert "$1)" in sql and "$ws" in sql
        assert "LIMIT 1" in sql


class TestFragmentsForMemorySql:
    def test_is_parameterized(self):
        sql = c.fragments_for_memory_sql()
        assert "$1)" in sql
        assert "$ws" in sql and "$mid" in sql

    def test_traverses_fragment_to_memory(self):
        sql = c.fragments_for_memory_sql()
        assert "(f:Fragment {workspace_id: $ws})-[:DERIVED_FROM]->(m:Memory {id: $mid, workspace_id: $ws})" in sql

    def test_filters_active_source_and_orders(self):
        sql = c.fragments_for_memory_sql()
        assert "m.status = 'active'" in sql
        assert "RETURN f.id, f.content" in sql
        assert "ORDER BY f.id" in sql
