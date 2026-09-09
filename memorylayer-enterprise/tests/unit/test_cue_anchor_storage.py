"""DB-free checks for the enterprise cue-anchor storage (cue retrieval channel).

Covers the ``CueAnchorModel`` schema (columns, table name, HNSW vector index),
the raw SQL migration ``010_cue_anchors.sql``, and the presence/signatures of the
PostgreSQL ``store_cue_anchors`` / ``search_cue_anchors`` overrides.

Vector-similarity query correctness (the ``<=>`` cosine search + MIN/GROUP-BY
dedup + join-to-memories) requires a live PostgreSQL+pgvector database and is NOT
exercised here — consistent with the other enterprise vector arms
(entity-anchor, fact, multivector), whose SQL paths are validated against live
PG. Run DB-free with the OSS server source on the path::

    PYTHONPATH=../../oss/memorylayer-core-python/src .venv/bin/python -m pytest \
        tests/unit/test_cue_anchor_storage.py
"""

import inspect
import pathlib

import pytest

from memorylayer_saas.storage.models import CueAnchorModel
from memorylayer_saas.storage.postgresql import PostgreSQLBackend


def test_cue_anchor_model_table_and_columns():
    assert CueAnchorModel.__tablename__ == "cue_anchors"
    cols = CueAnchorModel.__table__.columns
    for name in (
        "id", "workspace_id", "memory_id", "cue", "normalized_cue",
        "entity_id", "embedding", "created_at",
    ):
        assert name in cols, f"cue_anchors missing column {name}"
    # embedding is nullable (mirrors MemoryFragmentModel).
    assert cols["embedding"].nullable is True
    assert cols["cue"].nullable is False
    assert cols["normalized_cue"].nullable is False
    # entity_id links a cue to its canonical Entity; nullable with a FK to entities.
    assert cols["entity_id"].nullable is True
    fks = {fk.column.table.name for fk in cols["entity_id"].foreign_keys}
    assert "entities" in fks
    entity_fk = next(iter(cols["entity_id"].foreign_keys))
    assert entity_fk.ondelete == "SET NULL"


def test_cue_anchor_model_indexes():
    idx = {i.name for i in CueAnchorModel.__table__.indexes}
    assert "idx_cue_anchors_workspace" in idx
    assert "idx_cue_anchors_memory" in idx
    assert "idx_cue_anchors_normalized" in idx
    assert "idx_cue_anchors_entity" in idx
    # HNSW vector index over the cue embedding.
    assert "idx_cue_anchors_embedding" in idx
    emb_idx = next(i for i in CueAnchorModel.__table__.indexes if i.name == "idx_cue_anchors_embedding")
    assert emb_idx.dialect_options["postgresql"]["using"] == "hnsw"


def test_migration_010_exists_and_has_cue_table_and_vector_index():
    p = (
        pathlib.Path(__file__).resolve().parents[2]
        / "src" / "memorylayer_saas" / "storage" / "migrations" / "010_cue_anchors.sql"
    )
    assert p.exists(), f"migration not found: {p}"
    sql = p.read_text()
    assert "cue_anchors" in sql
    assert "CREATE TABLE IF NOT EXISTS cue_anchors" in sql
    assert "CREATE INDEX IF NOT EXISTS" in sql
    # HNSW vector index present.
    assert "hnsw" in sql
    assert "vector_cosine_ops" in sql


def test_migration_011_exists_and_adds_entity_id_column_and_index():
    p = (
        pathlib.Path(__file__).resolve().parents[2]
        / "src" / "memorylayer_saas" / "storage" / "migrations" / "011_cue_anchor_entity.sql"
    )
    assert p.exists(), f"migration not found: {p}"
    sql = p.read_text()
    assert "ALTER TABLE cue_anchors" in sql
    assert "ADD COLUMN IF NOT EXISTS entity_id" in sql
    assert "REFERENCES entities" in sql
    assert "ON DELETE SET NULL" in sql
    assert "CREATE INDEX IF NOT EXISTS idx_cue_anchors_entity" in sql


def test_postgres_backend_has_cue_methods():
    assert hasattr(PostgreSQLBackend, "store_cue_anchors")
    assert hasattr(PostgreSQLBackend, "search_cue_anchors")
    assert hasattr(PostgreSQLBackend, "expand_via_cues")

    store_sig = inspect.signature(PostgreSQLBackend.store_cue_anchors)
    assert list(store_sig.parameters) == ["self", "workspace_id", "memory_id", "cues"]

    search_sig = inspect.signature(PostgreSQLBackend.search_cue_anchors)
    assert list(search_sig.parameters) == ["self", "workspace_id", "query_embedding", "limit"]

    expand_sig = inspect.signature(PostgreSQLBackend.expand_via_cues)
    assert list(expand_sig.parameters) == ["self", "workspace_id", "memory_ids", "limit", "threshold"]
    assert expand_sig.parameters["limit"].default == 20
    assert expand_sig.parameters["threshold"].default == 0.85

    assert inspect.iscoroutinefunction(PostgreSQLBackend.store_cue_anchors)
    assert inspect.iscoroutinefunction(PostgreSQLBackend.search_cue_anchors)
    assert inspect.iscoroutinefunction(PostgreSQLBackend.expand_via_cues)


@pytest.mark.asyncio
async def test_store_cue_anchors_empty_is_noop():
    """store_cue_anchors with no cues short-circuits without touching the session."""

    class _Boom:
        def _session_factory(self):  # pragma: no cover - must never be called
            raise AssertionError("session factory should not be used for empty cues")

    # Call the unbound coroutine with a stub self; empty cues must return None
    # before any DB access.
    result = await PostgreSQLBackend.store_cue_anchors(_Boom(), "ws", "mem1", [])
    assert result is None
