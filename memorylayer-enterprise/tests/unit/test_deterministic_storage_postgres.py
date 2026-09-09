# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""PostgreSQL parity checks for deterministic-memory storage contracts."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
from memorylayer_server.models.entity_relation import (
    EntityRelation,
    EntityRelationEvidence,
    RelationEvidenceKind,
)
from sqlalchemy.dialects import postgresql

from memorylayer_saas.storage.models import Base
from memorylayer_saas.storage.postgresql import PostgreSQLBackend


class _Result:
    def __init__(self, *, one=None, one_or_none=None):
        self._one = one
        self._one_or_none = one_or_none

    def scalar_one(self):
        return self._one

    def scalar_one_or_none(self):
        return self._one_or_none


class _Session:
    def __init__(self, results):
        self.results = list(results)
        self.statements = []
        self.flushes = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def begin(self):
        return self

    async def execute(self, statement):
        self.statements.append(statement)
        return self.results.pop(0)

    async def flush(self):
        self.flushes += 1


def _backend_with_results(results):
    session = _Session(results)
    backend = object.__new__(PostgreSQLBackend)
    backend._session_factory = lambda: session
    return backend, session


def _relation_models(*, evidence_active=True):
    relation = EntityRelation(
        id="erel_pg_contract",
        workspace_id="ws_pg_contract",
        source_entity_id="ent_source",
        target_entity_id="ent_target",
        relationship="employed_by",
        confidence=0.9,
    )
    evidence = EntityRelationEvidence(
        id="eev_pg_contract",
        workspace_id=relation.workspace_id,
        relation_id=relation.id,
        source_memory_id="mem_evidence",
        evidence_kind=RelationEvidenceKind.EXPLICIT,
        source_span_start=0,
        source_span_end=12,
        excerpt_hash="a" * 64,
        confidence=0.9,
        extraction_method="explicit_api",
    )
    relation_model = SimpleNamespace(**relation.model_dump())
    evidence_model = SimpleNamespace(
        active=evidence_active,
        confidence=evidence.confidence,
        evidence_kind=evidence.evidence_kind.value,
        extraction_method=evidence.extraction_method,
    )
    return relation, evidence, relation_model, evidence_model


def test_revision_036_is_the_migration_head_successor():
    path = Path(__file__).parents[2] / "migrations/versions/036_deterministic_memory_storage.py"
    spec = spec_from_file_location("migration_036", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.revision == "036"
    assert module.down_revision == "035"


def test_postgres_models_publish_all_additive_tables_and_constraints():
    assert {
        "session_checkpoints",
        "session_context_events",
        "entity_relations",
        "entity_relation_evidence",
    } <= set(Base.metadata.tables)
    relation_indexes = {
        index.name: index for index in Base.metadata.tables["entity_relations"].indexes
    }
    active_edge = relation_indexes["uq_entity_relations_active_edge"]
    assert active_edge.unique is True
    assert str(active_edge.dialect_options["postgresql"]["where"]) == "active = true"
    checkpoint_constraints = {
        constraint.name
        for constraint in Base.metadata.tables["session_checkpoints"].constraints
    }
    assert "uq_session_checkpoint_idempotency" in checkpoint_constraints


def test_postgres_backend_advertises_deterministic_storage_capabilities():
    backend = object.__new__(PostgreSQLBackend)
    assert backend.supports_capability("session_checkpoints")
    assert backend.supports_capability("session_context_events")
    assert backend.supports_capability("entity_relations")
    assert not backend.supports_capability("unknown")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inserted_evidence_id", "evidence_active", "expected_duplicate"),
    [
        ("eev_pg_contract", True, False),
        (None, True, True),
        (None, False, False),
    ],
)
async def test_relation_upsert_is_conflict_safe_and_reactivates_evidence(
    inserted_evidence_id,
    evidence_active,
    expected_duplicate,
):
    relation, evidence, relation_model, evidence_model = _relation_models(
        evidence_active=evidence_active
    )
    backend, session = _backend_with_results(
        [
            _Result(one=2),
            _Result(one_or_none=evidence.source_memory_id),
            _Result(),
            _Result(one=relation_model),
            _Result(one_or_none=inserted_evidence_id),
            _Result(one_or_none=evidence_model),
            _Result(one=evidence.confidence),
        ]
    )

    stored, duplicate = await backend.upsert_entity_relation(relation, evidence)

    assert stored.id == relation.id
    assert duplicate is expected_duplicate
    assert evidence_model.active is True
    assert session.flushes == 2
    compiled_inserts = [
        str(statement.compile(dialect=postgresql.dialect()))
        for statement in (session.statements[2], session.statements[4])
    ]
    assert all("ON CONFLICT DO NOTHING" in sql for sql in compiled_inserts)


@pytest.mark.asyncio
async def test_relation_upsert_rejects_cross_workspace_evidence_before_io():
    relation, evidence, _relation_model, _evidence_model = _relation_models()
    evidence = evidence.model_copy(update={"workspace_id": "different_workspace"})
    backend, session = _backend_with_results([])

    with pytest.raises(ValueError, match="edge workspace"):
        await backend.upsert_entity_relation(relation, evidence)
    assert session.statements == []
