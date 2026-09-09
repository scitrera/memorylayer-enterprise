"""Contract tests for Enterprise PostgreSQL/AGE RPG storage support."""

from contextlib import asynccontextmanager

import pytest
from sqlalchemy.dialects import postgresql

from memorylayer_saas.services.graph_analysis.age import _RPG_SUBTYPES
from memorylayer_saas.storage.models import MemoryAssociationModel, MemoryModel
from memorylayer_saas.storage.postgresql import PostgreSQLBackend


class _EmptyScalarResult:
    def scalars(self):
        return self

    def all(self):
        return []


class _CaptureSession:
    def __init__(self):
        self.statement = None

    async def execute(self, statement, *args, **kwargs):
        self.statement = statement
        return _EmptyScalarResult()


def test_age_uses_all_rpg_ontology_subtypes():
    assert set(_RPG_SUBTYPES) == {
        "rpg_directory",
        "rpg_file",
        "rpg_class",
        "rpg_function",
        "rpg_method",
        "rpg_component",
        "rpg_module",
        "rpg_package",
        "rpg_interface",
    }


def test_rpg_storage_indexes_are_declared_on_fresh_schema():
    memory_indexes = {index.name for index in MemoryModel.__table__.indexes}
    association_indexes = {
        index.name for index in MemoryAssociationModel.__table__.indexes
    }

    assert {
        "idx_memories_rpg_context_subtype",
        "idx_memories_rpg_metadata",
    } <= memory_indexes
    assert {
        "idx_associations_rpg_source",
        "idx_associations_rpg_target",
    } <= association_indexes


@pytest.mark.asyncio
async def test_rpg_filter_query_uses_indexable_postgres_containment():
    backend = object.__new__(PostgreSQLBackend)
    session = _CaptureSession()

    @asynccontextmanager
    async def fake_session_factory():
        yield session

    backend._session_factory = fake_session_factory

    await backend.search_memories_by_filter(
        "ws_1",
        subtypes=["rpg_file"],
        tags=["rpg", "rpg_sync_meta"],
        metadata_filter={"rpg_node_id": "_rpg_sync_sentinel"},
        context_id="ws_1:rpg",
    )

    sql = str(
        session.statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": False},
        )
    )
    assert "memories.context_id =" in sql
    assert "memories.tags @>" in sql
    assert "memories.metadata @>" in sql
