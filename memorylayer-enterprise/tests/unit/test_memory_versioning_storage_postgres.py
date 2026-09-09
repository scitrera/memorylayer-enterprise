"""SQLite-backed contract tests for PostgreSQL semantic-memory versioning."""

import hashlib
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from memorylayer_server.models.memory import MemoryMutation, MemoryType, RememberInput
from memorylayer_server.models.versioned_resource import VersionedResourceConflictError
from memorylayer_server.services.memory.versioning import memory_semantic_state
from memorylayer_server.services.skills.versioning import canonical_hash
from memorylayer_saas.models.memory import Memory


class _MemoryBase(DeclarativeBase):
    pass


class _MemoryModelSQLite(_MemoryBase):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    tenant_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    context_id: Mapped[str | None] = mapped_column(sa.Text)
    user_id: Mapped[str | None] = mapped_column(sa.Text)
    logical_key: Mapped[str | None] = mapped_column(sa.Text)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(sa.Text, nullable=False)
    type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    subtype: Mapped[str | None] = mapped_column(sa.Text)
    importance: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.5)
    tags: Mapped[list] = mapped_column(sa.JSON, nullable=False, default=list)
    meta: Mapped[dict] = mapped_column("metadata", sa.JSON, nullable=False, default=dict)
    refinement_meta: Mapped[dict] = mapped_column(
        "refinement_metadata", sa.JSON, nullable=False, default=dict
    )
    abstract: Mapped[str | None] = mapped_column(sa.Text)
    overview: Mapped[str | None] = mapped_column(sa.Text)
    session_id: Mapped[str | None] = mapped_column(sa.Text)
    source_memory_id: Mapped[str | None] = mapped_column(sa.Text)
    category: Mapped[str | None] = mapped_column(sa.Text)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, default="active")
    pinned: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)
    observer_id: Mapped[str | None] = mapped_column(sa.Text)
    subject_id: Mapped[str | None] = mapped_column(sa.Text)
    source_document_id: Mapped[str | None] = mapped_column(sa.Text)
    source_page_id: Mapped[str | None] = mapped_column(sa.Text)
    source_dataset_id: Mapped[str | None] = mapped_column(sa.Text)
    source_thread_id: Mapped[str | None] = mapped_column(sa.Text)
    embedding: Mapped[list | None] = mapped_column(sa.JSON)
    multivector: Mapped[list | None] = mapped_column(sa.JSON)
    access_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    last_accessed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    decay_factor: Mapped[float] = mapped_column(sa.Float, nullable=False, default=1.0)
    archived_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    revision: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    etag: Mapped[str] = mapped_column(sa.Text, nullable=False, default="")
    event_time: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)

    __table_args__ = (
        sa.Index(
            "uq_test_memory_global_key",
            "workspace_id",
            "logical_key",
            unique=True,
            sqlite_where=sa.text("user_id IS NULL AND logical_key IS NOT NULL"),
        ),
        sa.Index(
            "uq_test_memory_user_key",
            "workspace_id",
            "user_id",
            "logical_key",
            unique=True,
            sqlite_where=sa.text("user_id IS NOT NULL AND logical_key IS NOT NULL"),
        ),
    )


class _MemoryRevisionModelSQLite(_MemoryBase):
    __tablename__ = "memory_revisions"

    sequence: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    workspace_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    memory_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    revision: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    snapshot: Mapped[dict] = mapped_column(sa.JSON, nullable=False)
    action: Mapped[str] = mapped_column(sa.Text, nullable=False)
    operation_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    request_hash: Mapped[str] = mapped_column(sa.Text, nullable=False)

    __table_args__ = (
        sa.UniqueConstraint(
            "tenant_id", "workspace_id", "memory_id", "revision"
        ),
    )


class _MemoryOperationModelSQLite(_MemoryBase):
    __tablename__ = "memory_operations"

    tenant_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    operation_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    request_hash: Mapped[str] = mapped_column(sa.Text, nullable=False)
    memory_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    revision: Mapped[int] = mapped_column(sa.Integer, nullable=False)


class _SessionContextEventModelSQLite(_MemoryBase):
    __tablename__ = "session_context_events"

    sequence: Mapped[int] = mapped_column(
        sa.Integer, primary_key=True, autoincrement=True
    )
    workspace_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    session_id: Mapped[str | None] = mapped_column(sa.Text)
    event_kind: Mapped[str] = mapped_column(sa.Text, nullable=False)
    subject_kind: Mapped[str] = mapped_column(sa.Text, nullable=False)
    subject_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    event_time: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False
    )
    meta: Mapped[dict] = mapped_column(
        "metadata", sa.JSON, nullable=False, default=dict
    )


class _EntityRelationEvidenceModelSQLite(_MemoryBase):
    __tablename__ = "entity_relation_evidence"

    id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    relation_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    source_memory_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    active: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=True)
    confidence: Mapped[float] = mapped_column(sa.Float, nullable=False, default=1.0)


@pytest_asyncio.fixture
async def backend():
    import memorylayer_saas.storage.postgresql as pg_module
    from memorylayer_saas.storage.postgresql import PostgreSQLBackend

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as connection:
        await connection.run_sync(_MemoryBase.metadata.create_all)

    with patch.object(PostgreSQLBackend, "__init__", lambda self, *a, **kw: None):
        instance = PostgreSQLBackend()
    instance._session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    instance.logger = MagicMock()
    instance._context_event_retention_days = 0

    with (
        patch.object(pg_module, "MemoryModel", _MemoryModelSQLite),
        patch.object(pg_module, "MemoryRevisionModel", _MemoryRevisionModelSQLite),
        patch.object(pg_module, "MemoryOperationModel", _MemoryOperationModelSQLite),
        patch.object(
            pg_module,
            "SessionContextEventModel",
            _SessionContextEventModelSQLite,
        ),
        patch.object(
            pg_module,
            "EntityRelationEvidenceModel",
            _EntityRelationEvidenceModelSQLite,
        ),
    ):
        yield instance

    await engine.dispose()


def _memory(memory_id: str, logical_key: str, content: str = "Original") -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=memory_id,
        logical_key=logical_key,
        tenant_id="tenant-a",
        workspace_id="project-a",
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        type=MemoryType.SEMANTIC,
        tags=["coding"],
        metadata={"producer": "test"},
        refinement_metadata={"audience": "agents"},
        created_at=now,
        updated_at=now,
    )


def _mutation(action: str, memory: Memory, operation_id: str, expected_etag: str):
    request_hash = canonical_hash(
        {
            "action": action,
            "state": memory_semantic_state(memory),
            "expected_etag": expected_etag,
        }
    )
    return MemoryMutation(
        action=action,
        memory=memory,
        operation_id=operation_id,
        request_hash=request_hash,
        expected_etag=expected_etag,
    )


@pytest.mark.asyncio
async def test_memory_cas_replay_history_and_derived_updates(backend):
    create = _mutation("create", _memory("mem-1", "policy"), "create-op", "*")
    created = await backend.mutate_memory(create)
    assert created.memory.revision == 1
    assert created.memory.etag.startswith('"memory-1-')

    replay = await backend.mutate_memory(create)
    assert replay.replayed is True
    assert replay.memory.etag == created.memory.etag

    old_etag = created.memory.etag
    derived = await backend.update_memory(
        "project-a", "mem-1", importance=0.9, metadata={"producer": "worker"}
    )
    assert derived.revision == 1
    assert derived.etag == old_etag
    assert derived.importance == 0.9
    assert derived.metadata == {"producer": "worker"}

    replacement = created.memory.model_copy(
        update={
            "content": "Replacement",
            "content_hash": hashlib.sha256(b"Replacement").hexdigest(),
            "updated_at": datetime.now(UTC),
        }
    )
    replaced = await backend.mutate_memory(
        _mutation("replace", replacement, "replace-op", old_etag)
    )
    assert replaced.memory.revision == 2
    assert replaced.memory.content == "Replacement"

    revisions = await backend.list_memory_revisions(
        "tenant-a", "project-a", "mem-1", limit=10
    )
    assert [revision.action for revision in revisions] == ["replace", "create"]
    assert revisions[-1].memory.embedding is None


@pytest.mark.asyncio
async def test_memory_tombstone_restore_and_scoped_key_reservation(backend):
    created = await backend.mutate_memory(
        _mutation("create", _memory("mem-2", "reserved"), "create-op", "*")
    )
    deleted_memory = created.memory.model_copy(
        update={"deleted_at": datetime.now(UTC), "updated_at": datetime.now(UTC)}
    )
    deleted = await backend.mutate_memory(
        _mutation("delete", deleted_memory, "delete-op", created.memory.etag)
    )
    assert deleted.memory.deleted_at is not None
    assert await backend.get_memory("project-a", "mem-2", track_access=False) is None
    tombstone = await backend.get_memory(
        "project-a", "mem-2", track_access=False, include_deleted=True
    )
    assert tombstone is not None

    with pytest.raises(VersionedResourceConflictError):
        await backend.mutate_memory(
            _mutation(
                "create", _memory("mem-3", "reserved"), "conflict-op", "*"
            )
        )

    restored_memory = deleted.memory.model_copy(
        update={"deleted_at": None, "updated_at": datetime.now(UTC)}
    )
    restored = await backend.mutate_memory(
        _mutation("restore", restored_memory, "restore-op", deleted.memory.etag)
    )
    assert restored.memory.deleted_at is None
    assert restored.memory.revision == 3


@pytest.mark.asyncio
async def test_legacy_create_path_writes_initial_revision(backend):
    created = await backend.create_memory(
        "project-b",
        RememberInput(
            tenant_id="tenant-a",
            logical_key="coding/conventions",
            content="Run focused tests.",
            type=MemoryType.PROCEDURAL,
            refinement_metadata={"source": "refinement"},
            pinned=True,
        ),
    )
    assert created.revision == 1
    assert created.logical_key == "coding/conventions"
    assert created.pinned is True
    revisions = await backend.list_memory_revisions(
        "tenant-a", "project-b", created.id, limit=10
    )
    assert len(revisions) == 1
    assert revisions[0].memory.refinement_metadata == {"source": "refinement"}
