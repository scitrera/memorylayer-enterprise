"""Unit tests for the store-phase gap primitives on PostgreSQLBackend.

Covers ``get_memory_source_page_ids`` (the set of page ids already represented
by a live memory) and ``get_document_memories``.

NOTE: These tests use an in-memory SQLite database via SQLAlchemy's async
engine instead of a live PostgreSQL instance — the SAME shim pattern as
``test_knowledgebase_storage_postgres.py``. The real ``MemoryModel`` carries
pgvector columns that SQLite cannot create, so we declare a SQLite-compatible
stand-in covering only the columns these two methods touch and patch
``pg_module.MemoryModel`` to it. This validates the actual query construction
(workspace scope, source_document/source_page filters, deleted_at exclusion,
NULL-page exclusion) end-to-end at the SQLAlchemy abstraction level.
"""
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
import sqlalchemy as _sa
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool


class _MemBase(DeclarativeBase):
    """Isolated declarative base for the SQLite-compatible memories schema."""
    pass


class _MemoryModelSQLite(_MemBase):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    tenant_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    context_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    user_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    content: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    type: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="semantic")
    subtype: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    importance: Mapped[float | None] = mapped_column(_sa.Float, nullable=True)
    tags: Mapped[object] = mapped_column(_sa.JSON, nullable=False, server_default="[]")
    meta: Mapped[object] = mapped_column("metadata", _sa.JSON, nullable=False, server_default="{}")
    abstract: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    overview: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    session_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    source_memory_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    category: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    status: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    pinned: Mapped[bool | None] = mapped_column(_sa.Boolean, nullable=True)
    observer_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    subject_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    source_document_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    source_page_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    source_dataset_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    source_thread_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    access_count: Mapped[int | None] = mapped_column(_sa.Integer, nullable=True)
    last_accessed_at: Mapped[datetime | None] = mapped_column(_sa.DateTime(timezone=True), nullable=True)
    decay_factor: Mapped[float | None] = mapped_column(_sa.Float, nullable=True)
    event_time: Mapped[datetime | None] = mapped_column(_sa.DateTime(timezone=True), nullable=True)
    # Vector columns are pgvector-only in production; here they are simple
    # nullable columns so _memory_model_to_domain (used by get_document_memories)
    # can read them without a pgvector codec.
    embedding: Mapped[object | None] = mapped_column(_sa.JSON, nullable=True)
    multivector: Mapped[object | None] = mapped_column(_sa.JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime | None] = mapped_column(_sa.DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(_sa.DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="module")
async def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(eng.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    async with eng.begin() as conn:
        await conn.run_sync(_MemBase.metadata.create_all)

    yield eng

    async with eng.begin() as conn:
        await conn.run_sync(_MemBase.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def backend(session_factory):
    import memorylayer_saas.storage.postgresql as pg_module
    from memorylayer_saas.storage.postgresql import PostgreSQLBackend

    with patch.object(PostgreSQLBackend, "__init__", lambda self, *a, **kw: None):
        b = PostgreSQLBackend()

    b._session_factory = session_factory
    b.logger = MagicMock()

    with patch.object(pg_module, "MemoryModel", _MemoryModelSQLite):
        yield b


async def _add_memory(
    session_factory, *, mem_id, workspace_id, document_id, page_id,
    deleted=False, created_offset=0, embedding=None,
):
    now = datetime.now(UTC) + timedelta(seconds=created_offset)
    async with session_factory() as session:
        session.add(_MemoryModelSQLite(
            embedding=embedding,
            id=mem_id,
            workspace_id=workspace_id,
            tenant_id="default_tenant",
            content="page content",
            content_hash="h_%s" % mem_id,
            type="semantic",
            importance=0.5,
            tags=[],
            meta={},
            status="active",
            pinned=False,
            access_count=0,
            decay_factor=1.0,
            source_document_id=document_id,
            source_page_id=page_id,
            created_at=now,
            updated_at=now,
            deleted_at=(now if deleted else None),
        ))
        await session.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetMemorySourcePageIds:

    @pytest.mark.asyncio
    async def test_returns_represented_page_ids(self, backend, session_factory):
        await _add_memory(session_factory, mem_id="m1", workspace_id="ws", document_id="doc1", page_id="page_a")
        await _add_memory(session_factory, mem_id="m2", workspace_id="ws", document_id="doc1", page_id="page_b")

        result = await backend.get_memory_source_page_ids("ws", "doc1")
        assert result == {"page_a", "page_b"}

    @pytest.mark.asyncio
    async def test_excludes_deleted(self, backend, session_factory):
        await _add_memory(session_factory, mem_id="m3", workspace_id="ws2", document_id="doc2", page_id="page_c")
        await _add_memory(
            session_factory, mem_id="m4", workspace_id="ws2", document_id="doc2",
            page_id="page_d", deleted=True,
        )

        result = await backend.get_memory_source_page_ids("ws2", "doc2")
        assert result == {"page_c"}

    @pytest.mark.asyncio
    async def test_excludes_null_page_id(self, backend, session_factory):
        await _add_memory(session_factory, mem_id="m5", workspace_id="ws3", document_id="doc3", page_id="page_e")
        await _add_memory(session_factory, mem_id="m6", workspace_id="ws3", document_id="doc3", page_id=None)

        result = await backend.get_memory_source_page_ids("ws3", "doc3")
        assert result == {"page_e"}

    @pytest.mark.asyncio
    async def test_scoped_by_workspace_and_document(self, backend, session_factory):
        await _add_memory(session_factory, mem_id="m7", workspace_id="ws4", document_id="doc4", page_id="page_f")
        # Different workspace, same document id.
        await _add_memory(session_factory, mem_id="m8", workspace_id="ws_other", document_id="doc4", page_id="page_g")
        # Same workspace, different document.
        await _add_memory(session_factory, mem_id="m9", workspace_id="ws4", document_id="doc_other", page_id="page_h")

        result = await backend.get_memory_source_page_ids("ws4", "doc4")
        assert result == {"page_f"}

    @pytest.mark.asyncio
    async def test_empty_when_no_memories(self, backend):
        result = await backend.get_memory_source_page_ids("ws_empty", "doc_empty")
        assert result == set()


class TestGetDocumentMemories:

    @pytest.mark.asyncio
    async def test_returns_live_memories_oldest_first(self, backend, session_factory):
        await _add_memory(
            session_factory, mem_id="dm2", workspace_id="ws5", document_id="doc5",
            page_id="page_y", created_offset=10,
        )
        await _add_memory(
            session_factory, mem_id="dm1", workspace_id="ws5", document_id="doc5",
            page_id="page_x", created_offset=0,
        )
        await _add_memory(
            session_factory, mem_id="dm3", workspace_id="ws5", document_id="doc5",
            page_id="page_z", deleted=True, created_offset=5,
        )

        mems = await backend.get_document_memories("ws5", "doc5")
        assert [m.id for m in mems] == ["dm1", "dm2"]


# ---------------------------------------------------------------------------
# Embedding deferral on bulk read paths
# ---------------------------------------------------------------------------

class TestEmbeddingDeferral:
    """Bulk reads must not materialize ``memories.embedding``.

    A dim-1920 vector is ~7.7 KB on the wire but ~61 KB once converted to a
    Python ``list[float]``, so a large candidate pool was the server's biggest
    per-request allocation. These paths rank in SQL and the API strips the
    vector before serializing, so nothing downstream needs it.
    """

    @pytest.mark.asyncio
    async def test_get_document_memories_does_not_load_embedding(
        self, backend, session_factory,
    ):
        """The row has an embedding; the domain object must not carry it.

        Also guards the async failure mode: if the column were merely absent
        from the SELECT but still touched during conversion, SQLAlchemy would
        emit a lazy load and raise ``MissingGreenlet`` here rather than return.
        """
        await _add_memory(
            session_factory, mem_id="e1", workspace_id="ws_emb",
            document_id="doc_emb", page_id="page_a", embedding=[0.1, 0.2, 0.3],
        )

        memories = await backend.get_document_memories("ws_emb", "doc_emb")

        assert [m.id for m in memories] == ["e1"]
        assert memories[0].embedding is None

    @pytest.mark.asyncio
    async def test_deferred_query_omits_embedding_column_from_sql(
        self, backend, session_factory, engine,
    ):
        """The column is left out of the SELECT, not fetched and discarded.

        This is the assertion that actually pins the win: dropping it after the
        fact would still pay the database and driver cost.
        """
        await _add_memory(
            session_factory, mem_id="e2", workspace_id="ws_sql",
            document_id="doc_sql", page_id="page_a", embedding=[0.4, 0.5],
        )

        seen: list[str] = []

        def _record(conn, cursor, statement, params, context, executemany):
            seen.append(statement)

        event.listen(engine.sync_engine, "before_cursor_execute", _record)
        try:
            await backend.get_document_memories("ws_sql", "doc_sql")
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", _record)

        selects = [s for s in seen if s.lstrip().upper().startswith("SELECT")]
        assert selects, "no SELECT was captured"
        assert not any("embedding" in s for s in selects), (
            "embedding column present in emitted SQL: %s" % selects
        )
        # Sanity: the query really did run against the memories table.
        assert any("memories" in s for s in selects)


class TestLoadedEmbeddingHelper:
    """``_loaded_embedding`` distinguishes 'not loaded' from 'has no vector'."""

    @pytest.mark.asyncio
    async def test_returns_list_when_column_is_loaded(self, session_factory):
        import memorylayer_saas.storage.postgresql as pg_module

        await _add_memory(
            session_factory, mem_id="h1", workspace_id="ws_h",
            document_id="doc_h", page_id="page_a", embedding=[0.5, 0.25],
        )

        async with session_factory() as session:
            model = (await session.execute(
                _sa.select(_MemoryModelSQLite).where(_MemoryModelSQLite.id == "h1")
            )).scalar_one()
            assert pg_module._loaded_embedding(model) == [0.5, 0.25]

    @pytest.mark.asyncio
    async def test_returns_none_when_column_is_deferred(self, session_factory):
        import memorylayer_saas.storage.postgresql as pg_module
        from sqlalchemy.orm import defer as _defer

        await _add_memory(
            session_factory, mem_id="h2", workspace_id="ws_h",
            document_id="doc_h", page_id="page_b", embedding=[0.5, 0.25],
        )

        async with session_factory() as session:
            model = (await session.execute(
                _sa.select(_MemoryModelSQLite)
                .where(_MemoryModelSQLite.id == "h2")
                .options(_defer(_MemoryModelSQLite.embedding))
            )).scalar_one()
            assert pg_module._loaded_embedding(model) is None

    @pytest.mark.asyncio
    async def test_returns_none_when_row_has_no_vector(self, session_factory):
        import memorylayer_saas.storage.postgresql as pg_module

        await _add_memory(
            session_factory, mem_id="h3", workspace_id="ws_h",
            document_id="doc_h", page_id="page_c", embedding=None,
        )

        async with session_factory() as session:
            model = (await session.execute(
                _sa.select(_MemoryModelSQLite).where(_MemoryModelSQLite.id == "h3")
            )).scalar_one()
            assert pg_module._loaded_embedding(model) is None
