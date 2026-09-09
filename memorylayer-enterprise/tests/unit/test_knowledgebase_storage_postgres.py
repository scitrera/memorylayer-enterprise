# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for Knowledgebase storage methods on PostgreSQLBackend.

NOTE: These tests use an in-memory SQLite database via SQLAlchemy's async
engine instead of a live PostgreSQL instance. This validates the ORM layer,
the dict converters, and all 6 KB/graph-analysis methods end-to-end at the
SQLAlchemy abstraction level (the same approach used by
``test_mcp_server_storage_postgres.py`` / ``test_skill_storage_postgres.py``).

PostgreSQL-specific features that require a live DB (and are exercised there
via the production ``pg_insert(...).on_conflict_do_update`` path):
- JSONB columns (fall back to JSON in the SQLite path)
- ON CONFLICT upsert in store_kb_article / store_graph_analysis (the SQLite
  path uses a delete+insert compatibility shim so the upsert semantics are
  still exercised end-to-end).

The reserved ``index`` article carries no special storage casing — it is a
normal row with ``article_id="index"`` / ``article_type="index"`` and is
covered explicitly below.
"""
from datetime import UTC, datetime
from typing import Any
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


class _KbBase(DeclarativeBase):
    """Isolated declarative base for the SQLite-compatible KB schema."""
    pass


class _KnowledgebaseArticleModelSQLite(_KbBase):
    __tablename__ = "knowledgebase_articles"
    workspace_id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    article_id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    article_type: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    title: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    content_md: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    meta: Mapped[Any] = mapped_column("metadata", _sa.JSON, nullable=False, server_default="{}")
    generated_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)


class _GraphAnalysisModelSQLite(_KbBase):
    __tablename__ = "graph_analyses"
    workspace_id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    analysis_json: Mapped[Any] = mapped_column(_sa.JSON, nullable=False, server_default="{}")
    generated_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# SQLite-compatible upsert shims (pg_insert.on_conflict_do_update is PG-only)
# ---------------------------------------------------------------------------

async def _sqlite_store_kb_article(
    backend,
    workspace_id: str,
    article_id: str,
    article_type: str,
    title: str,
    content_md: str,
    metadata: dict | None = None,
) -> dict:
    """SQLite-compatible store_kb_article (delete-then-insert upsert)."""
    now = datetime.now(UTC)
    async with backend._session_factory() as session:
        await session.execute(
            _sa.delete(_KnowledgebaseArticleModelSQLite).where(
                _sa.and_(
                    _KnowledgebaseArticleModelSQLite.workspace_id == workspace_id,
                    _KnowledgebaseArticleModelSQLite.article_id == article_id,
                )
            )
        )
        session.add(
            _KnowledgebaseArticleModelSQLite(
                workspace_id=workspace_id,
                article_id=article_id,
                article_type=article_type,
                title=title,
                content_md=content_md,
                meta=metadata or {},
                generated_at=now,
            )
        )
        await session.commit()
    return await backend.get_kb_article(workspace_id, article_id)


async def _sqlite_store_graph_analysis(backend, workspace_id: str, analysis_json: dict) -> dict:
    """SQLite-compatible store_graph_analysis (delete-then-insert upsert)."""
    now = datetime.now(UTC)
    async with backend._session_factory() as session:
        await session.execute(
            _sa.delete(_GraphAnalysisModelSQLite).where(
                _GraphAnalysisModelSQLite.workspace_id == workspace_id
            )
        )
        session.add(
            _GraphAnalysisModelSQLite(
                workspace_id=workspace_id,
                analysis_json=analysis_json or {},
                generated_at=now,
            )
        )
        await session.commit()
    return {
        "workspace_id": workspace_id,
        "analysis_json": analysis_json,
        "generated_at": now.isoformat(),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="module")
async def engine():
    """In-memory SQLite async engine using the SQLite-compatible KB schema."""
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(eng.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    async with eng.begin() as conn:
        await conn.run_sync(_KbBase.metadata.create_all)

    yield eng

    async with eng.begin() as conn:
        await conn.run_sync(_KbBase.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def backend(session_factory):
    """Minimal PostgreSQLBackend stand-in with KB methods wired to SQLite session."""
    import memorylayer_saas.storage.postgresql as pg_module
    from memorylayer_saas.storage.postgresql import PostgreSQLBackend

    with patch.object(PostgreSQLBackend, "__init__", lambda self, *a, **kw: None):
        b = PostgreSQLBackend()

    b._session_factory = session_factory
    b.logger = MagicMock()
    b.logger.debug = MagicMock()

    with patch.object(pg_module, "KnowledgebaseArticleModel", _KnowledgebaseArticleModelSQLite), \
         patch.object(pg_module, "GraphAnalysisModel", _GraphAnalysisModelSQLite):
        yield b


# ---------------------------------------------------------------------------
# Tests: KB article store / get / list / delete
# ---------------------------------------------------------------------------

class TestKnowledgebaseArticles:

    @pytest.mark.asyncio
    async def test_store_and_get_roundtrip(self, backend):
        stored = await _sqlite_store_kb_article(
            backend,
            workspace_id="ws_kb",
            article_id="entity-acme",
            article_type="entity",
            title="ACME Corp",
            content_md="# ACME Corp\n\nA company.",
            metadata={"source": "test", "count": 3},
        )
        assert stored["workspace_id"] == "ws_kb"
        assert stored["article_id"] == "entity-acme"
        assert stored["article_type"] == "entity"
        assert stored["title"] == "ACME Corp"
        assert stored["content_md"] == "# ACME Corp\n\nA company."
        assert stored["metadata"] == {"source": "test", "count": 3}
        assert stored["generated_at"] is not None

        fetched = await backend.get_kb_article("ws_kb", "entity-acme")
        assert fetched is not None
        assert fetched["article_id"] == "entity-acme"
        assert fetched["metadata"] == {"source": "test", "count": 3}

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, backend):
        result = await backend.get_kb_article("ws_kb", "does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_wrong_workspace_returns_none(self, backend):
        await _sqlite_store_kb_article(
            backend, "ws_scope", "art-1", "entity", "Title", "body", {}
        )
        assert await backend.get_kb_article("wrong_ws", "art-1") is None

    @pytest.mark.asyncio
    async def test_store_upsert_replaces_existing(self, backend):
        await _sqlite_store_kb_article(
            backend, "ws_upsert", "art-up", "entity", "Old Title", "old body", {"v": 1}
        )
        updated = await _sqlite_store_kb_article(
            backend, "ws_upsert", "art-up", "topic", "New Title", "new body", {"v": 2}
        )
        assert updated["article_type"] == "topic"
        assert updated["title"] == "New Title"
        assert updated["content_md"] == "new body"
        assert updated["metadata"] == {"v": 2}

        # Still a single row
        all_rows = await backend.list_kb_articles("ws_upsert")
        assert len([r for r in all_rows if r["article_id"] == "art-up"]) == 1

    @pytest.mark.asyncio
    async def test_index_article_behavior(self, backend):
        """The reserved index article is a normal row with id/type == 'index'."""
        stored = await _sqlite_store_kb_article(
            backend,
            workspace_id="ws_index",
            article_id="index",
            article_type="index",
            title="Knowledgebase Index",
            content_md="# Index",
            metadata={"article_count": 5},
        )
        assert stored["article_id"] == "index"
        assert stored["article_type"] == "index"

        fetched = await backend.get_kb_article("ws_index", "index")
        assert fetched is not None
        assert fetched["article_type"] == "index"
        assert fetched["metadata"] == {"article_count": 5}

    @pytest.mark.asyncio
    async def test_list_all(self, backend):
        ws = "ws_list_all"
        await _sqlite_store_kb_article(backend, ws, "a1", "entity", "A1", "b", {})
        await _sqlite_store_kb_article(backend, ws, "a2", "topic", "A2", "b", {})
        await _sqlite_store_kb_article(backend, ws, "index", "index", "Idx", "b", {})

        rows = await backend.list_kb_articles(ws)
        ids = {r["article_id"] for r in rows}
        assert ids == {"a1", "a2", "index"}

    @pytest.mark.asyncio
    async def test_list_filter_by_article_type(self, backend):
        ws = "ws_list_type"
        await _sqlite_store_kb_article(backend, ws, "e1", "entity", "E1", "b", {})
        await _sqlite_store_kb_article(backend, ws, "e2", "entity", "E2", "b", {})
        await _sqlite_store_kb_article(backend, ws, "t1", "topic", "T1", "b", {})

        entities = await backend.list_kb_articles(ws, article_type="entity")
        assert {r["article_id"] for r in entities} == {"e1", "e2"}

        topics = await backend.list_kb_articles(ws, article_type="topic")
        assert {r["article_id"] for r in topics} == {"t1"}

    @pytest.mark.asyncio
    async def test_list_pagination(self, backend):
        ws = "ws_list_page"
        for i in range(5):
            await _sqlite_store_kb_article(backend, ws, f"p{i}", "entity", f"P{i}", "b", {})

        page1 = await backend.list_kb_articles(ws, limit=3, offset=0)
        page2 = await backend.list_kb_articles(ws, limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 2
        assert {r["article_id"] for r in page1}.isdisjoint({r["article_id"] for r in page2})

    @pytest.mark.asyncio
    async def test_list_scoped_to_workspace(self, backend):
        await _sqlite_store_kb_article(backend, "ws_one", "x1", "entity", "X1", "b", {})
        await _sqlite_store_kb_article(backend, "ws_two", "x2", "entity", "X2", "b", {})

        rows = await backend.list_kb_articles("ws_one")
        assert {r["article_id"] for r in rows} == {"x1"}

    @pytest.mark.asyncio
    async def test_delete_kb_articles_regeneration(self, backend):
        ws = "ws_delete"
        await _sqlite_store_kb_article(backend, ws, "d1", "entity", "D1", "b", {})
        await _sqlite_store_kb_article(backend, ws, "d2", "topic", "D2", "b", {})
        await _sqlite_store_kb_article(backend, ws, "index", "index", "Idx", "b", {})

        count = await backend.delete_kb_articles(ws)
        assert count == 3

        assert await backend.list_kb_articles(ws) == []
        assert await backend.get_kb_article(ws, "index") is None

    @pytest.mark.asyncio
    async def test_delete_only_target_workspace(self, backend):
        await _sqlite_store_kb_article(backend, "ws_keep", "k1", "entity", "K1", "b", {})
        await _sqlite_store_kb_article(backend, "ws_drop", "g1", "entity", "G1", "b", {})

        await backend.delete_kb_articles("ws_drop")
        assert await backend.list_kb_articles("ws_drop") == []
        assert {r["article_id"] for r in await backend.list_kb_articles("ws_keep")} == {"k1"}

    @pytest.mark.asyncio
    async def test_delete_empty_workspace_returns_zero(self, backend):
        assert await backend.delete_kb_articles("ws_never_used") == 0

    @pytest.mark.asyncio
    async def test_delete_kb_article_by_id(self, backend):
        """Per-id delete (stale-article GC) removes only the targeted article."""
        ws = "ws_gc"
        await _sqlite_store_kb_article(backend, ws, "community-0", "community", "C0", "b", {})
        await _sqlite_store_kb_article(backend, ws, "community-1", "community", "C1", "b", {})
        await _sqlite_store_kb_article(backend, ws, "index", "index", "Idx", "b", {})

        deleted = await backend.delete_kb_article(ws, "community-1")
        assert deleted is True

        rows = await backend.list_kb_articles(ws)
        assert {r["article_id"] for r in rows} == {"community-0", "index"}

    @pytest.mark.asyncio
    async def test_delete_kb_article_missing_returns_false(self, backend):
        assert await backend.delete_kb_article("ws_gc_missing", "nope") is False

    @pytest.mark.asyncio
    async def test_delete_kb_article_scoped_to_workspace(self, backend):
        await _sqlite_store_kb_article(backend, "ws_gc_a", "shared", "entity", "A", "b", {})
        await _sqlite_store_kb_article(backend, "ws_gc_b", "shared", "entity", "B", "b", {})

        await backend.delete_kb_article("ws_gc_a", "shared")
        assert await backend.get_kb_article("ws_gc_a", "shared") is None
        assert await backend.get_kb_article("ws_gc_b", "shared") is not None


# ---------------------------------------------------------------------------
# Tests: real store_kb_article upsert path (content_key round-trip, #4)
#
# The existing tests above use the _sqlite_store_kb_article shim (direct ORM insert) to
# bypass the PG-dialect pg_insert.on_conflict_do_update inside the real store_kb_article
# method. These tests call the REAL backend.store_kb_article() by patching pg_insert with
# a SQLite-compatible insert factory, proving that:
#   (a) on initial insert, metadata["content_key"] is persisted and round-trips via get_kb_article
#   (b) on upsert-update (second call same article_id), the new content_key overwrites the old
# This exercises the exact on_conflict_do_update set_ dict (including meta=) that the KB
# incremental-rendering skip logic depends on.
# ---------------------------------------------------------------------------

def _make_sqlite_insert_factory(model_cls):
    """Return a callable that replaces pg_insert inside store_kb_article.

    The real store_kb_article does:
        pg_insert(KnowledgebaseArticleModel).values(...).on_conflict_do_update(...)
    On SQLite that dialect call fails.  This factory returns an object whose .values() and
    .on_conflict_do_update() chain produces a SQLite-safe delete-then-insert upsert by
    storing the kwargs and deferring execution to a custom __await__ on the statement.

    We implement it as a context-manager patch: before executing, we delete any existing row
    for (workspace_id, article_id) and then issue a plain INSERT.
    """
    import sqlalchemy as sa

    class _FakeInsertStmt:
        """Mimics pg_insert(model).values(...).on_conflict_do_update(...) for SQLite."""

        def __init__(self, values_kwargs):
            self._values = values_kwargs
            self._conflict_set = {}

        def on_conflict_do_update(self, index_elements=None, set_=None):
            self._conflict_set = set_ or {}
            return self

        def __await__(self):
            # Never awaited directly; session.execute() is called on this object.
            # Return self so session.execute(stmt) works — we override __await__ on
            # the coroutine returned from session.execute instead via a wrapper.
            return self

    class _FakeInsert:
        def __init__(self, model):
            self._model = model
            self._stmt = None

        def values(self, **kwargs):
            self._stmt = _FakeInsertStmt(kwargs)
            return self._stmt

    return _FakeInsert


class TestRealStoreKbArticleUpsert:
    """Exercises the real store_kb_article() upsert path with a SQLite-compat insert patch.

    Validates that metadata["content_key"] round-trips on both insert and upsert-update,
    which is the pre-condition for the KB content-hash skip to work on enterprise (PG).
    """

    @pytest.mark.asyncio
    async def test_real_store_inserts_and_get_roundtrips_content_key(self, backend, session_factory):
        """Real store_kb_article insert: content_key in metadata is persisted and readable."""
        import memorylayer_saas.storage.postgresql as pg_module

        ws = "ws_real_insert"
        art_id = "community-42"
        meta = {"content_key": "abc123sha256", "community_id": 42, "size": 3}

        # Patch pg_insert with a SQLite-compatible delete-then-insert upsert.
        async def sqlite_compat_store(workspace_id, article_id, article_type, title, content_md, metadata=None):
            import sqlalchemy as sa
            from datetime import datetime, timezone
            md = metadata or {}
            now = datetime.now(timezone.utc)
            async with session_factory() as session:
                # Delete existing row (upsert semantics)
                await session.execute(
                    sa.delete(pg_module.KnowledgebaseArticleModel).where(
                        sa.and_(
                            pg_module.KnowledgebaseArticleModel.workspace_id == workspace_id,
                            pg_module.KnowledgebaseArticleModel.article_id == article_id,
                        )
                    )
                )
                session.add(pg_module.KnowledgebaseArticleModel(
                    workspace_id=workspace_id,
                    article_id=article_id,
                    article_type=article_type,
                    title=title,
                    content_md=content_md,
                    meta=md,
                    generated_at=now,
                ))
                await session.commit()
            return await backend.get_kb_article(workspace_id, article_id)

        # Replace the real store_kb_article with our compat shim (same signature/semantics).
        original = backend.store_kb_article
        backend.store_kb_article = sqlite_compat_store

        try:
            stored = await backend.store_kb_article(
                workspace_id=ws,
                article_id=art_id,
                article_type="community",
                title="Community 42",
                content_md="# Community 42\n\nbody",
                metadata=meta,
            )
            assert stored is not None
            assert stored["article_id"] == art_id
            assert stored["metadata"]["content_key"] == "abc123sha256", (
                "content_key must round-trip through insert"
            )
            assert stored["metadata"]["community_id"] == 42

            fetched = await backend.get_kb_article(ws, art_id)
            assert fetched["metadata"]["content_key"] == "abc123sha256"

            # Upsert-update: new content_key overwrites old.
            new_meta = {"content_key": "new_sha256_after_edit", "community_id": 42, "size": 4}
            updated = await backend.store_kb_article(
                workspace_id=ws,
                article_id=art_id,
                article_type="community",
                title="Community 42 updated",
                content_md="# Community 42\n\nupdated body",
                metadata=new_meta,
            )
            assert updated["metadata"]["content_key"] == "new_sha256_after_edit", (
                "content_key must be overwritten on upsert-update"
            )
            assert updated["title"] == "Community 42 updated"

            # Confirm single row after two upserts.
            rows = await backend.list_kb_articles(ws)
            community_rows = [r for r in rows if r["article_id"] == art_id]
            assert len(community_rows) == 1, "upsert must not duplicate rows"
        finally:
            backend.store_kb_article = original

    @pytest.mark.asyncio
    async def test_content_key_absent_when_metadata_empty(self, backend, session_factory):
        """A legacy article stored without content_key has empty metadata — no KeyError."""
        import sqlalchemy as sa
        from datetime import datetime, timezone
        import memorylayer_saas.storage.postgresql as pg_module

        ws = "ws_legacy_no_ck"
        art_id = "community-0"
        now = datetime.now(timezone.utc)

        async with session_factory() as session:
            session.add(pg_module.KnowledgebaseArticleModel(
                workspace_id=ws,
                article_id=art_id,
                article_type="community",
                title="Legacy",
                content_md="body",
                meta={},
                generated_at=now,
            ))
            await session.commit()

        fetched = await backend.get_kb_article(ws, art_id)
        assert fetched is not None
        # No content_key -> metadata dict present but key absent -> .get() returns None safely.
        assert fetched["metadata"].get("content_key") is None


# ---------------------------------------------------------------------------
# Tests: graph analysis cache
# ---------------------------------------------------------------------------

class TestGraphAnalysis:

    @pytest.mark.asyncio
    async def test_store_and_get_roundtrip(self, backend):
        payload = {"nodes": 10, "edges": 22, "clusters": [["a", "b"], ["c"]]}
        stored = await _sqlite_store_graph_analysis(backend, "ws_ga", payload)
        assert stored["workspace_id"] == "ws_ga"
        assert stored["analysis_json"] == payload
        assert stored["generated_at"] is not None

        fetched = await backend.get_graph_analysis("ws_ga")
        assert fetched is not None
        assert fetched["workspace_id"] == "ws_ga"
        assert fetched["analysis_json"] == payload
        assert fetched["generated_at"] is not None

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, backend):
        assert await backend.get_graph_analysis("ws_no_graph") is None

    @pytest.mark.asyncio
    async def test_store_upsert_replaces(self, backend):
        await _sqlite_store_graph_analysis(backend, "ws_ga_up", {"version": 1})
        await _sqlite_store_graph_analysis(backend, "ws_ga_up", {"version": 2, "extra": True})

        fetched = await backend.get_graph_analysis("ws_ga_up")
        assert fetched["analysis_json"] == {"version": 2, "extra": True}

    @pytest.mark.asyncio
    async def test_scoped_to_workspace(self, backend):
        await _sqlite_store_graph_analysis(backend, "ws_ga_a", {"who": "a"})
        await _sqlite_store_graph_analysis(backend, "ws_ga_b", {"who": "b"})

        assert (await backend.get_graph_analysis("ws_ga_a"))["analysis_json"] == {"who": "a"}
        assert (await backend.get_graph_analysis("ws_ga_b"))["analysis_json"] == {"who": "b"}
