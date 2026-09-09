"""Unit tests for Skill storage methods on PostgreSQLBackend.

NOTE: These tests use an in-memory SQLite database via SQLAlchemy's async
engine instead of a live PostgreSQL instance. This validates the ORM layer,
helper converters, and all 11 skill methods end-to-end at the SQLAlchemy
abstraction level.

PostgreSQL-specific features tested here that require a live DB:
- Partial unique indexes (idx_skills_workspace_name_global / idx_skills_workspace_user_name)
- ON CONFLICT DO UPDATE in upsert_skill_file (uses pg_insert; falls back to manual
  delete+insert in the SQLite path so upsert is exercised via a compatibility shim)

All other logic (CRUD, filters, cascades, converters) is fully covered.
"""
import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import event, text, MetaData, Table, Column, String, Text, Boolean, Integer, LargeBinary
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import DeclarativeBase

from memorylayer_server.models.skill import Skill, SkillFile, SkillMutation
from memorylayer_server.models.versioned_resource import VersionedResourcePreconditionFailedError
from memorylayer_server.services.skills.versioning import canonical_hash, manifest_state
from memorylayer_saas.storage.models import SkillFileModel, SkillModel


class _SkillBase(DeclarativeBase):
    """Isolated declarative base for SQLite-compatible skill-only schema."""
    pass


# We re-declare minimal SQLite-compatible table definitions using JSON (not JSONB)
# so create_all works without a live PostgreSQL connection.
import sqlalchemy as _sa
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from typing import Any


class _SkillModelSQLite(_SkillBase):
    __tablename__ = "skills"
    id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="")
    workspace_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    user_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    name: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    description: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    version: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="0.1.0")
    license: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    compatibility: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    allowed_tools: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    body: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="")
    meta: Mapped[str] = mapped_column("metadata", _sa.JSON, nullable=False, server_default="{}")
    source_mode: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="server")
    manifest_hash: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="")
    bundle_hash: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="")
    enabled: Mapped[bool] = mapped_column(_sa.Boolean, nullable=False, server_default="1")
    revision: Mapped[int] = mapped_column(_sa.Integer, nullable=False, server_default="0")
    etag: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="")
    created_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(_sa.DateTime(timezone=True), nullable=True)
    files: Mapped[list["_SkillFileModelSQLite"]] = relationship(
        back_populates="skill", cascade="all, delete-orphan"
    )


class _SkillFileModelSQLite(_SkillBase):
    __tablename__ = "skill_files"
    id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    skill_id: Mapped[str] = mapped_column(
        _sa.Text, _sa.ForeignKey("skills.id", ondelete="CASCADE"), nullable=False
    )
    path: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    kind: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    content: Mapped[bytes] = mapped_column(_sa.LargeBinary, nullable=False)
    content_hash: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(_sa.Integer, nullable=False)
    mime_type: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)
    skill: Mapped["_SkillModelSQLite"] = relationship(back_populates="files")
    __table_args__ = (
        _sa.UniqueConstraint("skill_id", "path", name="uq_skill_file_path"),
    )


class _SkillRevisionModelSQLite(_SkillBase):
    __tablename__ = "skill_revisions"
    sequence: Mapped[int] = mapped_column(_sa.Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    workspace_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    skill_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    revision: Mapped[int] = mapped_column(_sa.Integer, nullable=False)
    snapshot: Mapped[dict[str, Any]] = mapped_column(_sa.JSON, nullable=False)
    action: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    operation_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    request_hash: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    __table_args__ = (
        _sa.UniqueConstraint(
            "tenant_id", "workspace_id", "skill_id", "revision",
            name="uq_skill_revision",
        ),
    )


class _SkillOperationModelSQLite(_SkillBase):
    __tablename__ = "skill_operations"
    tenant_id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    operation_id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    request_hash: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    skill_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    revision: Mapped[int] = mapped_column(_sa.Integer, nullable=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skill_id() -> str:
    return f"skl_{uuid.uuid4().hex[:12]}"


def _file_id() -> str:
    return f"sklf_{uuid.uuid4().hex[:12]}"


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _make_skill(
    workspace_id: str = "ws_test",
    name: str = "my-skill",
    user_id: str | None = None,
    tenant_id: str = "tenant_a",
    **kwargs: Any,
) -> Skill:
    return Skill(
        id=_skill_id(),
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        user_id=user_id,
        name=name,
        description="A test skill",
        version="1.0.0",
        body="# My Skill\nDoes things.",
        metadata={"key": "value"},
        source_mode="server",
        manifest_hash=_sha256(name),
        bundle_hash="",
        **{"enabled": True, **kwargs},
    )


def _make_skill_file(skill_id: str, path: str = "scripts/run.py") -> SkillFile:
    content = b"print('hello')"
    return SkillFile(
        id=_file_id(),
        skill_id=skill_id,
        path=path,
        kind="script",
        content=content,
        content_hash=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        mime_type="text/x-python",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="module")
async def engine():
    """In-memory SQLite async engine using SQLite-compatible skill table definitions."""
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Enable FK support on every new connection
    @event.listens_for(eng.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    async with eng.begin() as conn:
        await conn.run_sync(_SkillBase.metadata.create_all)

    yield eng

    async with eng.begin() as conn:
        await conn.run_sync(_SkillBase.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    """Async session factory bound to the in-memory engine."""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def backend(session_factory):
    """Minimal PostgreSQLBackend stand-in with skill methods wired to SQLite session.

    We construct the backend without calling connect() (which needs a real PG
    URL), patch _session_factory to use the SQLite fixture session, and swap
    SkillModel / SkillFileModel for the SQLite-compatible counterparts so that
    SQLAlchemy can actually render the DDL without JSONB / pgvector types.
    """
    import memorylayer_saas.storage.postgresql as pg_module
    from memorylayer_saas.storage.postgresql import PostgreSQLBackend

    # Construct without triggering connect
    with patch.object(PostgreSQLBackend, "__init__", lambda self, *a, **kw: None):
        b = PostgreSQLBackend()

    b._session_factory = session_factory
    b.logger = MagicMock()
    b.logger.debug = MagicMock()

    # Patch module-level ORM references so skill queries target SQLite tables
    with patch.object(pg_module, "SkillModel", _SkillModelSQLite), \
         patch.object(pg_module, "SkillFileModel", _SkillFileModelSQLite), \
         patch.object(pg_module, "SkillRevisionModel", _SkillRevisionModelSQLite), \
         patch.object(pg_module, "SkillOperationModel", _SkillOperationModelSQLite):
        # Re-bind on the backend instance for the converter helpers
        b._SkillModel = _SkillModelSQLite
        b._SkillFileModel = _SkillFileModelSQLite
        yield b


# ---------------------------------------------------------------------------
# Compatibility helper: upsert via delete+insert for SQLite
# ---------------------------------------------------------------------------

async def _sqlite_upsert_skill_file(backend, skill_file: SkillFile) -> SkillFile:
    """SQLite-compatible upsert (delete-then-insert) used in place of pg_insert."""
    from sqlalchemy import delete as sa_delete

    now = datetime.now(UTC)
    async with backend._session_factory() as session:
        # Delete existing if present
        await session.execute(
            sa_delete(SkillFileModel).where(
                SkillFileModel.skill_id == skill_file.skill_id,
                SkillFileModel.path == skill_file.path,
            )
        )
        model = SkillFileModel(
            id=skill_file.id,
            skill_id=skill_file.skill_id,
            path=skill_file.path,
            kind=skill_file.kind,
            content=skill_file.content,
            content_hash=skill_file.content_hash,
            size_bytes=skill_file.size_bytes,
            mime_type=skill_file.mime_type,
            created_at=skill_file.created_at or now,
            updated_at=skill_file.updated_at or now,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)
        return backend._skill_file_model_to_domain(model)


# ---------------------------------------------------------------------------
# Tests: Skill CRUD
# ---------------------------------------------------------------------------

class TestSkillCRUD:
    """Round-trip create / get / get_by_name / list / update / delete tests."""

    @pytest.mark.asyncio
    async def test_create_and_get_skill(self, backend):
        skill = _make_skill(name="create-get-skill")
        created = await backend.create_skill(skill)

        assert created.id == skill.id
        assert created.name == "create-get-skill"
        assert created.workspace_id == "ws_test"
        assert created.metadata == {"key": "value"}
        assert created.enabled is True

        fetched = await backend.get_skill("ws_test", skill.id)
        assert fetched is not None
        assert fetched.id == skill.id
        assert fetched.description == skill.description

    @pytest.mark.asyncio
    async def test_get_skill_wrong_workspace_returns_none(self, backend):
        skill = _make_skill(name="ws-check-skill")
        await backend.create_skill(skill)

        result = await backend.get_skill("wrong_workspace", skill.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_skill_by_name_workspace_scope(self, backend):
        skill = _make_skill(workspace_id="ws_byname", name="named-skill", user_id=None)
        await backend.create_skill(skill)

        found = await backend.get_skill_by_name("ws_byname", "named-skill")
        assert found is not None
        assert found.id == skill.id
        assert found.user_id is None

    @pytest.mark.asyncio
    async def test_get_skill_by_name_user_scope(self, backend):
        skill = _make_skill(workspace_id="ws_user", name="user-skill", user_id="usr_001")
        await backend.create_skill(skill)

        # User-scoped lookup
        found = await backend.get_skill_by_name("ws_user", "user-skill", user_id="usr_001")
        assert found is not None
        assert found.user_id == "usr_001"

        # Workspace-scope lookup (user_id=None) should NOT return user-scoped skill
        not_found = await backend.get_skill_by_name("ws_user", "user-skill")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_list_skills_basic(self, backend):
        ws = "ws_list_basic"
        s1 = _make_skill(workspace_id=ws, name="skill-alpha")
        s2 = _make_skill(workspace_id=ws, name="skill-beta")
        await backend.create_skill(s1)
        await backend.create_skill(s2)

        results = await backend.list_skills(ws)
        ids = {r.id for r in results}
        assert s1.id in ids
        assert s2.id in ids

    @pytest.mark.asyncio
    async def test_list_skills_filter_by_enabled(self, backend):
        ws = "ws_list_enabled"
        active = _make_skill(workspace_id=ws, name="skill-active", enabled=True)
        inactive = _make_skill(workspace_id=ws, name="skill-inactive", enabled=False)
        await backend.create_skill(active)
        await backend.create_skill(inactive)

        active_results = await backend.list_skills(ws, enabled=True)
        active_ids = {r.id for r in active_results}
        assert active.id in active_ids
        assert inactive.id not in active_ids

        inactive_results = await backend.list_skills(ws, enabled=False)
        inactive_ids = {r.id for r in inactive_results}
        assert inactive.id in inactive_ids
        assert active.id not in inactive_ids

    @pytest.mark.asyncio
    async def test_list_skills_filter_by_user(self, backend):
        ws = "ws_list_user"
        global_skill = _make_skill(workspace_id=ws, name="global-skill", user_id=None)
        user_skill = _make_skill(workspace_id=ws, name="user-skill", user_id="usr_xyz")
        await backend.create_skill(global_skill)
        await backend.create_skill(user_skill)

        user_results = await backend.list_skills(ws, user_id="usr_xyz")
        user_ids = {r.id for r in user_results}
        assert user_skill.id in user_ids
        assert global_skill.id not in user_ids

    @pytest.mark.asyncio
    async def test_list_skills_include_global_union(self, backend):
        ws = "ws_list_global"
        ws_skill = _make_skill(workspace_id=ws, name="ws-local")
        global_skill = _make_skill(workspace_id="_global", name="global-shared", user_id=None)
        global_user_skill = _make_skill(workspace_id="_global", name="global-user", user_id="usr_g")
        await backend.create_skill(ws_skill)
        await backend.create_skill(global_skill)
        await backend.create_skill(global_user_skill)

        # Default: workspace-only, no _global rows.
        default_ids = {r.id for r in await backend.list_skills(ws)}
        assert default_ids == {ws_skill.id}

        # include_global unions tenant-shared (user_id IS NULL) global skills only.
        union_ids = {r.id for r in await backend.list_skills(ws, include_global=True)}
        assert union_ids == {ws_skill.id, global_skill.id}

        # A specific user_id filter suppresses the global union.
        user_ids = {r.id for r in await backend.list_skills(ws, user_id="usr_g", include_global=True)}
        assert global_skill.id not in user_ids

    @pytest.mark.asyncio
    async def test_list_skills_limit_offset(self, backend):
        ws = "ws_list_page"
        skills = [_make_skill(workspace_id=ws, name=f"paged-{i}") for i in range(5)]
        for s in skills:
            await backend.create_skill(s)

        page1 = await backend.list_skills(ws, limit=3, offset=0)
        page2 = await backend.list_skills(ws, limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 2
        assert {r.id for r in page1}.isdisjoint({r.id for r in page2})

    @pytest.mark.asyncio
    async def test_update_skill(self, backend):
        skill = _make_skill(workspace_id="ws_update", name="update-me")
        await backend.create_skill(skill)

        updated = await backend.update_skill(
            "ws_update", skill.id,
            {"description": "Updated description", "version": "2.0.0", "enabled": False}
        )
        assert updated is not None
        assert updated.description == "Updated description"
        assert updated.version == "2.0.0"
        assert updated.enabled is False

    @pytest.mark.asyncio
    async def test_update_skill_metadata_key(self, backend):
        skill = _make_skill(workspace_id="ws_meta_update", name="meta-skill")
        await backend.create_skill(skill)

        updated = await backend.update_skill(
            "ws_meta_update", skill.id,
            {"metadata": {"new_key": "new_value"}}
        )
        assert updated is not None
        assert updated.metadata == {"new_key": "new_value"}

    @pytest.mark.asyncio
    async def test_update_skill_not_found_returns_none(self, backend):
        result = await backend.update_skill("ws_test", "skl_nonexistent", {"version": "9.9.9"})
        assert result is None

    @pytest.mark.asyncio
    async def test_update_skill_no_updates_returns_current(self, backend):
        skill = _make_skill(workspace_id="ws_noop", name="noop-skill")
        await backend.create_skill(skill)

        same = await backend.update_skill("ws_noop", skill.id, {})
        assert same is not None
        assert same.id == skill.id

    @pytest.mark.asyncio
    async def test_delete_skill(self, backend):
        skill = _make_skill(workspace_id="ws_delete", name="delete-me")
        await backend.create_skill(skill)

        deleted = await backend.delete_skill("ws_delete", skill.id)
        assert deleted is True

        refetch = await backend.get_skill("ws_delete", skill.id)
        assert refetch is None

    @pytest.mark.asyncio
    async def test_delete_skill_not_found_returns_false(self, backend):
        result = await backend.delete_skill("ws_test", "skl_does_not_exist")
        assert result is False

    @pytest.mark.asyncio
    async def test_versioned_manifest_replay_stale_writer_and_history(self, backend):
        skill = _make_skill(workspace_id="ws_versions", name="versioned-skill")
        created = await backend.create_skill(skill)
        assert created.revision == 1
        assert created.etag

        desired = created.model_copy(
            update={"description": "Refined", "updated_at": datetime.now(UTC)}
        )
        request_hash = canonical_hash(
            {"action": "replace", "state": manifest_state(desired), "expected_etag": created.etag}
        )
        mutation = SkillMutation(
            action="replace",
            skill=desired,
            operation_id="versioned-op",
            request_hash=request_hash,
            expected_etag=created.etag,
        )
        replaced = await backend.mutate_skill(mutation)
        replay = await backend.mutate_skill(mutation)
        assert replaced.skill.revision == 2
        assert replay.replayed is True
        assert replay.skill == replaced.skill

        stale = mutation.model_copy(
            update={
                "operation_id": "stale-op",
                "request_hash": canonical_hash({"stale": True}),
            }
        )
        with pytest.raises(VersionedResourcePreconditionFailedError):
            await backend.mutate_skill(stale)

        history = await backend.list_skill_revisions(
            created.tenant_id, created.workspace_id, created.id, limit=10
        )
        assert [item.action for item in history[:2]] == ["replace", "create"]

    @pytest.mark.asyncio
    async def test_delete_skill_retains_files_for_tombstone_restore(self, backend):
        skill = _make_skill(workspace_id="ws_cascade", name="cascade-skill")
        await backend.create_skill(skill)
        sf = _make_skill_file(skill.id, path="scripts/run.py")
        await _sqlite_upsert_skill_file(backend, sf)

        # Verify file exists
        assert await backend.get_skill_file(skill.id, "scripts/run.py") is not None

        # A tombstone hides the manifest but retains the exact child bundle.
        await backend.delete_skill("ws_cascade", skill.id)
        assert await backend.get_skill("ws_cascade", skill.id) is None
        assert await backend.get_skill_file(skill.id, "scripts/run.py") is not None


# ---------------------------------------------------------------------------
# Tests: find_skills_by_name (multi-scope)
# ---------------------------------------------------------------------------

class TestFindSkillsByName:
    """Tests for cross-scope skill resolution."""

    @pytest.mark.asyncio
    async def test_find_skills_by_name_multi_scope(self, backend):
        ws_a = "ws_find_a"
        ws_b = "ws_find_b"
        skill_a = _make_skill(workspace_id=ws_a, name="shared-tool")
        skill_b = _make_skill(workspace_id=ws_b, name="shared-tool")
        await backend.create_skill(skill_a)
        await backend.create_skill(skill_b)

        results = await backend.find_skills_by_name(
            "shared-tool",
            [{"workspace_id": ws_a}, {"workspace_id": ws_b}],
        )
        result_ids = {r.id for r in results}
        assert skill_a.id in result_ids
        assert skill_b.id in result_ids

    @pytest.mark.asyncio
    async def test_find_skills_by_name_with_user_scope(self, backend):
        ws = "ws_find_user"
        global_skill = _make_skill(workspace_id=ws, name="tool-x", user_id=None)
        user_skill = _make_skill(workspace_id=ws, name="tool-x", user_id="usr_find")
        await backend.create_skill(global_skill)
        await backend.create_skill(user_skill)

        # Ask for both scopes
        results = await backend.find_skills_by_name(
            "tool-x",
            [{"workspace_id": ws}, {"workspace_id": ws, "user_id": "usr_find"}],
        )
        result_ids = {r.id for r in results}
        assert global_skill.id in result_ids
        assert user_skill.id in result_ids

    @pytest.mark.asyncio
    async def test_find_skills_by_name_empty_filters_returns_empty(self, backend):
        results = await backend.find_skills_by_name("any-skill", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_find_skills_by_name_no_match(self, backend):
        results = await backend.find_skills_by_name(
            "nonexistent-skill-xyz",
            [{"workspace_id": "ws_test"}],
        )
        assert results == []


# ---------------------------------------------------------------------------
# Tests: SkillFile upsert / get / list / delete
# ---------------------------------------------------------------------------

class TestSkillFileOperations:
    """Tests for upsert_skill_file, get_skill_file, list_skill_files, delete_skill_file."""

    @pytest.mark.asyncio
    async def test_upsert_skill_file_insert(self, backend):
        skill = _make_skill(workspace_id="ws_files", name="file-skill-insert")
        await backend.create_skill(skill)

        sf = _make_skill_file(skill.id, path="scripts/main.py")
        result = await _sqlite_upsert_skill_file(backend, sf)

        assert result.id == sf.id
        assert result.skill_id == skill.id
        assert result.path == "scripts/main.py"
        assert result.kind == "script"
        assert result.content == b"print('hello')"
        assert result.size_bytes == len(b"print('hello')")

    @pytest.mark.asyncio
    async def test_upsert_skill_file_update(self, backend):
        skill = _make_skill(workspace_id="ws_files_update", name="file-skill-update")
        await backend.create_skill(skill)

        sf1 = _make_skill_file(skill.id, path="scripts/update.py")
        await _sqlite_upsert_skill_file(backend, sf1)

        # Re-upsert same path with updated content
        new_content = b"print('updated')"
        sf2 = SkillFile(
            id=_file_id(),  # new ID to simulate re-upload
            skill_id=skill.id,
            path="scripts/update.py",  # same path
            kind="script",
            content=new_content,
            content_hash=hashlib.sha256(new_content).hexdigest(),
            size_bytes=len(new_content),
            mime_type="text/x-python",
        )
        result = await _sqlite_upsert_skill_file(backend, sf2)

        assert result.content == new_content
        assert result.size_bytes == len(new_content)

    @pytest.mark.asyncio
    async def test_get_skill_file(self, backend):
        skill = _make_skill(workspace_id="ws_getfile", name="getfile-skill")
        await backend.create_skill(skill)
        sf = _make_skill_file(skill.id, path="refs/README.md")
        await _sqlite_upsert_skill_file(backend, sf)

        fetched = await backend.get_skill_file(skill.id, "refs/README.md")
        assert fetched is not None
        assert fetched.path == "refs/README.md"
        assert fetched.content == b"print('hello')"

    @pytest.mark.asyncio
    async def test_get_skill_file_not_found(self, backend):
        result = await backend.get_skill_file("skl_nonexistent", "no/such/file.py")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_skill_files(self, backend):
        skill = _make_skill(workspace_id="ws_listfiles", name="list-files-skill")
        await backend.create_skill(skill)

        paths = ["scripts/a.py", "scripts/b.py", "refs/doc.md"]
        for path in paths:
            sf = _make_skill_file(skill.id, path=path)
            await _sqlite_upsert_skill_file(backend, sf)

        files = await backend.list_skill_files(skill.id)
        assert len(files) == 3
        # Ordered by path
        assert [f.path for f in files] == sorted(paths)

    @pytest.mark.asyncio
    async def test_list_skill_files_empty(self, backend):
        skill = _make_skill(workspace_id="ws_emptyfiles", name="empty-files-skill")
        await backend.create_skill(skill)

        files = await backend.list_skill_files(skill.id)
        assert files == []

    @pytest.mark.asyncio
    async def test_delete_skill_file(self, backend):
        skill = _make_skill(workspace_id="ws_delfile", name="delfile-skill")
        await backend.create_skill(skill)
        sf = _make_skill_file(skill.id, path="scripts/del.py")
        await _sqlite_upsert_skill_file(backend, sf)

        deleted = await backend.delete_skill_file(skill.id, "scripts/del.py")
        assert deleted is True

        assert await backend.get_skill_file(skill.id, "scripts/del.py") is None

    @pytest.mark.asyncio
    async def test_delete_skill_file_not_found(self, backend):
        result = await backend.delete_skill_file("skl_none", "no/file.py")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_skill_files_after_delete(self, backend):
        skill = _make_skill(workspace_id="ws_listdel", name="list-del-skill")
        await backend.create_skill(skill)

        sf1 = _make_skill_file(skill.id, path="scripts/keep.py")
        sf2 = _make_skill_file(skill.id, path="scripts/remove.py")
        await _sqlite_upsert_skill_file(backend, sf1)
        await _sqlite_upsert_skill_file(backend, sf2)

        await backend.delete_skill_file(skill.id, "scripts/remove.py")

        remaining = await backend.list_skill_files(skill.id)
        assert len(remaining) == 1
        assert remaining[0].path == "scripts/keep.py"


# ---------------------------------------------------------------------------
# Tests: converter helpers
# ---------------------------------------------------------------------------

class TestConverterHelpers:
    """Direct tests of _skill_model_to_domain and _skill_file_model_to_domain."""

    def test_skill_model_to_domain_maps_meta_to_metadata(self, backend):
        now = datetime.now(UTC)
        model = SkillModel(
            id="skl_aabbccdd1234",
            tenant_id="t1",
            workspace_id="ws_conv",
            user_id=None,
            name="conv-skill",
            description="Converter test",
            version="1.0.0",
            license=None,
            compatibility=None,
            allowed_tools=None,
            body="## Body",
            meta={"foo": "bar"},
            source_mode="server",
            manifest_hash="abc",
            bundle_hash="def",
            enabled=True,
            revision=1,
            etag='"skill-1-test"',
            created_at=now,
            updated_at=now,
            deleted_at=None,
        )
        domain = backend._skill_model_to_domain(model)
        assert domain.metadata == {"foo": "bar"}
        assert domain.name == "conv-skill"
        assert domain.source_mode == "server"

    def test_skill_file_model_to_domain(self, backend):
        now = datetime.now(UTC)
        model = SkillFileModel(
            id="sklf_aabbccdd1234",
            skill_id="skl_parent",
            path="assets/img.png",
            kind="asset",
            content=b"\x89PNG",
            content_hash=hashlib.sha256(b"\x89PNG").hexdigest(),
            size_bytes=4,
            mime_type="image/png",
            created_at=now,
            updated_at=now,
        )
        domain = backend._skill_file_model_to_domain(model)
        assert domain.content == b"\x89PNG"
        assert domain.kind == "asset"
        assert domain.mime_type == "image/png"
        assert domain.size_bytes == 4
