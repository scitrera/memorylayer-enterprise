"""Unit tests for ApplicationService capability binding + bundle resolution.

Mirrors the SQLite-compatible test pattern used by test_skill_storage_postgres.py:
we redeclare SQLite-friendly versions of the application / skill / mcp_server
tables so create_all works without JSONB/ARRAY-only PG types, then bind the
service to an in-memory async SQLite engine. We exercise only logic that does
not rely on PostgreSQL-specific upsert syntax — the binding insert paths use
ON CONFLICT DO NOTHING via pg_insert and are stubbed for SQLite by direct
session.add (compatibility helper, mirroring _sqlite_upsert_skill_file).
"""
import uuid
from datetime import UTC, datetime
from typing import Any, Optional

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import StaticPool

from memorylayer_saas.models.application import Application, WorkspaceApplication
from memorylayer_saas.services.application import ApplicationService


# ---------------------------------------------------------------------------
# Isolated SQLite-compatible schema
# ---------------------------------------------------------------------------

class _Base(DeclarativeBase):
    pass


class _ApplicationModel(_Base):
    __tablename__ = "applications"
    id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    description: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    app_type: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="generic")
    enabled: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default="1")
    config: Mapped[dict] = mapped_column(sa.JSON, nullable=False, server_default="{}")
    meta: Mapped[dict] = mapped_column("metadata", sa.JSON, nullable=False, server_default="{}")
    default_skill_names: Mapped[list[str]] = mapped_column(sa.JSON, nullable=False, server_default="[]")
    default_mcp_server_names: Mapped[list[str]] = mapped_column(sa.JSON, nullable=False, server_default="[]")
    default_tool_names: Mapped[list[str]] = mapped_column(sa.JSON, nullable=False, server_default="[]")
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)


class _WorkspaceApplicationModel(_Base):
    __tablename__ = "workspace_applications"
    workspace_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    application_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    enabled: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default="1")
    config_overrides: Mapped[dict] = mapped_column(sa.JSON, nullable=False, server_default="{}")
    tool_names: Mapped[list[str]] = mapped_column(sa.JSON, nullable=False, server_default="[]")
    skill_override_mode: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="merge")
    mcp_override_mode: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="merge")
    tool_override_mode: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="merge")
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)


class _SkillModel(_Base):
    __tablename__ = "skills"
    id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="")
    workspace_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    user_id: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    description: Mapped[str] = mapped_column(sa.Text, nullable=False)
    version: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="0.1.0")
    license: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    compatibility: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    allowed_tools: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    body: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="")
    meta: Mapped[dict] = mapped_column("metadata", sa.JSON, nullable=False, server_default="{}")
    source_mode: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="server")
    manifest_hash: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="")
    bundle_hash: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="")
    enabled: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default="1")
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)


class _McpServerModel(_Base):
    __tablename__ = "mcp_servers"
    id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="_default")
    workspace_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    user_id: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    description: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    transport: Mapped[str] = mapped_column(sa.Text, nullable=False)
    command: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    args: Mapped[list[str]] = mapped_column(sa.JSON, nullable=False, server_default="[]")
    env: Mapped[dict] = mapped_column(sa.JSON, nullable=False, server_default="{}")
    url: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    headers: Mapped[dict] = mapped_column(sa.JSON, nullable=False, server_default="{}")
    meta: Mapped[dict] = mapped_column("metadata", sa.JSON, nullable=False, server_default="{}")
    source_mode: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="server")
    manifest_hash: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="")
    enabled: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default="1")
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)


class _WorkspaceApplicationSkillModel(_Base):
    __tablename__ = "workspace_application_skills"
    workspace_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    application_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    skill_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)


class _WorkspaceApplicationMcpServerModel(_Base):
    __tablename__ = "workspace_application_mcp_servers"
    workspace_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    application_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    mcp_server_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture()
async def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @sa.event.listens_for(eng.sync_engine, "connect")
    def _pragma(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    async with eng.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(_Base.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture()
async def session_factory(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture()
async def service(session_factory, monkeypatch):
    """ApplicationService bound to the SQLite session factory.

    The service module imports ORM models inside each method via
    ``from ..storage.models import ...`` — we patch that module's attributes
    so queries hit the SQLite-compatible classes declared above. The
    PostgreSQL-specific ``pg_insert`` upsert path is also patched in the
    service module so SQLite can satisfy the writes.
    """
    import memorylayer_saas.storage.models as orm_module
    import memorylayer_saas.services.application as svc_module

    # Swap each ORM symbol referenced by the service.
    for orig_name, sub in {
        "ApplicationModel": _ApplicationModel,
        "WorkspaceApplicationModel": _WorkspaceApplicationModel,
        "SkillModel": _SkillModel,
        "McpServerModel": _McpServerModel,
        "WorkspaceApplicationSkillModel": _WorkspaceApplicationSkillModel,
        "WorkspaceApplicationMcpServerModel": _WorkspaceApplicationMcpServerModel,
    }.items():
        monkeypatch.setattr(orm_module, orig_name, sub, raising=True)

    # Replace pg_insert with a SQLite-compatible shim: emulate
    # on_conflict_do_update / on_conflict_do_nothing via delete-then-insert.
    class _SQLiteUpsert:
        def __init__(self, model, values):
            self._model = model
            self._values = values
            self._update_set: Optional[dict] = None
            self._do_nothing = False
            self._index_elements: list[str] = []

        def on_conflict_do_update(self, *, index_elements, set_):
            self._index_elements = list(index_elements)
            self._update_set = set_
            return self

        def on_conflict_do_nothing(self, *, index_elements):
            self._index_elements = list(index_elements)
            self._do_nothing = True
            return self

        async def _execute(self, session):
            from sqlalchemy import select as _select
            cols = self._index_elements
            stmt = _select(self._model).where(
                *[getattr(self._model, c) == self._values[c] for c in cols]
            )
            existing = (await session.execute(stmt)).scalar_one_or_none()
            if existing is None:
                obj = self._model(**self._values)
                session.add(obj)
            elif self._do_nothing:
                return
            else:
                for k, v in (self._update_set or {}).items():
                    setattr(existing, k, v)

    # The service code calls ``await session.execute(pg_insert_stmt)``; we make
    # our shim awaitable by patching session.execute to recognise it.
    real_execute = AsyncSession.execute

    async def patched_execute(self, statement, *args, **kwargs):  # type: ignore[override]
        if isinstance(statement, _SQLiteUpsert):
            return await statement._execute(self)
        return await real_execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", patched_execute, raising=True)

    def _pg_insert_factory(model):
        def _ctor(**values):
            return _SQLiteUpsert(model, values)
        return _SQLiteUpsert  # placeholder; we'll use _Builder below

    class _Builder:
        def __init__(self, model):
            self._model = model

        def values(self, **vals):
            return _SQLiteUpsert(self._model, vals)

    def _pg_insert(model):
        return _Builder(model)

    # Patch the postgres dialect's insert symbol used inside the service.
    import sqlalchemy.dialects.postgresql as pg_dialect
    monkeypatch.setattr(pg_dialect, "insert", _pg_insert, raising=True)

    return ApplicationService(session_factory=session_factory)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _app_id() -> str:
    return f"app_{uuid.uuid4().hex[:12]}"


def _skl_id() -> str:
    return f"skl_{uuid.uuid4().hex[:12]}"


def _mcp_id() -> str:
    return f"mcp_{uuid.uuid4().hex[:12]}"


async def _insert_skill(session_factory, *, workspace_id: str, name: str, skill_id: Optional[str] = None) -> str:
    sid = skill_id or _skl_id()
    async with session_factory() as session:
        session.add(_SkillModel(
            id=sid,
            tenant_id="t1",
            workspace_id=workspace_id,
            user_id=None,
            name=name,
            description="x",
            version="1.0.0",
            license=None,
            compatibility=None,
            allowed_tools=None,
            body="",
            meta={},
            source_mode="server",
            manifest_hash="",
            bundle_hash="",
            enabled=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ))
        await session.commit()
    return sid


async def _insert_mcp_server(session_factory, *, workspace_id: str, name: str, server_id: Optional[str] = None) -> str:
    sid = server_id or _mcp_id()
    async with session_factory() as session:
        session.add(_McpServerModel(
            id=sid,
            tenant_id="t1",
            workspace_id=workspace_id,
            user_id=None,
            name=name,
            description=None,
            transport="http",
            command=None,
            args=[],
            env={},
            url="https://example.com",
            headers={},
            meta={},
            source_mode="server",
            manifest_hash="",
            enabled=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ))
        await session.commit()
    return sid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_application_round_trips_defaults(service):
    app = Application(
        id=_app_id(),
        tenant_id="t1",
        name="my-app",
        default_skill_names=["alpha", "beta"],
        default_mcp_server_names=["primary"],
        default_tool_names=["search", "code-run"],
    )
    created = await service.create_application(app)
    assert created.default_skill_names == ["alpha", "beta"]
    assert created.default_mcp_server_names == ["primary"]
    assert created.default_tool_names == ["search", "code-run"]


@pytest.mark.asyncio
async def test_associate_workspace_persists_override_modes(service):
    app = Application(id=_app_id(), tenant_id="t1", name="app2", default_tool_names=["a"])
    await service.create_application(app)

    binding = await service.associate_workspace(
        workspace_id="ws1",
        app_id=app.id,
        enabled=True,
        tool_names=["b"],
        skill_override_mode="replace",
        tool_override_mode="replace",
    )
    assert binding.tool_names == ["b"]
    assert binding.skill_override_mode == "replace"
    assert binding.tool_override_mode == "replace"
    assert binding.mcp_override_mode == "merge"


@pytest.mark.asyncio
async def test_add_skill_to_binding_rejects_cross_workspace(service, session_factory):
    app = Application(id=_app_id(), tenant_id="t1", name="app3")
    await service.create_application(app)
    await service.associate_workspace(workspace_id="ws1", app_id=app.id)

    # Skill lives in ws2, not ws1 — must be rejected.
    foreign_skill = await _insert_skill(session_factory, workspace_id="ws2", name="foreign-skill")

    with pytest.raises(ValueError, match="not found in workspace"):
        await service.add_skill_to_binding("ws1", app.id, foreign_skill)


@pytest.mark.asyncio
async def test_bundle_merge_unions_defaults_and_bindings(service, session_factory):
    app = Application(
        id=_app_id(), tenant_id="t1", name="app4",
        default_skill_names=["default-skill"],
        default_tool_names=["t1"],
    )
    await service.create_application(app)
    await service.associate_workspace(
        workspace_id="ws1", app_id=app.id, tool_names=["t2"],
    )

    # Skill named "default-skill" exists in ws1 — resolves via name match.
    default_skill_id = await _insert_skill(session_factory, workspace_id="ws1", name="default-skill")
    # Plus an extra skill bound only via the junction.
    extra_skill_id = await _insert_skill(session_factory, workspace_id="ws1", name="extra-skill")
    await service.add_skill_to_binding("ws1", app.id, extra_skill_id)

    bundle = await service.get_application_bundle(
        workspace_id="ws1", app_id=app.id, tenant_id="t1",
        expand={"skills", "tools"},
    )
    assert bundle is not None
    skill_ids = {s["id"] for s in bundle.skills}
    assert skill_ids == {default_skill_id, extra_skill_id}
    # Tools merge: defaults first, then binding additions.
    assert bundle.tool_names == ["t1", "t2"]


@pytest.mark.asyncio
async def test_bundle_replace_mode_drops_defaults(service, session_factory):
    app = Application(
        id=_app_id(), tenant_id="t1", name="app5",
        default_skill_names=["default-skill"],
        default_tool_names=["t1"],
    )
    await service.create_application(app)
    await service.associate_workspace(
        workspace_id="ws1", app_id=app.id,
        tool_names=["only-binding"],
        skill_override_mode="replace",
        tool_override_mode="replace",
    )

    # Insert the default-named skill so it WOULD match under merge mode.
    await _insert_skill(session_factory, workspace_id="ws1", name="default-skill")
    extra = await _insert_skill(session_factory, workspace_id="ws1", name="extra-skill")
    await service.add_skill_to_binding("ws1", app.id, extra)

    bundle = await service.get_application_bundle(
        workspace_id="ws1", app_id=app.id, tenant_id="t1",
        expand={"skills", "tools"},
    )
    assert bundle is not None
    assert {s["id"] for s in bundle.skills} == {extra}
    assert bundle.tool_names == ["only-binding"]


@pytest.mark.asyncio
async def test_bundle_merge_unions_global_default_skills(service, session_factory):
    """A default_skill_name present only in _global fans into the bundle (merge)."""
    app = Application(
        id=_app_id(), tenant_id="t1", name="app_global",
        default_skill_names=["shared-skill"],
    )
    await service.create_application(app)
    await service.associate_workspace(workspace_id="ws1", app_id=app.id)

    # The named default exists only as a tenant-shared _global skill.
    global_skill_id = await _insert_skill(session_factory, workspace_id="_global", name="shared-skill")

    bundle = await service.get_application_bundle(
        workspace_id="ws1", app_id=app.id, tenant_id="t1",
        expand={"skills"},
    )
    assert bundle is not None
    assert {s["id"] for s in bundle.skills} == {global_skill_id}


@pytest.mark.asyncio
async def test_bundle_workspace_skill_shadows_global(service, session_factory):
    """When a default name exists in both ws and _global, the workspace row wins."""
    app = Application(
        id=_app_id(), tenant_id="t1", name="app_shadow",
        default_skill_names=["shared-skill"],
    )
    await service.create_application(app)
    await service.associate_workspace(workspace_id="ws1", app_id=app.id)

    ws_skill_id = await _insert_skill(session_factory, workspace_id="ws1", name="shared-skill")
    await _insert_skill(session_factory, workspace_id="_global", name="shared-skill")

    bundle = await service.get_application_bundle(
        workspace_id="ws1", app_id=app.id, tenant_id="t1",
        expand={"skills"},
    )
    assert bundle is not None
    # Only the workspace-scoped row is returned; the _global duplicate is shadowed.
    assert {s["id"] for s in bundle.skills} == {ws_skill_id}


@pytest.mark.asyncio
async def test_bundle_returns_none_for_wrong_tenant(service):
    app = Application(id=_app_id(), tenant_id="t1", name="app6")
    await service.create_application(app)
    result = await service.get_application_bundle(
        workspace_id="ws1", app_id=app.id, tenant_id="other-tenant", expand=set(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_remove_skill_from_binding_returns_false_when_absent(service):
    app = Application(id=_app_id(), tenant_id="t1", name="app7")
    await service.create_application(app)
    await service.associate_workspace(workspace_id="ws1", app_id=app.id)
    removed = await service.remove_skill_from_binding("ws1", app.id, "skl_does_not_exist")
    assert removed is False


@pytest.mark.asyncio
async def test_set_binding_tool_names_requires_binding(service):
    app = Application(id=_app_id(), tenant_id="t1", name="app8")
    await service.create_application(app)
    with pytest.raises(ValueError, match="binding not found"):
        await service.set_binding_tool_names("ws-missing", app.id, ["t1"])


@pytest.mark.asyncio
async def test_mcp_binding_round_trip(service, session_factory):
    app = Application(id=_app_id(), tenant_id="t1", name="app9", default_mcp_server_names=["default-mcp"])
    await service.create_application(app)
    await service.associate_workspace(workspace_id="ws1", app_id=app.id)

    default_mcp = await _insert_mcp_server(session_factory, workspace_id="ws1", name="default-mcp")
    other_mcp = await _insert_mcp_server(session_factory, workspace_id="ws1", name="binding-mcp")
    await service.add_mcp_server_to_binding("ws1", app.id, other_mcp)

    bundle = await service.get_application_bundle(
        workspace_id="ws1", app_id=app.id, tenant_id="t1", expand={"mcp"},
    )
    assert {m["id"] for m in bundle.mcp_servers} == {default_mcp, other_mcp}
