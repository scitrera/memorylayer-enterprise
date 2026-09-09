"""Application service for MemoryLayer Enterprise."""
import logging
from datetime import datetime, timezone
from typing import Iterable, Optional
from uuid import uuid4

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..models.application import (
    Application,
    ApplicationBundle,
    OverrideMode,
    WorkspaceApplication,
)


class ApplicationService:
    """Async CRUD service for applications and workspace associations."""

    def __init__(self, session_factory: async_sessionmaker, logger: logging.Logger = None):
        self._session_factory = session_factory
        self.logger = logger or logging.getLogger(__name__)

    # ---- Application CRUD ----

    async def create_application(self, app: Application) -> Application:
        """Create a new application."""
        from ..storage.models import ApplicationModel

        app_id = app.id or f"app_{uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)

        async with self._session_factory() as session:
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            stmt = pg_insert(ApplicationModel).values(
                id=app_id,
                tenant_id=app.tenant_id,
                name=app.name,
                description=app.description,
                app_type=app.app_type,
                enabled=app.enabled,
                config=app.config or {},
                meta=app.metadata or {},
                default_skill_names=list(app.default_skill_names or []),
                default_mcp_server_names=list(app.default_mcp_server_names or []),
                default_tool_names=list(app.default_tool_names or []),
                created_at=now,
                updated_at=now,
            ).on_conflict_do_update(
                index_elements=['id'],
                set_={
                    'name': app.name,
                    'description': app.description,
                    'app_type': app.app_type,
                    'enabled': app.enabled,
                    'config': app.config or {},
                    'metadata': app.metadata or {},
                    'default_skill_names': list(app.default_skill_names or []),
                    'default_mcp_server_names': list(app.default_mcp_server_names or []),
                    'default_tool_names': list(app.default_tool_names or []),
                    'updated_at': now,
                },
            )
            await session.execute(stmt)
            await session.commit()
            result = await session.execute(
                select(ApplicationModel).where(ApplicationModel.id == app_id)
            )
            model = result.scalar_one()
            return self._to_domain(model)

    async def get_application(self, app_id: str, tenant_id: str) -> Optional[Application]:
        """Get application by ID."""
        from ..storage.models import ApplicationModel

        async with self._session_factory() as session:
            result = await session.execute(
                select(ApplicationModel).where(
                    and_(ApplicationModel.id == app_id, ApplicationModel.tenant_id == tenant_id)
                )
            )
            model = result.scalar_one_or_none()
            return self._to_domain(model) if model else None

    async def list_applications(
        self,
        tenant_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Application], int]:
        """List applications for a tenant."""
        from ..storage.models import ApplicationModel

        async with self._session_factory() as session:
            conditions = [ApplicationModel.tenant_id == tenant_id]

            count_result = await session.execute(
                select(func.count(ApplicationModel.id)).where(and_(*conditions))
            )
            total = count_result.scalar_one()

            result = await session.execute(
                select(ApplicationModel)
                .where(and_(*conditions))
                .order_by(ApplicationModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            apps = [self._to_domain(m) for m in result.scalars().all()]
            return apps, total

    async def update_application(self, app_id: str, tenant_id: str, **updates) -> Optional[Application]:
        """Update application fields."""
        from ..storage.models import ApplicationModel

        if not updates:
            return await self.get_application(app_id, tenant_id)

        async with self._session_factory() as session:
            result = await session.execute(
                select(ApplicationModel).where(
                    and_(ApplicationModel.id == app_id, ApplicationModel.tenant_id == tenant_id)
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None

            for key, value in updates.items():
                if key == "metadata":
                    setattr(model, "meta", value)
                elif hasattr(model, key):
                    setattr(model, key, value)
            model.updated_at = datetime.now(timezone.utc)

            await session.commit()
            await session.refresh(model)
            return self._to_domain(model)

    async def delete_application(self, app_id: str, tenant_id: str) -> bool:
        """Delete an application."""
        from ..storage.models import ApplicationModel

        async with self._session_factory() as session:
            result = await session.execute(
                delete(ApplicationModel).where(
                    and_(ApplicationModel.id == app_id, ApplicationModel.tenant_id == tenant_id)
                )
            )
            await session.commit()
            return result.rowcount > 0

    # ---- Workspace association ----

    async def associate_workspace(
        self,
        workspace_id: str,
        app_id: str,
        enabled: bool = True,
        config_overrides: dict = None,
        tool_names: Optional[list[str]] = None,
        skill_override_mode: Optional[OverrideMode] = None,
        mcp_override_mode: Optional[OverrideMode] = None,
        tool_override_mode: Optional[OverrideMode] = None,
    ) -> WorkspaceApplication:
        """Associate an application with a workspace.

        Upserts the binding; only fields explicitly supplied are written so an
        upsert from a partial request (e.g. only the enable flag) does not
        clobber previously configured overrides.
        """
        from ..storage.models import WorkspaceApplicationModel

        now = datetime.now(timezone.utc)

        async with self._session_factory() as session:
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            insert_values = dict(
                workspace_id=workspace_id,
                application_id=app_id,
                enabled=enabled,
                config_overrides=config_overrides or {},
                tool_names=list(tool_names or []),
                skill_override_mode=skill_override_mode or "merge",
                mcp_override_mode=mcp_override_mode or "merge",
                tool_override_mode=tool_override_mode or "merge",
                created_at=now,
            )
            update_set = dict(enabled=enabled, config_overrides=config_overrides or {})
            if tool_names is not None:
                update_set["tool_names"] = list(tool_names)
            if skill_override_mode is not None:
                update_set["skill_override_mode"] = skill_override_mode
            if mcp_override_mode is not None:
                update_set["mcp_override_mode"] = mcp_override_mode
            if tool_override_mode is not None:
                update_set["tool_override_mode"] = tool_override_mode

            stmt = pg_insert(WorkspaceApplicationModel).values(
                **insert_values
            ).on_conflict_do_update(
                index_elements=['workspace_id', 'application_id'],
                set_=update_set,
            )
            await session.execute(stmt)
            await session.commit()
            result = await session.execute(
                select(WorkspaceApplicationModel).where(
                    and_(
                        WorkspaceApplicationModel.workspace_id == workspace_id,
                        WorkspaceApplicationModel.application_id == app_id,
                    )
                )
            )
            model = result.scalar_one()
            return self._binding_to_domain(model)

    async def disassociate_workspace(self, workspace_id: str, app_id: str) -> bool:
        """Remove an application from a workspace."""
        from ..storage.models import WorkspaceApplicationModel

        async with self._session_factory() as session:
            result = await session.execute(
                delete(WorkspaceApplicationModel).where(
                    and_(
                        WorkspaceApplicationModel.workspace_id == workspace_id,
                        WorkspaceApplicationModel.application_id == app_id,
                    )
                )
            )
            await session.commit()
            return result.rowcount > 0

    async def get_workspace_binding(
        self, workspace_id: str, app_id: str
    ) -> Optional[WorkspaceApplication]:
        """Return the workspace_applications row for a given (workspace, app), or None."""
        from ..storage.models import WorkspaceApplicationModel

        async with self._session_factory() as session:
            result = await session.execute(
                select(WorkspaceApplicationModel).where(
                    and_(
                        WorkspaceApplicationModel.workspace_id == workspace_id,
                        WorkspaceApplicationModel.application_id == app_id,
                    )
                )
            )
            model = result.scalar_one_or_none()
            return self._binding_to_domain(model) if model else None

    async def list_workspace_applications(self, workspace_id: str) -> list[Application]:
        """List all applications associated with a workspace.

        Merges the association's config_overrides into each application's
        config dict so callers can access workspace-specific settings
        (e.g. min_role) without a separate query.
        """
        from ..storage.models import ApplicationModel, WorkspaceApplicationModel

        async with self._session_factory() as session:
            result = await session.execute(
                select(ApplicationModel, WorkspaceApplicationModel.config_overrides)
                .join(
                    WorkspaceApplicationModel,
                    ApplicationModel.id == WorkspaceApplicationModel.application_id,
                )
                .where(WorkspaceApplicationModel.workspace_id == workspace_id)
            )
            apps = []
            for app_model, config_overrides in result.all():
                app = self._to_domain(app_model)
                if config_overrides:
                    app.config = {**app.config, **config_overrides}
                apps.append(app)
            return apps

    # ---- Capability bindings: skills ----

    async def _ensure_skill_in_workspace(self, session, workspace_id: str, skill_id: str) -> None:
        from ..storage.models import SkillModel

        result = await session.execute(
            select(SkillModel.id).where(
                and_(SkillModel.id == skill_id, SkillModel.workspace_id == workspace_id)
            )
        )
        if result.scalar_one_or_none() is None:
            raise ValueError(f"Skill {skill_id} not found in workspace {workspace_id}")

    async def add_skill_to_binding(
        self, workspace_id: str, app_id: str, skill_id: str
    ) -> None:
        """Attach a skill to a (workspace, app) binding.

        Requires the binding to exist (via associate_workspace) and the skill
        to belong to the same workspace. Idempotent via ON CONFLICT DO NOTHING.
        """
        from ..storage.models import WorkspaceApplicationSkillModel

        async with self._session_factory() as session:
            await self._ensure_skill_in_workspace(session, workspace_id, skill_id)

            from sqlalchemy.dialects.postgresql import insert as pg_insert
            stmt = pg_insert(WorkspaceApplicationSkillModel).values(
                workspace_id=workspace_id,
                application_id=app_id,
                skill_id=skill_id,
                created_at=datetime.now(timezone.utc),
            ).on_conflict_do_nothing(
                index_elements=['workspace_id', 'application_id', 'skill_id']
            )
            await session.execute(stmt)
            await session.commit()

    async def remove_skill_from_binding(
        self, workspace_id: str, app_id: str, skill_id: str
    ) -> bool:
        from ..storage.models import WorkspaceApplicationSkillModel

        async with self._session_factory() as session:
            result = await session.execute(
                delete(WorkspaceApplicationSkillModel).where(
                    and_(
                        WorkspaceApplicationSkillModel.workspace_id == workspace_id,
                        WorkspaceApplicationSkillModel.application_id == app_id,
                        WorkspaceApplicationSkillModel.skill_id == skill_id,
                    )
                )
            )
            await session.commit()
            return result.rowcount > 0

    async def list_binding_skill_ids(self, workspace_id: str, app_id: str) -> list[str]:
        from ..storage.models import WorkspaceApplicationSkillModel

        async with self._session_factory() as session:
            result = await session.execute(
                select(WorkspaceApplicationSkillModel.skill_id).where(
                    and_(
                        WorkspaceApplicationSkillModel.workspace_id == workspace_id,
                        WorkspaceApplicationSkillModel.application_id == app_id,
                    )
                )
            )
            return [row[0] for row in result.all()]

    # ---- Capability bindings: MCP servers ----

    async def _ensure_mcp_server_in_workspace(
        self, session, workspace_id: str, mcp_server_id: str
    ) -> None:
        from ..storage.models import McpServerModel

        result = await session.execute(
            select(McpServerModel.id).where(
                and_(
                    McpServerModel.id == mcp_server_id,
                    McpServerModel.workspace_id == workspace_id,
                )
            )
        )
        if result.scalar_one_or_none() is None:
            raise ValueError(
                f"MCP server {mcp_server_id} not found in workspace {workspace_id}"
            )

    async def add_mcp_server_to_binding(
        self, workspace_id: str, app_id: str, mcp_server_id: str
    ) -> None:
        from ..storage.models import WorkspaceApplicationMcpServerModel

        async with self._session_factory() as session:
            await self._ensure_mcp_server_in_workspace(session, workspace_id, mcp_server_id)

            from sqlalchemy.dialects.postgresql import insert as pg_insert
            stmt = pg_insert(WorkspaceApplicationMcpServerModel).values(
                workspace_id=workspace_id,
                application_id=app_id,
                mcp_server_id=mcp_server_id,
                created_at=datetime.now(timezone.utc),
            ).on_conflict_do_nothing(
                index_elements=['workspace_id', 'application_id', 'mcp_server_id']
            )
            await session.execute(stmt)
            await session.commit()

    async def remove_mcp_server_from_binding(
        self, workspace_id: str, app_id: str, mcp_server_id: str
    ) -> bool:
        from ..storage.models import WorkspaceApplicationMcpServerModel

        async with self._session_factory() as session:
            result = await session.execute(
                delete(WorkspaceApplicationMcpServerModel).where(
                    and_(
                        WorkspaceApplicationMcpServerModel.workspace_id == workspace_id,
                        WorkspaceApplicationMcpServerModel.application_id == app_id,
                        WorkspaceApplicationMcpServerModel.mcp_server_id == mcp_server_id,
                    )
                )
            )
            await session.commit()
            return result.rowcount > 0

    async def list_binding_mcp_server_ids(
        self, workspace_id: str, app_id: str
    ) -> list[str]:
        from ..storage.models import WorkspaceApplicationMcpServerModel

        async with self._session_factory() as session:
            result = await session.execute(
                select(WorkspaceApplicationMcpServerModel.mcp_server_id).where(
                    and_(
                        WorkspaceApplicationMcpServerModel.workspace_id == workspace_id,
                        WorkspaceApplicationMcpServerModel.application_id == app_id,
                    )
                )
            )
            return [row[0] for row in result.all()]

    # ---- Capability bindings: tools ----

    async def set_binding_tool_names(
        self, workspace_id: str, app_id: str, tool_names: list[str]
    ) -> None:
        """Replace the binding's tool_names list. Caller-side validation only."""
        from ..storage.models import WorkspaceApplicationModel
        from sqlalchemy import update as sa_update

        async with self._session_factory() as session:
            result = await session.execute(
                sa_update(WorkspaceApplicationModel)
                .where(
                    and_(
                        WorkspaceApplicationModel.workspace_id == workspace_id,
                        WorkspaceApplicationModel.application_id == app_id,
                    )
                )
                .values(tool_names=list(tool_names))
            )
            if result.rowcount == 0:
                raise ValueError(
                    f"Workspace-application binding not found: ws={workspace_id} app={app_id}"
                )
            await session.commit()

    # ---- Bundle loader ----

    async def get_application_bundle(
        self,
        workspace_id: str,
        app_id: str,
        tenant_id: str,
        expand: Optional[Iterable[str]] = None,
    ) -> Optional[ApplicationBundle]:
        """Return app + workspace binding, optionally with expanded capability rows.

        ``expand`` is a subset of {"skills", "mcp", "tools"}. Skills and MCP
        servers are resolved against the workspace's own rows (merge mode unions
        binding rows with workspace rows whose ``name`` matches an app default;
        replace mode returns only the binding rows). Tools are name-only —
        merge unions defaults with binding ``tool_names``; replace returns just
        the binding's tool_names.
        """
        from ..storage.models import (
            ApplicationModel,
            McpServerModel,
            SkillModel,
            WorkspaceApplicationMcpServerModel,
            WorkspaceApplicationModel,
            WorkspaceApplicationSkillModel,
        )

        expand_set = set(expand or [])

        async with self._session_factory() as session:
            # Tenant guard on the application itself.
            app_result = await session.execute(
                select(ApplicationModel).where(
                    and_(
                        ApplicationModel.id == app_id,
                        ApplicationModel.tenant_id == tenant_id,
                    )
                )
            )
            app_model = app_result.scalar_one_or_none()
            if app_model is None:
                return None
            app = self._to_domain(app_model)

            # Optional binding row.
            binding_result = await session.execute(
                select(WorkspaceApplicationModel).where(
                    and_(
                        WorkspaceApplicationModel.workspace_id == workspace_id,
                        WorkspaceApplicationModel.application_id == app_id,
                    )
                )
            )
            binding_model = binding_result.scalar_one_or_none()
            binding = self._binding_to_domain(binding_model) if binding_model else None

            bundle = ApplicationBundle(application=app, workspace_binding=binding)
            if not expand_set:
                return bundle

            skill_mode: OverrideMode = (binding.skill_override_mode if binding else "merge")
            mcp_mode: OverrideMode = (binding.mcp_override_mode if binding else "merge")
            tool_mode: OverrideMode = (binding.tool_override_mode if binding else "merge")

            if "skills" in expand_set:
                bundle.skills = await self._resolve_skills(
                    session, workspace_id, app, binding, skill_mode
                )
            if "mcp" in expand_set:
                bundle.mcp_servers = await self._resolve_mcp_servers(
                    session, workspace_id, app, binding, mcp_mode
                )
            if "tools" in expand_set:
                bundle.tool_names = self._resolve_tool_names(app, binding, tool_mode)

            return bundle

    async def _resolve_skills(
        self,
        session,
        workspace_id: str,
        app: Application,
        binding: Optional[WorkspaceApplication],
        mode: OverrideMode,
    ) -> list[dict]:
        from memorylayer_server.config import GLOBAL_WORKSPACE_ID

        from ..storage.models import SkillModel, WorkspaceApplicationSkillModel

        binding_ids: list[str] = []
        if binding is not None:
            id_result = await session.execute(
                select(WorkspaceApplicationSkillModel.skill_id).where(
                    and_(
                        WorkspaceApplicationSkillModel.workspace_id == workspace_id,
                        WorkspaceApplicationSkillModel.application_id == app.id,
                    )
                )
            )
            binding_ids = [row[0] for row in id_result.all()]

        if mode == "replace":
            # Replace mode is an explicit, workspace-scoped skill set; _global
            # skills do not fan in.
            if not binding_ids:
                return []
            where = and_(
                SkillModel.workspace_id == workspace_id,
                SkillModel.id.in_(binding_ids),
            )
        else:  # merge
            name_filter = app.default_skill_names or []
            if not binding_ids and not name_filter:
                return []
            either = []
            if binding_ids:
                either.append(SkillModel.id.in_(binding_ids))
            if name_filter:
                # Tenant defaults resolve to workspace-global rows (user_id IS NULL).
                either.append(
                    and_(
                        SkillModel.name.in_(name_filter),
                        SkillModel.user_id.is_(None),
                    )
                )
            branches = [and_(SkillModel.workspace_id == workspace_id, or_(*either))]
            # Tenant-shared _global skills apply by name across every workspace.
            if name_filter and workspace_id != GLOBAL_WORKSPACE_ID:
                branches.append(
                    and_(
                        SkillModel.workspace_id == GLOBAL_WORKSPACE_ID,
                        SkillModel.name.in_(name_filter),
                        SkillModel.user_id.is_(None),
                    )
                )
            where = or_(*branches)

        result = await session.execute(select(SkillModel).where(where))
        rows = list(result.scalars().all())
        # Workspace skills shadow same-named _global skills (workspace > global).
        ws_names = {m.name for m in rows if m.workspace_id != GLOBAL_WORKSPACE_ID}
        rows = [m for m in rows if m.workspace_id != GLOBAL_WORKSPACE_ID or m.name not in ws_names]
        return [self._skill_model_to_dict(m) for m in rows]

    async def _resolve_mcp_servers(
        self,
        session,
        workspace_id: str,
        app: Application,
        binding: Optional[WorkspaceApplication],
        mode: OverrideMode,
    ) -> list[dict]:
        from ..storage.models import McpServerModel, WorkspaceApplicationMcpServerModel

        binding_ids: list[str] = []
        if binding is not None:
            id_result = await session.execute(
                select(WorkspaceApplicationMcpServerModel.mcp_server_id).where(
                    and_(
                        WorkspaceApplicationMcpServerModel.workspace_id == workspace_id,
                        WorkspaceApplicationMcpServerModel.application_id == app.id,
                    )
                )
            )
            binding_ids = [row[0] for row in id_result.all()]

        conditions = [McpServerModel.workspace_id == workspace_id]
        if mode == "replace":
            if not binding_ids:
                return []
            conditions.append(McpServerModel.id.in_(binding_ids))
        else:  # merge
            name_filter = app.default_mcp_server_names or []
            if not binding_ids and not name_filter:
                return []
            either = []
            if binding_ids:
                either.append(McpServerModel.id.in_(binding_ids))
            if name_filter:
                either.append(
                    and_(
                        McpServerModel.name.in_(name_filter),
                        McpServerModel.user_id.is_(None),
                    )
                )
            conditions.append(or_(*either))

        result = await session.execute(select(McpServerModel).where(and_(*conditions)))
        return [self._mcp_server_model_to_dict(m) for m in result.scalars().all()]

    @staticmethod
    def _resolve_tool_names(
        app: Application,
        binding: Optional[WorkspaceApplication],
        mode: OverrideMode,
    ) -> list[str]:
        binding_names = list(binding.tool_names) if binding else []
        if mode == "replace":
            return binding_names
        # merge: deduplicate while preserving order (defaults first, then binding).
        seen: set[str] = set()
        merged: list[str] = []
        for name in list(app.default_tool_names or []) + binding_names:
            if name not in seen:
                seen.add(name)
                merged.append(name)
        return merged

    # ---- Domain mappers ----

    @staticmethod
    def _to_domain(model) -> Application:
        return Application(
            id=model.id,
            tenant_id=model.tenant_id,
            name=model.name,
            description=model.description,
            app_type=model.app_type,
            enabled=model.enabled,
            config=model.config or {},
            metadata=model.meta or {},
            default_skill_names=list(model.default_skill_names or []),
            default_mcp_server_names=list(model.default_mcp_server_names or []),
            default_tool_names=list(model.default_tool_names or []),
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def _binding_to_domain(model) -> WorkspaceApplication:
        return WorkspaceApplication(
            workspace_id=model.workspace_id,
            application_id=model.application_id,
            enabled=model.enabled,
            config_overrides=model.config_overrides or {},
            tool_names=list(model.tool_names or []),
            skill_override_mode=model.skill_override_mode or "merge",
            mcp_override_mode=model.mcp_override_mode or "merge",
            tool_override_mode=model.tool_override_mode or "merge",
            created_at=model.created_at,
        )

    @staticmethod
    def _skill_model_to_dict(m) -> dict:
        return {
            "id": m.id,
            "tenant_id": m.tenant_id,
            "workspace_id": m.workspace_id,
            "user_id": m.user_id,
            "name": m.name,
            "description": m.description,
            "version": m.version,
            "license": m.license,
            "compatibility": m.compatibility,
            "allowed_tools": m.allowed_tools,
            "metadata": m.meta or {},
            "source_mode": m.source_mode,
            "manifest_hash": m.manifest_hash,
            "bundle_hash": m.bundle_hash,
            "enabled": m.enabled,
            "created_at": m.created_at,
            "updated_at": m.updated_at,
        }

    @staticmethod
    def _mcp_server_model_to_dict(m) -> dict:
        return {
            "id": m.id,
            "tenant_id": m.tenant_id,
            "workspace_id": m.workspace_id,
            "user_id": m.user_id,
            "name": m.name,
            "description": m.description,
            "transport": m.transport,
            "command": m.command,
            "args": list(m.args or []),
            "env": dict(m.env or {}),
            "url": m.url,
            "headers": dict(m.headers or {}),
            "metadata": m.meta or {},
            "source_mode": m.source_mode,
            "manifest_hash": m.manifest_hash,
            "enabled": m.enabled,
            "created_at": m.created_at,
            "updated_at": m.updated_at,
        }
