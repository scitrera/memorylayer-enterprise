# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""
Default Tenant Service - OSS single-tenant implementation.

For OSS, always returns the default tenant. Enterprise extends this.
Moved from OSS to enterprise as tenant service is only needed for multi-tenant deployments.
"""
from typing import Optional

from logging import Logger

from scitrera_app_framework.api import Variables

from memorylayer_server.config import DEFAULT_TENANT_ID, GLOBAL_USER_WORKSPACE_ID, GLOBAL_WORKSPACE_ID

from ...models.tenant import Tenant, TenantSettings
from .base import TenantService, TenantServicePluginBase


class DefaultTenantService(TenantService):
    """Default tenant service for OSS - returns hardcoded default tenant."""

    def __init__(self, workspace_service=None):
        self._default_tenant = Tenant(
            id=DEFAULT_TENANT_ID,
            name="Default Tenant",
            settings=TenantSettings()
        )
        self._workspace_service = workspace_service
        self._global_workspace_initialized = False

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID. In OSS mode, only the default tenant exists."""
        if tenant_id == DEFAULT_TENANT_ID:
            return self._default_tenant
        return None

    async def get_default_tenant(self) -> Tenant:
        """Get the default tenant."""
        await self._ensure_global_workspace()
        return self._default_tenant

    async def ensure_default_tenant(self) -> Tenant:
        """Ensure default tenant exists. In OSS mode, always exists in memory."""
        await self._ensure_global_workspace()
        return self._default_tenant

    async def _ensure_global_workspace(self):
        """Ensure the reserved _global and _global_user workspaces exist.

        Mirrors the SQLite bootstrap (storage/sqlite.py::_ensure_reserved_entities):
        - ``_global`` (GLOBAL_WORKSPACE_ID): tenant-wide shared memories.
        - ``_global_user`` (GLOBAL_USER_WORKSPACE_ID): per-user cross-workspace
          memories (USER scope), partitioned by user_id and recalled via the
          include_global_user fan-out. Without this row, USER-scope writes and
          the recall fan-out would FK-reject / find nothing on PostgreSQL.

        Idempotent: skips creation when each workspace already exists.
        """
        if self._global_workspace_initialized or not self._workspace_service:
            return

        try:
            from memorylayer_server.models.workspace import Workspace

            # Check if _global workspace exists
            existing = await self._workspace_service.get_workspace(GLOBAL_WORKSPACE_ID)
            if not existing:
                # Create _global workspace. Workspace.settings is a dict[str, Any]
                # (see models/workspace.py), so pass a plain dict.
                global_workspace = Workspace(
                    id=GLOBAL_WORKSPACE_ID,
                    tenant_id=DEFAULT_TENANT_ID,
                    name="Global Workspace",
                    # description="Shared workspace accessible by all workspaces",
                    settings={},
                )
                await self._workspace_service.create_workspace(global_workspace)

            # Check if _global_user workspace exists (per-user cross-workspace
            # memories: stored with a user_id filter, recalled via
            # include_global_user=True).
            existing_user = await self._workspace_service.get_workspace(GLOBAL_USER_WORKSPACE_ID)
            if not existing_user:
                global_user_workspace = Workspace(
                    id=GLOBAL_USER_WORKSPACE_ID,
                    tenant_id=DEFAULT_TENANT_ID,
                    name="Global User Workspace",
                    settings={},
                )
                await self._workspace_service.create_workspace(global_user_workspace)

            self._global_workspace_initialized = True
        except Exception as e:
            # May fail if workspace service not available yet (during initialization)
            import logging
            logging.getLogger(__name__).warning("Failed to ensure global workspace: %s", e)


class DefaultTenantServicePlugin(TenantServicePluginBase):
    """Default tenant service plugin."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        # Import here to avoid circular dependency
        from memorylayer_server.services.workspace import get_workspace_service
        try:
            workspace_service = get_workspace_service(v)
        except Exception:
            # Workspace service may not be available during early initialization
            logger.debug("Workspace service not yet available during tenant plugin init")
            workspace_service = None
        return DefaultTenantService(workspace_service=workspace_service)
