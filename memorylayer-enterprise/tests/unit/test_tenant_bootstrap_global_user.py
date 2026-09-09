# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Enterprise tenant bootstrap of the reserved global workspaces.

DefaultTenantService.ensure_default_tenant() must create BOTH the _global and
_global_user workspaces, mirroring the SQLite bootstrap. Without _global_user,
USER-scope writes and the include_global_user recall fan-out FK-reject / find
nothing on PostgreSQL.

Uses an in-memory fake workspace service so it runs without the eval PG.
"""

import pytest
from memorylayer_server.config import (
    DEFAULT_TENANT_ID,
    GLOBAL_USER_WORKSPACE_ID,
    GLOBAL_WORKSPACE_ID,
)

from memorylayer_saas.services.tenant.default import DefaultTenantService


class _FakeWorkspaceService:
    """Minimal async workspace service: in-memory id->Workspace store."""

    def __init__(self):
        self.workspaces = {}
        self.create_calls = []

    async def get_workspace(self, workspace_id):
        return self.workspaces.get(workspace_id)

    async def create_workspace(self, workspace):
        self.create_calls.append(workspace.id)
        self.workspaces[workspace.id] = workspace
        return workspace


@pytest.mark.asyncio
async def test_ensure_default_tenant_bootstraps_both_global_workspaces():
    """Both _global and _global_user are created on first bootstrap."""
    fake_ws = _FakeWorkspaceService()
    service = DefaultTenantService(workspace_service=fake_ws)

    await service.ensure_default_tenant()

    assert GLOBAL_WORKSPACE_ID in fake_ws.workspaces
    assert GLOBAL_USER_WORKSPACE_ID in fake_ws.workspaces

    user_ws = fake_ws.workspaces[GLOBAL_USER_WORKSPACE_ID]
    assert user_ws.tenant_id == DEFAULT_TENANT_ID
    assert user_ws.id == GLOBAL_USER_WORKSPACE_ID


@pytest.mark.asyncio
async def test_global_user_bootstrap_is_idempotent():
    """A pre-existing _global_user is not re-created."""
    fake_ws = _FakeWorkspaceService()

    # Pre-seed both reserved workspaces.
    from memorylayer_server.models.workspace import Workspace

    for ws_id in (GLOBAL_WORKSPACE_ID, GLOBAL_USER_WORKSPACE_ID):
        fake_ws.workspaces[ws_id] = Workspace(
            id=ws_id,
            tenant_id=DEFAULT_TENANT_ID,
            name=ws_id,
            settings={},
        )

    service = DefaultTenantService(workspace_service=fake_ws)
    await service.ensure_default_tenant()

    # Nothing was created — both already existed.
    assert fake_ws.create_calls == []


@pytest.mark.asyncio
async def test_bootstrap_runs_once_per_service():
    """The bootstrap guard prevents repeated work on subsequent calls."""
    fake_ws = _FakeWorkspaceService()
    service = DefaultTenantService(workspace_service=fake_ws)

    await service.ensure_default_tenant()
    first = list(fake_ws.create_calls)
    await service.get_default_tenant()  # also triggers _ensure_global_workspace

    # No additional creates after the first bootstrap pass.
    assert fake_ws.create_calls == first
    assert set(first) == {GLOBAL_WORKSPACE_ID, GLOBAL_USER_WORKSPACE_ID}
