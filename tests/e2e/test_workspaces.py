# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""E2E: Workspace management — create, list, delete."""

import uuid

from memorylayer import SyncMemoryLayerClient


def test_create_and_list_workspaces(e2e_url):
    """Create a workspace and verify it appears in the listing."""
    ws_name = f"e2e-ws-{uuid.uuid4().hex[:8]}"
    with SyncMemoryLayerClient(base_url=e2e_url, workspace_id=ws_name) as client:
        workspace = client.create_workspace(ws_name)
        assert workspace.name == ws_name

        workspaces = client.list_workspaces()
        ws_names = [w.name for w in workspaces]
        assert ws_name in ws_names

        # Cleanup
        client.delete_workspace(ws_name)


def test_delete_workspace(e2e_url):
    """Delete a workspace and verify it's removed."""
    ws_name = f"e2e-ws-del-{uuid.uuid4().hex[:8]}"
    with SyncMemoryLayerClient(base_url=e2e_url, workspace_id=ws_name) as client:
        client.create_workspace(ws_name)
        client.delete_workspace(ws_name)

        workspaces = client.list_workspaces()
        ws_names = [w.name for w in workspaces]
        assert ws_name not in ws_names
