# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""E2E: Workspace isolation — memories don't leak between workspaces."""

import time


def test_workspace_isolation(client_factory):
    """Memories in one workspace are not visible in another."""
    client_a = client_factory("e2e-isolation-ws-a")
    client_b = client_factory("e2e-isolation-ws-b")

    # Store in workspace A
    client_a.remember(content="This is workspace A's secret data")
    time.sleep(0.5)

    # Recall in workspace B should not find it
    results = client_b.recall("workspace A's secret data")
    contents = [m.content for m in results.memories]
    assert not any("workspace A" in c for c in contents), (
        "Workspace B should not see workspace A's memories"
    )


def test_same_query_different_workspaces(client_factory):
    """Same query in different workspaces returns different results."""
    client_a = client_factory("e2e-same-query-ws-a")
    client_b = client_factory("e2e-same-query-ws-b")

    client_a.remember(content="The preferred language is Rust")
    client_b.remember(content="The preferred language is Go")
    time.sleep(0.5)

    results_a = client_a.recall("preferred language")
    results_b = client_b.recall("preferred language")

    contents_a = " ".join(m.content for m in results_a.memories)
    contents_b = " ".join(m.content for m in results_b.memories)

    assert "Rust" in contents_a
    assert "Go" in contents_b
    assert "Go" not in contents_a
    assert "Rust" not in contents_b
