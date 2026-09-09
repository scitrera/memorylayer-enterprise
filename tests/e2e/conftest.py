# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""End-to-end test configuration.

These tests run against a live MemoryLayer + Aether stack.
Start the stack with: ./dev.sh up
Run tests with:       ./dev.sh test
"""

import uuid

import httpx
import pytest

from memorylayer import SyncMemoryLayerClient


def pytest_addoption(parser):
    parser.addoption(
        "--e2e-url",
        default="http://localhost:61001",
        help="MemoryLayer direct API URL",
    )
    parser.addoption(
        "--e2e-auth-url",
        default="http://localhost:40080",
        help="MemoryLayer API URL via auth-proxy",
    )


@pytest.fixture(scope="session")
def e2e_url(request):
    return request.config.getoption("--e2e-url")


@pytest.fixture(scope="session")
def e2e_auth_url(request):
    return request.config.getoption("--e2e-auth-url")


@pytest.fixture(scope="session", autouse=True)
def check_stack_health(e2e_url):
    """Skip all e2e tests if the stack isn't running."""
    try:
        resp = httpx.get(f"{e2e_url}/health", timeout=5)
        if resp.status_code != 200:
            pytest.skip(f"Stack not healthy at {e2e_url}: {resp.status_code}")
    except httpx.ConnectError:
        pytest.skip(
            f"Stack not reachable at {e2e_url} -- run './dev.sh up' first"
        )


@pytest.fixture
def workspace_name():
    """Generate a unique workspace name for test isolation."""
    return f"e2e-test-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def client(e2e_url, workspace_name):
    """Create a SDK client with a fresh workspace, clean up after."""
    with SyncMemoryLayerClient(base_url=e2e_url, workspace_id=workspace_name) as c:
        yield c
        # Cleanup: delete the workspace after the test
        try:
            c.delete_workspace(workspace_name)
        except Exception:
            pass


@pytest.fixture
def client_factory(e2e_url):
    """Factory for creating clients with specific workspace names."""
    created = []

    def _make(workspace_name: str | None = None):
        ws = workspace_name or f"e2e-test-{uuid.uuid4().hex[:12]}"
        c = SyncMemoryLayerClient(base_url=e2e_url, workspace_id=ws)
        c.__enter__()
        created.append((c, ws))
        return c

    yield _make

    for c, ws in created:
        try:
            c.delete_workspace(ws)
        except Exception:
            pass
        try:
            c.__exit__(None, None, None)
        except Exception:
            pass
