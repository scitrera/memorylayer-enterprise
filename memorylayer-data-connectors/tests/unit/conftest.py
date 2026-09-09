from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def explicit_standalone_vfs_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit route tests use direct HTTP, so opt into standalone auth mode."""
    monkeypatch.setenv("DC_VFS_AUTHORIZATION_MODE", "disabled")
