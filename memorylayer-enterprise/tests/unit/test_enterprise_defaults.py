"""Tests for enterprise preconfiguration defaults."""

import os
from unittest.mock import patch

import pytest
from scitrera_app_framework import Variables

from memorylayer_server.config import (
    MEMORYLAYER_AUTHENTICATION_SERVICE,
    MEMORYLAYER_AUTHORIZATION_SERVICE,
    MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER,
    MEMORYLAYER_GRAPH_QUERY_PROVIDER,
)
from memorylayer_saas.config import MEMORYLAYER_REQUIRE_SECURE_AUTH
from memorylayer_saas.dependencies import _enterprise_preconfigure_hook


def _run_hook(v: Variables) -> None:
    """Run the preconfigure hook with plugin registration patched out."""
    with (
        patch("memorylayer_saas.dependencies.register_package_plugins"),
        patch("memorylayer_saas.dependencies._supersede_oss_routers"),
    ):
        _enterprise_preconfigure_hook(v)


def test_enterprise_defaults_select_age_graph_backends():
    """Enterprise preconfigure should put AGE graph backends in the hot path."""
    v = Variables()
    _run_hook(v)

    assert v.get(MEMORYLAYER_GRAPH_QUERY_PROVIDER) == "age"
    assert v.get(MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER) == "age"


# ---------------------------------------------------------------------------
# B1 — auth/authz default to 'aether' + fail-closed startup guard
# ---------------------------------------------------------------------------


def test_auth_defaults_to_aether():
    """Hook must default both auth and authz to 'aether', not the OSS 'default'."""
    v = Variables()
    _run_hook(v)

    assert v.get(MEMORYLAYER_AUTHENTICATION_SERVICE) == "aether", (
        "MEMORYLAYER_AUTHENTICATION_SERVICE should default to 'aether' in enterprise"
    )
    assert v.get(MEMORYLAYER_AUTHORIZATION_SERVICE) == "aether", (
        "MEMORYLAYER_AUTHORIZATION_SERVICE should default to 'aether' in enterprise"
    )


def test_secure_auth_guard_raises_when_auth_forced_to_default(monkeypatch):
    """With MEMORYLAYER_REQUIRE_SECURE_AUTH=1, booting with auth='default' must raise.

    This pins the 'v.get() reflects env override' behaviour: even though the hook
    calls set_default_value(..., 'aether'), an explicit env var sets the resolved
    value back to 'default', and the guard must catch it.
    """
    monkeypatch.setenv(MEMORYLAYER_REQUIRE_SECURE_AUTH, "1")
    monkeypatch.setenv(MEMORYLAYER_AUTHENTICATION_SERVICE, "default")

    v = Variables()

    with pytest.raises(RuntimeError, match=MEMORYLAYER_REQUIRE_SECURE_AUTH):
        _run_hook(v)


def test_secure_auth_guard_passes_when_auth_is_aether(monkeypatch):
    """With MEMORYLAYER_REQUIRE_SECURE_AUTH=1 and auth resolved to 'aether', startup is OK."""
    monkeypatch.setenv(MEMORYLAYER_REQUIRE_SECURE_AUTH, "1")
    # No env override → set_default_value resolves to 'aether' → guard passes.

    v = Variables()
    _run_hook(v)  # must not raise

    assert v.get(MEMORYLAYER_AUTHENTICATION_SERVICE) == "aether"
    assert v.get(MEMORYLAYER_AUTHORIZATION_SERVICE) == "aether"


def test_secure_auth_guard_off_by_default():
    """Without MEMORYLAYER_REQUIRE_SECURE_AUTH set, even 'default' auth must not raise."""
    v = Variables()
    # Force auth back to 'default' via explicit local set to simulate a misconfigured
    # env where someone re-opened the allow-all service, but the guard is off.
    v[MEMORYLAYER_AUTHENTICATION_SERVICE] = "default"

    # Guard flag is NOT set — hook must complete without raising.
    _run_hook(v)
