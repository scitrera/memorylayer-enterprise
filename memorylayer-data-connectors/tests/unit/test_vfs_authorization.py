from __future__ import annotations

from types import SimpleNamespace

import pytest

from data_connectors.server.vfs_authorization import (
    ACCESS_READ,
    ACCESS_READ_WRITE,
    SCOPE_RECEIPT_KEY,
    VfsAuthorizationError,
    require_vfs_access,
    vfs_collection_resource,
    vfs_entry_resource,
)

NOW_MS = 1_786_640_000_000


def _request(receipt=None):
    return SimpleNamespace(scope={SCOPE_RECEIPT_KEY: receipt})


def _receipt(
    *,
    workspace: str = "ws-1",
    vfs_ref: str | None = "vfs-1",
    operation: str = "read",
    required: int = ACCESS_READ,
    expires_at_ms: int = NOW_MS + 30_000,
    delivery_target: str = "sv::data-connectors::dc-one",
    authority_mode: str = "direct",
):
    receipt = SimpleNamespace(
        decision_id="decision-1",
        request=SimpleNamespace(
            resource_type="vfs",
            resource_id=(
                vfs_entry_resource(workspace, vfs_ref)
                if vfs_ref is not None
                else vfs_collection_resource(workspace)
            ),
            operation=operation,
            workspace=workspace,
            required_access_level=required,
            correlation_id="proxy-1",
        ),
        allowed=True,
        decision="ALLOW",
        effective_access_level=required,
        actor=SimpleNamespace(
            principal_type="service", principal_id="sv::platform-server::one"
        ),
        subject=None,
        grant_id="",
        authority_mode=authority_mode,
        evaluated_at_ms=NOW_MS,
        expires_at_ms=expires_at_ms,
        delivery_target=delivery_target,
    )
    if authority_mode == "on_behalf_of":
        receipt.subject = SimpleNamespace(
            principal_type="user", principal_id="alice@example.com"
        )
        receipt.grant_id = "grant-1"
    return receipt


@pytest.fixture
def aether_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DC_VFS_AUTHORIZATION_MODE", "aether")
    monkeypatch.setenv("DC_SERVICE_SPECIFIER", "dc-one")


def test_resource_ids_percent_encode_segments() -> None:
    assert vfs_collection_resource("team/a b") == "workspaces/team%2Fa%20b/entries"
    assert vfs_entry_resource("team/a b", "vfs/x") == (
        "workspaces/team%2Fa%20b/entries/vfs%2Fx"
    )


def test_disabled_mode_is_explicit_bypass() -> None:
    authority = require_vfs_access(
        _request(), workspace_id="ws-1", operation="read",
        required_access_level=ACCESS_READ, vfs_ref="vfs-1", now_ms=NOW_MS,
    )
    assert authority.enforced is False


def test_exact_direct_receipt_is_accepted(aether_mode: None) -> None:
    authority = require_vfs_access(
        _request(_receipt()), workspace_id="ws-1", operation="read",
        required_access_level=ACCESS_READ, vfs_ref="vfs-1", now_ms=NOW_MS,
    )
    assert authority.enforced is True
    assert authority.actor_id == "sv::platform-server::one"
    assert authority.user_subject is None


def test_obo_receipt_derives_trusted_user_provenance(aether_mode: None) -> None:
    authority = require_vfs_access(
        _request(_receipt(authority_mode="on_behalf_of")),
        workspace_id="ws-1", operation="read",
        required_access_level=ACCESS_READ, vfs_ref="vfs-1", now_ms=NOW_MS,
    )
    assert authority.user_subject == "alice@example.com"
    assert authority.initiated_by == "us::alice@example.com"
    assert authority.grant_id == "grant-1"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda receipt: setattr(receipt.request, "operation", "write"), "operation"),
        (lambda receipt: setattr(receipt.request, "workspace", "ws-2"), "workspace"),
        (lambda receipt: setattr(receipt.request, "resource_id", "workspaces/ws-1/entries/other"), "resource_id"),
        (lambda receipt: setattr(receipt, "allowed", False), "allow"),
        (lambda receipt: setattr(receipt, "decision", "DENY"), "ALLOW"),
        (lambda receipt: setattr(receipt, "effective_access_level", 0), "insufficient"),
        (lambda receipt: setattr(receipt, "expires_at_ms", NOW_MS), "expired"),
        (lambda receipt: setattr(receipt, "delivery_target", "sv::data-connectors::other"), "delivery target"),
        (lambda receipt: setattr(receipt, "actor", None), "actor"),
    ],
)
def test_receipt_mismatch_fails_closed(aether_mode: None, mutate, message: str) -> None:
    receipt = _receipt()
    mutate(receipt)
    with pytest.raises(VfsAuthorizationError, match=message):
        require_vfs_access(
            _request(receipt), workspace_id="ws-1", operation="read",
            required_access_level=ACCESS_READ, vfs_ref="vfs-1", now_ms=NOW_MS,
        )


def test_missing_receipt_fails_closed(aether_mode: None) -> None:
    with pytest.raises(VfsAuthorizationError, match="required"):
        require_vfs_access(
            _request(), workspace_id="ws-1", operation="write",
            required_access_level=ACCESS_READ_WRITE, now_ms=NOW_MS,
        )
