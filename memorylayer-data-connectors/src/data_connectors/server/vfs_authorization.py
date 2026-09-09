# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Fail-closed authorization for VFS operations delivered through Aether.

The gateway evaluates a caller-supplied logical resource tuple and attaches a
short-lived receipt to the ProxyHTTP envelope. The ASGI bridge exposes that
receipt only through a private scope extension; JSON bodies and HTTP headers
are never accepted as authority.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from data_connectors.server.aether_service import service_topic

AUTHORIZATION_MODE_ENV = "DC_VFS_AUTHORIZATION_MODE"
MODE_AETHER = "aether"
MODE_DISABLED = "disabled"
SCOPE_RECEIPT_KEY = "aether.access_receipt"
RESOURCE_TYPE_VFS = "vfs"
ACCESS_READ = 10
ACCESS_READ_WRITE = 20


class VfsAuthorizationError(Exception):
    """A VFS request lacked an exact, current gateway authorization receipt."""


@dataclass(frozen=True)
class TrustedVfsAuthority:
    enforced: bool
    actor_type: str = ""
    actor_id: str = ""
    subject_type: str = ""
    subject_id: str = ""
    grant_id: str = ""

    @property
    def user_subject(self) -> str | None:
        if self.subject_type == "user" and self.subject_id:
            return self.subject_id
        if self.subject_type == "" and self.actor_type == "user" and self.actor_id:
            return self.actor_id
        return None

    @property
    def initiated_by(self) -> str | None:
        subject = self.user_subject
        if subject is None:
            return None
        return f"us::{subject.removeprefix('user:')}"


def _segment(value: str) -> str:
    return quote(value, safe="")


def vfs_collection_resource(workspace_id: str) -> str:
    return f"workspaces/{_segment(workspace_id)}/entries"


def vfs_entry_resource(workspace_id: str, vfs_ref: str) -> str:
    return f"{vfs_collection_resource(workspace_id)}/{_segment(vfs_ref)}"


def authorization_mode() -> str:
    mode = os.environ.get(AUTHORIZATION_MODE_ENV, MODE_AETHER).strip().lower()
    if mode not in {MODE_AETHER, MODE_DISABLED}:
        raise RuntimeError(
            f"{AUTHORIZATION_MODE_ENV} must be {MODE_AETHER!r} or {MODE_DISABLED!r}"
        )
    return mode


def _principal(receipt: Any, field: str) -> tuple[str, str]:
    ref = getattr(receipt, field, None)
    if ref is None:
        return "", ""
    return (
        str(getattr(ref, "principal_type", "") or ""),
        str(getattr(ref, "principal_id", "") or ""),
    )


def require_vfs_access(
    request: Any,
    *,
    workspace_id: str,
    operation: str,
    required_access_level: int,
    vfs_ref: str | None = None,
    now_ms: int | None = None,
) -> TrustedVfsAuthority:
    """Validate the gateway receipt for one collection or exact-entry action."""
    if authorization_mode() == MODE_DISABLED:
        return TrustedVfsAuthority(enforced=False)

    receipt = request.scope.get(SCOPE_RECEIPT_KEY)
    if receipt is None:
        raise VfsAuthorizationError("gateway access receipt is required")

    checked = getattr(receipt, "request", None)
    expected_resource = (
        vfs_entry_resource(workspace_id, vfs_ref)
        if vfs_ref is not None
        else vfs_collection_resource(workspace_id)
    )
    expected = {
        "resource_type": RESOURCE_TYPE_VFS,
        "resource_id": expected_resource,
        "operation": operation,
        "workspace": workspace_id,
        "required_access_level": required_access_level,
    }
    if checked is None:
        raise VfsAuthorizationError("receipt has no checked resource")
    for field, value in expected.items():
        if getattr(checked, field, None) != value:
            raise VfsAuthorizationError(f"receipt {field} does not match request")
    if not str(getattr(checked, "correlation_id", "") or ""):
        raise VfsAuthorizationError("receipt correlation is missing")

    current_ms = int(time.time() * 1000) if now_ms is None else now_ms
    if not bool(getattr(receipt, "allowed", False)):
        raise VfsAuthorizationError("receipt is not an allow decision")
    if str(getattr(receipt, "decision", "") or "") != "ALLOW":
        raise VfsAuthorizationError("receipt decision is not ALLOW")
    if int(getattr(receipt, "effective_access_level", 0) or 0) < required_access_level:
        raise VfsAuthorizationError("receipt access level is insufficient")
    if int(getattr(receipt, "expires_at_ms", 0) or 0) <= current_ms:
        raise VfsAuthorizationError("receipt is expired")
    if str(getattr(receipt, "delivery_target", "") or "") != service_topic():
        raise VfsAuthorizationError("receipt delivery target does not match this service")

    actor_type, actor_id = _principal(receipt, "actor")
    if not actor_type or not actor_id:
        raise VfsAuthorizationError("receipt actor is incomplete")
    authority_mode = str(getattr(receipt, "authority_mode", "") or "")
    subject_type, subject_id = _principal(receipt, "subject")
    grant_id = str(getattr(receipt, "grant_id", "") or "")
    if authority_mode == "direct":
        if subject_type or subject_id or grant_id:
            raise VfsAuthorizationError("direct receipt contains delegated authority")
    elif authority_mode == "on_behalf_of":
        if not subject_type or not subject_id or not grant_id:
            raise VfsAuthorizationError("delegated receipt is missing authority lineage")
    else:
        raise VfsAuthorizationError("receipt authority mode is invalid")

    return TrustedVfsAuthority(
        enforced=True,
        actor_type=actor_type,
        actor_id=actor_id,
        subject_type=subject_type,
        subject_id=subject_id,
        grant_id=grant_id,
    )
