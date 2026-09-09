# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Quarantine legacy comingled ``_user_chat`` chat threads (SECURITY).

Revision ID: 029
Revises: 028
Create Date: 2026-07-25 00:00:01

SECURITY BACKGROUND
-------------------
User-owned chat threads all live in the shared ``_user_chat`` sentinel workspace.
Before per-user materialization, the harness sent a shared client thread_id
(e.g. "_default") for every user and threads were resolved by
``(workspace_id, thread_id)`` only — so every user's ("_user_chat", "_default")
was ONE shared row and different users' chats COMINGLED (cross-user data leak,
see api/v1/chat.py chokepoint fix).

The fix materializes a per-user storage id ``<client_id>::u::<sha256(user)[:16]>``
for the sentinel. NEW per-user threads therefore start clean. This migration
stops SERVING the OLD comingled rows: any ``_user_chat`` thread whose id does
NOT match the per-user pattern is a legacy shared thread and is QUARANTINED.

WHAT THIS DOES
--------------
- Scope: rows with ``workspace_id = '_user_chat'`` whose ``id`` does NOT end in
  ``::u::`` followed by a 16-char lowercase hex digest (the materialized shape).
- Action: HIDE them (``hidden_at = now()`` so they drop out of listings) and
  stamp a quarantine marker into ``metadata`` for incident evidence. The rows
  and their messages are NOT deleted — this is user data + incident evidence
  that needs MANUAL REVIEW (the comingled content may belong to several users).
- Idempotent: rows already carrying ``metadata->>'quarantine_reason'`` are
  skipped, so re-running is a no-op.

downgrade() clears the quarantine marker and un-hides the affected rows (best
effort — only rows this migration marked, identified by the marker value).

NOTE: this is a DATA migration only (no schema change). It relies on the
``hidden_at`` column (rev 027) and JSONB ``metadata`` already present.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "029"
down_revision: Union[str, None] = "028"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_QUARANTINE_REASON = "legacy_comingled_user_chat_pre_per_user_scoping"

# A materialized per-user id ends with ``::u::`` + a 16-char lowercase hex digest.
# Anything in _user_chat NOT matching this is a legacy shared thread.
_MATERIALIZED_ID_REGEX = r"::u::[0-9a-f]{16}$"


def upgrade() -> None:
    """Hide + mark legacy comingled ``_user_chat`` threads (idempotent)."""
    op.execute(
        f"""
        UPDATE chat_threads
        SET
            hidden_at = COALESCE(hidden_at, now()),
            metadata = COALESCE(metadata, '{{}}'::jsonb)
                || jsonb_build_object(
                    'quarantine_reason', '{_QUARANTINE_REASON}',
                    'quarantined_at', to_jsonb(now())
                )
        WHERE workspace_id = '_user_chat'
          AND id !~ '{_MATERIALIZED_ID_REGEX}'
          AND (metadata->>'quarantine_reason') IS DISTINCT FROM '{_QUARANTINE_REASON}'
        """
    )


def downgrade() -> None:
    """Un-hide and clear the quarantine marker for rows this migration touched."""
    op.execute(
        f"""
        UPDATE chat_threads
        SET
            hidden_at = NULL,
            metadata = (metadata - 'quarantine_reason' - 'quarantined_at')
        WHERE workspace_id = '_user_chat'
          AND (metadata->>'quarantine_reason') = '{_QUARANTINE_REASON}'
        """
    )
