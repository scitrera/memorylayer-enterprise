# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add composite UNIQUE(workspace_id, id) constraint to chat_threads.

Revision ID: 018
Revises: 017
Create Date: 2026-05-23 00:00:01

Defense-in-depth schema enforcement for the user-scoped chat-thread design.

Earlier work (Agents A–C) introduced an ``ownership`` column and a
``list_user_threads`` query path. The chat-rail audit (.slop/chat_rail_frontend_handoff.md)
then identified a residual collision risk: ``chat_threads.id`` is a simple
TEXT primary key, so a thread with ``id='_default'`` in workspace ``ws1``
and another with the same id in workspace ``ws2`` are distinct rows the
database is happy to store, but the application layer cannot disambiguate
when callers omit ``workspace_id``.

The agreed application-layer fix is to route every user-scoped thread
through a sentinel workspace (``USER_CHAT_HOME_WORKSPACE = '_user_chat'``)
so the (workspace_id, id) tuple is once again unambiguous for the
user-scoped read path. This migration adds the matching database-level
constraint so the schema enforces the invariant even if a future code
path accidentally bypasses the sentinel.

PRESUMES no existing collisions: legacy ``_default`` threads have
distinct ``workspace_id`` values per the pre-redesign per-workspace
ownership model, so the constraint is satisfiable on existing data.
No deduplication is performed here — the data is correct today; the
constraint just enforces it going forward.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "018"
down_revision: Union[str, None] = "017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add UNIQUE(workspace_id, id) constraint to chat_threads."""
    op.create_unique_constraint(
        "uq_chat_threads_workspace_id",
        "chat_threads",
        ["workspace_id", "id"],
    )


def downgrade() -> None:
    """Drop the UNIQUE(workspace_id, id) constraint from chat_threads."""
    op.drop_constraint(
        "uq_chat_threads_workspace_id",
        "chat_threads",
        type_="unique",
    )
