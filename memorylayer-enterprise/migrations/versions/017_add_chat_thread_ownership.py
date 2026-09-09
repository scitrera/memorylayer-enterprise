# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add ownership column to chat_threads table.

Revision ID: 017
Revises: 016
Create Date: 2026-05-23 00:00:00

Adds a non-null ``ownership`` text column to ``chat_threads`` with a
server-side default of ``'user'``.  The column separates user-owned
threads (the user-session-scoped right rail in the web app) from
workspace-shared threads.

Existing rows backfill to ``'user'`` via ``server_default`` — this
matches the locked-in decision to leave legacy per-workspace ``_default``
threads alone (they simply become "user-owned threads that happen to
share a workspace/_default key and remain unreachable from the new UI").

Also adds a composite index on (tenant_id, user_id, ownership) to back
the cross-workspace ``list_user_threads`` query path.  Partial index on
``user_id IS NOT NULL`` mirrors the user-scope patterns used by
``SkillModel`` and ``McpServerModel``.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "017"
down_revision: Union[str, None] = "016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add ownership column and supporting index to chat_threads."""
    op.add_column(
        "chat_threads",
        sa.Column(
            "ownership",
            sa.String(length=16),
            nullable=False,
            server_default="user",
        ),
    )
    op.create_index(
        "idx_chat_threads_tenant_user_ownership",
        "chat_threads",
        ["tenant_id", "user_id", "ownership"],
        postgresql_where=sa.text("user_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Remove ownership column and supporting index from chat_threads."""
    op.drop_index("idx_chat_threads_tenant_user_ownership", table_name="chat_threads")
    op.drop_column("chat_threads", "ownership")
