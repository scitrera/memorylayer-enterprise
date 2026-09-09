# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add parent_thread to chat_threads for sub-threads.

Revision ID: 028
Revises: 027
Create Date: 2026-06-25 00:00:01

Adds a nullable ``parent_thread TEXT`` column to ``chat_threads``. NULL = a
top-level thread (the default; thread listings return only top-level threads
unless a parent is explicitly requested). A non-NULL value points at the parent
thread's id, making the row a sub-thread; a child always shares its parent's
workspace + ownership (enforced at the service layer). No DB-level self-FK is
added — child cascade on delete is handled in application code. A partial index
on ``parent_thread`` backs the "list a parent's children" query.

Existing rows backfill to NULL (all current threads are top-level), so there is
no behavior change for existing data. downgrade() drops the index and column.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "028"
down_revision: Union[str, None] = "027"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add chat_threads.parent_thread and its child-lookup index."""
    op.add_column("chat_threads", sa.Column("parent_thread", sa.Text(), nullable=True))
    op.create_index(
        "idx_chat_threads_parent",
        "chat_threads",
        ["parent_thread"],
        postgresql_where=sa.text("parent_thread IS NOT NULL"),
    )


def downgrade() -> None:
    """Drop the child-lookup index and the parent_thread column."""
    op.drop_index("idx_chat_threads_parent", table_name="chat_threads")
    op.drop_column("chat_threads", "parent_thread")
