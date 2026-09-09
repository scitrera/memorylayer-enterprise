"""Add per-thread idle policy + archive flag to chat_threads.

Revision ID: 027
Revises: 026
Create Date: 2026-06-25 00:00:00

Adds two nullable columns to ``chat_threads`` supporting idle-based thread
lifecycle and manual/automatic archiving:

- ``idle_action TEXT`` — per-thread idle policy: NULL = never idles out (current
  behavior), ``'hide'`` = archive once idle past the server idle threshold,
  ``'delete'`` = remove. Idle is measured on ``updated_at`` (bumped on every
  message append).
- ``hidden_at TIMESTAMPTZ`` — set when the thread is archived (manually or by the
  idle ``'hide'`` policy); cleared on revival (a new append). NULL = visible.

Both are NULL for existing rows (no backfill — existing threads keep current
"permanent, visible" behavior). Two partial indexes back the background cleanup
sweeps: ``idx_chat_threads_idle`` for the idle delete/hide scan and
``idx_chat_threads_hidden`` for the post-archive grace purge. downgrade() drops
the indexes and columns.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "027"
down_revision: Union[str, None] = "026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add chat_threads.idle_action + hidden_at and their scan indexes."""
    op.add_column("chat_threads", sa.Column("idle_action", sa.String(16), nullable=True))
    op.add_column(
        "chat_threads",
        sa.Column("hidden_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.create_index(
        "idx_chat_threads_idle",
        "chat_threads",
        ["updated_at"],
        postgresql_where=sa.text("idle_action IS NOT NULL"),
    )
    op.create_index(
        "idx_chat_threads_hidden",
        "chat_threads",
        ["hidden_at"],
        postgresql_where=sa.text("hidden_at IS NOT NULL"),
    )


def downgrade() -> None:
    """Drop the scan indexes and the idle-policy columns."""
    op.drop_index("idx_chat_threads_hidden", table_name="chat_threads")
    op.drop_index("idx_chat_threads_idle", table_name="chat_threads")
    op.drop_column("chat_threads", "hidden_at")
    op.drop_column("chat_threads", "idle_action")
