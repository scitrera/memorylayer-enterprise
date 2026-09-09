"""Add memories.event_time for the timeline/temporal index.

Revision ID: 020
Revises: 019
Create Date: 2026-06-01 12:00:00

Adds:
- ``memories.event_time`` (nullable timestamptz) — when the memory's content is
  *about* (event time), distinct from ``created_at`` (when it was recorded).
- A partial index on ``(workspace_id, event_time)`` for timeline queries.

Timeline ordering uses COALESCE(event_time, created_at), so undated memories
fall back to creation time; the index accelerates explicitly-dated lookups.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "020"
down_revision: Union[str, None] = "019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column("event_time", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.create_index(
        "idx_memories_event_time",
        "memories",
        ["workspace_id", "event_time"],
        postgresql_where=sa.text("event_time IS NOT NULL AND deleted_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_memories_event_time", table_name="memories")
    op.drop_column("memories", "event_time")
