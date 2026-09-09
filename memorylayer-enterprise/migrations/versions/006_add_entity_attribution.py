"""Add entity attribution columns (observer_id, subject_id) to memories table.

Syncs with OSS Memory model v3 entity attribution fields.
Each memory can now track "who remembers what about whom":
- observer_id: The entity doing the observing/remembering (agent ID, user ID, etc.)
- subject_id: The entity the memory is about

Revision ID: 006
Revises: 005
Create Date: 2026-03-10 12:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add entity attribution columns and indexes."""
    # Entity attribution columns
    op.add_column("memories", sa.Column("observer_id", sa.Text(), nullable=True))
    op.add_column("memories", sa.Column("subject_id", sa.Text(), nullable=True))

    # Partial indexes for entity-scoped queries (only index non-null values)
    op.create_index(
        "idx_memories_observer",
        "memories",
        ["workspace_id", "observer_id"],
        postgresql_where=sa.text("observer_id IS NOT NULL AND deleted_at IS NULL"),
    )
    op.create_index(
        "idx_memories_subject",
        "memories",
        ["workspace_id", "subject_id"],
        postgresql_where=sa.text("subject_id IS NOT NULL AND deleted_at IS NULL"),
    )
    # Composite index for entity pair lookups (who remembers what about whom)
    op.create_index(
        "idx_memories_observer_subject",
        "memories",
        ["workspace_id", "observer_id", "subject_id"],
        postgresql_where=sa.text(
            "observer_id IS NOT NULL AND subject_id IS NOT NULL AND deleted_at IS NULL"
        ),
    )


def downgrade() -> None:
    """Remove entity attribution columns and indexes."""
    op.drop_index("idx_memories_observer_subject", table_name="memories")
    op.drop_index("idx_memories_subject", table_name="memories")
    op.drop_index("idx_memories_observer", table_name="memories")
    op.drop_column("memories", "subject_id")
    op.drop_column("memories", "observer_id")
