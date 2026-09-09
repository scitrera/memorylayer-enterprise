"""Add audit_events table for tracking system and user actions.

Introduces the audit_events table to record immutable audit log entries
for all significant actions across tenants, workspaces, and users:
- audit_events: stores event type, action, actor, resource, and metadata

Revision ID: 008
Revises: 007
Create Date: 2026-03-15 12:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create audit_events table with indexes for common query patterns."""
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=True),
        sa.Column("user_id", sa.Text(), nullable=True),
        sa.Column("resource_type", sa.Text(), nullable=True),
        sa.Column("resource_id", sa.Text(), nullable=True),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column(
            "timestamp",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_index("idx_audit_events_tenant", "audit_events", ["tenant_id"])
    op.create_index("idx_audit_events_workspace", "audit_events", ["workspace_id"])
    op.create_index("idx_audit_events_user", "audit_events", ["user_id"])
    op.create_index("idx_audit_events_event_type", "audit_events", ["event_type"])
    op.create_index("idx_audit_events_timestamp", "audit_events", ["timestamp"])


def downgrade() -> None:
    """Drop audit_events indexes then the table."""
    op.drop_index("idx_audit_events_timestamp", table_name="audit_events")
    op.drop_index("idx_audit_events_event_type", table_name="audit_events")
    op.drop_index("idx_audit_events_user", table_name="audit_events")
    op.drop_index("idx_audit_events_workspace", table_name="audit_events")
    op.drop_index("idx_audit_events_tenant", table_name="audit_events")
    op.drop_table("audit_events")
