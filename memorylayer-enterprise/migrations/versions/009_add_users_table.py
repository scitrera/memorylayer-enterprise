"""Add users table for platform user management.

Revision ID: 009
Revises: 008
Create Date: 2026-03-24 12:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create users table with indexes."""
    op.create_table(
        "users",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=True),
        sa.Column("first_name", sa.Text(), nullable=True),
        sa.Column("last_name", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("licensed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_unique_constraint("uq_users_tenant_email", "users", ["tenant_id", "email"])
    op.create_index("idx_users_tenant", "users", ["tenant_id"])
    op.create_index("idx_users_email", "users", ["email"])


def downgrade() -> None:
    """Drop users table."""
    op.drop_index("idx_users_email", table_name="users")
    op.drop_index("idx_users_tenant", table_name="users")
    op.drop_constraint("uq_users_tenant_email", "users", type_="unique")
    op.drop_table("users")
