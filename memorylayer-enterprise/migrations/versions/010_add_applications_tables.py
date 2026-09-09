# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add applications and workspace_applications tables.

Revision ID: 010
Revises: 009
Create Date: 2026-03-24 12:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create applications and workspace_applications tables."""
    op.create_table(
        "applications",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("app_type", sa.Text(), nullable=False, server_default="generic"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("config", sa.dialects.postgresql.JSONB(), nullable=False, server_default="{}"),
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

    op.create_index("idx_applications_tenant", "applications", ["tenant_id"])

    op.create_table(
        "workspace_applications",
        sa.Column(
            "workspace_id",
            sa.Text(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "application_id",
            sa.Text(),
            sa.ForeignKey("applications.id", ondelete="CASCADE"),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("config_overrides", sa.dialects.postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_index("idx_workspace_apps_workspace", "workspace_applications", ["workspace_id"])
    op.create_index("idx_workspace_apps_application", "workspace_applications", ["application_id"])


def downgrade() -> None:
    """Drop workspace_applications and applications tables."""
    op.drop_index("idx_workspace_apps_application", table_name="workspace_applications")
    op.drop_index("idx_workspace_apps_workspace", table_name="workspace_applications")
    op.drop_table("workspace_applications")
    op.drop_index("idx_applications_tenant", table_name="applications")
    op.drop_table("applications")
