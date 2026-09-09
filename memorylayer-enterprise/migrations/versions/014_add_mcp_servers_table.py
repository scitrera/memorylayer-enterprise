# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add mcp_servers table for MCP Server Registry feature.

Revision ID: 014
Revises: 013
Create Date: 2026-04-25 12:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create mcp_servers table."""
    op.create_table(
        "mcp_servers",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column("tenant_id", sa.Text(), nullable=False, server_default="_default"),
        sa.Column(
            "workspace_id",
            sa.Text(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("transport", sa.Text(), nullable=False),
        sa.Column("command", sa.Text(), nullable=True),
        sa.Column(
            "args",
            sa.dialects.postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "env",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column(
            "headers",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "metadata",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("source_mode", sa.Text(), nullable=False, server_default="server"),
        sa.Column("manifest_hash", sa.Text(), nullable=False, server_default=""),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
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

    op.create_index("idx_mcp_servers_workspace", "mcp_servers", ["workspace_id"])
    op.create_index("idx_mcp_servers_tenant_workspace", "mcp_servers", ["tenant_id", "workspace_id"])
    op.create_index("idx_mcp_servers_name", "mcp_servers", ["name"])
    op.execute(
        "CREATE UNIQUE INDEX idx_mcp_servers_workspace_name_global "
        "ON mcp_servers (workspace_id, name) WHERE user_id IS NULL"
    )
    op.execute(
        "CREATE UNIQUE INDEX idx_mcp_servers_workspace_user_name "
        "ON mcp_servers (workspace_id, user_id, name) WHERE user_id IS NOT NULL"
    )


def downgrade() -> None:
    """Drop mcp_servers table."""
    op.drop_index("idx_mcp_servers_workspace_user_name", table_name="mcp_servers")
    op.drop_index("idx_mcp_servers_workspace_name_global", table_name="mcp_servers")
    op.drop_index("idx_mcp_servers_name", table_name="mcp_servers")
    op.drop_index("idx_mcp_servers_tenant_workspace", table_name="mcp_servers")
    op.drop_index("idx_mcp_servers_workspace", table_name="mcp_servers")
    op.drop_table("mcp_servers")
