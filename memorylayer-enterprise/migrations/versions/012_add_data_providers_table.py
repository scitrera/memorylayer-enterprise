# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add data_providers table for document ingestion sources.

Revision ID: 012
Revises: 011
Create Date: 2026-03-24 12:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create data_providers table."""
    op.create_table(
        "data_providers",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column(
            "workspace_id",
            sa.Text(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("provider_type", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("connection_args", sa.dialects.postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("encrypted_args", sa.dialects.postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("schedule", sa.Text(), nullable=True),
        sa.Column("last_sync_at", sa.TIMESTAMP(timezone=True), nullable=True),
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

    op.create_index("idx_data_providers_workspace", "data_providers", ["workspace_id"])
    op.create_index(
        "idx_data_providers_type", "data_providers", ["workspace_id", "provider_type"]
    )


def downgrade() -> None:
    """Drop data_providers table."""
    op.drop_index("idx_data_providers_type", table_name="data_providers")
    op.drop_index("idx_data_providers_workspace", table_name="data_providers")
    op.drop_table("data_providers")
