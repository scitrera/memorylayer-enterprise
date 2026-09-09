"""Initial schema: providers, vfs_entries, sync_checkpoints.

Revision ID: 001
Revises: None
Create Date: 2026-05-08

Follows the migration pattern from memorylayer-enterprise.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create providers, vfs_entries, and sync_checkpoints tables."""

    # --- providers ---
    op.create_table(
        "providers",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column("workspace_id", sa.Text(), nullable=False),
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
    op.create_index("idx_providers_workspace", "providers", ["workspace_id"])
    op.create_index("idx_providers_type", "providers", ["workspace_id", "provider_type"])

    # --- vfs_entries ---
    op.create_table(
        "vfs_entries",
        sa.Column("vfs_ref", sa.Text(), nullable=False, primary_key=True),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column(
            "connector_id",
            sa.Text(),
            sa.ForeignKey("providers.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("blob_key", sa.Text(), nullable=True),
        sa.Column("ml_doc_id", sa.Text(), nullable=True),
        sa.Column("ml_job_id", sa.Text(), nullable=True),
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
    op.create_index("idx_vfs_workspace", "vfs_entries", ["workspace_id"])
    op.create_index("idx_vfs_connector", "vfs_entries", ["connector_id"])
    op.create_index("idx_vfs_content_hash", "vfs_entries", ["workspace_id", "content_hash"])
    op.create_index("idx_vfs_ml_doc", "vfs_entries", ["ml_doc_id"])

    # --- sync_checkpoints ---
    op.create_table(
        "sync_checkpoints",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column(
            "provider_id",
            sa.Text(),
            sa.ForeignKey("providers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("checkpoint_data", sa.dialects.postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("entries_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "synced_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("idx_checkpoint_provider", "sync_checkpoints", ["provider_id"])
    op.create_index(
        "idx_checkpoint_latest",
        "sync_checkpoints",
        ["provider_id", "synced_at"],
    )


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table("sync_checkpoints")
    op.drop_table("vfs_entries")
    op.drop_table("providers")
