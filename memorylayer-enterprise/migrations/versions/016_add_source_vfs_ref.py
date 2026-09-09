"""Add source_vfs_ref column to documents table.

Revision ID: 016
Revises: 015
Create Date: 2026-05-08 12:00:00

Adds a nullable ``source_vfs_ref`` text column to ``documents`` for linking
documents created via the data-connectors VFS-based ingestion path back to
their VFS entry.  Indexed for dedup-by-vfs-ref lookups.  Also adds
'pending_fetch' to the document status check constraint to support the
new PENDING_FETCH lifecycle state.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add source_vfs_ref column and update status constraint."""
    # Add source_vfs_ref column
    op.add_column(
        "documents",
        sa.Column("source_vfs_ref", sa.Text(), nullable=True),
    )
    # Index for dedup-by-vfs-ref lookups
    op.create_index(
        "idx_documents_source_vfs_ref",
        "documents",
        ["source_vfs_ref"],
        unique=False,
    )

    # Update the status check constraint to include 'pending_fetch'
    op.drop_constraint("ck_document_status", "documents", type_="check")
    op.create_check_constraint(
        "ck_document_status",
        "documents",
        "status IN ('pending', 'pending_fetch', 'processing', 'completed', 'failed', 'partial')",
    )


def downgrade() -> None:
    """Remove source_vfs_ref column and revert status constraint."""
    # Revert the status check constraint
    op.drop_constraint("ck_document_status", "documents", type_="check")
    op.create_check_constraint(
        "ck_document_status",
        "documents",
        "status IN ('pending', 'processing', 'completed', 'failed', 'partial')",
    )

    # Remove source_vfs_ref column and index
    op.drop_index("idx_documents_source_vfs_ref", table_name="documents")
    op.drop_column("documents", "source_vfs_ref")
