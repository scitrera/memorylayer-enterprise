"""Add document_pages table and source attribution columns to memories.

Introduces per-page document storage with transcription and visual token support,
and links memories back to their source document/page for attribution:
- document_pages: stores page-level data (image path, transcript, multivector, etc.)
- memories.source_document_id: which document a memory was derived from
- memories.source_page_id: which specific page a memory was derived from

Revision ID: 007
Revises: 006
Create Date: 2026-03-11 12:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import ARRAY
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create document_pages table and add source attribution columns to memories."""
    # --- document_pages table ---
    op.create_table(
        "document_pages",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column(
            "document_id",
            sa.Text(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("page_no", sa.Integer(), nullable=False),
        sa.Column("image_storage_path", sa.Text(), nullable=True),
        sa.Column("transcript", sa.Text(), nullable=True),
        # multivector: ARRAY of Vector(128) for multi-vector retrieval.
        # Declared as ARRAY(Vector(128)) via pgvector; ORM model handles actual type.
        sa.Column("multivector", ARRAY(Vector(128)), nullable=True),
        sa.Column("transcript_model", sa.Text(), nullable=True),
        sa.Column(
            "transcript_attempts",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("visual_tokens", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column(
            "metadata",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("document_id", "page_no", name="uq_document_page"),
    )

    op.create_index(
        "idx_document_pages_workspace",
        "document_pages",
        ["workspace_id"],
    )
    op.create_index(
        "idx_document_pages_document",
        "document_pages",
        ["document_id"],
    )
    op.create_index(
        "idx_document_pages_workspace_document",
        "document_pages",
        ["workspace_id", "document_id"],
    )

    # --- Source attribution columns on memories ---
    # No FK constraints here: documents table may not exist in all deployments.
    # Referential integrity is enforced at the application level.
    op.add_column("memories", sa.Column("source_document_id", sa.Text(), nullable=True))
    op.add_column("memories", sa.Column("source_page_id", sa.Text(), nullable=True))

    # Partial indexes for source-scoped queries (only index non-null, non-deleted rows)
    op.create_index(
        "idx_memories_source_document",
        "memories",
        ["workspace_id", "source_document_id"],
        postgresql_where=sa.text("source_document_id IS NOT NULL AND deleted_at IS NULL"),
    )
    op.create_index(
        "idx_memories_source_page",
        "memories",
        ["source_page_id"],
        postgresql_where=sa.text("source_page_id IS NOT NULL AND deleted_at IS NULL"),
    )


def downgrade() -> None:
    """Remove source attribution indexes/columns from memories and drop document_pages."""
    # Drop memories indexes first
    op.drop_index("idx_memories_source_page", table_name="memories")
    op.drop_index("idx_memories_source_document", table_name="memories")

    # Drop memories columns
    op.drop_column("memories", "source_page_id")
    op.drop_column("memories", "source_document_id")

    # Drop document_pages indexes then table
    op.drop_index("idx_document_pages_workspace_document", table_name="document_pages")
    op.drop_index("idx_document_pages_document", table_name="document_pages")
    op.drop_index("idx_document_pages_workspace", table_name="document_pages")
    op.drop_table("document_pages")
