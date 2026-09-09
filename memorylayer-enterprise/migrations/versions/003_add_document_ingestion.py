"""Add document ingestion tables

Revision ID: 003
Revises: 002
Create Date: 2026-02-20 12:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply migration changes - create document ingestion tables."""
    # Create documents table
    op.create_table(
        "documents",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Text(), server_default="default_tenant", nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("document_type", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("mime_type", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), server_default="pending", nullable=False),
        sa.Column("target_context_id", sa.Text(), server_default="_default", nullable=False),
        sa.Column(
            "extraction_options", sa.dialects.postgresql.JSONB(), server_default="{}", nullable=False
        ),
        sa.Column("page_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("chunk_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("memory_ids", sa.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("deduplicated_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=True),
        sa.Column("retain_original", sa.Boolean(), server_default="true", nullable=False),
        sa.Column(
            "metadata", sa.dialects.postgresql.JSONB(), server_default="{}", nullable=False
        ),
        sa.Column(
            "extracted_metadata", sa.dialects.postgresql.JSONB(), server_default="{}", nullable=False
        ),
        sa.Column(
            "created_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("processing_started_at", sa.dialects.postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("processing_completed_at", sa.dialects.postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed', 'partial')",
            name="ck_document_status",
        ),
        sa.CheckConstraint(
            "document_type IN ('pdf', 'markdown', 'text', 'html', 'docx', 'pptx')",
            name="ck_document_type",
        ),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for documents
    op.create_index("idx_documents_workspace", "documents", ["workspace_id"])
    op.create_index("idx_documents_content_hash", "documents", ["workspace_id", "content_hash"])
    op.create_index("idx_documents_status", "documents", ["workspace_id", "status"])

    # Create ingestion_jobs table
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("document_ids", sa.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("status", sa.Text(), server_default="queued", nullable=False),
        sa.Column("progress_percent", sa.Integer(), server_default="0", nullable=False),
        sa.Column("documents_processed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("total_memories_created", sa.Integer(), server_default="0", nullable=False),
        sa.Column("webhook_url", sa.Text(), nullable=True),
        sa.Column(
            "errors", sa.dialects.postgresql.JSONB(), server_default="[]", nullable=False
        ),
        sa.Column(
            "created_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.dialects.postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("completed_at", sa.dialects.postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_job_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for ingestion_jobs
    op.create_index("idx_jobs_workspace", "ingestion_jobs", ["workspace_id"])
    op.create_index("idx_jobs_status", "ingestion_jobs", ["workspace_id", "status"])
    op.create_index("idx_jobs_created_at", "ingestion_jobs", ["created_at"])


def downgrade() -> None:
    """Revert migration changes - drop document ingestion tables."""
    # Drop ingestion_jobs indexes first
    op.drop_index("idx_jobs_created_at", table_name="ingestion_jobs")
    op.drop_index("idx_jobs_status", table_name="ingestion_jobs")
    op.drop_index("idx_jobs_workspace", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")

    # Drop documents indexes and table
    op.drop_index("idx_documents_status", table_name="documents")
    op.drop_index("idx_documents_content_hash", table_name="documents")
    op.drop_index("idx_documents_workspace", table_name="documents")
    op.drop_table("documents")
