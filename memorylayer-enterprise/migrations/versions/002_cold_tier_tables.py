"""Cold tier tables for LEANN storage

Revision ID: 002
Revises: 001
Create Date: 2026-01-27 12:00:00

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply migration changes - create LEANN cold tier tables."""
    # Create leann_graphs table for storing compressed neighbor graph structures
    op.create_table(
        "leann_graphs",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("graph_data", sa.LargeBinary(), nullable=False),
        sa.Column("node_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("edge_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("memory_ids", sa.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column(
            "metadata", sa.dialects.postgresql.JSONB(), server_default="{}", nullable=False
        ),
        sa.Column(
            "created_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for leann_graphs
    op.create_index("idx_leann_graphs_workspace", "leann_graphs", ["workspace_id"])
    op.create_index(
        "idx_leann_graphs_memory_ids",
        "leann_graphs",
        ["memory_ids"],
        postgresql_using="gin",
    )

    # Create leann_documents table for storing archived memory content
    op.create_table(
        "leann_documents",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("graph_id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("memory_id", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("cold_access_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_cold_access_at", sa.dialects.postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("memory_type", sa.Text(), nullable=True),
        sa.Column("memory_subtype", sa.Text(), nullable=True),
        sa.Column("importance", sa.Float(), server_default="0.5", nullable=False),
        sa.Column("tags", sa.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column(
            "metadata", sa.dialects.postgresql.JSONB(), server_default="{}", nullable=False
        ),
        sa.Column(
            "created_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["graph_id"], ["leann_graphs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("graph_id", "position", name="uq_leann_document_position"),
    )

    # Create indexes for leann_documents
    op.create_index("idx_leann_documents_workspace", "leann_documents", ["workspace_id"])
    op.create_index("idx_leann_documents_graph", "leann_documents", ["graph_id"])
    op.create_index("idx_leann_documents_memory", "leann_documents", ["memory_id"])
    op.create_index(
        "idx_leann_documents_tags",
        "leann_documents",
        ["tags"],
        postgresql_using="gin",
    )
    op.create_index(
        "idx_leann_documents_cold_access",
        "leann_documents",
        ["workspace_id", "cold_access_count"],
    )


def downgrade() -> None:
    """Revert migration changes - drop LEANN cold tier tables."""
    # Drop leann_documents indexes first
    op.drop_index("idx_leann_documents_cold_access", table_name="leann_documents")
    op.drop_index("idx_leann_documents_tags", table_name="leann_documents")
    op.drop_index("idx_leann_documents_memory", table_name="leann_documents")
    op.drop_index("idx_leann_documents_graph", table_name="leann_documents")
    op.drop_index("idx_leann_documents_workspace", table_name="leann_documents")
    op.drop_table("leann_documents")

    # Drop leann_graphs indexes and table
    op.drop_index("idx_leann_graphs_memory_ids", table_name="leann_graphs")
    op.drop_index("idx_leann_graphs_workspace", table_name="leann_graphs")
    op.drop_table("leann_graphs")
