"""Add collection_items table for vector collections.

Revision ID: 011
Revises: 010
Create Date: 2026-03-24 12:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create collection_items table with pgvector embedding column."""
    # Get embedding dimensions from environment (default 1536)
    import os
    embedding_dim = int(os.environ.get('MEMORYLAYER_EMBEDDING_DIMENSIONS', '1536'))

    op.create_table(
        "collection_items",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column(
            "workspace_id",
            sa.Text(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("collection_name", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("item_type", sa.Text(), nullable=True),
        sa.Column("tags", sa.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), nullable=False, server_default="{}"),
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

    # Add vector column via raw SQL (pgvector extension)
    op.execute(
        f"ALTER TABLE collection_items ADD COLUMN embedding vector({embedding_dim})"
    )

    op.create_index("idx_collection_items_workspace", "collection_items", ["workspace_id"])
    op.create_index(
        "idx_collection_items_collection", "collection_items", ["workspace_id", "collection_name"]
    )
    op.create_index(
        "idx_collection_items_tags", "collection_items", ["tags"], postgresql_using="gin"
    )

    # HNSW index on embedding for cosine similarity search
    op.execute(
        "CREATE INDEX idx_collection_items_embedding ON collection_items "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64) "
        "WHERE embedding IS NOT NULL"
    )


def downgrade() -> None:
    """Drop collection_items table."""
    op.execute("DROP INDEX IF EXISTS idx_collection_items_embedding")
    op.drop_index("idx_collection_items_tags", table_name="collection_items")
    op.drop_index("idx_collection_items_collection", table_name="collection_items")
    op.drop_index("idx_collection_items_workspace", table_name="collection_items")
    op.drop_table("collection_items")
