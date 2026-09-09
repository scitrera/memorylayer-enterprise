"""Initial schema for MemoryLayer.ai

Revision ID: 001
Revises:
Create Date: 2026-01-26 16:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply migration changes - create all tables for MemoryLayer.ai."""
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create workspaces table (tenant boundary)
    op.create_table(
        "workspaces",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("settings", sa.dialects.postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column(
            "created_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create memory_spaces table
    op.create_table(
        "memory_spaces",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("settings", sa.dialects.postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column(
            "created_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workspace_id", "name", name="uq_workspace_space_name"),
    )

    # Create memories table with pgvector embedding
    op.create_table(
        "memories",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("space_id", sa.Text(), nullable=True),
        sa.Column("user_id", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("subtype", sa.Text(), nullable=True),
        sa.Column("importance", sa.Float(), server_default="0.5", nullable=False),
        sa.Column("tags", sa.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("access_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_accessed_at", sa.dialects.postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("decay_factor", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("archived_at", sa.dialects.postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.dialects.postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "type IN ('episodic', 'semantic', 'procedural', 'working')", name="ck_memory_type"
        ),
        sa.CheckConstraint("importance >= 0 AND importance <= 1", name="ck_memory_importance"),
        sa.ForeignKeyConstraint(["space_id"], ["memory_spaces.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for memories
    op.create_index(
        "idx_memories_workspace",
        "memories",
        ["workspace_id"],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "idx_memories_workspace_type",
        "memories",
        ["workspace_id", "type"],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "idx_memories_workspace_user",
        "memories",
        ["workspace_id", "user_id"],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "idx_memories_tags",
        "memories",
        ["tags"],
        postgresql_using="gin",
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "idx_memories_embedding",
        "memories",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
        postgresql_where=sa.text("embedding IS NOT NULL AND deleted_at IS NULL"),
    )

    # Create memory_associations table (graph edges)
    op.create_table(
        "memory_associations",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("target_id", sa.Text(), nullable=False),
        sa.Column("relationship", sa.Text(), nullable=False),
        sa.Column("strength", sa.Float(), server_default="0.5", nullable=False),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column(
            "created_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("strength >= 0 AND strength <= 1", name="ck_association_strength"),
        sa.ForeignKeyConstraint(["source_id"], ["memories.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_id"], ["memories.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id", "target_id", "relationship", name="uq_association"),
    )

    # Create indexes for memory_associations
    op.create_index("idx_associations_workspace", "memory_associations", ["workspace_id"])
    op.create_index("idx_associations_source", "memory_associations", ["source_id"])
    op.create_index("idx_associations_target", "memory_associations", ["target_id"])

    # Create memory_fragments table
    op.create_table(
        "memory_fragments",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("memory_id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column(
            "created_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["memory_id"], ["memories.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("memory_id", "sequence", name="uq_fragment_sequence"),
    )

    # Create indexes for memory_fragments
    op.create_index("idx_fragments_memory", "memory_fragments", ["memory_id"])
    op.create_index(
        "idx_fragments_embedding",
        "memory_fragments",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    # Create resources table (raw data layer)
    op.create_table(
        "resources",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("content", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("processed", sa.Boolean(), server_default="false", nullable=False),
        sa.Column(
            "created_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create categories table (aggregated summaries)
    op.create_table(
        "categories",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("item_ids", sa.ARRAY(sa.Text()), server_default="{}", nullable=False),
        sa.Column(
            "last_updated",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workspace_id", "name", name="uq_workspace_category_name"),
    )

    # Create sessions table
    op.create_table(
        "sessions",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=True),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("expires_at", sa.dialects.postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create session_context table
    op.create_table(
        "session_context",
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("value", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("ttl_seconds", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("session_id", "key"),
    )

    # Create memory_access_log table (for analytics)
    op.create_table(
        "memory_access_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("memory_id", sa.Text(), nullable=True),
        sa.Column("access_type", sa.Text(), nullable=False),
        sa.Column("query", sa.Text(), nullable=True),
        sa.Column("relevance_score", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "accessed_at",
            sa.dialects.postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for memory_access_log
    op.create_index("idx_access_log_workspace", "memory_access_log", ["workspace_id"])
    op.create_index("idx_access_log_accessed_at", "memory_access_log", ["accessed_at"])


def downgrade() -> None:
    """Revert migration changes - drop all tables."""
    op.drop_index("idx_access_log_accessed_at", table_name="memory_access_log")
    op.drop_index("idx_access_log_workspace", table_name="memory_access_log")
    op.drop_table("memory_access_log")
    op.drop_table("session_context")
    op.drop_table("sessions")
    op.drop_table("categories")
    op.drop_table("resources")
    op.drop_index("idx_fragments_embedding", table_name="memory_fragments")
    op.drop_index("idx_fragments_memory", table_name="memory_fragments")
    op.drop_table("memory_fragments")
    op.drop_index("idx_associations_target", table_name="memory_associations")
    op.drop_index("idx_associations_source", table_name="memory_associations")
    op.drop_index("idx_associations_workspace", table_name="memory_associations")
    op.drop_table("memory_associations")
    op.drop_index("idx_memories_embedding", table_name="memories")
    op.drop_index("idx_memories_tags", table_name="memories")
    op.drop_index("idx_memories_workspace_user", table_name="memories")
    op.drop_index("idx_memories_workspace_type", table_name="memories")
    op.drop_index("idx_memories_workspace", table_name="memories")
    op.drop_table("memories")
    op.drop_table("memory_spaces")
    op.drop_table("workspaces")
    op.execute("DROP EXTENSION IF EXISTS vector")
