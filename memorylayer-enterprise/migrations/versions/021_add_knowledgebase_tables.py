"""Add knowledgebase_articles and graph_analyses tables.

Revision ID: 021
Revises: 020
Create Date: 2026-06-02 12:00:00

Adds the PostgreSQL KB storage layer that the SQLite backend already had:

- ``knowledgebase_articles`` — generated KB content, keyed by
  (workspace_id, article_id). The reserved ``article_id="index"`` row holds
  the index article and is upserted like any other.
- ``graph_analyses`` — cached graph-analysis payloads, one row per workspace.

Mirrors the SQLite definitions in
``memorylayer_server/services/storage/sqlite.py`` and the ORM models
``KnowledgebaseArticleModel`` / ``GraphAnalysisModel``.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "021"
down_revision: Union[str, None] = "020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create knowledgebase_articles and graph_analyses tables."""
    op.create_table(
        "knowledgebase_articles",
        sa.Column(
            "workspace_id",
            sa.Text(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("article_id", sa.Text(), nullable=False, primary_key=True),
        sa.Column("article_type", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("content_md", sa.Text(), nullable=True),
        sa.Column(
            "metadata",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "generated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "idx_kb_articles_workspace", "knowledgebase_articles", ["workspace_id"]
    )
    op.create_index(
        "idx_kb_articles_workspace_type",
        "knowledgebase_articles",
        ["workspace_id", "article_type"],
    )

    op.create_table(
        "graph_analyses",
        sa.Column(
            "workspace_id",
            sa.Text(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "analysis_json",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "generated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )


def downgrade() -> None:
    """Drop knowledgebase_articles and graph_analyses tables."""
    op.drop_table("graph_analyses")
    op.drop_index("idx_kb_articles_workspace_type", table_name="knowledgebase_articles")
    op.drop_index("idx_kb_articles_workspace", table_name="knowledgebase_articles")
    op.drop_table("knowledgebase_articles")
