# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add skills and skill_files tables for Agent Skills feature.

Revision ID: 013
Revises: 012
Create Date: 2026-04-25 12:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create skills and skill_files tables."""
    op.create_table(
        "skills",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column("tenant_id", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "workspace_id",
            sa.Text(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False, server_default="0.1.0"),
        sa.Column("license", sa.Text(), nullable=True),
        sa.Column("compatibility", sa.Text(), nullable=True),
        sa.Column("allowed_tools", sa.Text(), nullable=True),
        sa.Column("body", sa.Text(), nullable=False, server_default=""),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("source_mode", sa.Text(), nullable=False, server_default="server"),
        sa.Column("manifest_hash", sa.Text(), nullable=False, server_default=""),
        sa.Column("bundle_hash", sa.Text(), nullable=False, server_default=""),
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

    # Basic workspace lookup index
    op.create_index("idx_skills_workspace", "skills", ["workspace_id"])
    # Tenant + workspace filtering
    op.create_index("idx_skills_tenant_workspace", "skills", ["tenant_id", "workspace_id"])
    # Cross-scope name lookup
    op.create_index("idx_skills_name", "skills", ["name"])
    # Unique workspace-scoped name (no user scope) — partial index
    op.execute(
        "CREATE UNIQUE INDEX idx_skills_workspace_name_global "
        "ON skills (workspace_id, name) WHERE user_id IS NULL"
    )
    # Unique user-scoped name per user — partial index
    op.execute(
        "CREATE UNIQUE INDEX idx_skills_workspace_user_name "
        "ON skills (workspace_id, user_id, name) WHERE user_id IS NOT NULL"
    )

    op.create_table(
        "skill_files",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column(
            "skill_id",
            sa.Text(),
            sa.ForeignKey("skills.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("mime_type", sa.Text(), nullable=True),
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
        sa.UniqueConstraint("skill_id", "path", name="uq_skill_file_path"),
    )

    op.create_index("idx_skill_files_skill", "skill_files", ["skill_id"])


def downgrade() -> None:
    """Drop skills and skill_files tables."""
    # Drop skill_files first (FK dependency)
    op.drop_index("idx_skill_files_skill", table_name="skill_files")
    op.drop_table("skill_files")

    # Drop skills and its indexes
    op.drop_index("idx_skills_workspace_user_name", table_name="skills")
    op.drop_index("idx_skills_workspace_name_global", table_name="skills")
    op.drop_index("idx_skills_name", table_name="skills")
    op.drop_index("idx_skills_tenant_workspace", table_name="skills")
    op.drop_index("idx_skills_workspace", table_name="skills")
    op.drop_table("skills")
