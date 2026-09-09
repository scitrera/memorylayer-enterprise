"""Add native skill-manifest CAS and immutable history.

Revision ID: 032
Revises: 031
Create Date: 2026-08-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "032"
down_revision: str | None = "031"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "skills",
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "skills",
        sa.Column("etag", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "skills",
        sa.Column("deleted_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.create_index("idx_skills_active", "skills", ["workspace_id", "deleted_at"])

    op.create_table(
        "skill_revisions",
        sa.Column("sequence", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("skill_id", sa.Text(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("operation_id", sa.Text(), nullable=False),
        sa.Column("request_hash", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "tenant_id",
            "workspace_id",
            "skill_id",
            "revision",
            name="uq_skill_revision",
        ),
    )
    op.create_index(
        "idx_skill_revisions_resource",
        "skill_revisions",
        ["tenant_id", "workspace_id", "skill_id", "sequence"],
    )
    op.create_table(
        "skill_operations",
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("operation_id", sa.Text(), nullable=False),
        sa.Column("request_hash", sa.Text(), nullable=False),
        sa.Column("skill_id", sa.Text(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint(
            "tenant_id", "workspace_id", "operation_id",
            name="pk_skill_operations",
        ),
    )

    # Existing rows become deterministic revision-one heads. The migration
    # ETag is opaque and stable; the next application mutation uses the normal
    # canonical SHA-256 form. to_jsonb captures the exact adopted result.
    op.execute(sa.text("""
        UPDATE skills
        SET revision = 1,
            etag = '"skill-1-' || md5(
                tenant_id || ':' || workspace_id || ':' || id || ':' ||
                manifest_hash || ':' || enabled::text
            ) || '"'
        WHERE revision = 0 OR etag = ''
    """))
    op.execute(sa.text("""
        INSERT INTO skill_revisions (
            tenant_id, workspace_id, skill_id, revision, snapshot,
            action, operation_id, request_hash
        )
        SELECT tenant_id, workspace_id, id, revision, to_jsonb(skills),
               'create', 'legacy-adopt:' || id,
               md5('legacy-adopt:' || tenant_id || ':' || workspace_id || ':' || id)
        FROM skills
    """))
    op.execute(sa.text("""
        INSERT INTO skill_operations (
            tenant_id, workspace_id, operation_id, request_hash,
            skill_id, revision
        )
        SELECT tenant_id, workspace_id, operation_id, request_hash,
               skill_id, revision
        FROM skill_revisions
        WHERE operation_id LIKE 'legacy-adopt:%'
    """))


def downgrade() -> None:
    op.drop_table("skill_operations")
    op.drop_index("idx_skill_revisions_resource", table_name="skill_revisions")
    op.drop_table("skill_revisions")
    op.drop_index("idx_skills_active", table_name="skills")
    op.drop_column("skills", "deleted_at")
    op.drop_column("skills", "etag")
    op.drop_column("skills", "revision")
