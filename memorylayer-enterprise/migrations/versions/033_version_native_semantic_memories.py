# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add native semantic-memory CAS and immutable history.

Revision ID: 033
Revises: 032
Create Date: 2026-08-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "033"
down_revision: str | None = "032"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("logical_key", sa.Text(), nullable=True))
    op.add_column(
        "memories",
        sa.Column(
            "refinement_metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
    )
    op.add_column(
        "memories",
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "memories",
        sa.Column("etag", sa.Text(), nullable=False, server_default=""),
    )
    op.create_index(
        "idx_memories_workspace_logical_key_global",
        "memories",
        ["workspace_id", "logical_key"],
        unique=True,
        postgresql_where=sa.text("user_id IS NULL AND logical_key IS NOT NULL"),
    )
    op.create_index(
        "idx_memories_workspace_user_logical_key",
        "memories",
        ["workspace_id", "user_id", "logical_key"],
        unique=True,
        postgresql_where=sa.text("user_id IS NOT NULL AND logical_key IS NOT NULL"),
    )

    op.create_table(
        "memory_revisions",
        sa.Column("sequence", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("memory_id", sa.Text(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("operation_id", sa.Text(), nullable=False),
        sa.Column("request_hash", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "tenant_id",
            "workspace_id",
            "memory_id",
            "revision",
            name="uq_memory_revision",
        ),
    )
    op.create_index(
        "idx_memory_revisions_resource",
        "memory_revisions",
        ["tenant_id", "workspace_id", "memory_id", "sequence"],
    )
    op.create_table(
        "memory_operations",
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("operation_id", sa.Text(), nullable=False),
        sa.Column("request_hash", sa.Text(), nullable=False),
        sa.Column("memory_id", sa.Text(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint(
            "tenant_id", "workspace_id", "operation_id",
            name="pk_memory_operations",
        ),
    )

    # Adopt every existing typed row as revision one. The migration ETag is
    # opaque and stable; the next accepted mutation uses the canonical SHA-256
    # form. Large vector payloads and the generated FTS projection are omitted.
    op.execute(sa.text("""
        UPDATE memories
        SET revision = 1,
            etag = '"memory-1-' || md5(
                tenant_id || ':' || workspace_id || ':' || id || ':' ||
                content_hash || ':' || pinned::text || ':' ||
                coalesce(deleted_at::text, '')
            ) || '"'
        WHERE revision = 0 OR etag = ''
    """))
    op.execute(sa.text("""
        INSERT INTO memory_revisions (
            tenant_id, workspace_id, memory_id, revision, snapshot,
            action, operation_id, request_hash
        )
        SELECT tenant_id, workspace_id, id, revision,
               to_jsonb(memories) - 'embedding' - 'multivector' - 'fts_content',
               'create', 'legacy-adopt:' || id,
               md5('legacy-adopt:' || tenant_id || ':' || workspace_id || ':' || id)
        FROM memories
    """))
    op.execute(sa.text("""
        INSERT INTO memory_operations (
            tenant_id, workspace_id, operation_id, request_hash,
            memory_id, revision
        )
        SELECT tenant_id, workspace_id, operation_id, request_hash,
               memory_id, revision
        FROM memory_revisions
        WHERE operation_id LIKE 'legacy-adopt:%'
    """))


def downgrade() -> None:
    op.drop_table("memory_operations")
    op.drop_index("idx_memory_revisions_resource", table_name="memory_revisions")
    op.drop_table("memory_revisions")
    op.drop_index("idx_memories_workspace_user_logical_key", table_name="memories")
    op.drop_index("idx_memories_workspace_logical_key_global", table_name="memories")
    op.drop_column("memories", "etag")
    op.drop_column("memories", "revision")
    op.drop_column("memories", "refinement_metadata")
    op.drop_column("memories", "logical_key")
