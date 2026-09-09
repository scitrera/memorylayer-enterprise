# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add deterministic checkpoint, context-event, and entity-relation storage.

Revision ID: 036
Revises: 035
Create Date: 2026-09-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "036"
down_revision: str | None = "035"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "session_context_events",
        sa.Column("sequence", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=True),
        sa.Column("event_kind", sa.Text(), nullable=False),
        sa.Column("subject_kind", sa.Text(), nullable=False),
        sa.Column("subject_id", sa.Text(), nullable=False),
        sa.Column("event_time", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "idx_session_context_events_scope",
        "session_context_events",
        ["workspace_id", "session_id", "sequence"],
    )
    op.create_index(
        "idx_session_context_events_retention",
        "session_context_events",
        ["event_time"],
    )

    op.create_table(
        "session_checkpoints",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("raw_memory_id", sa.Text(), nullable=False),
        sa.Column("source_kind", sa.Text(), nullable=False),
        sa.Column("source_sequence", sa.BigInteger(), nullable=True),
        sa.Column("source_boundary", sa.BigInteger(), nullable=True),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("byte_count", sa.BigInteger(), nullable=False),
        sa.Column("capture_status", sa.Text(), nullable=False),
        sa.Column("index_status", sa.Text(), nullable=False),
        sa.Column("enrichment_status", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["raw_memory_id"], ["memories.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("session_id", "idempotency_key", name="uq_session_checkpoint_idempotency"),
    )
    op.create_index(
        "idx_session_checkpoints_scope",
        "session_checkpoints",
        ["workspace_id", "session_id", "created_at"],
    )

    op.create_table(
        "entity_relations",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("source_entity_id", sa.Text(), nullable=False),
        sa.Column("target_entity_id", sa.Text(), nullable=False),
        sa.Column("relationship", sa.Text(), nullable=False),
        sa.Column("direction", sa.Text(), nullable=False, server_default="outgoing"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("source_entity_id <> target_entity_id", name="ck_entity_relation_not_self"),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_entity_relation_confidence"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_entity_id"], ["entities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_entity_id"], ["entities.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "uq_entity_relations_active_edge",
        "entity_relations",
        ["workspace_id", "source_entity_id", "target_entity_id", "relationship"],
        unique=True,
        postgresql_where=sa.text("active = true"),
    )
    op.create_index(
        "idx_entity_relations_source",
        "entity_relations",
        ["workspace_id", "source_entity_id", "relationship"],
    )
    op.create_index(
        "idx_entity_relations_target",
        "entity_relations",
        ["workspace_id", "target_entity_id", "relationship"],
    )

    op.create_table(
        "entity_relation_evidence",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("relation_id", sa.Text(), nullable=False),
        sa.Column("source_memory_id", sa.Text(), nullable=False),
        sa.Column("evidence_kind", sa.Text(), nullable=False),
        sa.Column("source_span_start", sa.Integer(), nullable=True),
        sa.Column("source_span_end", sa.Integer(), nullable=True),
        sa.Column("excerpt_hash", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("extraction_method", sa.Text(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_entity_relation_evidence_confidence"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["relation_id"], ["entity_relations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_memory_id"], ["memories.id"], ondelete="CASCADE"),
    )
    op.execute(sa.text(
        "CREATE UNIQUE INDEX uq_entity_relation_evidence_source_span "
        "ON entity_relation_evidence (relation_id, source_memory_id, "
        "COALESCE(source_span_start, -1), COALESCE(source_span_end, -1), excerpt_hash)"
    ))
    op.create_index(
        "idx_entity_relation_evidence_relation",
        "entity_relation_evidence",
        ["workspace_id", "relation_id", "active"],
    )
    op.create_index(
        "idx_entity_relation_evidence_memory",
        "entity_relation_evidence",
        ["workspace_id", "source_memory_id", "active"],
    )


def downgrade() -> None:
    op.drop_table("entity_relation_evidence")
    op.drop_table("entity_relations")
    op.drop_table("session_checkpoints")
    op.drop_table("session_context_events")
