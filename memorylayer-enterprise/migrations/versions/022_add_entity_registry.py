"""Add entity registry tables (entities, entity_aliases, entity_members).

Revision ID: 022
Revises: 021
Create Date: 2026-06-03 12:00:00

The PostgreSQL mirror of the entity-registry storage the SQLite backend gained
in ``memorylayer_server/services/storage/sqlite.py`` (slice 1: canonical entity
nodes + aliases + member accretion). Mirrors the ORM models
``EntityModel`` / ``EntityAliasModel`` / ``EntityMemberModel``.

- ``entities`` — canonical, workspace-scoped entity nodes. The exact-match
  index is a PARTIAL unique index over active rows only
  (``WHERE status = 'active'``, following the ``006`` partial-index precedent)
  so merged tombstones do not collide with live rows.
- ``entity_aliases`` — alternate surface forms folded in over time.
- ``entity_members`` — membership edges (a memory mentions/belongs to an
  entity). Mirrors ``memory_associations``: CASCADE FKs, named indexes, a
  uniqueness constraint on (entity_id, memory_id, role).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "022"
down_revision: Union[str, None] = "021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create entities, entity_aliases, and entity_members tables."""
    op.create_table(
        "entities",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column(
            "workspace_id",
            sa.Text(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("entity_type", sa.Text(), nullable=False),
        sa.Column("canonical_name", sa.Text(), nullable=False),
        sa.Column("normalized_name", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column(
            "provenance",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("representative_memory_id", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="active"),
        sa.Column("merged_into", sa.Text(), nullable=True),
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
    op.create_index("idx_entities_workspace", "entities", ["workspace_id"])
    # Exact-match index: at most one ACTIVE entity per (workspace, type,
    # normalized_name). Partial unique index (active rows only) so merged
    # tombstones can keep the same normalized_name without colliding.
    op.create_index(
        "uq_entities_active_norm",
        "entities",
        ["workspace_id", "entity_type", "normalized_name"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    op.create_table(
        "entity_aliases",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column(
            "entity_id",
            sa.Text(),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("alias", sa.Text(), nullable=False),
        sa.Column("normalized_alias", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False, server_default="manual"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("entity_id", "normalized_alias", name="uq_entity_aliases_entity_norm"),
    )
    op.create_index(
        "idx_entity_aliases_norm", "entity_aliases", ["workspace_id", "normalized_alias"]
    )

    op.create_table(
        "entity_members",
        sa.Column("id", sa.Text(), nullable=False, primary_key=True),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column(
            "entity_id",
            sa.Text(),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "memory_id",
            sa.Text(),
            sa.ForeignKey("memories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.Text(), nullable=False, server_default="mention"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column(
            "meta",
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
        sa.UniqueConstraint("entity_id", "memory_id", "role", name="uq_entity_member"),
    )
    op.create_index("idx_entity_members_entity", "entity_members", ["workspace_id", "entity_id"])
    op.create_index("idx_entity_members_memory", "entity_members", ["workspace_id", "memory_id"])


def downgrade() -> None:
    """Drop entity registry tables."""
    op.drop_index("idx_entity_members_memory", table_name="entity_members")
    op.drop_index("idx_entity_members_entity", table_name="entity_members")
    op.drop_table("entity_members")
    op.drop_index("idx_entity_aliases_norm", table_name="entity_aliases")
    op.drop_table("entity_aliases")
    op.drop_index("uq_entities_active_norm", table_name="entities")
    op.drop_index("idx_entities_workspace", table_name="entities")
    op.drop_table("entities")
