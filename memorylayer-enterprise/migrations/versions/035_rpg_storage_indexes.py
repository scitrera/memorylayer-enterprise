# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add PostgreSQL indexes for Repository Planning Graph workloads.

Revision ID: 035
Revises: 034
Create Date: 2026-08-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "035"
down_revision: str | None = "034"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ACTIVE_RPG = sa.text(
    "deleted_at IS NULL AND status = 'active' "
    "AND tags @> ARRAY['rpg']::text[]"
)
_RPG_RELATIONSHIPS = sa.text(
    "relationship IN ('contains', 'contained_by', 'inherits', "
    "'inherited_by', 'invokes', 'invoked_by', 'imports', "
    "'imported_by', 'composes', 'composed_by', 'data_flow', "
    "'data_flow_from')"
)


def upgrade() -> None:
    op.create_index(
        "idx_memories_rpg_context_subtype",
        "memories",
        ["workspace_id", "context_id", "subtype"],
        postgresql_where=_ACTIVE_RPG,
    )
    op.create_index(
        "idx_memories_rpg_metadata",
        "memories",
        ["metadata"],
        postgresql_using="gin",
        postgresql_ops={"metadata": "jsonb_path_ops"},
        postgresql_where=_ACTIVE_RPG,
    )
    op.create_index(
        "idx_associations_rpg_source",
        "memory_associations",
        ["workspace_id", "source_id", "relationship"],
        postgresql_where=_RPG_RELATIONSHIPS,
    )
    op.create_index(
        "idx_associations_rpg_target",
        "memory_associations",
        ["workspace_id", "target_id", "relationship"],
        postgresql_where=_RPG_RELATIONSHIPS,
    )


def downgrade() -> None:
    op.drop_index("idx_associations_rpg_target", table_name="memory_associations")
    op.drop_index("idx_associations_rpg_source", table_name="memory_associations")
    op.drop_index("idx_memories_rpg_metadata", table_name="memories")
    op.drop_index("idx_memories_rpg_context_subtype", table_name="memories")
