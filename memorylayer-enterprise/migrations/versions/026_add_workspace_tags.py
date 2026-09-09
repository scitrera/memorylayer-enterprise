# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add tags array to workspaces for tag-based workspace lookup.

Revision ID: 026
Revises: 025
Create Date: 2026-06-12 00:00:00

Adds a free-form ``tags TEXT[]`` column to ``workspaces`` so workspaces can be
grouped and discovered by tag (e.g. a ``knowledge`` workspace, optionally further
tagged ``topic:finance``). This mirrors the existing ``memories.tags`` pattern
(``ARRAY(Text)`` + GIN index) and backs the new
``GET /v1/workspaces?tags=...&match=any|all`` lookup, which queries the column with
PostgreSQL array operators (``@>`` for ``all`` containment, ``&&`` for ``any``
overlap). The GIN index makes those containment/overlap lookups efficient.

The column is ``NOT NULL DEFAULT '{}'`` so existing rows backfill to an empty tag
list. downgrade() drops the index and the column.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "026"
down_revision: Union[str, None] = "025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the workspaces.tags column and its GIN index."""
    op.add_column(
        "workspaces",
        sa.Column("tags", sa.ARRAY(sa.Text()), server_default="{}", nullable=False),
    )
    op.create_index(
        "idx_workspaces_tags",
        "workspaces",
        ["tags"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    """Drop the GIN index and the tags column."""
    op.drop_index("idx_workspaces_tags", table_name="workspaces")
    op.drop_column("workspaces", "tags")
