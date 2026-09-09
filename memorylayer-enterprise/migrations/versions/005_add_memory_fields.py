# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add OSS Memory fields to enterprise memories table.

Revision ID: 005
Revises: 004
Create Date: 2026-02-20 12:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply migration changes - add OSS Memory fields to memories table."""
    op.add_column("memories", sa.Column("abstract", sa.Text(), nullable=True))
    op.add_column("memories", sa.Column("overview", sa.Text(), nullable=True))
    op.add_column("memories", sa.Column("session_id", sa.Text(), nullable=True))
    op.add_column("memories", sa.Column("source_memory_id", sa.Text(), nullable=True))
    op.add_column("memories", sa.Column("category", sa.Text(), nullable=True))
    op.add_column("memories", sa.Column("status", sa.Text(), nullable=False, server_default="active"))
    op.add_column("memories", sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.text("false")))


def downgrade() -> None:
    """Revert migration changes - drop OSS Memory fields from memories table."""
    op.drop_column("memories", "pinned")
    op.drop_column("memories", "status")
    op.drop_column("memories", "category")
    op.drop_column("memories", "source_memory_id")
    op.drop_column("memories", "session_id")
    op.drop_column("memories", "overview")
    op.drop_column("memories", "abstract")
