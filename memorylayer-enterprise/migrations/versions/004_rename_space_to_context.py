# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Rename memory_spaces to contexts and space_id to context_id.

Revision ID: 004
Revises: 003
Create Date: 2026-02-20 12:00:00

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply migration changes - rename memory_spaces to contexts and space_id to context_id."""
    # Rename table
    op.rename_table("memory_spaces", "contexts")

    # Rename column on memories table
    op.alter_column("memories", "space_id", new_column_name="context_id")

    # Rename unique constraint
    op.drop_constraint("uq_workspace_space_name", "contexts", type_="unique")
    op.create_unique_constraint("uq_workspace_context_name", "contexts", ["workspace_id", "name"])

    # Recreate foreign key with new names
    op.drop_constraint("memories_space_id_fkey", "memories", type_="foreignkey")
    op.create_foreign_key(
        "memories_context_id_fkey", "memories", "contexts",
        ["context_id"], ["id"], ondelete="SET NULL"
    )


def downgrade() -> None:
    """Revert migration changes - rename contexts back to memory_spaces and context_id to space_id."""
    op.drop_constraint("memories_context_id_fkey", "memories", type_="foreignkey")
    op.create_foreign_key(
        "memories_space_id_fkey", "memories", "memory_spaces",
        ["space_id"], ["id"], ondelete="SET NULL"
    )
    op.drop_constraint("uq_workspace_context_name", "contexts", type_="unique")
    op.create_unique_constraint("uq_workspace_space_name", "memory_spaces", ["workspace_id", "name"])
    op.alter_column("memories", "context_id", new_column_name="space_id")
    op.rename_table("contexts", "memory_spaces")
