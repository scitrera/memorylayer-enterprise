"""Add scope column to chat_threads table.

Revision ID: 015
Revises: 014
Create Date: 2026-04-28 00:00:00

Adds a nullable ``scope`` text column to ``chat_threads``.  The column is used
to separate threads created by the web application (scope='web') from threads
created by the Office add-in platform bridge (scope='office').

Existing rows receive NULL, which the application treats as equivalent to
'web'.  No backfill is required.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add nullable scope column to chat_threads."""
    op.add_column(
        "chat_threads",
        sa.Column("scope", sa.Text(), nullable=True),
    )
    op.create_index("idx_chat_threads_scope", "chat_threads", ["workspace_id", "scope"])


def downgrade() -> None:
    """Remove scope column from chat_threads."""
    op.drop_index("idx_chat_threads_scope", table_name="chat_threads")
    op.drop_column("chat_threads", "scope")
