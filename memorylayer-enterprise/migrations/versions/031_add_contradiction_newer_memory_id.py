"""Persist the supersession direction on contradiction records.

Revision ID: 031
Revises: 030
Create Date: 2026-08-03 00:00:00

BACKGROUND
----------
``ContradictionService`` has always computed which of two conflicting memories is the
CURRENT one — ``_determine_newer_memory`` compares ``event_time`` / ``created_at`` and the
detector sets ``ContradictionRecord.newer_memory_id`` on every record it writes. Neither
storage backend persisted it. The dataclass field existed, the column did not, so a stored
contradiction recorded that two memories conflict but not which one is stale, and the
direction was recomputed and discarded on every store.

That made recall-side supersession impossible to build: there was nothing durable saying
which side to demote or drop.

WHAT THIS DOES
--------------
Adds a nullable ``newer_memory_id TEXT`` to ``contradictions``, plus a partial index on
``(workspace_id, newer_memory_id) WHERE resolved_at IS NULL`` backing the read-path query
"of these memory ids, which are the stale side of an unresolved contradiction".

Existing rows backfill to NULL, and NULL is meaningful rather than missing: a record with
no recorded direction supersedes nothing. The read path treats it as "we know these two
conflict, we do not know which is current" and declines to guess, so pre-existing
contradictions keep their current behaviour exactly — they simply never trigger
supersession. No backfill is attempted because the information was never stored; it would
have to be recomputed from memory timestamps, which is a data decision, not a migration.

This mirrors the OSS sqlite change so the enterprise backend supports the same
MEMORYLAYER_RECALL_SUPERSESSION_MODE feature. Note the feature ships OFF by default in
both: the end-to-end QA arm on LongMemEval knowledge-update found no statistically
significant improvement (see docs/DESIGN_graphiti_adoption.md in memorylayer-core-python).
This migration exists so enterprise has the OPTION, not because the feature is recommended.

downgrade() drops the index and the column.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "031"
down_revision: Union[str, None] = "030"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add contradictions.newer_memory_id and its supersession-lookup index."""
    op.add_column("contradictions", sa.Column("newer_memory_id", sa.Text(), nullable=True))
    op.create_index(
        "idx_contradictions_superseded",
        "contradictions",
        ["workspace_id", "newer_memory_id"],
        postgresql_where=sa.text("resolved_at IS NULL"),
    )


def downgrade() -> None:
    """Drop the supersession-lookup index and the newer_memory_id column."""
    op.drop_index("idx_contradictions_superseded", table_name="contradictions")
    op.drop_column("contradictions", "newer_memory_id")
