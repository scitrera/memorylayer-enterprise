# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Split document knowledge phase out of ingestion status.

``documents.status`` conflates two questions that finish at very different
times: "is this document retrievable?" and "is fact extraction done?". The
first completes once pages, embeddings and composite memories are durable; the
second fans out to thousands of background tasks and can trail it by a long
way. Callers that only search or read pages should not wait on the second.

Existing rows get ``not_applicable`` rather than ``complete``: they predate the
split, nothing was recorded as scheduled for them, and asserting their
enrichment finished would be a fabrication. ``doc_verify`` re-derives the state
for documents it can still account for.

Revision ID: 034
Revises: 033
Create Date: 2026-08-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "034"
down_revision: str | None = "033"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "enrichment_status",
            sa.Text(),
            nullable=False,
            server_default="not_applicable",
        ),
    )
    # The memories scheduled for decomposition at ingest. Persisted so
    # completion is re-derived from stored state instead of counted from
    # events, which keeps it correct across retries, restarts and replays.
    op.add_column(
        "documents",
        sa.Column(
            "enrichment_memory_ids",
            sa.ARRAY(sa.Text()),
            nullable=False,
            server_default="{}",
        ),
    )
    # Partial index: the sweep only ever scans documents whose enrichment is
    # still outstanding, which is a small and shrinking slice of the table.
    op.create_index(
        "ix_documents_enrichment_pending",
        "documents",
        ["workspace_id"],
        postgresql_where=sa.text("enrichment_status = 'pending'"),
    )


def downgrade() -> None:
    op.drop_index("ix_documents_enrichment_pending", table_name="documents")
    op.drop_column("documents", "enrichment_memory_ids")
    op.drop_column("documents", "enrichment_status")
