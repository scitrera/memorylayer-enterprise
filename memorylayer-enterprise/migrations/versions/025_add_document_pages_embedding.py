# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add single-vector embedding column to document_pages (de-brittle the JSON stash).

Revision ID: 025
Revises: 024
Create Date: 2026-06-04 12:00:00

Replaces the brittle single-vector page embedding stash in
``document_pages.metadata['_embedding']`` with a dedicated ``embedding``
``vector(N)`` column (idempotent-ingestion design,
docs/DESIGN_idempotent_ingestion.md, item P2.4). ``N`` matches the deployed
embed dimension (``MEMORYLAYER_EMBEDDING_DIMENSIONS``, default 1536) so the SAME
embedding model that produces ``memories.embedding`` / ``fragments.embedding``
also produces page vectors — the column type MIRRORS ``MemoryModel.embedding``
exactly.

Two steps, in order:

Step A — ADD COLUMN ``document_pages.embedding`` ``vector(N)``, NULLABLE.
  Nullable because not every page carries text (image-only pages with no
  transcript never get a single-vector embedding), and because pre-migration
  rows start NULL until backfilled below.

Step B — BACKFILL from the legacy JSON stash where present.
  The embed phase previously wrote the vector as a JSON array into
  ``metadata['_embedding']``. pgvector accepts the bracketed, comma-separated
  text form (e.g. ``[0.1,0.2,0.3]``) as a vector literal, which is exactly what
  ``metadata->>'_embedding'`` renders for a JSON array of numbers, so the
  ``::vector`` cast round-trips the stored value directly. Only fill rows that
  have the key and don't already have a column value (idempotent / re-runnable).

The legacy ``metadata['_embedding']`` key is DROPPED from each backfilled row's
metadata AFTER the column is populated, so the source of truth is unambiguously
the column going forward and the meta JSON stops carrying duplicate vector blobs.
(The READ/GAP paths keep a short-lived fallback to the old key for any
not-yet-migrated environment; that fallback is removable once this migration is
universally applied.)

OSS SQLite does NOT get this column; this is an enterprise-only, pgvector-only
feature. Single alembic head: 025 -> 024.
"""
import os
from typing import Sequence, Union

from alembic import op

revision: str = "025"
down_revision: Union[str, None] = "024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Resolved at migration time from the same env var the ORM models read
# (storage/models.py ``_EMBEDDING_DIM``) so the column dim never drifts from the
# deployed embed model / the existing memory & fragment embedding columns.
_EMBEDDING_DIM = int(os.environ.get("MEMORYLAYER_EMBEDDING_DIMENSIONS", "1536"))


def upgrade() -> None:
    """Add the embedding vector column, backfill from the JSON stash, drop the key."""
    # pgvector is already enabled by earlier migrations (memories/fragments use
    # vector columns), so no CREATE EXTENSION is needed here. Add the column via
    # raw SQL (matching the entities.name_embedding / collection_items precedent).
    op.execute(
        f"ALTER TABLE document_pages ADD COLUMN embedding vector({_EMBEDDING_DIM})"
    )
    # Backfill the column from the legacy metadata['_embedding'] JSON array.
    # ``metadata->>'_embedding'`` renders the JSON array as bracketed text
    # (e.g. ``[0.1, 0.2, 0.3]``), which pgvector parses as a vector literal.
    op.execute(
        "UPDATE document_pages "
        "SET embedding = (metadata->>'_embedding')::vector "
        "WHERE metadata ? '_embedding' AND embedding IS NULL"
    )
    # Drop the now-redundant key from each backfilled row's metadata so the
    # column is the single source of truth (only touches rows that had it).
    op.execute(
        "UPDATE document_pages "
        "SET metadata = metadata - '_embedding' "
        "WHERE metadata ? '_embedding'"
    )


def downgrade() -> None:
    """Restore the JSON stash from the column, then drop the column.

    Mirrors upgrade in reverse: re-stash each non-NULL column value back under
    ``metadata['_embedding']`` (as a JSON array via ``to_jsonb``) so a downgrade
    does not lose the embeddings, then drop the column.
    """
    op.execute(
        "UPDATE document_pages "
        "SET metadata = jsonb_set("
        "  metadata, '{_embedding}', to_jsonb(embedding::text::real[])"
        ") "
        "WHERE embedding IS NOT NULL"
    )
    op.drop_column("document_pages", "embedding")
