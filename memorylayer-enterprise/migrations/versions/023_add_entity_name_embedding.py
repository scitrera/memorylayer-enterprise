# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add name_embedding to entities for enterprise embedding-fuzzy resolution.

Revision ID: 023
Revises: 022
Create Date: 2026-06-03 13:00:00

The enterprise embedding-fuzzy entity-resolution tier (the first
entity-registry follow-on) resolves surface forms like ``Caroline`` and
``Caroline Chen`` to one canonical entity via an ANN over a per-entity name
embedding. This migration adds the column + index that tier needs:

- ``entities.name_embedding`` — ``vector(N)``, NULLABLE. ``N`` matches the
  deployed embed dimension (``MEMORYLAYER_EMBEDDING_DIMENSIONS``, default 1536)
  so the SAME embedding model that produces memory/fragment vectors also
  produces entity-name vectors. Nullable because (a) the OSS/SQLite backend
  never populates it (OSS resolution stays exact+alias only), and (b) the
  enterprise backend degrades gracefully to exact+alias+create when no
  embedding service is available.
- ``idx_entities_name_embedding`` — HNSW cosine index mirroring
  ``idx_memories_embedding`` / ``idx_collection_items_embedding``
  (m=16, ef_construction=64, vector_cosine_ops), partial over active, embedded
  rows only.

OSS SQLite does NOT get this column; this is an enterprise-only, pgvector-only
feature. Single alembic head: 023 -> 022.
"""
import os
from typing import Sequence, Union

from alembic import op

revision: str = "023"
down_revision: Union[str, None] = "022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Resolved at migration time from the same env var the ORM models read
# (storage/models.py ``_EMBEDDING_DIM``) so the column dim never drifts from
# the deployed embed model / the existing memory & fragment embedding columns.
_EMBEDDING_DIM = int(os.environ.get("MEMORYLAYER_EMBEDDING_DIMENSIONS", "1536"))


def upgrade() -> None:
    """Add name_embedding vector column + HNSW cosine index to entities."""
    # pgvector is already enabled by earlier migrations (memories/fragments use
    # vector columns), so no CREATE EXTENSION is needed here. Add the column via
    # raw SQL (matching the collection_items precedent in migration 011).
    op.execute(
        f"ALTER TABLE entities ADD COLUMN name_embedding vector({_EMBEDDING_DIM})"
    )
    # HNSW cosine index, partial over active + embedded rows only (mirrors the
    # idx_memories_embedding / idx_collection_items_embedding precedent).
    op.execute(
        "CREATE INDEX idx_entities_name_embedding ON entities "
        "USING hnsw (name_embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64) "
        "WHERE name_embedding IS NOT NULL AND status = 'active'"
    )


def downgrade() -> None:
    """Drop the name_embedding index + column."""
    op.execute("DROP INDEX IF EXISTS idx_entities_name_embedding")
    op.drop_column("entities", "name_embedding")
