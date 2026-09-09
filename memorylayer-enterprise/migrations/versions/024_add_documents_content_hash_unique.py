# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Add UNIQUE(workspace_id, content_hash) to documents (gated on dedup cleanup).

Revision ID: 024
Revises: 023
Create Date: 2026-06-04 00:00:00

DB-enforces the "same bytes -> one document" LINK decision from the idempotent
-ingestion design (docs/DESIGN_idempotent_ingestion.md, decision #2 "LINK",
item P2.1). The running ``doc_added`` LINK logic already routes NEW arrivals of
identical bytes to the existing document; this migration adds the constraint
that makes that invariant a database guarantee and back-stops ``find_document_
by_hash`` (which was hardened to ``.first()`` in Phase 1 precisely because no
unique constraint existed yet).

Two steps, in order:

Step A — duplicate-hash CLEANUP (gated precondition).
  The constraint cannot be created while duplicate ``(workspace_id,
  content_hash)`` rows exist, so this migration first dedups them.

  Keeper selection per duplicate group: prefer ``status = 'completed'``, then
  earliest ``created_at``, tie-broken by ``id`` (deterministic). The keeper is
  the canonical document the LINK decision would have everyone share.

  Cleanup POLICY = HARD DELETE of the non-keeper duplicate rows and their
  dependent ``document_pages`` + ``memories``. We hard-delete (rather than
  soft-delete) because:
    * ``DocumentStatus`` has NO soft-delete state (no DELETED/ARCHIVED) and
      ``documents`` has NO ``deleted_at`` column — there is no less-destructive
      in-schema way to retire a duplicate document row.
    * Duplicates are BYTE-IDENTICAL to the keeper, so the keeper already carries
      an equivalent set of pages/memories; the duplicates' derived artifacts are
      redundant.
  This IS destructive, but bounded to provably-redundant rows. It is also
  idempotent: re-running finds no duplicate groups and deletes nothing.

  Dependency-order detail (why we delete memories explicitly):
    * ``document_pages.document_id`` FK is ``ON DELETE CASCADE`` -> a duplicate
      document's pages are removed automatically when the document is deleted.
    * ``memories.source_document_id`` / ``source_page_id`` FKs are ``ON DELETE
      SET NULL`` -> deleting a duplicate document/page would merely NULL the
      provenance on its (redundant) memory rows and leave them behind. So we
      DELETE those memories explicitly BEFORE deleting the document, keying on
      ``source_document_id = <dup id>``. (Memories do have a ``deleted_at``
      soft-delete column, but these rows are redundant copies of the keeper's
      memories, so per the design we hard-delete them.)

  KNOWN FOLLOW-UP / CAVEAT (cannot be fixed here): a duplicate document's
  ``source_vfs_ref`` points at a data-connectors VFS entry that, after this
  dedup, references a now-removed document. This migration runs only against the
  MemoryLayer relational store and CANNOT re-link the data-connectors side. The
  running LINK logic in ``doc_added`` handles NEW arrivals going forward; this
  cleanup is strictly about PRE-EXISTING duplicates, and any dangling VFS->doc
  reference from a removed duplicate must be reconciled on the data-connectors
  side as a separate follow-up.

Step B — create the unique constraint and drop the now-redundant index.
  ``uq_documents_workspace_content_hash`` on ``(workspace_id, content_hash)``.
  The pre-existing non-unique ``idx_documents_content_hash`` had IDENTICAL
  leading columns ``(workspace_id, content_hash)``, so the constraint's implicit
  index fully supersedes it; we DROP it to avoid a redundant duplicate index.
  (The ORM model in ``storage/models.py`` was updated to match: the constraint
  is present and the old Index() entry removed.)

downgrade() drops the constraint and restores the non-unique index. The Step A
data cleanup is NOT reversible — deleted duplicate documents/pages/memories are
gone; downgrade only reverts the schema, not the data.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "024"
down_revision: Union[str, None] = "023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Ranks each row within its (workspace_id, content_hash) group; rank 1 is the
# keeper. Keeper preference: status='completed' first, then earliest created_at,
# tie-broken by id for determinism.
_RANK_DUPLICATES_CTE = """
WITH ranked AS (
    SELECT
        id,
        workspace_id,
        content_hash,
        ROW_NUMBER() OVER (
            PARTITION BY workspace_id, content_hash
            ORDER BY
                (status = 'completed') DESC,
                created_at ASC,
                id ASC
        ) AS rn
    FROM documents
),
dups AS (
    SELECT id FROM ranked WHERE rn > 1
)
"""


def upgrade() -> None:
    """Step A: dedup duplicate-hash rows. Step B: add unique constraint."""
    # --- Step A: cleanup (idempotent) -------------------------------------
    # Delete the redundant memories of every non-keeper duplicate FIRST
    # (FK is ON DELETE SET NULL, so this won't cascade from the doc delete).
    op.execute(
        _RANK_DUPLICATES_CTE
        + """
        DELETE FROM memories
        WHERE source_document_id IN (SELECT id FROM dups)
        """
    )
    # Delete the non-keeper duplicate documents. document_pages cascade via
    # the ON DELETE CASCADE FK on document_pages.document_id.
    op.execute(
        _RANK_DUPLICATES_CTE
        + """
        DELETE FROM documents
        WHERE id IN (SELECT id FROM dups)
        """
    )

    # --- Step B: enforce + tidy -------------------------------------------
    op.create_unique_constraint(
        "uq_documents_workspace_content_hash",
        "documents",
        ["workspace_id", "content_hash"],
    )
    # Redundant now: identical leading columns to the constraint's index.
    op.execute("DROP INDEX IF EXISTS idx_documents_content_hash")


def downgrade() -> None:
    """Drop the unique constraint and restore the non-unique index.

    The Step A data cleanup is NOT reversible; this only reverts the schema.
    """
    op.drop_constraint(
        "uq_documents_workspace_content_hash",
        "documents",
        type_="unique",
    )
    op.create_index(
        "idx_documents_content_hash",
        "documents",
        ["workspace_id", "content_hash"],
    )
