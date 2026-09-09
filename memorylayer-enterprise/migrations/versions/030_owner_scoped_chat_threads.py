"""Replace per-user thread-id materialization with owner-scoped storage.

Revision ID: 030
Revises: 029
Create Date: 2026-07-26 00:00:00

BACKGROUND
----------
Per-user isolation for user-owned chat threads (the ``_user_chat`` sentinel
workspace) was implemented by ENCODING the owner into the thread id:
``<client_id>::u::<sha256(user)[:16]>``, with a materialize/de-materialize
translation wired ONLY into the REST response layer (``resolve_thread_id`` /
``_present_thread`` / ``client_thread_id``). Any channel that carried a thread id
back to the client through a DIFFERENT path (the harness-emitted ``rename``
control, the sidebar thread list, ...) bypassed that translation and leaked the
raw ``::u::`` id, which the frontend — that only knows the short client id —
could not match. It also lengthened ids that end up in URLs.

NEW MODEL
---------
The owner is a SCOPING COLUMN, not part of the id. Thread identity becomes
``(workspace_id, user_id, id)`` with ``id`` stored VERBATIM (the client id, e.g.
``_default`` / ``thread_<uuid>``). A new opaque surrogate ``row_id`` is the
primary key and the ONLY value ``chat_messages.thread_id`` references — it never
appears in any API response or event, so there is no mangled id to leak. Every
client-facing surface (REST, events, harness, frontend, URLs) uses the short
``id`` uniformly.

WHAT THIS DOES (all data + schema, single revision)
---------------------------------------------------
1. Add surrogate ``row_id`` (text uuid) and backfill.
2. DELETE the migration-029 quarantined comingled rows (they are known-leaked
   incident data; approved for deletion). CASCADE removes their messages.
3. Repoint ``chat_messages.thread_id`` from the thread ``id`` to ``row_id``.
4. Un-suffix ``chat_threads.id`` (and ``parent_thread``): strip the trailing
   ``::u::<hex16>`` so the stored id becomes the client id.
5. Collision merge: un-suffixing can make a materialized row collapse onto an
   "escaped" non-materialized shell of the SAME (workspace, user, client id)
   — e.g. an empty ``create_thread`` shell + the ``append`` thread that actually
   holds the messages. Per group keep the best row (most messages, then most
   recently updated), repoint the losers' messages onto it, re-sequence the
   merged message_index, and delete the loser rows.
6. Swap constraints: drop PK(id) + UNIQUE(workspace_id, id); PK becomes row_id;
   add UNIQUE(workspace_id, COALESCE(user_id,''), id); FK
   chat_messages.thread_id -> chat_threads.row_id ON DELETE CASCADE.

downgrade() restores the id-as-PK schema shape (best effort). It CANNOT restore
the deleted quarantined rows or re-encode the ``::u::`` suffix (the per-user hash
is one-way), so it raises to avoid a false sense of reversibility.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "030"
down_revision: Union[str, None] = "029"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The materialized suffix: ``::u::`` + a 16-char lowercase hex digest, always
# trailing (mirrors migration 029's _MATERIALIZED_ID_REGEX).
_SUFFIX_REGEX = r"::u::[0-9a-f]{16}$"


def upgrade() -> None:
    # 1. Surrogate PK column (text uuid; keeps chat_messages.thread_id a text FK
    #    with no type churn). gen_random_uuid() is core in PG13+.
    op.execute("ALTER TABLE chat_threads ADD COLUMN IF NOT EXISTS row_id TEXT")
    op.execute("UPDATE chat_threads SET row_id = gen_random_uuid()::text WHERE row_id IS NULL")

    # 2. Delete the 029-quarantined comingled rows (approved). The existing
    #    chat_messages FK is ON DELETE CASCADE, so their messages go with them.
    op.execute(
        "DELETE FROM chat_threads WHERE (metadata->>'quarantine_reason') IS NOT NULL"
    )

    # 3. Repoint messages to the surrogate BEFORE we mutate id (ids are still
    #    materialized + unique here, so the join on the old id is exact). Drop the
    #    old FK first so the repoint isn't gated by it.
    op.execute(
        "ALTER TABLE chat_messages DROP CONSTRAINT IF EXISTS chat_messages_thread_id_fkey"
    )
    op.execute(
        """
        UPDATE chat_messages m
        SET thread_id = t.row_id
        FROM chat_threads t
        WHERE m.thread_id = t.id
        """
    )

    # 4. Compute the client id (un-suffixed) into a scratch column, and un-suffix
    #    parent_thread (a client-id reference, not a FK) in place.
    op.execute("ALTER TABLE chat_threads ADD COLUMN IF NOT EXISTS _client_id TEXT")
    op.execute(
        f"UPDATE chat_threads SET _client_id = regexp_replace(id, '{_SUFFIX_REGEX}', '')"
    )
    op.execute(
        f"""
        UPDATE chat_threads
        SET parent_thread = regexp_replace(parent_thread, '{_SUFFIX_REGEX}', '')
        WHERE parent_thread IS NOT NULL
        """
    )

    # 5. Collision merge within (workspace_id, COALESCE(user_id,''), _client_id).
    #    Keeper = most messages, then most recently updated, then created, then
    #    row_id (deterministic tiebreak).
    op.execute(
        """
        CREATE TEMP TABLE _thread_keep ON COMMIT DROP AS
        SELECT
            row_id,
            first_value(row_id) OVER (
                PARTITION BY workspace_id, COALESCE(user_id, ''), _client_id
                ORDER BY message_count DESC,
                         updated_at DESC NULLS LAST,
                         created_at DESC,
                         row_id
            ) AS keep_row_id
        FROM chat_threads
        """
    )
    # Repoint loser messages onto the keeper.
    op.execute(
        """
        UPDATE chat_messages m
        SET thread_id = k.keep_row_id
        FROM _thread_keep k
        WHERE m.thread_id = k.row_id
          AND k.row_id <> k.keep_row_id
        """
    )
    # Delete the loser threads.
    op.execute(
        """
        DELETE FROM chat_threads t
        USING _thread_keep k
        WHERE t.row_id = k.row_id
          AND k.row_id <> k.keep_row_id
        """
    )
    # Re-sequence message_index per (now-merged) thread so a merge can't leave
    # duplicate indices, and recompute message_count from reality.
    op.execute(
        """
        WITH seq AS (
            SELECT id,
                   row_number() OVER (
                       PARTITION BY thread_id ORDER BY message_index, created_at, id
                   ) - 1 AS new_idx
            FROM chat_messages
        )
        UPDATE chat_messages m
        SET message_index = seq.new_idx
        FROM seq
        WHERE m.id = seq.id
          AND m.message_index <> seq.new_idx
        """
    )
    op.execute(
        """
        UPDATE chat_threads t
        SET message_count = COALESCE(c.cnt, 0)
        FROM (
            SELECT thread_id, count(*) AS cnt FROM chat_messages GROUP BY thread_id
        ) c
        WHERE t.row_id = c.thread_id
        """
    )
    op.execute(
        """
        UPDATE chat_threads t
        SET message_count = 0
        WHERE NOT EXISTS (SELECT 1 FROM chat_messages m WHERE m.thread_id = t.row_id)
        """
    )

    # 6. Swap the PK onto the surrogate BEFORE rewriting `id`.
    #
    #    ORDER IS LOAD-BEARING — this is what the first cut of 030 got wrong.
    #    The OLD pkey is on `id` ALONE (global), while the step-5 merge dedupes
    #    within the OWNER-SCOPED partition (workspace_id, user_id, _client_id),
    #    so two rows with the SAME client id but DIFFERENT owners survive on
    #    purpose — that is the entire point of owner scoping. Rewriting `id` to
    #    the un-suffixed client id while the global pkey is still in place makes
    #    exactly those rows collide:
    #
    #        duplicate key value violates unique constraint "chat_threads_pkey"
    #        DETAIL:  Key (id)=(_default) already exists.
    #
    #    i.e. the migration failed on the very collision it exists to make legal.
    #    Transactional DDL rolled it back, leaving alembic at 029 while the new
    #    ORM (expecting row_id) was already deployed.
    op.execute("ALTER TABLE chat_threads DROP CONSTRAINT IF EXISTS uq_chat_threads_workspace_id")
    op.execute("ALTER TABLE chat_threads DROP CONSTRAINT IF EXISTS chat_threads_pkey")
    op.execute("ALTER TABLE chat_threads ALTER COLUMN row_id SET NOT NULL")
    op.execute("ALTER TABLE chat_threads ADD CONSTRAINT chat_threads_pkey PRIMARY KEY (row_id)")

    # 7. Stored id becomes the client id. Safe now: `id` is no longer unique on
    #    its own, and the owner-scoped unique index is not created until step 8
    #    (after the values are final).
    op.execute("UPDATE chat_threads SET id = _client_id")
    op.execute("ALTER TABLE chat_threads DROP COLUMN _client_id")
    # 8. Owner-scoped identity. COALESCE(user_id,'') so workspace-owned threads
    # (user_id NULL) are scoped by (workspace_id, id) and never collide with a
    # user-owned thread of the same client id.
    op.execute(
        """
        CREATE UNIQUE INDEX uq_chat_threads_ws_user_id
        ON chat_threads (workspace_id, COALESCE(user_id, ''), id)
        """
    )
    # Re-establish the message FK against the surrogate.
    op.execute(
        """
        ALTER TABLE chat_messages
        ADD CONSTRAINT chat_messages_thread_id_fkey
        FOREIGN KEY (thread_id) REFERENCES chat_threads (row_id) ON DELETE CASCADE
        """
    )


def downgrade() -> None:
    # Irreversible: the ``::u::`` suffix is a one-way hash of the user and the
    # quarantined rows were deleted. Refuse rather than pretend.
    raise NotImplementedError(
        "030 is not reversible: it deletes quarantined rows and drops the "
        "one-way per-user id suffix. Restore from a backup to roll back."
    )
