"""Deleting a VFS entry together with the bytes it points at.

The catalog row and the blob are two halves of one file. Dropping only the
row leaves the bytes in object storage addressed by a key that no longer
appears in any table -- unreachable, unbilled-for-any-reason, and impossible
to find again without reading the storage bucket directly.

A blob key is minted per entry (``{workspace}/{connector}/{uuid}/{name}``)
and is never shared: the sync path's content-hash dedup SKIPS registering a
second entry rather than pointing one at an existing blob, and no component
outside data-connectors records a blob key at all -- MemoryLayer addresses
files by vfs_ref. So an entry is the sole referent of its blob, and the two
can be retired together.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeleteOutcome:
    """What actually happened, so callers can report it honestly.

    ``blob_failed`` is kept distinct from "no blob to delete": an orphaned
    blob is invisible in storage, so the count of them is the only signal
    that any exist.
    """
    found: bool
    entry_deleted: bool = False
    blob_deleted: bool = False
    blob_failed: bool = False


async def delete_entry_with_blob(catalog, blob_store, vfs_ref: str) -> DeleteOutcome:
    """Delete a VFS entry and the blob it points at.

    The blob is best-effort: if object storage refuses, the key is logged and
    the row is still removed. The alternative -- keeping a row whose bytes
    could not be deleted -- makes the file undeletable from the user's point
    of view and leaves it in listings, to fail again on every retry.
    """
    entry = await catalog.get(vfs_ref)
    if entry is None:
        return DeleteOutcome(found=False)

    blob_deleted = blob_failed = False
    if blob_store is not None and entry.blob_key:
        try:
            await blob_store.delete_object(entry.blob_key)
            blob_deleted = True
        except Exception:
            logger.warning(
                "Could not delete blob %s for %s; removing the catalog entry "
                "anyway (blob may be orphaned)",
                entry.blob_key, vfs_ref, exc_info=True,
            )
            blob_failed = True

    return DeleteOutcome(
        found=True,
        entry_deleted=await catalog.delete(vfs_ref),
        blob_deleted=blob_deleted,
        blob_failed=blob_failed,
    )
