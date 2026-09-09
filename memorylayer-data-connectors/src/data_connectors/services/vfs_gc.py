# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Garbage collection for uploads that were started but never completed.

An upload mints a VFS entry first and finalizes it once the bytes land. When
the client goes away in between — tab closed, network dropped, request
failed — the entry stays behind describing a file that does not exist, and
nothing will ever complete it. They accumulate silently: they are invisible
to listings, so the only symptom is a table that grows and never shrinks.

This sweeps them on a timer. It is deliberately conservative: it only ever
touches entries with no content hash, and only long after any real upload
could still be in flight.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from data_connectors.vfs.deletion import delete_entry_with_blob

logger = logging.getLogger(__name__)

# A sweep must never race a slow upload. The window is measured in DAYS
# rather than minutes for that reason: an entry hidden from listings after
# half an hour is merely invisible and still completable, whereas one
# deleted here is gone. Nothing legitimate stays unfinalized this long.
DEFAULT_MAX_AGE_SECONDS = 7 * 24 * 3600
DEFAULT_INTERVAL_SECONDS = 6 * 3600
DEFAULT_BATCH_LIMIT = 500


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r; using %d", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Ignoring non-positive %s=%d; using %d", name, value, default)
        return default
    return value


def gc_enabled() -> bool:
    return os.environ.get("DC_UPLOAD_GC_ENABLED", "1").lower() not in ("0", "false", "no")


# Arbitrary constant identifying this sweep's Postgres advisory lock. Any
# value works as long as it is stable and not shared with another lock.
_SWEEP_LOCK_ID = 0x4643475643  # "DCGVC"


@asynccontextmanager
async def _sweep_lock(catalog):
    """Hold a cluster-wide lock for the sweep, if the catalog is Postgres-backed.

    Concurrent sweeps are SAFE -- delete is idempotent and a second deleter
    just finds the row gone -- but they duplicate work and log spurious
    blob-delete failures for blobs a peer already removed. This keeps exactly
    one sweeper across any number of replicas.

    A Postgres advisory lock rather than a leader election or a queue: it is
    held on the session, so it releases automatically if the pod dies, and it
    adds no component that can itself fail and silently stop collection. A
    janitor whose whole job is to stop things accumulating must not be the
    thing that quietly stops running.

    Yields True when the sweep should proceed. In-memory catalogs (no
    database, single process by definition) always proceed.
    """
    session_scope = getattr(catalog, "_session_scope", None)
    if session_scope is None:
        yield True
        return

    from sqlalchemy import text

    async with session_scope() as session:
        acquired = await session.scalar(
            text("SELECT pg_try_advisory_lock(:k)"), {"k": _SWEEP_LOCK_ID},
        )
        try:
            yield bool(acquired)
        finally:
            if acquired:
                await session.execute(
                    text("SELECT pg_advisory_unlock(:k)"), {"k": _SWEEP_LOCK_ID},
                )


async def sweep_abandoned_uploads(catalog, blob_store=None, *,
                                  max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
                                  limit: int = DEFAULT_BATCH_LIMIT,
                                  dry_run: bool = False) -> dict:
    """Delete catalog entries for uploads that never completed.

    Returns a summary dict: ``{"found", "deleted", "blobs_deleted",
    "errors", "entries"}``. With ``dry_run`` nothing is deleted and the
    summary reports what would have been.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
    entries = await catalog.find_abandoned_uploads(cutoff, limit=limit)
    if not entries:
        return {"found": 0, "deleted": 0, "blobs_deleted": 0, "errors": 0, "entries": []}

    sample = [
        {"vfs_ref": e.vfs_ref, "workspace_id": e.workspace_id,
         "source_path": e.source_path, "created_at": e.created_at.isoformat()}
        for e in entries[:50]
    ]

    if dry_run:
        logger.info("Abandoned-upload GC (dry run): would delete %d entries older than %ds",
                    len(entries), max_age_seconds)
        return {"found": len(entries), "deleted": 0, "blobs_deleted": 0,
                "errors": 0, "dry_run": True, "entries": sample}

    deleted = blobs_deleted = errors = 0
    for entry in entries:
        # A key is assigned at mint, so bytes MAY have been written even
        # though the upload never finalized. Shares the normal delete path so
        # an abandoned upload is reclaimed exactly like a user-deleted file.
        try:
            outcome = await delete_entry_with_blob(catalog, blob_store, entry.vfs_ref)
            deleted += int(outcome.entry_deleted)
            blobs_deleted += int(outcome.blob_deleted)
            errors += int(outcome.blob_failed)
        except Exception:
            logger.warning("Abandoned-upload GC failed to delete entry %s",
                           entry.vfs_ref, exc_info=True)
            errors += 1

    logger.info(
        "Abandoned-upload GC: found=%d deleted=%d blobs_deleted=%d errors=%d "
        "(older than %ds)",
        len(entries), deleted, blobs_deleted, errors, max_age_seconds,
    )
    return {"found": len(entries), "deleted": deleted,
            "blobs_deleted": blobs_deleted, "errors": errors, "entries": sample}


async def run_gc_loop(catalog, blob_store=None) -> None:
    """Sweep on an interval until cancelled.

    Never raises: a failed sweep logs and waits for the next tick, because
    this runs for the lifetime of the process and one bad round (a database
    blip, say) must not silently end all future collection.
    """
    interval = _env_int("DC_UPLOAD_GC_INTERVAL_SECONDS", DEFAULT_INTERVAL_SECONDS)
    max_age = _env_int("DC_UPLOAD_GC_MAX_AGE_SECONDS", DEFAULT_MAX_AGE_SECONDS)
    limit = _env_int("DC_UPLOAD_GC_BATCH_LIMIT", DEFAULT_BATCH_LIMIT)

    logger.info(
        "Abandoned-upload GC started (interval=%ds, max_age=%ds, batch=%d)",
        interval, max_age, limit,
    )
    while True:
        # Wait FIRST so the sweep never competes with startup, and so a
        # crash-looping pod cannot issue a burst of deletes on each boot.
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Abandoned-upload GC stopped")
            raise

        try:
            async with _sweep_lock(catalog) as acquired:
                if not acquired:
                    logger.debug("Another replica holds the GC lock; skipping this round")
                    continue
                await sweep_abandoned_uploads(
                    catalog, blob_store, max_age_seconds=max_age, limit=limit,
                )
        except asyncio.CancelledError:
            logger.info("Abandoned-upload GC stopped")
            raise
        except Exception:
            logger.warning("Abandoned-upload GC sweep failed; will retry", exc_info=True)
