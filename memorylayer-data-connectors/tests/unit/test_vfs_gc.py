"""Reclaiming uploads that were started but never completed.

An upload mints an entry and finalizes it once the bytes land. When the
client disappears in between, the entry survives describing a file that does
not exist. They are invisible to listings, so nothing surfaces them and the
table only grows.

The risk in sweeping them is deleting a file someone still has, so most of
what follows is about what the sweep must NOT touch.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from data_connectors.services.vfs_gc import (
    DEFAULT_MAX_AGE_SECONDS,
    gc_enabled,
    sweep_abandoned_uploads,
)
from data_connectors.vfs.catalog import VfsCatalog


@pytest.fixture
def catalog():
    return VfsCatalog()


class _BlobStore:
    def __init__(self, fail_on=None):
        self.deleted = []
        self._fail_on = fail_on or set()

    async def delete_object(self, key):
        if key in self._fail_on:
            raise RuntimeError(f"blob store rejected {key}")
        self.deleted.append(key)


async def _entry(catalog, *, content_hash="", age_days=30, blob_key="blobs/x/f.pdf",
                 source_path="f.pdf", ml_doc_id=None):
    e = await catalog.register(
        workspace_id="ws-1",
        connector_id="manual_upload",
        source_path=source_path,
        content_hash=content_hash,
        blob_key=blob_key,
    )
    e.created_at = datetime.now(timezone.utc) - timedelta(days=age_days)
    if ml_doc_id:
        e.ml_doc_id = ml_doc_id
    return e


# =========================================================================
# what gets collected
# =========================================================================

@pytest.mark.asyncio
async def test_an_old_unfinalized_entry_is_deleted(catalog):
    e = await _entry(catalog)

    result = await sweep_abandoned_uploads(catalog)

    assert result["deleted"] == 1
    assert await catalog.get(e.vfs_ref) is None


@pytest.mark.asyncio
async def test_its_blob_goes_too(catalog):
    """A key is assigned at mint, so bytes may exist even unfinalized.

    Deleting only the row would strand them where nothing can find them.
    """
    await _entry(catalog, blob_key="blobs/abc/report.pdf")
    blobs = _BlobStore()

    result = await sweep_abandoned_uploads(catalog, blobs)

    assert blobs.deleted == ["blobs/abc/report.pdf"]
    assert result["blobs_deleted"] == 1


# =========================================================================
# what must survive
# =========================================================================

@pytest.mark.asyncio
async def test_a_finalized_entry_is_never_touched(catalog):
    """The whole safety property: a real file is not garbage at any age."""
    e = await _entry(catalog, content_hash="sha256:abc", age_days=4000)

    result = await sweep_abandoned_uploads(catalog)

    assert result["found"] == 0
    assert await catalog.get(e.vfs_ref) is not None


@pytest.mark.asyncio
async def test_a_recent_unfinalized_entry_is_left_alone(catalog):
    """It could still be in flight — a big upload over a slow link."""
    e = await _entry(catalog, age_days=0)

    result = await sweep_abandoned_uploads(catalog)

    assert result["found"] == 0
    assert await catalog.get(e.vfs_ref) is not None


@pytest.mark.asyncio
async def test_an_entry_just_inside_the_window_survives(catalog):
    """Boundary: the cutoff is an age, so this must not be off by a day."""
    e = await _entry(catalog, age_days=(DEFAULT_MAX_AGE_SECONDS / 86400) - 0.5)

    await sweep_abandoned_uploads(catalog)

    assert await catalog.get(e.vfs_ref) is not None


@pytest.mark.asyncio
async def test_an_unfinalized_entry_with_a_document_is_left_alone(catalog):
    """Contradictory state means something unmodelled — don't destroy it."""
    e = await _entry(catalog, ml_doc_id="doc_123")

    result = await sweep_abandoned_uploads(catalog)

    assert result["found"] == 0
    assert await catalog.get(e.vfs_ref) is not None


# =========================================================================
# operational behaviour
# =========================================================================

@pytest.mark.asyncio
async def test_dry_run_deletes_nothing_but_reports_what_it_would(catalog):
    e = await _entry(catalog)
    blobs = _BlobStore()

    result = await sweep_abandoned_uploads(catalog, blobs, dry_run=True)

    assert result["found"] == 1
    assert result["deleted"] == 0
    assert blobs.deleted == []
    assert await catalog.get(e.vfs_ref) is not None


@pytest.mark.asyncio
async def test_a_failed_blob_delete_still_removes_the_row(catalog):
    """A row that cannot be retired would block the sweep behind it forever.

    The blob key is logged instead, so the orphan stays recoverable.
    """
    e = await _entry(catalog, blob_key="blobs/bad/x.pdf")
    blobs = _BlobStore(fail_on={"blobs/bad/x.pdf"})

    result = await sweep_abandoned_uploads(catalog, blobs)

    assert result["errors"] == 1
    assert result["deleted"] == 1
    assert await catalog.get(e.vfs_ref) is None


@pytest.mark.asyncio
async def test_one_bad_entry_does_not_abort_the_others(catalog):
    await _entry(catalog, blob_key="blobs/bad/x.pdf", source_path="x.pdf")
    await _entry(catalog, blob_key="blobs/ok/y.pdf", source_path="y.pdf")
    blobs = _BlobStore(fail_on={"blobs/bad/x.pdf"})

    result = await sweep_abandoned_uploads(catalog, blobs)

    assert result["deleted"] == 2


@pytest.mark.asyncio
async def test_the_batch_limit_is_respected(catalog):
    for i in range(5):
        await _entry(catalog, source_path=f"f{i}.pdf")

    result = await sweep_abandoned_uploads(catalog, limit=2)

    assert result["deleted"] == 2


@pytest.mark.asyncio
async def test_the_oldest_entries_are_collected_first(catalog):
    """A capped sweep must drain the backlog, not resample the same rows."""
    newest = await _entry(catalog, age_days=10, source_path="new.pdf")
    oldest = await _entry(catalog, age_days=900, source_path="old.pdf")

    await sweep_abandoned_uploads(catalog, limit=1)

    assert await catalog.get(oldest.vfs_ref) is None
    assert await catalog.get(newest.vfs_ref) is not None


@pytest.mark.asyncio
async def test_an_empty_catalog_is_a_clean_no_op(catalog):
    result = await sweep_abandoned_uploads(catalog)

    assert result == {"found": 0, "deleted": 0, "blobs_deleted": 0,
                      "errors": 0, "entries": []}


# =========================================================================
# multi-replica safety
# =========================================================================

@pytest.mark.asyncio
async def test_an_in_memory_catalog_sweeps_without_a_lock(catalog):
    """No database means no advisory lock -- and a single process anyway."""
    from data_connectors.services.vfs_gc import _sweep_lock

    async with _sweep_lock(catalog) as acquired:
        assert acquired is True


@pytest.mark.asyncio
async def test_only_one_replica_sweeps_when_the_lock_is_held():
    """Concurrent sweeps are safe but duplicate work and log false failures
    for blobs a peer already deleted, so a loser skips the round."""
    from data_connectors.services.vfs_gc import _sweep_lock

    class _Session:
        def __init__(self, acquired):
            self._acquired = acquired
            self.unlocked = False

        async def scalar(self, *_a, **_kw):
            return self._acquired

        async def execute(self, *_a, **_kw):
            self.unlocked = True

    class _PgCatalog:
        def __init__(self, acquired):
            self.session = _Session(acquired)

        def _session_scope(self):
            import contextlib

            @contextlib.asynccontextmanager
            async def _scope():
                yield self.session
            return _scope()

    loser = _PgCatalog(acquired=False)
    async with _sweep_lock(loser) as acquired:
        assert acquired is False
    assert loser.session.unlocked is False, "must not unlock a lock it never held"

    winner = _PgCatalog(acquired=True)
    async with _sweep_lock(winner) as acquired:
        assert acquired is True
    assert winner.session.unlocked is True, "must release the lock when done"


@pytest.mark.asyncio
async def test_a_missing_blob_store_does_not_block_collection(catalog):
    """Standalone/in-memory deployments have no blob store wired up."""
    e = await _entry(catalog)

    result = await sweep_abandoned_uploads(catalog, None)

    assert result["deleted"] == 1
    assert await catalog.get(e.vfs_ref) is None


def test_gc_is_on_by_default(monkeypatch):
    monkeypatch.delenv("DC_UPLOAD_GC_ENABLED", raising=False)
    assert gc_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no"])
def test_gc_can_be_switched_off(monkeypatch, value):
    monkeypatch.setenv("DC_UPLOAD_GC_ENABLED", value)
    assert gc_enabled() is False
