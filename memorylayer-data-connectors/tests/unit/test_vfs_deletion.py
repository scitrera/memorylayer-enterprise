"""Deleting a VFS entry takes its bytes with it.

The catalog row and the blob are two halves of one file. Deleting only the
row left the bytes in object storage under a key that appeared in no table,
so nothing could ever find them again -- a leak on the ordinary user-facing
delete path, not just on abandoned uploads.

Safe because a blob has exactly one referent: keys are minted per entry with
a fresh uuid, the sync path's dedup skips re-registering rather than sharing
a blob, and no component outside data-connectors records a blob key at all.
"""
from __future__ import annotations

import pytest

from data_connectors.vfs.catalog import VfsCatalog
from data_connectors.vfs.deletion import delete_entry_with_blob


@pytest.fixture
def catalog():
    return VfsCatalog()


class _BlobStore:
    def __init__(self, fail=False):
        self.deleted = []
        self._fail = fail

    async def delete_object(self, key):
        if self._fail:
            raise RuntimeError("storage unavailable")
        self.deleted.append(key)


async def _entry(catalog, blob_key="ws-1/manual_upload/abc123/f.pdf"):
    return await catalog.register(
        workspace_id="ws-1",
        connector_id="manual_upload",
        source_path="f.pdf",
        content_hash="sha256:abc",
        blob_key=blob_key,
    )


@pytest.mark.asyncio
async def test_the_blob_is_deleted_with_the_entry(catalog):
    e = await _entry(catalog)
    blobs = _BlobStore()

    outcome = await delete_entry_with_blob(catalog, blobs, e.vfs_ref)

    assert blobs.deleted == ["ws-1/manual_upload/abc123/f.pdf"]
    assert outcome.entry_deleted and outcome.blob_deleted
    assert await catalog.get(e.vfs_ref) is None


@pytest.mark.asyncio
async def test_a_missing_entry_reports_not_found_and_touches_nothing(catalog):
    blobs = _BlobStore()

    outcome = await delete_entry_with_blob(catalog, blobs, "vfs_nope")

    assert outcome.found is False
    assert blobs.deleted == []


@pytest.mark.asyncio
async def test_a_failed_blob_delete_still_removes_the_entry(catalog):
    """Otherwise the file is undeletable: it stays listed and fails forever."""
    e = await _entry(catalog)
    blobs = _BlobStore(fail=True)

    outcome = await delete_entry_with_blob(catalog, blobs, e.vfs_ref)

    assert outcome.entry_deleted is True
    assert outcome.blob_failed is True
    assert outcome.blob_deleted is False
    assert await catalog.get(e.vfs_ref) is None


@pytest.mark.asyncio
async def test_an_entry_with_no_blob_deletes_cleanly(catalog):
    e = await _entry(catalog, blob_key=None)
    blobs = _BlobStore()

    outcome = await delete_entry_with_blob(catalog, blobs, e.vfs_ref)

    assert outcome.entry_deleted is True
    assert outcome.blob_deleted is False
    assert blobs.deleted == []


@pytest.mark.asyncio
async def test_no_blob_store_still_deletes_the_entry(catalog):
    """Standalone deployments run without object storage wired up."""
    e = await _entry(catalog)

    outcome = await delete_entry_with_blob(catalog, None, e.vfs_ref)

    assert outcome.entry_deleted is True
    assert await catalog.get(e.vfs_ref) is None
