"""Unit tests for the document blob-store orphan GC sweep.

Covers the BlobGarbageCollectionTaskHandler:
- orphan dir (no DB doc, old mtime) -> delete_tree called on the dir
- live doc dir -> NOT deleted
- live doc with a prompt_embeds/ subdir -> only that subdir delete_tree'd
- orphan dir within grace window -> skipped
- GC disabled -> get_schedule returns None

Also exercises the new BlobStorageService listing helpers (list_dir,
iter_document_dirs, newest_mtime) against a mocked fsspec filesystem.
"""
import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memorylayer_saas.services.document.blob_storage import BlobStorageService
from memorylayer_saas.tasks.blob_gc import BlobGarbageCollectionTaskHandler


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def make_mock_storage(doc=None, not_found_exc=None):
    """Storage mock; get_document returns `doc` or raises `not_found_exc`."""
    storage = AsyncMock()
    if not_found_exc is not None:
        storage.get_document.side_effect = not_found_exc
    else:
        storage.get_document.return_value = doc
    return storage


def make_mock_blob_storage(document_dirs=None, newest=None, exists=False):
    """Blob storage mock with the listing helpers used by the sweep."""
    blob = AsyncMock()
    blob.iter_document_dirs.return_value = document_dirs or []
    blob.newest_mtime.return_value = newest
    blob.exists.return_value = exists
    blob.delete_tree.return_value = None
    return blob


@contextmanager
def patch_handler(storage, blob_storage):
    """Patch get_extension / get_blob_storage_service for the GC handler."""
    from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

    def ext_side_effect(ext_name, v=None):
        return {EXT_STORAGE_BACKEND: storage}[ext_name]

    with patch("memorylayer_saas.tasks.blob_gc.get_extension", side_effect=ext_side_effect), \
         patch("memorylayer_saas.tasks.blob_gc.get_blob_storage_service",
               return_value=blob_storage), \
         patch("memorylayer_saas.tasks.blob_gc.get_logger", return_value=MagicMock()):
        yield


PAYLOAD = {"grace_seconds": 3600}


# ---------------------------------------------------------------------------
# get_task_type / get_schedule
# ---------------------------------------------------------------------------

class TestScheduleAndType:
    def test_get_task_type(self):
        assert BlobGarbageCollectionTaskHandler().get_task_type() == "blob_gc"

    def test_get_schedule_disabled_returns_none(self):
        """GC disabled -> get_schedule returns None."""
        v = MagicMock()
        v.environ.return_value = False  # MEMORYLAYER_DOCUMENT_BLOB_GC_ENABLED
        assert BlobGarbageCollectionTaskHandler().get_schedule(v) is None

    def test_get_schedule_enabled_returns_schedule(self):
        """GC enabled -> TaskSchedule with the configured interval and grace payload."""
        v = MagicMock()
        # enabled(bool), interval(int), grace(int) read in order.
        v.environ.side_effect = [True, 21600, 3600]
        schedule = BlobGarbageCollectionTaskHandler().get_schedule(v)
        assert schedule is not None
        assert schedule.interval_seconds == 21600
        assert schedule.default_payload == {"grace_seconds": 3600}


# ---------------------------------------------------------------------------
# handle() reclamation behaviour
# ---------------------------------------------------------------------------

class TestSweep:
    @pytest.mark.asyncio
    async def test_orphan_dir_old_mtime_deleted(self):
        """No DB doc + mtime older than grace -> delete_tree on the dir."""
        old = time.time() - 7200  # 2h ago, beyond 1h grace
        storage = make_mock_storage(doc=None)
        blob = make_mock_blob_storage(
            document_dirs=[("ws1", "doc1", "/blobs/ws1/documents/doc1")],
            newest=old,
        )
        with patch_handler(storage, blob):
            await BlobGarbageCollectionTaskHandler().handle(MagicMock(), PAYLOAD)

        blob.delete_tree.assert_called_once_with("/blobs/ws1/documents/doc1")

    @pytest.mark.asyncio
    async def test_orphan_dir_not_found_exception_treated_as_absent(self):
        """get_document raising FileNotFoundError counts as orphan."""
        old = time.time() - 7200
        storage = make_mock_storage(not_found_exc=FileNotFoundError("nope"))
        blob = make_mock_blob_storage(
            document_dirs=[("ws1", "doc1", "/blobs/ws1/documents/doc1")],
            newest=old,
        )
        with patch_handler(storage, blob):
            await BlobGarbageCollectionTaskHandler().handle(MagicMock(), PAYLOAD)

        blob.delete_tree.assert_called_once_with("/blobs/ws1/documents/doc1")

    @pytest.mark.asyncio
    async def test_live_doc_not_deleted(self):
        """Live doc with no obsolete subdir -> nothing deleted."""
        storage = make_mock_storage(doc=MagicMock(id="doc1"))
        blob = make_mock_blob_storage(
            document_dirs=[("ws1", "doc1", "/blobs/ws1/documents/doc1")],
            exists=False,  # no prompt_embeds/
        )
        with patch_handler(storage, blob):
            await BlobGarbageCollectionTaskHandler().handle(MagicMock(), PAYLOAD)

        blob.delete_tree.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_doc_with_prompt_embeds_only_subdir_deleted(self):
        """Live doc + prompt_embeds/ -> only that subdir delete_tree'd."""
        storage = make_mock_storage(doc=MagicMock(id="doc1"))
        blob = make_mock_blob_storage(
            document_dirs=[("ws1", "doc1", "/blobs/ws1/documents/doc1")],
            exists=True,  # prompt_embeds/ present
        )
        with patch_handler(storage, blob):
            await BlobGarbageCollectionTaskHandler().handle(MagicMock(), PAYLOAD)

        blob.delete_tree.assert_called_once_with(
            "/blobs/ws1/documents/doc1/prompt_embeds"
        )

    @pytest.mark.asyncio
    async def test_orphan_within_grace_skipped(self):
        """No DB doc but mtime within grace window -> skipped, no delete."""
        recent = time.time() - 60  # 1 min ago, inside 1h grace
        storage = make_mock_storage(doc=None)
        blob = make_mock_blob_storage(
            document_dirs=[("ws1", "doc1", "/blobs/ws1/documents/doc1")],
            newest=recent,
        )
        with patch_handler(storage, blob):
            await BlobGarbageCollectionTaskHandler().handle(MagicMock(), PAYLOAD)

        blob.delete_tree.assert_not_called()

    @pytest.mark.asyncio
    async def test_orphan_unknown_mtime_skipped(self):
        """No DB doc but undeterminable mtime -> conservatively skipped."""
        storage = make_mock_storage(doc=None)
        blob = make_mock_blob_storage(
            document_dirs=[("ws1", "doc1", "/blobs/ws1/documents/doc1")],
            newest=None,
        )
        with patch_handler(storage, blob):
            await BlobGarbageCollectionTaskHandler().handle(MagicMock(), PAYLOAD)

        blob.delete_tree.assert_not_called()

    @pytest.mark.asyncio
    async def test_per_item_error_does_not_abort_sweep(self):
        """An error on one doc dir is logged and the sweep continues."""
        old = time.time() - 7200
        storage = make_mock_storage(doc=None)
        # First dir's get_document raises an unexpected error; second is a clean orphan.
        storage.get_document.side_effect = [RuntimeError("db blip"), None]
        blob = make_mock_blob_storage(
            document_dirs=[
                ("ws1", "doc1", "/blobs/ws1/documents/doc1"),
                ("ws1", "doc2", "/blobs/ws1/documents/doc2"),
            ],
            newest=old,
        )
        with patch_handler(storage, blob):
            await BlobGarbageCollectionTaskHandler().handle(MagicMock(), PAYLOAD)

        # doc1 errored (skipped), doc2 reclaimed.
        blob.delete_tree.assert_called_once_with("/blobs/ws1/documents/doc2")


# ---------------------------------------------------------------------------
# BlobStorageService listing helpers
# ---------------------------------------------------------------------------

class TestListingHelpers:
    @pytest.fixture
    def mock_fs(self):
        fs = MagicMock()
        fs.exists.return_value = True
        return fs

    @pytest.fixture
    def service(self, mock_fs):
        return BlobStorageService(fs=mock_fs, base_path="/blobs", logger=MagicMock())

    @pytest.mark.asyncio
    async def test_list_dir_excludes_self_and_sorts(self, service, mock_fs):
        mock_fs.ls.return_value = [
            "/blobs/ws1/documents/doc1",  # the dir itself
            "/blobs/ws1/documents/doc1/pages",
            "/blobs/ws1/documents/doc1/transcripts",
        ]
        result = await service.list_dir("/blobs/ws1/documents/doc1")
        assert result == [
            "/blobs/ws1/documents/doc1/pages",
            "/blobs/ws1/documents/doc1/transcripts",
        ]

    @pytest.mark.asyncio
    async def test_list_dir_missing_returns_empty(self, service, mock_fs):
        mock_fs.exists.return_value = False
        assert await service.list_dir("/blobs/missing") == []

    @pytest.mark.asyncio
    async def test_iter_document_dirs_walks_two_levels(self, service, mock_fs):
        def ls_side_effect(path, detail=False):
            return {
                "/blobs": ["/blobs/ws1", "/blobs/ws2"],
                "/blobs/ws1/documents": ["/blobs/ws1/documents/docA"],
                "/blobs/ws2/documents": ["/blobs/ws2/documents/docB"],
            }[path]

        mock_fs.ls.side_effect = ls_side_effect
        result = await service.iter_document_dirs()
        assert ("ws1", "docA", "/blobs/ws1/documents/docA") in result
        assert ("ws2", "docB", "/blobs/ws2/documents/docB") in result
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_newest_mtime_picks_max_local(self, service, mock_fs):
        mock_fs.find.return_value = {
            "/blobs/ws1/documents/doc1/a": {"mtime": 100.0},
            "/blobs/ws1/documents/doc1/b": {"mtime": 250.5},
        }
        assert await service.newest_mtime("/blobs/ws1/documents/doc1") == 250.5

    @pytest.mark.asyncio
    async def test_newest_mtime_handles_s3_lastmodified(self, service, mock_fs):
        from datetime import datetime, timezone

        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mock_fs.find.return_value = {
            "/blobs/ws1/documents/doc1/a": {"LastModified": dt},
        }
        assert await service.newest_mtime("/blobs/ws1/documents/doc1") == dt.timestamp()

    @pytest.mark.asyncio
    async def test_newest_mtime_no_timestamp_returns_none(self, service, mock_fs):
        mock_fs.find.return_value = {
            "/blobs/ws1/documents/doc1/a": {"size": 10},
        }
        assert await service.newest_mtime("/blobs/ws1/documents/doc1") is None

    @pytest.mark.asyncio
    async def test_newest_mtime_missing_path_returns_none(self, service, mock_fs):
        mock_fs.exists.return_value = False
        assert await service.newest_mtime("/blobs/missing") is None
