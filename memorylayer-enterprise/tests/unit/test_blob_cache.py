"""Unit tests for the process-local image-embed/grid blob reuse cache."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import memorylayer_saas.services.document.blob_cache as blob_cache_mod
from memorylayer_saas.services.document.blob_cache import (
    BlobCache,
    cached_retrieve,
    get_blob_cache,
)


def _blob_storage(payload=b"blob-bytes"):
    blob = MagicMock()
    blob.retrieve_file = AsyncMock(return_value=payload)
    return blob


@pytest.fixture
def reset_singleton():
    """Reset the module-level cache singleton around each test."""
    blob_cache_mod._BLOB_CACHE = None
    yield
    blob_cache_mod._BLOB_CACHE = None


@pytest.mark.asyncio
async def test_same_path_fetched_once():
    cache = BlobCache(maxsize_bytes=1024 * 1024, ttl_sec=3600)
    blob = _blob_storage()

    a = await cache.cached_retrieve(blob, "/blobs/p0.pt.zst")
    b = await cache.cached_retrieve(blob, "/blobs/p0.pt.zst")

    assert a == b == b"blob-bytes"
    blob.retrieve_file.assert_awaited_once()
    assert cache.hits == 1 and cache.misses == 1


@pytest.mark.asyncio
async def test_different_paths_both_fetched():
    cache = BlobCache(maxsize_bytes=1024 * 1024, ttl_sec=3600)
    blob = _blob_storage()

    await cache.cached_retrieve(blob, "/blobs/p0.pt.zst")
    await cache.cached_retrieve(blob, "/blobs/p0.grid.pt")

    assert blob.retrieve_file.await_count == 2
    assert cache.misses == 2 and cache.hits == 0


@pytest.mark.asyncio
async def test_disabled_when_max_gb_zero():
    cache = BlobCache(maxsize_bytes=0, ttl_sec=3600)
    blob = _blob_storage()

    assert cache.enabled is False
    await cache.cached_retrieve(blob, "/blobs/p0.pt.zst")
    await cache.cached_retrieve(blob, "/blobs/p0.pt.zst")

    # Disabled cache always calls through.
    assert blob.retrieve_file.await_count == 2
    assert cache.hits == 0 and cache.misses == 0


@pytest.mark.asyncio
async def test_size_eviction():
    """Byte-bounded eviction: a maxsize smaller than two entries evicts the LRU."""
    # Each payload is 10 bytes; maxsize 15 holds at most one entry.
    cache = BlobCache(maxsize_bytes=15, ttl_sec=3600)
    blob = MagicMock()
    blob.retrieve_file = AsyncMock(side_effect=lambda path: b"0123456789")

    await cache.cached_retrieve(blob, "/a")  # miss -> store (size 10)
    await cache.cached_retrieve(blob, "/b")  # miss -> store evicts /a (10+10 > 15)
    await cache.cached_retrieve(blob, "/a")  # miss again: /a was evicted

    assert blob.retrieve_file.await_count == 3
    assert cache.misses == 3 and cache.hits == 0


@pytest.mark.asyncio
async def test_ttl_eviction():
    """Time-bounded eviction: after the TTL elapses the entry is re-fetched."""
    cache = BlobCache(maxsize_bytes=1024 * 1024, ttl_sec=0.05)
    blob = _blob_storage()

    await cache.cached_retrieve(blob, "/a")  # miss -> store
    await cache.cached_retrieve(blob, "/a")  # hit
    assert blob.retrieve_file.await_count == 1

    await asyncio.sleep(0.08)  # let the TTL elapse
    await cache.cached_retrieve(blob, "/a")  # expired -> miss -> re-fetch
    assert blob.retrieve_file.await_count == 2


@pytest.mark.asyncio
async def test_oversized_blob_not_cached_but_returned():
    """A single blob larger than maxsize is returned but not cached."""
    cache = BlobCache(maxsize_bytes=5, ttl_sec=3600)
    blob = MagicMock()
    blob.retrieve_file = AsyncMock(side_effect=lambda path: b"too-large-payload")

    data1 = await cache.cached_retrieve(blob, "/a")
    data2 = await cache.cached_retrieve(blob, "/a")

    assert data1 == data2 == b"too-large-payload"
    # Not cached (exceeds maxsize), so fetched both times.
    assert blob.retrieve_file.await_count == 2


@pytest.mark.asyncio
async def test_concurrent_same_path_single_fetch():
    """Concurrent retrievals of the same path hit the underlying store once."""
    cache = BlobCache(maxsize_bytes=1024 * 1024, ttl_sec=3600)
    calls = 0

    async def _retrieve(path):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)  # widen the race window
        return b"payload"

    blob = MagicMock()
    blob.retrieve_file = AsyncMock(side_effect=_retrieve)

    results = await asyncio.gather(
        *[cache.cached_retrieve(blob, "/same") for _ in range(8)]
    )
    assert all(r == b"payload" for r in results)
    assert calls == 1


@pytest.mark.asyncio
async def test_get_blob_cache_lazy_from_config(reset_singleton):
    from scitrera_app_framework import Variables

    v = Variables()
    v.set(blob_cache_mod.MEMORYLAYER_DOCUMENT_BLOB_CACHE_MAX_GB, "1")
    v.set(blob_cache_mod.MEMORYLAYER_DOCUMENT_BLOB_CACHE_TTL_SEC, "120")

    cache = get_blob_cache(v)
    assert cache.enabled is True
    # Same singleton on subsequent calls.
    assert get_blob_cache(v) is cache


@pytest.mark.asyncio
async def test_module_cached_retrieve_disabled(reset_singleton):
    from scitrera_app_framework import Variables

    v = Variables()
    v.set(blob_cache_mod.MEMORYLAYER_DOCUMENT_BLOB_CACHE_MAX_GB, "0")
    blob = _blob_storage()

    await cached_retrieve(blob, "/a", v=v)
    await cached_retrieve(blob, "/a", v=v)

    assert blob.retrieve_file.await_count == 2
