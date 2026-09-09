# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Process-local, byte-bounded, TTL'd LRU cache for blob reads.

The chat read path (``build_image_embeds_content_blocks``) re-reads each page's
``image_embeds`` blob (``.pt.zst``) and grid blob (``.grid.pt``) from the
authoritative blob store (local disk / S3) on EVERY chat turn. For an ongoing
multi-turn interaction over the same document this re-downloads identical blobs
repeatedly. This module adds a volatile, in-process reuse cache so a live
interaction reuses them.

The authoritative source of truth is always the blob store: cached entries can
be evicted (by size or TTL) at any time and re-fetched transparently. The cache
stores the RAW stored bytes (the embeds blob is zstd-compressed; the compressed
form is cached and decompression stays in the caller). The cache key is the
blob path string.

This cache is PROCESS-LOCAL: it is a module-level singleton, lazily built from
config on first use, and is NOT shared across workers. That is fine for the
single-worker dev/inference path. A multi-worker / shared cache (e.g. Redis)
is a possible future option.
"""

from __future__ import annotations

import asyncio

from cachetools import TTLCache
from scitrera_app_framework import Variables

from memorylayer_saas.config import (
    DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_CACHE_MAX_GB,
    DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_CACHE_TTL_SEC,
    MEMORYLAYER_DOCUMENT_BLOB_CACHE_MAX_GB,
    MEMORYLAYER_DOCUMENT_BLOB_CACHE_TTL_SEC,
)

_GIB = 1024 * 1024 * 1024


class BlobCache:
    """Byte-bounded + time-bounded reuse cache for raw blob bytes.

    Eviction is byte-bounded via ``TTLCache(getsizeof=len)`` (the sum of cached
    ``bytes`` lengths is capped at ``maxsize`` bytes) and time-bounded via
    ``ttl``. When ``maxsize`` is 0 the cache is disabled and every retrieval
    falls through to the blob store.
    """

    def __init__(self, *, maxsize_bytes: int, ttl_sec: float) -> None:
        self._enabled = maxsize_bytes > 0
        self._cache: TTLCache | None = (
            TTLCache(maxsize=maxsize_bytes, ttl=ttl_sec, getsizeof=len)
            if self._enabled
            else None
        )
        # A single lock around miss-fill is sufficient: it only serializes the
        # store-after-fetch and the double-check, not the (awaited) fetch of
        # unrelated paths. Concurrent turns asking for the SAME path will not
        # double-fetch because the second waiter re-checks the cache under lock.
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def cached_retrieve(self, blob_storage, path: str) -> bytes:
        """Return the raw bytes for ``path``: cache hit, or fetch + store on miss.

        On a miss the blob is fetched via ``await blob_storage.retrieve_file``
        and stored. Concurrent retrievals of the same path are guarded so the
        underlying store is hit once; the fetch itself is awaited without
        holding unrelated work under the lock.
        """
        if self._cache is None:
            return await blob_storage.retrieve_file(path)

        cached = self._cache.get(path)
        if cached is not None:
            self.hits += 1
            return cached

        async with self._lock:
            # Re-check under the lock: a concurrent miss for the same path may
            # have filled it while we waited.
            cached = self._cache.get(path)
            if cached is not None:
                self.hits += 1
                return cached

            self.misses += 1
            data = await blob_storage.retrieve_file(path)
            # A single blob larger than maxsize cannot be cached (cachetools
            # raises ValueError); skip caching it but still return the bytes.
            try:
                self._cache[path] = data
            except ValueError:
                pass
            return data


# Process-local singleton, lazily built from config on first use.
_BLOB_CACHE: BlobCache | None = None


def get_blob_cache(v: Variables) -> BlobCache:
    """Return the process-local blob cache, building it from config on first use.

    Reads ``MEMORYLAYER_DOCUMENT_BLOB_CACHE_MAX_GB`` (GiB, ``0`` disables) and
    ``MEMORYLAYER_DOCUMENT_BLOB_CACHE_TTL_SEC`` from the environment via
    ``Variables`` exactly once; the resulting cache lives for the process
    lifetime.
    """
    global _BLOB_CACHE
    if _BLOB_CACHE is None:
        max_gb = float(
            v.environ(
                MEMORYLAYER_DOCUMENT_BLOB_CACHE_MAX_GB,
                default=DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_CACHE_MAX_GB,
                type_fn=float,
            )
        )
        ttl_sec = float(
            v.environ(
                MEMORYLAYER_DOCUMENT_BLOB_CACHE_TTL_SEC,
                default=DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_CACHE_TTL_SEC,
                type_fn=float,
            )
        )
        _BLOB_CACHE = BlobCache(maxsize_bytes=int(max_gb * _GIB), ttl_sec=ttl_sec)
    return _BLOB_CACHE


async def cached_retrieve(blob_storage, path: str, *, v: Variables) -> bytes:
    """Module-level convenience wrapper over the process-local cache."""
    return await get_blob_cache(v).cached_retrieve(blob_storage, path)
