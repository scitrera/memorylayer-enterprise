"""In-memory enterprise cache service.

A single-process :class:`EnterpriseCacheService` backed by an LRU map with
optional per-key TTL, plus owner-verified in-process distributed-lock
emulation.  It mirrors the OSS ``LRUCacheService`` (see
``memorylayer_server.services.cache.lru``) for the cache surface and the
``coord.MemoryLocker`` semantics (Aether Go SDK) for the lock surface, so it is
a drop-in wherever code expects an ``EnterpriseCacheService`` -- primarily tests
and single-process deployments where a shared external backend is unnecessary.

The lock is genuine *within this process* only: ``acquire_lock`` succeeds iff
the key is absent or its lease has expired, and ``release_lock`` deletes iff the
caller is the current holder.  It provides no cross-node coordination.
"""
from __future__ import annotations

import time
from logging import Logger
from typing import Any, Optional

from scitrera_app_framework import Variables, get_logger

from .base import EnterpriseCacheService, CacheServicePluginBase

# Environment variable constants (specific to this implementation)
MEMORYLAYER_CACHE_INMEMORY_MAXSIZE = "MEMORYLAYER_CACHE_INMEMORY_MAXSIZE"
DEFAULT_MEMORYLAYER_CACHE_INMEMORY_MAXSIZE = 4096


class InMemoryCacheService(EnterpriseCacheService):
    """In-memory LRU cache with TTL and in-process distributed-lock emulation.

    Uses ``cachetools.LRUCache`` for O(1) bounded lookups; TTL is tracked
    alongside each entry as ``(monotonic_timestamp, ttl_seconds)``.  Locks are
    held in a separate map as ``(holder_id, expiry_monotonic)``.
    """

    def __init__(
        self,
        v: Variables = None,
        logger: Logger = None,
        maxsize: int = DEFAULT_MEMORYLAYER_CACHE_INMEMORY_MAXSIZE,
    ):
        from cachetools import LRUCache

        self._v = v
        self.logger = logger or get_logger(v, name=self.__class__.__name__)
        self._cache: LRUCache = LRUCache(maxsize=maxsize)
        self._timestamps: dict[str, tuple[float, Optional[int]]] = {}
        self._locks: dict[str, tuple[str, Optional[float]]] = {}
        self._maxsize = maxsize
        self.logger.info("Initialized InMemoryCacheService with maxsize=%s", maxsize)

    # ------------------------------------------------------------------
    # TTL helpers
    # ------------------------------------------------------------------
    def _is_expired(self, key: str) -> bool:
        """Return True if the entry is missing or past its TTL."""
        if key not in self._timestamps:
            return True
        timestamp, ttl_seconds = self._timestamps[key]
        if ttl_seconds is None:
            return False
        return (time.monotonic() - timestamp) > ttl_seconds

    # ------------------------------------------------------------------
    # CacheService interface
    # ------------------------------------------------------------------
    async def get(self, key: str) -> Any | None:
        """Get value from cache, evicting it if expired."""
        if key not in self._cache:
            return None
        if self._is_expired(key):
            await self.delete(key)
            return None
        return self._cache.get(key)

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        """Set value in cache with optional TTL."""
        self._cache[key] = value
        self._timestamps[key] = (time.monotonic(), ttl_seconds)
        self.logger.debug("Cache set: key=%s, ttl=%s", key, ttl_seconds)
        return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            self._timestamps.pop(key, None)
            self.logger.debug("Cache delete: key=%s", key)
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache and is not expired."""
        if key not in self._cache:
            return False
        if self._is_expired(key):
            await self.delete(key)
            return False
        return True

    async def clear_prefix(self, prefix: str) -> int:
        """Clear all keys with the given prefix. Returns number deleted."""
        keys_to_delete = [k for k in list(self._cache.keys()) if k.startswith(prefix)]
        for key in keys_to_delete:
            del self._cache[key]
            self._timestamps.pop(key, None)
        if keys_to_delete:
            self.logger.debug("Cache clear_prefix: prefix=%s, deleted=%s", prefix, len(keys_to_delete))
        return len(keys_to_delete)

    # ------------------------------------------------------------------
    # EnterpriseCacheService interface (in-process lock emulation)
    # ------------------------------------------------------------------
    def _lock_live_holder(self, lock_key: str) -> Optional[str]:
        """Return the current holder of ``lock_key``, expiring it if past TTL."""
        entry = self._locks.get(lock_key)
        if entry is None:
            return None
        holder, expiry = entry
        if expiry is not None and time.monotonic() > expiry:
            self._locks.pop(lock_key, None)
            return None
        return holder

    async def acquire_lock(self, lock_key: str, holder_id: str, ttl: int = 30) -> bool:
        """Acquire the lock iff it is currently unheld (or its lease expired)."""
        if self._lock_live_holder(lock_key) is not None:
            return False
        expiry = time.monotonic() + ttl if ttl and ttl > 0 else None
        self._locks[lock_key] = (holder_id, expiry)
        return True

    async def release_lock(self, lock_key: str, holder_id: str) -> bool:
        """Release the lock iff it is held by ``holder_id``."""
        if self._lock_live_holder(lock_key) != holder_id:
            return False
        self._locks.pop(lock_key, None)
        return True


class InMemoryCacheServicePlugin(CacheServicePluginBase):
    """Plugin for the in-memory enterprise cache service."""

    PROVIDER_NAME = "in-memory"

    def initialize(self, v: Variables, logger: Logger) -> InMemoryCacheService:
        maxsize = v.environ(
            MEMORYLAYER_CACHE_INMEMORY_MAXSIZE,
            default=DEFAULT_MEMORYLAYER_CACHE_INMEMORY_MAXSIZE,
            type_fn=int,
        )
        return InMemoryCacheService(v=v, logger=logger, maxsize=maxsize)
