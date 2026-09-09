# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Enterprise cache service plugin base."""
from abc import abstractmethod

# Import from OSS
from memorylayer_server.services.cache.base import (
    CacheServicePluginBase,
    EXT_CACHE_SERVICE,
    CacheService
)

# Enterprise Default Cache Service.
# 'aether-kv' uses the shared Aether KV store (no separate Redis needed);
# 'redis' and 'in-memory' remain selectable via MEMORYLAYER_CACHE_SERVICE.
DEFAULT_MEMORYLAYER_CACHE_SERVICE = 'aether-kv'


class EnterpriseCacheService(CacheService):
    """Extended cache service with enterprise features like distributed locking.

    Implementations should extend this class to provide distributed locking
    capabilities. Code can check `isinstance(cache, EnterpriseCacheService)`
    to determine if locking is available.
    """

    @abstractmethod
    async def acquire_lock(self, lock_key: str, holder_id: str, ttl: int = 30) -> bool:
        """Acquire a distributed lock.

        Args:
            lock_key: Unique identifier for the lock
            holder_id: Identifier for the lock holder (for safe release)
            ttl: Time-to-live in seconds (auto-release after this time)

        Returns:
            True if lock was acquired, False if already held
        """
        pass

    @abstractmethod
    async def release_lock(self, lock_key: str, holder_id: str) -> bool:
        """Release a distributed lock.

        Only releases if the lock is held by the specified holder_id.

        Args:
            lock_key: Unique identifier for the lock
            holder_id: Identifier for the lock holder

        Returns:
            True if lock was released, False if not held or held by another
        """
        pass


# Re-export from OSS
__all__ = (
    'CacheServicePluginBase',
    'EXT_CACHE_SERVICE',
    'CacheService',
    'EnterpriseCacheService',
    'DEFAULT_MEMORYLAYER_CACHE_SERVICE',
)
