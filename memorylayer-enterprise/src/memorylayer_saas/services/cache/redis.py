"""Cache layer using Redis/Valkey."""
import json
from logging import Logger
from typing import Optional, Any

import redis.asyncio as redis
from scitrera_app_framework import Variables, get_logger

from .base import EnterpriseCacheService, CacheServicePluginBase

# Redis settings (for SaaS implementations)
MEMORYLAYER_CACHE_REDIS_URL = 'MEMORYLAYER_CACHE_REDIS_URL'
DEFAULT_CACHE_REDIS_URL = 'redis://localhost:6379/0'

MEMORYLAYER_CACHE_REDIS_PREFIX = 'MEMORYLAYER_CACHE_REDIS_PREFIX'
DEFAULT_CACHE_REDIS_PREFIX = 'ml:'


class RedisCacheService(EnterpriseCacheService):
    """
    Async Redis/Valkey cache client implementing the CacheService interface.

    Used for:
    - Working memory (session context)
    - Recent memories cache
    - Distributed locking
    """

    def __init__(self, v: Variables = None, url: str = None, prefix: str = 'ml:'):
        self.url = url
        self.prefix = prefix
        self._client: Optional[redis.Redis] = None
        self.logger = get_logger(v, name=self.__class__.__name__)

    async def connect(self) -> None:
        """Connect to Redis/Valkey."""
        self._client = redis.from_url(self.url, decode_responses=True)
        await self._client.ping()
        self.logger.info("Connected to Redis at %s", self.url)

    async def disconnect(self) -> None:
        """Close connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def health_check(self) -> bool:
        """Check if cache is healthy."""
        if self._client:
            try:
                await self._client.ping()
                return True
            except redis.RedisError:
                return False
        return False

    def _prefixed_key(self, key: str) -> str:
        """Apply prefix to key."""
        return f"{self.prefix}{key}"

    # CacheService interface implementation
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._client:
            return None
        value = await self._client.get(self._prefixed_key(key))
        if value is None:
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self._client:
            return False
        try:
            serialized = json.dumps(value) if not isinstance(value, str) else value
            await self._client.set(self._prefixed_key(key), serialized, ex=ttl_seconds)
            return True
        except (TypeError, redis.RedisError):
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self._client:
            return False
        result = await self._client.delete(self._prefixed_key(key))
        return result > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self._client:
            return False
        return await self._client.exists(self._prefixed_key(key)) > 0

    async def clear_prefix(self, prefix: str) -> int:
        """Clear all keys with given prefix."""
        if not self._client:
            return 0
        full_prefix = self._prefixed_key(prefix)
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = await self._client.scan(cursor, match=f"{full_prefix}*", count=100)
            if keys:
                deleted += await self._client.delete(*keys)
            if cursor == 0:
                break
        return deleted

    # Session context operations (Redis-specific extensions)
    async def set_session_context(
            self, session_id: str, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> bool:
        """Store session context data."""
        cache_key = f"session:{session_id}:{key}"
        return await self.set(cache_key, value, ttl_seconds)

    async def get_session_context(self, session_id: str, key: str) -> Optional[Any]:
        """Retrieve session context data."""
        cache_key = f"session:{session_id}:{key}"
        return await self.get(cache_key)

    # Distributed locking (Redis-specific extensions)
    async def acquire_lock(self, lock_key: str, holder_id: str, ttl: int = 30) -> bool:
        """Acquire distributed lock using SET NX."""
        if not self._client:
            return False
        full_key = self._prefixed_key(f"lock:{lock_key}")
        result = await self._client.set(full_key, holder_id, nx=True, ex=ttl)
        return result is True

    async def release_lock(self, lock_key: str, holder_id: str) -> bool:
        """Release lock if we hold it."""
        if not self._client:
            return False
        full_key = self._prefixed_key(f"lock:{lock_key}")
        # Lua script for atomic check-and-delete
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        result = await self._client.eval(script, 1, full_key, holder_id)
        return result == 1


class RedisCacheServicePlugin(CacheServicePluginBase):
    """Plugin for redis cache service."""
    PROVIDER_NAME = 'redis'

    def initialize(self, v: Variables, logger: Logger) -> EnterpriseCacheService:
        return RedisCacheService(
            v=v,
            url=v.environ(MEMORYLAYER_CACHE_REDIS_URL, DEFAULT_CACHE_REDIS_URL),
            prefix=v.environ(MEMORYLAYER_CACHE_REDIS_PREFIX, DEFAULT_CACHE_REDIS_PREFIX),
        )

    async def async_ready(self, v: Variables, logger: Logger, value: object | None) -> None:
        if isinstance(value, RedisCacheService):
            await value.connect()

    async def async_stopping(self, v: Variables, logger: Logger, value: object | None) -> None:
        if isinstance(value, RedisCacheService):
            await value.disconnect()
