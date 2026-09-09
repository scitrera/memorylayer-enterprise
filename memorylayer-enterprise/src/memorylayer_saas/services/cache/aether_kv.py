"""Aether KV-backed cache service for MemoryLayer Enterprise.

Uses the shared ``AsyncServiceClient`` from :class:`AetherServiceConnection`
(the same client that backs rate limiting and kb_update debouncing) as a
distributed cache, removing the need for a separate Redis/Valkey deployment.

Design notes
------------
* Keys are prefixed (default ``ml:``) and stored under a single KV ``scope``
  (default ``global``) -- cache keys already embed the workspace id where
  isolation matters (e.g. ``recall:{workspace}:...``), mirroring the previous
  Redis single-keyspace behaviour.
* Values are JSON-encoded to ``bytes`` (Aether KV stores raw bytes); plain
  ``str`` values pass through unencoded, matching the Redis backend.
* ``clear_prefix`` uses ``kv_list(key_prefix=...)`` (the SCAN equivalent) and
  deletes each returned key.
* Distributed locking is *best-effort* and weaker than the Redis backend's
  ``SET NX`` lock.  The Python KV client exposes no SetNX/compare-and-set, so
  ``acquire_lock`` uses the atomic ``kv_increment_if(ceiling=1, ttl)`` lease
  primitive (the same pattern ``tasks/kb_update.py`` runs against live Aether)
  and ``release_lock`` deletes the key.  Two deliberate deviations from the
  :class:`EnterpriseCacheService` lock contract follow from the missing CAS:
    1. ``release_lock`` cannot verify ownership -- ``holder_id`` is advisory and
       a non-holder *can* release another holder's lock (the in-memory backend,
       which has CAS-equivalent in-process state, does honour ownership).
    2. The lock is non-reentrant and non-renewable -- once the counter is at the
       ceiling, even the current holder's re-acquire returns ``False``.  A
       critical section MUST complete within ``ttl``; a crashed holder is
       reclaimed only when the lease TTL expires.
  The fully owner-safe lock (Aether Go SDK ``coord.Locker``) is not expressible
  with the current Python KV verbs.  Locking has no live callers today.
* Every operation is *fail-open*: KV/connectivity errors and timeouts log and
  degrade to a cache miss (``None`` / ``False`` / ``0``) rather than raising,
  matching the rate-limit and kb_update conventions so a KV blip never breaks a
  request.
"""
from __future__ import annotations

import json
from logging import Logger
from typing import Any, Iterable, Optional

from scitrera_app_framework import Variables, get_logger, get_extension

from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION

from .base import EnterpriseCacheService, CacheServicePluginBase

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MEMORYLAYER_CACHE_AETHER_PREFIX = "MEMORYLAYER_CACHE_AETHER_PREFIX"
DEFAULT_CACHE_AETHER_PREFIX = "ml:"

MEMORYLAYER_CACHE_AETHER_SCOPE = "MEMORYLAYER_CACHE_AETHER_SCOPE"
DEFAULT_CACHE_AETHER_SCOPE = "global"

MEMORYLAYER_CACHE_AETHER_TIMEOUT = "MEMORYLAYER_CACHE_AETHER_TIMEOUT"
DEFAULT_CACHE_AETHER_TIMEOUT = 2.0


class AetherKVCacheService(EnterpriseCacheService):
    """Distributed cache backed by Aether KV via the shared service client."""

    def __init__(
        self,
        v: Variables = None,
        *,
        prefix: str = DEFAULT_CACHE_AETHER_PREFIX,
        scope: str = DEFAULT_CACHE_AETHER_SCOPE,
        timeout: float = DEFAULT_CACHE_AETHER_TIMEOUT,
    ) -> None:
        self._v = v
        self.prefix = prefix
        self.scope = scope
        self.timeout = timeout
        self._client = None
        self._workspace: str = "_system"
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info(
            "Initialized AetherKVCacheService (scope=%s, prefix=%s)", scope, prefix
        )

    # ------------------------------------------------------------------
    # Client binding (shared connection, not owned by this service)
    # ------------------------------------------------------------------
    def bind_client(self, agent_service) -> None:
        """Bind to the shared client from AetherServiceConnection."""
        self._client = agent_service.client
        self._workspace = agent_service.workspace
        self.logger.info(
            "Bound to shared Aether client (workspace=%s, connected=%s)",
            self._workspace,
            self._client is not None,
        )

    async def disconnect(self) -> None:
        """Release the client reference (do NOT close -- not our client)."""
        self._client = None

    @property
    def is_connected(self) -> bool:
        """Return True if the shared Aether client is available."""
        return self._client is not None

    def _prefixed_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    # ------------------------------------------------------------------
    # CacheService interface
    # ------------------------------------------------------------------
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache. Returns None on miss or any KV error."""
        if self._client is None:
            return None
        try:
            resp = await self._client.kv_get(
                self._prefixed_key(key),
                scope=self.scope,
                workspace=self._workspace,
                timeout=self.timeout,
            )
        except Exception:
            self.logger.error("Aether KV get failed for key %s (fail-open)", key, exc_info=True)
            return None
        # Timeout -> None; a GET miss returns success with an empty value.
        if resp is None or not getattr(resp, "value", b""):
            return None
        return _deserialize(resp.value)

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache. Returns False on any KV error.

        Note: an empty-string value serializes to empty bytes, which ``get``
        reports as a miss (Aether cannot distinguish a stored empty value from
        an absent key).  Callers should not rely on caching ``""``.
        """
        if self._client is None:
            return False
        try:
            data = _serialize(value)
        except (TypeError, ValueError):
            self.logger.error("Aether KV set: value not serializable for key %s", key, exc_info=True)
            return False
        # KV uses ttl=0 to mean "no expiry"; map None -> 0 explicitly.
        ttl = 0 if ttl_seconds is None else ttl_seconds
        try:
            resp = await self._client.kv_put(
                self._prefixed_key(key),
                data,
                scope=self.scope,
                workspace=self._workspace,
                ttl=ttl,
                timeout=self.timeout,
            )
        except Exception:
            self.logger.error("Aether KV set failed for key %s (fail-open)", key, exc_info=True)
            return False
        return bool(resp is not None and getattr(resp, "success", False))

    async def delete(self, key: str) -> bool:
        """Delete key from cache. Returns False on any KV error."""
        if self._client is None:
            return False
        try:
            resp = await self._client.kv_delete(
                self._prefixed_key(key),
                scope=self.scope,
                workspace=self._workspace,
                timeout=self.timeout,
            )
        except Exception:
            self.logger.error("Aether KV delete failed for key %s (fail-open)", key, exc_info=True)
            return False
        return bool(resp is not None and getattr(resp, "success", False))

    async def exists(self, key: str) -> bool:
        """Check if key exists (non-empty value) in cache."""
        if self._client is None:
            return False
        try:
            resp = await self._client.kv_get(
                self._prefixed_key(key),
                scope=self.scope,
                workspace=self._workspace,
                timeout=self.timeout,
            )
        except Exception:
            self.logger.error("Aether KV exists failed for key %s (fail-open)", key, exc_info=True)
            return False
        return bool(resp is not None and getattr(resp, "value", b""))

    async def clear_prefix(self, prefix: str) -> int:
        """Clear all keys with the given prefix via kv_list + kv_delete."""
        if self._client is None:
            return 0
        full_prefix = self._prefixed_key(prefix)
        try:
            resp = await self._client.kv_list(
                key_prefix=full_prefix,
                scope=self.scope,
                workspace=self._workspace,
                timeout=self.timeout,
            )
        except Exception:
            self.logger.error("Aether KV list failed for prefix %s (fail-open)", prefix, exc_info=True)
            return 0
        if resp is None:
            return 0
        keys = list(getattr(resp, "keys", []) or [])
        deleted = 0
        for full_key in keys:
            try:
                del_resp = await self._client.kv_delete(
                    full_key,
                    scope=self.scope,
                    workspace=self._workspace,
                    timeout=self.timeout,
                )
            except Exception:
                self.logger.warning("Aether KV delete failed for key %s during clear_prefix", full_key, exc_info=True)
                continue
            if del_resp is not None and getattr(del_resp, "success", False):
                deleted += 1
        if deleted < len(keys):
            self.logger.warning(
                "Aether KV clear_prefix(%s): deleted %d of %d listed keys",
                prefix, deleted, len(keys),
            )
        return deleted

    # ------------------------------------------------------------------
    # EnterpriseCacheService interface (best-effort lease-based lock)
    # ------------------------------------------------------------------
    async def acquire_lock(self, lock_key: str, holder_id: str, ttl: int = 30) -> bool:
        """Acquire a best-effort lease via kv_increment_if(ceiling=1, ttl).

        The first caller drives the counter 0 -> 1 (``applied=True``); a
        concurrent caller is guarded off (``applied=False``).  Returns False on
        any KV error -- we fail *closed* on acquire so a blip cannot grant two
        holders.
        """
        if self._client is None:
            return False
        full_key = self._prefixed_key(f"lock:{lock_key}")
        try:
            result = await self._client.kv_increment_if(
                full_key,
                delta=1,
                ceiling=1,
                scope=self.scope,
                workspace=self._workspace,
                ttl=ttl,
                timeout=self.timeout,
            )
        except Exception:
            self.logger.error("Aether KV acquire_lock failed for %s (fail-closed)", lock_key, exc_info=True)
            return False
        if result is None:
            return False
        _value, applied = result
        return bool(applied)

    async def release_lock(self, lock_key: str, holder_id: str) -> bool:
        """Release the lease by deleting the lock key (best-effort).

        ``holder_id`` is advisory: the Python KV verbs cannot atomically verify
        ownership before deletion.  TTL expiry bounds a crashed holder.
        """
        if self._client is None:
            return False
        full_key = self._prefixed_key(f"lock:{lock_key}")
        try:
            resp = await self._client.kv_delete(
                full_key,
                scope=self.scope,
                workspace=self._workspace,
                timeout=self.timeout,
            )
        except Exception:
            self.logger.error("Aether KV release_lock failed for %s (fail-open)", lock_key, exc_info=True)
            return False
        return bool(resp is not None and getattr(resp, "success", False))


# ---------------------------------------------------------------------------
# Serialization helpers (bytes <-> JSON, str passthrough)
# ---------------------------------------------------------------------------
def _serialize(value: Any) -> bytes:
    """Encode a cache value to bytes (str passthrough, else JSON)."""
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value).encode("utf-8")


def _deserialize(raw: bytes) -> Any:
    """Decode bytes from KV back into a value, falling back to the raw string."""
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------
class AetherKVCacheServicePlugin(CacheServicePluginBase):
    """Plugin that creates and manages an :class:`AetherKVCacheService`.

    Selected when ``MEMORYLAYER_CACHE_SERVICE=aether-kv`` (the enterprise
    default).

    Lifecycle:
        ``initialize``     -- constructs the service (no I/O).
        ``async_ready``    -- binds to the shared Aether client.
        ``async_stopping`` -- releases the client reference.
    """

    PROVIDER_NAME = "aether-kv"

    def initialize(self, v: Variables, logger: Logger) -> AetherKVCacheService:
        return AetherKVCacheService(
            v,
            prefix=v.environ(MEMORYLAYER_CACHE_AETHER_PREFIX, DEFAULT_CACHE_AETHER_PREFIX),
            scope=v.environ(MEMORYLAYER_CACHE_AETHER_SCOPE, DEFAULT_CACHE_AETHER_SCOPE),
            timeout=v.environ(MEMORYLAYER_CACHE_AETHER_TIMEOUT, DEFAULT_CACHE_AETHER_TIMEOUT, type_fn=float),
        )

    async def async_ready(self, v: Variables, logger: Logger, value: object | None) -> None:
        if not isinstance(value, AetherKVCacheService):
            return
        try:
            agent_service = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
            value.bind_client(agent_service)
        except Exception:
            logger.error("AetherKVCacheService failed to bind to shared Aether client", exc_info=True)

    async def async_stopping(self, v: Variables, logger: Logger, value: object | None) -> None:
        if isinstance(value, AetherKVCacheService):
            await value.disconnect()

    def get_dependencies(self, v: Variables) -> Iterable[str] | None:
        return (EXT_AETHER_SERVICE_CONNECTION,)
