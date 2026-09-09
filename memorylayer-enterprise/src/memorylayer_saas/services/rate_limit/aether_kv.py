"""Aether KV-backed rate limit service for MemoryLayer Enterprise.

Uses the shared ``AsyncServiceClient`` from :class:`AetherServiceConnection`
to maintain per-key rate limit counters visible across all nodes sharing
the same Aether workspace.

Design notes
------------
* Key format: ``ratelimit:{key}``
* Value: plain integer counter managed by Redis INCR (via ``kv_increment``)
* TTL is set on the first increment; subsequent increments just bump the
  counter.  When the TTL expires the key disappears and the next request
  starts a new window automatically.
* ``kv_increment`` is atomic -- no read-modify-write race conditions.
"""
from __future__ import annotations

import time
from logging import Logger
from typing import Optional, Iterable

from scitrera_app_framework import Variables, get_logger, get_extension, ext_parse_bool

from memorylayer_server.config import (
    MEMORYLAYER_RATE_LIMIT_REQUESTS,
    DEFAULT_MEMORYLAYER_RATE_LIMIT_REQUESTS,
    MEMORYLAYER_RATE_LIMIT_WINDOW_SECONDS,
    DEFAULT_MEMORYLAYER_RATE_LIMIT_WINDOW_SECONDS,
)
from memorylayer_server.services.rate_limit.base import (
    RateLimitResult,
    RateLimitService,
    RateLimitServicePluginBase,
)

from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
AETHER_RATELIMIT_ENABLED = "AETHER_RATELIMIT_ENABLED"
DEFAULT_AETHER_RATELIMIT_ENABLED = True

_KV_SCOPE = "workspace"
_KV_KEY_PREFIX = "ratelimit:"


class AetherKVRateLimitService(RateLimitService):
    """Rate limit service backed by Aether KV storage via the shared client.

    Each rate limit entry is stored under the key ``ratelimit:{key}`` as a
    plain integer counter incremented atomically via ``kv_increment``.  The
    TTL is set on the first increment so that Aether automatically expires
    stale entries and resets the window.
    """

    def __init__(
            self,
            v: Variables,
            *,
            default_limit: int = DEFAULT_MEMORYLAYER_RATE_LIMIT_REQUESTS,
            default_window_seconds: int = DEFAULT_MEMORYLAYER_RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        self._v = v
        self._default_limit = default_limit
        self._default_window_seconds = default_window_seconds
        self._client = None
        self._workspace: str = "_system"
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized AetherKVRateLimitService")

    # ------------------------------------------------------------------
    # Client binding (replaces own connect/disconnect lifecycle)
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
        """Release client reference (do NOT close — not our client)."""
        self._client = None

    @property
    def is_connected(self) -> bool:
        """Return ``True`` if the shared Aether client is available."""
        return self._client is not None

    # ------------------------------------------------------------------
    # RateLimitService interface
    # ------------------------------------------------------------------

    async def check_rate_limit(
            self,
            key: str,
            limit: int = 0,
            window_seconds: int = 0,
    ) -> RateLimitResult:
        """Check and increment the rate limit counter for ``key``.

        If the Aether client is not connected, the request is allowed
        (fail-open behaviour) to avoid blocking traffic due to connectivity
        issues.
        """
        effective_limit = limit if limit > 0 else self._default_limit
        effective_window = window_seconds if window_seconds > 0 else self._default_window_seconds

        now = time.time()
        reset_at = now + effective_window

        if self._client is None:
            self.logger.warning(
                "Aether client not connected; allowing request for key %s (fail-open)",
                key,
            )
            return RateLimitResult(
                allowed=True,
                limit=effective_limit,
                remaining=effective_limit,
                reset_at=reset_at,
            )

        kv_key = f"{_KV_KEY_PREFIX}{key}"

        try:
            response = await self._client.kv_increment(
                key=kv_key,
                scope=_KV_SCOPE,
                workspace=self._workspace,
                ttl=effective_window,
                timeout=2.0,
            )
            if response is None:
                self.logger.warning(
                    "Aether kv_increment timed out for key %s; allowing request (fail-open)",
                    key,
                )
                return RateLimitResult(
                    allowed=True,
                    limit=effective_limit,
                    remaining=effective_limit,
                    reset_at=reset_at,
                )
            count = response.counter_value

        except Exception:
            self.logger.error(
                "Aether KV error during rate limit check for key %s; allowing request (fail-open)",
                key,
                exc_info=True,
            )
            return RateLimitResult(
                allowed=True,
                limit=effective_limit,
                remaining=effective_limit,
                reset_at=reset_at,
            )

        allowed = count <= effective_limit
        remaining = max(0, effective_limit - count)

        self.logger.debug(
            "Rate limit check: key=%s count=%d limit=%d allowed=%s",
            key,
            count,
            effective_limit,
            allowed,
        )

        return RateLimitResult(
            allowed=allowed,
            limit=effective_limit,
            remaining=remaining,
            reset_at=reset_at,
        )

    async def get_usage(self, key: str) -> tuple[int, int]:
        """Return ``(current_count, max_limit)`` for a key."""
        if self._client is None:
            return 0, self._default_limit

        kv_key = f"{_KV_KEY_PREFIX}{key}"

        try:
            existing = await self._client.kv_get(
                kv_key,
                scope=_KV_SCOPE,
                workspace=self._workspace,
            )
            if existing is None:
                return 0, self._default_limit

            count = int(existing.value)
            return count, self._default_limit

        except Exception:
            self.logger.error(
                "Aether KV error during get_usage for key %s",
                key,
                exc_info=True,
            )
            return 0, self._default_limit


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class AetherKVRateLimitServicePlugin(RateLimitServicePluginBase):
    """Plugin that creates and manages an :class:`AetherKVRateLimitService`.

    Enabled when ``MEMORYLAYER_RATE_LIMIT_SERVICE=aether-kv``.

    Lifecycle:
        ``initialize``    -- constructs the service (no I/O).
        ``async_ready``   -- binds to the shared Aether client from AetherServiceConnection.
        ``async_stopping``-- releases the client reference.
    """

    PROVIDER_NAME = "aether-kv"

    def initialize(self, v: Variables, logger: Logger) -> Optional[AetherKVRateLimitService]:
        """Create and return the service instance (no I/O)."""
        default_limit = v.environ(
            MEMORYLAYER_RATE_LIMIT_REQUESTS,
            DEFAULT_MEMORYLAYER_RATE_LIMIT_REQUESTS,
            type_fn=int,
        )
        default_window_seconds = v.environ(
            MEMORYLAYER_RATE_LIMIT_WINDOW_SECONDS,
            DEFAULT_MEMORYLAYER_RATE_LIMIT_WINDOW_SECONDS,
            type_fn=int,
        )

        return AetherKVRateLimitService(
            v,
            default_limit=default_limit,
            default_window_seconds=default_window_seconds,
        )

    async def async_ready(self, v: Variables, logger: Logger, value: AetherKVRateLimitService) -> None:
        """Bind the rate limit service to the shared Aether client."""
        try:
            agent_service = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
            value.bind_client(agent_service)
        except Exception:
            logger.error("AetherKVRateLimitService failed to bind to shared Aether client", exc_info=True)

    async def async_stopping(self, v: Variables, logger: Logger, value: AetherKVRateLimitService) -> None:
        """Release the client reference."""
        await value.disconnect()

    def get_dependencies(self, v: Variables) -> Iterable[str] | None:
        return (EXT_AETHER_SERVICE_CONNECTION,)
