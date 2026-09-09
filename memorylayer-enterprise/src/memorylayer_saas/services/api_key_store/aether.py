# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Aether-backed API key store for MemoryLayer Enterprise.

Resolves provider API keys from Aether secure KV first, falling back to the
environment -- mirroring ``TenantInterface2.get_api_key`` so keys committed via
TI2 are picked up here. The wire-key structure is identical to TI2's so a key
written by one is readable by the other:

    wire key : ``enc:ti:<tenant>:ikv:api_key:<NAME>``
    KV scope : ``global``  (user_id="", workspace="")
    value    : msgpack-packed (``enc:`` at-rest encryption is currently inert;
               the value is cleartext-but-msgpack-wrapped)

``<NAME>`` is the raw secret/env-var name (e.g. ``OPENAI_API_KEY``); ``<tenant>``
is this deployment's tenant slug from ``SCITRERA_TENANT`` (MemoryLayer runs one
deployment per tenant).

Resolution is *fail-open*: any KV/connectivity error degrades to the env
fallback rather than raising, so a KV blip never breaks LLM/embedding routing.
A short TTL cache bounds per-call KV load while still picking up rotations.
"""

from __future__ import annotations

import time
from logging import Logger
from typing import Iterable

import msgpack

from scitrera_app_framework import Variables, get_extension, get_logger

from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION
from memorylayer_server.services.api_key_store import (
    DEFAULT_MEMORYLAYER_API_KEY_STORE_TTL,
    MEMORYLAYER_API_KEY_STORE_TTL,
    ApiKeyStore,
    ApiKeyStorePluginBase,
)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
# Per-deployment tenant slug (the ``ti:<tenant>:`` key prefix). Injected by the
# platform framework; MemoryLayer runs one deployment per tenant.
SCITRERA_TENANT = "SCITRERA_TENANT"

# KV scope used by TI2's secure internal-KV writes (global, tenant isolation is
# carried by the key prefix, NOT the scope).
MEMORYLAYER_API_KEY_STORE_AETHER_SCOPE = "MEMORYLAYER_API_KEY_STORE_AETHER_SCOPE"
DEFAULT_API_KEY_STORE_AETHER_SCOPE = "global"

MEMORYLAYER_API_KEY_STORE_AETHER_TIMEOUT = "MEMORYLAYER_API_KEY_STORE_AETHER_TIMEOUT"
DEFAULT_API_KEY_STORE_AETHER_TIMEOUT = 2.0

# Sentinel distinguishing "cached a real miss" from "not cached".
_MISS = object()


def _decode_value(raw) -> str | None:
    """Decode a KV value written by TI2 (msgpack) into a string, or ``None``."""
    if raw is None:
        return None
    try:
        val = msgpack.unpackb(raw, raw=False)
    except Exception:
        # Defensive: a value stored as raw bytes (not msgpack) still round-trips.
        val = raw
    if isinstance(val, (bytes, bytearray)):
        val = val.decode("utf-8", errors="replace")
    if val is None:
        return None
    val = str(val)
    return val or None


class AetherApiKeyStore(ApiKeyStore):
    """Resolve API keys from Aether secure KV (TI2-compatible), env fallback."""

    def __init__(
        self,
        v: Variables = None,
        *,
        tenant: str | None,
        scope: str = DEFAULT_API_KEY_STORE_AETHER_SCOPE,
        timeout: float = DEFAULT_API_KEY_STORE_AETHER_TIMEOUT,
        ttl: float = DEFAULT_MEMORYLAYER_API_KEY_STORE_TTL,
    ) -> None:
        self._v = v
        self._tenant = tenant
        self._scope = scope
        self._timeout = timeout
        self._ttl = ttl
        self._client = None
        # name -> (expires_at_monotonic, value-or-_MISS)
        self._cache: dict[str, tuple[float, object]] = {}
        self.logger = get_logger(v, name=self.__class__.__name__)
        if not tenant:
            self.logger.warning(
                "AetherApiKeyStore: SCITRERA_TENANT unset -- secure KV lookups are "
                "disabled; resolving API keys from the environment only."
            )
        self.logger.info(
            "Initialized AetherApiKeyStore (tenant=%s, scope=%s, ttl=%ss)",
            tenant, scope, ttl,
        )

    # ------------------------------------------------------------------
    # Client binding (shared connection, not owned by this service)
    # ------------------------------------------------------------------
    def bind_client(self, agent_service) -> None:
        """Bind to the shared client from AetherServiceConnection."""
        self._client = agent_service.client
        self.logger.info(
            "Bound AetherApiKeyStore to shared Aether client (connected=%s)",
            self._client is not None,
        )

    def disconnect(self) -> None:
        """Release the client reference (do NOT close -- not our client)."""
        self._client = None

    def _wire_key(self, name: str) -> str:
        # ``enc:`` MUST stay at the very front (it gates Aether at-rest
        # encryption); the tenant prefix follows it. Mirrors TI2's
        # AetherKVHelper._key("enc:ikv:api_key:<NAME>").
        return f"enc:ti:{self._tenant}:ikv:api_key:{name}"

    async def get_api_key(self, name: str, *, default: str | None = None) -> str | None:
        cached = self._cache.get(name)
        if cached is not None and cached[0] > time.monotonic():
            val = cached[1]
            return default if val is _MISS else val  # type: ignore[return-value]

        resolved = await self._resolve(name)
        # Cache the resolved value (or a miss sentinel) for the TTL window.
        self._cache[name] = (
            time.monotonic() + self._ttl,
            _MISS if resolved is None else resolved,
        )
        return default if resolved is None else resolved

    async def _resolve(self, name: str) -> str | None:
        # 1. Aether secure KV (TI2-compatible wire key), fail-open.
        if self._client is not None and self._tenant:
            try:
                resp = await self._client.kv_get(
                    self._wire_key(name),
                    scope=self._scope,
                    user_id="",
                    workspace="",
                    timeout=self._timeout,
                )
            except Exception:
                self.logger.warning(
                    "Aether KV api_key lookup failed for %r (fail-open to env)",
                    name, exc_info=True,
                )
                resp = None
            if resp is not None and getattr(resp, "value", b""):
                val = _decode_value(resp.value)
                if val:
                    return val

        # 2. Environment fallback (same name).
        if self._v is not None:
            env = self._v.environ(name, default=None)
            if env:
                return env
        return None


class AetherApiKeyStorePlugin(ApiKeyStorePluginBase):
    """Plugin selected when ``MEMORYLAYER_API_KEY_STORE=aether`` (enterprise default).

    Lifecycle:
        ``initialize``     -- constructs the store (no I/O).
        ``async_ready``    -- binds to the shared Aether client.
        ``async_stopping`` -- releases the client reference.
    """

    PROVIDER_NAME = "aether"

    def initialize(self, v: Variables, logger: Logger) -> AetherApiKeyStore:
        return AetherApiKeyStore(
            v,
            tenant=v.environ(SCITRERA_TENANT, default=None),
            scope=v.environ(MEMORYLAYER_API_KEY_STORE_AETHER_SCOPE, DEFAULT_API_KEY_STORE_AETHER_SCOPE),
            timeout=v.environ(
                MEMORYLAYER_API_KEY_STORE_AETHER_TIMEOUT,
                DEFAULT_API_KEY_STORE_AETHER_TIMEOUT, type_fn=float,
            ),
            ttl=v.environ(
                MEMORYLAYER_API_KEY_STORE_TTL,
                DEFAULT_MEMORYLAYER_API_KEY_STORE_TTL, type_fn=float,
            ),
        )

    async def async_ready(self, v: Variables, logger: Logger, value: object | None) -> None:
        if not isinstance(value, AetherApiKeyStore):
            return
        try:
            agent_service = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
            value.bind_client(agent_service)
        except Exception:
            logger.error("AetherApiKeyStore failed to bind to shared Aether client", exc_info=True)

    async def async_stopping(self, v: Variables, logger: Logger, value: object | None) -> None:
        if isinstance(value, AetherApiKeyStore):
            value.disconnect()

    def get_dependencies(self, v: Variables) -> Iterable[str] | None:
        return (EXT_AETHER_SERVICE_CONNECTION,)
