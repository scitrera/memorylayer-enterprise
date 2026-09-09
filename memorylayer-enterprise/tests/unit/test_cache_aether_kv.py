# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the Aether KV-backed enterprise cache service.

Drives :class:`AetherKVCacheService` against an in-memory fake KV client that
ports the map+TTL semantics of the Aether Go SDK ``coord.MemoryLocker`` and
returns ``KVResponse``-like objects.  Covers serialization round-trips, TTL
passthrough, ``clear_prefix`` via list+delete, the lease-based lock, and
fail-open behaviour on simulated errors and timeouts.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from memorylayer_saas.services.cache.base import EnterpriseCacheService
from memorylayer_saas.services.cache.aether_kv import (
    AetherKVCacheService,
    AetherKVCacheServicePlugin,
    DEFAULT_CACHE_AETHER_PREFIX,
)


# ---------------------------------------------------------------------------
# Fake KV client (ports coord.MemoryLocker map + guarded counter)
# ---------------------------------------------------------------------------

def _resp(**kw):
    """A KVResponse-like object with sensible defaults."""
    defaults = dict(success=True, value=b"", keys=[], counter_value=0, applied=False)
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class FakeKVClient:
    """Minimal async KV client backing AetherKVCacheService in tests."""

    def __init__(self):
        self.store: dict[str, bytes] = {}
        self.counters: dict[str, int] = {}

    async def kv_get(self, key, scope="global", workspace="", timeout=5.0, **kw):
        # A GET miss returns success with an empty value (Aether semantics).
        return _resp(value=self.store.get(key, b""))

    async def kv_put(self, key, value, scope="global", workspace="", ttl=0, timeout=5.0, **kw):
        self.store[key] = value
        return _resp(success=True)

    async def kv_list(self, key_prefix="", scope="global", workspace="", timeout=5.0, **kw):
        keys = [k for k in self.store if k.startswith(key_prefix)]
        return _resp(keys=keys)

    async def kv_delete(self, key, scope="global", workspace="", timeout=5.0, **kw):
        existed = self.store.pop(key, None) is not None
        self.counters.pop(key, None)
        return _resp(success=existed)

    async def kv_increment_if(self, key, delta=1, ceiling=0, scope="global",
                              workspace="", ttl=0, timeout=5.0, **kw):
        cur = self.counters.get(key, 0)
        proposed = cur + delta
        if proposed > ceiling:
            return (cur, False)
        self.counters[key] = proposed
        self.store[key] = str(proposed).encode()
        return (proposed, True)


class BoomKVClient:
    """KV client whose every op raises -- exercises the fail-open paths."""

    async def kv_get(self, *a, **kw):
        raise RuntimeError("boom")

    async def kv_put(self, *a, **kw):
        raise RuntimeError("boom")

    async def kv_list(self, *a, **kw):
        raise RuntimeError("boom")

    async def kv_delete(self, *a, **kw):
        raise RuntimeError("boom")

    async def kv_increment_if(self, *a, **kw):
        raise RuntimeError("boom")


class TimeoutKVClient:
    """KV client that returns None (timeout) from every op."""

    async def kv_get(self, *a, **kw):
        return None

    async def kv_put(self, *a, **kw):
        return None

    async def kv_list(self, *a, **kw):
        return None

    async def kv_delete(self, *a, **kw):
        return None

    async def kv_increment_if(self, *a, **kw):
        return None


def _bound(client) -> AetherKVCacheService:
    svc = AetherKVCacheService(prefix=DEFAULT_CACHE_AETHER_PREFIX, scope="global")
    svc.logger = MagicMock()
    svc.bind_client(SimpleNamespace(client=client, workspace="_system"))
    return svc


@pytest.fixture()
def fake():
    return FakeKVClient()


@pytest.fixture()
def cache(fake):
    return _bound(fake)


# ---------------------------------------------------------------------------
# Cache surface
# ---------------------------------------------------------------------------

def test_is_enterprise_cache_service(cache):
    assert isinstance(cache, EnterpriseCacheService)


async def test_set_applies_prefix_and_json(cache, fake):
    await cache.set("emb:abc", [0.1, 0.2, 0.3])
    assert "ml:emb:abc" in fake.store
    assert json.loads(fake.store["ml:emb:abc"].decode()) == [0.1, 0.2, 0.3]


async def test_set_get_roundtrip(cache):
    await cache.set("k", {"a": 1, "nested": [1, 2]})
    assert await cache.get("k") == {"a": 1, "nested": [1, 2]}


async def test_str_passthrough(cache, fake):
    await cache.set("s", "hello")
    assert fake.store["ml:s"] == b"hello"
    assert await cache.get("s") == "hello"


async def test_get_miss_returns_none(cache):
    assert await cache.get("absent") is None


async def test_ttl_passed_through(fake):
    cache = _bound(fake)
    captured = {}
    orig = fake.kv_put

    async def spy(key, value, **kw):
        captured.update(kw)
        return await orig(key, value, **kw)

    fake.kv_put = spy
    await cache.set("k", "v", ttl_seconds=300)
    assert captured.get("ttl") == 300


async def test_delete(cache):
    await cache.set("k", "v")
    assert await cache.delete("k") is True
    assert await cache.delete("k") is False  # gone -> success False


async def test_exists(cache):
    assert await cache.exists("k") is False
    await cache.set("k", "v")
    assert await cache.exists("k") is True


async def test_clear_prefix(cache, fake):
    await cache.set("recall:ws1:a", 1)
    await cache.set("recall:ws1:b", 2)
    await cache.set("assoc:ws1:c", 3)
    deleted = await cache.clear_prefix("recall:ws1:")
    assert deleted == 2
    assert await cache.get("recall:ws1:a") is None
    assert await cache.get("assoc:ws1:c") == 3


# ---------------------------------------------------------------------------
# Lock surface (lease via kv_increment_if)
# ---------------------------------------------------------------------------

async def test_acquire_lock_lease(cache):
    assert await cache.acquire_lock("job", "owner-a", ttl=30) is True
    # contended: ceiling=1 guard blocks the second acquire
    assert await cache.acquire_lock("job", "owner-b", ttl=30) is False


async def test_release_lock_frees_lease(cache):
    assert await cache.acquire_lock("job", "owner-a", ttl=30) is True
    assert await cache.release_lock("job", "owner-a") is True
    # re-acquirable after release
    assert await cache.acquire_lock("job", "owner-b", ttl=30) is True


async def test_lock_is_non_reentrant(cache):
    # Documented deviation: even the current holder's re-acquire returns False
    # (the lease counter is already at the ceiling).
    assert await cache.acquire_lock("job", "owner-a", ttl=30) is True
    assert await cache.acquire_lock("job", "owner-a", ttl=30) is False


async def test_release_lock_does_not_verify_holder(cache):
    # Documented deviation from EnterpriseCacheService: the Python KV verbs have
    # no compare-and-delete, so a non-holder CAN release another's lock. The
    # in-memory backend honours ownership; this backend cannot.
    assert await cache.acquire_lock("job", "owner-a", ttl=30) is True
    assert await cache.release_lock("job", "owner-b") is True  # non-holder succeeds
    # lock is now free
    assert await cache.acquire_lock("job", "owner-c", ttl=30) is True


# ---------------------------------------------------------------------------
# Falsy-value round-trips (pin the empty-value-as-miss limitation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0, False, [], {}, "hello"])
async def test_falsy_json_values_roundtrip(cache, value):
    # These serialize to NON-empty bytes, so they round-trip correctly.
    await cache.set("k", value)
    assert await cache.get("k") == value


async def test_numeric_string_roundtrips_as_json_type(cache):
    # Inherent ambiguity of the JSON+str-passthrough scheme (matches the Redis
    # backend): a string that is itself valid JSON reads back as the parsed type.
    await cache.set("k", "0")
    assert await cache.get("k") == 0


async def test_empty_string_is_indistinguishable_from_miss(cache):
    # Documented limitation: "" serializes to empty bytes -> read back as a miss.
    await cache.set("k", "")
    assert await cache.get("k") is None


async def test_get_or_set_inherited(cache):
    calls = []

    async def factory():
        calls.append(1)
        return {"computed": True}

    assert await cache.get_or_set("k", factory, ttl_seconds=60) == {"computed": True}
    assert await cache.get_or_set("k", factory, ttl_seconds=60) == {"computed": True}
    assert len(calls) == 1  # factory only invoked on miss


# ---------------------------------------------------------------------------
# Fail-open / fail-closed behaviour
# ---------------------------------------------------------------------------

async def test_unbound_client_is_inert():
    svc = AetherKVCacheService()
    svc.logger = MagicMock()
    assert await svc.get("k") is None
    assert await svc.set("k", "v") is False
    assert await svc.delete("k") is False
    assert await svc.exists("k") is False
    assert await svc.clear_prefix("p") == 0
    assert await svc.acquire_lock("l", "o") is False
    assert await svc.release_lock("l", "o") is False


async def test_errors_fail_open():
    svc = _bound(BoomKVClient())
    assert await svc.get("k") is None
    assert await svc.set("k", "v") is False
    assert await svc.delete("k") is False
    assert await svc.exists("k") is False
    assert await svc.clear_prefix("p") == 0
    # acquire fails *closed* so a blip can't grant two holders
    assert await svc.acquire_lock("l", "o") is False
    assert await svc.release_lock("l", "o") is False


async def test_timeouts_fail_open():
    svc = _bound(TimeoutKVClient())
    assert await svc.get("k") is None
    assert await svc.set("k", "v") is False
    assert await svc.clear_prefix("p") == 0
    assert await svc.acquire_lock("l", "o") is False


async def test_non_serializable_value_returns_false(cache):
    assert await cache.set("k", object()) is False


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

def test_plugin_metadata_and_dependencies():
    plugin = AetherKVCacheServicePlugin()
    assert plugin.PROVIDER_NAME == "aether-kv"
    v = MagicMock()
    v.environ = MagicMock(side_effect=lambda key, default=None, **kw: default)
    svc = plugin.initialize(v, MagicMock())
    assert isinstance(svc, AetherKVCacheService)
    from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION
    assert tuple(plugin.get_dependencies(v)) == (EXT_AETHER_SERVICE_CONNECTION,)


async def test_plugin_async_ready_binds_client(fake):
    plugin = AetherKVCacheServicePlugin()
    svc = AetherKVCacheService()
    svc.logger = MagicMock()

    import memorylayer_saas.services.cache.aether_kv as mod
    v = MagicMock()
    agent_service = SimpleNamespace(client=fake, workspace="_system")
    # patch get_extension used inside the module
    orig = mod.get_extension
    mod.get_extension = lambda ext, vv: agent_service
    try:
        await plugin.async_ready(v, MagicMock(), svc)
    finally:
        mod.get_extension = orig
    assert svc.is_connected is True
    await svc.set("k", "v")
    assert await svc.get("k") == "v"
