# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the in-memory enterprise cache service.

Covers the full CacheService surface (get/set/TTL/delete/exists/clear_prefix)
plus the in-process EnterpriseCacheService lock emulation
(acquire/contend/owner-safe release/TTL expiry).
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from memorylayer_saas.services.cache.base import EnterpriseCacheService
from memorylayer_saas.services.cache.in_memory import (
    InMemoryCacheService,
    InMemoryCacheServicePlugin,
)


@pytest.fixture()
def cache() -> InMemoryCacheService:
    return InMemoryCacheService(logger=MagicMock(), maxsize=128)


# ---------------------------------------------------------------------------
# Cache surface
# ---------------------------------------------------------------------------

def test_is_enterprise_cache_service(cache):
    assert isinstance(cache, EnterpriseCacheService)


async def test_set_get_roundtrip(cache):
    assert await cache.set("k", {"a": 1, "b": [1, 2, 3]}) is True
    assert await cache.get("k") == {"a": 1, "b": [1, 2, 3]}


async def test_get_missing_returns_none(cache):
    assert await cache.get("nope") is None


@pytest.mark.parametrize("value", [0, False, [], {}, ""])
async def test_falsy_values_roundtrip(cache, value):
    # In-memory stores the object directly, so all falsy values round-trip
    # (including ""), unlike the Aether backend which conflates "" with a miss.
    await cache.set("k", value)
    assert await cache.get("k") == value


async def test_ttl_expiry(cache, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    await cache.set("k", "v", ttl_seconds=10)
    assert await cache.get("k") == "v"
    now[0] += 11  # advance past TTL
    assert await cache.get("k") is None
    # expired entry is evicted
    assert "k" not in cache._cache


async def test_no_ttl_never_expires(cache, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    await cache.set("k", "v")  # no ttl
    now[0] += 10_000
    assert await cache.get("k") == "v"


async def test_delete(cache):
    await cache.set("k", "v")
    assert await cache.delete("k") is True
    assert await cache.get("k") is None
    assert await cache.delete("k") is False  # already gone


async def test_exists(cache, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    assert await cache.exists("k") is False
    await cache.set("k", "v", ttl_seconds=5)
    assert await cache.exists("k") is True
    now[0] += 6
    assert await cache.exists("k") is False


async def test_clear_prefix(cache):
    await cache.set("recall:ws1:a", 1)
    await cache.set("recall:ws1:b", 2)
    await cache.set("recall:ws2:c", 3)
    await cache.set("assoc:ws1:d", 4)
    deleted = await cache.clear_prefix("recall:ws1:")
    assert deleted == 2
    assert await cache.get("recall:ws1:a") is None
    assert await cache.get("recall:ws1:b") is None
    assert await cache.get("recall:ws2:c") == 3
    assert await cache.get("assoc:ws1:d") == 4


async def test_get_or_set(cache):
    calls = []

    async def factory():
        calls.append(1)
        return "computed"

    assert await cache.get_or_set("k", factory, ttl_seconds=60) == "computed"
    assert await cache.get_or_set("k", factory, ttl_seconds=60) == "computed"
    assert len(calls) == 1  # factory only invoked on miss


# ---------------------------------------------------------------------------
# Lock surface (in-process emulation)
# ---------------------------------------------------------------------------

async def test_acquire_and_release_lock(cache):
    assert await cache.acquire_lock("job", "owner-a", ttl=30) is True
    assert await cache.release_lock("job", "owner-a") is True
    # re-acquirable after release
    assert await cache.acquire_lock("job", "owner-b", ttl=30) is True


async def test_lock_contended(cache):
    assert await cache.acquire_lock("job", "owner-a", ttl=30) is True
    assert await cache.acquire_lock("job", "owner-b", ttl=30) is False


async def test_release_requires_owner(cache):
    assert await cache.acquire_lock("job", "owner-a", ttl=30) is True
    # a non-holder cannot release
    assert await cache.release_lock("job", "owner-b") is False
    # holder still holds it
    assert await cache.acquire_lock("job", "owner-c", ttl=30) is False
    assert await cache.release_lock("job", "owner-a") is True


async def test_lock_lease_expiry(cache, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    assert await cache.acquire_lock("job", "owner-a", ttl=10) is True
    now[0] += 11  # lease expires
    # another holder can now acquire
    assert await cache.acquire_lock("job", "owner-b", ttl=10) is True


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

def test_plugin_initialize_reads_maxsize():
    plugin = InMemoryCacheServicePlugin()
    v = MagicMock()
    v.environ = MagicMock(side_effect=lambda key, default=None, **kw: 256)
    svc = plugin.initialize(v, MagicMock())
    assert isinstance(svc, InMemoryCacheService)
    assert svc._maxsize == 256
    assert plugin.PROVIDER_NAME == "in-memory"
