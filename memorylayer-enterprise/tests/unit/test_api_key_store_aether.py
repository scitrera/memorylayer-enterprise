# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the Aether-backed API key store (TI2 wire-key parity).

These import ``memorylayer_server.services.api_key_store`` from the OSS server.
That package ships in the OSS release; until the enterprise dep is bumped to a
release that includes it, run locally with the OSS src on PYTHONPATH:

    PYTHONPATH=../../oss/memorylayer-core-python/src .venv/bin/python -m pytest \
        tests/unit/test_api_key_store_aether.py
"""

import os
from unittest.mock import patch

import msgpack
import pytest
from scitrera_app_framework.api import Variables

from memorylayer_saas.services.api_key_store.aether import (
    AetherApiKeyStore,
    AetherApiKeyStorePlugin,
    _decode_value,
)


class _Resp:
    def __init__(self, value):
        self.value = value


class _FakeClient:
    def __init__(self, backing, *, raises=False):
        self._backing = backing
        self._raises = raises
        self.calls = []

    async def kv_get(self, key, scope="", user_id="", workspace="", timeout=None):
        self.calls.append((key, scope, user_id, workspace))
        if self._raises:
            raise RuntimeError("kv boom")
        return _Resp(self._backing.get(key, b""))


class _FakeAgentService:
    def __init__(self, client):
        self.client = client


def _store(tenant="acme", *, v=None, ttl=60.0):
    return AetherApiKeyStore(v, tenant=tenant, scope="global", timeout=1.0, ttl=ttl)


@pytest.mark.asyncio
async def test_reads_ti2_wire_key_and_msgpack_value():
    wire = "enc:ti:acme:ikv:api_key:OPENAI_API_KEY"
    client = _FakeClient({wire: msgpack.packb("sk-aether", use_bin_type=True)})
    store = _store()
    store.bind_client(_FakeAgentService(client))

    assert await store.get_api_key("OPENAI_API_KEY") == "sk-aether"
    # Exact TI2 read contract: enc-prefixed tenant wire key, global scope, no user/workspace.
    assert client.calls[0] == (wire, "global", "", "")


@pytest.mark.asyncio
async def test_ttl_cache_avoids_repeat_kv_calls():
    wire = "enc:ti:acme:ikv:api_key:OPENAI_API_KEY"
    client = _FakeClient({wire: msgpack.packb("sk-aether", use_bin_type=True)})
    store = _store()
    store.bind_client(_FakeAgentService(client))

    assert await store.get_api_key("OPENAI_API_KEY") == "sk-aether"
    assert await store.get_api_key("OPENAI_API_KEY") == "sk-aether"
    assert len(client.calls) == 1  # second resolve served from cache


@pytest.mark.asyncio
@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}, clear=False)
async def test_env_fallback_on_kv_miss():
    client = _FakeClient({})  # KV miss -> empty value
    store = _store(v=Variables())
    store.bind_client(_FakeAgentService(client))

    assert await store.get_api_key("OPENAI_API_KEY") == "sk-env"


@pytest.mark.asyncio
@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}, clear=False)
async def test_fail_open_to_env_on_kv_error():
    client = _FakeClient({}, raises=True)  # KV blows up
    store = _store(v=Variables())
    store.bind_client(_FakeAgentService(client))

    # Must degrade to env, never raise.
    assert await store.get_api_key("OPENAI_API_KEY") == "sk-env"


@pytest.mark.asyncio
@patch.dict(os.environ, {}, clear=True)
async def test_unresolved_returns_default():
    client = _FakeClient({})
    store = _store(v=Variables())
    store.bind_client(_FakeAgentService(client))

    assert await store.get_api_key("OPENAI_API_KEY") is None
    assert await store.get_api_key("OPENAI_API_KEY", default="fb") == "fb"


@pytest.mark.asyncio
@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}, clear=False)
async def test_no_tenant_is_env_only():
    client = _FakeClient({"enc:ti::ikv:api_key:OPENAI_API_KEY": msgpack.packb("nope")})
    store = _store(tenant=None, v=Variables())
    store.bind_client(_FakeAgentService(client))

    # No tenant -> skip KV entirely, resolve from env.
    assert await store.get_api_key("OPENAI_API_KEY") == "sk-env"
    assert client.calls == []


def test_decode_value_variants():
    assert _decode_value(msgpack.packb("sk", use_bin_type=True)) == "sk"
    assert _decode_value(b"raw-bytes") == "raw-bytes"  # defensive non-msgpack path
    assert _decode_value(None) is None


def test_plugin_provider_name_is_aether():
    assert AetherApiKeyStorePlugin().PROVIDER_NAME == "aether"
