# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for Aether service registration (mocked transport)."""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_connectors.server._asgi_bridge import asgi_dispatch
from data_connectors.server.aether_service import (
    _SERVICE_IMPLEMENTATION,
    _TERMINATOR_ALLOW_PATHS,
    AetherServiceRegistration,
    service_topic,
)


class TestAetherServiceRegistration:
    """Verify Aether service registration with mocked SDK."""

    def test_initial_state(self):
        svc = AetherServiceRegistration()
        assert svc.client is None

    @pytest.mark.asyncio
    async def test_connect_creates_client(self):
        mock_app = MagicMock()
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.close = AsyncMock()

        mock_terminator = AsyncMock()
        mock_terminator.start = AsyncMock()
        mock_terminator.stop = AsyncMock()

        sdk_module = ModuleType("scitrera_aether_client")
        sdk_module.AsyncServiceClient = MagicMock(return_value=mock_client)
        terminator_module = ModuleType("scitrera_aether_client.proxy_terminator")
        terminator_module.ProxyHttpTerminator = MagicMock(return_value=mock_terminator)

        with (
            patch("data_connectors.server.aether_service.os") as mock_os,
            patch.dict(sys.modules, {
                "scitrera_aether_client": sdk_module,
                "scitrera_aether_client.proxy_terminator": terminator_module,
            }),
            patch("data_connectors.server._asgi_bridge.asgi_dispatch", new_callable=AsyncMock),
        ):
            mock_os.environ = {
                "AETHER_GATEWAY_ADDR": "localhost:50051",
                "AETHER_AUTH": "none",
            }
            mock_os.path.isfile = MagicMock(return_value=False)

            svc = AetherServiceRegistration()
            await svc.connect(mock_app)

            # Verify client was created with correct implementation
            sdk_module.AsyncServiceClient.assert_called_once()
            call_kwargs = sdk_module.AsyncServiceClient.call_args.kwargs
            assert call_kwargs["implementation"] == _SERVICE_IMPLEMENTATION
            assert call_kwargs["implementation"] == "data-connectors"

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        svc = AetherServiceRegistration()
        mock_client = AsyncMock()
        mock_terminator = AsyncMock()
        svc._client = mock_client
        svc._terminator = mock_terminator

        await svc.disconnect()

        mock_terminator.stop.assert_awaited_once()
        mock_client.close.assert_awaited_once()
        assert svc._client is None
        assert svc._terminator is None

    @pytest.mark.asyncio
    async def test_disconnect_idempotent(self):
        svc = AetherServiceRegistration()
        await svc.disconnect()  # Should not raise

    def test_service_implementation_name(self):
        assert _SERVICE_IMPLEMENTATION == "data-connectors"

    def test_terminator_allow_paths(self):
        assert "/v1/*" in _TERMINATOR_ALLOW_PATHS
        assert "/healthz" in _TERMINATOR_ALLOW_PATHS

    def test_service_topic_uses_exact_registered_specifier(self, monkeypatch):
        monkeypatch.setenv("DC_SERVICE_SPECIFIER", "pod-a")
        assert service_topic() == "sv::data-connectors::pod-a"


@pytest.mark.asyncio
async def test_asgi_bridge_exposes_receipt_only_as_private_scope_metadata():
    receipt = object()
    observed = {}

    async def app(scope, receive, send):
        observed.update(scope)
        await receive()
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    request = MagicMock()
    request.method = "GET"
    request.path = "/v1/vfs/entries/ref-1"
    request.query = ""
    request.headers = {}
    request.body = b""
    request.access_receipt = receipt

    status_code, _, _ = await asgi_dispatch(app, request)
    assert status_code == 204
    assert observed["aether.access_receipt"] is receipt
    assert observed["headers"] == []
