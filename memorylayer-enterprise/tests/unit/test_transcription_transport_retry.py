"""Transport-fault retry in the transcription provider.

The retry loop originally inspected only `response.status_code`, so an
exception raised by the POST propagated straight past it: HTTP-level overload
got three attempts while a dropped connection got none. In production that
surfaced as `Transcription provider unlimited failed page N: ReadError:` with
an empty message, against a vLLM whose own logs showed clean 200s -- because
the failing requests died on a stale pooled connection and never arrived.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import httpx
import pytest

from memorylayer_saas.services.transcription.contracts import build_unlimited_ocr_contract
from memorylayer_saas.services.transcription.provider import (
    _RETRYABLE_TRANSPORT,
    OpenAIChatTranscriptionProvider,
    ProviderConfig,
)

LOG = logging.getLogger("test-transcription-transport")


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": "text"}, "finish_reason": "stop"}]},
    )


def _provider(*side_effects, attempts: int = 3) -> OpenAIChatTranscriptionProvider:
    client = AsyncMock()
    client.post = AsyncMock(side_effect=list(side_effects))
    config = ProviderConfig(
        name="unlimited",
        base_url="http://embed-proxy.gpu.svc:8101/v1",
        contract=build_unlimited_ocr_contract(),
        model="baidu/Unlimited-OCR",
        overload_attempts=attempts,
        overload_backoff_sec=0.0,  # keep the test fast
    )
    return OpenAIChatTranscriptionProvider(config, LOG, client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [
    httpx.ReadError("connection lost"),
    httpx.ConnectError("refused"),
    httpx.WriteError("broken pipe"),
    httpx.RemoteProtocolError("server disconnected"),
    httpx.ConnectTimeout("connect timed out"),
])
async def test_a_dropped_connection_is_retried(exc):
    provider = _provider(exc, _ok_response())

    response = await provider._post_with_overload_retry({"messages": []})

    assert response.status_code == 200
    assert provider._client.post.await_count == 2


@pytest.mark.asyncio
async def test_a_read_timeout_is_still_not_retried():
    """Deliberate: on a multi-minute OCR budget a timeout means real trouble,
    and retrying doubles the load that caused it."""
    provider = _provider(httpx.ReadTimeout("too slow"))

    with pytest.raises(httpx.ReadTimeout):
        await provider._post_with_overload_retry({"messages": []})

    assert provider._client.post.await_count == 1


def test_the_retryable_family_excludes_timeouts():
    # Guards the relationship rather than trusting it: if httpx reparented
    # ReadTimeout under NetworkError, slow pages would silently start retrying.
    assert issubclass(httpx.ReadError, _RETRYABLE_TRANSPORT)
    assert not issubclass(httpx.ReadTimeout, _RETRYABLE_TRANSPORT)


@pytest.mark.asyncio
async def test_persistent_transport_failure_raises_after_the_budget():
    provider = _provider(*[httpx.ReadError("gone")] * 3, attempts=3)

    with pytest.raises(httpx.ReadError):
        await provider._post_with_overload_retry({"messages": []})

    assert provider._client.post.await_count == 3


@pytest.mark.asyncio
async def test_transport_and_status_retries_share_one_budget():
    # A drop then an overload then success: both kinds count toward the same
    # attempt budget, so a flapping endpoint cannot be retried indefinitely.
    provider = _provider(
        httpx.ReadError("gone"), httpx.Response(503), _ok_response(), attempts=3,
    )

    response = await provider._post_with_overload_retry({"messages": []})

    assert response.status_code == 200
    assert provider._client.post.await_count == 3


@pytest.mark.asyncio
async def test_a_dropped_connection_clears_the_cold_start_budget():
    """The endpoint answered the socket even though it then went away; a second
    cold-start budget would stall the whole batch behind one bad connection."""
    provider = _provider(httpx.ReadError("gone"), _ok_response())
    provider.config.cold_start_timeout_sec = 600.0

    await provider._post_with_overload_retry({"messages": []})

    assert provider._first_request_done is True
