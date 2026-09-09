# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for bounded concurrency and overload backoff on the direct path.

Two invariants matter here and neither is visible from a single-request test:

* pages run in parallel but never more than ``concurrency`` at once -- the lane
  is one GPU shared with other lanes, and fleet load is (workers x this);
* an upstream that says "busy" is retried, because the proxy in front of the
  lane has a FAIL-FAST bounded queue: without a retry a 503 at the ceiling
  becomes a silently dropped page rather than backpressure.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_saas.services.transcription.cascade import CascadeTranscriber
from memorylayer_saas.services.transcription.contracts import build_unlimited_ocr_contract
from memorylayer_saas.services.transcription.provider import (
    OpenAIChatTranscriptionProvider,
    ProviderConfig,
)

LOG = logging.getLogger("test")
IMAGE = "AAAA"


def _ok(content: str = "<|det|>text [1, 2, 3, 4]<|/det|>page text"):
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }
    return response


def _busy(status: int = 503, headers: dict | None = None):
    response = MagicMock()
    response.status_code = status
    response.headers = headers or {"x-envoy-overloaded": "true"}
    response.raise_for_status = MagicMock()
    response.json.return_value = {}
    return response


def _provider(client, **overrides) -> OpenAIChatTranscriptionProvider:
    defaults = dict(
        name="unlimited", base_url="http://proxy:8101/v1",
        contract=build_unlimited_ocr_contract(), model="baidu/Unlimited-OCR",
        overload_backoff_sec=0.001,   # keep retry tests fast; overridable below
    )
    defaults.update(overrides)
    return OpenAIChatTranscriptionProvider(ProviderConfig(**defaults), LOG, client=client)


# ---------------------------------------------------------------------------
# Bounded parallelism
# ---------------------------------------------------------------------------


class _TrackingClient:
    """Records peak simultaneous in-flight requests."""

    def __init__(self, delay: float = 0.02):
        self.delay = delay
        self.in_flight = 0
        self.peak = 0
        self.total = 0

    async def post(self, *_args, **_kwargs):
        self.in_flight += 1
        self.total += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            return _ok()
        finally:
            self.in_flight -= 1

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_pages_run_in_parallel_up_to_the_limit():
    client = _TrackingClient()
    cascade = CascadeTranscriber([_provider(client)], LOG, concurrency=4)
    pages = await cascade.transcribe_pages([IMAGE] * 12)
    assert len(pages) == 12
    assert client.total == 12
    assert client.peak == 4, "expected exactly `concurrency` in flight, saw %d" % client.peak


@pytest.mark.asyncio
async def test_concurrency_one_is_strictly_sequential():
    client = _TrackingClient()
    cascade = CascadeTranscriber([_provider(client)], LOG, concurrency=1)
    await cascade.transcribe_pages([IMAGE] * 5)
    assert client.peak == 1


@pytest.mark.asyncio
async def test_concurrency_is_floored_at_one():
    """A misconfigured 0 must not deadlock the batch."""
    assert CascadeTranscriber([], LOG, concurrency=0).concurrency == 1


@pytest.mark.asyncio
async def test_parallel_pages_keep_request_relative_indices():
    """Indices address the caller's batch; gather must not reorder them."""
    client = _TrackingClient()
    cascade = CascadeTranscriber([_provider(client)], LOG, concurrency=4)
    pages = await cascade.transcribe_pages([IMAGE] * 6)
    assert [p.request_index for p in pages] == [0, 1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_one_failing_page_does_not_fail_the_batch():
    """Failure isolation must survive the move to gather."""
    # Middle page returns nothing usable. NOTE a figure-only page would NOT do:
    # it renders as "[figure 1]", which is real content and a real success.
    responses = [_ok(), _ok(""), _ok()]
    client = AsyncMock()
    client.post = AsyncMock(side_effect=responses)
    pages = await CascadeTranscriber([_provider(client)], LOG, concurrency=3).transcribe_pages([IMAGE] * 3)
    assert [p.request_index for p in pages] == [0, 2]


# ---------------------------------------------------------------------------
# Overload backoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overloaded_upstream_is_retried_then_succeeds():
    """503 at the proxy ceiling must cost latency, not a page."""
    client = AsyncMock()
    client.post = AsyncMock(side_effect=[_busy(), _busy(), _ok()])
    attempt = await _provider(client, overload_attempts=3).transcribe_page(IMAGE)
    assert attempt.success is True
    assert client.post.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 502, 503, 504])
async def test_all_busy_statuses_are_retried(status):
    client = AsyncMock()
    client.post = AsyncMock(side_effect=[_busy(status), _ok()])
    attempt = await _provider(client, overload_attempts=2).transcribe_page(IMAGE)
    assert attempt.success is True and client.post.await_count == 2


@pytest.mark.asyncio
async def test_retries_are_bounded_and_the_failure_surfaces():
    client = AsyncMock()
    client.post = AsyncMock(side_effect=[_busy(), _busy(), _busy(), _ok()])
    attempt = await _provider(client, overload_attempts=3).transcribe_page(IMAGE)
    assert client.post.await_count == 3, "must stop at overload_attempts"
    assert attempt.success is False


@pytest.mark.asyncio
async def test_a_normal_error_is_not_retried():
    """A 400 is the request's fault; retrying just multiplies load."""
    bad = _busy(400, headers={})
    client = AsyncMock()
    client.post = AsyncMock(side_effect=[bad, _ok()])
    await _provider(client, overload_attempts=3).transcribe_page(IMAGE)
    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_success_makes_no_extra_calls():
    client = AsyncMock()
    client.post = AsyncMock(side_effect=[_ok()])
    attempt = await _provider(client, overload_attempts=3).transcribe_page(IMAGE)
    assert attempt.success is True and client.post.await_count == 1


def test_backoff_honors_retry_after_then_falls_back_to_jittered_exponential():
    provider = _provider(AsyncMock(), overload_backoff_sec=1.0)
    assert provider._backoff_delay(1, _busy(503, {"retry-after": "7"})) == 7.0
    # Unparseable Retry-After must not raise; fall through to the computed delay.
    # base * (0.5 + random()) -> attempt 1 spans [0.5, 1.5).
    assert 0.5 <= provider._backoff_delay(1, _busy(503, {"retry-after": "soon"})) < 1.5
    # Exponential in the attempt number: attempt 2 spans [1.0, 3.0).
    assert 1.0 <= provider._backoff_delay(2, _busy(503, {})) < 3.0
    # Jittered, so repeated calls must not all collide on one value.
    assert len({round(provider._backoff_delay(2, _busy(503, {})), 6) for _ in range(12)}) > 1
