"""Unit tests for the direct (no embed-server) transcription path.

Covers the provider's request shape, cascade fallthrough across N rungs, the
on-demand cold-start budget, auth header injection, and profile parsing.
No network, no GPU: httpx is injected as a stub.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from scitrera_app_framework.api import Variables

from memorylayer_saas.services.transcription import (
    CascadeTranscriber,
    DirectTranscriptionService,
    OpenAIChatTranscriptionProvider,
    ProviderConfig,
    build_provider_config,
    parse_cascade,
)
from memorylayer_saas.services.transcription.contracts import (
    build_generic_markdown_contract,
    build_unlimited_ocr_contract,
)

LOG = logging.getLogger("test")
IMAGE_B64 = "AAAA"


def _response(content: str, *, finish_reason: str = "stop", status_error: Exception | None = None):
    response = MagicMock()
    response.raise_for_status = MagicMock(side_effect=status_error)
    response.json.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 9},
    }
    return response


def _client(*responses):
    client = AsyncMock()
    client.post = AsyncMock(side_effect=list(responses))
    return client


def _provider(client, *, contract=None, name="unlimited", **overrides) -> OpenAIChatTranscriptionProvider:
    config = ProviderConfig(
        name=name,
        base_url="http://embed-proxy.gpu.svc:8006/v1",
        contract=contract or build_unlimited_ocr_contract(),
        model="baidu/Unlimited-OCR",
        **overrides,
    )
    return OpenAIChatTranscriptionProvider(config, LOG, client=client)


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_carries_the_unlimited_ocr_recipe():
    client = _client(_response("<|ref|>Hello<|/ref|><|det|>[[0,0,1,1]]<|/det|>"))
    attempt = await _provider(client).transcribe_page(IMAGE_B64)

    path, kwargs = client.post.call_args[0][0], client.post.call_args[1]
    payload = kwargs["json"]
    assert path == "/chat/completions"
    assert payload["model"] == "baidu/Unlimited-OCR"
    # Temperature is OMITTED so the model's serving default applies. It used to
    # be pinned to 0.0 -- greedy decoding, the most repetition-prone setting --
    # on a model that demonstrably loops on dense pages.
    assert "temperature" not in payload
    assert payload["skip_special_tokens"] is False
    assert payload["vllm_xargs"] == {"ngram_size": 35, "window_size": 1024}
    parts = payload["messages"][0]["content"]
    assert parts[0]["text"] == "<image>document parsing."
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,AAAA"

    assert attempt.success is True
    assert attempt.content == "Hello"
    assert attempt.tokens_in == 7 and attempt.tokens_out == 9


@pytest.mark.asyncio
async def test_already_formed_data_url_is_not_double_wrapped():
    client = _client(_response("text"))
    await _provider(client).transcribe_page("data:image/jpeg;base64,ZZZZ")
    parts = client.post.call_args[1]["json"]["messages"][0]["content"]
    assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,ZZZZ"


@pytest.mark.asyncio
async def test_generic_contract_sends_no_unlimited_ocr_fields():
    client = _client(_response("# Page"))
    await _provider(client, contract=build_generic_markdown_contract()).transcribe_page(IMAGE_B64)
    payload = client.post.call_args[1]["json"]
    assert "skip_special_tokens" not in payload
    assert "vllm_xargs" not in payload


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_output_is_a_failure_not_a_blank_page():
    """For a fixed-recipe model this means the contract drifted."""
    attempt = await _provider(_client(_response("<|det|>[[1,2,3,4]]<|/det|>"))).transcribe_page(IMAGE_B64)
    assert attempt.success is False
    assert attempt.error == "Empty content after cleaning"


@pytest.mark.asyncio
async def test_length_finish_reason_is_a_failure():
    attempt = await _provider(_client(_response("partial", finish_reason="length"))).transcribe_page(IMAGE_B64)
    assert attempt.success is False
    assert "Token limit reached" in attempt.error


@pytest.mark.asyncio
async def test_transport_error_is_captured_not_raised():
    client = AsyncMock()
    client.post = AsyncMock(side_effect=RuntimeError("connection refused"))
    attempt = await _provider(client).transcribe_page(IMAGE_B64)
    assert attempt.success is False
    assert "connection refused" in attempt.error


# ---------------------------------------------------------------------------
# Cold start (on-demand endpoints)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_request_uses_the_cold_start_budget_then_steady_state():
    """An on-demand endpoint boots a GPU on the first call; later calls are fast."""
    client = _client(_response("a"), _response("b"))
    provider = _provider(client, timeout_sec=30.0, cold_start_timeout_sec=600.0)

    await provider.transcribe_page(IMAGE_B64)
    assert client.post.call_args[1]["timeout"] == 600.0

    await provider.transcribe_page(IMAGE_B64)
    assert client.post.call_args[1]["timeout"] == 30.0


@pytest.mark.asyncio
async def test_a_failed_first_request_still_consumes_the_cold_start_budget():
    """Otherwise every failure would re-arm the multi-minute timeout."""
    client = AsyncMock()
    client.post = AsyncMock(side_effect=[RuntimeError("boom"), _response("ok")])
    provider = _provider(client, timeout_sec=30.0, cold_start_timeout_sec=600.0)

    await provider.transcribe_page(IMAGE_B64)
    await provider.transcribe_page(IMAGE_B64)
    assert client.post.call_args[1]["timeout"] == 30.0


@pytest.mark.asyncio
async def test_without_cold_start_every_request_uses_the_steady_budget():
    client = _client(_response("a"))
    await _provider(client, timeout_sec=45.0).transcribe_page(IMAGE_B64)
    assert client.post.call_args[1]["timeout"] == 45.0


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_auth_headers_per_kind():
    base = dict(name="p", base_url="http://x/v1", contract=build_generic_markdown_contract())
    assert ProviderConfig(**base).auth_headers() == {}
    assert ProviderConfig(**base, auth_kind="bearer", auth_token="t").auth_headers() == {
        "Authorization": "Bearer t"
    }
    assert ProviderConfig(
        **base, auth_kind="modal_proxy", auth_key="k", auth_secret="s"
    ).auth_headers() == {"Modal-Key": "k", "Modal-Secret": "s"}


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cascade_falls_through_to_the_next_rung():
    first = _provider(_client(_response("")), name="unlimited")
    second = _provider(
        _client(_response("# Recovered")),
        name="gemini",
        contract=build_generic_markdown_contract(),
    )
    pages = await CascadeTranscriber([first, second], LOG).transcribe_pages([IMAGE_B64])
    assert [p.content for p in pages] == ["# Recovered"]


@pytest.mark.asyncio
async def test_cascade_stops_at_the_first_success():
    second_client = _client(_response("unused"))
    cascade = CascadeTranscriber(
        [_provider(_client(_response("# Good"))), _provider(second_client, name="gemini")], LOG,
    )
    await cascade.transcribe_pages([IMAGE_B64])
    second_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_page_that_exhausts_every_rung_is_omitted():
    """Callers must leave it transcript=None, not persist a failure marker."""
    cascade = CascadeTranscriber(
        [_provider(_client(_response(""))), _provider(_client(_response("")), name="gemini")], LOG,
    )
    assert await cascade.transcribe_pages([IMAGE_B64]) == []


@pytest.mark.asyncio
async def test_one_bad_page_does_not_drop_the_rest_of_the_batch():
    provider = _provider(_client(_response("A"), _response(""), _response("C")))
    pages = await CascadeTranscriber([provider], LOG).transcribe_pages([IMAGE_B64] * 3)
    assert [(p.request_index, p.content) for p in pages] == [(0, "A"), (2, "C")]


@pytest.mark.asyncio
async def test_indices_are_request_relative():
    provider = _provider(_client(_response("A"), _response("B")))
    pages = await CascadeTranscriber([provider], LOG).transcribe_pages([IMAGE_B64] * 2)
    assert [p.request_index for p in pages] == [0, 1]


@pytest.mark.asyncio
async def test_direct_service_reports_its_cascade_and_ignores_max_tokens():
    provider = _provider(_client(_response("A")))
    service = DirectTranscriptionService(CascadeTranscriber([provider], LOG), LOG)
    assert service.provider_names == ["unlimited"]
    # max_tokens is per-provider; passing it must not blow up the shared signature.
    pages = await service.transcribe_pages([IMAGE_B64], max_tokens=99)
    assert [p.content for p in pages] == ["A"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_parse_cascade_preserves_order_and_drops_blanks():
    assert parse_cascade(" unlimited , , gemini ") == ["unlimited", "gemini"]
    assert parse_cascade("") == []


def _vars(**env) -> Variables:
    v = Variables()
    for key, value in env.items():
        v.set(key, value)
    return v


def test_build_provider_config_reads_the_profile_keys():
    v = _vars(**{
        "MEMORYLAYER_TRANSCRIBE_PROFILE_UNLIMITED_URL": "http://embed-proxy.gpu.svc:8006/v1",
        "MEMORYLAYER_TRANSCRIBE_PROFILE_UNLIMITED_CONTRACT": "unlimited_ocr",
        "MEMORYLAYER_TRANSCRIBE_PROFILE_UNLIMITED_MODEL": "baidu/Unlimited-OCR",
        "MEMORYLAYER_TRANSCRIBE_PROFILE_UNLIMITED_WINDOW_SIZE": "1024",
        "MEMORYLAYER_TRANSCRIBE_PROFILE_UNLIMITED_COLD_START_TIMEOUT_SEC": "600",
    })
    config = build_provider_config(v, "unlimited")
    assert config.base_url == "http://embed-proxy.gpu.svc:8006/v1"
    assert config.model == "baidu/Unlimited-OCR"
    assert config.contract.extra_body["vllm_xargs"]["window_size"] == 1024
    assert config.cold_start_timeout_sec == 600.0


def test_profile_without_url_fails_at_configuration_time():
    """A silently short cascade is worse than a boot failure."""
    with pytest.raises(ValueError, match="has no MEMORYLAYER_TRANSCRIBE_PROFILE_GEMINI_URL"):
        build_provider_config(_vars(), "gemini")


def test_profile_with_unknown_auth_is_rejected():
    v = _vars(**{
        "MEMORYLAYER_TRANSCRIBE_PROFILE_X_URL": "http://x/v1",
        "MEMORYLAYER_TRANSCRIBE_PROFILE_X_AUTH": "oauth",
    })
    with pytest.raises(ValueError, match="unknown auth"):
        build_provider_config(v, "x")


@pytest.mark.asyncio
async def test_temperature_is_sent_when_explicitly_configured():
    """Omitted by default, but still settable -- the point is to stop pinning
    greedy decoding, not to make sampling unreachable."""
    client = _client(_response("text"))
    provider = _provider(client, temperature=0.2)

    await provider.transcribe_page("aGVsbG8=")

    assert client.post.call_args[1]["json"]["temperature"] == 0.2
