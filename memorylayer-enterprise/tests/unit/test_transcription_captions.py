# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for context-grounded figure captioning."""

from __future__ import annotations

import base64
import io
import logging
from unittest.mock import AsyncMock

import pytest
from memorylayer_server.models.llm import LLMResponse, LLMRole

from memorylayer_saas.services.transcription.captions import (
    FigureCaptioner,
    build_figure_context,
)
from memorylayer_saas.services.transcription.cascade import CascadeTranscriber
from memorylayer_saas.services.transcription.regions import (
    PageRegion,
    parse_grounded_page,
    render_transcript,
)
from memorylayer_saas.services.transcription.result import TranscribedPage
from memorylayer_saas.services.transcription.service import DirectTranscriptionService

LOG = logging.getLogger("test")

PAGE_WITH_FIGURE = (
    "<|det|>title [113, 92, 336, 108]<|/det|>Visitor Food Service Outlets\n"
    "<|det|>text [113, 113, 614, 129]<|/det|>Below, we outline current food locations.\n"
    "<|det|>title [113, 152, 312, 168]<|/det|>The Macaw Landing Grille\n"
    "<|det|>text [112, 171, 881, 248]<|/det|>Macaw Landing Grille is a primary dining outlet.\n"
    "<|det|>image [235, 325, 460, 504]<|/det|>"
    "<|det|>text [113, 269, 378, 285]<|/det|>Location: The Macaw Landing Grille\n"
)


def _response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content, model="ml-default", prompt_tokens=7, completion_tokens=9,
        total_tokens=16, finish_reason="stop",
    )


def _chat(*contents) -> AsyncMock:
    """Stub LLMService: complete(request, profile=...) -> LLMResponse."""
    client = AsyncMock()
    client.complete = AsyncMock(
        side_effect=[_response(c) if isinstance(c, str) else c for c in contents]
    )
    return client


def _captioner(client, **kw) -> FigureCaptioner:
    return FigureCaptioner(llm_service=client, logger=LOG, **kw)


def _page_b64(width: int = 1700, height: int = 2200) -> str:
    pil_image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    pil_image.new("RGB", (width, height), (255, 255, 255)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def test_context_pulls_the_nearest_preceding_title_and_text():
    regions = parse_grounded_page(PAGE_WITH_FIGURE)
    figure_index = next(i for i, r in enumerate(regions) if r.is_figure)
    context = build_figure_context(regions, figure_index)
    assert "The Macaw Landing Grille" in context
    assert "primary dining outlet" in context


def test_context_includes_text_after_the_figure():
    """A figure's caption often follows it rather than preceding it."""
    regions = parse_grounded_page(PAGE_WITH_FIGURE)
    figure_index = next(i for i, r in enumerate(regions) if r.is_figure)
    assert "Location: The Macaw Landing Grille" in build_figure_context(regions, figure_index)


def test_context_respects_the_char_budget():
    regions = parse_grounded_page(PAGE_WITH_FIGURE)
    figure_index = next(i for i, r in enumerate(regions) if r.is_figure)
    assert len(build_figure_context(regions, figure_index, max_chars=40)) <= 80


def test_context_is_empty_when_the_page_has_no_text():
    regions = [PageRegion("image", (1, 2, 3, 4), "")]
    assert build_figure_context(regions, 0) == ""


# ---------------------------------------------------------------------------
# Captioning one figure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_caption_goes_through_the_llm_profile_with_image_and_context():
    client = _chat("Exterior of the Macaw Landing Grille dining pavilion.")
    caption = await _captioner(client).caption(b"\x89PNG-bytes", context="The Macaw Landing Grille")

    request = client.complete.await_args.args[0]
    message = request.messages[0]
    assert client.complete.await_args.kwargs["profile"] == "default"
    # Model unset -> the profile's own default model applies.
    assert request.model is None
    assert message.role == LLMRole.USER
    assert message.content.startswith("Surrounding page text:")
    assert "The Macaw Landing Grille" in message.content
    assert "at most 25 words" in message.content
    # LLMMessage.images is the vision seam; providers serialize it themselves.
    assert message.images and message.images[0] == base64.b64encode(b"\x89PNG-bytes").decode()
    assert caption == "Exterior of the Macaw Landing Grille dining pavilion."


@pytest.mark.asyncio
async def test_caption_omits_the_context_preamble_when_there_is_none():
    client = _chat("A bar chart.")
    await _captioner(client).caption(b"png")
    message = client.complete.await_args.args[0].messages[0]
    assert not message.content.startswith("Surrounding page text:")


@pytest.mark.asyncio
async def test_caption_honors_an_explicit_profile_and_model_override():
    client = _chat("A map.")
    await _captioner(client, profile="vision", model="pinned/model").caption(b"png")
    assert client.complete.await_args.kwargs["profile"] == "vision"
    assert client.complete.await_args.args[0].model == "pinned/model"


@pytest.mark.asyncio
async def test_caption_is_collapsed_to_one_clean_line():
    client = _chat('  "A photo of\n  the entrance."  ')
    assert await _captioner(client).caption(b"png") == "A photo of the entrance."


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["", "   "])
async def test_caption_returns_none_on_empty_completions(bad):
    assert await _captioner(_chat(bad)).caption(b"png") is None


@pytest.mark.asyncio
async def test_caption_returns_none_when_the_response_has_no_content():
    client = AsyncMock()
    client.complete = AsyncMock(return_value=object())
    assert await _captioner(client).caption(b"png") is None


@pytest.mark.asyncio
async def test_caption_swallows_transport_errors():
    """A caption is an enrichment; it must never fail an ingest."""
    client = AsyncMock()
    client.complete = AsyncMock(side_effect=RuntimeError("model offline"))
    assert await _captioner(client).caption(b"png") is None


# ---------------------------------------------------------------------------
# Rendering with captions
# ---------------------------------------------------------------------------


def test_caption_is_folded_into_the_figure_marker():
    regions = parse_grounded_page(PAGE_WITH_FIGURE)
    out = render_transcript(regions, figure_captions={1: "Exterior of the grille."})
    assert "[figure 1: Exterior of the grille.]" in out


def test_uncaptioned_figures_keep_the_bare_marker():
    raw = "<|det|>image [1, 2, 3, 4]<|/det|><|det|>image [5, 6, 7, 8]<|/det|>"
    out = render_transcript(parse_grounded_page(raw), figure_captions={2: "A map."})
    assert "[figure 1]" in out and "[figure 2: A map.]" in out


# ---------------------------------------------------------------------------
# Service integration
# ---------------------------------------------------------------------------


class _StubCascade(CascadeTranscriber):
    def __init__(self, page: TranscribedPage):
        super().__init__([], LOG)
        self._page = page

    async def transcribe_pages(self, images_b64, *, system_prompt=None):
        return [self._page]


def _page_from(raw: str) -> TranscribedPage:
    regions = parse_grounded_page(raw)
    return TranscribedPage(0, render_transcript(regions), "unlimited-ocr", tuple(regions))


@pytest.mark.asyncio
async def test_service_rewrites_the_transcript_with_captions():
    page = _page_from(PAGE_WITH_FIGURE)
    assert "[figure 1]" in page.content

    service = DirectTranscriptionService(
        _StubCascade(page), LOG, _captioner(_chat("Exterior of the dining pavilion.")),
    )
    out = (await service.transcribe_pages([_page_b64()]))[0]
    assert "[figure 1: Exterior of the dining pavilion.]" in out.content
    assert "## The Macaw Landing Grille" in out.content


@pytest.mark.asyncio
async def test_service_leaves_pages_without_figures_alone():
    page = _page_from("<|det|>text [1, 2, 3, 4]<|/det|>Just prose.")
    client = _chat("unused")
    service = DirectTranscriptionService(_StubCascade(page), LOG, _captioner(client))
    out = (await service.transcribe_pages([_page_b64()]))[0]
    assert out.content == page.content
    client.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_service_keeps_the_transcript_when_captioning_fails():
    page = _page_from(PAGE_WITH_FIGURE)
    client = AsyncMock()
    client.complete = AsyncMock(side_effect=RuntimeError("model offline"))
    service = DirectTranscriptionService(_StubCascade(page), LOG, _captioner(client))
    out = (await service.transcribe_pages([_page_b64()]))[0]
    assert out.content == page.content
    assert "[figure 1]" in out.content


@pytest.mark.asyncio
async def test_service_without_a_captioner_does_not_touch_pages():
    page = _page_from(PAGE_WITH_FIGURE)
    service = DirectTranscriptionService(_StubCascade(page), LOG)
    out = (await service.transcribe_pages([_page_b64()]))[0]
    assert out is page


# ---------------------------------------------------------------------------
# Thinking suppression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reasoning_is_suppressed_by_default():
    """The default profile is a THINKING model whose chain-of-thought lands in
    `content` (it returns no separate reasoning_content), so with reasoning on
    the "caption" is raw reasoning. Measured live before this default existed:
    6 of 7 captions were chain-of-thought at ~700-1024 tokens each; with "none"
    they are clean sentences at ~21. This default is the difference between
    working and garbage, so it is pinned."""
    client = _chat("A commercial kitchen.")
    await _captioner(client).caption(b"png")
    assert client.complete.await_args.args[0].reasoning_effort == "none"


@pytest.mark.asyncio
async def test_reasoning_effort_can_be_disabled_for_providers_that_reject_it():
    client = _chat("A commercial kitchen.")
    await _captioner(client, reasoning_effort=None).caption(b"png")
    assert client.complete.await_args.args[0].reasoning_effort is None


@pytest.mark.asyncio
async def test_token_budget_is_not_caption_sized():
    """A caption-sized cap truncates a still-thinking model mid-thought and
    yields chain-of-thought instead of a sentence."""
    client = _chat("A commercial kitchen.")
    await _captioner(client).caption(b"png")
    assert client.complete.await_args.args[0].max_tokens >= 256
