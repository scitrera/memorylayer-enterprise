"""The transcription seam the ingestion pipeline talks to.

Two implementations, one contract:

``embed_server``
    Delegates to the embed-server REST client (``POST /v1/transcribe``) and
    normalizes the response. This is today's behavior and the OSS default; it
    requires an embed-server process.

``direct``
    Runs the cascade in-process against model endpoints served by sparkrun
    recipes (or any OpenAI-compatible endpoint). **No embed-server.**

Both return ``list[TranscribedPage]``, so the ingestion paths never learn which
one is configured.
"""

from __future__ import annotations

import asyncio
import base64
from abc import ABC, abstractmethod
from dataclasses import replace
from logging import Logger

from .captions import FigureCaptioner
from .cascade import CascadeTranscriber
from .figures import crop_figures
from .regions import render_transcript
from .result import TranscribedPage, pages_from_embed_server_response


class TranscriptionService(ABC):
    """Transcribe a batch of page images to markdown."""

    async def connect(self) -> None:
        """Open any upstream connections. Idempotent."""
        return

    async def close(self) -> None:
        """Release upstream connections. Idempotent."""
        return

    @abstractmethod
    async def transcribe_pages(
        self, images_b64: list[str], *, system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> list[TranscribedPage]:
        """Return the successfully transcribed pages of this batch.

        Indices are relative to ``images_b64``, not to the document. Pages that
        could not be transcribed are omitted rather than represented by a
        failure placeholder.
        """


class EmbedServerTranscriptionService(TranscriptionService):
    """Transcription via an embed-server ``POST /v1/transcribe`` cascade."""

    def __init__(self, embed_client, logger: Logger):
        self._embed = embed_client
        self.logger = logger

    async def connect(self) -> None:
        await self._embed.connect()

    async def close(self) -> None:
        await self._embed.close()

    async def transcribe_pages(
        self, images_b64: list[str], *, system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> list[TranscribedPage]:
        result = await self._embed.transcribe_pages(
            images_b64, system_prompt=system_prompt, max_tokens=max_tokens,
        )
        return pages_from_embed_server_response(result)


class DirectTranscriptionService(TranscriptionService):
    """Transcription run in-process against model endpoints.

    ``max_tokens`` is deliberately ignored: each cascade rung carries its own
    budget, because a per-model token cap is part of that model's serving
    contract rather than a per-request preference. The parameter stays in the
    signature so the two implementations remain substitutable.

    When a ``captioner`` is supplied, each page's figures are cropped and
    captioned with the multimodal model and the transcript is re-rendered with
    those captions. That turns a bare ``[figure 1]`` into a statement of what
    the illustration depicts, which is the difference between a consumer having
    to fetch every crop and being able to decide from the text.
    """

    def __init__(
        self,
        cascade: CascadeTranscriber,
        logger: Logger,
        captioner: FigureCaptioner | None = None,
    ):
        self._cascade = cascade
        self._captioner = captioner
        self.logger = logger

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self._cascade.providers]

    async def connect(self) -> None:
        await self._cascade.connect()

    async def close(self) -> None:
        await self._cascade.close()

    async def transcribe_pages(
        self, images_b64: list[str], *, system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> list[TranscribedPage]:
        del max_tokens  # per-provider; see the class docstring
        pages = await self._cascade.transcribe_pages(images_b64, system_prompt=system_prompt)
        if self._captioner is None:
            return pages
        # Pages captioned in parallel; each page's figures are bounded by the
        # captioner's own semaphore, so total LLM concurrency stays capped.
        return list(
            await asyncio.gather(*(self._caption_page(page, images_b64) for page in pages))
        )

    async def _caption_page(
        self, page: TranscribedPage, images_b64: list[str],
    ) -> TranscribedPage:
        """Re-render one page's transcript with figure captions.

        Any failure returns the page untouched: a caption is an enrichment, and
        losing it must never cost the transcript.
        """
        if not page.figures or not (0 <= page.request_index < len(images_b64)):
            return page
        try:
            crops = crop_figures(
                base64.b64decode(images_b64[page.request_index]), list(page.regions),
            )
            if not crops:
                return page
            captions = await self._captioner.caption_page_figures(page.regions, crops)
            if not captions:
                return page
            return replace(
                page,
                content=render_transcript(list(page.regions), figure_captions=captions),
                figure_captions=captions,
            )
        except Exception as e:  # noqa: BLE001 — enrichment must not fail an ingest
            self.logger.warning(
                "Figure captioning failed for page %d: %s", page.request_index, e,
            )
            return page
