"""Ordered fallback across N transcription providers."""

from __future__ import annotations

import asyncio
from logging import Logger

from .provider import (
    DEFAULT_TRANSCRIBE_CONCURRENCY,
    OpenAIChatTranscriptionProvider,
    TranscriptionAttempt,
)
from .result import TranscribedPage


class CascadeTranscriber:
    """Try each provider in order per page; first success wins.

    The cascade is per **page**, not per batch: one hard page falling through to
    the next rung must not drag the rest of the document with it.

    Pages that exhaust every provider are simply absent from the returned list —
    the same contract as :func:`.result.pages_from_embed_server_response`, so
    callers leave those pages with ``transcript=None`` instead of persisting a
    failure marker.
    """

    def __init__(
        self,
        providers: list[OpenAIChatTranscriptionProvider],
        logger: Logger,
        concurrency: int = DEFAULT_TRANSCRIBE_CONCURRENCY,
    ):
        self.providers = providers
        self.logger = logger
        #: Pages in flight at once. Bounded rather than unlimited because the
        #: lane is one GPU shared with other lanes, and because fleet load is
        #: (worker replicas x this), which the proxy ceiling has to cover.
        self.concurrency = max(1, int(concurrency))

    async def connect(self) -> None:
        for provider in self.providers:
            await provider.connect()

    async def close(self) -> None:
        for provider in self.providers:
            try:
                await provider.close()
            except Exception as e:  # noqa: BLE001 — close must not mask real errors
                self.logger.warning("Closing transcription provider %s failed: %s", provider.name, e)

    async def transcribe_page(
        self, image_b64: str, *, request_index: int, system_prompt: str | None = None,
    ) -> tuple[TranscribedPage | None, list[TranscriptionAttempt]]:
        attempts: list[TranscriptionAttempt] = []
        for provider in self.providers:
            attempt = await provider.transcribe_page(image_b64, system_prompt=system_prompt)
            attempts.append(attempt)
            if attempt.success:
                self.logger.debug(
                    "Page %d transcribed by %s (%s) in %.0fms",
                    request_index, attempt.provider, attempt.model, attempt.latency_ms,
                )
                return (
                    TranscribedPage(
                        request_index=request_index,
                        content=attempt.content,
                        model=attempt.model or None,
                        regions=tuple(attempt.regions),
                    ),
                    attempts,
                )
            self.logger.info(
                "Transcription provider %s failed page %d: %s",
                attempt.provider, request_index, attempt.error,
            )
        return None, attempts

    async def transcribe_pages(
        self, images_b64: list[str], *, system_prompt: str | None = None,
    ) -> list[TranscribedPage]:
        """Transcribe a batch, indexed by position **within this request**.

        Pages run concurrently up to ``concurrency``. Each page walks the
        cascade independently, so failure isolation is unchanged; ``gather``
        preserves order, so ``request_index`` still lines up with the caller's
        batch.
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run(index: int, image_b64: str):
            async with semaphore:
                return await self.transcribe_page(
                    image_b64, request_index=index, system_prompt=system_prompt,
                )

        results = await asyncio.gather(
            *(run(index, b64) for index, b64 in enumerate(images_b64))
        )

        pages: list[TranscribedPage] = []
        failed = 0
        for index, (page, attempts) in enumerate(results):
            if page is not None:
                pages.append(page)
            else:
                failed += 1
                self.logger.warning(
                    "All %d transcription provider(s) failed page %d: %s",
                    len(attempts), index,
                    "; ".join("%s=%s" % (a.provider, a.error) for a in attempts),
                )
        if failed:
            self.logger.warning(
                "Transcribed %d/%d pages in this batch", len(pages), len(images_b64),
            )
        return pages
