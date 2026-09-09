# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Normalization of the embed-server ``POST /v1/transcribe`` response.

The wire shape is the embed-server's ``TranscriptionResponse``::

    {"results": [{"page_index": 0, "content": "...", "success": true,
                  "model_used": "...", "provider_used": "...", "attempts": [...]}],
     "stats": {...}}

Both ingestion paths (the ``document_transcribe`` task and the monolithic
``IngestionService._transcribe_pages``) previously dug into that dict inline,
and both read a shape the server has never emitted -- ``result["pages"]`` with
``page_number`` / ``model`` keys. ``dict.get("pages", [])`` yields an empty
list, so the loop body never ran and **no page was ever assigned a transcript**,
silently: no exception, no warning, just ``0/N pages transcribed``. Both unit
suites mocked ``transcribe_pages`` with the same imagined shape, so the seam
drifted with green tests.

Normalizing in one place is the fix and the guard: the wire shape is now read
once, and the callers consume typed records.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .regions import PageRegion


@dataclass(frozen=True)
class TranscribedPage:
    """One successfully transcribed page.

    ``request_index`` is the page's position **within the request batch**, not
    its ``page_no`` in the document -- the embed server numbers pages 0-based
    per call, and pages are sent in fixed-size batches, so the two diverge after
    the first batch.

    ``regions`` is the page's layout segmentation when the provider produced one
    (grounded OCR models do; the embed-server path does not, since that server
    returns already-flattened text). It is retained so callers can act on
    structure the transcript cannot carry -- most importantly the figure boxes,
    which are the only handle on illustrations the model emits no text for.
    """

    request_index: int
    content: str
    model: str | None
    regions: tuple[PageRegion, ...] = field(default_factory=tuple)
    #: Caption per 1-based figure number, when captioning ran. Kept alongside
    #: the transcript (rather than only inside its ``[figure N: ...]`` marker)
    #: so the persisted figure record can carry it without re-parsing text.
    figure_captions: dict[int, str] = field(default_factory=dict)

    @property
    def figures(self) -> tuple[PageRegion, ...]:
        """Figure regions, in page order. Croppable via :mod:`.figures`."""
        return tuple(r for r in self.regions if r.is_figure)


def pages_from_embed_server_response(result: dict) -> list[TranscribedPage]:
    """Extract the successfully transcribed pages from a ``/v1/transcribe`` body.

    Pages the cascade failed on are **omitted** rather than returned. On total
    failure the server substitutes the literal string
    ``"**Transcription Failed for this page**"`` as ``content`` and sets
    ``success: false``; persisting that as a page transcript would put a failure
    marker into blob storage, the page record, and every memory derived from it.
    Omitting it leaves ``transcript`` as ``None`` -- the same state as
    "not transcribed yet", which the downstream image-embed and gap-analysis
    paths already handle.

    Malformed or non-dict entries are skipped rather than raising: one bad page
    in a batch should not fail the whole document.
    """
    pages: list[TranscribedPage] = []
    for entry in result.get("results") or ():
        if not isinstance(entry, dict) or not entry.get("success"):
            continue
        index = entry.get("page_index")
        content = entry.get("content")
        if not isinstance(index, int) or not content:
            continue
        pages.append(
            TranscribedPage(
                request_index=index,
                content=content,
                model=entry.get("model_used"),
            )
        )
    return pages
