# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for persisting figures cropped out of a page render."""

from __future__ import annotations

import base64
import io
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_saas.services.document.page_figures import (
    PAGE_FIGURES_METADATA_KEY,
    store_page_figures,
)
from memorylayer_saas.services.transcription.regions import PageRegion, parse_grounded_page

LOG = logging.getLogger("test")
WS, DOC = "ws_1", "doc_1"


def _page_b64(width: int = 1700, height: int = 2200) -> str:
    pil_image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    pil_image.new("RGB", (width, height), (255, 255, 255)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _blob() -> MagicMock:
    blob = MagicMock()
    blob.store_file = AsyncMock()
    blob.page_figure_path = MagicMock(
        side_effect=lambda ws, doc, page, fig: "/b/%s/%s/figures/page_%04d_fig_%02d.png" % (ws, doc, page, fig)
    )
    return blob


async def _store(regions, blob=None, page_b64=None, captions=None):
    blob = blob or _blob()
    return blob, await store_page_figures(
        blob_storage=blob, workspace_id=WS, doc_id=DOC, page_no=7,
        page_image_b64=page_b64 or _page_b64(), regions=regions, logger=LOG,
        captions=captions,
    )


@pytest.mark.asyncio
async def test_stores_one_blob_per_figure_numbered_from_one():
    """figure_no is 1-based so it lines up with the [figure N] transcript marker."""
    regions = parse_grounded_page(
        "<|det|>image [235, 325, 460, 504]<|/det|>"
        "<|det|>text [1, 2, 300, 400]<|/det|>body"
        "<|det|>image [468, 325, 765, 504]<|/det|>"
    )
    blob, stored = await _store(regions)
    assert [r["storage_path"] for r in stored] == [
        "/b/ws_1/doc_1/figures/page_0007_fig_01.png",
        "/b/ws_1/doc_1/figures/page_0007_fig_02.png",
    ]
    assert [r["figure_no"] for r in stored] == [1, 2]
    assert blob.store_file.await_count == 2
    assert blob.store_file.await_args_list[0].args[1][:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.asyncio
async def test_pages_without_figures_do_no_work():
    blob, stored = await _store(parse_grounded_page("<|det|>text [1, 2, 3, 4]<|/det|>body"))
    assert stored == []
    blob.store_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_corrupt_page_image_does_not_fail_the_ingest():
    """Figures are a bonus; losing them must never cost the document."""
    regions = [PageRegion("image", (100, 100, 400, 400), "")]
    blob, stored = await _store(regions, page_b64=base64.b64encode(b"not-a-png").decode())
    assert stored == []
    blob.store_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_failed_store_does_not_abort_the_others():
    regions = parse_grounded_page(
        "<|det|>image [100, 100, 400, 400]<|/det|><|det|>image [500, 500, 800, 800]<|/det|>"
    )
    blob = _blob()
    blob.store_file = AsyncMock(side_effect=[OSError("disk full"), None])
    _, stored = await _store(regions, blob=blob)
    assert [r["storage_path"] for r in stored] == ["/b/ws_1/doc_1/figures/page_0007_fig_02.png"]
    assert [r["figure_no"] for r in stored] == [2], "numbering must not shift when one store fails"


# ---------------------------------------------------------------------------
# Persistable records — what makes a stored crop discoverable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_carries_bbox_and_caption():
    """A stored blob nobody can enumerate is unfetchable, so the record is the
    feature: bbox locates the figure on the page, caption lets a consumer skip
    the fetch entirely."""
    regions = parse_grounded_page("<|det|>image [235, 325, 460, 504]<|/det|>")
    _, stored = await _store(regions, captions={1: "Exterior of the grille."})
    assert stored == [{
        "figure_no": 1,
        "storage_path": "/b/ws_1/doc_1/figures/page_0007_fig_01.png",
        "bbox": [235, 325, 460, 504],
        "caption": "Exterior of the grille.",
    }]


@pytest.mark.asyncio
async def test_uncaptioned_figure_still_gets_a_record():
    regions = parse_grounded_page("<|det|>image [235, 325, 460, 504]<|/det|>")
    _, stored = await _store(regions)
    assert stored[0]["caption"] is None
    assert stored[0]["bbox"] == [235, 325, 460, 504]


def test_metadata_key_is_stable():
    """The API route reads figures out of page metadata under this key; a rename
    would silently 404 every figure fetch."""
    assert PAGE_FIGURES_METADATA_KEY == "figures"
