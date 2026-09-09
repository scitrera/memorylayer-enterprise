# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for grounded-layout parsing, rendering policy, and figure crops.

Shapes here are taken from real Unlimited-OCR output captured against the live
lane, not from the model card -- the card documents
``<|ref|>text<|/ref|><|det|>[box]<|/det|>`` pairs, and the model actually emits
``<|det|>label [box]<|/det|>text``. Both are handled; only the second occurs in
practice.
"""

from __future__ import annotations

import io

import pytest

from memorylayer_saas.services.transcription.figures import bbox_to_pixels, crop_figures
from memorylayer_saas.services.transcription.regions import (
    BBOX_SCALE,
    PageRegion,
    extract_grounded_page,
    parse_grounded_page,
    render_transcript,
)

# Verbatim structure of a real page: running headers, a figure that emits no
# text, a heading, body prose, an HTML table, and a page number.
REAL_PAGE = (
    "<|det|>header [115, 47, 244, 62]<|/det|>JGL Consultants\n"
    "<|det|>header [636, 47, 884, 63]<|/det|>Tulsa Zoo Request for Proposal\n"
    "<|det|>image [143, 90, 855, 282]<|/det|>"
    "<|det|>title [114, 305, 339, 319]<|/det|>Historic Visitor Food Revenue\n"
    "<|det|>text [113, 324, 839, 357]<|/det|>Below, we provide revenue data.\n"
    "<|det|>table [290, 361, 709, 604]<|/det|>"
    "<table><tr><td colspan=\"4\">Sales</td></tr></table>\n"
    "<|det|>page_number [489, 938, 511, 952]<|/det|>12"
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parses_the_real_emission_format():
    regions = parse_grounded_page(REAL_PAGE)
    assert [r.label for r in regions] == [
        "header", "header", "image", "title", "text", "table", "page_number",
    ]


def test_bboxes_are_captured_for_every_region():
    regions = parse_grounded_page(REAL_PAGE)
    assert regions[0].bbox == (115, 47, 244, 62)
    assert regions[2].bbox == (143, 90, 855, 282)


def test_figure_region_has_a_box_and_no_text():
    """The whole reason regions are retained: the model emits zero characters
    for an illustration, so the box is the only handle on it."""
    figure = next(r for r in parse_grounded_page(REAL_PAGE) if r.is_figure)
    assert figure.content == ""
    assert figure.bbox == (143, 90, 855, 282)


def test_table_html_survives_parsing_intact():
    table = next(r for r in parse_grounded_page(REAL_PAGE) if r.label == "table")
    assert table.content.startswith("<table>") and "colspan" in table.content


def test_documented_ref_det_pairing_is_also_handled():
    regions = parse_grounded_page("<|ref|>Section 1<|/ref|><|det|>[[12, 34, 56, 78]]<|/det|>")
    assert [(r.label, r.content, r.bbox) for r in regions] == [("text", "Section 1", (12, 34, 56, 78))]


def test_ungrounded_output_becomes_one_text_region():
    regions = parse_grounded_page("Just plain prose, no markup at all.")
    assert [(r.label, r.bbox) for r in regions] == [("text", None)]
    assert regions[0].content == "Just plain prose, no markup at all."


def test_text_before_the_first_box_is_kept():
    regions = parse_grounded_page("preamble\n<|det|>text [1, 2, 3, 4]<|/det|>body")
    assert [r.content for r in regions] == ["preamble", "body"]


def test_truncated_unpaired_ref_terminates():
    """Regression: the fallback scan used to re-find the marker at its own
    position, leaving the parser spinning forever on a page truncated at
    max_tokens."""
    regions = parse_grounded_page("<|det|>text [1, 2, 3, 4]<|/det|>ok\n<|ref|>truncated heading")
    assert "ok" in [r.content for r in regions]
    assert any("truncated heading" in r.content for r in regions)


def test_missing_box_is_tolerated():
    regions = parse_grounded_page("<|det|>text<|/det|>body")
    assert regions[0].bbox is None and regions[0].content == "body"


def test_label_is_normalized():
    assert parse_grounded_page("<|det|>Page Number [1, 2, 3, 4]<|/det|>7")[0].label == "page_number"


# ---------------------------------------------------------------------------
# Rendering policy
# ---------------------------------------------------------------------------


def test_render_drops_running_headers_and_page_numbers():
    out = render_transcript(parse_grounded_page(REAL_PAGE))
    assert "JGL Consultants" not in out
    assert "Tulsa Zoo Request for Proposal" not in out
    assert not out.rstrip().endswith("12")


def test_render_promotes_titles_to_headings():
    out = render_transcript(parse_grounded_page(REAL_PAGE))
    assert "## Historic Visitor Food Revenue" in out


def test_render_emits_a_figure_placeholder():
    """Without this the page transcribes as if the illustration never existed."""
    assert "[figure 1]" in render_transcript(parse_grounded_page(REAL_PAGE))


def test_figure_placeholders_are_numbered_per_page():
    raw = "<|det|>image [1, 2, 3, 4]<|/det|><|det|>image [5, 6, 7, 8]<|/det|>"
    out = render_transcript(parse_grounded_page(raw))
    assert "[figure 1]" in out and "[figure 2]" in out


def test_render_keeps_tables_as_html():
    """9/9 real tables use colspan/rowspan, which markdown pipe tables cannot
    express -- converting would silently corrupt them."""
    out = render_transcript(parse_grounded_page(REAL_PAGE))
    assert "<table>" in out and "colspan" in out


def test_render_policy_is_overridable():
    regions = parse_grounded_page(REAL_PAGE)
    out = render_transcript(regions, drop_labels=frozenset(), heading_level=3, figure_placeholder=None)
    assert "JGL Consultants" in out
    assert "### Historic Visitor Food Revenue" in out
    assert "[figure" not in out


def test_extract_returns_both_markdown_and_regions():
    text, regions = extract_grounded_page(REAL_PAGE)
    assert "## Historic Visitor Food Revenue" in text
    assert len(regions) == 7


# ---------------------------------------------------------------------------
# Figure cropping
# ---------------------------------------------------------------------------


def test_bbox_maps_from_normalized_space_to_pixels():
    """Boxes are normalized to BBOX_SCALE, not page pixels -- verified against
    1700x2200 renders."""
    assert bbox_to_pixels((0, 0, BBOX_SCALE, BBOX_SCALE), 1700, 2200) == (0, 0, 1700, 2200)
    assert bbox_to_pixels((235, 325, 460, 504), 1700, 2200) == (400, 715, 782, 1109)


def test_bbox_is_clamped_and_normalized():
    assert bbox_to_pixels((-50, 0, 2000, 500), 1000, 1000) == (0, 0, 1000, 500)
    assert bbox_to_pixels((800, 600, 200, 100), 1000, 1000) == (200, 100, 800, 600)


def _page_png(width: int = 1700, height: int = 2200) -> bytes:
    pil_image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    pil_image.new("RGB", (width, height), (255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_crop_figures_returns_one_crop_per_figure():
    pil_image = pytest.importorskip("PIL.Image")
    regions = parse_grounded_page(
        "<|det|>image [235, 325, 460, 504]<|/det|><|det|>text [1, 2, 3, 4]<|/det|>body"
    )
    crops = crop_figures(_page_png(), regions)
    assert len(crops) == 1
    region, data = crops[0]
    assert region.is_figure
    with pil_image.open(io.BytesIO(data)) as im:
        assert im.size == (382, 394)


def test_crop_skips_degenerate_and_boxless_regions():
    """One bad box must not fail a document's ingestion."""
    regions = [
        PageRegion("image", None, ""),
        PageRegion("image", (500, 500, 501, 501), ""),  # ~2px -> below min
    ]
    assert crop_figures(_page_png(), regions) == []


def test_crop_ignores_non_figure_labels_by_default():
    regions = parse_grounded_page("<|det|>table [100, 100, 400, 400]<|/det|><table></table>")
    assert crop_figures(_page_png(), regions) == []
    assert len(crop_figures(_page_png(), regions, labels=("table",))) == 1


def test_nested_bracket_box_form_is_parsed_not_leaked_as_text():
    """A bare det with the card's [[x,y,x,y]] box must not leak coordinates into
    the transcript: a box pattern stopping at the first ']' fails the match and
    the whole block falls through to plain text."""
    regions = parse_grounded_page("<|det|>[[1, 2, 3, 4]]<|/det|>")
    assert [(r.label, r.bbox, r.content) for r in regions] == [("text", (1, 2, 3, 4), "")]
    assert render_transcript(regions) == ""
