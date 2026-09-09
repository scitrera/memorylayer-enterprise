# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Structured layout parsing of grounded OCR output.

Unlimited-OCR does not return prose — it returns a **layout segmentation**. Each
region is emitted as a labelled bounding box followed by its text::

    <|det|>header [114, 47, 244, 62]<|/det|>JGL Consultants
    <|det|>title [113, 92, 336, 108]<|/det|>Visitor Food Service Outlets
    <|det|>table [290, 361, 709, 604]<|/det|><table>...</table>
    <|det|>image [235, 325, 460, 504]<|/det|>
    <|det|>page_number [494, 939, 506, 952]<|/det|>8

Blindly stripping that markup (the previous behavior) threw away four useful
things, measured over a 10-page sample of a real RFP:

* **figures vanished entirely** — every ``image`` region emits ZERO text (7/7),
  so a page of photographs transcribed to silence. The box is exact, though, so
  the figure can be cropped out of the page render instead of lost;
* **headings flattened into prose** — 26 ``title`` regions became indistinguishable
  from body text, costing downstream chunking its structure;
* **running headers duplicated per page** — the same two strings on 10/10 pages,
  landing in every page's memory and embedding;
* **page numbers inlined** — bare integers glued onto the body text.

So the parse is kept and the *rendering* becomes policy. Boxes are retained on
every region so a caller can crop figures (see :mod:`.figures`).

Two emission formats are handled: the labelled form above (what the model
actually produces) and the ``<|ref|>text<|/ref|><|det|>[box]<|/det|>`` pairing
the model card documents. Neither is assumed — a parser that required the
documented pairing would return nothing on real output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Bounding boxes are normalized to this square, NOT page pixels. Verified
#: against 1700x2200 renders: boxes top out near 950, and a page_number box maps
#: to the bottom-centre of the page.
BBOX_SCALE = 1000

LABEL_TEXT = "text"
LABEL_TITLE = "title"
LABEL_HEADER = "header"
LABEL_TABLE = "table"
LABEL_IMAGE = "image"
LABEL_PAGE_NUMBER = "page_number"

#: Boilerplate that repeats on every page and adds nothing to a page's meaning.
DEFAULT_DROP_LABELS = frozenset({LABEL_HEADER, LABEL_PAGE_NUMBER})

DEFAULT_FIGURE_PLACEHOLDER = "[figure %d]"

_REF_DET_RE = re.compile(
    r"<\|ref\|>(?P<text>.*?)<\|/ref\|>\s*<\|det\|>(?P<box>.*?)<\|/det\|>", re.DOTALL
)
#: The box is written either ``[x, y, x, y]`` (labelled form) or ``[[x, y, x, y]]``
#: (the model card's form). ``\[\[?[^\]]*\]\]?`` accepts both in linear time; a
#: plain ``\[[^\]]*\]`` stops at the first ``]`` and fails the whole match on the
#: nested form, which then leaks raw coordinates into the transcript as text.
_DET_RE = re.compile(
    r"<\|det\|>\s*(?P<label>[A-Za-z_][A-Za-z0-9_ -]*)?\s*(?P<box>\[\[?[^\]]*\]\]?)?\s*<\|/det\|>",
    re.DOTALL,
)
_NEXT_MARKER_RE = re.compile(r"<\|(?:ref|det)\|>")
_INT_RE = re.compile(r"-?\d+")
# Special-token markers that survive when skip_special_tokens=False (EOS, pad).
_RESIDUAL_TOKEN_RE = re.compile(r"<[|｜][^<>\n]{0,64}?[|｜]>")


@dataclass(frozen=True)
class PageRegion:
    """One labelled region of a page.

    ``bbox`` is ``(x0, y0, x1, y1)`` in the normalized :data:`BBOX_SCALE` space,
    or ``None`` when the model emitted no box. ``content`` is the region's text
    — empty for figures, HTML for tables.
    """

    label: str
    bbox: tuple[int, int, int, int] | None
    content: str

    @property
    def is_figure(self) -> bool:
        return self.label == LABEL_IMAGE


def _parse_bbox(raw: str | None) -> tuple[int, int, int, int] | None:
    if not raw:
        return None
    values = [int(v) for v in _INT_RE.findall(raw)]
    if len(values) < 4:
        return None
    return tuple(values[:4])  # a [[x,y,x,y]] nesting yields the same first four


def _clean_fragment(text: str) -> str:
    return _RESIDUAL_TOKEN_RE.sub("", text).strip()


def parse_grounded_page(raw: str) -> list[PageRegion]:
    """Split grounded output into ordered regions.

    Unlabelled prose (output with no grounding markup at all, or text preceding
    the first box) becomes a ``text`` region with no bbox, so this degrades to
    "one region holding everything" rather than losing content.
    """
    regions: list[PageRegion] = []
    position = 0
    length = len(raw)

    while position < length:
        ref_match = _REF_DET_RE.match(raw, position)
        if ref_match:
            content = _clean_fragment(ref_match.group("text"))
            if content:
                regions.append(
                    PageRegion(LABEL_TEXT, _parse_bbox(ref_match.group("box")), content)
                )
            position = ref_match.end()
            continue

        det_match = _DET_RE.match(raw, position)
        if det_match:
            next_marker = _NEXT_MARKER_RE.search(raw, det_match.end())
            end = next_marker.start() if next_marker else length
            label = (det_match.group("label") or LABEL_TEXT).strip().lower().replace(" ", "_")
            regions.append(
                PageRegion(
                    label,
                    _parse_bbox(det_match.group("box")),
                    _clean_fragment(raw[det_match.end():end]),
                )
            )
            position = end
            continue

        # Plain text, or a marker that matched neither form -- an unpaired
        # "<|ref|>" from output truncated at max_tokens is the real case.
        # Scan from position+1 so the scan ALWAYS advances: searching from
        # `position` would re-find that same marker, leave `end == position`,
        # and spin forever on a truncated page.
        next_marker = _NEXT_MARKER_RE.search(raw, position + 1)
        end = next_marker.start() if next_marker else length
        content = _clean_fragment(raw[position:end])
        if content:
            regions.append(PageRegion(LABEL_TEXT, None, content))
        position = end

    return regions


def render_transcript(
    regions: list[PageRegion],
    *,
    drop_labels: frozenset[str] | set[str] = DEFAULT_DROP_LABELS,
    heading_level: int = 2,
    figure_placeholder: str | None = DEFAULT_FIGURE_PLACEHOLDER,
    figure_captions: dict[int, str] | None = None,
) -> str:
    """Render regions to markdown.

    ``title`` regions become headings; ``drop_labels`` are omitted; figures
    become a placeholder so the transcript records that something was there.

    ``figure_captions`` maps 1-based figure number to a caption, which is
    appended to that figure's placeholder. With it the transcript states what
    the illustration depicts, so a consumer can tell from the text alone whether
    it needs to fetch the crop.

    Tables are emitted **as the model produced them — HTML**. Converting to
    markdown pipe tables is tempting and wrong: every table measured in the
    sample (9/9) used ``colspan``/``rowspan``, which pipe tables cannot express,
    so the conversion would silently corrupt real data.
    """
    heading = "#" * max(1, heading_level)
    figure_index = 0
    blocks: list[str] = []

    for region in regions:
        if region.label in drop_labels:
            continue
        if region.is_figure:
            figure_index += 1
            if figure_placeholder:
                marker = figure_placeholder % figure_index
                caption = (figure_captions or {}).get(figure_index)
                if caption:
                    # "[figure 1]" -> "[figure 1: <caption>]" so the marker stays
                    # a single greppable token that still carries the meaning.
                    marker = "%s: %s]" % (marker.rstrip("]"), caption)
                blocks.append(marker)
            # A figure region carries a caption only rarely; emit it if present.
            if region.content:
                blocks.append(region.content)
            continue
        if not region.content:
            continue
        if region.label == LABEL_TITLE:
            blocks.append("%s %s" % (heading, region.content))
        else:
            blocks.append(region.content)

    return "\n\n".join(blocks).strip()


def extract_grounded_page(raw: str) -> tuple[str, list[PageRegion]]:
    """Contract hook: grounded output -> (markdown, regions)."""
    regions = parse_grounded_page(raw)
    return render_transcript(regions), regions
