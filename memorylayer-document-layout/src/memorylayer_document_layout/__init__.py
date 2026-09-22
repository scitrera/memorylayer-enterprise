# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only
"""Locate quotations in recorded page geometry without I/O or source verification."""

from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser
from typing import Any, Literal, TypedDict


class LocatedRegion(TypedDict):
    region_id: str
    image_sha256: str
    bbox: list[float]
    origin: Literal["ocr"]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(type(v) not in (int, float) or not 0 <= v <= 1 for v in value):
        return None
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        return None
    return [float(v) for v in value]


def validate_layout(
    value: Any, text: str, *, image_sha256: str | None = None
) -> dict | None:
    """Return the public layout fields only when transcript/image identity agrees.

    Unknown/private metadata is omitted. Invalid boxes become unlocated blocks;
    an invalid layout or duplicate region identity rejects the entire layout.
    """
    if (
        not isinstance(text, str)
        or not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != 1
        or value.get("coordinate_space") != "normalized_0_1"
        or value.get("transcript_sha256") != text_sha256(text)
        or not isinstance(value.get("image_sha256"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", value["image_sha256"])
        or (image_sha256 is not None and value["image_sha256"] != image_sha256)
        or not isinstance(value.get("regions"), list)
    ):
        return None
    rows, seen = [], set()
    for region in value["regions"]:
        if (
            not isinstance(region, dict)
            or not isinstance(region.get("id"), str)
            or not region["id"]
            or region["id"] in seen
            or not isinstance(region.get("text"), str)
            or not isinstance(region.get("label", "text"), str)
        ):
            return None
        seen.add(region["id"])
        rows.append(
            {
                "id": region["id"],
                "text": region["text"],
                "label": region.get("label", "text"),
                "bbox": _valid_box(region.get("bbox")),
            }
        )
    return {
        "version": 1,
        "coordinate_space": "normalized_0_1",
        "image_sha256": value["image_sha256"],
        "transcript_sha256": value["transcript_sha256"],
        "regions": rows,
    }


def _region(region: dict, layout: dict) -> LocatedRegion:
    return {
        "region_id": region["id"],
        "image_sha256": layout["image_sha256"],
        "bbox": region["bbox"],
        "origin": "ocr",
    }


def locate_quote(
    quote: str,
    text: str,
    layout: Any,
    *,
    mode: Literal["text", "table_row"] = "text",
    image_sha256: str | None = None,
) -> list[LocatedRegion]:
    """Locate a unique quotation using existing whole-block geometry.

    ``text`` requires a unique whitespace-normalized literal passage, possibly
    spanning contiguous OCR blocks. ``table_row`` requires one complete ordered
    pipe-separated row in one recorded HTML table; it tolerates the documented
    OCR currency/math delimiter error without changing digits/signs/percentages.

    This grants no source access and establishes no claim-verification result.
    Authorize the source before calling; pass the expected image hash when it is
    available. Missing, stale, ambiguous or unlocated matches return an empty list.
    Neither mode invents row/cell rectangles or modifies the input layout/text.
    """
    if mode not in ("text", "table_row"):
        raise ValueError("Unknown quote locator mode")
    layout = validate_layout(layout, text, image_sha256=image_sha256)
    if layout is None or not isinstance(quote, str) or not quote.strip():
        return []
    if mode == "text":
        return _text_regions(quote, text, layout)
    return _table_quote_regions(quote, text, layout)


def _normalized(text):
    return " ".join(re.sub(r"(</tr>)\s+(?=<tr(?:\s|>))", r"\1", text).split())


def _text_regions(quote, page_text, layout):
    """Link a unique exact passage to whole OCR blocks, possibly spanning blocks.

    No fuzzy matching or word-level precision is claimed. Duplicate quotations,
    absent geometry or a passage crossing an unlocated block use page fallback.
    """
    needle = _normalized(quote)
    if not layout or not needle or _normalized(page_text).count(needle) != 1:
        return []
    pieces, spans, cursor = [], [], 0
    for region in layout["regions"]:
        text = _normalized(region["text"])
        if not text:
            continue
        pieces.append(text)
        spans.append((cursor, cursor + len(text), region))
        cursor += len(text) + 1
    joined = " ".join(pieces)
    start = joined.find(needle)
    if start < 0 or joined.find(needle, start + 1) >= 0:
        return []
    selected = [
        r for left, right, r in spans if left < start + len(needle) and right > start
    ]
    if not selected or any(r["bbox"] is None for r in selected):
        return []
    return [_region(r, layout) for r in selected]


class _TableRows(HTMLParser):
    """Extract complete HTML table rows for display-only exact cell matching."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.row, self.cell = [], None, None
        self.table_depth = 0
        self.invalid = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table_depth += 1
            if self.table_depth > 1:
                self.invalid = True
        elif self.table_depth == 1:
            if tag == "tr":
                if self.row is not None:
                    self.invalid = True
                self.row = []
            elif tag in ("td", "th"):
                if self.row is None or self.cell is not None:
                    self.invalid = True
                self.cell = []
            elif tag == "br" and self.cell is not None:
                self.cell.append(" ")
            elif tag in ("script", "style"):
                self.invalid = True

    def handle_endtag(self, tag):
        if tag == "table":
            if self.row is not None or self.cell is not None:
                self.invalid = True
            self.table_depth -= 1
        elif self.table_depth == 1:
            if tag in ("td", "th"):
                if self.row is None or self.cell is None:
                    self.invalid = True
                else:
                    self.row.append(" ".join("".join(self.cell).split()))
                self.cell = None
            elif tag == "tr":
                if self.row is None or self.cell is not None:
                    self.invalid = True
                else:
                    self.rows.append(self.row)
                self.row = None

    def handle_data(self, data):
        if self.table_depth == 1 and self.cell is not None:
            self.cell.append(data)


def _table_rows(text):
    parser = _TableRows()
    parser.feed(text)
    parser.close()
    return (
        []
        if parser.invalid or parser.table_depth or parser.row is not None
        else parser.rows
    )


def _table_cell_matches(actual, quoted):
    if actual == quoted:
        return True
    # OCR can mistake currency dollars across adjacent cells for a math span,
    # leaving \( / \) delimiters in those cells. This is display location only:
    # preserve every digit, sign, percentage and all other punctuation.
    if r"\(" not in actual and r"\)" not in actual:
        return False
    actual = actual.replace(r"\(", "").replace(r"\)", "").strip()
    if re.fullmatch(r"\$[+-]?\d[\d,]*(?:\.\d+)?(?:\s*\(\d+(?:\.\d+)?%\))?", quoted):
        quoted = quoted[1:]
    return actual == quoted


def _table_row_matches(row, cells):
    return len(row) == len(cells) and all(
        _table_cell_matches(a, b) for a, b in zip(row, cells)
    )


def _table_quote_regions(quote, page_text, layout):
    """Locate one exact pipe-separated row in one recorded OCR table block.

    This is a display locator, not source verification. It never infers row/cell
    rectangles, changes numbers, skips cells or crosses rows. Ambiguity falls
    back to page/transcript navigation, including duplicate rows on the page.
    """
    if not layout or "|" not in quote:
        return []
    cells = [" ".join(cell.split()) for cell in quote.strip().strip("|").split("|")]
    if len(cells) < 2 or not all(cells):
        return []
    if sum(_table_row_matches(row, cells) for row in _table_rows(page_text)) != 1:
        return []
    matches = [
        r
        for r in layout["regions"]
        if r.get("label") == "table"
        and sum(_table_row_matches(row, cells) for row in _table_rows(r["text"])) == 1
    ]
    if len(matches) != 1 or matches[0]["bbox"] is None:
        return []
    region = matches[0]
    return [_region(region, layout)]
