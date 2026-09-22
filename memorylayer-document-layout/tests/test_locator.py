# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only
import copy
import json
from types import SimpleNamespace

import pytest
from memorylayer_document_layout import locate_quote, text_sha256, validate_layout


def page(text="Optional service. Other terms.", rows=None):
    rows = rows or [
        ("first", "Optional service.", [0.1, 0.1, 0.8, 0.3]),
        ("second", "Other terms.", [0.1, 0.4, 0.8, 0.6]),
    ]
    layout = {
        "version": 1,
        "coordinate_space": "normalized_0_1",
        "image_sha256": "a" * 64,
        "transcript_sha256": text_sha256(text),
        "raw_ocr": {"storage_path": "/private/raw.txt"},
        "provider": "private-provider",
        "regions": [dict(id=i, text=t, label="text", bbox=b) for i, t, b in rows],
    }
    return SimpleNamespace(text=text, layout=layout)


def table_quote_regions(quote, page):
    return [
        SimpleNamespace(**r)
        for r in locate_quote(quote, page.text, page.layout, mode="table_row")
    ]


def test_exact_text_and_contiguous_blocks_use_existing_geometry_without_mutation():
    p = page()
    before = copy.deepcopy(p.layout)
    result = locate_quote("service. Other", p.text, p.layout, image_sha256="a" * 64)
    assert [r["region_id"] for r in result] == ["first", "second"]
    assert result[0] == dict(
        region_id="first",
        image_sha256="a" * 64,
        bbox=[0.1, 0.1, 0.8, 0.3],
        origin="ocr",
    )
    assert p.layout == before
    sanitized = validate_layout(p.layout, p.text)
    assert "raw_ocr" not in sanitized and "provider" not in sanitized
    assert "/private" not in json.dumps(sanitized)
    assert locate_quote("service. Other", p.text, sanitized) == result
    result[0]["bbox"][0] = 0.9
    assert p.layout == before


@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "missing",
        "unlocated",
        "stale_transcript",
        "unsupported_version",
        "duplicate_id",
        "wrong_space",
        "wrong_hash",
        "wrong_image",
        "empty_quote",
        "absent_quote",
    ],
)
def test_missing_stale_or_ambiguous_sources_do_not_gain_geometry(case):
    p = page()
    quote = "Optional service."
    image = "a" * 64
    if case == "duplicate":
        p = page(
            "Optional service. Optional service.",
            [
                ("a", "Optional service.", [0, 0, 1, 0.4]),
                ("b", "Optional service.", [0, 0.5, 1, 1]),
            ],
        )
    if case == "missing":
        p.layout = None
    if case == "unlocated":
        p.layout["regions"][0]["bbox"] = None
    if case == "stale_transcript":
        p.text = "Changed. Optional service."
    if case == "unsupported_version":
        p.layout["version"] = 2
    if case == "duplicate_id":
        p.layout["regions"][1]["id"] = "first"
    if case == "wrong_space":
        p.layout["coordinate_space"] = "pixels"
    if case == "wrong_hash":
        p.layout["image_sha256"] = "invalid"
    if case == "wrong_image":
        image = "b" * 64
    if case == "empty_quote":
        quote = " "
    if case == "absent_quote":
        quote = "Invented."
    assert locate_quote(quote, p.text, p.layout, image_sha256=image) == []


@pytest.mark.parametrize(
    "box",
    [
        [0, 0, 2, 1],
        [0, 0, 0, 1],
        [0, 1, 1, 0],
        [True, 0, 1, 1],
        ["0", 0, 1, 1],
        [0, 0, float("nan"), 1],
        [0, 0, float("inf"), 1],
        [0, 0, 10**400, 1],
        None,
        [],
    ],
)
def test_invalid_rectangles_remain_unlocated(box):
    p = page()
    p.layout["regions"][0]["bbox"] = box
    assert locate_quote("Optional service.", p.text, p.layout) == []


def test_unknown_mode_is_explicit_and_text_mode_does_not_apply_table_relaxation():
    p = page()
    with pytest.raises(ValueError, match="mode"):
        locate_quote("Optional service.", p.text, p.layout, mode="fuzzy")
    table = "<table><tr><td>Income</td><td>$100</td></tr></table>"
    p = page(table, [("table", table, [0.1, 0.2, 0.8, 0.4])])
    p.layout["regions"][0]["label"] = "table"
    assert locate_quote("Income | $100", p.text, p.layout) == []
    assert locate_quote("Income | $100", p.text, p.layout, mode="table_row")


@pytest.mark.parametrize(
    "case",
    [
        "",
        "wrong_number",
        "subset",
        "reordered",
        "cross_row",
        "duplicate_row",
        "duplicate_region",
        "missing_box",
        "nested",
        "malformed",
        "stale",
        "literal_pipes",
    ],
)
def test_table_quote_locates_only_one_exact_complete_row(case):
    table = (
        "<table><tr><td>Operating costs</td><td>$40 (4%)</td><td>$50 (5%)</td></tr>"
        "<tr><td><b>Net</b> Income</td><td>$100 (10%)</td><td>$200 (20%)</td></tr></table>"
    )
    quote = "Net Income | $100 (10%) | $200 (20%)"
    if case == "wrong_number":
        quote = quote.replace("$100", "$101")
    if case == "subset":
        quote = "Net Income | $100 (10%)"
    if case == "reordered":
        quote = "Net Income | $200 (20%) | $100 (10%)"
    if case == "cross_row":
        quote = "$50 (5%) | Net Income | $100 (10%)"
    if case == "duplicate_row":
        table += table
    if case == "nested":
        table = "<table><tr><td>" + table + "</td></tr></table>"
    if case == "malformed":
        table = table.replace("</td>", "", 1)
    if case == "literal_pipes":
        table = "Net Income | $100 (10%) | $200 (20%)"
    p = page(table, [("table", table, [0.1, 0.2, 0.8, 0.4])])
    p.layout["regions"][0]["label"] = "table"
    if case == "duplicate_region":
        p.layout["regions"].append({**p.layout["regions"][0], "id": "other"})
    if case == "missing_box":
        p.layout["regions"][0]["bbox"] = None
    if case == "stale":
        p.text += "changed"
    result = table_quote_regions(quote, p)
    assert [r.region_id for r in result] == ([] if case else ["table"])
    if result:
        assert result[0].bbox == [0.1, 0.2, 0.8, 0.4]


@pytest.mark.parametrize(
    "quoted, matches",
    [
        ("Net Income | $100 (10%) | $200 (20%)", True),
        ("Net Income | $101 (10%) | $200 (20%)", False),
        ("Net Income | $-100 (10%) | $200 (20%)", False),
        ("Net Income | $100 (11%) | $200 (20%)", False),
        ("Net Income | $100 | $200", False),
    ],
)
def test_table_location_recognizes_ocr_currency_math_delimiters_only(quoted, matches):
    table = r"<table><tr><td>Net Income</td><td>\(100 (10%)</td><td>200 (20%)\)</td></tr></table>"
    p = page(table, [("table", table, [0.1, 0.2, 0.8, 0.4])])
    p.layout["regions"][0]["label"] = "table"
    assert bool(table_quote_regions(quoted, p)) == matches
