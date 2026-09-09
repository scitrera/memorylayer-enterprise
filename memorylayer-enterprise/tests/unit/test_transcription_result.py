# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the embed-server transcription response normalizer.

Regression cover for a silent data-loss bug: both ingestion paths read
``result["pages"]`` with ``page_number`` / ``model`` keys, but the embed server
emits ``results`` with ``page_index`` / ``model_used``. ``dict.get("pages", [])``
returned ``[]``, so every page kept ``transcript=None`` with no error raised —
and both unit suites mocked the same fictional shape, so they stayed green.

These tests assert against the shape the server actually builds in
``memorylayer_embed_server.api.v1.transcription`` /
``models.transcription.TranscriptionResponse``.
"""

from __future__ import annotations

from memorylayer_saas.services.transcription import (
    TranscribedPage,
    pages_from_embed_server_response,
)


def _page(index: int, content: str, *, success: bool = True, model: str | None = "vlm-v1") -> dict:
    """One entry as the embed server's TranscriptionResult serializes it."""
    return {
        "page_index": index,
        "content": content,
        "success": success,
        "model_used": model,
        "provider_used": "glm-ocr",
        "attempts": [],
    }


def _response(*entries: dict) -> dict:
    return {
        "results": list(entries),
        "stats": {
            "total_pages": len(entries),
            "successful_pages": sum(1 for e in entries if e.get("success")),
            "failed_pages": sum(1 for e in entries if not e.get("success")),
        },
    }


# ---------------------------------------------------------------------------
# The regression itself
# ---------------------------------------------------------------------------


def test_reads_the_shape_the_server_actually_emits():
    pages = pages_from_embed_server_response(_response(_page(0, "# Title")))
    assert pages == [TranscribedPage(request_index=0, content="# Title", model="vlm-v1")]


def test_legacy_pages_key_yields_nothing():
    """The pre-fix callers dug for this shape; it must not silently half-work."""
    legacy = {"pages": [{"page_number": 0, "content": "text", "model": "vlm-v1"}]}
    assert pages_from_embed_server_response(legacy) == []


def test_request_index_is_request_relative_not_page_no():
    """Pages are sent in batches; the server numbers them per call."""
    pages = pages_from_embed_server_response(_response(_page(0, "a"), _page(1, "b")))
    assert [p.request_index for p in pages] == [0, 1]


# ---------------------------------------------------------------------------
# Failed pages
# ---------------------------------------------------------------------------


def test_failed_page_is_omitted_not_stored_as_a_transcript():
    """The server substitutes a failure marker as `content` when the cascade
    exhausts; persisting it would put that string into blob storage, the page
    record, and every memory derived from the page."""
    response = _response(
        _page(0, "**Transcription Failed for this page**", success=False, model=None),
        _page(1, "real text"),
    )
    pages = pages_from_embed_server_response(response)
    assert [p.content for p in pages] == ["real text"]
    assert [p.request_index for p in pages] == [1]


def test_empty_content_is_omitted_even_when_marked_successful():
    assert pages_from_embed_server_response(_response(_page(0, ""))) == []


# ---------------------------------------------------------------------------
# Robustness — one bad entry must not fail the document
# ---------------------------------------------------------------------------


def test_missing_and_malformed_entries_are_skipped():
    response = {
        "results": [
            {"content": "no index", "success": True},           # missing page_index
            {"page_index": "1", "content": "str index", "success": True},  # wrong type
            "not-a-dict",
            _page(3, "good"),
        ]
    }
    pages = pages_from_embed_server_response(response)
    assert [(p.request_index, p.content) for p in pages] == [(3, "good")]


def test_absent_or_null_results_key_is_empty_not_an_error():
    assert pages_from_embed_server_response({}) == []
    assert pages_from_embed_server_response({"results": None}) == []


def test_missing_model_used_is_none_not_a_keyerror():
    entry = {"page_index": 0, "content": "text", "success": True}
    assert pages_from_embed_server_response({"results": [entry]})[0].model is None
