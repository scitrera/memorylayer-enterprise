# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the OCR model contracts.

Unlimited-OCR returns **empty output** if any part of its prompt/decode recipe
is wrong, and an empty transcript is indistinguishable from "this page was
blank". Each element of the recipe is pinned here so a well-meaning edit fails
in CI rather than silently producing blank documents.
"""

from __future__ import annotations

import pytest

from memorylayer_saas.services.transcription.contracts import (
    UNLIMITED_OCR_NGRAM_SIZE,
    UNLIMITED_OCR_PROMPT,
    UNLIMITED_OCR_WINDOW_SIZE,
    build_contract,
    build_generic_markdown_contract,
    build_unlimited_ocr_contract,
    clean_transcription_output,
    strip_grounding_tokens,
)

DATA_URL = "data:image/png;base64,AAAA"


# ---------------------------------------------------------------------------
# unlimited_ocr
# ---------------------------------------------------------------------------


def test_unlimited_ocr_puts_text_before_image():
    """The literal <image> placeholder leads the prompt, so ordering matters."""
    parts = build_unlimited_ocr_contract().build_messages(None, DATA_URL)[0]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"] == DATA_URL


def test_unlimited_ocr_prompt_is_the_exact_trained_recipe():
    parts = build_unlimited_ocr_contract().build_messages(None, DATA_URL)[0]["content"]
    assert parts[0]["text"] == UNLIMITED_OCR_PROMPT
    assert parts[0]["text"].startswith("<image>")


def test_unlimited_ocr_ignores_the_caller_system_prompt():
    """A generic markdown system prompt would degrade this model's output."""
    contract = build_unlimited_ocr_contract()
    parts = contract.build_messages("You are a transcription assistant.", DATA_URL)[0]["content"]
    assert parts[0]["text"] == UNLIMITED_OCR_PROMPT
    assert contract.fixed_prompt is True


def test_unlimited_ocr_pins_skip_special_tokens_false():
    """The grounding markup IS special tokens; skipping them empties the output."""
    assert build_unlimited_ocr_contract().extra_body["skip_special_tokens"] is False


def test_unlimited_ocr_sends_ngram_xargs_by_default():
    xargs = build_unlimited_ocr_contract().extra_body["vllm_xargs"]
    assert xargs == {"ngram_size": UNLIMITED_OCR_NGRAM_SIZE, "window_size": UNLIMITED_OCR_WINDOW_SIZE}


def test_unlimited_ocr_window_size_is_overridable():
    xargs = build_unlimited_ocr_contract(window_size=128).extra_body["vllm_xargs"]
    assert xargs["window_size"] == 128
    assert xargs["ngram_size"] == UNLIMITED_OCR_NGRAM_SIZE


def test_the_default_window_covers_a_realistic_repetition_cycle():
    """128 was too small to see the loop that actually occurs.

    A dense page degenerated into a ladder of zero-width regions ~34 lines
    (~530 tokens) long before repeating. With a 128-token lookback the blocker
    never had the repeat in view, so it ran to the 16384-token cap and the whole
    page was discarded for finish_reason=length.
    """
    assert UNLIMITED_OCR_WINDOW_SIZE >= 1024


def test_unlimited_ocr_extract_returns_markdown_and_regions():
    text, regions = build_unlimited_ocr_contract().extract(
        "<|det|>title [1, 2, 3, 4]<|/det|>Hi"
    )
    assert text == "## Hi"
    assert [(r.label, r.bbox) for r in regions] == [("title", (1, 2, 3, 4))]


# ---------------------------------------------------------------------------
# generic_markdown
# ---------------------------------------------------------------------------


def test_generic_markdown_puts_image_first():
    parts = build_generic_markdown_contract().build_messages(None, DATA_URL)[0]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[1]["type"] == "text"


def test_generic_markdown_folds_in_the_system_prompt():
    parts = build_generic_markdown_contract().build_messages("Only tables.", DATA_URL)[0]["content"]
    assert parts[1]["text"].startswith("Only tables.")


def test_generic_markdown_sends_no_extra_body():
    """Unlimited-OCR's decode knobs must not leak onto chat-templated models."""
    contract = build_generic_markdown_contract()
    assert contract.extra_body is None
    assert contract.extract is None
    assert contract.fixed_prompt is False


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_build_contract_resolves_known_names():
    assert build_contract("unlimited_ocr").name == "unlimited_ocr"
    assert build_contract("generic_markdown").name == "generic_markdown"


def test_build_contract_rejects_unknown_name():
    """A typo'd contract must fail at config time, not transcribe nothing at 3am."""
    with pytest.raises(ValueError, match="unknown transcription contract"):
        build_contract("unlimted_ocr")


def test_build_contract_drops_none_overrides():
    """Unset per-profile knobs fall back to the model's defaults."""
    xargs = build_contract("unlimited_ocr", ngram_size=None, window_size=None).extra_body["vllm_xargs"]
    assert xargs["window_size"] == UNLIMITED_OCR_WINDOW_SIZE


def test_build_contract_ignores_irrelevant_overrides_for_generic():
    assert build_contract("generic_markdown", ngram_size=None, window_size=None).extra_body is None


# ---------------------------------------------------------------------------
# extraction helpers
# ---------------------------------------------------------------------------


def test_strip_grounding_keeps_text_drops_boxes():
    raw = "<|ref|>Title<|/ref|><|det|>[[0,0,1,1]]<|/det|> tail"
    assert strip_grounding_tokens(raw) == "Title tail"


def test_strip_grounding_handles_multiline_spans():
    assert strip_grounding_tokens("<|ref|>a\nb<|/ref|><|det|>[[0,0,1,1]]<|/det|>") == "a\nb"


def test_strip_grounding_drops_unpaired_opener_from_truncated_output():
    assert strip_grounding_tokens("<|ref|>Truncated heading") == "Truncated heading"


def test_strip_grounding_drops_residual_special_tokens():
    """skip_special_tokens=False leaves the EOS marker in the decoded text."""
    assert strip_grounding_tokens("<|ref|>Body<|/ref|><|end of sentence|>") == "Body"
    assert strip_grounding_tokens("<|ref|>Body<|/ref|><｜end▁of▁sentence｜>") == "Body"


def test_strip_grounding_leaves_ordinary_markdown_alone():
    md = "# Heading\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n`x <| y` stays"
    assert strip_grounding_tokens(md) == md


def test_clean_output_strips_thinking_and_fences():
    assert clean_transcription_output("<think>hmm</think>\n```markdown\n# H\n```") == "# H"
    assert clean_transcription_output("◁think▷x◁/think▷# H") == "# H"


# ---------------------------------------------------------------------------
# DeepSeek-OCR-2 (cascade rung 2)
# ---------------------------------------------------------------------------

def test_deepseek_prompt_is_grounded_markdown_and_leads_the_image():
    from memorylayer_saas.services.transcription.contracts import build_deepseek_ocr_contract

    parts = build_deepseek_ocr_contract().build_messages(None, "data:image/png;base64,AA")[0]["content"]
    # The literal <image> placeholder is positional, so the text precedes the
    # image part -- same requirement as Unlimited-OCR.
    assert parts[0]["text"] == "<image>\n<|grounding|>Convert the document to markdown."
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,AA"


def test_deepseek_whitelists_the_table_cell_tokens():
    """<td>/</td> must survive the repetition blocker.

    A table is legitimately full of near-identical cell tags; suppressing them
    as "repetition" would collapse the row structure -- the same loss the
    transcript sanitiser had to be fixed for downstream.
    """
    from memorylayer_saas.services.transcription.contracts import build_deepseek_ocr_contract

    xargs = build_deepseek_ocr_contract().extra_body["vllm_xargs"]
    assert xargs["whitelist_token_ids"] == [128821, 128822]


def test_deepseek_keeps_special_tokens_and_suppresses_repetition():
    from memorylayer_saas.services.transcription.contracts import build_deepseek_ocr_contract

    body = build_deepseek_ocr_contract().extra_body
    # Grounding markup IS special tokens; dropping them decodes to nothing.
    assert body["skip_special_tokens"] is False
    assert body["vllm_xargs"]["ngram_size"] == 30
    # Deliberately above the model card's 90: a window shorter than the
    # repetition cycle cannot see the repeat (measured ~530 tokens on the
    # sibling model, which ran to the token cap emitting a coordinate ladder).
    assert body["vllm_xargs"]["window_size"] >= 1024


def test_deepseek_is_resolvable_by_name():
    contract = build_contract("deepseek_ocr")
    assert contract.name == "deepseek_ocr"
    assert contract.fixed_prompt is True


def test_deepseek_overrides_apply():
    contract = build_contract("deepseek_ocr", ngram_size=12, window_size=64)
    assert contract.extra_body["vllm_xargs"]["ngram_size"] == 12
    assert contract.extra_body["vllm_xargs"]["window_size"] == 64
