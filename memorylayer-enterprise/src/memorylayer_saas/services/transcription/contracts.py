# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Model-specific prompt / decode / extraction contracts for page OCR.

A *contract* is everything about talking to one OCR model that is not the URL:
how the chat message is built, what non-OpenAI request fields ride along, and
how the raw completion is turned back into markdown. Endpoints are
interchangeable; contracts are not.

This exists because Unlimited-OCR is trained against one exact prompt/decode
recipe and returns **empty output** if any part of it is wrong — a failure mode
that looks identical to "the model is bad at this page". Pinning it in data,
next to its tests, is what keeps that from being rediscovered.

> The same contract logic also lives in the embed-server transcription
> providers (``memorylayer_embed_server.services.transcription``), which remains
> the OSS / no-sparkrun path. Folding both onto one copy in OSS core is a
> follow-up, deliberately deferred while that tree is busy.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .regions import PageRegion, extract_grounded_page

# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------

_THINKING_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINKING_ALT_RE = re.compile(r"◁think▷.*?◁/think▷", re.DOTALL)

# Grounding markup from the DeepSeek-OCR lineage (DeepSeek-OCR-2, Unlimited-OCR):
# each recognized region is a text span followed by its bounding box —
#     <|ref|>Section 1<|/ref|><|det|>[[12, 34, 56, 78]]<|/det|>
# The text belongs in the markdown; the coordinates do not.
_GROUNDING_REF_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)
_GROUNDING_DET_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
# Markers that survive the pass above: an unpaired opener from output truncated
# at max_tokens, or a special token left in because the request ran with
# skip_special_tokens=False (which Unlimited-OCR requires). Matches both the
# ASCII bar and the fullwidth bar the DeepSeek tokenizers use. Length-bounded so
# a stray "<|" in real document text cannot eat a paragraph.
_RESIDUAL_SPECIAL_TOKEN_RE = re.compile(r"<[|｜][^<>\n]{0,64}?[|｜]>")


def strip_thinking_tokens(content: str) -> str:
    content = _THINKING_RE.sub("", content)
    return _THINKING_ALT_RE.sub("", content).strip()


def strip_markdown_wrapper(content: str) -> str:
    """Drop a ```markdown fence wrapping the whole response."""
    content = content.strip()
    for fence in ("```markdown", "```md"):
        if content.startswith(fence) and content.endswith("```"):
            return content[len(fence):-3].strip()
    if content.startswith("```") and content.endswith("```"):
        first, _, _ = content.partition("\n")
        if first.strip() == "```":
            return content[3:-3].strip()
    return content


def strip_grounding_tokens(content: str) -> str:
    """Unwrap ``<|ref|>`` spans and drop ``<|det|>`` coordinate boxes."""
    content = _GROUNDING_DET_RE.sub("", content)
    content = _GROUNDING_REF_RE.sub(lambda m: m.group(1), content)
    return _RESIDUAL_SPECIAL_TOKEN_RE.sub("", content)


def clean_transcription_output(content: str) -> str:
    """Shared final pass applied to every provider's output."""
    return strip_markdown_wrapper(strip_thinking_tokens(content)).strip()


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OcrModelContract:
    """How to prompt one OCR model and how to read it back.

    ``build_messages`` receives the caller's system prompt (which a fixed-recipe
    model ignores) and a ``data:image/png;base64,...`` URL, and returns the
    OpenAI ``messages`` array. ``extra_body`` carries non-OpenAI request fields.

    ``extract`` turns the raw completion into ``(markdown, regions)``. It
    returns regions as well as text because a grounded model emits a layout
    segmentation, and some of that structure -- figure boxes above all -- cannot
    be expressed in the transcript but is worth keeping. Contracts for models
    that return plain prose leave it ``None``.
    """

    name: str
    build_messages: Callable[[str | None, str], list[dict]]
    extra_body: dict | None = None
    extract: Callable[[str], tuple[str, list[PageRegion]]] | None = None
    #: True when the model ignores the caller's system prompt entirely.
    fixed_prompt: bool = False


DEFAULT_TRANSCRIBE_INSTRUCTION = "Please transcribe the document in the image to markdown."

# Unlimited-OCR's serving contract. The literal "<image>" prefix leads the
# prompt, which is why the text block precedes the image block.
UNLIMITED_OCR_PROMPT = "<image>document parsing."
UNLIMITED_OCR_NGRAM_SIZE = 35
#: Lookback the n-gram repetition blocker searches for a repeat. The model card
#: suggests 128 for single-image (gundam crop mode) and 1024 for multi-page /
#: PDF input, but 128 proved far too small on dense pages.
#:
#: Measured on a marketing infographic (a dashboard of ~15 nested metric cards):
#: the model transcribed the title and body correctly, then fell into emitting a
#: ladder of ZERO-WIDTH regions -- <|det|>image [500, y, 500, y+20]<|/det|> --
#: stepping 25px down the page and starting over, ~34 lines (~530 tokens) per
#: cycle. It ran to the 16384-token cap: 1017 lines, 120 unique, and the whole
#: page then discarded for finish_reason=length.
#:
#: A 128-token window cannot see a ~530-token cycle, so the blocker never had
#: the repeat in view. 1024 covers it with room to spare.
UNLIMITED_OCR_WINDOW_SIZE = 1024


def build_unlimited_ocr_contract(
    *,
    prompt: str = UNLIMITED_OCR_PROMPT,
    ngram_size: int = UNLIMITED_OCR_NGRAM_SIZE,
    window_size: int = UNLIMITED_OCR_WINDOW_SIZE,
) -> OcrModelContract:
    """Baidu Unlimited-OCR — four things must all hold or the output is empty.

    * user text is exactly ``"<image>document parsing."``, **before** the image
      part (the literal ``<image>`` placeholder is positional);
    * ``skip_special_tokens: false`` — the grounding markup the model emits *is*
      made of special tokens, so dropping them at decode leaves an empty string;
    * ``ngram_size`` / ``window_size`` ride along as ``vllm_xargs`` for the
      arch's n-gram logits processor (without it, long documents loop on
      ``<|det|>`` coordinate tokens instead of terminating);
    * the transcript comes back as grounded markup and must be unwrapped.

    The server side of this recipe is pinned in
    ``sparkrun_recipes/transcribe_unlimited-ocr.yaml``.
    """

    def build_messages(system_prompt: str | None, data_url: str) -> list[dict]:
        del system_prompt  # trained for one prompt; anything else degrades it
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

    return OcrModelContract(
        name="unlimited_ocr",
        build_messages=build_messages,
        extra_body={
            "skip_special_tokens": False,
            "vllm_xargs": {"ngram_size": int(ngram_size), "window_size": int(window_size)},
        },
        extract=extract_grounded_page,
        fixed_prompt=True,
    )


# DeepSeek-OCR-2's serving contract. The literal "<image>" leads the prompt, as
# with Unlimited-OCR, so the text block precedes the image block.
DEEPSEEK_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
DEEPSEEK_OCR_NGRAM_SIZE = 30
#: The model card suggests 90. We default far higher for the same reason
#: UNLIMITED_OCR_WINDOW_SIZE does: a window shorter than the repetition cycle
#: cannot see the repeat, and a dense page's cycle measured ~530 tokens on the
#: sibling model. 90 would be blind to anything but the tightest loop. Override
#: per profile if this model turns out to behave differently.
DEEPSEEK_OCR_WINDOW_SIZE = 1024
#: Token ids for ``<td>`` / ``</td>``. Whitelisted so the repetition blocker
#: cannot suppress them: a table is legitimately full of near-identical cell
#: tags, and blocking those as "repetition" would collapse its structure --
#: which is exactly the row-boundary loss the transcript sanitiser had to be
#: fixed for downstream.
DEEPSEEK_OCR_TABLE_TOKEN_IDS = (128821, 128822)


def build_deepseek_ocr_contract(
    *,
    prompt: str = DEEPSEEK_OCR_PROMPT,
    ngram_size: int = DEEPSEEK_OCR_NGRAM_SIZE,
    window_size: int = DEEPSEEK_OCR_WINDOW_SIZE,
    whitelist_token_ids: tuple[int, ...] = DEEPSEEK_OCR_TABLE_TOKEN_IDS,
) -> OcrModelContract:
    """DeepSeek-OCR-2 — grounded document-to-markdown.

    Same shape as the Unlimited-OCR contract and the same four requirements:
    positional ``<image>`` in the user text before the image part,
    ``skip_special_tokens: false`` (the grounding markup IS special tokens),
    n-gram suppression via ``vllm_xargs``, and grounded output that must be
    unwrapped. The grounded form it emits --
    ``<|ref|>text<|/ref|><|det|>[[x,y,x,y]]<|/det|>`` -- is already what
    ``extract_grounded_page`` parses.

    Distinct from Unlimited-OCR in two ways: the ``<|grounding|>`` prompt asks
    for markdown rather than a region dump, and table cell tokens are
    whitelisted out of the repetition blocker.

    Serves as the cascade rung AFTER unlimited: a second opinion for pages the
    first model cannot transcribe, not a replacement. The server side is pinned
    in ``thunder_tmp/recipes-local/transcribe-uvbuilder-ds-ocr2.yaml``.
    """

    def build_messages(system_prompt: str | None, data_url: str) -> list[dict]:
        del system_prompt  # trained for its own prompts; instructions degrade it
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

    return OcrModelContract(
        name="deepseek_ocr",
        build_messages=build_messages,
        extra_body={
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": int(ngram_size),
                "window_size": int(window_size),
                "whitelist_token_ids": [int(t) for t in whitelist_token_ids],
            },
        },
        extract=extract_grounded_page,
        fixed_prompt=True,
    )


def build_generic_markdown_contract(
    *, instruction: str = DEFAULT_TRANSCRIBE_INSTRUCTION,
) -> OcrModelContract:
    """Any chat-templated VLM: GLM-OCR, DeepSeek-OCR, or a hosted model reached
    through an OpenAI-compatible gateway.

    Image part first, caller's system prompt folded into the user message —
    the shape that works across models whose chat templates handle the
    ``system`` role differently.
    """

    def build_messages(system_prompt: str | None, data_url: str) -> list[dict]:
        text = f"{system_prompt}\n\n{instruction}" if system_prompt else instruction
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": text},
                ],
            }
        ]

    return OcrModelContract(name="generic_markdown", build_messages=build_messages)


_CONTRACT_BUILDERS: dict[str, Callable[..., OcrModelContract]] = {
    "unlimited_ocr": build_unlimited_ocr_contract,
    "deepseek_ocr": build_deepseek_ocr_contract,
    "generic_markdown": build_generic_markdown_contract,
}

KNOWN_CONTRACTS = tuple(sorted(_CONTRACT_BUILDERS))


def build_contract(name: str, **options) -> OcrModelContract:
    """Resolve a contract by name, applying any per-profile overrides.

    Raises ``ValueError`` on an unknown name — a typo'd contract must fail at
    configuration time, not by silently transcribing nothing at 3am.
    """
    try:
        builder = _CONTRACT_BUILDERS[name]
    except KeyError:
        raise ValueError(
            "unknown transcription contract %r (known: %s)" % (name, ", ".join(KNOWN_CONTRACTS))
        ) from None
    return builder(**{k: val for k, val in options.items() if val is not None})
