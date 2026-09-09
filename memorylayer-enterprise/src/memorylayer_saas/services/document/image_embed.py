# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Per-page image-embed precompute + blob persistence.

During the embed phase, for each page that has a rendered image, we ask the
embed-server's enterprise visual tokenizer to produce that page's raw
vision-tower output (``image_embeds`` tensor, shape ``[N, hidden_dim]``) plus
the ``image_grid_thw`` tensor describing the temporal/height/width patch grid.
Both are persisted to blob storage and a reference is recorded on the page's
``visual_tokens`` JSONB column.

These raw vision embeds + grid are exactly what vLLM expects as an
``image_embeds`` chat content part: vLLM then computes the correct 3-D M-RoPE
positions for the image tokens. (The old flat ``prompt_embeds`` path gave image
tokens 1-D positions, so Qwen3.5/3.6 could not read them; that path is gone.)

Storage is keyed by model slug so multiple models can coexist for the same
page without a schema change::

    page.visual_tokens = {
        "qwen--qwen3.6-27b-fp8": {
            "embed_kind": "image_embeds",
            "embeds_blob_path": ".../image_embeds/<slug>/page_0000.pt.zst",
            "grid_blob_path":   ".../image_embeds/<slug>/page_0000.grid.pt",
            "num_image_tokens": 812,
            "hidden_dim": 5120,
            "image_grid_thw": [1, 38, 28],
            "created_at": "…",
        },
        # a second model would add another key here — no migration needed.
    }

The embeds blob is stored zstd-COMPRESSED (the embed-server compresses the raw
``torch.save`` bytes before base64). The grid blob is stored UNCOMPRESSED (raw
``torch.save`` bytes). This module needs no torch: it base64-decodes the wire
payloads for storage, and zstd-decompresses the embeds back to raw torch.save
bytes only when re-base64-ing them for the vLLM ``image_embeds`` content part.
Only ``zstandard`` + ``base64`` are used here.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from logging import Logger
from typing import Any

import zstandard
from scitrera_app_framework import Variables

from memorylayer_saas.services.document.blob_cache import cached_retrieve

from .visual_tokenize import visual_tokenize

# zstd frame magic number prefix; used to detect compressed blobs.
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def build_page_metadata(
    *, filename: str | None, page_no: int, document_id: str, source: str | None = None,
) -> dict[str, Any]:
    """Assemble the stable per-page provenance metadata for the tokenizer call."""
    meta: dict[str, Any] = {"page_no": page_no, "document_id": document_id}
    if filename:
        meta["filename"] = filename
    if source:
        meta["source"] = source
    return meta


def page_metadata_text(page, filename: str | None) -> str:
    """Human-readable page marker interleaved before each page's image part."""
    suffix = f" of {filename}" if filename else ""
    return f"[Page {page.page_no + 1}{suffix}]"


def _slug_from_result(result: dict, response_model: str | None) -> str:
    """Resolve the model slug for cache/storage keying."""
    slug = result.get("model_slug")
    if slug:
        return slug
    base = response_model or "unknown-model"
    return base.replace("/", "--").replace(" ", "_").lower()


async def precompute_and_store_image_embeds(
    *,
    embed_client,
    blob_storage,
    storage,
    pages,
    workspace_id: str,
    document_id: str,
    filename: str | None,
    logger: Logger,
    source: str | None = None,
    persist: bool = True,
) -> int:
    """Precompute per-page image-embeds for ``pages`` and persist them + a ref.

    ``pages`` should already be filtered to those with an
    ``image_storage_path``. The caller owns the ``embed_client`` connection
    lifecycle. Returns the number of pages successfully stored.

    The embeds/grid tensors are always written to blob storage (keyed by
    ``page_no``, no page id needed). The ``visual_tokens`` reference is always
    set on the in-memory page object; when ``persist`` is True it is also written
    to the page row via ``storage.update_page(page.id, ...)``. Callers whose
    pages are not yet persisted (e.g. the inline pipeline, which persists pages
    in a later phase) pass ``persist=False`` and rely on that later persist to
    write the in-memory ``visual_tokens``.

    Failures for individual pages are logged and skipped; the image-embed
    feature is additive and must never fail the ingestion pipeline.
    """
    if not pages:
        return 0

    images_b64: list[str] = []
    metadatas: list[dict] = []
    for page in pages:
        img_bytes = await blob_storage.retrieve_file(page.image_storage_path)
        images_b64.append(base64.b64encode(img_bytes).decode("ascii"))
        metadatas.append(
            build_page_metadata(
                filename=filename,
                page_no=page.page_no,
                document_id=document_id,
                source=source,
            )
        )

    response = await visual_tokenize(embed_client, images_b64, metadata=metadatas)
    results = response.get("results", [])
    response_model = response.get("model")

    stored = 0
    for page, result in zip(pages, results):
        if not result.get("success") or not result.get("image_embeds_b64"):
            logger.warning(
                "Image-embed missing for page %s (doc %s): %s",
                page.id, document_id, result.get("error"),
            )
            continue

        model_slug = _slug_from_result(result, response_model)

        # Skip-guard (re-run/gap-fill perf): a page that already carries an
        # ``image_embeds`` ref for THIS model slug is already done — re-storing
        # would overwrite the same blobs/ref (correctness-neutral). Counted as
        # stored so the caller's success accounting is unchanged. A different
        # model slug still adds its own entry (no migration, see module docstring),
        # and a fresh ingest has no entry so all pages are (re)computed as before.
        if (page.visual_tokens or {}).get(model_slug):
            stored += 1
            continue
        try:
            # Embeds: zstd-compressed torch.save bytes — store verbatim.
            embeds_bytes = base64.b64decode(result["image_embeds_b64"])
            # Grid: uncompressed raw torch.save bytes — store verbatim.
            grid_bytes = base64.b64decode(result["image_grid_thw_b64"])

            embeds_blob_path = blob_storage.page_image_embeds_path(
                workspace_id, document_id, page.page_no, model_slug,
            )
            grid_blob_path = blob_storage.page_image_grid_path(
                workspace_id, document_id, page.page_no, model_slug,
            )
            await blob_storage.store_file(embeds_blob_path, embeds_bytes)
            await blob_storage.store_file(grid_blob_path, grid_bytes)

            entry = {
                "embed_kind": result.get("embed_kind", "image_embeds"),
                "embeds_blob_path": embeds_blob_path,
                "grid_blob_path": grid_blob_path,
                "num_image_tokens": result.get("num_image_tokens", 0),
                "hidden_dim": result.get("hidden_dim", 0),
                "image_grid_thw": result.get("image_grid_thw"),
                "created_at": datetime.now(UTC).isoformat(),
            }
            updated_vt = dict(page.visual_tokens or {})
            updated_vt[model_slug] = entry
            # Set the in-memory ref first so a same-phase consumer (document-chat
            # ingestion building image_embeds content blocks) sees it without a
            # re-fetch, and so a later persist phase writes it for the inline
            # pipeline (persist=False).
            page.visual_tokens = updated_vt
            if persist:
                await storage.update_page(page.id, visual_tokens=updated_vt)
            stored += 1
        except Exception as exc:  # noqa: BLE001 - additive feature, never fatal
            logger.warning(
                "Failed to persist image-embed for page %s (doc %s): %s",
                page.id, document_id, exc,
            )

    logger.info(
        "Image-embed phase: stored %d/%d page tensors for document %s",
        stored, len(pages), document_id,
    )
    return stored


async def build_image_embeds_content_blocks(
    *,
    blob_storage,
    pages,
    model_slug: str,
    v: Variables,
) -> list[dict]:
    """Load stored image-embeds for ``pages`` and build vLLM content blocks.

    For each page that has a ``visual_tokens[model_slug]`` reference with an
    ``embeds_blob_path``, the persisted blobs are loaded from blob storage and
    base64-encoded into a vLLM ``image_embeds`` content part of the form::

        {"type": "image_embeds",
         "image_embeds": {"image_embeds": <b64 raw torch.save>,
                          "image_grid_thw": <b64 raw torch.save>}}

    The embeds blob is stored zstd-compressed, so it is decompressed back to the
    raw ``torch.save`` bytes (which vLLM decodes via ``torch.load``) before
    base64-ing. The grid blob is stored uncompressed (already raw) and is
    base64'd as-is. Pages without an entry for ``model_slug`` are skipped, so
    the returned list is aligned to the included pages in input order.

    The two per-page blob reads go through the process-local ``cached_retrieve``
    reuse cache (keyed by blob path) so a live multi-turn interaction does not
    re-download the same blobs from the authoritative store every turn; ``v`` is
    needed to read the cache's config on first use.
    """
    blocks: list[dict] = []
    for page in pages:
        ref = (page.visual_tokens or {}).get(model_slug)
        if not ref or not ref.get("embeds_blob_path"):
            continue

        embeds_blob = await cached_retrieve(blob_storage, ref["embeds_blob_path"], v=v)
        if embeds_blob[:4] == _ZSTD_MAGIC:
            raw_embeds = zstandard.ZstdDecompressor().decompress(embeds_blob)
        else:
            raw_embeds = embeds_blob
        embeds_b64 = base64.b64encode(raw_embeds).decode("ascii")

        grid_blob = await cached_retrieve(blob_storage, ref["grid_blob_path"], v=v)
        grid_b64 = base64.b64encode(grid_blob).decode("ascii")

        blocks.append(
            {
                "type": "image_embeds",
                "image_embeds": {
                    "image_embeds": embeds_b64,
                    "image_grid_thw": grid_b64,
                },
            }
        )
    return blocks


async def generate_page_text_from_image_embeds(
    *,
    inference_client,
    blob_storage,
    page,
    model: str,
    model_slug: str,
    filename: str | None,
    instruction: str,
    max_tokens: int,
    v: Variables,
    logger: Logger,
    temperature: float = 0.2,
) -> str | None:
    """Generate faithful per-page Markdown text via the prompt-embeds LLM.

    This is the document-chat ingestion building block: it injects the page's
    precomputed ``image_embeds`` (visual-tokenizer output) into a single-turn
    chat completion on the inference LLM and returns the generated text, to be
    used as the page's memory content in place of OCR transcription.

    Returns ``None`` (caller skips the page, non-fatal) when the page has no
    stored ``image_embeds`` for ``model_slug``, or when the completion is empty
    or malformed.
    """
    blocks = await build_image_embeds_content_blocks(
        blob_storage=blob_storage, pages=[page], model_slug=model_slug, v=v,
    )
    if not blocks:
        return None

    # Interleave the page marker before its image_embeds block, then the
    # transcription instruction — byte-identical leading layout to the
    # /v1/documents/chat read path so vLLM prefix-caching still applies.
    content = [
        {"type": "text", "text": page_metadata_text(page, filename)},
        blocks[0],
        {"type": "text", "text": instruction},
    ]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    completion = await inference_client.chat_completions(payload, stream=False)
    try:
        text = completion["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logger.warning(
            "document-chat ingest: malformed completion for page %s (doc %s)",
            getattr(page, "id", "?"), getattr(page, "document_id", "?"),
        )
        return None

    if not text or not text.strip():
        return None
    return text
