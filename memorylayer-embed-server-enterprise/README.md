# memorylayer-embed-server-enterprise

AGPLv3 plugin overlay for `memorylayer-embed-server`. Adds the
Qwen3.5/3.6 **visual tokenizer** and the `/v1/visual-tokenize` API routes.

When installed alongside `memorylayer-embed-server`, the OSS server's
plugin discovery picks up this package automatically — no changes to
the server entrypoint or CLI.

## What it does

For each page image (plus stable metadata: filename, page number, source),
the tokenizer produces the **merged input-embedding sequence** — the token
embeddings for the metadata text with the vision tower's output scattered into
the image-placeholder positions. This `[seq_len, hidden]` tensor is exactly
what vLLM's `--enable-prompt-embeds` consumes, so a chat request can be given
page context by injecting precomputed embeds instead of re-encoding images at
query time.

Production happens with **HF Transformers only — no vLLM** (it just needs the
model's embedding layer + vision tower + merger), so it can colocate with
ColPali preprocessing at far lower VRAM than the full decode model. The legacy
`embed_kind=image_embeds` path (vision-encoder output only) is retained.

The tensor is returned base64-encoded as a `torch.save` payload, matching
vLLM's `safe_load_prompt_embeds`, so callers can persist it and feed it back
verbatim to a vLLM `--enable-prompt-embeds` chat profile.

## Install

```bash
cd .. # repository root
python3 scripts/dev.py embed
# (also brings in transformers, torch, accelerate, safetensors)
```

Core and the embedding base come from one OSS source selected by
`oss-source.toml`, `MEMORYLAYER_OSS_REF`, or `MEMORYLAYER_OSS_PATH`. See
[combined development](../docs/DEVELOPMENT.md#combined-development).

## Enable

Set `MEMORYLAYER_EMBED_VISUAL_TOKENIZER_ENABLED=true` in the environment of
the embed-server process. The plugin gates itself on this flag.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORYLAYER_EMBED_VISUAL_TOKENIZER_ENABLED` | `false` | Master enable flag. |
| `MEMORYLAYER_EMBED_VISUAL_TOKENIZER_MODEL` | `Qwen/Qwen3.6-27B-FP8` | HF model id. Qwen3.5/3.6 share the `Qwen3_5*` arch family (distinct from `Qwen3VL*`). |
| `MEMORYLAYER_EMBED_VISUAL_TOKENIZER_EMBED_KIND` | `prompt_embeds` | `prompt_embeds` (merged metadata+image sequence) or `image_embeds` (legacy vision-only). |
| `MEMORYLAYER_EMBED_VISUAL_TOKENIZER_VISION_ONLY` | `false` | Vision-only model load; only valid for `image_embeds` (ignored for `prompt_embeds`, which needs the text embedding layer). |
| `MEMORYLAYER_EMBED_VISUAL_TOKENIZER_PARTIAL_LOAD` | `true` | Prompt-embeds only: instantiate with the text decoder elided and the generation-only `lm_head` dropped, so only the embedding layer + vision tower + merger remain. Byte-identical output at a fraction of the VRAM (verified on 27B-FP8: 33.6 GB → **3.2 GB** resident weights; ~+5 GB transient during a page forward). Falls back to a full load if reduced-config instantiation fails. Designed to colocate with GLM-OCR + ColPali on a 24 GB L4. |
| `MEMORYLAYER_EMBED_VISUAL_TOKENIZER_TORCH_DTYPE` | `auto` | `auto` / `float16` / `bfloat16` / `float32`. |
| `MEMORYLAYER_EMBED_VISUAL_TOKENIZER_CACHE_DIR` | `/tmp/memorylayer-visual-cache` | On-disk safetensors cache root. |
| `MEMORYLAYER_EMBED_VISUAL_TOKENIZER_CACHE_MAX_GB` | `50` | LRU eviction threshold. |
| `MEMORYLAYER_EMBED_VISUAL_TOKENIZER_CACHE_TTL_HOURS` | `0` | `0` = no TTL. |

For `prompt_embeds`, the disk cache key includes a fingerprint of the page
metadata, so the same image with different metadata does not collide.

## API

`POST /v1/visual-tokenize`

```jsonc
{
  "images": ["<base64-png>", ...],
  "metadata": [{"filename": "report.pdf", "page_no": 0, "source": "..."}, ...],
  "return_tensors": true,   // include base64 torch.save payload per page
  "async_mode": false
}
```

Each result carries `prompt_embeds_b64` (when `return_tensors`),
`prompt_token_count`, `hidden_dim`, `embed_kind`, `model_slug`, and
`metadata_fingerprint`. Poll async jobs at
`GET /v1/visual-tokenize/status/{job_id}`.

## Serving the embeds

Run the model under a vLLM LLM profile on the OSS embed-server with
prompt-embeds enabled, e.g.:

```bash
MEMORYLAYER_EMBED_LLM_ENABLED=true
MEMORYLAYER_EMBED_LLM_PROFILES=qwen36
MEMORYLAYER_EMBED_LLM_PROFILE_QWEN36_MODEL=Qwen/Qwen3.6-27B-FP8
MEMORYLAYER_EMBED_LLM_PROFILE_QWEN36_ENABLE_PROMPT_EMBEDS=true
```

Then send chat requests with a `{"type": "prompt_embeds", "data": <b64>}`
content block per page ahead of the user's text block.

## Release prerequisites and license

Requires the matching MemoryLayer 0.2.0 embedding server; see
[development requirements](../docs/DEVELOPMENT.md). The extensions use
[AGPL-3.0-only](LICENSE). Model weights retain their own licenses.
