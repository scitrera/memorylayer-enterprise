# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Configuration keys for the enterprise visual-tokenizer overlay.

These keys used to live in ``memorylayer_embed_server.config`` but moved
here when the visual-tokenizer service was relocated into the enterprise
plugin package. Default values preserved verbatim from the OSS source.
"""

EMBED_SERVER_VISUAL_TOKENIZER_ENABLED = 'MEMORYLAYER_EMBED_VISUAL_TOKENIZER_ENABLED'
DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_ENABLED = False

EMBED_SERVER_VISUAL_TOKENIZER_MODEL = 'MEMORYLAYER_EMBED_VISUAL_TOKENIZER_MODEL'
DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_MODEL = 'Qwen/Qwen3.6-27B-FP8'

# Output kind. Only ``image_embeds`` is supported: the provider emits the
# projected vision-tower per-image embeds + ``image_grid_thw`` for a vLLM
# ``image_embeds`` content part (the flat ``prompt_embeds`` path was dropped
# because it gave image tokens 1-D positions, defeating M-RoPE).
EMBED_SERVER_VISUAL_TOKENIZER_EMBED_KIND = 'MEMORYLAYER_EMBED_VISUAL_TOKENIZER_EMBED_KIND'
DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_EMBED_KIND = 'image_embeds'

# Vision-only model load (saves VRAM): load just the vision tower when the
# checkpoint exposes a standalone ``Qwen3_5VisionModel``, falling back to the
# full multimodal model otherwise.
EMBED_SERVER_VISUAL_TOKENIZER_VISION_ONLY = 'MEMORYLAYER_EMBED_VISUAL_TOKENIZER_VISION_ONLY'
DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_VISION_ONLY = False

# Partial load: instantiate the full multimodal model with the text decoder
# layers elided so only the input-embedding layer + vision tower + merger are
# materialized. The decoder is never run for image-embed extraction, so this
# cuts VRAM dramatically (e.g. ~27B → a few GB) while producing identical
# vision embeds. Falls back to a full load if the reduced-config
# instantiation fails.
EMBED_SERVER_VISUAL_TOKENIZER_PARTIAL_LOAD = 'MEMORYLAYER_EMBED_VISUAL_TOKENIZER_PARTIAL_LOAD'
DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_PARTIAL_LOAD = True

EMBED_SERVER_VISUAL_TOKENIZER_BATCH_SIZE = 'MEMORYLAYER_EMBED_VISUAL_TOKENIZER_BATCH_SIZE'
DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_BATCH_SIZE = 4

EMBED_SERVER_VISUAL_TOKENIZER_MAX_JOBS = 'MEMORYLAYER_EMBED_VISUAL_TOKENIZER_MAX_JOBS'
DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_MAX_JOBS = 64

EMBED_SERVER_VISUAL_TOKENIZER_JOB_TTL_SECONDS = 'MEMORYLAYER_EMBED_VISUAL_TOKENIZER_JOB_TTL_SECONDS'
DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_JOB_TTL_SECONDS = 3600.0

EMBED_SERVER_VISUAL_TOKENIZER_CACHE_DIR = 'MEMORYLAYER_EMBED_VISUAL_TOKENIZER_CACHE_DIR'
DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_CACHE_DIR = '/tmp/memorylayer-visual-cache'

# The visual-tokenizer cache is an EPHEMERAL produce-and-forward buffer: the
# embed-server computes embeds and hands them to the caller, which persists them
# in the authoritative blob store. The cache only avoids recompute on retries,
# so keep it tight (small size cap + TTL) rather than a large retention store.
EMBED_SERVER_VISUAL_TOKENIZER_CACHE_MAX_GB = 'MEMORYLAYER_EMBED_VISUAL_TOKENIZER_CACHE_MAX_GB'
DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_CACHE_MAX_GB = 4.0

EMBED_SERVER_VISUAL_TOKENIZER_CACHE_TTL_HOURS = 'MEMORYLAYER_EMBED_VISUAL_TOKENIZER_CACHE_TTL_HOURS'
DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_CACHE_TTL_HOURS = 24

EMBED_SERVER_VISUAL_TOKENIZER_TORCH_DTYPE = 'MEMORYLAYER_EMBED_VISUAL_TOKENIZER_TORCH_DTYPE'
DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_TORCH_DTYPE = 'auto'

# Optional downsample cap applied before the image hits the Qwen processor /
# vision tower. The value is the maximum allowed size (in pixels) of the
# image's LARGEST dimension (max of width/height); larger images are scaled
# down preserving aspect ratio so vision-token count (which scales with pixel
# area) drops. 0 disables the resize entirely (current behavior preserved).
EMBED_SERVER_VISUAL_TOKENIZER_MAX_IMAGE_DIM = 'MEMORYLAYER_EMBED_VISUAL_TOKENIZER_MAX_IMAGE_DIM'
DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_MAX_IMAGE_DIM = 0

# ---------------------------------------------------------------------------
# GLiNER2 NER service (POST /v1/ner)
#
# Productionizes the GLiNER2 typed-NER endpoint into the main embed server so
# the app-side ``GLiNER2ExtractionService`` can call ``/v1/ner`` on the same
# host (:61051) it already uses for embeddings, instead of a separate
# standalone process.
# ---------------------------------------------------------------------------

# Disabled by default: the gliner2 model is NOT bundled into the base image's
# HF cache, so on first enable it is downloaded to the mounted HF-cache volume.
# Operators opt in explicitly (mirrors the visual-tokenizer ENABLED default).
EMBED_SERVER_GLINER2_ENABLED = 'MEMORYLAYER_EMBED_GLINER2_ENABLED'
DEFAULT_EMBED_SERVER_GLINER2_ENABLED = False

EMBED_SERVER_GLINER2_MODEL = 'MEMORYLAYER_EMBED_GLINER2_MODEL'
DEFAULT_EMBED_SERVER_GLINER2_MODEL = 'fastino/gliner2-base-v1'

# Default entity labels when a request omits ``labels``. The authoritative
# label taxonomy lives app-side (EntityType -> label); this is only the
# fallback used for ad-hoc requests that send no labels.
EMBED_SERVER_GLINER2_LABELS = 'MEMORYLAYER_EMBED_GLINER2_LABELS'
DEFAULT_EMBED_SERVER_GLINER2_LABELS = 'person,organization,location,event,product'
