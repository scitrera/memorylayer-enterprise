# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Central configuration for MemoryLayer Enterprise.

NOTE: Most configurations should live closest to where it's used (in service base.py files).
Only truly central configuration belongs here - e.g., which service provider to use.

Individual service configuration (like tiering thresholds) should be defined
in the service's own module and read via v.environ() in the plugin's initialize().
"""

# make OSS common keys available here as passthrough

# LLM config lives co-located with the service in OSS (not in central config)

# Central service selection environment variables
# These control which plugin provider is used for each service

# Context service provider selection -- only exists in enterprise
MEMORYLAYER_CONTEXT_SERVICE = 'MEMORYLAYER_CONTEXT_SERVICE'
DEFAULT_MEMORYLAYER_CONTEXT_SERVICE = 'default'

# Tiering service provider selection -- only exists in enterprise
MEMORYLAYER_TIERING_SERVICE = 'MEMORYLAYER_TIERING_SERVICE'
DEFAULT_MEMORYLAYER_TIERING_SERVICE = 'default'

# Trajectory service provider selection -- only exists in enterprise
MEMORYLAYER_TRAJECTORY_SERVICE = 'MEMORYLAYER_TRAJECTORY_SERVICE'
DEFAULT_MEMORYLAYER_TRAJECTORY_SERVICE = 'default'

# Memory service provider selection (enterprise uses 'saas')
MEMORYLAYER_MEMORY_SERVICE = 'MEMORYLAYER_MEMORY_SERVICE'
DEFAULT_MEMORYLAYER_MEMORY_SERVICE = 'saas'

# Storage service provider selection (enterprise uses 'postgresql')
DEFAULT_MEMORYLAYER_STORAGE_BACKEND = 'postgresql'

# Opt in to caller identity headers only for operator-controlled LLM gateways.
# MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS may supply an explicit hostname allowlist.
# The public distribution sends no tenant identity headers by default.
DEFAULT_MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS = ''

# Compression service provider selection -- only exists in enterprise
MEMORYLAYER_COMPRESSION_SERVICE = 'MEMORYLAYER_COMPRESSION_SERVICE'
DEFAULT_MEMORYLAYER_COMPRESSION_SERVICE = 'default'

# Redis Cache by default for enterprise
DEFAULT_MEMORYLAYER_CACHE_SERVICE = 'redis'

# Use multivector compatible service by default for enterprise
DEFAULT_MEMORYLAYER_EMBEDDING_SERVICE = 'mv'  # use multivector compatible embedding service for enterprise
# Enterprise default routes all heavy embedding through memorylayer-embed-server.
# The legacy in-process providers (qwen3-vl, colpali) were removed in the
# Aether-convergence cleanup; operators must run an embed-server peer.
DEFAULT_MEMORYLAYER_EMBEDDING_PROVIDER = 'embed_server'

# Reranking also delegates to memorylayer-embed-server (MaxSim via /v1/score).
DEFAULT_MEMORYLAYER_RERANKER_PROVIDER = 'embed_server'

# Enterprise graph reads use Apache AGE by default. The Postgres image built by
# postgres-container/ includes AGE; the AGE plugins fail fast if selected
# against a non-PostgreSQL storage backend. OSS keeps its relational defaults.
DEFAULT_MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER = 'age'
DEFAULT_MEMORYLAYER_GRAPH_QUERY_PROVIDER = 'age'

# ============================================
# Config Service
# ============================================
MEMORYLAYER_CONFIG_SERVICE = 'MEMORYLAYER_CONFIG_SERVICE'
DEFAULT_MEMORYLAYER_CONFIG_SERVICE = 'default'

# ============================================
# Tenant Service
# ============================================
MEMORYLAYER_TENANT_SERVICE = 'MEMORYLAYER_TENANT_SERVICE'
DEFAULT_MEMORYLAYER_TENANT_SERVICE = 'default'

# ============================================
# Embed Server Connection (relocated to OSS in Phase 3 of the Aether
# convergence; re-exported here so legacy enterprise imports still resolve).
# ============================================
from memorylayer_server.config import (  # noqa: E402,F401
    DEFAULT_MEMORYLAYER_EMBED_AETHER_TARGET,
    DEFAULT_MEMORYLAYER_EMBED_SERVER_TIMEOUT,
    DEFAULT_MEMORYLAYER_EMBED_SERVER_URL,
    DEFAULT_MEMORYLAYER_EMBED_TRANSPORT,
    MEMORYLAYER_EMBED_AETHER_TARGET,
    MEMORYLAYER_EMBED_SERVER_TIMEOUT,
    MEMORYLAYER_EMBED_SERVER_URL,
    MEMORYLAYER_EMBED_TRANSPORT,
)

# ============================================
# Service-selector env-var-name constants relocated to OSS. The enterprise
# preconfigure hook (dependencies.py) imports these MEMORYLAYER_*_(SERVICE|
# BACKEND|PROVIDER) names from THIS module, but only their DEFAULT_ siblings
# survived a prior config split — re-export the canonical OSS definitions here
# so the hook resolves (the server crashed at boot with ImportError otherwise).
# ============================================
from memorylayer_server.config import (  # noqa: E402,F401
    MEMORYLAYER_ASSOCIATION_SERVICE,
    MEMORYLAYER_AUDIT_SERVICE,
    MEMORYLAYER_AUTHENTICATION_SERVICE,
    MEMORYLAYER_AUTHORIZATION_SERVICE,
    MEMORYLAYER_CACHE_SERVICE,
    MEMORYLAYER_EMBEDDING_PROVIDER,
    MEMORYLAYER_EMBEDDING_SERVICE,
    MEMORYLAYER_ENTITY_REGISTRY_PROVIDER,
    MEMORYLAYER_EXTRACTION_SERVICE,
    MEMORYLAYER_KNOWLEDGEBASE_PROVIDER,
    MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS,
    MEMORYLAYER_METRICS_SERVICE,
    MEMORYLAYER_RATE_LIMIT_SERVICE,
    MEMORYLAYER_REFLECT_SERVICE,
    MEMORYLAYER_SESSION_SERVICE,
    MEMORYLAYER_STORAGE_BACKEND,
    MEMORYLAYER_WORKSPACE_IMPLICIT_CREATE,
    MEMORYLAYER_WORKSPACE_SERVICE,
)
from memorylayer_server.services.llm.base import (  # noqa: E402,F401
    MEMORYLAYER_LLM_REGISTRY,
    MEMORYLAYER_LLM_SERVICE,
)

# ============================================
# tokenary inference-client adapter (enterprise-only)
# ============================================
# Opt-in: set MEMORYLAYER_EMBED_SERVER_SERVICE=tokenary to swap the embed-server
# REST client for the TokenaryClient (reroutes single-vec, multi-vec,
# image, and MaxSim score with no OSS change). Each per-concern URL points at one
# tokenary instance (tokenary = one model/process). Each defaults to "" and is
# resolved to MEMORYLAYER_EMBED_SERVER_URL at read time, so a single-instance
# deployment and mixed deployments both work. Chat (MEMORYLAYER_EMBED_LLM_SERVER_URL)
# and NER (MEMORYLAYER_GLINER2_NER_URL) already have their own URL knobs.
MEMORYLAYER_TOKENARY_TEXTVEC_URL = 'MEMORYLAYER_TOKENARY_TEXTVEC_URL'
DEFAULT_MEMORYLAYER_TOKENARY_TEXTVEC_URL = ''  # -> MEMORYLAYER_EMBED_SERVER_URL
MEMORYLAYER_TOKENARY_MULTIVEC_URL = 'MEMORYLAYER_TOKENARY_MULTIVEC_URL'
DEFAULT_MEMORYLAYER_TOKENARY_MULTIVEC_URL = ''  # -> MEMORYLAYER_EMBED_SERVER_URL
MEMORYLAYER_TOKENARY_SCORE_URL = 'MEMORYLAYER_TOKENARY_SCORE_URL'
DEFAULT_MEMORYLAYER_TOKENARY_SCORE_URL = ''  # usually == multivec instance
MEMORYLAYER_TOKENARY_VISUALTOK_URL = 'MEMORYLAYER_TOKENARY_VISUALTOK_URL'
DEFAULT_MEMORYLAYER_TOKENARY_VISUALTOK_URL = ''  # -> MEMORYLAYER_EMBED_SERVER_URL

# ============================================
# Blob Storage
# ============================================
MEMORYLAYER_BLOB_STORAGE_TYPE = 'MEMORYLAYER_BLOB_STORAGE_TYPE'
DEFAULT_MEMORYLAYER_BLOB_STORAGE_TYPE = 'local'
MEMORYLAYER_BLOB_STORAGE_BASE_PATH = 'MEMORYLAYER_BLOB_STORAGE_BASE_PATH'
DEFAULT_MEMORYLAYER_BLOB_STORAGE_BASE_PATH = '/var/lib/memorylayer/blobs'
MEMORYLAYER_BLOB_S3_ENDPOINT_URL = 'MEMORYLAYER_BLOB_S3_ENDPOINT_URL'
MEMORYLAYER_BLOB_S3_ACCESS_KEY = 'MEMORYLAYER_BLOB_S3_ACCESS_KEY'
MEMORYLAYER_BLOB_S3_SECRET_KEY = 'MEMORYLAYER_BLOB_S3_SECRET_KEY'
MEMORYLAYER_BLOB_S3_REGION = 'MEMORYLAYER_BLOB_S3_REGION'

# ============================================
# Document Ingestion
# ============================================
MEMORYLAYER_DOCUMENT_INGESTION_SERVICE = 'MEMORYLAYER_DOCUMENT_INGESTION_SERVICE'
DEFAULT_MEMORYLAYER_DOCUMENT_INGESTION_SERVICE = 'default'
MEMORYLAYER_DOCUMENT_MAX_FILE_SIZE = 'MEMORYLAYER_DOCUMENT_MAX_FILE_SIZE'
DEFAULT_MEMORYLAYER_DOCUMENT_MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Page image sent to the multimodal LLM for fact decomposition. The stored render
# stays full-res (200 DPI PNG — for viewing + OCR/transcription); only the copy
# ATTACHED to the extraction LLM is downscaled + re-encoded as JPEG, to keep the
# request body under the AI-gateway ingress client_max_body_size (avoids the
# nginx 413) and cut multimodal latency/cost. Max longest-side px; 0 disables
# downscaling (still JPEG-encoded). Quality is the JPEG quality (1-95).
MEMORYLAYER_LLM_IMAGE_MAX_DIM = 'MEMORYLAYER_LLM_IMAGE_MAX_DIM'
DEFAULT_MEMORYLAYER_LLM_IMAGE_MAX_DIM = 1200
MEMORYLAYER_LLM_IMAGE_JPEG_QUALITY = 'MEMORYLAYER_LLM_IMAGE_JPEG_QUALITY'
DEFAULT_MEMORYLAYER_LLM_IMAGE_JPEG_QUALITY = 85

# Toggle the transcription (OCR) phase of the ingestion pipeline. When false,
# document_render schedules document_embed directly (skipping transcribe), so
# pages carry no transcript. Useful when only image-derived signals are needed
# (multi-vector + visual-tokenizer prompt-embeds), or when no OCR backend is
# available. Default true (transcribe runs).
MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED = 'MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED'
DEFAULT_MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED = True

# ============================================
# Transcription service selection
# ============================================
# 'embed_server' (default) routes page OCR through an embed-server process via
# POST /v1/transcribe. 'direct' runs the cascade in-process against model
# endpoints (sparkrun-served lanes, or anything OpenAI-compatible), which is
# what removes embed-server from the deployment topology.
MEMORYLAYER_TRANSCRIPTION_SERVICE = 'MEMORYLAYER_TRANSCRIPTION_SERVICE'
DEFAULT_MEMORYLAYER_TRANSCRIPTION_SERVICE = 'embed_server'

# Ordered, comma-separated cascade of profile names tried per page; first
# success wins. Each name resolves the per-profile keys below. N rungs are
# supported -- 'unlimited' alone, or e.g. 'unlimited,gemini' once a hosted
# fallback is reachable through an OpenAI-compatible endpoint.
MEMORYLAYER_TRANSCRIBE_CASCADE = 'MEMORYLAYER_TRANSCRIBE_CASCADE'
DEFAULT_MEMORYLAYER_TRANSCRIBE_CASCADE = 'unlimited'

# Per-profile keys: MEMORYLAYER_TRANSCRIBE_PROFILE_<NAME>_<FIELD>, where <NAME>
# is the upper-cased cascade entry. Fields:
#   URL                  base URL including the /v1 suffix (required)
#   CONTRACT             unlimited_ocr | generic_markdown  (default unlimited_ocr)
#   MODEL                model id sent in the request body
#   AUTH                 none | bearer | modal_proxy       (default none)
#   AUTH_TOKEN           bearer token
#   AUTH_KEY/AUTH_SECRET modal proxy-auth credentials
#   MAX_TOKENS           per-model output cap
#   TIMEOUT_SEC          steady-state request budget
#   COLD_START_TIMEOUT_SEC  budget for this profile's FIRST request; an
#                        on-demand endpoint boots a GPU on demand, so the first
#                        call can take minutes while steady state is seconds
#   NGRAM_SIZE/WINDOW_SIZE  unlimited_ocr decode knobs (WINDOW_SIZE 1024 for
#                        multi-page input)
# Pages in flight at once per worker process, and figure captions in flight per
# page (separate pools -- pages hit the OCR lane, captions hit the LLM profile).
#
# FLEET load is (worker replicas x this), and the proxy in front of the lane
# enforces the fleet-wide ceiling; keep the two sized against each other. The
# lane also shares its GPU with the chat lane, so bigger is not free.
MEMORYLAYER_TRANSCRIBE_CONCURRENCY = 'MEMORYLAYER_TRANSCRIBE_CONCURRENCY'

# Attempts per page when the upstream reports overload (503/429/502/504). The
# proxy's queue is fail-fast: without a retry, hitting the ceiling turns into a
# DROPPED PAGE rather than backpressure.
MEMORYLAYER_TRANSCRIBE_OVERLOAD_ATTEMPTS = 'MEMORYLAYER_TRANSCRIBE_OVERLOAD_ATTEMPTS'
MEMORYLAYER_TRANSCRIBE_OVERLOAD_BACKOFF_SEC = 'MEMORYLAYER_TRANSCRIBE_OVERLOAD_BACKOFF_SEC'

MEMORYLAYER_TRANSCRIBE_PROFILE_PREFIX = 'MEMORYLAYER_TRANSCRIBE_PROFILE_'
DEFAULT_MEMORYLAYER_TRANSCRIBE_CONTRACT = 'unlimited_ocr'

# ============================================
# Figure captioning
# ============================================
# A grounded OCR model marks an illustration with a box and emits NO text for
# it, so an uncaptioned transcript records only that something was there.
# Captioning each cropped figure with the (multimodal) default inference model
# puts what it depicts into the transcript, so a consumer can tell from the text
# alone whether it needs to fetch the crop.
#
# Routed through the STANDARD LLM PROFILES mechanism (LLMService.complete with
# a profile), not the document-chat prompt-embeds client -- this is an ordinary
# multimodal completion, and the default profile is a multimodal
# OpenAI-compatible model.
#
# Costs one LLM call per figure. Degrades gracefully: with no LLM service
# configured, captioning is skipped and figures keep their bare placeholder.
MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTIONS = 'MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTIONS'
DEFAULT_MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTIONS = True

# LLM profile used for captions.
MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_PROFILE = 'MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_PROFILE'
DEFAULT_MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_PROFILE = 'default'

# Optional model override. Empty (the default) means "whatever model the chosen
# profile is configured with" -- pinning a name here would silently diverge from
# the profile's own configuration.
MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_MODEL = 'MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_MODEL'
DEFAULT_MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_MODEL = ''

# Captioning is a description task, not a reasoning one. The default profile is
# a thinking model whose chain-of-thought lands in the response CONTENT (it
# returns no separate reasoning_content), so with thinking on the 'caption' is
# raw reasoning. 'none' suppresses it: measured ~700-1024 tokens of
# chain-of-thought per figure before, ~21 tokens of clean caption after.
# Set empty to omit the field for a provider that rejects it.
MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_REASONING_EFFORT = 'MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_REASONING_EFFORT'
DEFAULT_MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_REASONING_EFFORT = 'none'

MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_MAX_TOKENS = 'MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_MAX_TOKENS'
MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_CONTEXT_CHARS = 'MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_CONTEXT_CHARS'
MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_INSTRUCTION = 'MEMORYLAYER_TRANSCRIBE_FIGURE_CAPTION_INSTRUCTION'

# ============================================
# Document Blob Garbage Collection
# ============================================
# Periodic, low-priority reconciliation sweep that reclaims orphaned blob-store
# artifacts the delete-time cleanup missed (failed deletes, crashes, pre-existing
# stale data such as the obsolete prompt_embeds/ subdirs from the old pipeline).
MEMORYLAYER_DOCUMENT_BLOB_GC_ENABLED = 'MEMORYLAYER_DOCUMENT_BLOB_GC_ENABLED'
DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_GC_ENABLED = True
# Interval between sweeps. Default 6h.
MEMORYLAYER_DOCUMENT_BLOB_GC_INTERVAL_SEC = 'MEMORYLAYER_DOCUMENT_BLOB_GC_INTERVAL_SEC'
DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_GC_INTERVAL_SEC = 21600  # 6 hours
# Skip directories modified within this grace window to avoid racing in-flight
# ingestion. Default 1h.
MEMORYLAYER_DOCUMENT_BLOB_GC_GRACE_SEC = 'MEMORYLAYER_DOCUMENT_BLOB_GC_GRACE_SEC'
DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_GC_GRACE_SEC = 3600  # 1 hour

# ============================================
# Periodic ingestion-job orphan-reconcile sweep. Safety net that cancels
# queued/running ingestion_jobs whose referenced documents are ALL already
# completed -- catches jobs that slip past create-time coalescing +
# reconcile-on-complete (Aether task replays, crashed workers that marked a job
# started then died at 0% forever).
MEMORYLAYER_JOB_RECONCILE_ENABLED = 'MEMORYLAYER_JOB_RECONCILE_ENABLED'
DEFAULT_MEMORYLAYER_JOB_RECONCILE_ENABLED = True
# Interval between sweeps. Default 1h -- reconciliation is a cheap single UPDATE
# and the accumulation is otherwise unbounded.
MEMORYLAYER_JOB_RECONCILE_INTERVAL_SEC = 'MEMORYLAYER_JOB_RECONCILE_INTERVAL_SEC'
DEFAULT_MEMORYLAYER_JOB_RECONCILE_INTERVAL_SEC = 3600  # 1 hour

# ============================================
# Visual Tokenizer (prompt-embeds)
# ============================================
# Feature flag: during the embed phase, precompute per-page prompt-embeds via
# the enterprise embed-server's /v1/visual-tokenize endpoint and persist the
# tensor to blob storage, referenced from DocumentPage.visual_tokens keyed by
# model slug. Default off; requires the embed-server to run the enterprise
# overlay with the visual tokenizer enabled.
MEMORYLAYER_VISUAL_TOKENIZER_ENABLED = 'MEMORYLAYER_VISUAL_TOKENIZER_ENABLED'
DEFAULT_MEMORYLAYER_VISUAL_TOKENIZER_ENABLED = False

# ============================================
# Grounded Document Chat (prompt-embed inference)
# ============================================
# Feature flag for the POST /v1/documents/chat route. Enabled by default in
# enterprise; deployments without a prompt-embeds-capable inference target can
# still disable it explicitly.
MEMORYLAYER_DOCUMENT_CHAT_ENABLED = 'MEMORYLAYER_DOCUMENT_CHAT_ENABLED'
DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_ENABLED = True

# LLM inference target — SEPARATE from the (L4) preprocessing embed-server.
# In production the prompt-embeds-enabled vLLM runs on a high-VRAM node; it need
# not be a memorylayer-embed-server (any OpenAI-compatible vLLM with
# --enable-prompt-embeds works). When BOTH URL and aether target are empty, the
# endpoint falls back to the default preprocessing embed client (colocated dev,
# e.g. the GB10 running both roles).
MEMORYLAYER_EMBED_LLM_TRANSPORT = 'MEMORYLAYER_EMBED_LLM_TRANSPORT'
DEFAULT_MEMORYLAYER_EMBED_LLM_TRANSPORT = 'http'  # 'http' | 'aether'
MEMORYLAYER_EMBED_LLM_SERVER_URL = 'MEMORYLAYER_EMBED_LLM_SERVER_URL'
DEFAULT_MEMORYLAYER_EMBED_LLM_SERVER_URL = ''
MEMORYLAYER_EMBED_LLM_AETHER_TARGET = 'MEMORYLAYER_EMBED_LLM_AETHER_TARGET'
DEFAULT_MEMORYLAYER_EMBED_LLM_AETHER_TARGET = ''
MEMORYLAYER_EMBED_LLM_TIMEOUT = 'MEMORYLAYER_EMBED_LLM_TIMEOUT'
DEFAULT_MEMORYLAYER_EMBED_LLM_TIMEOUT = 600.0
# Default model name used when a /documents/chat request omits "model".
MEMORYLAYER_EMBED_LLM_DEFAULT_MODEL = 'MEMORYLAYER_EMBED_LLM_DEFAULT_MODEL'
DEFAULT_MEMORYLAYER_EMBED_LLM_DEFAULT_MODEL = 'Qwen/Qwen3.6-27B-FP8'

# ============================================
# Document-Chat Ingestion (memory text from image_embeds)
# ============================================
# Feature flag: during the embed phase, when a page has no transcript (OCR
# transcription disabled), generate the page's memory text by prompting the
# prompt-embeds inference LLM against the page's precomputed image_embeds — the
# same mechanism as /v1/documents/chat — instead of OCR transcription. The
# generated text is stored on the page transcript field so the rest of the
# pipeline (single-vector embedding, memory creation, KB folding) is unchanged.
# Requires the visual tokenizer (image_embeds) to be enabled; pages without
# image_embeds for the inference model slug are skipped (non-fatal). Default off.
MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED = 'MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED'
DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED = False

# Instruction appended after the page's image_embeds when generating memory
# text. Asks for a faithful markdown rendering (transcription-equivalent) so the
# produced text matches what OCR transcription would have yielded. Overridable
# per-document via DocumentExtractionOptions.system_prompt.
MEMORYLAYER_DOCUMENT_CHAT_INGEST_PROMPT = 'MEMORYLAYER_DOCUMENT_CHAT_INGEST_PROMPT'
DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_PROMPT = (
    "Transcribe this page into clean, faithful Markdown. Preserve all text, "
    "headings, lists, and table structure. Describe figures, charts, and images "
    "concisely in brackets. Output only the page content, with no preamble or "
    "commentary."
)
# Max tokens for the generated per-page memory text.
MEMORYLAYER_DOCUMENT_CHAT_INGEST_MAX_TOKENS = 'MEMORYLAYER_DOCUMENT_CHAT_INGEST_MAX_TOKENS'
DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_MAX_TOKENS = 8192

# Process-local LRU cache for image-embed/grid blob reads on the chat read path.
# Live multi-turn interactions re-read the same per-page blobs every turn; this
# byte-bounded + TTL'd cache lets a live interaction reuse them instead of
# re-fetching from the authoritative blob store each turn. MAX_GB == 0 disables
# the cache entirely (always re-fetch). See blob_cache.py.
MEMORYLAYER_DOCUMENT_BLOB_CACHE_MAX_GB = 'MEMORYLAYER_DOCUMENT_BLOB_CACHE_MAX_GB'
DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_CACHE_MAX_GB = 2.0
MEMORYLAYER_DOCUMENT_BLOB_CACHE_TTL_SEC = 'MEMORYLAYER_DOCUMENT_BLOB_CACHE_TTL_SEC'
DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_CACHE_TTL_SEC = 3600

# ============================================
# Data-Connectors Integration
# ============================================
# NOTE: there is deliberately no flag for routing page-image writes through
# data-connectors. One existed (MEMORYLAYER_DC_BLOB_WRITE_VIA_DC) and was never
# read by anything: page images, tensors, and source blobs are written by
# BlobStorageService, and data-connectors has no endpoint that would accept
# them. Its VFS catalog is one entry per user-visible file -- a document's ~3N
# derived per-page artifacts do not fit that model, and routing them there
# would also give up the deterministic paths that delete_tree, blob_gc, and
# reprocess-from-phase address blobs by.
# Aether topic for data-connectors service.
# Implementation-only address (no specifier) so the gateway routes to any
# healthy replica. data-connectors self-registers as
# sv::data-connectors:{hostname} by default (see
# data_connectors/server/aether_service.py::_resolve_specifier), so pinning
# to ::default would never match.
MEMORYLAYER_DATA_CONNECTORS_TOPIC = 'MEMORYLAYER_DATA_CONNECTORS_TOPIC'
DEFAULT_MEMORYLAYER_DATA_CONNECTORS_TOPIC = 'sv::data-connectors'

# In-flight freshness window for idempotent ingestion. When a doc_added arrives
# for a document already PROCESSING/PENDING* whose processing_started_at is
# within this window, another worker is assumed to own it and the redelivery is
# a NO-OP. Past the window (a crashed/orphaned PROCESSING doc) the document is
# re-driven via gap-fill. Backs the document-level in-flight protection (Aether
# POOL tasks have no schedule-time dedup). Default 900s (15 min).
MEMORYLAYER_INGEST_INFLIGHT_TTL = 'MEMORYLAYER_INGEST_INFLIGHT_TTL'
DEFAULT_MEMORYLAYER_INGEST_INFLIGHT_TTL = 900

# ============================================
# Document Verify / Reconcile Sweep (Phase 3)
# ============================================
# Periodic reconcile pass that heals documents left incomplete by crashed
# workers: it gap-analyzes PARTIAL/FAILED/stale-PROCESSING documents (and
# re-schedules any never-run fact decomposition) and resumes the fixable ones
# at their first missing phase via the chained document_* pipeline. Reuses the
# same analyze_document_gaps + resume_at primitives as doc_added. Also runnable
# on-demand against a single document.
MEMORYLAYER_DOC_VERIFY_ENABLED = 'MEMORYLAYER_DOC_VERIFY_ENABLED'
DEFAULT_MEMORYLAYER_DOC_VERIFY_ENABLED = True
# Interval between scheduled sweeps. Default 6h.
MEMORYLAYER_DOC_VERIFY_INTERVAL = 'MEMORYLAYER_DOC_VERIFY_INTERVAL'
DEFAULT_MEMORYLAYER_DOC_VERIFY_INTERVAL = 21600  # 6 hours
# Maximum documents inspected per status, per workspace, per sweep run. Bounds
# the work a single recurring task does (no silent truncation — the count swept
# is logged). Default 100.
MEMORYLAYER_DOC_VERIFY_BATCH_LIMIT = 'MEMORYLAYER_DOC_VERIFY_BATCH_LIMIT'
DEFAULT_MEMORYLAYER_DOC_VERIFY_BATCH_LIMIT = 100
# Per-document cross-run reprocess cap (stored in doc.metadata
# 'reprocess_attempts'). At/over the cap the sweep marks the doc terminally
# FAILED ('reconcile_terminal': true) and never re-drives it again. Prevents
# permanently-broken docs (e.g. corrupt files failing render every run) from
# churning every sweep cycle indefinitely. Default 5.
MEMORYLAYER_DOC_VERIFY_MAX_ATTEMPTS = 'MEMORYLAYER_DOC_VERIFY_MAX_ATTEMPTS'
DEFAULT_MEMORYLAYER_DOC_VERIFY_MAX_ATTEMPTS = 5

# ============================================
# Dataset Service
# ============================================
MEMORYLAYER_DATASET_SERVICE = 'MEMORYLAYER_DATASET_SERVICE'
DEFAULT_MEMORYLAYER_DATASET_SERVICE = 'default'
MEMORYLAYER_DATASET_MAX_FILE_SIZE = 'MEMORYLAYER_DATASET_MAX_FILE_SIZE'
DEFAULT_MEMORYLAYER_DATASET_MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

# ============================================
# Audit Service (enterprise default: postgresql)
# ============================================
DEFAULT_MEMORYLAYER_AUDIT_SERVICE = 'postgresql'

# ============================================
# Rate Limiting (enterprise default: aether-kv)
# ============================================

DEFAULT_MEMORYLAYER_RATE_LIMIT_SERVICE = 'aether-kv'

# ============================================
# Metrics (enterprise default: prometheus)
# ============================================

DEFAULT_MEMORYLAYER_METRICS_SERVICE = 'prometheus'

# ============================================
# Encryption (for DataProvider credentials)
# ============================================
MEMORYLAYER_ENCRYPTION_KEY = 'MEMORYLAYER_ENCRYPTION_KEY'

# ============================================
# Workflow Registration
# ============================================
# When true, MemoryLayer Enterprise registers an Aether workflow rule that
# automatically triggers a kb_update task whenever a document finishes
# ingesting (memorylayer.ingest_complete event).  Default on; operators can
# disable by setting the env var to 'false'.
MEMORYLAYER_AUTO_KB_UPDATE_ON_INGEST = 'MEMORYLAYER_AUTO_KB_UPDATE_ON_INGEST'
DEFAULT_MEMORYLAYER_AUTO_KB_UPDATE_ON_INGEST = True

# ============================================
# Representation Consolidation (P3 maintained-profile layer)
# ============================================
# Sub-flag (under MEMORYLAYER_REPRESENTATION_ENABLED) for the leased/coalesced
# background task that DERIVES + PERSISTS a maintained per-(observer,subject)
# representation (profile + derived beliefs) so get_representation can serve a
# maintained profile cheaply instead of deriving on every call.
#
# Ships DARK (default OFF): the consolidation task is registered but is a no-op,
# and get_representation alway derives on-demand (the 0a79fb9 behavior) until
# this flag is turned on. Any failure of the maintained path silently falls back
# to on-demand derivation, so turning it on is strictly a speedup + a
# persisted-beliefs upgrade, never a regression.
MEMORYLAYER_REPRESENTATION_CONSOLIDATION_ENABLED = 'MEMORYLAYER_REPRESENTATION_CONSOLIDATION_ENABLED'
DEFAULT_MEMORYLAYER_REPRESENTATION_CONSOLIDATION_ENABLED = False

# Per-(workspace,observer,subject) debounce lease TTL (seconds): an upper bound
# on a single consolidation run so a crashed holder self-heals.
MEMORYLAYER_REPRESENTATION_CONSOLIDATION_LEASE_TTL = 'MEMORYLAYER_REPRESENTATION_CONSOLIDATION_LEASE_TTL'
DEFAULT_MEMORYLAYER_REPRESENTATION_CONSOLIDATION_LEASE_TTL = 600

# TTL (seconds) for a persisted maintained representation in the cache. A
# generous default — freshness is governed by the workspace change watermark
# (a watermark mismatch forces a re-derive regardless of TTL), so the TTL only
# bounds unbounded growth of stale entries.
MEMORYLAYER_REPRESENTATION_CONSOLIDATION_RECORD_TTL = 'MEMORYLAYER_REPRESENTATION_CONSOLIDATION_RECORD_TTL'
DEFAULT_MEMORYLAYER_REPRESENTATION_CONSOLIDATION_RECORD_TTL = 604800  # 7 days

# ============================================
# Secure-auth startup guard (fail-closed)
# ============================================
# MemoryLayer Enterprise is the platform "data authority" and must NOT fall back
# to the OSS allow-all "default" auth/authz services in production. The enterprise
# preconfigure hook now defaults both auth and authz to "aether"; this opt-in flag
# additionally hardens startup: when set truthy, the server REFUSES to boot if the
# resolved authentication or authorization service is still "default" (e.g. an env
# override re-opened it). Off by default so local/dev runs with explicit
# MEMORYLAYER_AUTHENTICATION_SERVICE=default still work, but available so prod
# deployments can guarantee fail-closed auth by exporting MEMORYLAYER_REQUIRE_SECURE_AUTH=1.
MEMORYLAYER_REQUIRE_SECURE_AUTH = 'MEMORYLAYER_REQUIRE_SECURE_AUTH'
DEFAULT_MEMORYLAYER_REQUIRE_SECURE_AUTH = False

# ============================================
# Chat-thread auto-titling (dark by default)
# ============================================
# LLM-generated display titles for chat threads. When enabled, the enterprise
# chat service schedules a `chat_thread_title` background task as a thread's
# message_count crosses configured checkpoints; the task synthesizes a short
# title from the opening messages, persists it, and broadcasts a change event to
# the thread owner. Ships DARK: no behavior changes until the flag is truthy.
MEMORYLAYER_CHAT_TITLE_ENABLED = 'MEMORYLAYER_CHAT_TITLE_ENABLED'
DEFAULT_MEMORYLAYER_CHAT_TITLE_ENABLED = False

# Message-count checkpoints (comma-separated) at which a title (re)generation is
# scheduled. The first message is not guaranteed to capture the thread's purpose,
# so early ramped checkpoints let a weak initial title be corrected.
MEMORYLAYER_CHAT_TITLE_CHECKPOINTS = 'MEMORYLAYER_CHAT_TITLE_CHECKPOINTS'
DEFAULT_MEMORYLAYER_CHAT_TITLE_CHECKPOINTS = '1,5,10'

# After the explicit checkpoints, keep re-reviewing every N messages. 0 disables
# the recurring interval (only the explicit checkpoints fire).
MEMORYLAYER_CHAT_TITLE_INTERVAL = 'MEMORYLAYER_CHAT_TITLE_INTERVAL'
DEFAULT_MEMORYLAYER_CHAT_TITLE_INTERVAL = 10

# How many opening messages to feed the LLM as titling context.
MEMORYLAYER_CHAT_TITLE_MAX_MESSAGES = 'MEMORYLAYER_CHAT_TITLE_MAX_MESSAGES'
DEFAULT_MEMORYLAYER_CHAT_TITLE_MAX_MESSAGES = 10

# Max tokens for the generated title (titles are short by design).
MEMORYLAYER_CHAT_TITLE_MAX_TOKENS = 'MEMORYLAYER_CHAT_TITLE_MAX_TOKENS'
DEFAULT_MEMORYLAYER_CHAT_TITLE_MAX_TOKENS = 1024

# Instruction prompt for the titling LLM. The rendered conversation is appended
# after this instruction (a "{convo}" placeholder is also honored if present, so
# operator overrides can position the conversation explicitly). Kept short so
# operators can override without needing the placeholder.
MEMORYLAYER_CHAT_TITLE_PROMPT = 'MEMORYLAYER_CHAT_TITLE_PROMPT'
DEFAULT_MEMORYLAYER_CHAT_TITLE_PROMPT = (
    "Write a concise title of at most 6 words that captures the main topic of "
    "the conversation below. Respond with only the title text: no surrounding "
    "quotes, no markdown, no trailing punctuation, no preamble."
)

# ============================================
# Cold-tier archival job (dark by default)
# ============================================
# Master switch for the recurring archival sweep (tasks/tiering_archival.py).
# Ships OFF: tiering is opt-in and archival is destructive-ish (it drops the hot
# embedding and moves content to the LEANN cold tier), so operators enable it
# deliberately after validating impact via the dry-run admin endpoint
# (POST /v1/admin/tiering/run with dry_run=true). Per-workspace
# settings["tiering"].cold_tier_enabled additionally gates which workspaces the
# sweep touches.
MEMORYLAYER_TIERING_ARCHIVAL_ENABLED = 'MEMORYLAYER_TIERING_ARCHIVAL_ENABLED'
DEFAULT_MEMORYLAYER_TIERING_ARCHIVAL_ENABLED = False

# How often the archival sweep runs (seconds) when enabled.
MEMORYLAYER_TIERING_ARCHIVAL_INTERVAL_SEC = 'MEMORYLAYER_TIERING_ARCHIVAL_INTERVAL_SEC'
DEFAULT_MEMORYLAYER_TIERING_ARCHIVAL_INTERVAL_SEC = 86400  # 24 hours

# Max memories archived per workspace per sweep pass.
MEMORYLAYER_TIERING_ARCHIVAL_BATCH_SIZE = 'MEMORYLAYER_TIERING_ARCHIVAL_BATCH_SIZE'
DEFAULT_MEMORYLAYER_TIERING_ARCHIVAL_BATCH_SIZE = 100

# ============================================
# Knowledgebase refresh reconciler (enabled by default)
# ============================================
# The KB is normally refreshed event-driven: ingest_complete / decompose_complete
# feed a kb-coalesce join that fires a kb_update task. Those events are
# best-effort (see services/events.emit_event) — a transient Aether failure
# during decomposition can drop the signal, leaving a KB permanently stale
# relative to its graph. This recurring reconciler is the KB analogue of
# doc_verify / blob_gc: it walks workspaces that already have a KB and, using the
# workspace change-watermark, regenerates any whose KB fell behind. Idle /
# up-to-date KBs are a cheap no-op (KnowledgebaseService.generate skips when the
# watermark is unchanged). Defaults ON, matching the sibling reconcilers
# (doc_verify / blob_gc): a stale KB is a correctness gap the reconciler should
# heal without an operator opt-in. Set to false to disable.
MEMORYLAYER_KB_REFRESH_ENABLED = 'MEMORYLAYER_KB_REFRESH_ENABLED'
DEFAULT_MEMORYLAYER_KB_REFRESH_ENABLED = True

# How often the reconciler runs (seconds) when enabled.
MEMORYLAYER_KB_REFRESH_INTERVAL_SEC = 'MEMORYLAYER_KB_REFRESH_INTERVAL_SEC'
DEFAULT_MEMORYLAYER_KB_REFRESH_INTERVAL_SEC = 21600  # 6 hours

# Optional cap on workspaces processed per sweep (0 = no cap).
MEMORYLAYER_KB_REFRESH_MAX_WORKSPACES = 'MEMORYLAYER_KB_REFRESH_MAX_WORKSPACES'
DEFAULT_MEMORYLAYER_KB_REFRESH_MAX_WORKSPACES = 0

# ============================================
# Enterprise KB: canonical-entity articles (registry-derived "entity graph")
# ============================================
# The enterprise KnowledgebaseService (MEMORYLAYER_KNOWLEDGEBASE_PROVIDER=enterprise)
# adds one article per canonical Entity Registry entity on top of the OSS
# communities + god-nodes + index. Additive + fail-safe: with the registry
# unavailable (or this flag off) the KB is exactly the OSS output. Default ON.
MEMORYLAYER_KB_ENTITY_ARTICLES_ENABLED = 'MEMORYLAYER_KB_ENTITY_ARTICLES_ENABLED'
DEFAULT_MEMORYLAYER_KB_ENTITY_ARTICLES_ENABLED = True

# Upper bound on how many canonical entities become articles per generation
# (deterministic registry order). Keeps a huge registry from producing thousands
# of articles in one run.
MEMORYLAYER_KB_MAX_ENTITY_ARTICLES = 'MEMORYLAYER_KB_MAX_ENTITY_ARTICLES'
DEFAULT_MEMORYLAYER_KB_MAX_ENTITY_ARTICLES = 100

# Minimum member memories an entity needs to earn an article. 1-mention entities
# are noise (same principle as MEMORYLAYER_KB_MIN_COMMUNITY_SIZE for communities).
MEMORYLAYER_KB_MIN_ENTITY_MEMBERS = 'MEMORYLAYER_KB_MIN_ENTITY_MEMBERS'
DEFAULT_MEMORYLAYER_KB_MIN_ENTITY_MEMBERS = 2

# Whether to ALSO emit the OSS central-memory "god node" entity articles. Those
# are proxies for the most-connected memories; when the canonical entity-registry
# layer is active it supersedes them, so enterprise suppresses them by default to
# keep the KB's entity surface a real typed entity graph (not central-memory
# stand-ins). Set true to keep both. Only takes effect when entity articles are
# enabled AND the registry is available; otherwise god nodes are kept as fallback
# so the KB is never left with zero entity articles.
MEMORYLAYER_KB_GODNODE_ARTICLES_ENABLED = 'MEMORYLAYER_KB_GODNODE_ARTICLES_ENABLED'
DEFAULT_MEMORYLAYER_KB_GODNODE_ARTICLES_ENABLED = False
