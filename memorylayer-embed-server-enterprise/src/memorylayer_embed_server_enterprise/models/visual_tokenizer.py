# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Pydantic models for visual tokenizer API."""

from typing import Annotated

from pydantic import BaseModel, Field

_MAX_VISUAL_IMAGES = 32
_MAX_BASE64_IMAGE_CHARS = 16_000_000

Base64Image = Annotated[str, Field(max_length=_MAX_BASE64_IMAGE_CHARS)]


class VisualTokenizeRequest(BaseModel):
    """Request to extract vision-tower image embeds from page images."""
    images: list[Base64Image] = Field(
        ...,
        max_length=_MAX_VISUAL_IMAGES,
        description="List of base64-encoded images",
    )
    metadata: list[dict] | None = Field(
        None,
        description="Optional stable per-page metadata (filename, page_no, source), "
                    "aligned by index with images. Informational only.",
    )
    batch_size: int = Field(4, ge=1, le=32, description="GPU batch size for extraction")
    cache_key: str | None = Field(None, description="Optional cache key prefix")
    force_recompute: bool = Field(False, description="Ignore cache and re-extract")
    return_tensors: bool = Field(
        True,
        description="Include the base64-encoded image-embeds + grid tensors in the "
                    "response so the caller can persist them (torch.save payloads). "
                    "For async jobs, false stores restartable metadata only in "
                    "the job result directory; true also retains safetensors until "
                    "job TTL/delete cleanup removes the job.",
    )
    async_mode: bool = Field(False, description="Return job_id immediately for async processing")


class VisualTokenMetadata(BaseModel):
    """Metadata for a single page's vision-tower image embeds."""
    image_index: int
    success: bool = True
    num_visual_tokens: int = 0
    num_image_tokens: int = 0
    hidden_dim: int = 0
    image_grid_thw: list[int] = Field(default_factory=list)
    embed_kind: str | None = None
    model_slug: str | None = None
    image_embeds_b64: str | None = Field(
        None,
        description="Base64( zstd_level1( torch.save(vision_tensor) ) ) of the "
                    "[num_image_tokens, hidden_dim] projected vision-tower embeds "
                    "(present when return_tensors=true). Feed to a vLLM chat profile "
                    "as an image_embeds content part.",
    )
    image_grid_thw_b64: str | None = Field(
        None,
        description="Base64( torch.save( torch.tensor([t,h,w], int64) ) ), uncompressed "
                    "(present when return_tensors=true). Sent alongside image_embeds so "
                    "vLLM can reconstruct the spatial grid for M-RoPE.",
    )
    cache_path: str | None = None
    from_cache: bool = False
    latency_ms: float = 0.0
    error: str | None = None


class VisualTokenizeStats(BaseModel):
    """Aggregate statistics for a visual tokenization request."""
    total_images: int
    successful: int
    failed: int
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0
    total_visual_tokens: int = 0


class VisualTokenizeResponse(BaseModel):
    """Response from visual tokenization endpoint."""
    results: list[VisualTokenMetadata]
    stats: VisualTokenizeStats
    model: str


class VisualTokenizeJobResponse(BaseModel):
    """Response when async_mode=True, returns a job_id for polling."""
    job_id: str
    status: str = "pending"
    total_images: int


class JobStatusResponse(BaseModel):
    """Response from job status polling endpoint."""
    job_id: str
    status: str
    progress: int = 0
    total: int = 0
    result: VisualTokenizeResponse | None = None
    error: str | None = None
    tensors_persisted: bool = Field(
        False,
        description="Whether this async job result retained safetensors artifacts.",
    )
    artifact_retention: str | None = Field(
        None,
        description="Async job-result retention policy for this job directory.",
    )
