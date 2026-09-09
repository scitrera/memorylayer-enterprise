# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Visual tokenizer API endpoint."""

import base64
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from memorylayer_embed_server.api import EXT_MULTI_API_ROUTERS
from memorylayer_embed_server.lifecycle.fastapi import get_logger, get_variables_dep
from scitrera_app_framework import Plugin, Variables

from ...models.visual_tokenizer import (
    JobStatusResponse,
    VisualTokenizeJobResponse,
    VisualTokenizeRequest,
    VisualTokenizeResponse,
    VisualTokenizeStats,
    VisualTokenMetadata,
)
from ...services.visual_tokenizer.base import VisualFeatureMetadata
from ...services.visual_tokenizer.service import VisualTokenizerJobLimitError

router = APIRouter(prefix="/v1", tags=['visual-tokenizer'])


def _get_service(v: Variables):
    """Get the visual tokenizer service or raise 503."""
    service = v.get('visual_tokenizer_service', default=None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Visual tokenizer service not configured",
        )
    return service


def _feature_metadata_from_result(ext_result) -> VisualFeatureMetadata | None:
    """Return tensor-free feature metadata from live or restored results."""
    feats = ext_result.features
    if feats is not None:
        return VisualFeatureMetadata(
            image_grid_thw=feats.image_grid_thw,
            num_visual_tokens=feats.num_visual_tokens,
            hidden_dim=feats.hidden_dim,
            dtype=feats.dtype,
            original_image_size=feats.original_image_size,
            embed_kind=feats.embed_kind,
        )
    return ext_result.feature_metadata


def _result_to_metadata(ext_result, *, return_tensors: bool, model_slug: str | None) -> VisualTokenMetadata:
    """Convert an ExtractionResult into the API metadata model.

    Encodes the vision-tower image embeds (zstd-compressed torch.save payload)
    and the uncompressed grid tensor to base64 when ``return_tensors`` is set,
    the extraction succeeded, and the restored result includes tensor artifacts.
    """
    meta = VisualTokenMetadata(
        image_index=ext_result.image_index,
        success=ext_result.success,
        from_cache=ext_result.from_cache,
        cache_path=ext_result.cache_path,
        latency_ms=ext_result.latency_ms,
        error=ext_result.error,
        model_slug=model_slug,
    )
    feature_metadata = _feature_metadata_from_result(ext_result)
    if ext_result.success and feature_metadata is not None:
        meta.num_visual_tokens = feature_metadata.num_visual_tokens
        meta.num_image_tokens = feature_metadata.num_visual_tokens
        meta.hidden_dim = feature_metadata.hidden_dim
        meta.image_grid_thw = feature_metadata.image_grid_thw
        meta.embed_kind = feature_metadata.embed_kind
        if return_tensors and ext_result.features is not None:
            import torch

            from ...services.visual_tokenizer.codec import tensor_to_b64

            meta.image_embeds_b64 = tensor_to_b64(ext_result.features.image_embeds, compress=True)
            meta.image_grid_thw_b64 = tensor_to_b64(
                torch.tensor(feature_metadata.image_grid_thw, dtype=torch.int64), compress=False,
            )
    return meta


def _is_successful_result(ext_result) -> bool:
    return ext_result.success and _feature_metadata_from_result(ext_result) is not None


def _visual_token_count(ext_result) -> int:
    feature_metadata = _feature_metadata_from_result(ext_result)
    if feature_metadata is None:
        return 0
    return feature_metadata.num_visual_tokens


@router.post("/visual-tokenize", response_model=VisualTokenizeResponse | VisualTokenizeJobResponse)
async def visual_tokenize(
    request: VisualTokenizeRequest,
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> VisualTokenizeResponse | VisualTokenizeJobResponse:
    """Extract visual features from page images.

    Processes images through the Qwen3.5 vision encoder and caches
    the resulting visual tokens to disk in safetensors format.

    When async_mode=True, returns a job_id immediately for polling.
    """
    service = _get_service(v)

    # Decode base64 images
    image_bytes_list = []
    for idx, image_b64 in enumerate(request.images):
        try:
            image_bytes_list.append(base64.b64decode(image_b64))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 in image at index {idx}: {e}",
            )

    logger.info("Visual tokenizing %d image(s)", len(image_bytes_list))

    # Validate optional metadata alignment.
    metadatas = request.metadata
    if metadatas is not None and len(metadatas) != len(image_bytes_list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"metadata length ({len(metadatas)}) must match images length "
                   f"({len(image_bytes_list)})",
        )

    # Async mode: start job and return immediately
    if request.async_mode:
        try:
            job_id = service.start_async_job(
                images=image_bytes_list,
                batch_size=request.batch_size,
                force_recompute=request.force_recompute,
                metadatas=metadatas,
                return_tensors=request.return_tensors,
            )
        except VisualTokenizerJobLimitError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
            )
        return VisualTokenizeJobResponse(
            job_id=job_id,
            status="pending",
            total_images=len(image_bytes_list),
        )

    # Synchronous mode: process and return results
    extraction_results = await service.tokenize_images(
        images=image_bytes_list,
        batch_size=request.batch_size,
        force_recompute=request.force_recompute,
        metadatas=metadatas,
    )

    model_info = service.get_model_info()
    model_slug = model_info.get("model_slug")

    # Convert to response models
    results = []
    successful = 0
    failed = 0
    cache_hits = 0
    cache_misses = 0
    total_latency_ms = 0.0
    total_visual_tokens = 0

    for ext_result in extraction_results:
        metadata = _result_to_metadata(
            ext_result, return_tensors=request.return_tensors, model_slug=model_slug,
        )

        if _is_successful_result(ext_result):
            successful += 1
            total_visual_tokens += _visual_token_count(ext_result)
        else:
            failed += 1

        if ext_result.from_cache:
            cache_hits += 1
        elif ext_result.success:
            cache_misses += 1

        total_latency_ms += ext_result.latency_ms
        results.append(metadata)

    stats = VisualTokenizeStats(
        total_images=len(extraction_results),
        successful=successful,
        failed=failed,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        total_latency_ms=total_latency_ms,
        total_visual_tokens=total_visual_tokens,
    )

    return VisualTokenizeResponse(
        results=results,
        stats=stats,
        model=model_info.get("model_name", "unknown"),
    )


@router.get("/visual-tokenize/status/{job_id}", response_model=JobStatusResponse)
async def visual_tokenize_job_status(
    job_id: str,
    v: Variables = Depends(get_variables_dep),
) -> JobStatusResponse:
    """Poll status of an async visual tokenization job."""
    service = _get_service(v)

    job = service.get_job_status(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    response = JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0),
        total=job.get("total", 0),
        error=job.get("error"),
        tensors_persisted=bool(job.get("tensors_persisted", False)),
        artifact_retention=job.get("artifact_retention"),
    )

    # If completed, build the full response
    if job["status"] == "completed":
        extraction_results = service.get_job_result(job_id)
        if extraction_results is None:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail=f"Job {job_id} result is no longer available",
            )
        model_info = service.get_model_info()
        model_slug = model_info.get("model_slug")
        results = []
        successful = 0
        failed = 0
        cache_hits = 0
        total_visual_tokens = 0
        total_latency_ms = 0.0

        for ext_result in extraction_results:
            metadata = _result_to_metadata(
                ext_result,
                return_tensors=bool(job.get("return_tensors", False)),
                model_slug=model_slug,
            )

            if _is_successful_result(ext_result):
                successful += 1
                total_visual_tokens += _visual_token_count(ext_result)
            else:
                failed += 1

            if ext_result.from_cache:
                cache_hits += 1
            total_latency_ms += ext_result.latency_ms
            results.append(metadata)

        response.result = VisualTokenizeResponse(
            results=results,
            stats=VisualTokenizeStats(
                total_images=len(extraction_results),
                successful=successful,
                failed=failed,
                cache_hits=cache_hits,
                cache_misses=successful - cache_hits,
                total_latency_ms=total_latency_ms,
                total_visual_tokens=total_visual_tokens,
            ),
            model=model_info.get("model_name", "unknown"),
        )

    return response


class VisualTokenizerAPIPlugin(Plugin):
    """Plugin to register visual tokenizer API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False  # multi-extension pattern

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
