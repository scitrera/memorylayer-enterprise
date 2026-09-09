"""Visual tokenizer orchestration service.

Coordinates the provider (GPU feature extraction) with the disk cache,
supporting both synchronous and asynchronous (job-based) workflows.
"""

import asyncio
import time
import uuid
from logging import Logger

from scitrera_app_framework import Variables, get_logger

from .base import ExtractionResult, VisualTokenizerProvider
from .cache import VisualTokenCache
from .job_store import VisualTokenizerJobResultStore

_TERMINAL_JOB_STATUSES = {"completed", "failed"}


class VisualTokenizerJobLimitError(RuntimeError):
    """Raised when the visual-tokenizer async job queue is full."""


class VisualTokenizerService:
    """Orchestrates visual feature extraction with caching.

    Handles cache-hit/miss routing, batch extraction, and async jobs.
    """

    def __init__(
        self,
        provider: VisualTokenizerProvider,
        cache: VisualTokenCache,
        v: Variables = None,
        max_jobs: int = 64,
        job_ttl_seconds: float = 3600.0,
        job_result_store: VisualTokenizerJobResultStore | None = None,
    ):
        self.provider = provider
        self.cache = cache
        self.logger: Logger = get_logger(v, name="VisualTokenizerService")
        self.max_jobs = max(1, int(max_jobs))
        self.job_ttl_seconds = max(1.0, float(job_ttl_seconds))
        self._jobs: dict[str, dict] = {}
        self._job_tasks: dict[str, asyncio.Task] = {}
        self._job_result_store = job_result_store or VisualTokenizerJobResultStore(
            self.cache.cache_dir / "_jobs",
        )
        self.logger.info("VisualTokenizerService initialized")

    async def preload(self):
        """Preload the underlying provider model."""
        await self.provider.preload()

    def _cache_key(self, image_data: bytes, metadata: dict | None) -> str:
        """Composite cache key = image hash + optional provider suffix.

        For the prompt-embeds path the provider folds the page-metadata
        fingerprint into the suffix so the same image with different metadata
        does not collide.
        """
        img_hash = self.cache.image_hash(image_data)
        suffix = self.provider.cache_key_suffix(metadata)
        return f"{img_hash}.{suffix}" if suffix else img_hash

    async def tokenize_images(
        self,
        images: list[bytes],
        batch_size: int = 4,
        force_recompute: bool = False,
        metadatas: list[dict] | None = None,
    ) -> list[ExtractionResult]:
        """Extract features for a list of pages with caching.

        Phase 1: Cache lookup per page (keyed by image hash + metadata).
        Phase 2: Batch extract cache misses via provider.
        Phase 3: Save new results to cache.

        Args:
            images: List of raw image bytes.
            batch_size: GPU batch size for extraction.
            force_recompute: If True, ignore cache and re-extract all.
            metadatas: Optional per-image metadata dicts, aligned by index.

        Returns:
            List of ExtractionResult sorted by original index.
        """
        start_time = time.monotonic()
        results: dict[int, ExtractionResult] = {}
        to_extract: list[tuple[int, bytes, str, dict | None]] = []  # (index, data, key, meta)

        # Phase 1: Cache lookup
        for idx, image_data in enumerate(images):
            meta = metadatas[idx] if metadatas and idx < len(metadatas) else None
            key = self._cache_key(image_data, meta)

            if not force_recompute and self.cache.has(key):
                cached_features = self.cache.get(key)
                if cached_features is not None:
                    cache_path = str(self.cache._tensor_path(key))
                    results[idx] = ExtractionResult(
                        image_index=idx,
                        success=True,
                        features=cached_features,
                        from_cache=True,
                        cache_path=cache_path,
                        latency_ms=0.0,
                    )
                    continue

            to_extract.append((idx, image_data, key, meta))

        cache_hits = len(results)
        self.logger.info(
            "Cache lookup: %d hits, %d misses out of %d images",
            cache_hits, len(to_extract), len(images),
        )

        # Phase 2: Batch extract cache misses
        if to_extract:
            miss_images = [data for _, data, _, _ in to_extract]
            miss_metas = [meta for _, _, _, meta in to_extract]
            extraction_results = await self.provider.extract_features_batch(
                miss_images, batch_size=batch_size, metadatas=miss_metas,
            )

            # Phase 3: Save to cache and assemble results
            for (orig_idx, _image_data, key, _meta), ext_result in zip(
                to_extract, extraction_results,
            ):
                ext_result.image_index = orig_idx

                if ext_result.success and ext_result.features is not None:
                    try:
                        cache_path = self.cache.put(key, ext_result.features)
                        ext_result.cache_path = cache_path
                    except Exception as e:
                        self.logger.warning(
                            "Failed to cache features for image %d: %s", orig_idx, e,
                        )

                results[orig_idx] = ext_result

        total_ms = (time.monotonic() - start_time) * 1000
        self.logger.info(
            "Tokenization complete: %d images in %.1fms (%d cached, %d extracted)",
            len(images), total_ms, cache_hits, len(to_extract),
        )

        # Return sorted by original index
        return [results[i] for i in range(len(images))]

    def _prune_jobs(self, now: float | None = None) -> None:
        """Remove expired terminal jobs and keep the in-memory table bounded."""
        current = time.time() if now is None else now
        expired = [
            job_id for job_id, job in self._jobs.items()
            if job.get("status") in _TERMINAL_JOB_STATUSES
            and current - float(job.get("updated_at", job.get("created_at", current))) > self.job_ttl_seconds
        ]
        for job_id in expired:
            self._drop_job(job_id)

        if len(self._jobs) < self.max_jobs:
            return

        terminal = sorted(
            (
                (float(job.get("updated_at", job.get("created_at", current))), job_id)
                for job_id, job in self._jobs.items()
                if job.get("status") in _TERMINAL_JOB_STATUSES
            ),
        )
        for _updated_at, job_id in terminal:
            if len(self._jobs) < self.max_jobs:
                break
            self._drop_job(job_id)

    def _drop_job(self, job_id: str) -> None:
        """Remove job metadata, task tracking, and persisted terminal results."""
        self._jobs.pop(job_id, None)
        self._job_tasks.pop(job_id, None)
        self._job_result_store.delete(job_id)

    def start_async_job(
        self,
        images: list[bytes],
        batch_size: int = 4,
        force_recompute: bool = False,
        metadatas: list[dict] | None = None,
        return_tensors: bool = True,
    ) -> str:
        """Start an async tokenization job.

        Returns a job_id that can be polled with get_job_status().
        """
        now = time.time()
        self._prune_jobs(now)
        if len(self._jobs) >= self.max_jobs:
            raise VisualTokenizerJobLimitError(
                f"visual tokenizer job queue full ({len(self._jobs)}/{self.max_jobs})"
            )

        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "total": len(images),
            "result": None,
            "result_ref": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "return_tensors": return_tensors,
            "tensors_persisted": False,
            "artifact_retention": "metadata_only_until_job_ttl",
        }
        self._job_result_store.put_status(job_id, self._jobs[job_id])

        async def _run_job():
            try:
                self._jobs[job_id]["status"] = "running"
                self._jobs[job_id]["updated_at"] = time.time()
                self._job_result_store.put_status(job_id, self._jobs[job_id])
                result = await self.tokenize_images(
                    images, batch_size=batch_size, force_recompute=force_recompute,
                    metadatas=metadatas,
                )
                updated_at = time.time()
                result_ref = self._job_result_store.put(
                    job_id,
                    result,
                    return_tensors=return_tensors,
                    created_at=now,
                    updated_at=updated_at,
                )
                tensors_persisted = return_tensors and any(
                    item.success and item.features is not None for item in result
                )
                self._jobs[job_id]["status"] = "completed"
                self._jobs[job_id]["result"] = None
                self._jobs[job_id]["result_ref"] = result_ref
                self._jobs[job_id]["progress"] = len(images)
                self._jobs[job_id]["updated_at"] = updated_at
                self._jobs[job_id]["tensors_persisted"] = tensors_persisted
                self._jobs[job_id]["artifact_retention"] = (
                    "metadata_and_tensors_until_job_ttl"
                    if tensors_persisted
                    else "metadata_only_until_job_ttl"
                )
            except Exception as e:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = str(e)
                self._jobs[job_id]["updated_at"] = time.time()
                self._jobs[job_id]["tensors_persisted"] = False
                self._jobs[job_id]["artifact_retention"] = "metadata_only_until_job_ttl"
                self._job_result_store.put_status(job_id, self._jobs[job_id])
                self.logger.warning("Async job %s failed: %s", job_id, e)

        task = asyncio.create_task(_run_job())
        self._job_tasks[job_id] = task
        task.add_done_callback(lambda _task: self._job_tasks.pop(job_id, None))
        self.logger.info("Started async job %s for %d images", job_id, len(images))
        return job_id

    def get_job_status(self, job_id: str) -> dict | None:
        """Get status of an async job.

        Returns None if job_id not found.
        """
        self._prune_jobs()
        job = self._jobs.get(job_id)
        if job is not None:
            return job

        restored = self._job_result_store.get_status(job_id)
        if restored is None:
            return None
        self._jobs[job_id] = restored
        self._prune_jobs()
        return self._jobs.get(job_id)

    def get_job_result(self, job_id: str) -> list[ExtractionResult] | None:
        """Load persisted results for a completed async job."""
        job = self.get_job_status(job_id)
        if job is None or job.get("status") != "completed":
            return None
        return self._job_result_store.get(job_id)

    def cancel_all_jobs(self) -> None:
        """Cancel all in-flight async jobs, used by lifecycle shutdown hooks."""
        for task in list(self._job_tasks.values()):
            if not task.done():
                task.cancel()
        self._job_tasks.clear()

    def get_model_info(self) -> dict:
        """Get model info from the underlying provider."""
        return self.provider.get_model_info()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return self.cache.get_cache_stats()
