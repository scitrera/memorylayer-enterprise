"""Unit tests for the visual tokenizer service."""

import asyncio
import base64
import io
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from memorylayer_embed_server_enterprise.models.visual_tokenizer import (
    JobStatusResponse,
    VisualTokenizeJobResponse,
    VisualTokenizeRequest,
    VisualTokenizeResponse,
    VisualTokenizeStats,
    VisualTokenMetadata,
)
from memorylayer_embed_server_enterprise.services.visual_tokenizer.base import (
    ExtractionResult,
    VisualFeatureMetadata,
    VisualFeatures,
    VisualTokenizerProvider,
)
from memorylayer_embed_server_enterprise.services.visual_tokenizer.cache import VisualTokenCache
from memorylayer_embed_server_enterprise.services.visual_tokenizer.job_store import VisualTokenizerJobResultStore
from memorylayer_embed_server_enterprise.services.visual_tokenizer.service import VisualTokenizerService

# ============================================
# Fixtures
# ============================================

def _make_test_image(width: int = 64, height: int = 64) -> bytes:
    """Create a minimal test image as bytes."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_test_image_b64(width: int = 64, height: int = 64) -> str:
    """Create a minimal test image as base64 string."""
    return base64.b64encode(_make_test_image(width, height)).decode()


def _make_visual_features(
    num_tokens: int = 256,
    hidden_dim: int = 2048,
) -> VisualFeatures:
    """Create test VisualFeatures with random tensors."""
    return VisualFeatures(
        image_embeds=torch.randn(num_tokens, hidden_dim),
        image_grid_thw=[1, 16, 16],
        num_visual_tokens=num_tokens,
        hidden_dim=hidden_dim,
        dtype="torch.float32",
        original_image_size=(64, 64),
    )


class MockProvider(VisualTokenizerProvider):
    """Mock provider for testing."""

    PROVIDER_NAME = "mock"

    def __init__(self, features: VisualFeatures = None):
        self._features = features or _make_visual_features()
        self.preload_called = False
        self.extract_count = 0
        self.logger = MagicMock()

    async def preload(self):
        self.preload_called = True

    async def extract_features(self, image_data: bytes, metadata: dict = None) -> ExtractionResult:
        self.extract_count += 1
        self.last_metadata = metadata
        return ExtractionResult(
            image_index=0,
            success=True,
            features=self._features,
            latency_ms=10.0,
        )

    def get_model_info(self) -> dict:
        return {
            "provider": "mock",
            "model_name": "mock-model",
            "loaded": True,
        }


# ============================================
# Pydantic Model Tests
# ============================================

class TestPydanticModels:
    """Test Pydantic request/response model serialization."""

    def test_visual_tokenize_request_defaults(self):
        req = VisualTokenizeRequest(images=["abc123"])
        assert req.batch_size == 4
        assert req.cache_key is None
        assert req.force_recompute is False
        assert req.async_mode is False

    def test_visual_tokenize_request_custom(self):
        req = VisualTokenizeRequest(
            images=["img1", "img2"],
            batch_size=8,
            force_recompute=True,
            async_mode=True,
        )
        assert len(req.images) == 2
        assert req.batch_size == 8
        assert req.force_recompute is True
        assert req.async_mode is True

    def test_visual_tokenize_request_rejects_oversized_batch(self):
        with pytest.raises(ValidationError):
            VisualTokenizeRequest(images=["abc123"] * 33)

    def test_visual_tokenize_request_rejects_oversized_image_payload(self):
        with pytest.raises(ValidationError):
            VisualTokenizeRequest(images=["a" * 16_000_001])

    def test_visual_tokenize_request_batch_size_bounds(self):
        with pytest.raises(Exception):
            VisualTokenizeRequest(images=["img1"], batch_size=0)
        with pytest.raises(Exception):
            VisualTokenizeRequest(images=["img1"], batch_size=33)

    def test_visual_token_metadata(self):
        meta = VisualTokenMetadata(
            image_index=0,
            success=True,
            num_visual_tokens=256,
            hidden_dim=2048,
            image_grid_thw=[1, 16, 16],
            from_cache=True,
            latency_ms=5.2,
        )
        data = meta.model_dump()
        assert data["num_visual_tokens"] == 256
        assert data["hidden_dim"] == 2048
        assert data["from_cache"] is True

    def test_visual_tokenize_response(self):
        resp = VisualTokenizeResponse(
            results=[
                VisualTokenMetadata(image_index=0, success=True, num_visual_tokens=100, hidden_dim=2048),
            ],
            stats=VisualTokenizeStats(
                total_images=1, successful=1, failed=0,
                cache_hits=0, cache_misses=1,
                total_latency_ms=50.0, total_visual_tokens=100,
            ),
            model="Qwen/Qwen3.5-35B-A3B-FP8",
        )
        data = resp.model_dump()
        assert data["model"] == "Qwen/Qwen3.5-35B-A3B-FP8"
        assert len(data["results"]) == 1
        assert data["stats"]["successful"] == 1

    def test_job_response(self):
        resp = VisualTokenizeJobResponse(
            job_id="abc-123",
            total_images=5,
        )
        assert resp.status == "pending"

    def test_job_status_response(self):
        resp = JobStatusResponse(
            job_id="abc-123",
            status="completed",
            progress=5,
            total=5,
        )
        assert resp.result is None
        assert resp.error is None


# ============================================
# Cache Tests
# ============================================

class TestVisualTokenCache:
    """Test disk cache for visual features."""

    def _make_cache(self, tmpdir: str, **kwargs) -> VisualTokenCache:
        return VisualTokenCache(
            cache_dir=tmpdir,
            model_slug="test-model",
            **kwargs,
        )

    def test_image_hash_deterministic(self):
        data = b"test image data"
        h1 = VisualTokenCache.image_hash(data)
        h2 = VisualTokenCache.image_hash(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_image_hash_different_for_different_data(self):
        h1 = VisualTokenCache.image_hash(b"image_a")
        h2 = VisualTokenCache.image_hash(b"image_b")
        assert h1 != h2

    def test_cache_miss(self, tmp_path):
        cache = self._make_cache(str(tmp_path))
        assert cache.has("nonexistent") is False
        assert cache.get("nonexistent") is None

    def test_cache_roundtrip(self, tmp_path):
        """Test put -> has -> get produces identical tensors."""
        cache = self._make_cache(str(tmp_path))
        features = _make_visual_features(num_tokens=128, hidden_dim=1024)
        img_hash = "abc123def456"

        # Put
        path = cache.put(img_hash, features)
        assert os.path.exists(path)
        assert path.endswith(".safetensors")

        # Has
        assert cache.has(img_hash) is True

        # Get
        loaded = cache.get(img_hash)
        assert loaded is not None
        assert loaded.num_visual_tokens == 128
        assert loaded.hidden_dim == 1024
        assert loaded.image_grid_thw == [1, 16, 16]
        assert loaded.dtype == "torch.float32"
        assert loaded.original_image_size == (64, 64)
        assert torch.allclose(loaded.image_embeds, features.image_embeds)

    def test_cache_metadata_sidecar(self, tmp_path):
        """Test that metadata JSON sidecar is written correctly."""
        cache = self._make_cache(str(tmp_path))
        features = _make_visual_features()
        img_hash = "metadata_test"
        cache.put(img_hash, features)

        metadata_path = cache._metadata_path(img_hash)
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["model_slug"] == "test-model"
        assert metadata["num_visual_tokens"] == 256
        assert metadata["hidden_dim"] == 2048
        assert "cached_at" in metadata

    def test_cache_corruption_recovery(self, tmp_path):
        """Test that corrupted cache entries are deleted on load."""
        cache = self._make_cache(str(tmp_path))
        img_hash = "corrupt_test"

        # Write garbage to tensor file
        tensor_path = cache._tensor_path(img_hash)
        tensor_path.write_bytes(b"not a safetensor")
        metadata_path = cache._metadata_path(img_hash)
        metadata_path.write_text('{"image_grid_thw": [1, 1, 1]}')

        # Should return None and clean up
        result = cache.get(img_hash)
        assert result is None
        assert not tensor_path.exists()

    def test_cache_clear(self, tmp_path):
        cache = self._make_cache(str(tmp_path))
        for i in range(3):
            cache.put(f"hash_{i}", _make_visual_features(num_tokens=16, hidden_dim=64))

        stats = cache.get_cache_stats()
        assert stats["total_entries"] == 3

        cache.clear()

        stats = cache.get_cache_stats()
        assert stats["total_entries"] == 0

    def test_cache_stats(self, tmp_path):
        cache = self._make_cache(str(tmp_path))
        cache.put("test_hash", _make_visual_features(num_tokens=16, hidden_dim=64))

        stats = cache.get_cache_stats()
        assert stats["total_entries"] == 1
        assert stats["model_slug"] == "test-model"
        # Small tensors may round to 0.000 GB, check the file exists instead
        assert cache._tensor_path("test_hash").stat().st_size > 0

    def test_cache_eviction_by_size(self, tmp_path):
        """Test LRU eviction when cache exceeds max size."""
        # Tiny max size to trigger eviction
        cache = self._make_cache(str(tmp_path), max_cache_size_gb=0.000001)

        # First entry
        cache.put("first", _make_visual_features(num_tokens=64, hidden_dim=256))
        assert cache.has("first")

        # Second entry should trigger eviction of first
        cache.put("second", _make_visual_features(num_tokens=64, hidden_dim=256))
        # At least one entry should exist
        stats = cache.get_cache_stats()
        assert stats["total_entries"] >= 1

    def test_cache_ttl_expiry(self, tmp_path):
        """Test that expired entries are not returned."""
        cache = self._make_cache(str(tmp_path), ttl_hours=0.0001)  # ~0.36 seconds
        features = _make_visual_features(num_tokens=16, hidden_dim=64)
        img_hash = "ttl_test"

        cache.put(img_hash, features)
        assert cache.has(img_hash) is True

        # Manually backdate the file mtime
        import time
        tensor_path = cache._tensor_path(img_hash)
        old_time = time.time() - 3600  # 1 hour ago
        os.utime(tensor_path, (old_time, old_time))

        assert cache.has(img_hash) is False


# ============================================
# Service Tests
# ============================================

class TestVisualTokenizerService:
    """Test the orchestration service."""

    def _make_service(self, tmp_path, provider=None) -> VisualTokenizerService:
        provider = provider or MockProvider()
        cache = VisualTokenCache(
            cache_dir=str(tmp_path),
            model_slug="test-model",
        )
        return VisualTokenizerService(provider=provider, cache=cache)

    @pytest.mark.asyncio
    async def test_tokenize_single_image(self, tmp_path):
        service = self._make_service(tmp_path)
        image_data = _make_test_image()

        results = await service.tokenize_images([image_data])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].features is not None
        assert results[0].from_cache is False

    @pytest.mark.asyncio
    async def test_tokenize_caches_result(self, tmp_path):
        service = self._make_service(tmp_path)
        image_data = _make_test_image()

        # First call - cache miss
        results1 = await service.tokenize_images([image_data])
        assert results1[0].from_cache is False
        assert results1[0].cache_path is not None

        # Second call - cache hit
        results2 = await service.tokenize_images([image_data])
        assert results2[0].from_cache is True
        assert results2[0].success is True

    @pytest.mark.asyncio
    async def test_tokenize_force_recompute(self, tmp_path):
        provider = MockProvider()
        service = self._make_service(tmp_path, provider=provider)
        image_data = _make_test_image()

        # First call
        await service.tokenize_images([image_data])
        assert provider.extract_count == 1

        # Second call with force_recompute - should re-extract
        await service.tokenize_images([image_data], force_recompute=True)
        assert provider.extract_count == 2

    @pytest.mark.asyncio
    async def test_tokenize_multiple_images(self, tmp_path):
        service = self._make_service(tmp_path)
        images = [_make_test_image(w, w) for w in (32, 48, 64)]

        results = await service.tokenize_images(images)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.image_index == i
            assert result.success is True

    @pytest.mark.asyncio
    async def test_tokenize_mixed_cache_hit_miss(self, tmp_path):
        service = self._make_service(tmp_path)
        img_a = _make_test_image(32, 32)
        img_b = _make_test_image(48, 48)

        # Cache img_a
        await service.tokenize_images([img_a])

        # Now request both - img_a should be cached, img_b should miss
        results = await service.tokenize_images([img_a, img_b])
        assert results[0].from_cache is True
        assert results[1].from_cache is False

    @pytest.mark.asyncio
    async def test_preload_delegates_to_provider(self, tmp_path):
        provider = MockProvider()
        service = self._make_service(tmp_path, provider=provider)

        await service.preload()
        assert provider.preload_called is True

    @pytest.mark.asyncio
    async def test_async_job_lifecycle(self, tmp_path):
        service = self._make_service(tmp_path)
        image_data = _make_test_image()

        # Start job
        job_id = service.start_async_job([image_data])
        assert job_id is not None

        # Poll until done
        for _ in range(50):
            status = service.get_job_status(job_id)
            if status["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.05)

        status = service.get_job_status(job_id)
        assert status["status"] == "completed"
        assert status["result"] is None
        assert status["result_ref"]
        assert status["return_tensors"] is True
        assert status["tensors_persisted"] is True
        assert status["artifact_retention"] == "metadata_and_tensors_until_job_ttl"
        assert any(
            path.endswith(".safetensors")
            for path in os.listdir(status["result_ref"])
        )

        stored_results = service.get_job_result(job_id)
        assert stored_results is not None
        assert len(stored_results) == 1
        assert stored_results[0].success is True
        assert stored_results[0].features is not None
        assert stored_results[0].feature_metadata is not None

    @pytest.mark.asyncio
    async def test_async_job_tracks_return_tensors_preference(self, tmp_path):
        service = self._make_service(tmp_path)
        image_data = _make_test_image()

        job_id = service.start_async_job([image_data], return_tensors=False)

        for _ in range(50):
            status = service.get_job_status(job_id)
            if status["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.05)

        status = service.get_job_status(job_id)
        assert status["status"] == "completed"
        assert status["return_tensors"] is False
        assert status["tensors_persisted"] is False
        assert status["artifact_retention"] == "metadata_only_until_job_ttl"
        assert not any(
            path.endswith(".safetensors")
            for path in os.listdir(status["result_ref"])
        )

        stored_results = service.get_job_result(job_id)
        assert stored_results is not None
        assert stored_results[0].features is None
        assert stored_results[0].feature_metadata is not None
        assert stored_results[0].feature_metadata.num_visual_tokens == 256

    @pytest.mark.asyncio
    async def test_completed_job_prune_deletes_persisted_result(self, tmp_path):
        service = self._make_service(tmp_path)
        service.job_ttl_seconds = 1.0
        image_data = _make_test_image()

        job_id = service.start_async_job([image_data])

        for _ in range(50):
            status = service.get_job_status(job_id)
            if status["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.05)

        status = service.get_job_status(job_id)
        assert status["status"] == "completed"
        result_ref = status["result_ref"]
        assert os.path.isdir(result_ref)

        status["updated_at"] = 1.0
        assert service.get_job_status(job_id) is None
        assert not os.path.exists(result_ref)

    def test_job_result_store_round_trips_failure_without_tensor(self, tmp_path):
        store = VisualTokenizerJobResultStore(tmp_path / "jobs")
        result = ExtractionResult(image_index=2, success=False, error="bad image")

        job_id = "00000000-0000-4000-8000-000000000001"
        store.put(
            job_id,
            [result],
            return_tensors=False,
            created_at=10.0,
            updated_at=11.0,
        )
        loaded = store.get(job_id)

        assert loaded is not None
        assert loaded[0].image_index == 2
        assert loaded[0].success is False
        assert loaded[0].features is None
        assert loaded[0].error == "bad image"

    def test_job_result_store_keeps_metadata_without_tensor_artifact(self, tmp_path):
        store = VisualTokenizerJobResultStore(tmp_path / "jobs")
        result = ExtractionResult(
            image_index=0,
            success=True,
            features=_make_visual_features(num_tokens=128, hidden_dim=1024),
        )

        job_id = "00000000-0000-4000-8000-000000000003"
        result_ref = store.put(
            job_id,
            [result],
            return_tensors=False,
            created_at=10.0,
            updated_at=11.0,
        )
        loaded = store.get(job_id)

        assert not any(path.endswith(".safetensors") for path in os.listdir(result_ref))
        assert loaded is not None
        assert loaded[0].success is True
        assert loaded[0].features is None
        assert loaded[0].feature_metadata is not None
        assert loaded[0].feature_metadata.num_visual_tokens == 128
        assert loaded[0].feature_metadata.hidden_dim == 1024

    @pytest.mark.asyncio
    async def test_completed_job_status_restores_from_persistent_store(self, tmp_path):
        service = self._make_service(tmp_path)
        image_data = _make_test_image()

        job_id = service.start_async_job([image_data], return_tensors=False)

        for _ in range(50):
            status = service.get_job_status(job_id)
            if status["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.05)

        restored_service = self._make_service(tmp_path)
        status = restored_service.get_job_status(job_id)
        assert status is not None
        assert status["status"] == "completed"
        assert status["return_tensors"] is False
        assert status["tensors_persisted"] is False
        assert status["artifact_retention"] == "metadata_only_until_job_ttl"
        assert status["result"] is None
        assert status["result_ref"]

        restored_results = restored_service.get_job_result(job_id)
        assert restored_results is not None
        assert restored_results[0].features is None
        assert restored_results[0].feature_metadata is not None
        assert restored_results[0].feature_metadata.num_visual_tokens == 256

    @pytest.mark.asyncio
    async def test_pending_job_status_restores_from_persistent_store(self, tmp_path):
        service = self._make_service(tmp_path)
        image_data = _make_test_image()

        job_id = service.start_async_job([image_data])

        restored_service = self._make_service(tmp_path)
        status = restored_service.get_job_status(job_id)
        assert status is not None
        assert status["status"] in {"pending", "running", "completed"}
        assert status["total"] == 1

    @pytest.mark.asyncio
    async def test_failed_job_status_restores_from_persistent_store(self, tmp_path):
        class FailingProvider(MockProvider):
            async def extract_features(self, image_data: bytes, metadata: dict = None) -> ExtractionResult:
                raise RuntimeError("provider failed")

        service = self._make_service(tmp_path, provider=FailingProvider())
        image_data = _make_test_image()

        job_id = service.start_async_job([image_data])

        for _ in range(50):
            status = service.get_job_status(job_id)
            if status["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.05)

        restored_service = self._make_service(tmp_path, provider=FailingProvider())
        status = restored_service.get_job_status(job_id)
        assert status is not None
        assert status["status"] == "failed"
        assert "provider failed" in status["error"]

    def test_job_store_partial_json_read_does_not_delete_results(self, tmp_path):
        store = VisualTokenizerJobResultStore(tmp_path / "jobs")
        job_id = "00000000-0000-4000-8000-000000000002"
        result = ExtractionResult(image_index=0, success=True, features=_make_visual_features())
        result_ref = store.put(
            job_id,
            [result],
            return_tensors=True,
            created_at=1.0,
            updated_at=2.0,
        )
        metadata_path = os.path.join(result_ref, "results.json")
        with open(metadata_path, "w") as file:
            file.write("{")

        assert store.get_status(job_id) is None
        assert os.path.isdir(result_ref)

    def test_async_job_limit_rejects_unbounded_growth(self, tmp_path):
        service = self._make_service(tmp_path)
        service.max_jobs = 1
        service._jobs["existing"] = {
            "status": "running",
            "progress": 0,
            "total": 1,
            "result": None,
            "error": None,
            "created_at": 1.0,
            "updated_at": 1.0,
            "return_tensors": True,
        }

        with pytest.raises(RuntimeError, match="job queue full"):
            service.start_async_job([_make_test_image()])

    def test_get_job_status_not_found(self, tmp_path):
        service = self._make_service(tmp_path)
        assert service.get_job_status("nonexistent") is None

    def test_get_model_info(self, tmp_path):
        service = self._make_service(tmp_path)
        info = service.get_model_info()
        assert info["provider"] == "mock"
        assert info["model_name"] == "mock-model"

    def test_get_cache_stats(self, tmp_path):
        service = self._make_service(tmp_path)
        stats = service.get_cache_stats()
        assert stats["model_slug"] == "test-model"
        assert stats["total_entries"] == 0


# ============================================
# Provider Base Tests
# ============================================

class TestExtractionResult:
    """Test ExtractionResult dataclass."""

    def test_defaults(self):
        result = ExtractionResult(image_index=0)
        assert result.success is False
        assert result.features is None
        assert result.error is None
        assert result.from_cache is False

    def test_with_features(self):
        features = _make_visual_features()
        result = ExtractionResult(
            image_index=0,
            success=True,
            features=features,
            latency_ms=42.5,
        )
        assert result.features.num_visual_tokens == 256
        assert result.latency_ms == 42.5


class TestVisualFeatures:
    """Test VisualFeatures dataclass."""

    def test_creation(self):
        features = _make_visual_features(num_tokens=100, hidden_dim=512)
        assert features.image_embeds.shape == (100, 512)
        assert features.num_visual_tokens == 100
        assert features.hidden_dim == 512
        assert features.image_grid_thw == [1, 16, 16]


# ============================================
# API Endpoint Tests
# ============================================

class TestVisualTokenizerAPI:
    """Test API endpoints using FastAPI TestClient with mocked service."""

    @pytest.fixture
    def mock_service(self):
        """Create a mock service for API testing."""
        service = MagicMock()
        service.get_model_info.return_value = {
            "provider": "qwen3.5",
            "model_name": "Qwen/Qwen3.5-35B-A3B-FP8",
        }

        async def mock_tokenize(images, batch_size=4, force_recompute=False, metadatas=None):
            return [
                ExtractionResult(
                    image_index=i,
                    success=True,
                    features=_make_visual_features(),
                    latency_ms=50.0,
                    cache_path=f"/tmp/cache/{i}.safetensors",
                )
                for i in range(len(images))
            ]

        service.tokenize_images = mock_tokenize
        service.start_async_job.return_value = "test-job-123"
        service.get_job_status.return_value = {
            "status": "pending",
            "progress": 0,
            "total": 1,
            "result": None,
            "result_ref": None,
            "error": None,
            "return_tensors": True,
        }
        service.get_job_result.return_value = [
            ExtractionResult(
                image_index=0,
                success=True,
                features=_make_visual_features(),
                latency_ms=50.0,
                cache_path="/tmp/cache/0.safetensors",
            )
        ]
        return service

    @pytest.fixture
    def client(self, mock_service):
        """Create a TestClient with mocked service."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        from memorylayer_embed_server_enterprise.api.v1.visual_tokenizer import router

        app.include_router(router)

        # Mock the Variables dependency
        mock_v = MagicMock()
        mock_v.get.return_value = mock_service

        async def mock_get_variables(request=None):
            return mock_v

        async def mock_get_logger(request=None):
            import logging
            return logging.getLogger("test")

        from memorylayer_embed_server.lifecycle.fastapi import get_logger, get_variables_dep
        app.dependency_overrides[get_variables_dep] = mock_get_variables
        app.dependency_overrides[get_logger] = mock_get_logger

        return TestClient(app)

    def test_post_visual_tokenize_sync(self, client):
        response = client.post("/v1/visual-tokenize", json={
            "images": [_make_test_image_b64()],
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["success"] is True
        assert data["stats"]["successful"] == 1
        assert data["model"] == "Qwen/Qwen3.5-35B-A3B-FP8"

    def test_post_visual_tokenize_multiple_images(self, client):
        response = client.post("/v1/visual-tokenize", json={
            "images": [_make_test_image_b64(), _make_test_image_b64(48, 48)],
            "batch_size": 2,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["stats"]["total_images"] == 2

    def test_post_visual_tokenize_async(self, client):
        response = client.post("/v1/visual-tokenize", json={
            "images": [_make_test_image_b64()],
            "async_mode": True,
        })
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["total_images"] == 1

    def test_post_visual_tokenize_invalid_base64(self, client):
        response = client.post("/v1/visual-tokenize", json={
            "images": ["not-valid-base64!!!"],
        })
        assert response.status_code == 400
        assert "index 0" in response.json()["detail"]

    def test_return_tensors_true_includes_b64(self, client):
        response = client.post("/v1/visual-tokenize", json={
            "images": [_make_test_image_b64()],
            "return_tensors": True,
        })
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["image_embeds_b64"]
        assert result["image_grid_thw_b64"]

    def test_return_tensors_false_omits_b64(self, client):
        response = client.post("/v1/visual-tokenize", json={
            "images": [_make_test_image_b64()],
            "return_tensors": False,
        })
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["image_embeds_b64"] is None
        assert result["image_grid_thw_b64"] is None

    def test_metadata_length_mismatch(self, client):
        response = client.post("/v1/visual-tokenize", json={
            "images": [_make_test_image_b64(), _make_test_image_b64(48, 48)],
            "metadata": [{"page_no": 0}],
        })
        assert response.status_code == 400

    def test_get_job_status(self, client):
        response = client.get("/v1/visual-tokenize/status/test-job-123")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["status"] == "pending"

    def test_get_completed_job_status_loads_persisted_result(self, client, mock_service):
        mock_service.get_job_status.return_value = {
            "status": "completed",
            "progress": 1,
            "total": 1,
            "result": None,
            "result_ref": "/tmp/jobs/test-job-123",
            "error": None,
            "return_tensors": True,
            "tensors_persisted": True,
            "artifact_retention": "metadata_and_tensors_until_job_ttl",
        }

        response = client.get("/v1/visual-tokenize/status/test-job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["tensors_persisted"] is True
        assert data["artifact_retention"] == "metadata_and_tensors_until_job_ttl"
        assert data["result"]["results"][0]["image_embeds_b64"]
        assert data["result"]["stats"]["successful"] == 1
        mock_service.get_job_result.assert_called_once_with("test-job-123")

    def test_get_completed_job_status_reports_metadata_only_result(self, client, mock_service):
        mock_service.get_job_status.return_value = {
            "status": "completed",
            "progress": 1,
            "total": 1,
            "result": None,
            "result_ref": "/tmp/jobs/test-job-123",
            "error": None,
            "return_tensors": False,
            "tensors_persisted": False,
            "artifact_retention": "metadata_only_until_job_ttl",
        }
        mock_service.get_job_result.return_value = [
            ExtractionResult(
                image_index=0,
                success=True,
                feature_metadata=VisualFeatureMetadata(
                    image_grid_thw=[1, 16, 16],
                    num_visual_tokens=256,
                    hidden_dim=2048,
                    dtype="torch.float32",
                    original_image_size=(64, 64),
                ),
                latency_ms=50.0,
            )
        ]

        response = client.get("/v1/visual-tokenize/status/test-job-123")

        assert response.status_code == 200
        data = response.json()
        result = data["result"]["results"][0]
        assert data["tensors_persisted"] is False
        assert result["num_visual_tokens"] == 256
        assert result["hidden_dim"] == 2048
        assert result["image_embeds_b64"] is None
        assert data["result"]["stats"]["successful"] == 1
        assert data["result"]["stats"]["total_visual_tokens"] == 256

    def test_get_job_status_not_found(self, client, mock_service):
        mock_service.get_job_status.return_value = None
        response = client.get("/v1/visual-tokenize/status/nonexistent")
        assert response.status_code == 404

    def test_completed_job_status_missing_result_returns_gone(self, client, mock_service):
        mock_service.get_job_status.return_value = {
            "status": "completed",
            "progress": 1,
            "total": 1,
            "result": None,
            "result_ref": "/tmp/jobs/test-job-123",
            "error": None,
            "return_tensors": True,
        }
        mock_service.get_job_result.return_value = None

        response = client.get("/v1/visual-tokenize/status/test-job-123")

        assert response.status_code == 410

    def test_service_unavailable(self):
        """Test 503 when service is not configured."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from memorylayer_embed_server.lifecycle.fastapi import get_logger, get_variables_dep

        from memorylayer_embed_server_enterprise.api.v1.visual_tokenizer import router

        app = FastAPI()
        app.include_router(router)

        mock_v = MagicMock()
        mock_v.get.return_value = None  # Service not configured

        async def mock_get_variables(request=None):
            return mock_v

        async def mock_get_logger(request=None):
            import logging
            return logging.getLogger("test")

        app.dependency_overrides[get_variables_dep] = mock_get_variables
        app.dependency_overrides[get_logger] = mock_get_logger

        client = TestClient(app)
        response = client.post("/v1/visual-tokenize", json={
            "images": [_make_test_image_b64()],
        })
        assert response.status_code == 503


# ============================================
# Qwen3.5 Provider Unit Tests (mocked model)
# ============================================

class TestQwen35VisualTokenizer:
    """Test Qwen3.5 provider with mocked HuggingFace models."""

    def test_model_slug_generation(self):
        from memorylayer_embed_server_enterprise.services.visual_tokenizer.qwen35_provider import (
            Qwen35VisualTokenizer,
        )
        slug = Qwen35VisualTokenizer._make_model_slug("Qwen/Qwen3.5-35B-A3B-FP8")
        assert slug == "qwen--qwen3.5-35b-a3b-fp8"
        assert "/" not in slug

    def test_model_slug_property(self):
        from memorylayer_embed_server_enterprise.services.visual_tokenizer.qwen35_provider import (
            Qwen35VisualTokenizer,
        )
        provider = Qwen35VisualTokenizer.__new__(Qwen35VisualTokenizer)
        provider._model_slug = "test-slug"
        assert provider.model_slug == "test-slug"

    def test_get_model_info_unloaded(self):
        from memorylayer_embed_server_enterprise.services.visual_tokenizer.qwen35_provider import (
            Qwen35VisualTokenizer,
        )
        provider = Qwen35VisualTokenizer.__new__(Qwen35VisualTokenizer)
        provider.model_name = "test-model"
        provider._model_slug = "test-slug"
        provider._vision_only_loaded = False
        provider._partial_loaded = False
        provider.torch_dtype = "auto"
        provider.embed_kind = "image_embeds"
        provider.max_image_dim = 0
        provider._model = None

        info = provider.get_model_info()
        assert info["loaded"] is False
        assert info["provider"] == "qwen3.5"
        assert info["embed_kind"] == "image_embeds"
        assert info["partial_loaded"] is False
        assert info["max_image_dim"] == 0


# ============================================
# Image-embed + grid codec
# ============================================

class TestImageEmbedCodec:
    """Round-trip + compression behavior of the base64 tensor codec."""

    def test_roundtrip_compressed_2d_float(self):
        from memorylayer_embed_server_enterprise.services.visual_tokenizer.codec import (
            b64_to_tensor,
            tensor_to_b64,
        )
        t = torch.randn(7, 16)
        restored = b64_to_tensor(tensor_to_b64(t, compress=True))
        assert restored.shape == (7, 16)
        assert torch.allclose(restored, t)

    def test_roundtrip_raw_grid_int64(self):
        from memorylayer_embed_server_enterprise.services.visual_tokenizer.codec import (
            b64_to_tensor,
            tensor_to_b64,
        )
        grid = torch.tensor([1, 16, 16], dtype=torch.int64)
        restored = b64_to_tensor(tensor_to_b64(grid, compress=False))
        assert restored.dtype == torch.int64
        assert restored.tolist() == [1, 16, 16]

    def test_roundtrip_uncompressed_2d_float(self):
        """Auto-detect must also read an uncompressed float payload."""
        from memorylayer_embed_server_enterprise.services.visual_tokenizer.codec import (
            b64_to_tensor,
            tensor_to_b64,
        )
        t = torch.randn(3, 8)
        restored = b64_to_tensor(tensor_to_b64(t, compress=False))
        assert torch.allclose(restored, t)

    def test_compressed_payload_carries_zstd_magic(self):
        import base64

        from memorylayer_embed_server_enterprise.services.visual_tokenizer.codec import (
            ZSTD_MAGIC,
            tensor_to_b64,
        )
        raw = base64.b64decode(tensor_to_b64(torch.randn(5, 8), compress=True), validate=True)
        assert raw[:4] == ZSTD_MAGIC

    def test_uncompressed_payload_loads_via_torch(self):
        """The uncompressed grid payload must deserialize via torch.load directly."""
        import base64
        from io import BytesIO

        from memorylayer_embed_server_enterprise.services.visual_tokenizer.codec import tensor_to_b64

        encoded = tensor_to_b64(torch.tensor([1, 2, 3], dtype=torch.int64), compress=False)
        tensor = torch.load(
            BytesIO(base64.b64decode(encoded, validate=True)),
            weights_only=True, map_location="cpu",
        )
        assert isinstance(tensor, torch.Tensor)
        assert tensor.tolist() == [1, 2, 3]


# ============================================
# Image-embed extraction (fake vision model)
# ============================================

class _FakeVisionOutput:
    """Mimics an HF vision output object exposing ``pooler_output``."""

    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class _FakeVisionModel:
    """Minimal stand-in exposing get_image_features + a config grid factor."""

    def __init__(self, hidden: int = 8, n_image_tokens: int = 4, spatial_merge_size: int = 1):
        self.device = "cpu"
        self.hidden = hidden
        self.n_image_tokens = n_image_tokens
        self.config = SimpleNamespace(
            hidden_size=hidden,
            vision_config=SimpleNamespace(spatial_merge_size=spatial_merge_size),
        )

    def get_image_features(self, pixel_values, image_grid_thw):
        # Distinct, recognizable rows so we can assert they survive the codec.
        rows = torch.arange(
            self.n_image_tokens * self.hidden, dtype=torch.float32,
        ).reshape(self.n_image_tokens, self.hidden)
        return _FakeVisionOutput(rows)


class _FakeImageProcessor:
    """Callable processor returning pixel_values + image_grid_thw."""

    def __init__(self, n_image_tokens: int, grid=(1, 2, 2)):
        self._n = n_image_tokens
        self._grid = grid

    def __call__(self, images, text="", return_tensors="pt"):
        return {
            "pixel_values": torch.randn(self._n, 4),
            "image_grid_thw": torch.tensor([list(self._grid)], dtype=torch.long),
        }


def _make_image_provider(hidden=8, n_image_tokens=4, spatial_merge_size=1, grid=(1, 2, 2)):
    from memorylayer_embed_server_enterprise.services.visual_tokenizer.qwen35_provider import (
        EMBED_KIND_IMAGE,
        Qwen35VisualTokenizer,
    )
    p = Qwen35VisualTokenizer.__new__(Qwen35VisualTokenizer)
    p.logger = MagicMock()
    p.model_name = "Qwen/Qwen3.6-27B-FP8"
    p._model_slug = "qwen--qwen3.6-27b-fp8"
    p.embed_kind = EMBED_KIND_IMAGE
    p.vision_only = False
    p.torch_dtype = "auto"
    p.max_image_dim = 0
    p._vision_only_loaded = False
    p._partial_loaded = False
    p._model = _FakeVisionModel(
        hidden=hidden, n_image_tokens=n_image_tokens, spatial_merge_size=spatial_merge_size,
    )
    p._image_token_id = None
    p._processor = _FakeImageProcessor(n_image_tokens, grid=grid)
    return p


class TestImageEmbedExtraction:
    """Vision-tower image-embed extraction."""

    def test_extract_shape_and_kind(self):
        hidden, n_img = 8, 4
        # grid (1,2,2) with spatial_merge_size=1 → prod//1 = 4 tokens.
        p = _make_image_provider(hidden=hidden, n_image_tokens=n_img, spatial_merge_size=1)
        feats = p._extract_single(_make_test_image(), {"filename": "a.pdf", "page_no": 0})

        assert feats.embed_kind == "image_embeds"
        assert feats.image_embeds.shape == (n_img, hidden)
        assert feats.hidden_dim == hidden
        assert feats.num_visual_tokens == n_img
        assert feats.image_grid_thw == [1, 2, 2]
        assert feats.image_embeds.is_contiguous()

        # The projected vision rows (arange) survive verbatim.
        expected = torch.arange(n_img * hidden, dtype=torch.float32).reshape(n_img, hidden)
        assert torch.allclose(feats.image_embeds, expected)

    def test_extract_honors_spatial_merge_size(self):
        # grid (1,4,4)=16, spatial_merge_size=2 → 16//4 = 4 tokens.
        p = _make_image_provider(n_image_tokens=4, spatial_merge_size=2, grid=(1, 4, 4))
        feats = p._extract_single(_make_test_image(), None)
        assert feats.num_visual_tokens == 4

    def test_token_count_mismatch_raises(self):
        # Grid (1,2,2)=4 with merge=1 implies 4 tokens, but model yields 3.
        p = _make_image_provider(n_image_tokens=3, spatial_merge_size=1, grid=(1, 2, 2))
        with pytest.raises(RuntimeError, match="mismatch"):
            p._extract_single(_make_test_image(), None)

    def test_default_cache_key_suffix_empty(self):
        from memorylayer_embed_server_enterprise.services.visual_tokenizer.qwen35_provider import (
            Qwen35VisualTokenizer,
        )
        p = Qwen35VisualTokenizer.__new__(Qwen35VisualTokenizer)
        p.embed_kind = "image_embeds"
        p.max_image_dim = 0
        assert p.cache_key_suffix({"filename": "a.pdf"}) == "image_embeds-d0"


class TestPartialLoadConfig:
    """The decoder-elision logic used by the partial (low-VRAM) load."""

    def test_elides_nested_text_config(self):
        from memorylayer_embed_server_enterprise.services.visual_tokenizer.qwen35_provider import (
            Qwen35VisualTokenizer,
        )
        cfg = SimpleNamespace(
            num_hidden_layers=48,
            text_config=SimpleNamespace(num_hidden_layers=48, num_nextn_predict_layers=1),
        )
        applied = Qwen35VisualTokenizer._elide_decoder_layers(cfg)
        assert applied is True
        assert cfg.num_hidden_layers == 0
        assert cfg.text_config.num_hidden_layers == 0
        assert cfg.text_config.num_nextn_predict_layers == 0

    def test_elides_flat_config(self):
        from memorylayer_embed_server_enterprise.services.visual_tokenizer.qwen35_provider import (
            Qwen35VisualTokenizer,
        )
        cfg = SimpleNamespace(num_hidden_layers=32)
        assert Qwen35VisualTokenizer._elide_decoder_layers(cfg) is True
        assert cfg.num_hidden_layers == 0

    def test_returns_false_when_no_layers_attr(self):
        from memorylayer_embed_server_enterprise.services.visual_tokenizer.qwen35_provider import (
            Qwen35VisualTokenizer,
        )
        assert Qwen35VisualTokenizer._elide_decoder_layers(SimpleNamespace(foo=1)) is False

    def test_strip_generation_head_replaces_lm_head(self):
        from memorylayer_embed_server_enterprise.services.visual_tokenizer.qwen35_provider import (
            Qwen35VisualTokenizer,
        )
        p = Qwen35VisualTokenizer.__new__(Qwen35VisualTokenizer)
        p.logger = MagicMock()
        p._model = SimpleNamespace(lm_head=torch.nn.Linear(8, 32))
        p._strip_generation_head()
        assert isinstance(p._model.lm_head, torch.nn.Identity)
        # Idempotent: a second call is a no-op.
        p._strip_generation_head()
        assert isinstance(p._model.lm_head, torch.nn.Identity)

    def test_partial_load_honored_for_image_embeds(self):
        from memorylayer_embed_server_enterprise.services.visual_tokenizer.qwen35_provider import (
            Qwen35VisualTokenizer,
        )
        # Lazy load — constructing does not touch the model. partial_load is no
        # longer gated on embed_kind (the only kind is image_embeds).
        on = Qwen35VisualTokenizer(model_name="x/y", embed_kind="image_embeds", partial_load=True)
        off = Qwen35VisualTokenizer(model_name="x/y", embed_kind="image_embeds", partial_load=False)
        assert on.partial_load is True
        assert off.partial_load is False
        assert on.embed_kind == "image_embeds"


# ============================================
# Max-image-dim downsample
# ============================================

def _make_downscale_provider(max_image_dim: int):
    from memorylayer_embed_server_enterprise.services.visual_tokenizer.qwen35_provider import (
        Qwen35VisualTokenizer,
    )
    p = Qwen35VisualTokenizer.__new__(Qwen35VisualTokenizer)
    p.logger = MagicMock()
    p.max_image_dim = max_image_dim
    return p


class TestMaybeDownscale:
    """``_maybe_downscale`` caps the largest dimension while preserving aspect."""

    def test_resizes_1700x2200_to_927x1200(self):
        # Sample-page geometry: cap 1200 -> 1700*1200/2200=927, height clamped to cap.
        p = _make_downscale_provider(max_image_dim=1200)
        img = Image.new("RGB", (1700, 2200))
        out = p._maybe_downscale(img)
        assert out.size == (927, 1200)

    def test_image_under_cap_untouched(self):
        p = _make_downscale_provider(max_image_dim=1200)
        img = Image.new("RGB", (800, 600))
        out = p._maybe_downscale(img)
        # Returned verbatim (same object, no resize).
        assert out is img
        assert out.size == (800, 600)

    def test_disabled_when_zero(self):
        p = _make_downscale_provider(max_image_dim=0)
        img = Image.new("RGB", (1700, 2200))
        out = p._maybe_downscale(img)
        assert out is img
        assert out.size == (1700, 2200)

    def test_min_dimension_clamped_to_one(self):
        # Extremely thin image: scaled height rounds below 1 and is clamped.
        p = _make_downscale_provider(max_image_dim=10)
        img = Image.new("RGB", (1000, 3))
        out = p._maybe_downscale(img)
        assert out.size == (10, 1)
