"""Base visual tokenizer provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger

import torch
from scitrera_app_framework import Variables, get_logger


@dataclass
class VisualFeatureMetadata:
    """Tensor-free metadata for one page's extracted visual features."""
    image_grid_thw: list[int]               # [temporal, height, width]
    num_visual_tokens: int                  # count of vision-encoder tokens
    hidden_dim: int
    dtype: str
    original_image_size: tuple[int, int]    # (width, height)
    embed_kind: str = "image_embeds"        # always "image_embeds"


@dataclass
class VisualFeatures:
    """Extracted features for a single page.

    Holds the **vision-tower image embeds** — the projected per-image
    ``pooler_output`` of shape ``[num_visual_tokens, hidden_dim]`` (where
    ``num_visual_tokens == prod(image_grid_thw) // spatial_merge_size**2``).
    These are shipped to vLLM as an ``image_embeds`` content part alongside
    ``image_grid_thw`` so the model can apply M-RoPE 2-D positions to the image
    tokens (the dropped flat ``prompt_embeds`` path gave them 1-D positions).

    The tensor lives in ``image_embeds`` (also the safetensors cache key).
    """
    image_embeds: torch.Tensor              # [num_visual_tokens, hidden_dim]
    image_grid_thw: list[int]               # [temporal, height, width]
    num_visual_tokens: int                  # count of vision-encoder tokens
    hidden_dim: int
    dtype: str
    original_image_size: tuple[int, int]    # (width, height)
    embed_kind: str = "image_embeds"        # always "image_embeds"


@dataclass
class ExtractionResult:
    """Result of extracting visual features from a single image."""
    image_index: int
    success: bool = False
    features: VisualFeatures | None = None
    feature_metadata: VisualFeatureMetadata | None = None
    error: str | None = None
    latency_ms: float = 0.0
    from_cache: bool = False
    cache_path: str | None = None


class VisualTokenizerProvider(ABC):
    """Abstract base class for visual tokenizer providers."""

    PROVIDER_NAME: str = ""

    def __init__(self, v: Variables = None):
        self.logger: Logger = get_logger(v, name=self.__class__.__name__)

    @abstractmethod
    async def preload(self):
        """Preload model resources onto GPU."""

    @abstractmethod
    async def extract_features(
        self, image_data: bytes, metadata: dict | None = None,
    ) -> ExtractionResult:
        """Extract features from a single page.

        Args:
            image_data: Raw image bytes (PNG/JPEG).
            metadata: Optional stable page metadata (filename, page_no, source
                attribution). Informational only; the image-embeds path derives
                features from the image alone.

        Returns:
            ExtractionResult with extracted features or error.
        """

    async def extract_features_batch(
        self,
        images: list[bytes],
        batch_size: int = 4,
        metadatas: list[dict] | None = None,
    ) -> list[ExtractionResult]:
        """Extract features from multiple pages.

        Default implementation processes sequentially. Subclasses may
        override with true batched GPU inference.

        Args:
            images: List of raw image bytes.
            batch_size: GPU batch size (used by subclass implementations).
            metadatas: Optional per-image metadata dicts, aligned by index.

        Returns:
            List of ExtractionResult, one per input image.
        """
        results = []
        for idx, image_data in enumerate(images):
            meta = metadatas[idx] if metadatas and idx < len(metadatas) else None
            result = await self.extract_features(image_data, meta)
            result.image_index = idx
            results.append(result)
        return results

    def cache_key_suffix(self, metadata: dict | None) -> str:
        """Extra cache-key component that distinguishes results for the same image.

        The base cache keys by image bytes alone. Providers whose output
        depends on more than the image override this to return a stable
        fingerprint so different inputs for the same image don't collide.
        Default: no suffix (image hash is sufficient).
        """
        return ""

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return metadata about the loaded model."""
