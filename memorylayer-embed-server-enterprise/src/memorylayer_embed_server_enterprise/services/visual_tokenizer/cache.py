# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Disk cache for visual features using safetensors."""

import hashlib
import json
import time
from logging import Logger
from pathlib import Path

from scitrera_app_framework import Variables, get_logger

from .base import VisualFeatures


class VisualTokenCache:
    """Disk-based cache for extracted visual features.

    Stores tensors in safetensors format for zero-copy mmap loading.
    Metadata is stored in JSON sidecar files.

    Cache layout:
        {cache_dir}/{model_slug}/{image_sha256}.safetensors
        {cache_dir}/{model_slug}/{image_sha256}.metadata.json
    """

    def __init__(
        self,
        cache_dir: str,
        model_slug: str,
        max_cache_size_gb: float = 50.0,
        ttl_hours: float = 0,
        v: Variables = None,
    ):
        self.cache_dir = Path(cache_dir) / model_slug
        self.model_slug = model_slug
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024**3)
        self.ttl_seconds = ttl_hours * 3600 if ttl_hours > 0 else 0
        self.logger: Logger = get_logger(v, name="VisualTokenCache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "Visual token cache initialized: dir=%s, max_size=%.1fGB, ttl=%.0fh",
            self.cache_dir, max_cache_size_gb, ttl_hours,
        )

    @staticmethod
    def image_hash(image_data: bytes) -> str:
        """Compute SHA-256 hash of image bytes."""
        return hashlib.sha256(image_data).hexdigest()

    def _tensor_path(self, image_hash: str) -> Path:
        return self.cache_dir / f"{image_hash}.safetensors"

    def _metadata_path(self, image_hash: str) -> Path:
        return self.cache_dir / f"{image_hash}.metadata.json"

    def has(self, image_hash: str) -> bool:
        """Check if features are cached for the given image hash."""
        tensor_path = self._tensor_path(image_hash)
        metadata_path = self._metadata_path(image_hash)

        if not tensor_path.exists() or not metadata_path.exists():
            return False

        # Check TTL expiry
        if self.ttl_seconds > 0:
            mtime = tensor_path.stat().st_mtime
            if time.time() - mtime > self.ttl_seconds:
                self.logger.debug("Cache entry expired: %s", image_hash[:16])
                self._delete_entry(image_hash)
                return False

        return True

    def get(self, image_hash: str) -> VisualFeatures | None:
        """Load cached visual features.

        Returns None if cache miss or corruption detected.
        """
        tensor_path = self._tensor_path(image_hash)
        metadata_path = self._metadata_path(image_hash)

        if not tensor_path.exists() or not metadata_path.exists():
            return None

        try:
            from safetensors.torch import load_file
            tensors = load_file(str(tensor_path))
            image_embeds = tensors["image_embeds"]

            with open(metadata_path) as f:
                metadata = json.load(f)

            # Touch mtime for LRU tracking
            tensor_path.touch()

            return VisualFeatures(
                image_embeds=image_embeds,
                image_grid_thw=metadata["image_grid_thw"],
                num_visual_tokens=metadata["num_visual_tokens"],
                hidden_dim=metadata["hidden_dim"],
                dtype=metadata["dtype"],
                original_image_size=tuple(metadata["original_image_size"]),
                embed_kind=metadata.get("embed_kind", "image_embeds"),
            )
        except Exception as e:
            self.logger.warning(
                "Cache corruption for %s, deleting: %s", image_hash[:16], e,
            )
            self._delete_entry(image_hash)
            return None

    def put(self, image_hash: str, features: VisualFeatures) -> str:
        """Save visual features to cache.

        Returns the path to the saved tensor file.
        """
        self._evict_if_needed(features.image_embeds.nbytes)

        tensor_path = self._tensor_path(image_hash)
        metadata_path = self._metadata_path(image_hash)

        try:
            from safetensors.torch import save_file
            save_file({"image_embeds": features.image_embeds}, str(tensor_path))

            metadata = {
                "image_grid_thw": features.image_grid_thw,
                "num_visual_tokens": features.num_visual_tokens,
                "hidden_dim": features.hidden_dim,
                "dtype": features.dtype,
                "original_image_size": list(features.original_image_size),
                "embed_kind": features.embed_kind,
                "model_slug": self.model_slug,
                "cached_at": time.time(),
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.debug(
                "Cached features: %s (%d tokens, %d dim)",
                image_hash[:16], features.num_visual_tokens, features.hidden_dim,
            )
            return str(tensor_path)

        except Exception as e:
            self.logger.warning("Failed to cache features for %s: %s", image_hash[:16], e)
            self._delete_entry(image_hash)
            raise

    def _delete_entry(self, image_hash: str):
        """Delete a cache entry (tensor + metadata)."""
        for path in (self._tensor_path(image_hash), self._metadata_path(image_hash)):
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                self.logger.debug("Failed to delete %s: %s", path, e)

    def _evict_if_needed(self, incoming_bytes: int = 0):
        """Evict oldest entries (by mtime) if cache exceeds max size."""
        if self.max_cache_size_bytes <= 0:
            return

        current_size = self._get_cache_size()
        if current_size + incoming_bytes <= self.max_cache_size_bytes:
            return

        self.logger.info(
            "Cache eviction triggered: current=%.2fGB, max=%.2fGB",
            current_size / 1024**3, self.max_cache_size_bytes / 1024**3,
        )

        # Collect safetensor files sorted by mtime (oldest first)
        entries = []
        for st_path in self.cache_dir.glob("*.safetensors"):
            try:
                stat = st_path.stat()
                entries.append((stat.st_mtime, stat.st_size, st_path.stem))
            except OSError:
                continue

        entries.sort()  # oldest first

        target = self.max_cache_size_bytes - incoming_bytes
        evicted = 0
        for _mtime, size, image_hash in entries:
            if current_size <= target:
                break
            self._delete_entry(image_hash)
            current_size -= size
            evicted += 1

        if evicted:
            self.logger.info("Evicted %d cache entries", evicted)

    def _get_cache_size(self) -> int:
        """Get total size of all cached files in bytes."""
        total = 0
        for path in self.cache_dir.iterdir():
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def clear(self):
        """Clear all cached entries."""
        count = 0
        for path in self.cache_dir.iterdir():
            try:
                path.unlink()
                count += 1
            except OSError:
                continue
        self.logger.info("Cleared %d cache files", count)

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total_size = 0
        tensor_count = 0
        oldest_mtime = float("inf")
        newest_mtime = 0.0

        for st_path in self.cache_dir.glob("*.safetensors"):
            try:
                stat = st_path.stat()
                total_size += stat.st_size
                tensor_count += 1
                oldest_mtime = min(oldest_mtime, stat.st_mtime)
                newest_mtime = max(newest_mtime, stat.st_mtime)
            except OSError:
                continue

        return {
            "cache_dir": str(self.cache_dir),
            "model_slug": self.model_slug,
            "total_entries": tensor_count,
            "total_size_gb": round(total_size / 1024**3, 3),
            "max_size_gb": round(self.max_cache_size_bytes / 1024**3, 3),
            "ttl_hours": self.ttl_seconds / 3600 if self.ttl_seconds > 0 else 0,
            "oldest_entry_age_hours": (
                round((time.time() - oldest_mtime) / 3600, 1)
                if tensor_count > 0 else 0
            ),
        }
