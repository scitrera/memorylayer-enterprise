# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Plugin that wires the Qwen3.5 visual-tokenizer service into the embed-server.

When this enterprise package is installed alongside ``memorylayer-embed-server``
the OSS server's ``register_package_plugins`` loop discovers this plugin and
calls ``initialize`` (synchronously) followed by ``async_ready`` (during
startup). The service instance is also stashed under the ``visual_tokenizer_service``
Variables key so any opportunistic OSS-side lookups continue to work.

The plugin gates itself on ``EMBED_SERVER_VISUAL_TOKENIZER_ENABLED``; if the
operator has not explicitly enabled visual tokenization, ``is_enabled`` returns
False and the plugin is skipped.
"""
from __future__ import annotations

from logging import Logger

from scitrera_app_framework import Plugin, Variables, ext_parse_bool

from ..config import (
    DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_CACHE_DIR,
    DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_CACHE_MAX_GB,
    DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_CACHE_TTL_HOURS,
    DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_EMBED_KIND,
    DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_ENABLED,
    DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_JOB_TTL_SECONDS,
    DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_MAX_IMAGE_DIM,
    DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_MAX_JOBS,
    DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_MODEL,
    DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_PARTIAL_LOAD,
    DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_TORCH_DTYPE,
    DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_VISION_ONLY,
    EMBED_SERVER_VISUAL_TOKENIZER_CACHE_DIR,
    EMBED_SERVER_VISUAL_TOKENIZER_CACHE_MAX_GB,
    EMBED_SERVER_VISUAL_TOKENIZER_CACHE_TTL_HOURS,
    EMBED_SERVER_VISUAL_TOKENIZER_EMBED_KIND,
    EMBED_SERVER_VISUAL_TOKENIZER_ENABLED,
    EMBED_SERVER_VISUAL_TOKENIZER_JOB_TTL_SECONDS,
    EMBED_SERVER_VISUAL_TOKENIZER_MAX_IMAGE_DIM,
    EMBED_SERVER_VISUAL_TOKENIZER_MAX_JOBS,
    EMBED_SERVER_VISUAL_TOKENIZER_MODEL,
    EMBED_SERVER_VISUAL_TOKENIZER_PARTIAL_LOAD,
    EMBED_SERVER_VISUAL_TOKENIZER_TORCH_DTYPE,
    EMBED_SERVER_VISUAL_TOKENIZER_VISION_ONLY,
)

EXT_VISUAL_TOKENIZER_SERVICE = "visual-tokenizer-service"


class VisualTokenizerServicePlugin(Plugin):
    """Lifecycle plugin that constructs and preloads the visual-tokenizer service."""

    def name(self) -> str:
        return EXT_VISUAL_TOKENIZER_SERVICE

    def extension_point_name(self, v: Variables) -> str:
        return EXT_VISUAL_TOKENIZER_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return v.environ(
            EMBED_SERVER_VISUAL_TOKENIZER_ENABLED,
            default=DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_ENABLED,
            type_fn=ext_parse_bool,
        )

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        try:
            from ..services.visual_tokenizer.qwen35_provider import Qwen35VisualTokenizer
        except ImportError as e:
            logger.warning(
                "Qwen3.5 visual tokenizer unavailable (transformers not installed): %s", e,
            )
            return None

        from ..services.visual_tokenizer.cache import VisualTokenCache
        from ..services.visual_tokenizer.service import VisualTokenizerService

        model_name = v.environ(
            EMBED_SERVER_VISUAL_TOKENIZER_MODEL,
            default=DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_MODEL,
        )
        vision_only = v.environ(
            EMBED_SERVER_VISUAL_TOKENIZER_VISION_ONLY,
            default=DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_VISION_ONLY,
            type_fn=ext_parse_bool,
        )
        embed_kind = v.environ(
            EMBED_SERVER_VISUAL_TOKENIZER_EMBED_KIND,
            default=DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_EMBED_KIND,
        )
        partial_load = v.environ(
            EMBED_SERVER_VISUAL_TOKENIZER_PARTIAL_LOAD,
            default=DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_PARTIAL_LOAD,
            type_fn=ext_parse_bool,
        )
        torch_dtype = v.environ(
            EMBED_SERVER_VISUAL_TOKENIZER_TORCH_DTYPE,
            default=DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_TORCH_DTYPE,
        )
        cache_dir = v.environ(
            EMBED_SERVER_VISUAL_TOKENIZER_CACHE_DIR,
            default=DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_CACHE_DIR,
        )
        cache_max_gb = v.environ(
            EMBED_SERVER_VISUAL_TOKENIZER_CACHE_MAX_GB,
            default=DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_CACHE_MAX_GB,
            type_fn=float,
        )
        cache_ttl_hours = v.environ(
            EMBED_SERVER_VISUAL_TOKENIZER_CACHE_TTL_HOURS,
            default=DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_CACHE_TTL_HOURS,
            type_fn=float,
        )
        max_image_dim = v.environ(
            EMBED_SERVER_VISUAL_TOKENIZER_MAX_IMAGE_DIM,
            default=DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_MAX_IMAGE_DIM,
            type_fn=int,
        )
        max_jobs = v.environ(
            EMBED_SERVER_VISUAL_TOKENIZER_MAX_JOBS,
            default=DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_MAX_JOBS,
            type_fn=int,
        )
        job_ttl_seconds = v.environ(
            EMBED_SERVER_VISUAL_TOKENIZER_JOB_TTL_SECONDS,
            default=DEFAULT_EMBED_SERVER_VISUAL_TOKENIZER_JOB_TTL_SECONDS,
            type_fn=float,
        )

        provider = Qwen35VisualTokenizer(
            v=v,
            model_name=model_name,
            vision_only=vision_only,
            torch_dtype=torch_dtype,
            embed_kind=embed_kind,
            partial_load=partial_load,
            max_image_dim=max_image_dim,
        )
        cache = VisualTokenCache(
            cache_dir=cache_dir,
            model_slug=provider.model_slug,
            max_cache_size_gb=cache_max_gb,
            ttl_hours=cache_ttl_hours,
            v=v,
        )
        service = VisualTokenizerService(
            provider=provider,
            cache=cache,
            v=v,
            max_jobs=max_jobs,
            job_ttl_seconds=job_ttl_seconds,
        )

        # Stash under the canonical Variables key for any code that still
        # looks the service up by name (tests, future overlays).
        v.set("visual_tokenizer_service", service)

        # Register a health-check contributor so /health/ready surfaces
        # visual-tokenizer status without OSS having to know about it.
        def _vt_health_check(checks: dict) -> None:
            checks.setdefault("services", {})["visual_tokenizer"] = {
                "status": "available",
                **service.get_model_info(),
            }

        existing_hooks = v.get("health_check_callables", default=[])
        v.set("health_check_callables", [*existing_hooks, _vt_health_check])

        logger.info(
            "Visual tokenizer configured: model=%s, vision_only=%s, cache_dir=%s",
            model_name, vision_only, cache_dir,
        )
        return service


    def shutdown(self, v: Variables, logger: Logger, value: object | None) -> None:
        """Cancel in-flight async visual-tokenizer jobs during service shutdown."""
        if value is None or not hasattr(value, "cancel_all_jobs"):
            return
        value.cancel_all_jobs()
        logger.info("Visual tokenizer async jobs cancelled")

    async def async_ready(self, v: Variables, logger: Logger, value: object | None) -> None:
        """Preload the model at startup so first request isn't blocked on warm-up."""
        if value is None:
            return
        try:
            await value.preload()
            logger.info("Visual tokenizer preloaded")
        except Exception as e:  # noqa: BLE001 - non-fatal startup hardening
            logger.warning("Visual tokenizer preload failed (non-fatal): %s", e)
