# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Plugin that wires the GLiNER2 NER service into the embed-server.

When this enterprise package is installed alongside ``memorylayer-embed-server``
the OSS server's ``register_package_plugins`` loop discovers this plugin and
calls ``initialize`` (synchronously) followed by ``async_ready`` (during
startup, which preloads the model). The service instance is stashed under the
``gliner2_ner_service`` Variables key so the ``/v1/ner`` router can look it up.

The plugin gates itself on ``MEMORYLAYER_EMBED_GLINER2_ENABLED`` (default False);
if the operator has not explicitly enabled NER, ``is_enabled`` returns False and
the plugin is skipped.
"""
from __future__ import annotations

from logging import Logger

from scitrera_app_framework import Plugin, Variables, ext_parse_bool

from ..config import (
    DEFAULT_EMBED_SERVER_GLINER2_ENABLED,
    DEFAULT_EMBED_SERVER_GLINER2_LABELS,
    DEFAULT_EMBED_SERVER_GLINER2_MODEL,
    EMBED_SERVER_GLINER2_ENABLED,
    EMBED_SERVER_GLINER2_LABELS,
    EMBED_SERVER_GLINER2_MODEL,
)

EXT_GLINER2_NER_SERVICE = "gliner2-ner-service"


class GLiNER2NERServicePlugin(Plugin):
    """Lifecycle plugin that constructs and preloads the GLiNER2 NER service."""

    def name(self) -> str:
        return EXT_GLINER2_NER_SERVICE

    def extension_point_name(self, v: Variables) -> str:
        return EXT_GLINER2_NER_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return v.environ(
            EMBED_SERVER_GLINER2_ENABLED,
            default=DEFAULT_EMBED_SERVER_GLINER2_ENABLED,
            type_fn=ext_parse_bool,
        )

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        from ..services.ner import GLiNER2NERService

        model_name = v.environ(
            EMBED_SERVER_GLINER2_MODEL,
            default=DEFAULT_EMBED_SERVER_GLINER2_MODEL,
        )
        labels_csv = v.environ(
            EMBED_SERVER_GLINER2_LABELS,
            default=DEFAULT_EMBED_SERVER_GLINER2_LABELS,
        )
        default_labels = [lbl.strip() for lbl in labels_csv.split(",") if lbl.strip()]

        service = GLiNER2NERService(
            model_name=model_name,
            default_labels=default_labels,
            v=v,
        )

        # Stash under the canonical Variables key so the /v1/ner router resolves
        # the service from the framework.
        v.set("gliner2_ner_service", service)

        # Register a health-check contributor so /health/ready surfaces NER
        # status without OSS having to know about it.
        def _ner_health_check(checks: dict) -> None:
            checks.setdefault("services", {})["gliner2_ner"] = {
                "status": "available" if service.is_ready else "loading",
                **service.get_model_info(),
            }

        existing_hooks = v.get("health_check_callables", default=[])
        v.set("health_check_callables", [*existing_hooks, _ner_health_check])

        logger.info("GLiNER2 NER configured: model=%s", model_name)
        return service

    async def async_ready(self, v: Variables, logger: Logger, value: object | None) -> None:
        """Preload the model at startup so the first request isn't blocked on warm-up."""
        if value is None:
            return
        try:
            await value.preload()
            logger.info("GLiNER2 NER preloaded")
        except Exception as e:  # noqa: BLE001 - non-fatal; endpoint 503s until loaded
            logger.warning("GLiNER2 NER preload failed (non-fatal): %s", e)
