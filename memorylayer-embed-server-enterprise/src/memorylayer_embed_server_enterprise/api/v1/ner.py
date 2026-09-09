# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""GLiNER2 typed-NER API endpoint (POST /v1/ner).

Integrated replacement for the standalone ``ner_service.py`` experiment: the
router is registered on the main embed server via ``EXT_MULTI_API_ROUTERS`` and
pulls the preloaded :class:`GLiNER2NERService` from the framework's Variables.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from memorylayer_embed_server.api import EXT_MULTI_API_ROUTERS
from memorylayer_embed_server.lifecycle.fastapi import get_logger, get_variables_dep
from scitrera_app_framework import Plugin, Variables

from ...models.ner import NERRequest, NERResponse, NERTextResult

router = APIRouter(prefix="/v1", tags=["ner"])


def _get_service(v: Variables):
    """Get the GLiNER2 NER service or raise 503.

    Returns 503 both when the service was never configured (plugin disabled) and
    when it was configured but the model failed to load.
    """
    service = v.get("gliner2_ner_service", default=None)
    if service is None or not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NER service not available (model not loaded)",
        )
    return service


@router.post("/ner", response_model=NERResponse)
async def ner(
    request: NERRequest,
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> NERResponse:
    """Extract typed entities from a batch of texts.

    Each text is processed independently; results are returned in the same order
    as the input texts. When ``labels`` is omitted the server's configured
    default labels are used.
    """
    if not request.texts:
        service = _get_service(v)
        return NERResponse(results=[], model=service.model_name)

    service = _get_service(v)

    try:
        batch_results = await service.extract_batch(request.texts, request.labels)
    except Exception as exc:  # noqa: BLE001 - surface as 500
        logger.error("NER batch inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NER inference failed: {exc}",
        )

    logger.info("NER request: %d text(s)", len(request.texts))
    return NERResponse(
        results=[NERTextResult(entities=ents) for ents in batch_results],
        model=service.model_name,
    )


class NERAPIPlugin(Plugin):
    """Plugin to register NER API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False  # multi-extension pattern

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
