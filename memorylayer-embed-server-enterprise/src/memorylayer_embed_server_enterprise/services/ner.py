"""GLiNER2 typed-NER service for the embed server.

Loads ``fastino/gliner2-base-v1`` (config-overridable) once at startup and runs
synchronous ``extract_entities`` calls inside ``asyncio.to_thread`` so the async
event loop stays unblocked. If the model fails to load the service NO-OPs
gracefully (``is_ready`` stays False) and the endpoint returns 503 rather than
crashing the server.

Ported from the standalone ``ner_service.py`` experiment (which ran a separate
FastAPI app on :61055) into the embed-server package so ``/v1/ner`` is served by
the main embed server (:61051) and survives restarts.

GLiNER2 model contract:
    GLiNER2.from_pretrained(model_id).extract_entities(text, [labels])
        -> {"entities": {label: [spans]}}
"""

from __future__ import annotations

import asyncio
from logging import Logger

from scitrera_app_framework import Variables, get_logger


class GLiNER2NERService:
    """Wraps a GLiNER2 model with async-friendly batch entity extraction.

    The model load is deferred to :meth:`preload` (called from the lifecycle
    plugin's ``async_ready`` hook) so the heavy import + weight download happens
    once at startup rather than on the first request.
    """

    def __init__(
        self,
        *,
        model_name: str,
        default_labels: list[str],
        v: Variables = None,
    ):
        self.model_name = model_name
        self.default_labels = default_labels
        self.logger: Logger = get_logger(v, name="GLiNER2NERService")
        self._model = None
        self.logger.info(
            "GLiNER2NERService configured: model=%s default_labels=%s",
            model_name, default_labels,
        )

    @property
    def is_ready(self) -> bool:
        """True once the model is loaded and able to serve requests."""
        return self._model is not None

    async def preload(self) -> None:
        """Load the GLiNER2 model. Safe to call once; subsequent calls no-op.

        Loading runs in a thread because ``from_pretrained`` is blocking (and may
        download weights). A load failure is logged and left non-fatal so the
        rest of the embed server still starts.
        """
        if self._model is not None:
            return
        self.logger.info("Loading GLiNER2 model: %s", self.model_name)
        try:
            from gliner2 import GLiNER2
            self._model = await asyncio.to_thread(GLiNER2.from_pretrained, self.model_name)
            self.logger.info("GLiNER2 model loaded successfully")
        except Exception as exc:  # noqa: BLE001 - non-fatal startup hardening
            self.logger.error("Failed to load GLiNER2 model %s: %s", self.model_name, exc)
            self._model = None

    async def extract_batch(
        self,
        texts: list[str],
        labels: list[str] | None = None,
    ) -> list[dict[str, list[str]]]:
        """Extract typed entities for each text.

        Returns a list aligned with ``texts``; each element is a
        ``{label: [spans]}`` map. Per-text model failures are raised so callers
        can fall back instead of silently treating failures as no entities.

        Raises ``RuntimeError`` if the model is not loaded so the endpoint can
        surface a 503.
        """
        if self._model is None:
            raise RuntimeError("GLiNER2 model not loaded")
        if not texts:
            return []

        effective_labels = labels if labels else self.default_labels

        def _run_batch() -> list[dict[str, list[str]]]:
            out: list[dict[str, list[str]]] = []
            for index, text in enumerate(texts):
                try:
                    result = self._model.extract_entities(text, effective_labels)
                except Exception as exc:  # noqa: BLE001 - converted at service boundary
                    raise RuntimeError(
                        f"NER inference failed for text index {index}: {exc}"
                    ) from exc
                out.append(result.get("entities", {}))
            return out

        return await asyncio.to_thread(_run_batch)

    def get_model_info(self) -> dict:
        """Health/info payload for diagnostics."""
        return {
            "model_name": self.model_name,
            "loaded": self.is_ready,
            "default_labels": self.default_labels,
        }
