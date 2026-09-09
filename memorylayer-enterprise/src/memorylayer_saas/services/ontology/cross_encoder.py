# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Cross-encoder (NLI) relationship classifier for MemoryLayer enterprise.

Replaces the per-pair / batched **LLM** relationship classification
(``DefaultOntologyService.classify_relationships_batch`` → a reasoning-model
call per new memory) with a small **NLI cross-encoder** served on the embed
stack. The cross-encoder scores each ``(anchor, candidate)`` pair as one of
``{entailment, neutral, contradiction}`` in a single batched HTTP call — no
generative LLM tokens, local GPU, ~free at the margin.

Why a cross-encoder and not NER/spaCy: auto-association's cost is *pairwise
relationship typing* between two memory chunks, which is a natural-language
inference problem, not entity extraction. NER (GLiNER2) already runs upstream to
supply entity-overlap signal; it cannot say how two memories relate. An NLI
cross-encoder maps directly onto the high-value relationship types:

    entailment    -> supports        (A entails/justifies B)
    contradiction -> contradicts     (gated; see below)
    neutral       -> related_to      (falls through to the similar_to default)

It intentionally does NOT reproduce the full ~60-type ontology — those fine
types (causes, blocks, preferred_over, …) are not measurably improving recall
(see the moat eval) and are exactly what makes the LLM call expensive. The
coarse {supports, contradicts, related_to} set is the part worth paying for.

Contradiction overlap (IMPORTANT):
    A separate deterministic ``ContradictionService`` already runs in the same
    post-store pipeline, detects contradictions (negation pairs + value
    conflicts), writes to its own ``contradictions`` table, and owns resolution
    (soft-delete / merge). Emitting ``contradicts`` graph edges here would be a
    THIRD, disjoint contradiction detector. So by default this provider does
    NOT emit ``contradicts`` (contradiction → ``related_to``); the deterministic
    service remains the single contradiction authority. Set
    ``MEMORYLAYER_CROSS_ENCODER_EMIT_CONTRADICTS=1`` only once the two paths are
    unified (the cross-encoder is a strictly better detector and is the natural
    place to converge them — see docs).

Selection:
    MEMORYLAYER_ONTOLOGY_SERVICE=cross_encoder

Config (env):
    MEMORYLAYER_NLI_URL                        NLI service base URL. When unset,
                                               resolves from MEMORYLAYER_EMBED_SERVER_URL.
    MEMORYLAYER_NLI_TIMEOUT                     HTTP timeout seconds (default 10).
    MEMORYLAYER_NLI_MIN_CONFIDENCE             Min softmax prob to accept a non-neutral
                                               label; below it -> related_to (default 0.60).
    MEMORYLAYER_CROSS_ENCODER_EMIT_CONTRADICTS Emit contradicts edges (default off;
                                               defer to ContradictionService).

Endpoint contract (POST {nli_url}/v1/nli):
    request : {"pairs": [{"premise": <anchor>, "hypothesis": <candidate>}, ...]}
    response: {"results": [{"label": "entailment"|"neutral"|"contradiction",
                            "score": <float 0-1>}, ...]}  # same order as pairs

Fail-safe:
    On any HTTP / parse error the batch degrades to ``related_to`` for every
    candidate (cheap, no LLM, never blocks ingest). Set the fallback to the
    inherited LLM path explicitly with ``MEMORYLAYER_CROSS_ENCODER_LLM_FALLBACK=1``
    if you would rather pay for LLM classification when the NLI lane is down.
"""

import asyncio
import json
import logging
import urllib.request

from memorylayer_server.services.llm import EXT_LLM_SERVICE
from memorylayer_server.services.ontology.base import OntologyServicePluginBase
from memorylayer_server.services.ontology.default import DefaultOntologyService
from scitrera_app_framework import Variables, ext_parse_bool, get_extension

# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------

MEMORYLAYER_NLI_URL = "MEMORYLAYER_NLI_URL"
DEFAULT_MEMORYLAYER_NLI_URL = ""  # unset -> resolve from embed server url

MEMORYLAYER_NLI_TIMEOUT = "MEMORYLAYER_NLI_TIMEOUT"
DEFAULT_MEMORYLAYER_NLI_TIMEOUT = "10"

MEMORYLAYER_NLI_MIN_CONFIDENCE = "MEMORYLAYER_NLI_MIN_CONFIDENCE"
DEFAULT_MEMORYLAYER_NLI_MIN_CONFIDENCE = "0.60"

MEMORYLAYER_CROSS_ENCODER_EMIT_CONTRADICTS = "MEMORYLAYER_CROSS_ENCODER_EMIT_CONTRADICTS"
MEMORYLAYER_CROSS_ENCODER_LLM_FALLBACK = "MEMORYLAYER_CROSS_ENCODER_LLM_FALLBACK"

# NLI label -> ontology relationship type. ``contradiction`` is resolved at call
# time because whether it emits ``contradicts`` or ``related_to`` is a config flag.
_ENTAILMENT_REL = "supports"
_NEUTRAL_REL = "related_to"


class CrossEncoderOntologyService(DefaultOntologyService):
    """Ontology service whose batched classifier uses an NLI cross-encoder.

    Inherits every relationship/subtype/entity-type method from
    :class:`DefaultOntologyService`; overrides only the batched classifier
    (and the single-pair path, which routes through it). The LLM service is
    still injected so the optional LLM fallback works.
    """

    def __init__(
        self,
        *,
        nli_url: str,
        nli_timeout: float,
        nli_min_confidence: float,
        emit_contradicts: bool,
        llm_fallback: bool,
        v: Variables = None,
        llm_service=None,
    ):
        super().__init__(v=v, llm_service=llm_service)
        self._nli_url = nli_url.rstrip("/")
        self._nli_timeout = nli_timeout
        self._nli_min_confidence = nli_min_confidence
        self._emit_contradicts = emit_contradicts
        self._llm_fallback = llm_fallback
        self.logger.info(
            "CrossEncoderOntologyService: nli_url=%s timeout=%ss min_conf=%.2f emit_contradicts=%s llm_fallback=%s",
            self._nli_url,
            self._nli_timeout,
            self._nli_min_confidence,
            self._emit_contradicts,
            self._llm_fallback,
        )

    async def classify_relationship(
        self,
        content_a: str,
        content_b: str,
        tenant_id: str = "_default",
        workspace_id: str | None = None,
    ) -> str:
        """Single-pair classification routed through the batch path."""
        result = await self.classify_relationships_batch(
            content_a=content_a,
            candidates=[("_", content_b)],
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        return result.get("_", _NEUTRAL_REL)

    async def classify_relationships_batch(
        self,
        content_a: str,
        candidates: list[tuple[str, str]],
        tenant_id: str = "_default",
        workspace_id: str | None = None,
    ) -> dict[str, str]:
        """Score each ``(content_a, candidate)`` pair with the NLI cross-encoder.

        One HTTP call for the whole batch. Maps NLI labels to coarse ontology
        relationships, honoring the min-confidence gate and the contradicts
        flag. On failure degrades to ``related_to`` (or the inherited LLM path
        when ``llm_fallback`` is set).
        """
        if not candidates:
            return {}

        pairs = [{"premise": content_a, "hypothesis": cand_content} for _, cand_content in candidates]
        try:
            nli_results = await asyncio.to_thread(self._call_nli_service, pairs)
        except Exception as exc:  # noqa: BLE001 - never block ingest
            self.logger.warning("NLI cross-encoder call failed: %s", exc)
            if self._llm_fallback:
                self.logger.info("Falling back to LLM batch classification")
                return await super().classify_relationships_batch(
                    content_a=content_a, candidates=candidates, tenant_id=tenant_id, workspace_id=workspace_id
                )
            return {cand_id: _NEUTRAL_REL for cand_id, _ in candidates}

        results: dict[str, str] = {}
        for (cand_id, _), nli in zip(candidates, nli_results):
            results[cand_id] = self._map_label(nli)
        # Any missing tail (short/misaligned response) defaults to related_to.
        for cand_id, _ in candidates:
            results.setdefault(cand_id, _NEUTRAL_REL)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_label(self, nli: dict) -> str:
        """Map one ``{"label", "score"}`` NLI result to a relationship type."""
        label = (nli.get("label") or "").strip().lower()
        try:
            score = float(nli.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if score < self._nli_min_confidence:
            return _NEUTRAL_REL
        if label == "entailment":
            return _ENTAILMENT_REL
        if label == "contradiction":
            return "contradicts" if self._emit_contradicts else _NEUTRAL_REL
        return _NEUTRAL_REL

    def _call_nli_service(self, pairs: list[dict]) -> list[dict]:
        """POST the pair batch to the NLI endpoint; return the results list.

        Sync (runs under ``asyncio.to_thread``). Raises on any network / HTTP /
        parse error so the async caller can apply the fallback policy.
        """
        payload = json.dumps({"pairs": pairs}).encode()
        endpoint = f"{self._nli_url}/v1/nli"
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._nli_timeout) as resp:
            body = json.loads(resp.read())
        results = body.get("results", [])
        if not isinstance(results, list):
            raise ValueError(f"NLI response 'results' is not a list: {type(results).__name__}")
        return results


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

class CrossEncoderOntologyServicePlugin(OntologyServicePluginBase):
    """Registers the cross-encoder ontology provider (``cross_encoder``)."""

    PROVIDER_NAME = "cross_encoder"

    def get_dependencies(self, v: Variables):
        return ()  # LLM optional (only used for the opt-in fallback)

    def initialize(self, v: Variables, logger: logging.Logger) -> CrossEncoderOntologyService:
        from memorylayer_server.config import (
            DEFAULT_MEMORYLAYER_EMBED_SERVER_URL,
            MEMORYLAYER_EMBED_SERVER_URL,
        )

        nli_url = v.environ(MEMORYLAYER_NLI_URL, default=DEFAULT_MEMORYLAYER_NLI_URL)
        if not nli_url:
            nli_url = v.environ(MEMORYLAYER_EMBED_SERVER_URL, default=DEFAULT_MEMORYLAYER_EMBED_SERVER_URL)
        nli_timeout = float(v.environ(MEMORYLAYER_NLI_TIMEOUT, default=DEFAULT_MEMORYLAYER_NLI_TIMEOUT))
        nli_min_confidence = float(v.environ(MEMORYLAYER_NLI_MIN_CONFIDENCE, default=DEFAULT_MEMORYLAYER_NLI_MIN_CONFIDENCE))
        emit_contradicts = ext_parse_bool(v.environ(MEMORYLAYER_CROSS_ENCODER_EMIT_CONTRADICTS, default="0"))
        llm_fallback = ext_parse_bool(v.environ(MEMORYLAYER_CROSS_ENCODER_LLM_FALLBACK, default="0"))

        # LLM only needed for the opt-in fallback; never hard-require it.
        llm_service = None
        try:
            llm_service = get_extension(EXT_LLM_SERVICE, v)
        except Exception:  # noqa: BLE001
            logger.debug("LLM service unavailable for cross-encoder fallback")

        return CrossEncoderOntologyService(
            nli_url=nli_url,
            nli_timeout=nli_timeout,
            nli_min_confidence=nli_min_confidence,
            emit_contradicts=emit_contradicts,
            llm_fallback=llm_fallback,
            v=v,
            llm_service=llm_service,
        )
