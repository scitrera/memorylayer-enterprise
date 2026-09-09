"""Unit tests for the cross-encoder (NLI) ontology relationship classifier.

These exercise the NLI-label -> relationship mapping WITHOUT a live NLI service:
``_call_nli_service`` is monkeypatched to return canned ``{label, score}`` results
(or to raise, for the fail-safe path). Under test:

  * entailment -> supports, neutral -> related_to.
  * contradiction is deferred to ContradictionService by default (-> related_to)
    and only becomes a ``contradicts`` edge when emit_contradicts is set.
  * a below-min-confidence non-neutral label degrades to related_to.
  * the whole candidate batch is scored in ONE service call.
  * on service failure the batch degrades to related_to (no LLM) by default.
"""

import pytest

from memorylayer_saas.services.ontology.cross_encoder import CrossEncoderOntologyService


def _make_service(*, emit_contradicts=False, llm_fallback=False, min_conf=0.60):
    return CrossEncoderOntologyService(
        nli_url="http://localhost:61051",
        nli_timeout=1.0,
        nli_min_confidence=min_conf,
        emit_contradicts=emit_contradicts,
        llm_fallback=llm_fallback,
        v=None,
        llm_service=None,
    )


@pytest.mark.asyncio
async def test_label_mapping_single_call(monkeypatch):
    svc = _make_service()
    calls = {"n": 0}

    def fake_call(pairs):
        calls["n"] += 1
        # Order matches the input pairs.
        return [
            {"label": "entailment", "score": 0.95},
            {"label": "neutral", "score": 0.99},
            {"label": "contradiction", "score": 0.95},
        ]

    monkeypatch.setattr(svc, "_call_nli_service", fake_call)
    result = await svc.classify_relationships_batch(
        content_a="anchor",
        candidates=[("a", "ca"), ("b", "cb"), ("c", "cc")],
    )
    assert calls["n"] == 1, "the whole batch must be one NLI call"
    assert result["a"] == "supports"
    assert result["b"] == "related_to"
    # contradiction deferred to ContradictionService by default
    assert result["c"] == "related_to"


@pytest.mark.asyncio
async def test_contradiction_emitted_when_enabled(monkeypatch):
    svc = _make_service(emit_contradicts=True)
    monkeypatch.setattr(svc, "_call_nli_service", lambda pairs: [{"label": "contradiction", "score": 0.9}])
    result = await svc.classify_relationships_batch(content_a="anchor", candidates=[("c", "cc")])
    assert result["c"] == "contradicts"


@pytest.mark.asyncio
async def test_low_confidence_degrades_to_related_to(monkeypatch):
    svc = _make_service(min_conf=0.60)
    monkeypatch.setattr(svc, "_call_nli_service", lambda pairs: [{"label": "entailment", "score": 0.40}])
    result = await svc.classify_relationships_batch(content_a="anchor", candidates=[("a", "ca")])
    assert result["a"] == "related_to"


@pytest.mark.asyncio
async def test_service_failure_degrades_without_llm(monkeypatch):
    svc = _make_service(llm_fallback=False)

    def boom(pairs):
        raise RuntimeError("nli down")

    monkeypatch.setattr(svc, "_call_nli_service", boom)
    result = await svc.classify_relationships_batch(
        content_a="anchor",
        candidates=[("a", "ca"), ("b", "cb")],
    )
    assert result == {"a": "related_to", "b": "related_to"}


@pytest.mark.asyncio
async def test_empty_candidates_no_call(monkeypatch):
    svc = _make_service()

    def should_not_run(pairs):
        raise AssertionError("must not call NLI for empty candidates")

    monkeypatch.setattr(svc, "_call_nli_service", should_not_run)
    assert await svc.classify_relationships_batch(content_a="anchor", candidates=[]) == {}


@pytest.mark.asyncio
async def test_single_pair_routes_through_batch(monkeypatch):
    svc = _make_service()
    monkeypatch.setattr(svc, "_call_nli_service", lambda pairs: [{"label": "entailment", "score": 0.99}])
    rel = await svc.classify_relationship("anchor", "candidate")
    assert rel == "supports"
