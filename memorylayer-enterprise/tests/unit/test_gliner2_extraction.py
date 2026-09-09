# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the GLiNER2 typed-NER extraction provider.

These exercise the type-carrying contract WITHOUT a live GLiNER2 HTTP service:
the NER call is monkeypatched to return a canned ``{label: [spans]}`` map (and
to raise, for the fail-safe path). What is under test:

  * NER labels are DERIVED from the OSS ``EntityType`` taxonomy (single source of
    truth) and exclude ``concept`` (the cleanliness win).
  * ``extract_entities`` preserves types: ``entity_types`` maps each name onto an
    ``EntityType`` value via the reverse label map; ``entities`` stays a flat,
    back-compat list; the regex speaker is included and typed PERSON.
  * On NER failure it falls back to the regex extractor (untyped, entity_types={}).
"""

import pytest

from memorylayer_server.models.entity_registry import EntityType

from memorylayer_saas.services.extraction.gliner2 import (
    DEFAULT_NER_LABELS,
    ENTITY_TYPE_TO_LABEL,
    LABEL_TO_ENTITY_TYPE,
    GLiNER2ExtractionService,
)


def _make_service(labels=None):
    return GLiNER2ExtractionService(
        ner_url="http://localhost:61055",
        ner_labels=labels if labels is not None else list(DEFAULT_NER_LABELS),
        ner_timeout=1.0,
        llm_service=None,
        storage=None,
        deduplication_service=None,
        embedding_service=None,
    )


def test_labels_derived_from_entity_type_taxonomy():
    """The default label list comes from EntityType and excludes 'concept'."""
    # Every label corresponds to an EntityType (single source of truth).
    for etype, label in ENTITY_TYPE_TO_LABEL.items():
        assert isinstance(etype, EntityType)
        assert LABEL_TO_ENTITY_TYPE[label] == etype.value
    # CONCEPT is intentionally excluded as a NER label (cleanliness win).
    assert EntityType.CONCEPT not in ENTITY_TYPE_TO_LABEL
    assert "concept" not in DEFAULT_NER_LABELS
    # The expected concrete labels are present.
    assert set(DEFAULT_NER_LABELS) == {"person", "organization", "project", "location", "event"}


def test_derived_labels_are_sent_to_ner_service(monkeypatch):
    """The EntityType-derived labels are passed to the NER service call."""
    svc = _make_service()
    captured = {}

    def _fake_call(text, labels):
        captured["labels"] = labels
        return {}

    monkeypatch.setattr(svc, "_call_ner_service", _fake_call)
    svc.extract_entities("[2026-06-01 10:00] Alice: hello")
    assert captured["labels"] == list(DEFAULT_NER_LABELS)


def test_typed_response_populates_entity_types(monkeypatch):
    """A typed {label: [spans]} response -> entity_types maps names to EntityType."""
    svc = _make_service()

    def _fake_call(text, labels):
        return {
            "person": ["Bob"],
            "organization": ["Acme"],
            "location": ["Paris"],
        }

    monkeypatch.setattr(svc, "_call_ner_service", _fake_call)
    result = svc.extract_entities("[2026-06-01 10:00] Alice: Bob at Acme in Paris")

    # Speaker (regex) is typed PERSON.
    assert result["speaker"] == "Alice"
    assert result["entity_types"]["Alice"] == EntityType.PERSON.value
    # Typed NER spans carry their mapped EntityType values.
    assert result["entity_types"]["Bob"] == EntityType.PERSON.value
    assert result["entity_types"]["Acme"] == EntityType.ORG.value
    assert result["entity_types"]["Paris"] == EntityType.PLACE.value
    # entities stays a flat back-compat list containing every name + speaker.
    assert set(result["entities"]) == {"Alice", "Bob", "Acme", "Paris"}
    # All stored type values are valid EntityType values.
    for value in result["entity_types"].values():
        assert EntityType(value)


def test_empty_ner_response_only_speaker(monkeypatch):
    """No NER spans -> only the speaker (typed PERSON) is returned."""
    svc = _make_service()
    monkeypatch.setattr(svc, "_call_ner_service", lambda text, labels: {})
    result = svc.extract_entities("[2026-06-01 10:00] Alice: nothing typed here")
    assert result["entities"] == ["Alice"]
    assert result["entity_types"] == {"Alice": EntityType.PERSON.value}


def test_ner_failure_falls_back_to_regex_untyped(monkeypatch):
    """On NER failure -> regex fallback, untyped (entity_types == {})."""
    svc = _make_service()

    def _boom(text, labels):
        raise RuntimeError("NER service down")

    monkeypatch.setattr(svc, "_call_ner_service", _boom)
    result = svc.extract_entities("[2026-06-01 10:00] Alice: I shipped Orion")
    assert result["speaker"] == "Alice"
    assert "Orion" in result["entities"]
    # Regex fallback cannot type -> empty entity_types (no junk typing).
    assert result["entity_types"] == {}


def test_empty_content():
    svc = _make_service()
    assert svc.extract_entities("") == {"speaker": None, "entities": [], "entity_types": {}}
