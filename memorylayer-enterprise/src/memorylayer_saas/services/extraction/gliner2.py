"""GLiNER2-backed NER extraction provider for MemoryLayer enterprise.

Replaces the OSS regex ``extract_entities`` with real typed NER by calling the
integrated ``/v1/ner`` endpoint on the main embed server (the same host used for
embeddings, :61051 by default), while keeping the reliable regex SPEAKER_RE for
dialogue speaker extraction (LoCoMo format).

The endpoint moved from the standalone experiment on :61055 into the embed-server
package, so by default this provider targets ``MEMORYLAYER_EMBED_SERVER_URL``.

Selection:
    MEMORYLAYER_EXTRACTION_SERVICE=gliner2

Config (env):
    MEMORYLAYER_GLINER2_NER_URL      - NER service base URL. Optional override;
                                       when unset it defaults to the main embed
                                       server URL (MEMORYLAYER_EMBED_SERVER_URL,
                                       default http://localhost:61051) where the
                                       integrated /v1/ner endpoint now lives.
    MEMORYLAYER_GLINER2_NER_LABELS   - CSV entity labels. Optional override; when
                                       unset the labels are DERIVED from the OSS
                                       ``EntityType`` taxonomy (see
                                       ``ENTITY_TYPE_TO_LABEL``) so types and NER
                                       labels stay a single source of truth.
    MEMORYLAYER_GLINER2_NER_TIMEOUT  - HTTP timeout seconds (default: 10)

Fail-safe:
    On any HTTP or parse error the provider falls back to
    ``super().extract_entities()`` (regex, untyped), so ingest is never blocked.
"""

import logging
import re
import urllib.error
import urllib.parse
import urllib.request

from memorylayer_server.models.entity_registry import EntityType
from memorylayer_server.services.extraction.base import ExtractionServicePluginBase
from memorylayer_server.services.extraction.default import DefaultExtractionService
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND, StorageBackend
from memorylayer_server.services.embedding import EXT_EMBEDDING_SERVICE, EmbeddingService
from memorylayer_server.services.deduplication import EXT_DEDUPLICATION_SERVICE, DeduplicationService
from memorylayer_server.services.llm import EXT_LLM_SERVICE, LLMService
from scitrera_app_framework import Variables, get_logger

# ---------------------------------------------------------------------------
# EntityType <-> GLiNER2 label mapping (single source of truth)
#
# The NER label list is DERIVED from the OSS ``EntityType`` taxonomy so the types
# we accrete into the entity registry and the labels we ask GLiNER2 to extract
# never drift apart. ``CONCEPT`` is intentionally EXCLUDED as a label: it is the
# catch-all bucket for untyped spans, and asking the NER model for "concept"
# floods the result with common words ("Good", "Yep"). Excluding it means
# GLiNER2 only emits concrete typed entities (the cleanliness win); anything not
# typed simply isn't extracted. ``entity_types`` stores raw ``EntityType``
# *values* (strings) so OSS accretion can ``EntityType(value)`` them directly.
# ---------------------------------------------------------------------------

ENTITY_TYPE_TO_LABEL: dict[EntityType, str] = {
    EntityType.PERSON: "person",
    EntityType.ORG: "organization",
    EntityType.PROJECT: "project",
    EntityType.PLACE: "location",
    EntityType.EVENT: "event",
}
# Reverse map: GLiNER2 label -> EntityType value (stored on metadata).
LABEL_TO_ENTITY_TYPE: dict[str, str] = {label: etype.value for etype, label in ENTITY_TYPE_TO_LABEL.items()}
# Default NER labels derived from the taxonomy (single source of truth).
DEFAULT_NER_LABELS: list[str] = list(ENTITY_TYPE_TO_LABEL.values())

# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------

MEMORYLAYER_GLINER2_NER_URL = "MEMORYLAYER_GLINER2_NER_URL"
# Empty default: when unset the URL is resolved from MEMORYLAYER_EMBED_SERVER_URL
# (the main embed server, default http://localhost:61051) where the integrated
# /v1/ner endpoint now lives — replacing the old standalone :61055 service.
DEFAULT_MEMORYLAYER_GLINER2_NER_URL = ""

# Empty default: when unset the EntityType-derived DEFAULT_NER_LABELS are used.
MEMORYLAYER_GLINER2_NER_LABELS = "MEMORYLAYER_GLINER2_NER_LABELS"
DEFAULT_MEMORYLAYER_GLINER2_NER_LABELS = ""

MEMORYLAYER_GLINER2_NER_TIMEOUT = "MEMORYLAYER_GLINER2_NER_TIMEOUT"
DEFAULT_MEMORYLAYER_GLINER2_NER_TIMEOUT = "10"

# Speaker regex — structural and perfectly reliable for [ts] Name: dialogue.
_SPEAKER_RE = re.compile(r"^\[[^\]]*\]\s*([A-Z][a-zA-Z]+):")


class GLiNER2ExtractionService(DefaultExtractionService):
    """Extraction service that uses GLiNER2 NER for entity extraction.

    All session extraction / LLM-backed methods are inherited from
    ``DefaultExtractionService``.  Only ``extract_entities`` is overridden to
    call the GLiNER2 NER HTTP service and return typed, high-precision entities
    instead of the regex proper-noun scan.
    """

    def __init__(
        self,
        *,
        ner_url: str,
        ner_labels: list[str],
        ner_timeout: float,
        label_to_type: dict[str, str] | None = None,
        llm_service: LLMService | None = None,
        storage: StorageBackend | None = None,
        deduplication_service=None,
        embedding_service: EmbeddingService | None = None,
        v: Variables = None,
    ):
        super().__init__(
            llm_service=llm_service,
            storage=storage,
            deduplication_service=deduplication_service,
            embedding_service=embedding_service,
            v=v,
        )
        self._ner_url = ner_url.rstrip("/")
        self._ner_labels = ner_labels
        self._ner_timeout = ner_timeout
        # NER-label -> EntityType-value map. Sourced from the ontology's entity-type
        # vocabulary (so new domain types flow through automatically); falls back to
        # the core enum-derived map when the ontology is unavailable.
        self._label_to_type = label_to_type or dict(LABEL_TO_ENTITY_TYPE)
        self.logger.info(
            "GLiNER2ExtractionService: ner_url=%s labels=%s timeout=%ss",
            self._ner_url,
            self._ner_labels,
            self._ner_timeout,
        )

    # ------------------------------------------------------------------
    # Public API (overrides base)
    # ------------------------------------------------------------------

    def extract_entities(self, content: str) -> dict:
        """Extract speaker (regex) + typed entities (GLiNER2 NER).

        Returns ``{"speaker": str | None, "entities": list[str], "entity_types":
        dict[str, str]}`` where ``entity_types`` maps each typed entity name to an
        ``EntityType`` value (via ``LABEL_TO_ENTITY_TYPE``) so OSS registry
        accretion assigns proper types. ``entities`` stays a flat list for the
        back-compat entity-anchor consumers. The speaker is always PERSON-typed.

        On NER service failure falls back to the regex extractor (untyped:
        ``entity_types={}``) so ingest is never blocked.
        """
        if not content:
            return {"speaker": None, "entities": [], "entity_types": {}}

        # Speaker is structural — always use regex.
        m = _SPEAKER_RE.match(content)
        speaker = m.group(1) if m else None

        # Entity extraction via NER service — preserves the typed {label: [spans]}.
        try:
            entities_map = self._call_ner_service(content, self._ner_labels)
        except Exception as exc:
            self.logger.warning(
                "GLiNER2 NER call failed, falling back to regex: %s", exc
            )
            return super().extract_entities(content)

        # Map each label bucket to an EntityType value, building both the flat
        # entity list (back-compat) and the per-name entity_types map.
        ents: list[str] = []
        entity_types: dict[str, str] = {}
        for label, spans in entities_map.items():
            etype_value = self._label_to_type.get(label)
            for span in spans:
                if not span:
                    continue
                if span not in entity_types:
                    ents.append(span)
                    if etype_value is not None:
                        entity_types[span] = etype_value

        # Always include the speaker in the entity set, typed PERSON.
        if speaker:
            if speaker not in entity_types:
                entity_types[speaker] = EntityType.PERSON.value
            if speaker not in ents:
                ents.append(speaker)

        return {
            "speaker": speaker,
            "entities": sorted(set(ents)),
            "entity_types": entity_types,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_ner_service(self, text: str, labels: list[str]) -> dict[str, list[str]]:
        """POST to the NER service and return the typed ``{label: [spans]}`` map.

        Uses stdlib ``urllib`` (sync) — this method is called from the sync
        ``extract_entities`` path so no event loop is needed. Returns the typed
        entity map verbatim (the caller maps labels onto ``EntityType`` values);
        previously this flattened-and-discarded the types.

        Raises on any network / HTTP / parse error (caller handles fallback).
        """
        import json as _json

        payload = _json.dumps({"texts": [text], "labels": labels}).encode()
        endpoint = f"{self._ner_url}/v1/ner"

        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._ner_timeout) as resp:
            body = _json.loads(resp.read())

        # body: {"results": [{"entities": {label: [spans]}}]}
        results = body.get("results", [])
        if not results:
            return {}

        entities_map: dict[str, list[str]] = results[0].get("entities", {})
        return entities_map


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

class GLiNER2ExtractionServicePlugin(ExtractionServicePluginBase):
    """Plugin that registers the GLiNER2 extraction provider."""

    PROVIDER_NAME = "gliner2"

    def initialize(self, v: Variables, logger: logging.Logger) -> GLiNER2ExtractionService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        llm_service: LLMService = self.get_extension(EXT_LLM_SERVICE, v)
        deduplication_service: DeduplicationService = self.get_extension(EXT_DEDUPLICATION_SERVICE, v)
        embedding_service: EmbeddingService = self.get_extension(EXT_EMBEDDING_SERVICE, v)

        # NER URL: explicit override wins; otherwise default to the main embed
        # server (where the integrated /v1/ner endpoint now lives).
        from memorylayer_server.config import (
            DEFAULT_MEMORYLAYER_EMBED_SERVER_URL,
            MEMORYLAYER_EMBED_SERVER_URL,
        )
        ner_url: str = v.environ(MEMORYLAYER_GLINER2_NER_URL, default=DEFAULT_MEMORYLAYER_GLINER2_NER_URL)
        if not ner_url:
            ner_url = v.environ(MEMORYLAYER_EMBED_SERVER_URL, default=DEFAULT_MEMORYLAYER_EMBED_SERVER_URL)

        # Entity-type vocabulary is owned by the OntologyService (the canonical
        # authority for relationship types, memory subtypes, AND entity types). The
        # extractor derives its NER label set + reverse (label -> type) map from it,
        # so a deployment can add domain types (MEMORYLAYER_ENTITY_TYPES / an
        # OntologyContributor) without touching extractor code. Fail-safe: if the
        # ontology can't be resolved, fall back to the core EntityType-derived map.
        from memorylayer_server.config import DEFAULT_TENANT_ID, MEMORYLAYER_TENANT_ID
        from memorylayer_server.services.ontology import EXT_ONTOLOGY_SERVICE
        from scitrera_app_framework import get_extension

        onto_labels: list[str] = list(DEFAULT_NER_LABELS)
        label_to_type: dict[str, str] = dict(LABEL_TO_ENTITY_TYPE)
        try:
            ontology = get_extension(EXT_ONTOLOGY_SERVICE, v)
        except Exception:  # noqa: BLE001 - ontology optional; degrade to core defaults
            ontology = None
            logger.debug("OntologyService unavailable for GLiNER2; using core entity-type defaults")
        if ontology is not None:
            tenant_id = v.environ(MEMORYLAYER_TENANT_ID, default=DEFAULT_TENANT_ID)
            try:
                onto_labels = ontology.get_ner_labels(tenant_id) or onto_labels
                label_to_type = ontology.ner_label_to_entity_type(tenant_id) or label_to_type
            except Exception:  # noqa: BLE001 - never block extraction bring-up
                logger.warning("Reading entity-type vocabulary from ontology failed; using core defaults")

        # Explicit CSV env override still wins for ad-hoc tuning; else ontology labels.
        ner_labels_csv: str = v.environ(MEMORYLAYER_GLINER2_NER_LABELS, default=DEFAULT_MEMORYLAYER_GLINER2_NER_LABELS)
        env_labels: list[str] = [lbl.strip() for lbl in ner_labels_csv.split(",") if lbl.strip()]
        ner_labels: list[str] = env_labels or onto_labels or list(DEFAULT_NER_LABELS)
        ner_timeout: float = float(v.environ(MEMORYLAYER_GLINER2_NER_TIMEOUT, default=DEFAULT_MEMORYLAYER_GLINER2_NER_TIMEOUT))

        return GLiNER2ExtractionService(
            ner_url=ner_url,
            ner_labels=ner_labels,
            ner_timeout=ner_timeout,
            label_to_type=label_to_type,
            llm_service=llm_service,
            storage=storage,
            deduplication_service=deduplication_service,
            embedding_service=embedding_service,
            v=v,
        )
