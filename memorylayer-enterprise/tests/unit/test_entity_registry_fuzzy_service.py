# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the enterprise embedding-fuzzy EntityRegistryService.

These tests exercise the ``PostgreSQLEntityRegistryService.resolve`` tier
control-flow (exact -> alias -> embedding-fuzzy -> create) WITHOUT a live
PostgreSQL: a lightweight in-memory fake storage implements only the entity
methods ``resolve`` touches, and the ANN method does an in-process cosine scan
over stored ``name_embedding`` vectors. The embeddings come from the DETERMINISTIC
``hash`` provider (feature-hashed bag-of-words), so cosine similarities are
fixed and the tier decisions are reproducible offline:

  cos('caroline chen', 'chen caroline')   == 1.0000  -> HIGH  (>= 0.92) MATCH
  cos('caroline chen', 'caroline a chen')  == 0.8165 -> AMBIG ([0.80, 0.92)) record-only
  cos('caroline chen', 'microsoft')        == 0.0000 -> create (no candidate)

The exact + alias tiers (inherited from the OSS default) still take precedence
over the fuzzy tier, and an unavailable embedding service degrades to
exact+alias+create.
"""
import pytest

from memorylayer_server.models.entity_registry import EntityType
from memorylayer_server.services.embedding.hash import HashEmbeddingProvider
from memorylayer_server.services.entity_registry._normalize import normalize_entity_name
from memorylayer_server.utils import generate_id, utc_now_iso

from memorylayer_saas.services.entity_registry.postgresql import (
    PostgreSQLEntityRegistryService,
)


# ---------------------------------------------------------------------------
# Deterministic hash embedding service (offline, no embed server)
# ---------------------------------------------------------------------------

class _HashService:
    """Minimal embedding-service shim wrapping the deterministic hash provider."""

    def __init__(self, dimensions: int = 384):
        self._provider = HashEmbeddingProvider(v=None, dimensions=dimensions)

    async def embed(self, text: str) -> list[float]:
        return await self._provider.embed(text)


def _cosine(a: list[float], b: list[float]) -> float:
    # Hash provider returns L2-normalized vectors, so dot product == cosine.
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# In-memory fake storage implementing only the entity methods resolve() uses
# ---------------------------------------------------------------------------

class _FakeStorage:
    """In-memory entity store with an in-process cosine ANN.

    Implements exactly the StorageBackend entity surface that
    ``PostgreSQLEntityRegistryService.resolve`` calls: exact-name lookup, alias
    lookup, name-embedding ANN, store, and add-alias. Enough to drive the tier
    control-flow deterministically without PostgreSQL/pgvector.
    """

    def __init__(self):
        self.entities: dict[str, dict] = {}
        # (workspace_id, entity_id) -> list of normalized aliases
        self.aliases: dict[tuple[str, str], list[tuple[str, str]]] = {}

    async def find_entity_by_normalized_name(self, workspace_id, entity_type, normalized_name):
        for e in self.entities.values():
            if (
                e["workspace_id"] == workspace_id
                and e["entity_type"] == entity_type
                and e["normalized_name"] == normalized_name
                and e["status"] == "active"
            ):
                return self._with_aliases(e)
        return None

    async def find_entities_by_normalized_alias(self, workspace_id, normalized_alias, entity_type=None):
        out = []
        for e in self.entities.values():
            if e["workspace_id"] != workspace_id or e["status"] != "active":
                continue
            if entity_type is not None and e["entity_type"] != entity_type:
                continue
            norms = [na for (_a, na) in self.aliases.get((workspace_id, e["id"]), [])]
            if normalized_alias in norms:
                out.append(self._with_aliases(e))
        out.sort(key=lambda d: d["id"])
        return out

    async def find_entities_by_name_embedding(
        self, workspace_id, entity_type, embedding, *, limit=5, min_score=0.0
    ):
        scored = []
        for e in self.entities.values():
            if (
                e["workspace_id"] != workspace_id
                or e["entity_type"] != entity_type
                or e["status"] != "active"
                or e.get("name_embedding") is None
            ):
                continue
            score = _cosine(embedding, e["name_embedding"])
            if score < min_score:
                continue
            scored.append((score, e))
        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for score, e in scored[:limit]:
            d = self._with_aliases(e)
            d["score"] = score
            out.append(d)
        return out

    async def store_entity(self, entity: dict) -> dict:
        entity_id = entity.get("id") or generate_id("ent")
        row = {
            "id": entity_id,
            "workspace_id": entity["workspace_id"],
            "entity_type": entity["entity_type"],
            "canonical_name": entity["canonical_name"],
            "normalized_name": entity["normalized_name"],
            "confidence": entity.get("confidence", 1.0),
            "provenance": dict(entity.get("provenance") or {}),
            "representative_memory_id": entity.get("representative_memory_id"),
            "status": entity.get("status", "active"),
            "merged_into": entity.get("merged_into"),
            "name_embedding": entity.get("name_embedding"),
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        self.entities[entity_id] = row
        self.aliases.setdefault((row["workspace_id"], entity_id), [])
        for alias in entity.get("aliases") or []:
            await self.add_entity_alias(
                row["workspace_id"], entity_id, alias, normalize_entity_name(alias)
            )
        return self._with_aliases(row)

    async def add_entity_alias(self, workspace_id, entity_id, alias, normalized_alias, source="manual"):
        key = (workspace_id, entity_id)
        existing = self.aliases.setdefault(key, [])
        if normalized_alias not in [na for (_a, na) in existing]:
            existing.append((alias, normalized_alias))

    def _with_aliases(self, e: dict) -> dict:
        d = dict(e)
        d["aliases"] = [a for (a, _na) in self.aliases.get((e["workspace_id"], e["id"]), [])]
        return d


WS = "ws-fuzzy"


def _service(embedding_service=_HashService(), **kw):
    return PostgreSQLEntityRegistryService(
        storage=_FakeStorage(), v=None, embedding_service=embedding_service, **kw
    )


@pytest.mark.asyncio
class TestEmbeddingFuzzyTier:
    async def test_high_similarity_matches_existing_and_adds_alias(self):
        svc = _service()
        # Seed an existing entity (also stores its name_embedding).
        first = await svc.resolve(WS, "Caroline Chen", EntityType.PERSON)
        assert first.matched_via == "created"

        # 'Chen Caroline' normalizes to a DIFFERENT key (token order preserved in
        # normalization) so exact+alias miss; bag-of-words cosine == 1.0 -> MATCH.
        second = await svc.resolve(WS, "Chen Caroline", EntityType.PERSON)
        assert second.matched_via == "embedding"
        assert second.entity.id == first.entity.id
        assert second.score >= 0.92

        # The new surface form was folded in as an alias (self-organizing): the
        # NEXT occurrence resolves via the cheap deterministic alias tier.
        third = await svc.resolve(WS, "Chen Caroline", EntityType.PERSON)
        assert third.matched_via == "alias"
        assert third.entity.id == first.entity.id

        # Still exactly one entity total.
        assert len(svc._storage.entities) == 1

    async def test_unrelated_name_creates_new(self):
        svc = _service()
        await svc.resolve(WS, "Caroline Chen", EntityType.PERSON)
        res = await svc.resolve(WS, "Microsoft", EntityType.PERSON)
        assert res.matched_via == "created"
        # No fuzzy candidate recorded (cosine 0.0 < ambig band).
        assert "fuzzy_candidates" not in res.entity.provenance
        assert len(svc._storage.entities) == 2

    async def test_ambiguous_creates_new_and_records_candidate_no_merge(self):
        svc = _service()
        first = await svc.resolve(WS, "Caroline Chen", EntityType.PERSON)
        # cos('caroline chen','caroline a chen') == 0.8165 -> AMBIG band.
        res = await svc.resolve(WS, "Caroline A Chen", EntityType.PERSON)
        # Conservative: a weak signal must NEVER silently merge distinct entities.
        assert res.matched_via == "created"
        assert res.entity.id != first.entity.id
        # The near-miss is recorded for future lazy-LLM adjudication.
        cands = res.entity.provenance.get("fuzzy_candidates")
        assert cands and cands[0]["entity_id"] == first.entity.id
        assert 0.80 <= cands[0]["score"] < 0.92
        # Two distinct entities now exist (no auto-merge).
        assert len(svc._storage.entities) == 2

    async def test_exact_tier_takes_precedence_over_fuzzy(self):
        svc = _service()
        first = await svc.resolve(WS, "Caroline Chen", EntityType.PERSON)
        # Same normalized key -> exact, never reaches the fuzzy tier.
        again = await svc.resolve(WS, "caroline chen", EntityType.PERSON)
        assert again.matched_via == "exact"
        assert again.entity.id == first.entity.id

    async def test_alias_tier_takes_precedence_over_fuzzy(self):
        svc = _service()
        created = await svc.upsert(WS, "Robert", EntityType.PERSON, aliases=["Bob"])
        res = await svc.resolve(WS, "Bob", EntityType.PERSON)
        assert res.matched_via == "alias"
        assert res.entity.id == created.id

    async def test_embedding_unavailable_falls_back(self):
        # No embedding service -> fuzzy tier disabled; exact+alias+create only.
        svc = _service(embedding_service=None)
        first = await svc.resolve(WS, "Caroline Chen", EntityType.PERSON)
        assert first.matched_via == "created"
        # A high-sim surface form would MATCH if fuzzy were active; here it must
        # CREATE a second entity (graceful degradation).
        second = await svc.resolve(WS, "Chen Caroline", EntityType.PERSON)
        assert second.matched_via == "created"
        assert second.entity.id != first.entity.id
        assert len(svc._storage.entities) == 2

    async def test_embedding_failure_is_failsafe(self):
        class _BoomService:
            async def embed(self, text):
                raise RuntimeError("embed backend down")

        svc = _service(embedding_service=_BoomService())
        first = await svc.resolve(WS, "Caroline Chen", EntityType.PERSON)
        assert first.matched_via == "created"
        # Embed raises -> tier skipped -> create (no crash, no match).
        second = await svc.resolve(WS, "Chen Caroline", EntityType.PERSON)
        assert second.matched_via == "created"
        assert len(svc._storage.entities) == 2

    async def test_allow_create_false_after_fuzzy_miss_raises(self):
        svc = _service()
        await svc.resolve(WS, "Caroline Chen", EntityType.PERSON)
        with pytest.raises(LookupError):
            await svc.resolve(WS, "Microsoft", EntityType.PERSON, allow_create=False)
