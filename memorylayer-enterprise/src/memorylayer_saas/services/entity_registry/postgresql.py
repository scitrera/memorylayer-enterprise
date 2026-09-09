# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Enterprise embedding-fuzzy Entity Registry Service (entity-registry follow-on 1).

``PostgreSQLEntityRegistryService`` extends the OSS
``DefaultEntityRegistryService`` (normalize -> exact -> alias -> create) with a
single extra resolution tier inserted AFTER an exact+alias miss but BEFORE
create: a **semantic / embedding-fuzzy** match over existing canonical entities.
This lets surface forms like ``Caroline`` and ``Caroline Chen`` collapse to one
canonical entity even when neither the normalized key nor any alias matches.

Why enterprise-only: the tier needs (a) an embedding service and (b) pgvector
ANN over a per-entity ``name_embedding`` column (migration 023). OSS stays
exact+alias only (the parity reference), so the OSS ``resolve`` never calls the
ANN storage method.

Conservative gating (a weak signal must NEVER silently merge distinct entities):
  * ``score >= HIGH_THRESHOLD`` (default 0.92 cosine) -> MATCH the existing
    entity (``matched_via="embedding"``) AND fold the new surface form in as an
    alias, so the next occurrence hits the cheap deterministic alias tier
    (self-organizing).
  * ``AMBIG_BAND <= score < HIGH_THRESHOLD`` (default 0.80) -> do NOT merge.
    Create a new entity, but record the near-miss in
    ``provenance["fuzzy_candidates"]`` so a future lazy-LLM adjudication slice
    can resolve the ambiguity offline.
  * ``score < AMBIG_BAND`` -> create normally (no candidate worth recording).

Fail-safe: if no embedding service is available, or embedding/ANN raises, the
tier is skipped entirely and resolution degrades to the deterministic OSS path
(exact+alias+create). Behind ``MEMORYLAYER_ENTITY_REGISTRY_ENABLED`` (default
OFF) like the rest of the registry; selected only when
``MEMORYLAYER_ENTITY_REGISTRY_PROVIDER=postgresql``.
"""

import logging

from scitrera_app_framework import Variables, get_extension, get_logger

from memorylayer_server.models.entity_registry import EntityResolution, EntityType
from memorylayer_server.services._constants import (
    EXT_EMBEDDING_SERVICE,
    EXT_STORAGE_BACKEND,
    EXT_TASK_SERVICE,
)
from memorylayer_server.services.embedding import EmbeddingService
from memorylayer_server.services.entity_registry import EntityRegistryServicePluginBase
from memorylayer_server.services.entity_registry._normalize import normalize_entity_name
from memorylayer_server.services.entity_registry.default import (
    DefaultEntityRegistryService,
    _entity_from_dict,
)
from memorylayer_server.services.storage import StorageBackend
from memorylayer_server.utils import utc_now_iso

# Conservative cosine thresholds for the embedding-fuzzy tier.
#
# HIGH (auto-merge): 0.92 — deliberately high. We only auto-attach a new surface
#   form to an existing entity when the names are near-synonymous in embedding
#   space. False merges are far costlier than false splits (a split can be merged
#   later; an erroneous merge silently conflates two real-world entities).
# AMBIG (record-only band lower bound): 0.80 — strong-enough to be worth flagging
#   for later LLM adjudication, weak-enough that auto-merging would be reckless.
# Both overridable via env so operators can tune per embed model.
MEMORYLAYER_ENTITY_REGISTRY_FUZZY_HIGH_THRESHOLD = "MEMORYLAYER_ENTITY_REGISTRY_FUZZY_HIGH_THRESHOLD"
DEFAULT_FUZZY_HIGH_THRESHOLD = 0.92
MEMORYLAYER_ENTITY_REGISTRY_FUZZY_AMBIG_BAND = "MEMORYLAYER_ENTITY_REGISTRY_FUZZY_AMBIG_BAND"
DEFAULT_FUZZY_AMBIG_BAND = 0.80
MEMORYLAYER_ENTITY_REGISTRY_FUZZY_ANN_LIMIT = "MEMORYLAYER_ENTITY_REGISTRY_FUZZY_ANN_LIMIT"
DEFAULT_FUZZY_ANN_LIMIT = 5


class PostgreSQLEntityRegistryService(DefaultEntityRegistryService):
    """Entity registry with an embedding-fuzzy tier over pgvector (enterprise)."""

    PROVIDER_NAME = "postgresql"

    def __init__(
        self,
        storage: StorageBackend,
        v: Variables,
        embedding_service: EmbeddingService | None = None,
        *,
        high_threshold: float = DEFAULT_FUZZY_HIGH_THRESHOLD,
        ambig_band: float = DEFAULT_FUZZY_AMBIG_BAND,
        ann_limit: int = DEFAULT_FUZZY_ANN_LIMIT,
        task_service=None,
    ):
        super().__init__(storage=storage, v=v)
        # Re-tag the logger so operators can tell the enterprise tier apart.
        self.logger = get_logger(v, name="EntityRegistryService[pg-fuzzy]")
        self._embedding = embedding_service
        self._high_threshold = high_threshold
        self._ambig_band = ambig_band
        self._ann_limit = ann_limit
        # Optional: used to enqueue the lazy-LLM entity_adjudication task when the
        # fuzzy tier records ambiguous candidates. When absent the candidates are
        # still recorded (a future backfill can pick them up) — graceful.
        self._task_service = task_service
        if self._embedding is None:
            self.logger.warning(
                "No embedding service available — embedding-fuzzy resolution tier "
                "disabled; falling back to exact+alias+create."
            )

    async def _embed_name(self, normalized: str) -> list[float] | None:
        """Embed the normalized name; return None on any failure (fail-safe)."""
        if self._embedding is None:
            return None
        try:
            return await self._embedding.embed(normalized)
        except Exception:  # noqa: BLE001 - fuzzy tier must never break resolution
            self.logger.exception(
                "Embedding the entity name failed; skipping fuzzy tier for %r", normalized
            )
            return None

    async def resolve(
        self,
        workspace_id: str,
        name: str,
        entity_type: EntityType,
        *,
        source_memory_id: str | None = None,
        observer_id: str | None = None,
        allow_create: bool = True,
        promote: bool = False,
    ) -> EntityResolution:
        from memorylayer_server.services.entity_registry.default import _etype_str

        etype = _etype_str(entity_type)
        normalized = normalize_entity_name(name)

        # 0. Name-first resolution (accretion path only; default OFF). Reuses the
        #    OSS deterministic PERSON-promotion path BEFORE the embedding-fuzzy
        #    tier: it unifies a name seen as both speaker (PERSON) and mention
        #    (CONCEPT) onto one node. Only fires when an existing same-name entity
        #    is found; otherwise resolution falls through to the fuzzy/create path.
        if promote:
            promoted = await self._resolve_name_first(
                workspace_id, name, normalized, etype, source_memory_id, observer_id
            )
            if promoted is not None:
                return promoted

        # 1. Exact match on (workspace, type, normalized_name).
        exact = await self._storage.find_entity_by_normalized_name(workspace_id, etype, normalized)
        if exact is not None:
            return EntityResolution(entity=_entity_from_dict(exact), matched_via="exact", score=1.0)

        # 2. Alias match on the normalized alias (restricted to the same type).
        alias_hits = await self._storage.find_entities_by_normalized_alias(
            workspace_id, normalized, entity_type=etype
        )
        if alias_hits:
            if len(alias_hits) > 1:
                self.logger.debug(
                    "Ambiguous alias %r matched %d entities in workspace %s (type=%s) — "
                    "using lowest entity id; consider merging duplicates",
                    name, len(alias_hits), workspace_id, etype,
                )
            entity = _entity_from_dict(alias_hits[0])
            return EntityResolution(entity=entity, matched_via="alias", score=entity.confidence)

        # 3. Embedding-fuzzy tier (enterprise-only). Embed the normalized name and
        #    ANN over same-type, same-workspace ACTIVE entities. Fail-safe: any
        #    miss (no service, embed failure, no candidates) drops straight to the
        #    deterministic create path below.
        embedding = await self._embed_name(normalized)
        fuzzy_candidates: list[dict] = []
        if embedding is not None:
            try:
                candidates = await self._storage.find_entities_by_name_embedding(
                    workspace_id,
                    etype,
                    embedding,
                    limit=self._ann_limit,
                    min_score=self._ambig_band,
                )
            except NotImplementedError:
                # Storage backend has no ANN (should not happen for PG, but keep
                # the fuzzy tier strictly optional / fail-safe).
                candidates = []
            except Exception:  # noqa: BLE001 - fuzzy tier must never break resolution
                self.logger.exception(
                    "ANN over name embeddings failed; skipping fuzzy tier for %r", name
                )
                candidates = []

            for cand in candidates:
                score = float(cand.get("score", 0.0))
                if score >= self._high_threshold:
                    # High-confidence MATCH: attach the new surface form as an
                    # alias so the next occurrence resolves via the cheap alias
                    # tier (self-organizing), then return the existing entity.
                    matched = _entity_from_dict(cand)
                    await self._storage.add_entity_alias(
                        workspace_id,
                        matched.id,
                        name,
                        normalized,
                        source="embedding",
                    )
                    self.logger.debug(
                        "Embedding-fuzzy MATCH %r -> entity %s (score=%.4f >= %.2f); "
                        "added alias",
                        name, matched.id, score, self._high_threshold,
                    )
                    return EntityResolution(
                        entity=matched, matched_via="embedding", score=score
                    )
                # AMBIG band [ambig_band, high_threshold): record the near-miss but
                # DO NOT merge. Explicit lower bound so candidate recording is
                # self-documenting and decoupled from the storage min_score floor.
                elif score >= self._ambig_band:
                    fuzzy_candidates.append({"entity_id": cand["id"], "score": score})

            if fuzzy_candidates:
                self.logger.debug(
                    "Embedding-fuzzy AMBIGUOUS for %r: %d candidate(s) in [%0.2f, %0.2f) — "
                    "creating new entity + recording fuzzy_candidates (no auto-merge)",
                    name, len(fuzzy_candidates), self._ambig_band, self._high_threshold,
                )

        # 4. Create (only when allowed). Carries the name embedding (when
        #    available) and any recorded near-miss fuzzy candidates.
        if not allow_create:
            raise LookupError(
                f"No {etype} entity matching {name!r} in workspace {workspace_id} (allow_create=False)"
            )

        from memorylayer_server.services.entity_registry.provenance import build_provenance

        _now = utc_now_iso()
        provenance = build_provenance(
            "entity.create",
            generated_at=_now,
            matched_via="created",
            created_at=_now,  # back-compat key (existing readers)
            source_name=name,
            source_memory_id=source_memory_id,
            observer_id=observer_id,
            # Surfaced for the lazy-LLM adjudication slice.
            fuzzy_candidates=fuzzy_candidates or None,
        )

        entity_row: dict = {
            "workspace_id": workspace_id,
            "entity_type": etype,
            "canonical_name": name,
            "normalized_name": normalized,
            "confidence": 1.0,
            "provenance": provenance,
            "representative_memory_id": source_memory_id,
            "status": "active",
        }
        if embedding is not None:
            entity_row["name_embedding"] = embedding

        stored = await self._storage.store_entity(entity_row)
        # Race tolerance (mirrors DefaultEntityRegistryService.resolve): the
        # partial-unique-index collision is handled in store_entity, which
        # returns the winner's row.
        if (
            stored["normalized_name"] == normalized
            and stored.get("provenance", {}).get("matched_via") != "created"
        ):
            matched_via = "exact"
        else:
            matched_via = "created"

        # Lazy-LLM adjudication enqueue: when WE created a new entity carrying
        # ambiguous fuzzy candidates, schedule the offline entity_adjudication
        # task to decide (conservatively) whether to merge it into a candidate.
        # Skipped silently if we lost the create race (matched_via != "created")
        # or no task service is wired (the candidates stay recorded for a future
        # backfill — graceful, never auto-merges here).
        if matched_via == "created" and fuzzy_candidates and self._task_service is not None:
            await self._maybe_enqueue_adjudication(workspace_id, stored)

        return EntityResolution(entity=_entity_from_dict(stored), matched_via=matched_via, score=1.0)

    async def _maybe_enqueue_adjudication(self, workspace_id: str, stored: dict) -> None:
        """Schedule one entity_adjudication task for a freshly-created ambiguous
        entity, debounced so we never enqueue twice for the same entity.

        Debounce: a ``provenance["adjudication_enqueued"]`` flag is the guard —
        if it is already set we skip. Best-effort: any failure to enqueue leaves
        the recorded ``fuzzy_candidates`` intact (a future backfill can resolve
        them); it must NEVER break resolution and NEVER auto-merge.
        """
        entity_id = stored["id"]
        try:
            prov = stored.get("provenance") or {}
            if prov.get("adjudication_enqueued"):
                return  # already enqueued for this entity (debounce)
            await self._task_service.schedule_task(
                "entity_adjudication",
                {"workspace_id": workspace_id, "entity_id": entity_id},
            )
            # Persist the debounce flag so a later resolve() for the same row
            # (or a retry) does not double-enqueue.
            updated_prov = {**prov, "adjudication_enqueued": True}
            await self._storage.update_entity(workspace_id, entity_id, provenance=updated_prov)
        except Exception:  # noqa: BLE001 - enqueue is best-effort; never break resolve
            self.logger.exception(
                "Failed to enqueue entity_adjudication for %s; candidates remain "
                "recorded for a future backfill", entity_id,
            )

    async def dedupe_workspace(
        self, workspace_id: str, *, threshold: float | None = None, limit: int = 1000
    ) -> dict:
        """Batch-dedupe via the pgvector name-embedding ANN + union-find clustering.

        For each active entity, ANN its same-type neighbors at/above ``threshold``
        (default the resolve-time high threshold) to build similarity edges,
        union-find them into clusters, and merge each cluster (>1) into its most
        established representative. Best-effort: a missing embedding, an ANN error,
        or a merge failure never aborts the pass. Returns ``{clusters, merged}``.
        """
        from memorylayer_server.services.entity_registry.dedup import UnionFind, pick_representative
        from memorylayer_server.services.entity_registry.default import _etype_str

        threshold = self._high_threshold if threshold is None else float(threshold)
        entities = await self.list_entities(workspace_id, status="active", limit=limit)
        by_id = {e.id: e for e in entities}

        uf = UnionFind()
        for e in entities:
            uf.add(e.id)
            embedding = await self._embed_name(e.normalized_name)
            if embedding is None:
                continue
            try:
                candidates = await self._storage.find_entities_by_name_embedding(
                    workspace_id, _etype_str(e.entity_type), embedding,
                    limit=self._ann_limit, min_score=threshold,
                )
            except Exception:  # noqa: BLE001 - ANN best-effort
                candidates = []
            for cand in candidates:
                cid = cand.get("id")
                if cid and cid != e.id and cid in by_id and float(cand.get("score", 0.0)) >= threshold:
                    uf.union(e.id, cid)

        clusters = [ids for ids in uf.groups() if len(ids) > 1]
        merged = 0
        for ids in clusters:
            members = [by_id[i] for i in ids if i in by_id]
            counts: dict = {}
            for e in members:
                try:
                    counts[e.id] = len(await self.list_members(workspace_id, e.id, limit=1000))
                except Exception:  # noqa: BLE001
                    counts[e.id] = 0
            rep = pick_representative(members, counts)
            for e in members:
                if e.id == rep.id:
                    continue
                try:
                    await self.merge(
                        workspace_id, e.id, rep.id,
                        reason="batch-dedupe: embedding-fuzzy cluster",
                    )
                    merged += 1
                except Exception as ex:  # noqa: BLE001 - one merge failure never aborts the pass
                    self.logger.debug("Dedupe merge %s -> %s failed: %s", e.id, rep.id, ex)

        self.logger.info(
            "Entity dedupe for %s (threshold=%.2f): clusters=%d merged=%d",
            workspace_id, threshold, len(clusters), merged,
        )
        return {"clusters": len(clusters), "merged": merged}


class PostgreSQLEntityRegistryServicePlugin(EntityRegistryServicePluginBase):
    """Plugin for the enterprise embedding-fuzzy entity registry service.

    Auto-discovered by the enterprise ``register_package_plugins(services...,
    recursive=True)`` scan. Selected only when
    ``MEMORYLAYER_ENTITY_REGISTRY_PROVIDER=postgresql``. Depends on storage +
    the embedding service, but the embedding dep is best-effort: if it cannot be
    resolved the service still initializes and degrades to exact+alias+create.
    """

    PROVIDER_NAME = "postgresql"

    def get_dependencies(self, v: Variables):
        # Storage is required; embedding is soft (we degrade gracefully).
        return EXT_STORAGE_BACKEND, EXT_EMBEDDING_SERVICE

    def initialize(self, v: Variables, logger: logging.Logger) -> PostgreSQLEntityRegistryService:
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        try:
            embedding_service = get_extension(EXT_EMBEDDING_SERVICE, v)
        except Exception:  # noqa: BLE001 - embedding is optional; degrade gracefully
            embedding_service = None
            logger.warning(
                "Embedding service unavailable for entity registry; "
                "embedding-fuzzy tier disabled (exact+alias+create only)."
            )

        high_threshold = v.environ(
            MEMORYLAYER_ENTITY_REGISTRY_FUZZY_HIGH_THRESHOLD,
            default=DEFAULT_FUZZY_HIGH_THRESHOLD,
            type_fn=float,
        )
        ambig_band = v.environ(
            MEMORYLAYER_ENTITY_REGISTRY_FUZZY_AMBIG_BAND,
            default=DEFAULT_FUZZY_AMBIG_BAND,
            type_fn=float,
        )
        ann_limit = v.environ(
            MEMORYLAYER_ENTITY_REGISTRY_FUZZY_ANN_LIMIT,
            default=DEFAULT_FUZZY_ANN_LIMIT,
            type_fn=int,
        )

        # Task service is soft: used only to enqueue the lazy-LLM
        # entity_adjudication task for ambiguous fuzzy candidates. If it cannot
        # be resolved the registry still works (candidates stay recorded for a
        # future backfill) — never auto-merges, never breaks resolution.
        try:
            task_service = get_extension(EXT_TASK_SERVICE, v)
        except Exception:  # noqa: BLE001 - task service is optional; degrade gracefully
            task_service = None
            logger.warning(
                "Task service unavailable for entity registry; lazy-LLM "
                "adjudication enqueue disabled (fuzzy candidates still recorded)."
            )

        return PostgreSQLEntityRegistryService(
            storage=storage,
            v=v,
            embedding_service=embedding_service,
            high_threshold=high_threshold,
            ambig_band=ambig_band,
            ann_limit=ann_limit,
            task_service=task_service,
        )
