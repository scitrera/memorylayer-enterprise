"""Enterprise Knowledgebase Service — canonical-entity ("proper entity graph") articles.

``EnterpriseKnowledgebaseService`` extends the OSS ``DefaultKnowledgebaseService``
with ONE additive layer: alongside the OSS communities + memory-god-node + index
articles, it emits one article per canonical entity from the enterprise Entity
Registry (the GLiNER2/NER-derived Person/Org/Concept nodes with their accreted
member memories).

Why this exists: the OSS "entity" articles are built from GRAPH god-nodes — the
most-connected individual *memories* — which are proxies, not real entities. The
enterprise registry already resolves surface names to stable canonical entities
and accretes the memories that mention each one. Turning those into KB articles
gives the "proper entity graph" view (a real Person/Org/Concept page grounded in
the exact memories that mention it) instead of a central-memory proxy.

╔══════════════════════════════════════════════════════════════════════════════╗
║ CARDINAL CONSTRAINT — this layer is STRICTLY ADDITIVE and STRICTLY FAIL-SAFE.  ║
║ Everything the OSS service produces is UNCHANGED. Entity articles are produced ║
║ via the ``_generate_extra_articles`` seam, whose failures are caught in the    ║
║ OSS ``generate`` (→ no extra articles = exact OSS output). No registry, no LLM,║
║ a registry error, or the feature disabled all degrade to the OSS KB.           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Grounding matches the community summaries: each member memory is fed id-prefixed
(``[memory:<id>]``) within the shared char budget and the model is asked to cite
the ids its claims draw from (never invent one). Reuse is content-hashed on the
entity identity + aliases + member content-versions, so an unchanged entity's
article is reused verbatim with no LLM call (same discipline as community/god-node
articles).

Selected only when ``MEMORYLAYER_KNOWLEDGEBASE_PROVIDER=enterprise`` (the
enterprise ``dependencies.py`` sets that as the default). Gated by
``MEMORYLAYER_KB_ENTITY_ARTICLES_ENABLED`` (default ON).
"""

import hashlib
import logging
from datetime import UTC, datetime

from memorylayer_server.models.graph_analysis import Community, GraphAnalysis
from memorylayer_server.models.generation import GenerationActivity
from memorylayer_server.services._constants import (
    EXT_CONTRADICTION_SERVICE,
    EXT_ENTITY_REGISTRY_SERVICE,
    EXT_GRAPH_ANALYSIS_SERVICE,
    EXT_INFERENCE_SERVICE,
    EXT_REFLECT_SERVICE,
    EXT_STORAGE_BACKEND,
)
from memorylayer_server.services.knowledgebase import (
    KnowledgebaseServicePluginBase,
)
from memorylayer_server.services.knowledgebase import differ as kb_differ
from memorylayer_server.services.knowledgebase.base import Article
from memorylayer_server.services.knowledgebase.citations import (
    CitationReport,
    audit_citations,
    summary_snippet,
)
from memorylayer_server.services.knowledgebase.default import DefaultKnowledgebaseService
from memorylayer_server.services.knowledgebase.linkcheck import (
    RENDER_FORMAT_VERSION,
    LinkIndex,
)
from memorylayer_server.services.llm import EXT_LLM_SERVICE
from scitrera_app_framework import Variables, ext_parse_bool, get_extension

from memorylayer_saas.config import (
    DEFAULT_MEMORYLAYER_KB_ENTITY_ARTICLES_ENABLED,
    DEFAULT_MEMORYLAYER_KB_GODNODE_ARTICLES_ENABLED,
    DEFAULT_MEMORYLAYER_KB_MAX_ENTITY_ARTICLES,
    DEFAULT_MEMORYLAYER_KB_MIN_ENTITY_MEMBERS,
    MEMORYLAYER_KB_ENTITY_ARTICLES_ENABLED,
    MEMORYLAYER_KB_GODNODE_ARTICLES_ENABLED,
    MEMORYLAYER_KB_MAX_ENTITY_ARTICLES,
    MEMORYLAYER_KB_MIN_ENTITY_MEMBERS,
)


class EnterpriseKnowledgebaseService(DefaultKnowledgebaseService):
    """OSS KB + one additive layer of canonical-entity-registry articles."""

    # Prompt for a single entity's description. Mirrors the community prompt's
    # grounding contract (cite numbered references; never invent one). The gateway
    # strips reasoning, so the only requirement is a token budget large enough to
    # finish (reuses self.summary_max_tokens).
    _ENTITY_PROMPT = (
        "You are writing the knowledge-base article for a single entity named "
        '"{name}" (type: {etype}). Below are memories that mention this entity, '
        "each numbered, e.g. [m1], [m2].\n"
        "Write a concise 2-5 sentence description in neutral, factual, third "
        "person: what this entity is and the key facts or relationships the "
        "memories establish about it. Ground each claim by citing the reference "
        "number(s) it draws from in square brackets, e.g. [m1] or [m2][m5]. Only "
        "cite numbers shown in the list below; never invent a number. Output ONLY "
        "the description — no title, no preamble.\n\n"
        "Memories:\n{memories}"
    )

    def __init__(
        self,
        *args,
        entity_registry=None,
        entity_articles_enabled: bool = DEFAULT_MEMORYLAYER_KB_ENTITY_ARTICLES_ENABLED,
        max_entity_articles: int = DEFAULT_MEMORYLAYER_KB_MAX_ENTITY_ARTICLES,
        min_entity_members: int = DEFAULT_MEMORYLAYER_KB_MIN_ENTITY_MEMBERS,
        godnode_articles_enabled: bool = DEFAULT_MEMORYLAYER_KB_GODNODE_ARTICLES_ENABLED,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Entity registry is SOFT: absent -> the enterprise layer is a no-op and
        # the service behaves exactly like the OSS default.
        self.entity_registry = entity_registry
        self.entity_articles_enabled = entity_articles_enabled
        self.max_entity_articles = max_entity_articles
        self.min_entity_members = min_entity_members
        self.godnode_articles_enabled = godnode_articles_enabled
        self.logger.info(
            "Initialized EnterpriseKnowledgebaseService (entity_articles=%s, registry=%s, "
            "max=%d, min_members=%d, godnodes=%s)",
            entity_articles_enabled,
            entity_registry is not None,
            max_entity_articles,
            min_entity_members,
            godnode_articles_enabled,
        )

    def _select_god_nodes(self, analysis, options):
        """Suppress the OSS central-memory god-node proxies when the registry-entity
        layer is actually producing articles — those canonical typed entities are the
        real 'entity' surface, so the memory proxies are redundant noise. Keeps god
        nodes when explicitly re-enabled, or when the registry layer is inactive (so
        the KB is never left with zero entity articles)."""
        if self.godnode_articles_enabled:
            return super()._select_god_nodes(analysis, options)
        if self.entity_articles_enabled and self.entity_registry is not None:
            return []
        return super()._select_god_nodes(analysis, options)

    async def _generate_extra_articles(
        self,
        workspace_id: str,
        analysis: GraphAnalysis,
        community_by_id: dict[int, Community],
        force_regenerate: bool = False,
        contradicted_ids: set[str] | None = None,
        link_index: LinkIndex | None = None,
    ) -> list[Article]:
        """One article per canonical entity in the registry (fail-safe, additive)."""
        if not (self.entity_articles_enabled and self.entity_registry):
            return []

        try:
            entities = await self.entity_registry.list_entities(
                workspace_id, status="active", limit=self.max_entity_articles
            )
        except NotImplementedError:
            return []
        except Exception as e:  # noqa: BLE001 - registry optional; degrade to OSS KB
            self.logger.debug("Entity-registry KB: list_entities failed for %s: %s", workspace_id, e)
            return []

        # Compute the entity co-occurrence graph ONCE for the whole batch (best-effort)
        # so each article's Connections section is a dict lookup rather than a per-entity
        # membership scan. Degrades cleanly to no related section if unsupported.
        cooccurrence: dict[str, dict[str, int]] = {}
        try:
            cooccurrence = await self.entity_registry.cooccurrence_map(workspace_id)
        except Exception as e:  # noqa: BLE001 - relatedness is additive; never fail the batch
            self.logger.debug("Entity-registry KB: cooccurrence_map unavailable for %s: %s", workspace_id, e)
        # Phase 1 — resolve membership and eligibility for EVERY candidate before any
        # article is rendered. An entity below min_entity_members yields no article, so
        # registering only the eligible ones is what lets a Connections link be emitted
        # exclusively for entities that actually have a page. Members are carried into
        # phase 2 rather than fetched twice.
        eligible: list[tuple] = []
        for entity in entities:
            try:
                members = await self.entity_registry.list_members(
                    workspace_id, entity.id, limit=self.community_max_members
                )
            except Exception as e:  # noqa: BLE001 - skip this entity on member-fetch error
                self.logger.debug("Entity-registry KB: list_members failed for %s: %s", entity.id, e)
                continue
            # Skip thin entities (1 mention is noise, same principle as min_community_size).
            if len(members) < self.min_entity_members:
                continue
            article_id = self._registry_entity_article_id(entity)
            if link_index is not None:
                link_index.add_entity(entity.id, article_id)
            eligible.append((entity, members, article_id))

        # Only entities that reached phase 2 are linkable targets, which is what makes
        # _top_related's "every link points at a real article" contract true.
        name_by_id = {entity.id: entity.canonical_name for entity, _members, _aid in eligible}

        articles: list[Article] = []
        for entity, members, article_id in eligible:
            try:
                related = self._top_related(entity.id, cooccurrence, name_by_id)
                article = await self._generate_registry_entity_article(
                    workspace_id,
                    entity,
                    force_regenerate,
                    contradicted_ids,
                    related=related,
                    members=members,
                    article_id=article_id,
                    link_index=link_index,
                )
                if article is not None:
                    articles.append(article)
            except Exception as e:  # noqa: BLE001 - one bad entity never fails the batch
                self.logger.debug(
                    "Entity-registry KB: article failed for entity %s: %s",
                    getattr(entity, "id", None),
                    e,
                )
        self.logger.info(
            "Entity-registry KB: produced %d entity articles for workspace=%s "
            "(from %d entities, %d eligible)",
            len(articles),
            workspace_id,
            len(entities),
            len(eligible),
        )
        return articles

    @staticmethod
    def _top_related(
        entity_id: str, cooccurrence: dict[str, dict[str, int]], name_by_id: dict[str, str], *, limit: int = 8
    ) -> list[dict]:
        """Top co-occurring entities for one entity, resolved to names.

        Only neighbors that themselves have an article (present in ``name_by_id``) are
        kept, so every Connections link points at a real entity article; ranked by
        shared-memory count (id as a stable tiebreak).
        """
        others = cooccurrence.get(entity_id) or {}
        ranked = sorted(others.items(), key=lambda kv: (-kv[1], kv[0]))
        out: list[dict] = []
        for rid, shared in ranked:
            name = name_by_id.get(rid)
            if name:
                # The id is what resolves the link to the neighbor's article; the name
                # is only what the link displays.
                out.append({"id": rid, "name": name, "shared": shared})
            if len(out) >= limit:
                break
        return out

    def _registry_entity_article_id(self, entity) -> str:
        """Article id for a canonical-entity article.

        The ONE place this id is derived, so the link index and the article itself
        cannot disagree. The entity-id suffix keeps entities that slugify identically
        ("C++"/"C") apart and stops a clash with an OSS god-node article (those suffix
        a memory_id). GC + content-hash reuse are keyed on this article_id.
        """
        return f"entity-{self.renderer.slugify(entity.canonical_name)}-{entity.id[:8]}"

    async def _generate_registry_entity_article(
        self,
        workspace_id: str,
        entity,
        force_regenerate: bool,
        contradicted_ids: set[str] | None = None,
        related: list[dict] | None = None,
        members: list | None = None,
        article_id: str | None = None,
        link_index: LinkIndex | None = None,
    ) -> Article | None:
        """Build one grounded article for a canonical entity, or None to skip it.

        ``related`` is the pre-computed top co-occurring entities
        (``[{"id", "name", "shared"}]``) rendered as the article's Connections section.
        ``members`` and ``article_id`` come from the caller's eligibility pass so this
        method neither re-fetches membership nor re-derives the id.
        """
        # Membership and the thin-entity skip were resolved by the caller's phase-1
        # eligibility pass; members arrive already fetched.
        members = members or []

        # Fetch member contents (+ content-versions for the reuse hash).
        member_dicts: list[dict] = []
        content_versions: dict[str, str] = {}
        for member in members:
            mem_id = member.memory_id
            try:
                mem = await self.storage.get_memory(workspace_id, mem_id, track_access=False)
            except Exception as e:  # noqa: BLE001 - a missing member just drops out
                self.logger.debug("Entity-registry KB: get_memory %s failed: %s", mem_id, e)
                continue
            if mem:
                member_dicts.append(self._member_dict(mem))
                content_versions[mem_id] = self._content_version(mem)

        if len(member_dicts) < self.min_entity_members:
            return None

        slug = self.renderer.slugify(entity.canonical_name)
        # Resolved in phase 1 so the link index and this article agree; recomputed only
        # when this method is driven directly. See _registry_entity_article_id.
        article_id = article_id or self._registry_entity_article_id(entity)

        # Content-hash skip: reuse verbatim if identity + aliases + members unchanged.
        content_key = self._registry_entity_content_key(entity, content_versions)
        if not force_regenerate:
            reuse = await self._maybe_reuse_article(workspace_id, article_id, content_key)
            if reuse is not None:
                return reuse

        # Grounding metric (measurement only — never rewrites the summary). The report
        # is audited against the numbered member block the model was shown; it flags
        # any out-of-range (invented) reference number.
        summary, report = await self._summarize_registry_entity(entity, member_dicts)
        if report.invalid:
            self.logger.warning(
                "KB entity %s (%s) summary cited %d out-of-range reference(s): %s",
                entity.id, entity.canonical_name, report.invalid, report.invalid_ids[:5],
            )

        # Type + aliases lead the insights, then the grounded summary. render_entity
        # renders insights as a bullet list, so keep each a single line.
        insights: list[str] = [f"**Type:** {entity.entity_type}"]
        if entity.aliases:
            insights.append(f"**Also known as:** {', '.join(entity.aliases[:8])}")
        insights.append(summary)

        # OKF v0.1 + trust/provenance + identity frontmatter (Obsidian properties).
        now = datetime.now(UTC)
        fm_fields = {
            "type": "entity",
            "title": entity.canonical_name,
            "description": summary_snippet(summary),
            "entity_type": entity.entity_type,
            "aliases": list(entity.aliases)[:8],
            "timestamp": now.isoformat(),
            "member_count": len(member_dicts),
            **self._provenance_fields(member_dicts, contradicted_ids),
            "citation_coverage": report.as_metadata()["citation_coverage"],
        }
        # Related entities → Connections section. Strength is the co-occurrence share
        # relative to this entity's own member set (shared / own members), so a related
        # entity that appears alongside it in most of its memories scores near 1.0.
        denom = max(len(member_dicts), 1)
        connections = [
            {
                "target_id": r["id"],
                "target_title": r["name"],
                "relationship": "related_to",
                "strength": min(r["shared"] / denom, 1.0),
            }
            for r in (related or [])
        ]
        content_md = self.renderer.render_frontmatter(fm_fields) + self.renderer.render_entity(
            entity_id=entity.id,
            title=entity.canonical_name,
            entity_card={"insights": insights},
            connections=connections,
            community=None,
            source_memories=member_dicts,
            link_index=link_index,
        )

        return Article(
            id=article_id,
            article_type="entity",
            title=entity.canonical_name,
            content_md=content_md,
            metadata={
                "entity_id": entity.id,
                "entity_type": entity.entity_type,
                "aliases": list(entity.aliases),
                "member_count": len(member_dicts),
                "related_count": len(connections),
                "member_ids": self.member_citation_ids(member_dicts),
                "slug": slug,
                "source": "entity_registry",
                **report.as_metadata(),
                kb_differ.CONTENT_KEY_FIELD: content_key,
            },
            generated_at=now,
        )

    async def _summarize_registry_entity(self, entity, members: list[dict]) -> tuple[str, CitationReport]:
        """Grounded one-paragraph entity description (falls back deterministically).

        Returns ``(description, citation_report)`` — the report is audited against
        the number of members actually shown to the model.
        """
        combined, snippets, member_count = self._build_member_block(members)

        def _fallback() -> tuple[str, CitationReport]:
            joined = "; ".join(snippets[:3])
            return (
                f"{entity.canonical_name} is referenced by {len(members)} "
                f"memories in this workspace. Representative mentions: {joined}.",
                CitationReport(),
            )

        if not self.llm or not combined:
            return _fallback()

        try:
            raw = await self.llm.synthesize(
                prompt=self._ENTITY_PROMPT.format(
                    name=entity.canonical_name,
                    etype=entity.entity_type,
                    memories=combined,
                ),
                max_tokens=self.summary_max_tokens,
                profile="reflection",
                activity=GenerationActivity.SYNTHESIS,
            )
        except Exception as e:  # noqa: BLE001 - any LLM failure -> deterministic body
            self.logger.debug("Entity-registry KB: summarize failed for %s: %s", entity.id, e)
            return _fallback()

        summary = (raw or "").strip()
        if not summary:
            return _fallback()
        return summary, audit_citations(summary, member_count)

    @staticmethod
    def _registry_entity_content_key(entity, content_versions: dict[str, str]) -> str:
        """Deterministic content hash for entity-article reuse.

        Covers the entity identity, canonical name, aliases, and each member's
        content-version — a change in any of them re-summarizes; otherwise the
        prior article is reused verbatim (no LLM call).

        The render-format version is folded in so a change to the RENDERER (not just
        the inputs) also busts reuse; without it a rendering fix would never reach an
        otherwise-unchanged article.
        """
        h = hashlib.sha256()
        h.update(RENDER_FORMAT_VERSION.encode("utf-8"))
        h.update(b"\0")
        h.update(entity.id.encode("utf-8"))
        h.update(b"\0")
        h.update(entity.canonical_name.encode("utf-8"))
        h.update(b"\0")
        h.update("|".join(sorted(entity.aliases)).encode("utf-8"))
        h.update(b"\0")
        for mid in sorted(content_versions):
            h.update(mid.encode("utf-8"))
            h.update(b"=")
            h.update(content_versions[mid].encode("utf-8"))
            h.update(b";")
        return h.hexdigest()


class EnterpriseKnowledgebaseServicePlugin(KnowledgebaseServicePluginBase):
    """Plugin for the enterprise knowledgebase service (registry-entity articles).

    Auto-discovered by the enterprise ``register_package_plugins(services...,
    recursive=True)`` scan. Selected only when
    ``MEMORYLAYER_KNOWLEDGEBASE_PROVIDER=enterprise``. Storage + graph analysis
    are required (the OSS KB deps); the entity registry + reflect + inference +
    LLM services are SOFT — any that cannot be resolved simply degrade the
    corresponding layer (no registry -> OSS KB; no LLM -> deterministic summaries).
    """

    PROVIDER_NAME = "enterprise"

    def get_dependencies(self, v: Variables):
        # Registry listed so the framework initializes it before us when present;
        # it is still resolved defensively (soft) in initialize().
        return (
            EXT_STORAGE_BACKEND,
            EXT_GRAPH_ANALYSIS_SERVICE,
            EXT_ENTITY_REGISTRY_SERVICE,
            EXT_LLM_SERVICE,
        )

    def initialize(self, v: Variables, logger: logging.Logger) -> EnterpriseKnowledgebaseService:
        storage = self.get_extension(EXT_STORAGE_BACKEND, v)
        graph_service = self.get_extension(EXT_GRAPH_ANALYSIS_SERVICE, v)

        def _soft(ext_name: str, label: str):
            try:
                return get_extension(ext_name, v)
            except Exception:  # noqa: BLE001 - optional dependency; degrade gracefully
                logger.debug("%s not available for EnterpriseKnowledgebaseService", label)
                return None

        entity_registry = _soft(EXT_ENTITY_REGISTRY_SERVICE, "EntityRegistryService")
        reflect_service = _soft(EXT_REFLECT_SERVICE, "ReflectService")
        inference_service = _soft(EXT_INFERENCE_SERVICE, "InferenceService")
        llm_service = _soft(EXT_LLM_SERVICE, "LLMService")
        contradiction_service = _soft(EXT_CONTRADICTION_SERVICE, "ContradictionService")

        entity_articles_enabled = ext_parse_bool(
            v.environ(
                MEMORYLAYER_KB_ENTITY_ARTICLES_ENABLED,
                default=DEFAULT_MEMORYLAYER_KB_ENTITY_ARTICLES_ENABLED,
            )
        )
        max_entity_articles = v.environ(
            MEMORYLAYER_KB_MAX_ENTITY_ARTICLES,
            default=DEFAULT_MEMORYLAYER_KB_MAX_ENTITY_ARTICLES,
            type_fn=int,
        )
        min_entity_members = v.environ(
            MEMORYLAYER_KB_MIN_ENTITY_MEMBERS,
            default=DEFAULT_MEMORYLAYER_KB_MIN_ENTITY_MEMBERS,
            type_fn=int,
        )
        godnode_articles_enabled = ext_parse_bool(
            v.environ(
                MEMORYLAYER_KB_GODNODE_ARTICLES_ENABLED,
                default=DEFAULT_MEMORYLAYER_KB_GODNODE_ARTICLES_ENABLED,
            )
        )

        if entity_articles_enabled and entity_registry is None:
            logger.warning(
                "EnterpriseKnowledgebaseService: entity articles enabled but the entity "
                "registry is unavailable; entity-article layer is a no-op (OSS KB preserved)."
            )

        return EnterpriseKnowledgebaseService(
            storage=storage,
            graph_service=graph_service,
            reflect_service=reflect_service,
            inference_service=inference_service,
            llm_service=llm_service,
            contradiction_service=contradiction_service,
            v=v,
            entity_registry=entity_registry,
            entity_articles_enabled=entity_articles_enabled,
            max_entity_articles=max_entity_articles,
            min_entity_members=min_entity_members,
            godnode_articles_enabled=godnode_articles_enabled,
        )
