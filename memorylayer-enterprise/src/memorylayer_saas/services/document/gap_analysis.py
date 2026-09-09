"""Path-agnostic document ingestion gap analysis (Phase 1).

``analyze_document_gaps`` reads the *persisted* state of a document (pages,
per-page artifacts, store-phase memories) and the *effective* ingest feature
flags to decide which phases are complete and, if not, which phase to resume
at. It is the shared core reused by the ``doc_added`` entry point (classify →
no-op | resume | fresh) and a future ``doc_verify`` reconcile sweep.

Completeness is judged against the EFFECTIVE flag set: flags pinned on
``doc.metadata['ingest_flags']`` at finalize (see ``document_finalize``) take
precedence over the current config, so a later flag flip does not retroactively
mark old documents incomplete.

Store-completeness rule (per design decision #4): a transcribed page is
store-complete when it has ≥1 composite memory (``page.id`` present in
``get_memory_source_page_ids``). Fact-decomposition / graph materialization are
additive / self-healing and are NOT gap phases here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from scitrera_app_framework import Variables, ext_parse_bool

from ...config import (
    DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED,
    DEFAULT_MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
    DEFAULT_MEMORYLAYER_VISUAL_TOKENIZER_ENABLED,
    MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED,
    MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
    MEMORYLAYER_VISUAL_TOKENIZER_ENABLED,
)
from memorylayer_server.config import (
    MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
    DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
)

from ...models.document import Document, DocumentEnrichmentStatus


# Phase order — ``first_missing_phase`` is the earliest of these that has a gap.
PHASE_RENDER = "render"
PHASE_TRANSCRIBE = "transcribe"
PHASE_EMBED = "embed"
PHASE_STORE = "store"
_PHASE_ORDER = (PHASE_RENDER, PHASE_TRANSCRIBE, PHASE_EMBED, PHASE_STORE)


@dataclass
class EffectiveFlags:
    """The effective ingest feature flags that decide which phases are required."""
    transcribe: bool
    chat_ingest: bool
    visual_tokenizer: bool
    #: Whether page memories get decomposed into atomic facts. Unlike the other
    #: flags this does NOT change what "complete" means -- the composite page
    #: memory is still created either way, so gap analysis is unaffected. It
    #: only suppresses the fan-out, which is the expensive part: one decomposed
    #: page becomes ~35-40 facts, each scheduling its own tiering and
    #: association work.
    decompose: bool = True

    def as_dict(self) -> dict:
        return {
            "transcribe": self.transcribe,
            "chat_ingest": self.chat_ingest,
            "visual_tokenizer": self.visual_tokenizer,
            "decompose": self.decompose,
        }


@dataclass
class DocumentGaps:
    """Per-document gap analysis result.

    ``pages_missing_*`` hold the page ids (or page numbers, for render) that are
    missing each artifact. ``first_missing_phase`` is the earliest phase with a
    gap and drives "resume at phase X"; it is ``None`` when ``is_complete``.
    """
    document_id: str
    has_pages: bool
    expected_page_count: int
    missing_render: bool
    # Factual page tallies, independent of the flag gates below. ``is_complete``
    # answers "should anything still run?"; these answer "how far along is it?".
    # They differ when transcription is disabled: every page can lack a
    # transcript while the document is legitimately complete.
    page_count: int = 0
    pages_with_transcript: int = 0
    pages_missing_transcript: list[str] = field(default_factory=list)
    pages_missing_image_embed: list[str] = field(default_factory=list)
    pages_missing_text_embedding: list[str] = field(default_factory=list)
    pages_missing_multivector: list[str] = field(default_factory=list)
    pages_missing_memory: list[str] = field(default_factory=list)
    is_complete: bool = False
    first_missing_phase: Optional[str] = None
    flags: Optional[EffectiveFlags] = None


def resolve_effective_flags(
    v: Variables, doc: Document, flags: Optional[EffectiveFlags] = None,
) -> EffectiveFlags:
    """Resolve the effective ingest flags for a document.

    Precedence: explicit ``flags`` arg > flags pinned on ``doc.metadata``
    (``ingest_flags``, written at finalize) > flags REQUESTED at upload
    (``requested_ingest_flags``, seeded at document creation from the VFS
    entry) > current config defaults.

    The upload request lives under its OWN key rather than seeding
    ``ingest_flags`` directly. ``ingest_flags`` is written at finalize as a
    COMPLETE set (``eff.as_dict()``) and the pinned branch below reads absent
    keys as hardcoded defaults — so seeding a PARTIAL dict there would silently
    freeze transcribe/chat_ingest/visual_tokenizer at those hardcoded values and
    override the deployment's env config. Keeping "what the uploader asked for"
    separate from "what was in force at ingest time" also keeps a gap-fill
    re-run repeating the pinned decision rather than re-deriving it.
    """
    if flags is not None:
        return flags

    md = doc.metadata or {}

    requested = md.get("requested_ingest_flags")
    if not isinstance(requested, dict):
        requested = {}
    # An upload-time request outranks the env default but not a finalize pin.
    decompose_default = (
        bool(requested["decompose"]) if "decompose" in requested
        else _decompose_default(v)
    )

    pinned = md.get("ingest_flags")
    if isinstance(pinned, dict):
        return EffectiveFlags(
            transcribe=bool(pinned.get("transcribe", True)),
            chat_ingest=bool(pinned.get("chat_ingest", False)),
            visual_tokenizer=bool(pinned.get("visual_tokenizer", False)),
            # Absent means "not specified", which must fall back to the global
            # default rather than read as a decision to disable. Documents
            # pinned before this flag existed have no key and keep decomposing.
            decompose=bool(pinned.get("decompose", decompose_default)),
        )

    return EffectiveFlags(
        transcribe=v.environ(
            MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
            default=DEFAULT_MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
            type_fn=ext_parse_bool,
        ),
        chat_ingest=v.environ(
            MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED,
            default=DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED,
            type_fn=ext_parse_bool,
        ),
        visual_tokenizer=v.environ(
            MEMORYLAYER_VISUAL_TOKENIZER_ENABLED,
            default=DEFAULT_MEMORYLAYER_VISUAL_TOKENIZER_ENABLED,
            type_fn=ext_parse_bool,
        ),
        # Already folded in the upload request, if there was one.
        decompose=decompose_default,
    )


def _decompose_default(v: Variables) -> bool:
    """The decomposition default when a document does not specify one.

    Reads the same OSS switch MemoryService consults, so an unspecified
    per-document flag and no per-document flag at all behave identically.
    """
    return v.environ(
        MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
        default=DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
        type_fn=ext_parse_bool,
    )


def _page_has_text_embedding(page) -> bool:
    """Single-vector embedding lives in the dedicated ``page.embedding`` column.

    (Mirrors document_embed.py / document_finalize.py which write/read it there;
    see DESIGN_idempotent_ingestion.md P2.4.)

    Transition fallback: also accept the legacy ``metadata['_embedding']`` stash
    for any not-yet-migrated row. Removable once the P2.4 migration is
    universally applied.
    """
    if page.embedding is not None:
        return True
    meta = page.metadata or {}
    return meta.get("_embedding") is not None


def _page_has_image_embed(page) -> bool:
    """Image-embeds are recorded under ``page.visual_tokens`` keyed by the
    embed-server's model slug. The slug is only known at embed time, so gap
    analysis treats a non-empty ``visual_tokens`` dict as "image-embed present"
    rather than pinning a specific slug.
    """
    return bool(page.visual_tokens)


async def analyze_document_gaps(
    v: Variables,
    storage,
    doc: Document,
    *,
    flags: Optional[EffectiveFlags] = None,
) -> DocumentGaps:
    """Analyze which ingestion artifacts a document is missing.

    Args:
        v: Variables instance (for flag resolution).
        storage: Storage backend (needs ``get_pages`` and
            ``get_memory_source_page_ids``).
        doc: Document domain model.
        flags: Optional explicit effective flags (else resolved from the
            document's pinned flags or current config).

    Returns:
        A ``DocumentGaps`` describing the missing artifacts and the first phase
        to resume at.
    """
    eff = resolve_effective_flags(v, doc, flags)

    pages = await storage.get_pages(doc.id, doc.workspace_id)

    # Store-phase: which page ids already have ≥1 composite memory.
    mem_page_ids = await storage.get_memory_source_page_ids(doc.workspace_id, doc.id)

    return _gaps_from_pages(doc, pages, mem_page_ids, eff)


async def analyze_documents_gaps(
    v: Variables,
    storage,
    docs: list[Document],
    *,
    flags: Optional[EffectiveFlags] = None,
) -> list[DocumentGaps]:
    """:func:`analyze_document_gaps` over several documents, batched.

    Same verdict per document, but resolved with two queries total instead of
    two *per document*. Falls back to sequential per-document analysis when the
    storage backend lacks the batch primitives, so any backend keeps working.

    Flags are still resolved per document, since each document may carry its own
    pinned ``ingest_flags`` from finalize time.

    Args:
        v: Variables instance (for flag resolution).
        storage: Storage backend. Uses ``get_pages_for_documents`` and
            ``get_memory_source_page_ids_for_documents`` when available.
        docs: Document domain models. All must share a workspace.
        flags: Optional explicit effective flags, applied to every document.

    Returns:
        One ``DocumentGaps`` per input document, in the same order.
    """
    if not docs:
        return []

    batch_pages = getattr(storage, "get_pages_for_documents", None)
    batch_mem = getattr(storage, "get_memory_source_page_ids_for_documents", None)
    if batch_pages is None or batch_mem is None:
        return [
            await analyze_document_gaps(v, storage, doc, flags=flags)
            for doc in docs
        ]

    workspace_id = docs[0].workspace_id
    doc_ids = [d.id for d in docs]

    all_pages = await batch_pages(doc_ids, workspace_id)
    mem_page_ids = await batch_mem(workspace_id, doc_ids)

    pages_by_doc: dict[str, list] = {d.id: [] for d in docs}
    for page in all_pages:
        # A page for an id we did not ask about cannot happen via the batch
        # query, but guard rather than KeyError if a backend returns extra.
        if page.document_id in pages_by_doc:
            pages_by_doc[page.document_id].append(page)

    return [
        _gaps_from_pages(
            doc,
            pages_by_doc[doc.id],
            mem_page_ids,
            resolve_effective_flags(v, doc, flags),
        )
        for doc in docs
    ]


def _gaps_from_pages(
    doc: Document,
    pages: list,
    mem_page_ids: set,
    eff: EffectiveFlags,
) -> DocumentGaps:
    """Decide a document's gaps from its already-loaded pages and memory ids.

    Split out of :func:`analyze_document_gaps` so the batch variant shares the
    exact same rules rather than reimplementing them — the flag-gate logic below
    is subtle enough that two copies would drift.

    ``mem_page_ids`` may cover several documents (the batch caller passes one
    combined set); membership is by page id, which is globally unique, so an
    over-broad set is harmless.
    """
    has_pages = bool(pages)
    expected = doc.page_count or 0

    # Render gap: no pages at all, or fewer pages than the document declares.
    missing_render = (not has_pages) or (expected > 0 and len(pages) < expected)

    pages_missing_transcript: list[str] = []
    pages_missing_image_embed: list[str] = []
    pages_missing_text_embedding: list[str] = []
    pages_missing_multivector: list[str] = []
    pages_missing_memory: list[str] = []

    # A page needs a transcript only when transcription is the required source
    # of page text. When transcribe is disabled AND chat-ingest is enabled, the
    # embed phase supplies the text from image-embeds, so a missing transcript
    # is not itself a transcribe-phase gap (it is filled in embed).
    transcript_required = eff.transcribe

    # Image-embeds are required only when the visual tokenizer is enabled, or
    # when chat-ingest needs them to generate page text (mirrors the embed
    # phase's precompute gate: vt_enabled or chat_ingest_enabled).
    image_embed_required = eff.visual_tokenizer or eff.chat_ingest

    pages_with_transcript = 0

    for page in pages:
        pid = page.id
        has_transcript = bool(page.transcript)
        if has_transcript:
            pages_with_transcript += 1

        if transcript_required and not has_transcript:
            pages_missing_transcript.append(pid)

        if image_embed_required and not _page_has_image_embed(page):
            pages_missing_image_embed.append(pid)

        # Multivector is produced for every rendered (image) page.
        if page.image_storage_path and page.multivector is None:
            pages_missing_multivector.append(pid)

        # A page that has (or will have) text needs a single-vector embedding.
        # Only meaningful once the page carries a transcript; a page that never
        # gets text (no transcript + no chat-ingest) is not a text-embed gap.
        if has_transcript and not _page_has_text_embedding(page):
            pages_missing_text_embedding.append(pid)

        # Store gap: a transcribed page with no composite memory yet.
        if has_transcript and pid not in mem_page_ids:
            pages_missing_memory.append(pid)

    # Determine the first missing phase by pipeline order.
    first_missing_phase: Optional[str] = None
    if missing_render:
        first_missing_phase = PHASE_RENDER
    elif pages_missing_transcript:
        first_missing_phase = PHASE_TRANSCRIBE
    elif (
        pages_missing_image_embed
        or pages_missing_multivector
        or pages_missing_text_embedding
    ):
        first_missing_phase = PHASE_EMBED
    elif pages_missing_memory:
        first_missing_phase = PHASE_STORE

    # Complete = pages exist, render not missing, and every required artifact is
    # present on every page (including ≥1 memory per transcribed page). A
    # document with pages but zero memories is NOT complete.
    has_any_memory = bool(mem_page_ids)
    is_complete = (
        has_pages
        and not missing_render
        and first_missing_phase is None
        and has_any_memory
    )

    return DocumentGaps(
        document_id=doc.id,
        has_pages=has_pages,
        expected_page_count=expected,
        missing_render=missing_render,
        page_count=len(pages),
        pages_with_transcript=pages_with_transcript,
        pages_missing_transcript=pages_missing_transcript,
        pages_missing_image_embed=pages_missing_image_embed,
        pages_missing_text_embedding=pages_missing_text_embedding,
        pages_missing_multivector=pages_missing_multivector,
        pages_missing_memory=pages_missing_memory,
        is_complete=is_complete,
        first_missing_phase=first_missing_phase,
        flags=eff,
    )


def resolve_enrichment_status(
    doc: Document, outstanding_parents: list[str],
) -> tuple[DocumentEnrichmentStatus, list[str]]:
    """Re-derive a document's knowledge phase from stored state.

    Pure, and deliberately so: the phase is recomputed from what is already on
    the document plus the fact-gap result the caller has in hand, rather than
    counted from completion events. Counters drift under retries, replays and
    worker restarts; re-derivation converges to the truth every sweep.

    The two inputs answer different halves and are BOTH needed:

    * ``doc.enrichment_memory_ids`` — what was actually scheduled at ingest.
      Without it, a page whose content was already atomic (decomposition
      early-returns, leaving the memory ACTIVE and factless forever) reads as
      permanently outstanding and the document never reaches COMPLETE.
    * ``outstanding_parents`` — of those, which have not produced facts yet,
      from :func:`analyze_fact_gaps`. Without it there is nothing to converge.

    Args:
        doc: Document domain model.
        outstanding_parents: Composite memory ids still missing facts, i.e. the
            return of :func:`analyze_fact_gaps` for the same document.

    Returns:
        ``(status, still_outstanding_ids)``.
    """
    scheduled = list(doc.enrichment_memory_ids or [])
    if not scheduled:
        # Nothing was ever handed to decomposition, so there is nothing to
        # finish. Distinct from COMPLETE: "never ran" should stay legible.
        return DocumentEnrichmentStatus.NOT_APPLICABLE, []

    outstanding = set(outstanding_parents or ())
    still = [mid for mid in scheduled if mid in outstanding]
    status = (
        DocumentEnrichmentStatus.PENDING if still else DocumentEnrichmentStatus.COMPLETE
    )
    return status, still


async def analyze_fact_gaps(storage, doc: Document) -> list[str]:
    """Detect page composite memories whose fact decomposition never ran.

    This is the deferred completeness hole (design §8 / decision #4): a page's
    composite memory exists (so the document looks store-complete) but the
    ``decompose_facts`` task scheduled for it by ``enqueue_post_store`` never
    produced any atomic fact — typically because the worker crashed. It is kept
    OUT of ``analyze_document_gaps`` / finalize on purpose: fact decomposition is
    additive and best-effort, so a missing fact must NEVER mark a document
    incomplete or fail it. ``doc_verify`` calls this separately and re-schedules
    ``decompose_facts`` for the returned memory ids.

    Detection: among the document's live page composite memories (those with a
    ``source_page_id``), a still-``ACTIVE`` memory with NO derived
    ``subtype='fact'`` chaining to it (via ``source_memory_id``) is a candidate.
    Composites that were already decomposed are ARCHIVED by ``decompose_facts``
    and so are skipped. Re-scheduling is idempotent: the handler self-corrects
    any already-atomic / already-archived parent (early-returns without creating
    duplicates), so over-inclusion here is harmless.

    Args:
        storage: Storage backend (needs ``get_document_memories`` and
            ``get_fact_memory_parent_ids``).
        doc: Document domain model.

    Returns:
        Composite memory ids that should have ``decompose_facts`` re-scheduled.
        Empty when nothing is missing or the storage backend lacks the methods.
    """
    get_doc_mems = getattr(storage, "get_document_memories", None)
    get_fact_parents = getattr(storage, "get_fact_memory_parent_ids", None)
    if get_doc_mems is None or get_fact_parents is None:
        return []

    memories = await get_doc_mems(doc.workspace_id, doc.id)

    # Only page composite memories are decomposition sources. A fact memory
    # (subtype='fact') or a non-page memory is never itself a decompose target.
    composites = [
        m for m in memories
        if getattr(m, "source_page_id", None)
        and getattr(m, "subtype", None) != "fact"
    ]
    if not composites:
        return []

    composite_ids = {m.id for m in composites}
    parents_with_facts = await get_fact_parents(doc.workspace_id, composite_ids)

    missing: list[str] = []
    for mem in composites:
        # Skip parents already decomposed (archived) or that produced facts.
        status = getattr(mem, "status", None)
        status_value = getattr(status, "value", status)
        if status_value == "archived":
            continue
        if mem.id not in parents_with_facts:
            missing.append(mem.id)
    return missing
