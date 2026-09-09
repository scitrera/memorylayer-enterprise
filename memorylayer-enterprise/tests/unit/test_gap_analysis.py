"""Unit tests for analyze_document_gaps (Phase 1 gap-fill primitive).

Uses lightweight fakes for storage + Variables — the primitive only reads
``get_pages`` / ``get_memory_source_page_ids`` and the effective flags, so no DB
is needed.
"""
from unittest.mock import MagicMock

import pytest

from memorylayer_saas.models.document import (
    Document,
    DocumentPage,
    DocumentStatus,
    DocumentType,
)
from memorylayer_saas.services.document.gap_analysis import (
    EffectiveFlags,
    analyze_document_gaps,
)


class _FakeStorage:
    def __init__(self, pages, mem_page_ids):
        self._pages = pages
        self._mem_page_ids = set(mem_page_ids)

    async def get_pages(self, document_id, workspace_id=None):
        return list(self._pages)

    async def get_memory_source_page_ids(self, workspace_id, document_id):
        return set(self._mem_page_ids)


def _make_v():
    """Variables whose environ returns the default (transcribe on, others off)."""
    v = MagicMock()
    v.environ = MagicMock(side_effect=lambda key, default=None, **kw: default)
    return v


def _make_doc(page_count, metadata=None):
    return Document(
        id="doc1",
        workspace_id="ws",
        filename="report.pdf",
        document_type=DocumentType.PDF,
        content_hash="h",
        size_bytes=1,
        status=DocumentStatus.PROCESSING,
        page_count=page_count,
        metadata=metadata or {},
    )


def _page(page_id, page_no, *, transcript=None, image_path="img", multivector=None,
          embedding=None, visual_tokens=None):
    return DocumentPage(
        id=page_id,
        document_id="doc1",
        workspace_id="ws",
        page_no=page_no,
        image_storage_path=image_path,
        transcript=transcript,
        embedding=embedding,
        multivector=multivector,
        visual_tokens=visual_tokens,
        metadata={},
    )


# Default flags used by most tests: transcribe ON, chat-ingest OFF, vt OFF.
_FLAGS_DEFAULT = EffectiveFlags(transcribe=True, chat_ingest=False, visual_tokenizer=False)


@pytest.mark.asyncio
async def test_complete_document_is_complete():
    pages = [
        _page("p0", 0, transcript="hi", multivector=[[1.0]], embedding=[0.1]),
        _page("p1", 1, transcript="yo", multivector=[[1.0]], embedding=[0.2]),
    ]
    storage = _FakeStorage(pages, {"p0", "p1"})
    doc = _make_doc(2)

    gaps = await analyze_document_gaps(_make_v(), storage, doc, flags=_FLAGS_DEFAULT)

    assert gaps.is_complete is True
    assert gaps.first_missing_phase is None
    assert gaps.missing_render is False
    assert gaps.pages_missing_memory == []


@pytest.mark.asyncio
async def test_missing_memory_on_a_page_is_store_gap():
    pages = [
        _page("p0", 0, transcript="hi", multivector=[[1.0]], embedding=[0.1]),
        _page("p1", 1, transcript="yo", multivector=[[1.0]], embedding=[0.2]),
    ]
    # Only p0 has a memory.
    storage = _FakeStorage(pages, {"p0"})
    doc = _make_doc(2)

    gaps = await analyze_document_gaps(_make_v(), storage, doc, flags=_FLAGS_DEFAULT)

    assert gaps.is_complete is False
    assert gaps.first_missing_phase == "store"
    assert gaps.pages_missing_memory == ["p1"]


@pytest.mark.asyncio
async def test_no_pages_is_render_gap():
    storage = _FakeStorage([], set())
    doc = _make_doc(0)

    gaps = await analyze_document_gaps(_make_v(), storage, doc, flags=_FLAGS_DEFAULT)

    assert gaps.has_pages is False
    assert gaps.missing_render is True
    assert gaps.first_missing_phase == "render"
    assert gaps.is_complete is False


@pytest.mark.asyncio
async def test_fewer_pages_than_expected_is_render_gap():
    pages = [_page("p0", 0, transcript="hi", multivector=[[1.0]], embedding=[0.1])]
    storage = _FakeStorage(pages, {"p0"})
    doc = _make_doc(3)  # expected 3, only 1 rendered

    gaps = await analyze_document_gaps(_make_v(), storage, doc, flags=_FLAGS_DEFAULT)

    assert gaps.missing_render is True
    assert gaps.first_missing_phase == "render"


@pytest.mark.asyncio
async def test_missing_transcript_is_transcribe_gap():
    pages = [
        _page("p0", 0, transcript=None, multivector=[[1.0]]),
        _page("p1", 1, transcript="yo", multivector=[[1.0]], embedding=[0.2]),
    ]
    storage = _FakeStorage(pages, {"p1"})
    doc = _make_doc(2)

    gaps = await analyze_document_gaps(_make_v(), storage, doc, flags=_FLAGS_DEFAULT)

    assert gaps.pages_missing_transcript == ["p0"]
    assert gaps.first_missing_phase == "transcribe"
    assert gaps.is_complete is False


@pytest.mark.asyncio
async def test_missing_text_embedding_is_embed_gap():
    pages = [
        _page("p0", 0, transcript="hi", multivector=[[1.0]], embedding=None),
    ]
    storage = _FakeStorage(pages, {"p0"})
    doc = _make_doc(1)

    gaps = await analyze_document_gaps(_make_v(), storage, doc, flags=_FLAGS_DEFAULT)

    assert gaps.pages_missing_text_embedding == ["p0"]
    assert gaps.first_missing_phase == "embed"


@pytest.mark.asyncio
async def test_missing_multivector_is_embed_gap():
    pages = [
        _page("p0", 0, transcript="hi", multivector=None, embedding=[0.1]),
    ]
    storage = _FakeStorage(pages, {"p0"})
    doc = _make_doc(1)

    gaps = await analyze_document_gaps(_make_v(), storage, doc, flags=_FLAGS_DEFAULT)

    assert gaps.pages_missing_multivector == ["p0"]
    assert gaps.first_missing_phase == "embed"


@pytest.mark.asyncio
async def test_transcribe_disabled_skips_transcript_gap():
    # transcribe OFF + chat_ingest ON → missing transcript is NOT a transcribe
    # gap; image-embeds are required instead.
    flags = EffectiveFlags(transcribe=False, chat_ingest=True, visual_tokenizer=False)
    pages = [
        _page("p0", 0, transcript="text-from-chat", multivector=[[1.0]],
              embedding=[0.1], visual_tokens={"slug": {}}),
    ]
    storage = _FakeStorage(pages, {"p0"})
    doc = _make_doc(1)

    gaps = await analyze_document_gaps(_make_v(), storage, doc, flags=flags)

    assert gaps.pages_missing_transcript == []
    assert gaps.pages_missing_image_embed == []
    assert gaps.is_complete is True


@pytest.mark.asyncio
async def test_image_embed_required_when_visual_tokenizer_on():
    flags = EffectiveFlags(transcribe=True, chat_ingest=False, visual_tokenizer=True)
    pages = [
        _page("p0", 0, transcript="hi", multivector=[[1.0]], embedding=[0.1],
              visual_tokens=None),  # no image embed
    ]
    storage = _FakeStorage(pages, {"p0"})
    doc = _make_doc(1)

    gaps = await analyze_document_gaps(_make_v(), storage, doc, flags=flags)

    assert gaps.pages_missing_image_embed == ["p0"]
    assert gaps.first_missing_phase == "embed"


@pytest.mark.asyncio
async def test_flags_pinned_on_metadata_take_precedence():
    # Config (via _make_v default) would say transcribe ON, but the pinned flags
    # say transcribe OFF + chat_ingest ON, so a chat-supplied transcript page
    # with image-embeds is complete and a missing transcript is not a gap.
    doc = _make_doc(1, metadata={"ingest_flags": {
        "transcribe": False, "chat_ingest": True, "visual_tokenizer": False,
    }})
    pages = [
        _page("p0", 0, transcript="chat-text", multivector=[[1.0]],
              embedding=[0.1], visual_tokens={"slug": {}}),
    ]
    storage = _FakeStorage(pages, {"p0"})

    gaps = await analyze_document_gaps(_make_v(), storage, doc)  # no explicit flags

    assert gaps.flags.transcribe is False
    assert gaps.flags.chat_ingest is True
    assert gaps.is_complete is True


# ===========================================================================
# Batch analysis (analyze_documents_gaps)
# ===========================================================================

class _FakeBatchStorage:
    """Storage exposing the batch primitives, counting queries issued."""

    def __init__(self, pages_by_doc, mem_page_ids):
        self._pages_by_doc = pages_by_doc
        self._mem_page_ids = set(mem_page_ids)
        self.page_queries = 0
        self.mem_queries = 0

    async def get_pages(self, document_id, workspace_id=None):
        self.page_queries += 1
        return list(self._pages_by_doc.get(document_id, []))

    async def get_memory_source_page_ids(self, workspace_id, document_id):
        self.mem_queries += 1
        return set(self._mem_page_ids)

    async def get_pages_for_documents(self, document_ids, workspace_id=None,
                                      limit=None, offset=0):
        self.page_queries += 1
        out = []
        for doc_id in document_ids:
            out.extend(self._pages_by_doc.get(doc_id, []))
        return out

    async def get_memory_source_page_ids_for_documents(self, workspace_id, document_ids):
        self.mem_queries += 1
        return set(self._mem_page_ids)


def _doc(doc_id, page_count, metadata=None):
    return Document(
        id=doc_id,
        workspace_id="ws",
        filename=f"{doc_id}.pdf",
        document_type=DocumentType.PDF,
        content_hash="h",
        size_bytes=1,
        status=DocumentStatus.PROCESSING,
        page_count=page_count,
        metadata=metadata or {},
    )


def _pg(doc_id, page_id, page_no, **kw):
    page = _page(page_id, page_no, **kw)
    page.document_id = doc_id
    return page


@pytest.mark.asyncio
async def test_batch_issues_two_queries_regardless_of_document_count():
    """The whole point: query count is constant, not one pair per document."""
    from memorylayer_saas.services.document.gap_analysis import analyze_documents_gaps

    pages = {
        f"doc{i}": [_pg(f"doc{i}", f"d{i}p0", 0, transcript="t",
                        multivector=[[1.0]], embedding=[0.1])]
        for i in range(5)
    }
    storage = _FakeBatchStorage(pages, {f"d{i}p0" for i in range(5)})
    docs = [_doc(f"doc{i}", 1) for i in range(5)]

    await analyze_documents_gaps(_make_v(), storage, docs, flags=_FLAGS_DEFAULT)

    assert storage.page_queries == 1
    assert storage.mem_queries == 1


@pytest.mark.asyncio
async def test_batch_matches_single_document_analysis():
    """Batch must not be a second, subtly different implementation."""
    from memorylayer_saas.services.document.gap_analysis import analyze_documents_gaps

    pages = {
        "doc1": [
            _pg("doc1", "p0", 0, transcript="hi", multivector=[[1.0]], embedding=[0.1]),
            _pg("doc1", "p1", 1, multivector=[[1.0]]),  # no transcript
        ],
    }
    storage = _FakeBatchStorage(pages, {"p0"})
    doc = _doc("doc1", 2)

    single = await analyze_document_gaps(_make_v(), storage, doc, flags=_FLAGS_DEFAULT)
    batch, = await analyze_documents_gaps(_make_v(), storage, [doc], flags=_FLAGS_DEFAULT)

    assert batch.is_complete == single.is_complete
    assert batch.first_missing_phase == single.first_missing_phase
    assert batch.pages_missing_transcript == single.pages_missing_transcript


@pytest.mark.asyncio
async def test_batch_attributes_pages_to_the_right_document():
    from memorylayer_saas.services.document.gap_analysis import analyze_documents_gaps

    pages = {
        "doc1": [_pg("doc1", "a0", 0, transcript="t", multivector=[[1.0]], embedding=[0.1])],
        "doc2": [_pg("doc2", "b0", 0, multivector=[[1.0]])],  # missing transcript
    }
    storage = _FakeBatchStorage(pages, {"a0"})

    results = await analyze_documents_gaps(
        _make_v(), storage, [_doc("doc1", 1), _doc("doc2", 1)], flags=_FLAGS_DEFAULT,
    )

    by_id = {g.document_id: g for g in results}
    assert by_id["doc1"].is_complete is True
    assert by_id["doc2"].pages_missing_transcript == ["b0"]
    assert by_id["doc2"].first_missing_phase == "transcribe"


@pytest.mark.asyncio
async def test_batch_preserves_input_order():
    from memorylayer_saas.services.document.gap_analysis import analyze_documents_gaps

    pages = {f"doc{i}": [] for i in (3, 1, 2)}
    storage = _FakeBatchStorage(pages, set())
    docs = [_doc("doc3", 0), _doc("doc1", 0), _doc("doc2", 0)]

    results = await analyze_documents_gaps(_make_v(), storage, docs, flags=_FLAGS_DEFAULT)

    assert [g.document_id for g in results] == ["doc3", "doc1", "doc2"]


@pytest.mark.asyncio
async def test_batch_falls_back_when_storage_lacks_batch_primitives():
    """Any backend without the new methods must keep working."""
    from memorylayer_saas.services.document.gap_analysis import analyze_documents_gaps

    storage = _FakeStorage(
        [_page("p0", 0, transcript="t", multivector=[[1.0]], embedding=[0.1])], {"p0"},
    )
    results = await analyze_documents_gaps(
        _make_v(), storage, [_make_doc(1)], flags=_FLAGS_DEFAULT,
    )

    assert len(results) == 1
    assert results[0].is_complete is True


@pytest.mark.asyncio
async def test_batch_empty_input_short_circuits():
    from memorylayer_saas.services.document.gap_analysis import analyze_documents_gaps

    storage = _FakeBatchStorage({}, set())
    assert await analyze_documents_gaps(_make_v(), storage, []) == []
    assert storage.page_queries == 0


# ===========================================================================
# Page tallies (page_count / pages_with_transcript)
# ===========================================================================

@pytest.mark.asyncio
async def test_tallies_count_actual_pages_and_transcripts():
    storage = _FakeStorage(
        [
            _page("p0", 0, transcript="hi", multivector=[[1.0]], embedding=[0.1]),
            _page("p1", 1, multivector=[[1.0]]),
            _page("p2", 2, transcript="", multivector=[[1.0]]),  # empty == no text
        ],
        {"p0"},
    )
    gaps = await analyze_document_gaps(_make_v(), storage, _make_doc(3), flags=_FLAGS_DEFAULT)

    assert gaps.page_count == 3
    assert gaps.pages_with_transcript == 1


@pytest.mark.asyncio
async def test_tallies_are_factual_even_when_transcribe_is_disabled():
    """The tallies report reality; is_complete reports the gate-aware verdict.

    With transcription off, a document with zero transcripts is COMPLETE (there
    is nothing left to run) — but pages_with_transcript must still read 0, or a
    progress display would claim text that does not exist.
    """
    flags_no_transcribe = EffectiveFlags(
        transcribe=False, chat_ingest=False, visual_tokenizer=False,
    )
    storage = _FakeStorage(
        [_page("p0", 0, multivector=[[1.0]]), _page("p1", 1, multivector=[[1.0]])],
        {"p0"},
    )
    gaps = await analyze_document_gaps(
        _make_v(), storage, _make_doc(2), flags=flags_no_transcribe,
    )

    assert gaps.pages_with_transcript == 0
    assert gaps.page_count == 2
    assert gaps.pages_missing_transcript == []
    assert gaps.is_complete is True
