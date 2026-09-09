# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the shared post-store enqueue path in document ingestion.

Phase 1 of the memory-lifecycle work routes document-ingested memories through
``MemoryService.enqueue_post_store`` so they receive the same decomposition +
enrichment lifecycle as memories created via ``remember()``.

These tests verify:
- ``DocumentIngestionService._store_as_memories`` (inline path) enqueues
  post-store once per page-with-transcript, passing the precomputed embedding.
- ``DocumentFinalizeTaskHandler`` (distributed path) delegates to the shared
  ``create_memories_for_pages`` which performs the same enqueue.
- Both create-sites build an IDENTICAL ``RememberInput`` (DRY guard), including
  the ``context_id`` ``"_default"`` -> ``None`` sentinel handling.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memorylayer_saas.models.document import (
    Document,
    DocumentExtractionOptions,
    DocumentPage,
    DocumentStatus,
    DocumentType,
)
from memorylayer_saas.services.document.ingestion_service import DocumentIngestionService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_document(target_context_id: str = "_default") -> Document:
    return Document(
        id="doc_aaa000000001",
        workspace_id="ws_test",
        filename="report.pdf",
        document_type=DocumentType.PDF,
        content_hash="deadbeef" * 8,
        size_bytes=2048,
        status=DocumentStatus.PROCESSING,
        extraction_options=DocumentExtractionOptions(
            target_context_id=target_context_id,
        ),
        target_context_id=target_context_id,
    )


def make_page(
    page_id: str = "page_001",
    page_no: int = 0,
    transcript: str | None = "Page content.",
    embedding=None,
    multivector=None,
) -> DocumentPage:
    return DocumentPage(
        id=page_id,
        document_id="doc_aaa000000001",
        workspace_id="ws_test",
        page_no=page_no,
        transcript=transcript,
        embedding=embedding,
        multivector=multivector,
    )


def make_service(memory_service):
    storage = AsyncMock()
    storage.create_memory.side_effect = lambda **kw: MagicMock(id="mem_%s" % kw["input"].metadata["page_number"])
    v = MagicMock()
    v.environ.side_effect = lambda key, default=None, **kw: default
    return DocumentIngestionService(
        v=v,
        storage_backend=storage,
        blob_storage=AsyncMock(),
        embed_client=AsyncMock(),
        task_service=AsyncMock(),
        memory_service=memory_service,
        max_file_size=10 * 1024 * 1024,
        logger=MagicMock(),
    ), storage


# ---------------------------------------------------------------------------
# _store_as_memories (inline ingestion path)
# ---------------------------------------------------------------------------

class TestStoreAsMemoriesEnqueue:
    """The inline ingestion path enqueues post-store per page-with-transcript."""

    @pytest.mark.asyncio
    async def test_enqueues_once_per_page_with_transcript(self):
        ms = AsyncMock()
        ms.enqueue_post_store.return_value = None
        service, storage = make_service(ms)

        pages = [
            make_page(page_id="page_001", page_no=0, embedding=[0.1, 0.2]),
            make_page(page_id="page_002", page_no=1, embedding=[0.3, 0.4]),
            make_page(page_id="page_003", page_no=2, transcript=None),  # skipped
        ]
        doc = make_document()

        memory_ids = await service._store_as_memories(doc, pages, job_id="job_xyz")

        assert memory_ids == ["mem_0", "mem_1"]
        # enqueue_post_store called once per transcribed page (2), not for the
        # transcript-less page.
        assert ms.enqueue_post_store.call_count == 2

    @pytest.mark.asyncio
    async def test_enqueue_passes_precomputed_embedding_and_job_id(self):
        ms = AsyncMock()
        ms.enqueue_post_store.return_value = None
        service, storage = make_service(ms)

        page = make_page(page_id="page_001", page_no=0, embedding=[0.5, 0.6, 0.7])
        doc = make_document()

        await service._store_as_memories(doc, [page], job_id="job_xyz")

        ms.enqueue_post_store.assert_called_once()
        call = ms.enqueue_post_store.call_args
        # Positional: workspace_id, memory.
        assert call.args[0] == "ws_test"
        assert call.args[1].id == "mem_0"
        # Keyword: precomputed embedding + job_id.
        assert call.kwargs["embedding"] == [0.5, 0.6, 0.7]
        assert call.kwargs["job_id"] == "job_xyz"

    @pytest.mark.asyncio
    async def test_no_job_id_passes_none(self):
        ms = AsyncMock()
        ms.enqueue_post_store.return_value = None
        service, storage = make_service(ms)

        page = make_page(embedding=[0.1])
        await service._store_as_memories(make_document(), [page])

        assert ms.enqueue_post_store.call_args.kwargs["job_id"] is None


# ---------------------------------------------------------------------------
# document_finalize task (distributed path)
# ---------------------------------------------------------------------------

class TestDocumentFinalizeEnqueue:
    """The distributed finalize task delegates to the shared create + enqueue."""

    @pytest.mark.asyncio
    async def test_finalize_enqueues_via_shared_create(self):
        from memorylayer_saas.tasks.document_finalize import DocumentFinalizeTaskHandler

        ms = AsyncMock()
        ms.enqueue_post_store.return_value = None
        service, storage = make_service(ms)

        # Pages carry the embed-phase embedding on the dedicated 'embedding'
        # column (populated by the storage layer).
        page1 = make_page(page_id="page_001", page_no=0, embedding=[0.1, 0.2])
        page2 = make_page(page_id="page_002", page_no=1, embedding=[0.3, 0.4])
        doc = make_document()

        storage.get_document.return_value = doc
        storage.get_pages.return_value = [page1, page2]

        from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

        def ext_side_effect(ext_name, v=None):
            return {EXT_STORAGE_BACKEND: storage}[ext_name]

        payload = {
            "document_id": "doc_aaa000000001",
            "job_id": "job_xyz",
            "workspace_id": "ws_test",
        }

        with patch(
            "memorylayer_saas.tasks.document_finalize.get_extension",
            side_effect=ext_side_effect,
        ), patch(
            "memorylayer_saas.tasks.document_finalize.get_document_ingestion_service",
            return_value=service,
        ):
            handler = DocumentFinalizeTaskHandler()
            await handler.handle(MagicMock(), payload)

        # Embedding carried on each page column into create.
        assert page1.embedding == [0.1, 0.2]
        assert page2.embedding == [0.3, 0.4]
        # enqueue_post_store ran once per page, with the restored embedding +
        # threaded job_id.
        assert ms.enqueue_post_store.call_count == 2
        first = ms.enqueue_post_store.call_args_list[0]
        assert first.kwargs["embedding"] == [0.1, 0.2]
        assert first.kwargs["job_id"] == "job_xyz"


# ---------------------------------------------------------------------------
# DRY guard: both create-sites build identical RememberInput
# ---------------------------------------------------------------------------

class TestRememberInputDryGuard:
    """The inline path and the finalize task share one RememberInput builder."""

    def test_default_context_id_maps_to_none(self):
        doc = make_document(target_context_id="_default")
        page = make_page()

        result = DocumentIngestionService._build_page_memory_input(doc, page)

        # '_default' sentinel -> NULL context_id (the correct ingestion form).
        assert result.context_id is None
        assert result.content == "Page content."
        assert result.source_document_id == "doc_aaa000000001"
        assert result.source_page_id == "page_001"
        assert result.metadata["page_number"] == 0
        assert "document_ingestion" in result.tags

    def test_explicit_context_id_preserved(self):
        doc = make_document(target_context_id="ctx_custom")
        page = make_page()

        result = DocumentIngestionService._build_page_memory_input(doc, page)
        assert result.context_id == "ctx_custom"

    def test_connector_metadata_is_normalized_without_copying_source_payload(self):
        doc = make_document()
        doc.metadata = {
            "connector_type": "gdrive",
            "owners": [{"displayName": "Alice Owner", "emailAddress": "alice@example.test"}],
            "authorization_header": "Bearer must-not-copy",
        }
        page = make_page()

        result = DocumentIngestionService._build_page_memory_input(doc, page)

        assert result.metadata["knowledge_work"]["subject"]["name"] == "report.pdf"
        assert result.metadata["knowledge_work"]["owner"]["name"] == "Alice Owner"
        assert "authorization_header" not in result.metadata

    @pytest.mark.asyncio
    async def test_both_create_sites_build_identical_input(self):
        """The inline path and finalize task produce identical RememberInputs."""
        # Capture the input each create-site hands to storage.create_memory.
        inline_inputs = []
        finalize_inputs = []

        # --- inline path ---
        ms = AsyncMock()
        service_inline, storage_inline = make_service(ms)
        storage_inline.create_memory.side_effect = None

        def capture_inline(**kw):
            inline_inputs.append(kw["input"])
            return MagicMock(id="mem_x")

        storage_inline.create_memory.side_effect = capture_inline

        doc = make_document(target_context_id="_default")
        page = make_page(page_id="page_001", page_no=0, embedding=[0.1])
        await service_inline._store_as_memories(doc, [page], job_id="job_xyz")

        # --- finalize path (via the shared create_memories_for_pages) ---
        service_final, storage_final = make_service(ms)

        def capture_final(**kw):
            finalize_inputs.append(kw["input"])
            return MagicMock(id="mem_y")

        storage_final.create_memory.side_effect = capture_final

        page_f = make_page(page_id="page_001", page_no=0, embedding=[0.1])
        await service_final.create_memories_for_pages(doc, [page_f], job_id="job_xyz")

        assert len(inline_inputs) == 1
        assert len(finalize_inputs) == 1
        # model_dump gives a structural equality check across both inputs.
        assert inline_inputs[0].model_dump() == finalize_inputs[0].model_dump()
