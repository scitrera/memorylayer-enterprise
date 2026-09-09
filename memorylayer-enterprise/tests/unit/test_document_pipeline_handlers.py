"""
Unit tests for the distributed document ingestion pipeline task handlers.

Tests each phase handler in isolation with mocked services:
- document_render: renders pages, persists to DB, schedules document_transcribe
- document_transcribe: loads pages, transcribes, updates pages, schedules document_embed
- document_embed: loads pages, embeds, stores embeddings, schedules document_finalize
- document_finalize: loads pages, creates memories, updates doc/job to COMPLETED

Error handling tests verify each handler marks doc/job as FAILED without
scheduling the next phase.
"""
import logging

import pytest
from datetime import datetime, timezone
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from memorylayer_saas.models.document import (
    Document,
    DocumentExtractionOptions,
    DocumentPage,
    DocumentStatus,
    DocumentType,
    IngestionJob,
    JobStatus,
)
from memorylayer_saas.tasks.document_render import DocumentRenderTaskHandler
from memorylayer_saas.tasks.document_transcribe import DocumentTranscribeTaskHandler
from memorylayer_saas.tasks.document_embed import (
    DocumentEmbedTaskHandler,
    RetryableEmbedError,
    build_embed_retry_policy,
)
from memorylayer_saas.tasks.document_finalize import DocumentFinalizeTaskHandler

# The embed RetryPolicy the forward pipeline attaches when scheduling
# document_embed. Resolves to ``None`` when the installed aether SDK predates the
# RetryPolicy proto (version drift) and to the real proto once it is present;
# asserting against the builder keeps these tests correct across that upgrade.
_EMBED_RETRY_POLICY = build_embed_retry_policy()


# ---------------------------------------------------------------------------
# Helpers / Factories
# ---------------------------------------------------------------------------

def make_document(
    doc_id: str = "doc_aaa000000001",
    workspace_id: str = "ws_test",
    filename: str = "test.pdf",
    document_type: DocumentType = DocumentType.PDF,
    status: DocumentStatus = DocumentStatus.PENDING,
    storage_path: str = "/blobs/ws_test/documents/doc_aaa000000001/test.pdf",
    memory_ids: list[str] | None = None,
) -> Document:
    return Document(
        id=doc_id,
        workspace_id=workspace_id,
        filename=filename,
        document_type=document_type,
        content_hash="deadbeef" * 8,
        size_bytes=2048,
        status=status,
        storage_path=storage_path,
        memory_ids=memory_ids or [],
        extraction_options=DocumentExtractionOptions(),
    )


def make_job(
    job_id: str = "job_aaa000000001",
    workspace_id: str = "ws_test",
    errors: list[dict] | None = None,
) -> IngestionJob:
    return IngestionJob(
        id=job_id,
        workspace_id=workspace_id,
        document_ids=["doc_aaa000000001"],
        status=JobStatus.RUNNING,
        errors=errors or [],
    )


def make_page(
    page_id: str = "page_001",
    document_id: str = "doc_aaa000000001",
    workspace_id: str = "ws_test",
    page_no: int = 0,
    image_storage_path: str | None = "/blobs/ws_test/documents/doc_aaa000000001/pages/page_0000.png",
    transcript: str | None = None,
    multivector: list | None = None,
    embedding: list | None = None,
    metadata: dict | None = None,
) -> DocumentPage:
    return DocumentPage(
        id=page_id,
        document_id=document_id,
        workspace_id=workspace_id,
        page_no=page_no,
        image_storage_path=image_storage_path,
        transcript=transcript,
        multivector=multivector,
        embedding=embedding,
        metadata=metadata or {},
    )


PAYLOAD = {
    "document_id": "doc_aaa000000001",
    "job_id": "job_aaa000000001",
    "workspace_id": "ws_test",
}


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------

def make_mock_storage(pages=None, doc=None, job=None):
    storage = AsyncMock()
    storage.get_document.return_value = doc or make_document()
    storage.get_pages.return_value = pages or []
    # get_job is consulted by the embed phase only when it needs to append
    # non-fatal error entries; default to a job with an empty errors list so the
    # append logic can read job.errors.
    storage.get_job.return_value = job or make_job()
    storage.create_page.side_effect = lambda workspace_id, document_id, page: page
    storage.update_page.return_value = None
    storage.update_document.return_value = None
    storage.update_job.return_value = None
    storage.create_memory.return_value = MagicMock(id="mem_001")
    return storage


def make_mock_task_service():
    ts = AsyncMock()
    ts.schedule_task.return_value = None
    return ts


def make_embed_v(visual_tokenizer_enabled: bool = False, chat_ingest_enabled: bool = False):
    """Build a Variables stand-in with key-aware ``environ`` resolution.

    The embed handler reads several flags via ``v.environ(name, default=...,
    type_fn=...)``: the visual-tokenizer toggle, the document-chat ingest toggle,
    and (for chat-ingest) the prompt + max-tokens. A bare ``MagicMock`` would
    return a truthy mock for ALL of them and spuriously enable additive paths
    that have their own dedicated tests. This returns the requested booleans only
    for their specific flags and falls through to the caller-supplied ``default``
    for everything else (prompt, max_tokens), honoring the production defaults
    (both features disabled) unless a test opts in.
    """
    from memorylayer_saas.config import (
        MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED,
        MEMORYLAYER_VISUAL_TOKENIZER_ENABLED,
    )

    def _environ(name, default=None, **kw):
        if name == MEMORYLAYER_VISUAL_TOKENIZER_ENABLED:
            return visual_tokenizer_enabled
        if name == MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED:
            return chat_ingest_enabled
        return default

    v = MagicMock()
    v.environ.side_effect = _environ
    return v


def make_mock_embed_client(transcript_result=None, embeddings=None, mv_results=None):
    embed = AsyncMock()
    embed.connect.return_value = None
    embed.close.return_value = None
    # MUST match the embed server's real TranscriptionResponse shape
    # ("results"/"page_index"/"model_used"/"success"). An earlier mock used
    # "pages"/"page_number"/"model", which the server has never emitted -- the
    # callers dug for those keys, found nothing, and silently transcribed zero
    # pages while this suite stayed green.
    embed.transcribe_pages.return_value = transcript_result or {
        "results": [
            {
                "page_index": 0,
                "content": "Hello world.",
                "success": True,
                "model_used": "vlm-v1",
                "provider_used": "glm-ocr",
                "attempts": [],
            }
        ],
        "stats": {"total_pages": 1, "successful_pages": 1, "failed_pages": 0},
    }
    embed.embed_texts.return_value = embeddings or [[0.1, 0.2, 0.3]]
    embed.embed_images_multivector.return_value = mv_results or [
        {"vectors": [[0.1, 0.2], [0.3, 0.4]], "num_vectors": 2}
    ]
    return embed


def make_mock_blob_storage():
    blob = AsyncMock()
    blob.retrieve_file.return_value = b"fake-image-bytes"
    blob.store_file.return_value = "/blobs/stored"
    blob.page_transcript_path.return_value = (
        "/blobs/ws_test/documents/doc_aaa000000001/transcripts/page_0000.md"
    )
    blob.page_image_path.return_value = (
        "/blobs/ws_test/documents/doc_aaa000000001/pages/page_0000.png"
    )
    return blob


def make_mock_ingestion_service(logger=None, created_memory_ids=None):
    svc = MagicMock()
    svc.logger = logger or MagicMock()
    svc._render_pages = AsyncMock(return_value=[
        make_page(page_id="page_001", image_storage_path="/blobs/.../page_0000.png")
    ])
    svc._finalize = AsyncMock(return_value=None)
    # The finalize task delegates per-page create + post-store enrichment to the
    # shared create_memories_for_pages on the ingestion service.
    svc.create_memories_for_pages = AsyncMock(
        return_value=created_memory_ids if created_memory_ids is not None else ["mem_001"]
    )
    return svc


# ---------------------------------------------------------------------------
# Patching helpers
# ---------------------------------------------------------------------------

@contextmanager
def patch_render_handler(storage=None, task_service=None, ingestion_service=None):
    """Patch all dependencies for DocumentRenderTaskHandler."""
    storage = storage or make_mock_storage()
    task_service = task_service or make_mock_task_service()
    ingestion_service = ingestion_service or make_mock_ingestion_service()

    from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
    from memorylayer_server.services.tasks.base import EXT_TASK_SERVICE

    def ext_side_effect(ext_name, v=None):
        return {
            EXT_STORAGE_BACKEND: storage,
            EXT_TASK_SERVICE: task_service,
        }[ext_name]

    with patch("memorylayer_saas.tasks.document_render.get_extension", side_effect=ext_side_effect), \
         patch("memorylayer_saas.tasks.document_render.get_document_ingestion_service",
               return_value=ingestion_service):
        yield storage, task_service, ingestion_service


@contextmanager
def patch_transcribe_handler(
    storage=None, task_service=None, embed_client=None,
    blob_storage=None, ingestion_service=None,
):
    """Patch all dependencies for DocumentTranscribeTaskHandler."""
    storage = storage or make_mock_storage()
    task_service = task_service or make_mock_task_service()
    embed_client = embed_client or make_mock_embed_client()
    blob_storage = blob_storage or make_mock_blob_storage()
    ingestion_service = ingestion_service or make_mock_ingestion_service()

    from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
    from memorylayer_server.services.tasks.base import EXT_TASK_SERVICE
    from memorylayer_saas.services.document import (
        EXT_EMBED_SERVER_CLIENT,
        EXT_BLOB_STORAGE_SERVICE,
    )
    from memorylayer_saas.services.transcription import (
        EXT_TRANSCRIPTION_SERVICE,
        EmbedServerTranscriptionService,
    )

    # Wrap the mock embed client in the REAL embed-server transcription service
    # rather than mocking the service outright, so the wire-shape normalization
    # stays under test end-to-end -- that seam is exactly where the
    # "pages"/"page_number" drift silently discarded every transcript.
    transcription = EmbedServerTranscriptionService(embed_client, logging.getLogger("test"))

    def ext_side_effect(ext_name, v=None):
        return {
            EXT_STORAGE_BACKEND: storage,
            EXT_TASK_SERVICE: task_service,
            EXT_EMBED_SERVER_CLIENT: embed_client,
            EXT_TRANSCRIPTION_SERVICE: transcription,
            EXT_BLOB_STORAGE_SERVICE: blob_storage,
        }[ext_name]

    with patch("memorylayer_saas.tasks.document_transcribe.get_extension", side_effect=ext_side_effect), \
         patch("memorylayer_saas.tasks.document_transcribe.get_document_ingestion_service",
               return_value=ingestion_service):
        yield storage, task_service, embed_client, blob_storage, ingestion_service


@contextmanager
def patch_embed_handler(
    storage=None, task_service=None, embed_client=None,
    blob_storage=None, ingestion_service=None,
):
    """Patch all dependencies for DocumentEmbedTaskHandler."""
    storage = storage or make_mock_storage()
    task_service = task_service or make_mock_task_service()
    embed_client = embed_client or make_mock_embed_client()
    blob_storage = blob_storage or make_mock_blob_storage()
    ingestion_service = ingestion_service or make_mock_ingestion_service()

    from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
    from memorylayer_server.services.tasks.base import EXT_TASK_SERVICE
    from memorylayer_saas.services.document import (
        EXT_EMBED_SERVER_CLIENT,
        EXT_BLOB_STORAGE_SERVICE,
    )

    def ext_side_effect(ext_name, v=None):
        return {
            EXT_STORAGE_BACKEND: storage,
            EXT_TASK_SERVICE: task_service,
            EXT_EMBED_SERVER_CLIENT: embed_client,
            EXT_BLOB_STORAGE_SERVICE: blob_storage,
        }[ext_name]

    with patch("memorylayer_saas.tasks.document_embed.get_extension", side_effect=ext_side_effect), \
         patch("memorylayer_saas.tasks.document_embed.get_document_ingestion_service",
               return_value=ingestion_service):
        yield storage, task_service, embed_client, blob_storage, ingestion_service


@contextmanager
def patch_finalize_handler(storage=None, ingestion_service=None):
    """Patch all dependencies for DocumentFinalizeTaskHandler."""
    storage = storage or make_mock_storage()
    ingestion_service = ingestion_service or make_mock_ingestion_service()

    from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

    def ext_side_effect(ext_name, v=None):
        return {EXT_STORAGE_BACKEND: storage}[ext_name]

    with patch("memorylayer_saas.tasks.document_finalize.get_extension", side_effect=ext_side_effect), \
         patch("memorylayer_saas.tasks.document_finalize.get_document_ingestion_service",
               return_value=ingestion_service):
        yield storage, ingestion_service


# ---------------------------------------------------------------------------
# DocumentRenderTaskHandler tests
# ---------------------------------------------------------------------------

class TestDocumentRenderTaskHandler:
    """Tests for the document_render task handler."""

    def test_get_task_type(self):
        handler = DocumentRenderTaskHandler()
        assert handler.get_task_type() == "document_render"

    def test_get_schedule_is_none(self):
        handler = DocumentRenderTaskHandler()
        assert handler.get_schedule(MagicMock()) is None

    @pytest.mark.asyncio
    async def test_happy_path_marks_processing_and_schedules_transcribe(self):
        """Render handler marks doc as PROCESSING and schedules document_transcribe."""
        rendered_page = make_page(
            page_id="page_001",
            image_storage_path="/blobs/.../page_0000.png",
        )
        ingestion_svc = make_mock_ingestion_service()
        ingestion_svc._render_pages = AsyncMock(return_value=[rendered_page])
        storage = make_mock_storage()

        with patch_render_handler(storage=storage, ingestion_service=ingestion_svc) as (
            storage, task_service, _
        ):
            handler = DocumentRenderTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        # Document updated to PROCESSING
        update_calls = [c.args for c in storage.update_document.call_args_list]
        assert any(
            args[0] == "doc_aaa000000001" and "status" in storage.update_document.call_args_list[i].kwargs
            for i, args in enumerate(update_calls)
        )
        processing_calls = [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.PROCESSING.value
        ]
        assert len(processing_calls) == 1

        # Page persisted
        storage.create_page.assert_called_once()
        # Progress updated to 20
        storage.update_job.assert_any_call("job_aaa000000001", progress_percent=20)
        # Next phase scheduled (transcribe path carries no embed retry policy)
        task_service.schedule_task.assert_called_once_with(
            "document_transcribe",
            {"document_id": "doc_aaa000000001", "job_id": "job_aaa000000001", "workspace_id": "ws_test"},
            priority=3,
            retry_policy=None,
        )

    @pytest.mark.asyncio
    async def test_renders_multiple_pages(self):
        """Render handler persists all rendered pages."""
        pages = [
            make_page(page_id="page_001", page_no=0),
            make_page(page_id="page_002", page_no=1),
            make_page(page_id="page_003", page_no=2),
        ]
        ingestion_svc = make_mock_ingestion_service()
        ingestion_svc._render_pages = AsyncMock(return_value=pages)
        storage = make_mock_storage()

        with patch_render_handler(storage=storage, ingestion_service=ingestion_svc) as (
            storage, task_service, _
        ):
            handler = DocumentRenderTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        assert storage.create_page.call_count == 3
        storage.update_document.assert_any_call("doc_aaa000000001", page_count=3)

    @pytest.mark.asyncio
    async def test_error_marks_failed_no_next_phase(self):
        """On render failure, doc and job are marked FAILED; no next phase scheduled."""
        ingestion_svc = make_mock_ingestion_service()
        ingestion_svc._render_pages = AsyncMock(
            side_effect=RuntimeError("PDF render failed")
        )
        storage = make_mock_storage()

        with patch_render_handler(storage=storage, ingestion_service=ingestion_svc) as (
            storage, task_service, _
        ):
            handler = DocumentRenderTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        # Document and job marked FAILED
        failed_doc_calls = [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.FAILED.value
        ]
        assert len(failed_doc_calls) == 1

        failed_job_calls = [
            c for c in storage.update_job.call_args_list
            if c.kwargs.get("status") == JobStatus.FAILED.value
        ]
        assert len(failed_job_calls) == 1
        assert failed_job_calls[0].kwargs["errors"] == [
            {"document_id": "doc_aaa000000001", "error": "PDF render failed"}
        ]
        # No next phase
        task_service.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_document_not_found(self):
        """Missing document causes FAILED state without scheduling next phase."""
        storage = make_mock_storage()
        storage.get_document.return_value = None
        ingestion_svc = make_mock_ingestion_service()

        with patch_render_handler(storage=storage, ingestion_service=ingestion_svc) as (
            storage, task_service, _
        ):
            handler = DocumentRenderTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        failed_calls = [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.FAILED.value
        ]
        assert len(failed_calls) == 1
        task_service.schedule_task.assert_not_called()


# ---------------------------------------------------------------------------
# DocumentTranscribeTaskHandler tests
# ---------------------------------------------------------------------------

class TestDocumentTranscribeTaskHandler:
    """Tests for the document_transcribe task handler."""

    def test_get_task_type(self):
        handler = DocumentTranscribeTaskHandler()
        assert handler.get_task_type() == "document_transcribe"

    def test_get_schedule_is_none(self):
        handler = DocumentTranscribeTaskHandler()
        assert handler.get_schedule(MagicMock()) is None

    @pytest.mark.asyncio
    async def test_happy_path_transcribes_and_schedules_embed(self):
        """Transcribe handler updates pages with transcripts and schedules document_embed."""
        page = make_page(
            page_id="page_001",
            image_storage_path="/blobs/.../page_0000.png",
            transcript=None,
        )
        storage = make_mock_storage(pages=[page], doc=make_document())
        embed_client = make_mock_embed_client(transcript_result={
            "results": [
                {
                    "page_index": 0,
                    "content": "Transcribed.",
                    "success": True,
                    "model_used": "vlm-v1",
                    "provider_used": "glm-ocr",
                    "attempts": [],
                }
            ],
            "stats": {"total_pages": 1, "successful_pages": 1, "failed_pages": 0},
        })
        blob_storage = make_mock_blob_storage()
        ingestion_svc = make_mock_ingestion_service()

        with patch_transcribe_handler(
            storage=storage,
            embed_client=embed_client,
            blob_storage=blob_storage,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, embed_client, blob_storage, _):
            handler = DocumentTranscribeTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        # Transcript stored to blob
        blob_storage.store_file.assert_called()
        # Page updated with transcript
        storage.update_page.assert_any_call(
            "page_001",
            transcript="Transcribed.",
            transcript_model="vlm-v1",
        )
        # Progress and next phase (embed carries the retry policy)
        storage.update_job.assert_any_call("job_aaa000000001", progress_percent=40)
        task_service.schedule_task.assert_called_once_with(
            "document_embed",
            {"document_id": "doc_aaa000000001", "job_id": "job_aaa000000001", "workspace_id": "ws_test"},
            priority=3,
            retry_policy=_EMBED_RETRY_POLICY,
        )

    @pytest.mark.asyncio
    async def test_text_page_with_existing_transcript_updated(self):
        """Text pages that already have transcripts are still persisted via update_page."""
        page = make_page(
            page_id="page_001",
            image_storage_path=None,
            transcript="Pre-existing text.",
        )
        storage = make_mock_storage(pages=[page], doc=make_document(
            document_type=DocumentType.TEXT
        ))
        ingestion_svc = make_mock_ingestion_service()

        with patch_transcribe_handler(
            storage=storage, ingestion_service=ingestion_svc
        ) as (storage, task_service, _, _, _):
            handler = DocumentTranscribeTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        # update_page called for the pre-existing transcript page
        storage.update_page.assert_called()
        # Next phase still scheduled (embed carries the retry policy)
        task_service.schedule_task.assert_called_once_with(
            "document_embed",
            {"document_id": "doc_aaa000000001", "job_id": "job_aaa000000001", "workspace_id": "ws_test"},
            priority=3,
            retry_policy=_EMBED_RETRY_POLICY,
        )

    @pytest.mark.asyncio
    async def test_error_marks_failed_no_next_phase(self):
        """Transcription failure marks doc/job as FAILED without scheduling document_embed."""
        page = make_page(image_storage_path="/blobs/.../page_0000.png", transcript=None)
        storage = make_mock_storage(pages=[page], doc=make_document())
        embed_client = make_mock_embed_client()
        embed_client.transcribe_pages = AsyncMock(
            side_effect=ConnectionError("Embed server offline")
        )
        ingestion_svc = make_mock_ingestion_service()

        with patch_transcribe_handler(
            storage=storage,
            embed_client=embed_client,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, _, _, _):
            handler = DocumentTranscribeTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        failed_doc_calls = [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.FAILED.value
        ]
        assert len(failed_doc_calls) == 1

        failed_job_calls = [
            c for c in storage.update_job.call_args_list
            if c.kwargs.get("status") == JobStatus.FAILED.value
        ]
        assert len(failed_job_calls) == 1
        assert failed_job_calls[0].kwargs["errors"] == [
            {"document_id": "doc_aaa000000001", "error": "Embed server offline"}
        ]
        task_service.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_no_pages(self):
        """Missing pages raises an error and marks doc/job as FAILED."""
        storage = make_mock_storage(pages=[], doc=make_document())
        ingestion_svc = make_mock_ingestion_service()

        with patch_transcribe_handler(
            storage=storage, ingestion_service=ingestion_svc
        ) as (storage, task_service, _, _, _):
            handler = DocumentTranscribeTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        failed_calls = [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.FAILED.value
        ]
        assert len(failed_calls) == 1
        task_service.schedule_task.assert_not_called()


# ---------------------------------------------------------------------------
# DocumentEmbedTaskHandler tests
# ---------------------------------------------------------------------------

class TestDocumentEmbedTaskHandler:
    """Tests for the document_embed task handler."""

    def test_get_task_type(self):
        handler = DocumentEmbedTaskHandler()
        assert handler.get_task_type() == "document_embed"

    def test_get_schedule_is_none(self):
        handler = DocumentEmbedTaskHandler()
        assert handler.get_schedule(MagicMock()) is None

    @pytest.mark.asyncio
    async def test_happy_path_embeds_and_schedules_finalize(self):
        """Embed handler stores embeddings in page metadata and schedules document_finalize."""
        page = make_page(
            page_id="page_001",
            transcript="Hello world.",
            image_storage_path="/blobs/.../page_0000.png",
        )
        storage = make_mock_storage(pages=[page])
        embed_client = make_mock_embed_client(
            embeddings=[[0.1, 0.2, 0.3]],
            mv_results=[{"vectors": [[0.1, 0.2], [0.3, 0.4]], "num_vectors": 2}],
        )
        blob_storage = make_mock_blob_storage()
        ingestion_svc = make_mock_ingestion_service()

        with patch_embed_handler(
            storage=storage,
            embed_client=embed_client,
            blob_storage=blob_storage,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, _, _, _):
            handler = DocumentEmbedTaskHandler()
            await handler.handle(make_embed_v(), PAYLOAD)

        # Single-vec embedding written to the dedicated 'embedding' column.
        embedding_update_calls = [
            c for c in storage.update_page.call_args_list
            if "embedding" in c.kwargs
        ]
        assert len(embedding_update_calls) == 1
        assert embedding_update_calls[0].kwargs["embedding"] == [0.1, 0.2, 0.3]

        # Multivector stored
        storage.update_page.assert_any_call(
            "page_001",
            multivector=[[0.1, 0.2], [0.3, 0.4]],
        )
        # All signals succeeded → no non-fatal error entries appended to the job.
        error_job_calls = [
            c for c in storage.update_job.call_args_list
            if "errors" in c.kwargs
        ]
        assert error_job_calls == []
        storage.get_job.assert_not_called()

        # Greppable summary line emitted once.
        summary_calls = [
            c for c in ingestion_svc.logger.info.call_args_list
            if c.args and isinstance(c.args[0], str) and c.args[0].startswith("embed summary doc=")
        ]
        assert len(summary_calls) == 1

        # Progress and next phase
        storage.update_job.assert_any_call("job_aaa000000001", progress_percent=70)
        task_service.schedule_task.assert_called_once_with(
            "document_finalize",
            {"document_id": "doc_aaa000000001", "job_id": "job_aaa000000001", "workspace_id": "ws_test"},
            priority=3,
        )

    @pytest.mark.asyncio
    async def test_text_page_no_image_only_single_vector(self):
        """Text pages without images only get single-vector embeddings (no multivector call)."""
        page = make_page(
            page_id="page_001",
            transcript="Text content.",
            image_storage_path=None,  # No image
        )
        storage = make_mock_storage(pages=[page])
        embed_client = make_mock_embed_client(embeddings=[[0.5, 0.6]])
        ingestion_svc = make_mock_ingestion_service()

        with patch_embed_handler(
            storage=storage,
            embed_client=embed_client,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, embed_client, _, _):
            handler = DocumentEmbedTaskHandler()
            await handler.handle(make_embed_v(), PAYLOAD)

        # Only embed_texts called, not embed_images_multivector
        embed_client.embed_texts.assert_called_once()
        embed_client.embed_images_multivector.assert_not_called()
        task_service.schedule_task.assert_called_once_with(
            "document_finalize",
            {"document_id": "doc_aaa000000001", "job_id": "job_aaa000000001", "workspace_id": "ws_test"},
            priority=3,
        )

    @pytest.mark.asyncio
    async def test_no_transcribed_pages_skips_embedding_and_still_schedules(self):
        """Pages without transcripts skip embedding but still schedule finalize."""
        page = make_page(page_id="page_001", transcript=None)
        storage = make_mock_storage(pages=[page])
        embed_client = make_mock_embed_client()
        ingestion_svc = make_mock_ingestion_service()

        with patch_embed_handler(
            storage=storage,
            embed_client=embed_client,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, embed_client, _, _):
            handler = DocumentEmbedTaskHandler()
            await handler.handle(make_embed_v(), PAYLOAD)

        embed_client.embed_texts.assert_not_called()
        task_service.schedule_task.assert_called_once_with(
            "document_finalize",
            {"document_id": "doc_aaa000000001", "job_id": "job_aaa000000001", "workspace_id": "ws_test"},
            priority=3,
        )

    @pytest.mark.asyncio
    async def test_error_marks_failed_no_next_phase(self):
        """Embed failure marks doc/job as FAILED without scheduling document_finalize."""
        page = make_page(page_id="page_001", transcript="Text.")
        storage = make_mock_storage(pages=[page])
        embed_client = make_mock_embed_client()
        embed_client.embed_texts = AsyncMock(side_effect=RuntimeError("Embed server error"))
        ingestion_svc = make_mock_ingestion_service()

        with patch_embed_handler(
            storage=storage,
            embed_client=embed_client,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, _, _, _):
            handler = DocumentEmbedTaskHandler()
            await handler.handle(make_embed_v(), PAYLOAD)

        failed_doc_calls = [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.FAILED.value
        ]
        assert len(failed_doc_calls) == 1

        failed_job_calls = [
            c for c in storage.update_job.call_args_list
            if c.kwargs.get("status") == JobStatus.FAILED.value
        ]
        assert len(failed_job_calls) == 1
        assert failed_job_calls[0].kwargs["errors"] == [
            {"document_id": "doc_aaa000000001", "error": "Embed server error"}
        ]
        task_service.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_multivector_failure_records_non_fatal_and_still_finalizes(self):
        """A signal that fails for ALL pages appends a non_fatal job-error entry,
        logs the summary, but does NOT mark the job FAILED and still schedules
        finalize (the signal is additive)."""
        page = make_page(
            page_id="page_001",
            transcript="Hello world.",
            image_storage_path="/blobs/.../page_0000.png",
        )
        storage = make_mock_storage(pages=[page])
        embed_client = make_mock_embed_client(embeddings=[[0.1, 0.2, 0.3]])
        # Multi-vector embedding raises for the whole batch.
        embed_client.embed_images_multivector = AsyncMock(
            side_effect=RuntimeError("colpali offline")
        )
        blob_storage = make_mock_blob_storage()
        ingestion_svc = make_mock_ingestion_service()

        with patch_embed_handler(
            storage=storage,
            embed_client=embed_client,
            blob_storage=blob_storage,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, _, _, _):
            handler = DocumentEmbedTaskHandler()
            await handler.handle(make_embed_v(), PAYLOAD)

        # Non-fatal error appended to the job's existing errors list.
        error_job_calls = [
            c for c in storage.update_job.call_args_list
            if "errors" in c.kwargs
        ]
        assert len(error_job_calls) == 1
        appended = error_job_calls[0].kwargs["errors"]
        assert appended == [
            {
                "document_id": "doc_aaa000000001",
                "signal": "multivector",
                "error": "colpali offline",
                "non_fatal": True,
            }
        ]

        # Job NOT marked FAILED; document NOT marked FAILED.
        failed_job_calls = [
            c for c in storage.update_job.call_args_list
            if c.kwargs.get("status") == JobStatus.FAILED.value
        ]
        assert failed_job_calls == []
        failed_doc_calls = [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.FAILED.value
        ]
        assert failed_doc_calls == []

        # Summary line emitted; finalize still scheduled.
        summary_calls = [
            c for c in ingestion_svc.logger.info.call_args_list
            if c.args and isinstance(c.args[0], str) and c.args[0].startswith("embed summary doc=")
        ]
        assert len(summary_calls) == 1
        task_service.schedule_task.assert_called_once_with(
            "document_finalize",
            {"document_id": "doc_aaa000000001", "job_id": "job_aaa000000001", "workspace_id": "ws_test"},
            priority=3,
        )

    @pytest.mark.asyncio
    async def test_all_signals_failed_raises_retryable_and_does_not_finalize(self):
        """When embedding was attempted but EVERY signal failed (total_attempted>0,
        total_succeeded==0 — the observed embed-backend-down case: multivector=0/N
        with no text pages), the handler raises RetryableEmbedError, does NOT
        schedule finalize, and does NOT mark the doc/job FAILED (so Aether retries
        per the RetryPolicy)."""
        # Image page with NO transcript (transcription disabled → text_attempted=0),
        # so the ONLY attempted signal is the multivector — mirroring the observed
        # multivector=0/26 failure. Its per-signal guard records the failure
        # (succeeded=0) rather than propagating, so the all-signals-failed check
        # below fires.
        page = make_page(
            page_id="page_001",
            transcript=None,
            image_storage_path="/blobs/.../page_0000.png",
        )
        storage = make_mock_storage(pages=[page])
        embed_client = make_mock_embed_client()
        embed_client.embed_images_multivector = AsyncMock(
            side_effect=RuntimeError("colpali offline")
        )
        blob_storage = make_mock_blob_storage()
        ingestion_svc = make_mock_ingestion_service()

        with patch_embed_handler(
            storage=storage,
            embed_client=embed_client,
            blob_storage=blob_storage,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, _, _, _):
            handler = DocumentEmbedTaskHandler()
            with pytest.raises(RetryableEmbedError):
                await handler.handle(make_embed_v(), PAYLOAD)

        # No finalize scheduled — the phase is being retried, not completed.
        task_service.schedule_task.assert_not_called()
        # Doc/job NOT marked FAILED (the RetryableEmbedError branch re-raises
        # before the failed-marking path).
        failed_doc_calls = [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.FAILED.value
        ]
        assert failed_doc_calls == []
        failed_job_calls = [
            c for c in storage.update_job.call_args_list
            if c.kwargs.get("status") == JobStatus.FAILED.value
        ]
        assert failed_job_calls == []

    @pytest.mark.asyncio
    async def test_partial_success_does_not_raise_and_finalizes(self):
        """PARTIAL success (multivector fails but text embed succeeds → some signal
        produced >0) does NOT raise RetryableEmbedError; it proceeds to finalize
        exactly as today (records the failed signal as non-fatal)."""
        page = make_page(
            page_id="page_001",
            transcript="Hello world.",
            image_storage_path="/blobs/.../page_0000.png",
        )
        storage = make_mock_storage(pages=[page])
        embed_client = make_mock_embed_client(embeddings=[[0.1, 0.2, 0.3]])
        # Image signal fails, but text embed succeeds → total_succeeded>0.
        embed_client.embed_images_multivector = AsyncMock(
            side_effect=RuntimeError("colpali offline")
        )
        blob_storage = make_mock_blob_storage()
        ingestion_svc = make_mock_ingestion_service()

        with patch_embed_handler(
            storage=storage,
            embed_client=embed_client,
            blob_storage=blob_storage,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, _, _, _):
            handler = DocumentEmbedTaskHandler()
            # Must NOT raise.
            await handler.handle(make_embed_v(), PAYLOAD)

        # Finalize scheduled normally (partial success proceeds).
        task_service.schedule_task.assert_called_once_with(
            "document_finalize",
            {"document_id": "doc_aaa000000001", "job_id": "job_aaa000000001", "workspace_id": "ws_test"},
            priority=3,
        )

    @pytest.mark.asyncio
    async def test_non_fatal_errors_appended_to_existing_job_errors(self):
        """Non-fatal entries are appended to (not replacing) the job's existing
        errors list."""
        page = make_page(
            page_id="page_001",
            transcript="Hello world.",
            image_storage_path="/blobs/.../page_0000.png",
        )
        existing = [{"document_id": "doc_aaa000000001", "error": "prior", "non_fatal": True}]
        storage = make_mock_storage(pages=[page], job=make_job(errors=existing))
        embed_client = make_mock_embed_client(embeddings=[[0.1, 0.2, 0.3]])
        embed_client.embed_images_multivector = AsyncMock(
            side_effect=RuntimeError("colpali offline")
        )
        blob_storage = make_mock_blob_storage()
        ingestion_svc = make_mock_ingestion_service()

        with patch_embed_handler(
            storage=storage,
            embed_client=embed_client,
            blob_storage=blob_storage,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, _, _, _):
            handler = DocumentEmbedTaskHandler()
            await handler.handle(make_embed_v(), PAYLOAD)

        error_job_calls = [
            c for c in storage.update_job.call_args_list
            if "errors" in c.kwargs
        ]
        assert len(error_job_calls) == 1
        appended = error_job_calls[0].kwargs["errors"]
        assert appended[0] == existing[0]
        assert appended[1]["signal"] == "multivector"
        assert appended[1]["non_fatal"] is True

    @pytest.mark.asyncio
    async def test_image_embeds_zero_of_n_records_non_fatal(self):
        """When the visual tokenizer is enabled but stores 0/N page tensors, a
        non_fatal image_embeds entry is recorded without failing the job."""
        page = make_page(
            page_id="page_001",
            transcript="Hello world.",
            image_storage_path="/blobs/.../page_0000.png",
        )
        storage = make_mock_storage(pages=[page])
        embed_client = make_mock_embed_client(
            embeddings=[[0.1, 0.2, 0.3]],
            mv_results=[{"vectors": [[0.1, 0.2], [0.3, 0.4]], "num_vectors": 2}],
        )
        blob_storage = make_mock_blob_storage()
        ingestion_svc = make_mock_ingestion_service()

        with patch_embed_handler(
            storage=storage,
            embed_client=embed_client,
            blob_storage=blob_storage,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, _, _, _), \
             patch(
                 "memorylayer_saas.tasks.document_embed.precompute_and_store_image_embeds",
                 new=AsyncMock(return_value=0),
             ):
            handler = DocumentEmbedTaskHandler()
            await handler.handle(make_embed_v(visual_tokenizer_enabled=True), PAYLOAD)

        error_job_calls = [
            c for c in storage.update_job.call_args_list
            if "errors" in c.kwargs
        ]
        assert len(error_job_calls) == 1
        appended = error_job_calls[0].kwargs["errors"]
        assert appended == [
            {
                "document_id": "doc_aaa000000001",
                "signal": "image_embeds",
                "error": "produced 0/1 image_embeds embeddings",
                "non_fatal": True,
            }
        ]
        # Job not FAILED; finalize still scheduled.
        failed_job_calls = [
            c for c in storage.update_job.call_args_list
            if c.kwargs.get("status") == JobStatus.FAILED.value
        ]
        assert failed_job_calls == []
        task_service.schedule_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_pages_with_existing_multivector(self):
        """Re-run/gap-fill: a page already carrying a multivector is NOT re-embedded
        (embed_images_multivector not called); a fresh page IS embedded."""
        done = make_page(
            page_id="page_done",
            transcript="Done.",
            image_storage_path="/blobs/.../page_0000.png",
            multivector=[[9.0, 9.0]],  # already embedded
        )
        fresh = make_page(
            page_id="page_fresh",
            page_no=1,
            transcript="Fresh.",
            image_storage_path="/blobs/.../page_0001.png",
            multivector=None,  # needs embedding
        )
        storage = make_mock_storage(pages=[done, fresh])
        embed_client = make_mock_embed_client(
            embeddings=[[0.1], [0.2]],
            mv_results=[{"vectors": [[0.3, 0.4]], "num_vectors": 1}],
        )
        blob_storage = make_mock_blob_storage()
        ingestion_svc = make_mock_ingestion_service()

        with patch_embed_handler(
            storage=storage, embed_client=embed_client,
            blob_storage=blob_storage, ingestion_service=ingestion_svc,
        ) as (storage, task_service, embed_client, _, _):
            handler = DocumentEmbedTaskHandler()
            await handler.handle(make_embed_v(), PAYLOAD)

        # Only the fresh page was sent for multivector embedding.
        embed_client.embed_images_multivector.assert_called_once()
        mv_update_calls = [
            c for c in storage.update_page.call_args_list
            if "multivector" in c.kwargs
        ]
        assert len(mv_update_calls) == 1
        assert mv_update_calls[0].args[0] == "page_fresh"
        task_service.schedule_task.assert_called_once_with(
            "document_finalize",
            {"document_id": "doc_aaa000000001", "job_id": "job_aaa000000001", "workspace_id": "ws_test"},
            priority=3,
        )

    @pytest.mark.asyncio
    async def test_all_multivectors_present_skips_embed_call(self):
        """When every image page already has a multivector, the multivector embed
        call is skipped entirely (pure perf, no non-fatal error)."""
        done = make_page(
            page_id="page_done",
            transcript="Done.",
            image_storage_path="/blobs/.../page_0000.png",
            multivector=[[9.0, 9.0]],
            embedding=[0.7],  # also already single-vec embedded
        )
        storage = make_mock_storage(pages=[done])
        embed_client = make_mock_embed_client()
        blob_storage = make_mock_blob_storage()
        ingestion_svc = make_mock_ingestion_service()

        with patch_embed_handler(
            storage=storage, embed_client=embed_client,
            blob_storage=blob_storage, ingestion_service=ingestion_svc,
        ) as (storage, task_service, embed_client, _, _):
            handler = DocumentEmbedTaskHandler()
            await handler.handle(make_embed_v(), PAYLOAD)

        embed_client.embed_images_multivector.assert_not_called()
        # No non-fatal error recorded for multivector (it was skipped, not 0/N).
        error_job_calls = [
            c for c in storage.update_job.call_args_list if "errors" in c.kwargs
        ]
        assert error_job_calls == []
        task_service.schedule_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_pages_with_existing_single_vec_embedding(self):
        """Re-run/gap-fill: a page that already has an 'embedding' is NOT
        re-embedded (only the fresh page's text is sent + stored)."""
        done = make_page(
            page_id="page_done",
            transcript="Done.",
            image_storage_path=None,
            embedding=[0.5, 0.6],  # already single-vec embedded
        )
        fresh = make_page(
            page_id="page_fresh",
            page_no=1,
            transcript="Fresh.",
            image_storage_path=None,
            embedding=None,  # needs single-vec embedding
        )
        storage = make_mock_storage(pages=[done, fresh])
        embed_client = make_mock_embed_client(embeddings=[[0.9, 1.0]])
        ingestion_svc = make_mock_ingestion_service()

        with patch_embed_handler(
            storage=storage, embed_client=embed_client,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, embed_client, _, _):
            handler = DocumentEmbedTaskHandler()
            await handler.handle(make_embed_v(), PAYLOAD)

        # Only the fresh page's transcript was embedded.
        embed_client.embed_texts.assert_called_once()
        assert embed_client.embed_texts.call_args.args[0] == ["Fresh."]
        # Only the fresh page got an 'embedding' column write.
        emb_update_calls = [
            c for c in storage.update_page.call_args_list
            if "embedding" in c.kwargs
        ]
        assert len(emb_update_calls) == 1
        assert emb_update_calls[0].args[0] == "page_fresh"
        task_service.schedule_task.assert_called_once()


class TestDocumentEmbedChatIngest:
    """Tests for the document-chat ingestion path of the embed handler.

    When OCR transcription is disabled (pages have no transcript) and
    document-chat ingestion is enabled, the embed handler generates per-page
    memory text from the page's image_embeds via the inference LLM, persists it
    as the page transcript, then single-vector embeds it.
    """

    @pytest.mark.asyncio
    async def test_generates_transcript_then_single_vector_embeds(self):
        """Chat-ingest fills a missing transcript, then it gets single-vector embedded."""
        page = make_page(
            page_id="page_001",
            transcript=None,
            image_storage_path="/blobs/.../page_0000.png",
        )
        storage = make_mock_storage(pages=[page])
        embed_client = make_mock_embed_client(
            embeddings=[[0.1, 0.2, 0.3]],
            mv_results=[{"vectors": [[0.1, 0.2]], "num_vectors": 1}],
        )
        blob_storage = make_mock_blob_storage()
        ingestion_svc = make_mock_ingestion_service()
        inference = AsyncMock()

        with patch_embed_handler(
            storage=storage,
            embed_client=embed_client,
            blob_storage=blob_storage,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, _, _, _), \
             patch(
                 "memorylayer_saas.tasks.document_embed.precompute_and_store_image_embeds",
                 new=AsyncMock(return_value=1),
             ), \
             patch(
                 "memorylayer_saas.tasks.document_embed.get_inference_client",
                 new=AsyncMock(return_value=inference),
             ), \
             patch(
                 "memorylayer_saas.tasks.document_embed.generate_page_text_from_image_embeds",
                 new=AsyncMock(return_value="# Page 1\n\nGenerated content."),
             ) as gen:
            handler = DocumentEmbedTaskHandler()
            await handler.handle(make_embed_v(chat_ingest_enabled=True), PAYLOAD)

        # Page text generated from image_embeds and persisted as the transcript.
        gen.assert_called_once()
        transcript_updates = [
            c for c in storage.update_page.call_args_list
            if c.kwargs.get("transcript") == "# Page 1\n\nGenerated content."
        ]
        assert len(transcript_updates) == 1

        # The generated transcript was then single-vector embedded (written to the
        # 'embedding' column) — proving the text flows downstream.
        embed_client.embed_texts.assert_called_once()
        embedding_updates = [
            c for c in storage.update_page.call_args_list
            if "embedding" in c.kwargs
        ]
        assert len(embedding_updates) == 1

        # No non-fatal errors; finalize scheduled.
        error_job_calls = [c for c in storage.update_job.call_args_list if "errors" in c.kwargs]
        assert error_job_calls == []
        task_service.schedule_task.assert_called_once_with(
            "document_finalize",
            {"document_id": "doc_aaa000000001", "job_id": "job_aaa000000001", "workspace_id": "ws_test"},
            priority=3,
        )

    @pytest.mark.asyncio
    async def test_missing_embeds_skips_page_records_non_fatal(self):
        """A page with no image_embeds yields no text → non-fatal chat_ingest entry,
        no transcript, no text embedding, but finalize still scheduled."""
        page = make_page(
            page_id="page_001",
            transcript=None,
            image_storage_path="/blobs/.../page_0000.png",
        )
        storage = make_mock_storage(pages=[page])
        embed_client = make_mock_embed_client(
            mv_results=[{"vectors": [[0.1, 0.2]], "num_vectors": 1}],
        )
        blob_storage = make_mock_blob_storage()
        ingestion_svc = make_mock_ingestion_service()
        inference = AsyncMock()

        with patch_embed_handler(
            storage=storage,
            embed_client=embed_client,
            blob_storage=blob_storage,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, _, _, _), \
             patch(
                 "memorylayer_saas.tasks.document_embed.precompute_and_store_image_embeds",
                 new=AsyncMock(return_value=1),
             ), \
             patch(
                 "memorylayer_saas.tasks.document_embed.get_inference_client",
                 new=AsyncMock(return_value=inference),
             ), \
             patch(
                 "memorylayer_saas.tasks.document_embed.generate_page_text_from_image_embeds",
                 new=AsyncMock(return_value=None),  # no image_embeds for this page
             ):
            handler = DocumentEmbedTaskHandler()
            await handler.handle(make_embed_v(chat_ingest_enabled=True), PAYLOAD)

        # No transcript persisted; no text embedding attempted.
        transcript_updates = [
            c for c in storage.update_page.call_args_list if "transcript" in c.kwargs
        ]
        assert transcript_updates == []
        embed_client.embed_texts.assert_not_called()

        # Non-fatal chat_ingest entry recorded (1/1 attempted, 0 succeeded).
        error_job_calls = [c for c in storage.update_job.call_args_list if "errors" in c.kwargs]
        assert len(error_job_calls) == 1
        assert error_job_calls[0].kwargs["errors"] == [
            {
                "document_id": "doc_aaa000000001",
                "signal": "chat_ingest",
                "error": "produced 0/1 chat_ingest embeddings",
                "non_fatal": True,
            }
        ]
        # Job not FAILED; finalize still scheduled.
        failed_job_calls = [
            c for c in storage.update_job.call_args_list
            if c.kwargs.get("status") == JobStatus.FAILED.value
        ]
        assert failed_job_calls == []
        task_service.schedule_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_disabled_does_not_generate(self):
        """With chat-ingest disabled, no text is generated for transcript-less pages."""
        page = make_page(page_id="page_001", transcript=None, image_storage_path="/img.png")
        storage = make_mock_storage(pages=[page])
        embed_client = make_mock_embed_client(
            mv_results=[{"vectors": [[0.1, 0.2]], "num_vectors": 1}],
        )
        ingestion_svc = make_mock_ingestion_service()

        with patch_embed_handler(
            storage=storage,
            embed_client=embed_client,
            ingestion_service=ingestion_svc,
        ) as (storage, task_service, _, _, _), \
             patch(
                 "memorylayer_saas.tasks.document_embed.generate_page_text_from_image_embeds",
                 new=AsyncMock(return_value="should not be used"),
             ) as gen:
            handler = DocumentEmbedTaskHandler()
            await handler.handle(make_embed_v(chat_ingest_enabled=False), PAYLOAD)

        gen.assert_not_called()
        embed_client.embed_texts.assert_not_called()
        task_service.schedule_task.assert_called_once()


# ---------------------------------------------------------------------------
# DocumentFinalizeTaskHandler tests
# ---------------------------------------------------------------------------

class TestDocumentFinalizeTaskHandler:
    """Tests for the document_finalize task handler."""

    def test_get_task_type(self):
        handler = DocumentFinalizeTaskHandler()
        assert handler.get_task_type() == "document_finalize"

    def test_get_schedule_is_none(self):
        handler = DocumentFinalizeTaskHandler()
        assert handler.get_schedule(MagicMock()) is None

    @pytest.mark.asyncio
    async def test_happy_path_creates_memories_and_finalizes(self):
        """Finalize handler delegates to the shared create + post-store path.

        The per-page create and post-store enrichment now live in
        ``create_memories_for_pages`` on the ingestion service (shared with the
        inline path). The single-vector embedding is carried on the dedicated
        ``page.embedding`` column (populated by the storage layer), so it is
        already set when the handler delegates; it then calls ``_finalize`` with
        the returned ids.
        """
        page = make_page(
            page_id="page_001",
            transcript="Document content here.",
            multivector=[[0.1, 0.2], [0.3, 0.4]],
            embedding=[0.1, 0.2, 0.3],
        )
        doc = make_document()
        storage = make_mock_storage(pages=[page], doc=doc)
        ingestion_svc = make_mock_ingestion_service(created_memory_ids=["mem_001"])

        with patch_finalize_handler(storage=storage, ingestion_service=ingestion_svc) as (
            storage, ingestion_svc
        ):
            handler = DocumentFinalizeTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        # Delegated to the shared create + post-store path with the job_id threaded.
        ingestion_svc.create_memories_for_pages.assert_called_once()
        call = ingestion_svc.create_memories_for_pages.call_args
        assert call.args[0] is doc
        assert call.args[1] == [page]
        assert call.kwargs.get("job_id") == "job_aaa000000001"
        # Embedding carried on the page column into delegation.
        assert page.embedding == [0.1, 0.2, 0.3]
        # _finalize called with doc and the returned memory ids.
        ingestion_svc._finalize.assert_called_once_with(doc, ["mem_001"], "job_aaa000000001")

    @pytest.mark.asyncio
    async def test_page_without_transcript_passed_to_shared_create(self):
        """Pages without transcripts are filtered by the shared create path.

        The handler delegates the (transcript-aware) create to
        ``create_memories_for_pages``; with no creatable pages the mock returns
        an empty id list and _finalize is called with it.
        """
        page = make_page(page_id="page_001", transcript=None)
        doc = make_document()
        storage = make_mock_storage(pages=[page], doc=doc)
        ingestion_svc = make_mock_ingestion_service(created_memory_ids=[])

        with patch_finalize_handler(storage=storage, ingestion_service=ingestion_svc) as (
            storage, ingestion_svc
        ):
            handler = DocumentFinalizeTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        ingestion_svc.create_memories_for_pages.assert_called_once()
        # _finalize still called with empty memory list
        ingestion_svc._finalize.assert_called_once_with(doc, [], "job_aaa000000001")

    @pytest.mark.asyncio
    async def test_page_without_embedding_stays_none(self):
        """Pages with neither a column embedding nor the legacy stash stay None."""
        page = make_page(
            page_id="page_001",
            transcript="Content.",
            embedding=None,
            metadata={},  # No legacy _embedding key either
        )
        doc = make_document()
        storage = make_mock_storage(pages=[page], doc=doc)
        ingestion_svc = make_mock_ingestion_service(created_memory_ids=["mem_001"])

        with patch_finalize_handler(storage=storage, ingestion_service=ingestion_svc) as (
            storage, ingestion_svc
        ):
            handler = DocumentFinalizeTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        ingestion_svc.create_memories_for_pages.assert_called_once()
        # No embedding anywhere -> page.embedding remains None.
        assert page.embedding is None

    @pytest.mark.asyncio
    async def test_page_embedding_restored_from_legacy_metadata_fallback(self):
        """Transition fallback: a not-yet-migrated page (column NULL but legacy
        metadata['_embedding'] present) still has its embedding restored."""
        page = make_page(
            page_id="page_001",
            transcript="Content.",
            embedding=None,  # column not populated (pre-migration row)
            metadata={"_embedding": [0.4, 0.5]},  # legacy stash still present
        )
        doc = make_document()
        storage = make_mock_storage(pages=[page], doc=doc)
        ingestion_svc = make_mock_ingestion_service(created_memory_ids=["mem_001"])

        with patch_finalize_handler(storage=storage, ingestion_service=ingestion_svc) as (
            storage, ingestion_svc
        ):
            handler = DocumentFinalizeTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        ingestion_svc.create_memories_for_pages.assert_called_once()
        assert page.embedding == [0.4, 0.5]

    @pytest.mark.asyncio
    async def test_multiple_pages_delegates_with_all_pages(self):
        """All pages are handed to the shared create path; ids flow to _finalize."""
        pages = [
            make_page(
                page_id="page_001", page_no=0,
                transcript="Page 1 content.",
                embedding=[0.1, 0.2],
            ),
            make_page(
                page_id="page_002", page_no=1,
                transcript="Page 2 content.",
                embedding=[0.3, 0.4],
            ),
        ]
        storage = make_mock_storage(pages=pages, doc=make_document())
        ingestion_svc = make_mock_ingestion_service(
            created_memory_ids=["mem_001", "mem_002"]
        )

        with patch_finalize_handler(storage=storage, ingestion_service=ingestion_svc) as (
            storage, ingestion_svc
        ):
            handler = DocumentFinalizeTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        ingestion_svc.create_memories_for_pages.assert_called_once()
        assert ingestion_svc.create_memories_for_pages.call_args.args[1] == pages
        ingestion_svc._finalize.assert_called_once()
        _, finalize_args, _ = ingestion_svc._finalize.mock_calls[0]
        assert finalize_args[1] == ["mem_001", "mem_002"]

    @pytest.mark.asyncio
    async def test_error_marks_failed(self):
        """Exception in finalize handler marks doc/job as FAILED."""
        doc = make_document()
        storage = make_mock_storage(doc=doc)
        storage.get_pages.side_effect = RuntimeError("DB unavailable")
        ingestion_svc = make_mock_ingestion_service()

        with patch_finalize_handler(storage=storage, ingestion_service=ingestion_svc) as (
            storage, _
        ):
            handler = DocumentFinalizeTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        failed_doc_calls = [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.FAILED.value
        ]
        assert len(failed_doc_calls) == 1

        failed_job_calls = [
            c for c in storage.update_job.call_args_list
            if c.kwargs.get("status") == JobStatus.FAILED.value
        ]
        assert len(failed_job_calls) == 1

    @pytest.mark.asyncio
    async def test_error_document_not_found(self):
        """Missing document causes FAILED state."""
        storage = make_mock_storage()
        storage.get_document.return_value = None
        ingestion_svc = make_mock_ingestion_service()

        with patch_finalize_handler(storage=storage, ingestion_service=ingestion_svc) as (
            storage, _
        ):
            handler = DocumentFinalizeTaskHandler()
            await handler.handle(MagicMock(), PAYLOAD)

        failed_calls = [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.FAILED.value
        ]
        assert len(failed_calls) == 1


# ---------------------------------------------------------------------------
# Integration: task chaining smoke test
# ---------------------------------------------------------------------------

class TestPipelineChaining:
    """Verify the correct chaining: render -> transcribe -> embed -> finalize."""

    def test_each_handler_schedules_correct_next_phase(self):
        assert DocumentRenderTaskHandler().get_task_type() == "document_render"
        assert DocumentTranscribeTaskHandler().get_task_type() == "document_transcribe"
        assert DocumentEmbedTaskHandler().get_task_type() == "document_embed"
        assert DocumentFinalizeTaskHandler().get_task_type() == "document_finalize"

    def test_all_handlers_return_none_schedule(self):
        v = MagicMock()
        for handler_cls in [
            DocumentRenderTaskHandler,
            DocumentTranscribeTaskHandler,
            DocumentEmbedTaskHandler,
            DocumentFinalizeTaskHandler,
        ]:
            assert handler_cls().get_schedule(v) is None
