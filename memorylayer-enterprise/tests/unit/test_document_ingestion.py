"""
Unit tests for DocumentIngestionService.

Tests:
- upload_document: validation, dedup, blob store, DB record creation, task scheduling
- delete_document: blob cleanup, optional memory deletion
- cancel_job: status update, terminal-state guard
- _detect_document_type: extension mapping including unsupported types
- process_document: happy-path pipeline, failure handling
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from memorylayer_saas.services.document.embed_client import EmbedServerClient
from memorylayer_saas.services.document.blob_storage import BlobStorageService
import memorylayer_saas.services.document.ingestion_service as isvc
from memorylayer_saas.services.document.ingestion_service import DocumentIngestionService
from memorylayer_saas.models.document import (
    Document,
    DocumentExtractionOptions,
    DocumentPage,
    DocumentStatus,
    DocumentType,
    IngestionJob,
    JobStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_document(
    doc_id: str = "doc_test000001",
    workspace_id: str = "ws_test",
    filename: str = "test.pdf",
    document_type: DocumentType = DocumentType.PDF,
    status: DocumentStatus = DocumentStatus.PENDING,
    storage_path: str = "/blobs/ws_test/documents/doc_test000001/test.pdf",
    memory_ids: list[str] | None = None,
    source_vfs_ref: str | None = None,
) -> Document:
    """Factory for Document domain model instances."""
    return Document(
        id=doc_id,
        workspace_id=workspace_id,
        filename=filename,
        document_type=document_type,
        content_hash="deadbeef" * 8,
        size_bytes=1024,
        status=status,
        storage_path=storage_path,
        memory_ids=memory_ids or [],
        source_vfs_ref=source_vfs_ref,
    )


def make_job(
    job_id: str = "job_test000001",
    workspace_id: str = "ws_test",
    document_ids: list[str] | None = None,
    status: JobStatus = JobStatus.QUEUED,
) -> IngestionJob:
    """Factory for IngestionJob domain model instances."""
    return IngestionJob(
        id=job_id,
        workspace_id=workspace_id,
        document_ids=document_ids or ["doc_test000001"],
        status=status,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_storage():
    """Create a fully mocked storage backend."""
    storage = AsyncMock()
    storage.find_document_by_hash.return_value = None
    storage.create_document.return_value = make_document()
    storage.create_job.return_value = make_job()
    storage.get_document.return_value = make_document()
    storage.get_job.return_value = make_job()
    storage.update_document.return_value = None
    storage.update_job.return_value = None
    # Job-coalescing lookup: default to no in-flight jobs so the create/finalize
    # paths' supersede step is a no-op unless a test overrides it.
    storage.list_active_jobs_for_documents.return_value = []
    storage.delete_document.return_value = None
    storage.delete_pages.return_value = 0
    storage.get_pages.return_value = []
    storage.delete_memory.return_value = None
    storage.create_memory.return_value = MagicMock(id="mem_001")
    storage.create_page.return_value = MagicMock(id="page_persisted_001")
    return storage


@pytest.fixture
def mock_blob():
    """Create a fully mocked BlobStorageService."""
    blob = AsyncMock(spec=BlobStorageService)
    blob.document_path.return_value = "/blobs/ws_test/documents/doc_test000001/test.pdf"
    blob.page_image_path.return_value = "/blobs/ws_test/documents/doc_test000001/pages/page_0000.png"
    blob.page_transcript_path.return_value = "/blobs/ws_test/documents/doc_test000001/transcripts/page_0000.md"
    blob.document_pages_prefix.return_value = "/blobs/ws_test/documents/doc_test000001/pages"
    blob.document_image_embeds_prefix.return_value = "/blobs/ws_test/documents/doc_test000001/image_embeds"
    blob.document_prompt_embeds_prefix.return_value = "/blobs/ws_test/documents/doc_test000001/prompt_embeds"
    blob.document_transcripts_prefix.return_value = "/blobs/ws_test/documents/doc_test000001/transcripts"
    blob.store_file.return_value = "/blobs/ws_test/documents/doc_test000001/test.pdf"
    blob.retrieve_file.return_value = b"fake content"
    blob.delete_tree.return_value = None
    return blob


@pytest.fixture
def mock_embed():
    """Create a fully mocked EmbedServerClient."""
    embed = AsyncMock(spec=EmbedServerClient)
    embed.connect.return_value = None
    embed.close.return_value = None
    # Real embed-server TranscriptionResponse shape -- see the note in
    # test_document_pipeline_handlers.make_mock_embed_client.
    embed.transcribe_pages.return_value = {
        "results": [
            {
                "page_index": 0,
                "content": "Transcribed text.",
                "success": True,
                "model_used": "vlm-v1",
                "provider_used": "glm-ocr",
                "attempts": [],
            }
        ],
        "stats": {"total_pages": 1, "successful_pages": 1, "failed_pages": 0},
    }
    embed.embed_texts.return_value = [[0.1, 0.2, 0.3]]
    embed.embed_images_multivector.return_value = [{"vectors": [[0.1, 0.2]], "num_vectors": 1}]
    return embed


@pytest.fixture
def mock_tasks():
    """Create a mocked task service."""
    tasks = AsyncMock()
    tasks.schedule_task.return_value = None
    return tasks


@pytest.fixture
def mock_memory_service():
    """Create a mocked MemoryService exposing the post-store hook."""
    ms = AsyncMock()
    ms.enqueue_post_store.return_value = None
    return ms


@pytest.fixture
def service(mock_storage, mock_blob, mock_embed, mock_tasks, mock_memory_service):
    """Create a DocumentIngestionService with all dependencies mocked.

    ``v.environ`` returns the caller-supplied default for every lookup so the
    pipeline's feature flags resolve to their production defaults (transcription
    enabled, visual tokenizer + document-chat ingestion disabled) unless a test
    overrides ``v`` explicitly.
    """
    v = MagicMock()
    v.environ.side_effect = lambda key, default=None, **kw: default
    return DocumentIngestionService(
        v=v,
        storage_backend=mock_storage,
        blob_storage=mock_blob,
        embed_client=mock_embed,
        task_service=mock_tasks,
        memory_service=mock_memory_service,
        max_file_size=10 * 1024 * 1024,  # 10 MB
        logger=MagicMock(),
    )


# ---------------------------------------------------------------------------
# _detect_document_type (static helper)
# ---------------------------------------------------------------------------

class TestDetectDocumentType:
    """Tests for the _detect_document_type static method."""

    def test_detect_pdf(self):
        """Test that .pdf extension maps to DocumentType.PDF."""
        result = DocumentIngestionService._detect_document_type("report.pdf")
        assert result == DocumentType.PDF

    def test_detect_markdown_md(self):
        """Test that .md extension maps to DocumentType.MARKDOWN."""
        result = DocumentIngestionService._detect_document_type("readme.md")
        assert result == DocumentType.MARKDOWN

    def test_detect_markdown_markdown(self):
        """Test that .markdown extension maps to DocumentType.MARKDOWN."""
        result = DocumentIngestionService._detect_document_type("notes.markdown")
        assert result == DocumentType.MARKDOWN

    def test_detect_text_txt(self):
        """Test that .txt extension maps to DocumentType.TEXT."""
        result = DocumentIngestionService._detect_document_type("data.txt")
        assert result == DocumentType.TEXT

    def test_detect_text_text(self):
        """Test that .text extension maps to DocumentType.TEXT."""
        result = DocumentIngestionService._detect_document_type("log.text")
        assert result == DocumentType.TEXT

    def test_detect_html(self):
        """Test that .html extension maps to DocumentType.HTML."""
        result = DocumentIngestionService._detect_document_type("page.html")
        assert result == DocumentType.HTML

    def test_detect_htm(self):
        """Test that .htm extension maps to DocumentType.HTML."""
        result = DocumentIngestionService._detect_document_type("page.htm")
        assert result == DocumentType.HTML

    def test_detect_docx(self):
        """Test that .docx extension maps to DocumentType.DOCX."""
        result = DocumentIngestionService._detect_document_type("document.docx")
        assert result == DocumentType.DOCX

    def test_detect_pptx(self):
        """Test that .pptx extension maps to DocumentType.PPTX."""
        result = DocumentIngestionService._detect_document_type("slides.pptx")
        assert result == DocumentType.PPTX

    def test_detect_case_insensitive(self):
        """Test that extension detection is case-insensitive."""
        result = DocumentIngestionService._detect_document_type("Report.PDF")
        assert result == DocumentType.PDF

    def test_detect_unsupported_raises(self):
        """Test that an unsupported extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            DocumentIngestionService._detect_document_type("archive.xyz")

    def test_detect_no_extension_raises(self):
        """Test that a filename with no extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            DocumentIngestionService._detect_document_type("noextension")


# ---------------------------------------------------------------------------
# upload_document
# ---------------------------------------------------------------------------

class TestUploadDocument:
    """Tests for upload_document() Phase 1 pipeline."""

    @pytest.mark.asyncio
    async def test_upload_document_success(self, service, mock_storage, mock_tasks):
        """Test successful document upload creates DB records and schedules processing."""
        mock_storage.find_document_by_hash.return_value = None
        expected_doc = make_document()
        expected_job = make_job()
        mock_storage.create_document.return_value = expected_doc
        mock_storage.create_job.return_value = expected_job

        doc, job = await service.upload_document(
            workspace_id="ws_test",
            file_data=b"fake pdf bytes",
            filename="test.pdf",
        )

        assert doc.id == expected_doc.id
        assert job.id == expected_job.id
        mock_storage.create_document.assert_called_once()
        mock_storage.create_job.assert_called_once()
        # upload_document generates fresh UUIDs for doc_id/job_id internally,
        # so verify structure rather than exact IDs
        mock_tasks.schedule_task.assert_called_once()
        call_args = mock_tasks.schedule_task.call_args
        assert call_args[0][0] == "document_render"
        payload = call_args[0][1]
        assert payload["workspace_id"] == "ws_test"
        assert payload["document_id"].startswith("doc_")
        assert payload["job_id"].startswith("job_")
        assert call_args[1]["priority"] == 3

    @pytest.mark.asyncio
    async def test_upload_document_too_large_raises(self, service):
        """Test that a file exceeding max_file_size raises ValueError."""
        oversized_data = b"x" * (10 * 1024 * 1024 + 1)  # 1 byte over 10 MB

        with pytest.raises(ValueError, match="exceeds maximum"):
            await service.upload_document(
                workspace_id="ws_test",
                file_data=oversized_data,
                filename="huge.pdf",
            )

    @pytest.mark.asyncio
    async def test_upload_document_duplicate_raises(self, service, mock_storage):
        """Test that a duplicate content hash raises ValueError."""
        existing_doc = make_document(doc_id="doc_existing")
        mock_storage.find_document_by_hash.return_value = existing_doc

        with pytest.raises(ValueError, match="Duplicate document"):
            await service.upload_document(
                workspace_id="ws_test",
                file_data=b"some content",
                filename="test.pdf",
            )

    @pytest.mark.asyncio
    async def test_upload_document_unsupported_type_raises(self, service, mock_storage):
        """Test that an unsupported file extension raises ValueError."""
        mock_storage.find_document_by_hash.return_value = None

        with pytest.raises(ValueError, match="Unsupported file type"):
            await service.upload_document(
                workspace_id="ws_test",
                file_data=b"data",
                filename="archive.xyz",
            )

    @pytest.mark.asyncio
    async def test_upload_document_stores_blob_when_retain_original(
        self, service, mock_storage, mock_blob
    ):
        """Test that original file is stored in blob storage when retain_original=True."""
        mock_storage.find_document_by_hash.return_value = None
        mock_storage.create_document.return_value = make_document()
        mock_storage.create_job.return_value = make_job()

        options = DocumentExtractionOptions(retain_original=True)
        await service.upload_document(
            workspace_id="ws_test",
            file_data=b"pdf content",
            filename="test.pdf",
            extraction_options=options,
        )

        mock_blob.store_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_document_skips_blob_when_not_retain(
        self, service, mock_storage, mock_blob
    ):
        """Test that blob storage is skipped when retain_original=False."""
        mock_storage.find_document_by_hash.return_value = None
        mock_storage.create_document.return_value = make_document()
        mock_storage.create_job.return_value = make_job()

        options = DocumentExtractionOptions(retain_original=False)
        await service.upload_document(
            workspace_id="ws_test",
            file_data=b"pdf content",
            filename="test.pdf",
            extraction_options=options,
        )

        mock_blob.store_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_document_explicit_type_overrides_detection(
        self, service, mock_storage
    ):
        """Test that an explicit document_type bypasses extension detection."""
        mock_storage.find_document_by_hash.return_value = None
        # Return document with TEXT type to verify it was stored
        mock_storage.create_document.return_value = make_document(
            document_type=DocumentType.TEXT, filename="notes.log"
        )
        mock_storage.create_job.return_value = make_job()

        doc, _ = await service.upload_document(
            workspace_id="ws_test",
            file_data=b"plain text",
            filename="notes.log",
            document_type=DocumentType.TEXT,
        )

        # The doc from storage is returned directly; verify create_document was called
        # with the explicit type (check call args)
        call_args = mock_storage.create_document.call_args
        created_doc: Document = call_args[0][0]
        assert created_doc.document_type == DocumentType.TEXT


# ---------------------------------------------------------------------------
# delete_document
# ---------------------------------------------------------------------------

class TestDeleteDocument:
    """Tests for delete_document()."""

    @pytest.mark.asyncio
    async def test_delete_document_success(self, service, mock_storage, mock_blob):
        """Test that delete_document removes the blob tree and DB record."""
        doc = make_document()
        mock_storage.get_document.return_value = doc

        await service.delete_document("doc_test000001", workspace_id="ws_test")

        mock_blob.delete_tree.assert_called_once()
        mock_storage.delete_document.assert_called_once_with("doc_test000001")

    @pytest.mark.asyncio
    async def test_delete_document_not_found_raises(self, service, mock_storage):
        """Test that deleting a non-existent document raises ValueError."""
        mock_storage.get_document.return_value = None

        with pytest.raises(ValueError, match="Document not found"):
            await service.delete_document("doc_missing", workspace_id="ws_test")

    @pytest.mark.asyncio
    async def test_delete_document_with_memories(self, service, mock_storage, mock_blob):
        """Test that memory records are deleted when delete_memories=True."""
        doc = make_document(memory_ids=["mem_001", "mem_002"])
        mock_storage.get_document.return_value = doc

        await service.delete_document(
            "doc_test000001", workspace_id="ws_test", delete_memories=True
        )

        assert mock_storage.delete_memory.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_document_without_memories(self, service, mock_storage, mock_blob):
        """Test that memory records are not deleted when delete_memories=False."""
        doc = make_document(memory_ids=["mem_001"])
        mock_storage.get_document.return_value = doc

        await service.delete_document(
            "doc_test000001", workspace_id="ws_test", delete_memories=False
        )

        mock_storage.delete_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_document_memory_failure_is_logged_not_raised(
        self, service, mock_storage, mock_blob
    ):
        """Test that a memory deletion failure is logged but does not abort the delete."""
        doc = make_document(memory_ids=["mem_bad"])
        mock_storage.get_document.return_value = doc
        mock_storage.delete_memory.side_effect = RuntimeError("DB error")

        # Should not raise
        await service.delete_document(
            "doc_test000001", workspace_id="ws_test", delete_memories=True
        )

        # Blob and document deletion should still proceed
        mock_blob.delete_tree.assert_called_once()
        mock_storage.delete_document.assert_called_once()


# ---------------------------------------------------------------------------
# reprocess_document
# ---------------------------------------------------------------------------

class TestReprocessDocument:
    """Tests for reprocess_document() phase handling and derived-state clearing."""

    @pytest.mark.asyncio
    async def test_reprocess_from_render_clears_pages_and_blobs(
        self, service, mock_storage, mock_blob, mock_tasks
    ):
        """from_phase='render' deletes page rows + derived blob subtrees and schedules render."""
        doc = make_document()
        mock_storage.get_document.return_value = doc
        mock_storage.delete_pages.return_value = 3

        job = await service.reprocess_document(
            "doc_test000001", workspace_id="ws_test",
        )

        # Page rows cleared via the storage method
        mock_storage.delete_pages.assert_awaited_once_with("doc_test000001", "ws_test")

        # Derived blob subtrees removed; the doc-root/original file is preserved
        deleted_prefixes = {
            call.args[0] for call in mock_blob.delete_tree.call_args_list
        }
        assert "/blobs/ws_test/documents/doc_test000001/pages" in deleted_prefixes
        assert "/blobs/ws_test/documents/doc_test000001/image_embeds" in deleted_prefixes
        # Never delete_tree the doc root or the original uploaded file
        assert "/blobs/ws_test/documents/doc_test000001" not in deleted_prefixes
        assert "/blobs/ws_test/documents/doc_test000001/test.pdf" not in deleted_prefixes

        # Render phase scheduled
        call_args = mock_tasks.schedule_task.call_args
        assert call_args[0][0] == "document_render"
        assert job.id is not None

    @pytest.mark.asyncio
    async def test_reprocess_from_render_resets_page_count(
        self, service, mock_storage
    ):
        """from_phase='render' zeroes page_count when resetting metadata."""
        mock_storage.get_document.return_value = make_document()

        await service.reprocess_document("doc_test000001", workspace_id="ws_test")

        reset_calls = [
            call.kwargs
            for call in mock_storage.update_document.call_args_list
            if "page_count" in call.kwargs
        ]
        assert reset_calls
        assert reset_calls[-1]["page_count"] == 0

    @pytest.mark.asyncio
    async def test_reprocess_from_embed_skips_clear_and_schedules_embed(
        self, service, mock_storage, mock_blob, mock_tasks
    ):
        """from_phase='embed' reuses pages: no page/blob deletes, schedules document_embed."""
        doc = make_document()
        mock_storage.get_document.return_value = doc
        mock_storage.get_pages.return_value = [
            DocumentPage(
                id="page_001",
                document_id="doc_test000001",
                workspace_id="ws_test",
                page_no=0,
                transcript="text",
            )
        ]

        job = await service.reprocess_document(
            "doc_test000001", workspace_id="ws_test", from_phase="embed",
        )

        # Existing rendered pages must be preserved
        mock_storage.delete_pages.assert_not_called()
        mock_blob.delete_tree.assert_not_called()

        # Embed phase scheduled directly
        call_args = mock_tasks.schedule_task.call_args
        assert call_args[0][0] == "document_embed"

        # page_count must NOT be zeroed on the embed reset
        reset_calls = [
            call.kwargs for call in mock_storage.update_document.call_args_list
        ]
        assert all("page_count" not in kwargs for kwargs in reset_calls)
        assert job.id is not None

    @pytest.mark.asyncio
    async def test_reprocess_from_embed_without_pages_raises(
        self, service, mock_storage
    ):
        """from_phase='embed' with no rendered pages raises ValueError."""
        mock_storage.get_document.return_value = make_document()
        mock_storage.get_pages.return_value = []

        with pytest.raises(ValueError, match="reprocess from render"):
            await service.reprocess_document(
                "doc_test000001", workspace_id="ws_test", from_phase="embed",
            )

    @pytest.mark.asyncio
    async def test_reprocess_invalid_from_phase_raises(self, service, mock_storage):
        """An invalid from_phase raises ValueError (mapped to HTTP 400 at the endpoint)."""
        mock_storage.get_document.return_value = make_document()

        with pytest.raises(ValueError, match="Invalid from_phase"):
            await service.reprocess_document(
                "doc_test000001", workspace_id="ws_test", from_phase="bogus",
            )

    @pytest.mark.asyncio
    async def test_reprocess_not_found_raises(self, service, mock_storage):
        """Reprocessing a missing document raises ValueError."""
        mock_storage.get_document.return_value = None

        with pytest.raises(ValueError, match="Document not found"):
            await service.reprocess_document("doc_missing", workspace_id="ws_test")

    @pytest.mark.asyncio
    async def test_reprocess_render_blob_delete_failure_is_logged_not_raised(
        self, service, mock_storage, mock_blob, mock_tasks
    ):
        """A missing/failed derived subtree must not abort the reprocess."""
        mock_storage.get_document.return_value = make_document()
        mock_blob.delete_tree.side_effect = FileNotFoundError("absent subtree")

        # Should not raise despite blob delete failures
        await service.reprocess_document("doc_test000001", workspace_id="ws_test")

        # Render still scheduled
        assert mock_tasks.schedule_task.call_args[0][0] == "document_render"


# ---------------------------------------------------------------------------
# cancel_job
# ---------------------------------------------------------------------------

class TestCancelJob:
    """Tests for cancel_job()."""

    @pytest.mark.asyncio
    async def test_cancel_job_success(self, service, mock_storage):
        """Test that a queued job can be cancelled."""
        job = make_job(status=JobStatus.QUEUED)
        mock_storage.get_job.return_value = job

        await service.cancel_job("job_test000001")

        mock_storage.update_job.assert_called_once()
        call_kwargs = mock_storage.update_job.call_args[1]
        assert call_kwargs["status"] == JobStatus.CANCELLED.value

    @pytest.mark.asyncio
    async def test_cancel_running_job_success(self, service, mock_storage):
        """Test that a running job can also be cancelled."""
        job = make_job(status=JobStatus.RUNNING)
        mock_storage.get_job.return_value = job

        await service.cancel_job("job_test000001")

        call_kwargs = mock_storage.update_job.call_args[1]
        assert call_kwargs["status"] == JobStatus.CANCELLED.value

    @pytest.mark.asyncio
    async def test_cancel_job_not_found_raises(self, service, mock_storage):
        """Test that cancelling a non-existent job raises ValueError."""
        mock_storage.get_job.return_value = None

        with pytest.raises(ValueError, match="Job not found"):
            await service.cancel_job("job_missing")

    @pytest.mark.asyncio
    async def test_cancel_completed_job_raises(self, service, mock_storage):
        """Test that cancelling a completed job raises ValueError."""
        job = make_job(status=JobStatus.COMPLETED)
        mock_storage.get_job.return_value = job

        with pytest.raises(ValueError, match="terminal state"):
            await service.cancel_job("job_test000001")

    @pytest.mark.asyncio
    async def test_cancel_failed_job_raises(self, service, mock_storage):
        """Test that cancelling a failed job raises ValueError."""
        job = make_job(status=JobStatus.FAILED)
        mock_storage.get_job.return_value = job

        with pytest.raises(ValueError, match="terminal state"):
            await service.cancel_job("job_test000001")

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled_job_raises(self, service, mock_storage):
        """Test that cancelling an already-cancelled job raises ValueError."""
        job = make_job(status=JobStatus.CANCELLED)
        mock_storage.get_job.return_value = job

        with pytest.raises(ValueError, match="terminal state"):
            await service.cancel_job("job_test000001")

    @pytest.mark.asyncio
    async def test_cancel_job_records_completed_at(self, service, mock_storage):
        """Test that cancel_job records a completed_at timestamp."""
        job = make_job(status=JobStatus.QUEUED)
        mock_storage.get_job.return_value = job

        await service.cancel_job("job_test000001")

        call_kwargs = mock_storage.update_job.call_args[1]
        assert "completed_at" in call_kwargs
        assert isinstance(call_kwargs["completed_at"], datetime)


# ---------------------------------------------------------------------------
# process_document (pipeline)
# ---------------------------------------------------------------------------

class TestProcessDocument:
    """Tests for process_document() full pipeline execution.

    These exercise ORCHESTRATION — phase order, progress, terminal status — so
    they stub the render phase rather than driving a real one. TEXT documents no
    longer produce a page inline: they convert to PDF via LibreOffice and flow
    through the shared raster path, so a test that drove rendering here would be
    asserting soffice availability instead of the pipeline it names. Rendering
    itself is covered by ``TestRenderPdfPagesOffloading`` and the office tests.
    """

    @staticmethod
    def _rendered_page(transcript="Some plain text content."):
        """One already-rendered, already-transcribed page.

        Carrying a transcript is what makes it eligible for embedding, which is
        the pipeline behaviour these tests are about.
        """
        return DocumentPage(
            id="page_001",
            document_id="doc_test000001",
            workspace_id="ws_test",
            page_no=0,
            image_storage_path="/blobs/ws_test/doc_test000001/page_0000.png",
            transcript=transcript,
        )

    @pytest.mark.asyncio
    async def test_process_document_text_success(self, service, mock_storage, mock_blob, mock_embed):
        """Full pipeline over one rendered page: embed -> memory -> COMPLETED."""
        text_doc = make_document(
            document_type=DocumentType.TEXT,
            filename="notes.txt",
            storage_path="/blobs/ws_test/documents/doc_test000001/notes.txt",
        )
        mock_storage.get_document.return_value = text_doc
        mock_embed.embed_texts.return_value = [[0.1, 0.2, 0.3]]
        mock_storage.create_memory.return_value = MagicMock(id="mem_001")

        with patch.object(
            service, "_render_pages",
            new=AsyncMock(return_value=[self._rendered_page()]),
        ):
            await service.process_document(
                document_id="doc_test000001",
                job_id="job_test000001",
                workspace_id="ws_test",
            )

        # Document and job should be marked PROCESSING at start
        first_update = mock_storage.update_document.call_args_list[0]
        assert first_update[1]["status"] == DocumentStatus.PROCESSING.value

        # Embedding should have been called for the text page
        mock_embed.embed_texts.assert_called_once()

        # Memory should be created
        mock_storage.create_memory.assert_called_once()

        # Final job status should be COMPLETED
        final_job_update = mock_storage.update_job.call_args_list[-1]
        assert final_job_update[1]["status"] == JobStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_process_document_failure_sets_failed_status(
        self, service, mock_storage, mock_blob
    ):
        """Test that an exception during processing sets FAILED status on doc and job."""
        doc = make_document()
        mock_storage.get_document.return_value = doc
        # Simulate a failure in _render_pages by making retrieve_file raise
        mock_blob.retrieve_file.side_effect = RuntimeError("Blob read error")

        # Patch the doc type so _render_pages tries to retrieve the file
        doc_text = make_document(
            document_type=DocumentType.TEXT,
            storage_path="/blobs/ws_test/documents/doc_test000001/notes.txt",
        )
        mock_storage.get_document.return_value = doc_text

        await service.process_document(
            document_id="doc_test000001",
            job_id="job_test000001",
            workspace_id="ws_test",
        )

        # Find update calls that set FAILED
        doc_updates = [
            call[1]
            for call in mock_storage.update_document.call_args_list
            if call[1].get("status") == DocumentStatus.FAILED.value
        ]
        assert len(doc_updates) >= 1

        job_updates = [
            call[1]
            for call in mock_storage.update_job.call_args_list
            if call[1].get("status") == JobStatus.FAILED.value
        ]
        assert len(job_updates) >= 1

    @pytest.mark.asyncio
    async def test_process_document_not_found_sets_failed_status(
        self, service, mock_storage
    ):
        """Test that a missing document during pipeline sets FAILED status."""
        # First call (update to PROCESSING) then get_document returns None
        mock_storage.get_document.return_value = None

        await service.process_document(
            document_id="doc_missing",
            job_id="job_test000001",
            workspace_id="ws_test",
        )

        job_updates = [
            call[1]
            for call in mock_storage.update_job.call_args_list
            if call[1].get("status") == JobStatus.FAILED.value
        ]
        assert len(job_updates) >= 1

    @pytest.mark.asyncio
    async def test_process_document_updates_progress(
        self, service, mock_storage, mock_blob, mock_embed
    ):
        """Test that process_document updates job progress_percent at each phase."""
        text_doc = make_document(
            document_type=DocumentType.TEXT,
            filename="doc.txt",
            storage_path="/blobs/ws_test/documents/doc_test000001/doc.txt",
        )
        mock_storage.get_document.return_value = text_doc
        mock_embed.embed_texts.return_value = [[0.1, 0.2]]
        mock_storage.create_memory.return_value = MagicMock(id="mem_001")

        with patch.object(
            service, "_render_pages",
            new=AsyncMock(return_value=[self._rendered_page("Content.")]),
        ):
            await service.process_document(
                document_id="doc_test000001",
                job_id="job_test000001",
                workspace_id="ws_test",
            )

        # At least one progress update should have been made to the job
        progress_updates = [
            call[1]
            for call in mock_storage.update_job.call_args_list
            if "progress_percent" in call[1]
        ]
        assert len(progress_updates) >= 1

    @pytest.mark.asyncio
    async def test_process_document_creates_memories_for_each_page(
        self, service, mock_storage, mock_blob, mock_embed
    ):
        """Test that one memory is created per page with a transcript."""
        text_doc = make_document(
            document_type=DocumentType.TEXT,
            filename="doc.txt",
            storage_path="/blobs/ws_test/documents/doc_test000001/doc.txt",
        )
        mock_storage.get_document.return_value = text_doc
        mock_embed.embed_texts.return_value = [[0.1, 0.2, 0.3]]
        mock_storage.create_memory.return_value = MagicMock(id="mem_001")

        with patch.object(
            service, "_render_pages",
            new=AsyncMock(return_value=[self._rendered_page("Single page text content.")]),
        ):
            await service.process_document(
                document_id="doc_test000001",
                job_id="job_test000001",
                workspace_id="ws_test",
            )

        # One page -> one memory
        assert mock_storage.create_memory.call_count == 1


# ---------------------------------------------------------------------------
# Inline document-chat ingestion parity (transcription off -> chat text)
# ---------------------------------------------------------------------------

def _chat_ingest_v(transcribe=False, chat_ingest=True):
    """A Variables stand-in with the transcribe + chat-ingest flags set.

    Returns the caller-supplied default for every other lookup (prompt,
    max_tokens, visual tokenizer), so only the two flags under test are forced.
    """
    from memorylayer_saas.config import (
        MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED,
        MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
    )

    def _environ(key, default=None, **kw):
        if key == MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED:
            return transcribe
        if key == MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED:
            return chat_ingest
        return default

    v = MagicMock()
    v.environ.side_effect = _environ
    return v


def _make_service(v, mock_storage, mock_blob, mock_embed, mock_tasks, mock_memory_service=None):
    if mock_memory_service is None:
        mock_memory_service = AsyncMock()
        mock_memory_service.enqueue_post_store.return_value = None
    return DocumentIngestionService(
        v=v,
        storage_backend=mock_storage,
        blob_storage=mock_blob,
        embed_client=mock_embed,
        task_service=mock_tasks,
        memory_service=mock_memory_service,
        max_file_size=10 * 1024 * 1024,
        logger=MagicMock(),
    )


class TestInlineChatIngest:
    """The inline pipeline generates page text from image-embeds when OCR
    transcription is disabled, at parity with the distributed embed task."""

    @pytest.mark.asyncio
    async def test_embed_pages_generates_transcript_and_single_vector(
        self, mock_storage, mock_blob, mock_embed, mock_tasks
    ):
        import memorylayer_saas.services.document.ingestion_service as isvc

        v = _chat_ingest_v(transcribe=False, chat_ingest=True)
        service = _make_service(v, mock_storage, mock_blob, mock_embed, mock_tasks)
        doc = make_document(document_type=DocumentType.PDF)
        page = DocumentPage(
            id=None, document_id=doc.id, workspace_id=doc.workspace_id,
            page_no=0, image_storage_path="/blobs/.../page_0000.png", transcript=None,
        )

        with patch.object(
                 isvc, "precompute_and_store_image_embeds",
                 new=AsyncMock(return_value=1),
             ) as precompute, \
             patch.object(isvc, "get_inference_client", new=AsyncMock(return_value=AsyncMock())), \
             patch.object(
                 isvc, "generate_page_text_from_image_embeds",
                 new=AsyncMock(return_value="# Generated page text."),
             ) as gen:
            await service._embed_pages([page], doc)

        # Chat-ingest forces image-embed precompute even with the visual-tokenizer
        # flag off (it is the prerequisite for generating page text).
        precompute.assert_called_once()
        gen.assert_called_once()
        # Chat-generated text becomes the page transcript and is single-vector
        # embedded in-memory (so the later store phase turns it into a memory).
        assert page.transcript == "# Generated page text."
        assert page.embedding == [0.1, 0.2, 0.3]
        mock_embed.embed_texts.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_bytes_transcription_off_creates_memory(
        self, mock_storage, mock_blob, mock_embed, mock_tasks
    ):
        """End-to-end inline: transcription off + chat-ingest on -> a memory is
        created for the image page (regression of the 'zero memories' gap)."""
        import memorylayer_saas.services.document.ingestion_service as isvc

        v = _chat_ingest_v(transcribe=False, chat_ingest=True)
        service = _make_service(v, mock_storage, mock_blob, mock_embed, mock_tasks)
        doc = make_document(document_type=DocumentType.PDF)
        mock_storage.get_document.return_value = doc
        mock_storage.create_memory.return_value = MagicMock(id="mem_001")

        rendered = DocumentPage(
            id="page_001", document_id=doc.id, workspace_id=doc.workspace_id,
            page_no=0, image_storage_path="/blobs/.../page_0000.png", transcript=None,
        )

        with patch.object(service, "_render_pages", new=AsyncMock(return_value=[rendered])), \
             patch.object(service, "_transcribe_pages", new=AsyncMock()) as transcribe, \
             patch.object(
                 isvc, "precompute_and_store_image_embeds",
                 new=AsyncMock(return_value=1),
             ), \
             patch.object(isvc, "get_inference_client", new=AsyncMock(return_value=AsyncMock())), \
             patch.object(
                 isvc, "generate_page_text_from_image_embeds",
                 new=AsyncMock(return_value="Generated content for the page."),
             ):
            await service.process_bytes(doc, b"fake-pdf-bytes", "job_test000001")

        # Transcription was skipped (flag off); a memory was still created from
        # the chat-generated text.
        transcribe.assert_not_called()
        mock_storage.create_memory.assert_called_once()
        memory_input = mock_storage.create_memory.call_args.kwargs["input"]
        assert memory_input.content == "Generated content for the page."


# ---------------------------------------------------------------------------
# Office / HTML rendering (LibreOffice -> PDF -> shared PDF render path)
# ---------------------------------------------------------------------------

class TestRenderOfficePages:
    """Tests that HTML/DOCX/PPTX render via conversion to PDF then the PDF path."""

    @pytest.mark.parametrize(
        "doc_type, filename, source_ext",
        [
            (DocumentType.DOCX, "report.docx", "docx"),
            (DocumentType.PPTX, "slides.pptx", "pptx"),
            (DocumentType.HTML, "page.html", "html"),
        ],
    )
    @pytest.mark.asyncio
    async def test_office_dispatches_to_pdf_render_path(
        self, service, mock_blob, doc_type, filename, source_ext
    ):
        """DOCX/PPTX/HTML convert to PDF then feed the existing PDF render path.

        The soffice/conversion call is mocked to return a small PDF; the test
        asserts ``_render_pdf_pages`` is invoked with those converted bytes.
        """
        import memorylayer_saas.services.document.ingestion_service as isvc

        doc = make_document(
            document_type=doc_type,
            filename=filename,
            storage_path="/blobs/ws_test/documents/doc_test000001/%s" % filename,
        )
        mock_blob.retrieve_file.return_value = b"raw-office-bytes"

        fake_pdf = b"%PDF-1.4 fake"
        rendered_page = DocumentPage(
            id="page_001", document_id=doc.id, workspace_id=doc.workspace_id, page_no=0,
        )

        with patch.object(
            isvc, "convert_office_bytes_to_pdf",
            new=AsyncMock(return_value=fake_pdf),
        ) as convert, \
             patch.object(
                 service, "_render_pdf_pages",
                 new=AsyncMock(return_value=[rendered_page]),
             ) as render_pdf:
            pages = await service._render_pages(doc)

        convert.assert_awaited_once_with(b"raw-office-bytes", source_ext)
        render_pdf.assert_awaited_once_with(doc, pdf_bytes=fake_pdf)
        assert pages == [rendered_page]

    @pytest.mark.asyncio
    async def test_office_conversion_failure_propagates(self, service, mock_blob):
        """When conversion fails, OfficeConversionError surfaces from render.

        ``process_document`` catches this and records it as the job error, so
        such documents fail at the render stage with an actionable message
        instead of a bare NotImplementedError.
        """
        import memorylayer_saas.services.document.ingestion_service as isvc
        from memorylayer_saas.services.document.office_convert import (
            OfficeConversionError,
        )

        doc = make_document(
            document_type=DocumentType.DOCX,
            filename="report.docx",
            storage_path="/blobs/ws_test/documents/doc_test000001/report.docx",
        )
        mock_blob.retrieve_file.return_value = b"raw-office-bytes"

        with patch.object(
            isvc, "convert_office_bytes_to_pdf",
            new=AsyncMock(side_effect=OfficeConversionError("soffice missing")),
        ):
            with pytest.raises(OfficeConversionError):
                await service._render_pages(doc)


class TestRenderPdfPagesOffloading:
    """The CPU-bound rasterize + PNG-encode must run OFF the event loop (in
    ``_rasterize_pdf_batch`` via a worker thread); only the async blob stores run
    on the loop. Regression guard: encoding each page's PNG on the event loop
    starved the ``/livez`` probe and caused liveness restarts during large-PDF
    ingestion.
    """

    @pytest.mark.asyncio
    async def test_render_pdf_delegates_encode_offloop_and_stores_pages(
        self, service, mock_blob
    ):
        import memorylayer_saas.services.document.ingestion_service as isvc

        doc = make_document(
            document_type=DocumentType.PDF,
            filename="big.pdf",
            storage_path="/blobs/ws_test/documents/doc_test000001/big.pdf",
        )
        mock_blob.retrieve_file.return_value = b"%PDF-1.4 fake"
        mock_blob.page_image_path.side_effect = (
            lambda ws, did, i: "/blobs/%s/%s/page_%04d.png" % (ws, did, i)
        )

        # The off-loop helper returns pre-encoded PNG bytes, so the render loop
        # itself does NO PIL work on the event loop — it only awaits store_file.
        with patch("pdf2image.pdfinfo_from_bytes", return_value={"Pages": 3}), \
             patch.object(
                 isvc, "_rasterize_pdf_batch",
                 return_value=[b"png-0", b"png-1", b"png-2"],
             ) as raster:
            pages = await service._render_pdf_pages(doc)

        # Rasterize+encode was delegated to the off-loop helper (one batch covers
        # 3 pages at the default batch size).
        raster.assert_called_once()
        assert len(pages) == 3
        assert [p.page_no for p in pages] == [0, 1, 2]
        # Each page's PNG bytes were stored to blob, and no inline base64 kept.
        assert mock_blob.store_file.await_count == 3
        stored = [c.args[1] for c in mock_blob.store_file.await_args_list]
        assert stored == [b"png-0", b"png-1", b"png-2"]
        assert all(p.image_storage_path for p in pages)
        assert all(p.image_b64 is None for p in pages)

    @pytest.mark.asyncio
    async def test_render_stamps_the_source_vfs_ref_on_every_page(
        self, service, mock_blob
    ):
        """Render is the only point holding the owning Document.

        The page payload the API returns carries page fields only, so a reader
        holding a page row has no route back to the source entry unless the ref
        is stamped here. Consumers resolve a human-readable path from the VFS
        catalog through it.
        """
        import memorylayer_saas.services.document.ingestion_service as isvc

        doc = make_document(
            document_type=DocumentType.PDF,
            source_vfs_ref="vfs_9c1e2f",
        )
        mock_blob.retrieve_file.return_value = b"%PDF-1.4 fake"
        mock_blob.page_image_path.side_effect = (
            lambda ws, did, i: "/blobs/%s/%s/page_%04d.png" % (ws, did, i)
        )

        with patch("pdf2image.pdfinfo_from_bytes", return_value={"Pages": 3}), \
             patch.object(
                 isvc, "_rasterize_pdf_batch",
                 return_value=[b"png-0", b"png-1", b"png-2"],
             ):
            pages = await service._render_pdf_pages(doc)

        assert [p.metadata.get("vfs_ref") for p in pages] == ["vfs_9c1e2f"] * 3

    @pytest.mark.asyncio
    async def test_each_page_gets_its_own_metadata_dict(self, service, mock_blob):
        """Pages must not share one dict: the transcribe phase merges figure
        records into page metadata per page, and a shared mapping would leak
        one page's figures onto every other page of the document.
        """
        import memorylayer_saas.services.document.ingestion_service as isvc

        doc = make_document(document_type=DocumentType.PDF, source_vfs_ref="vfs_9c1e2f")
        mock_blob.retrieve_file.return_value = b"%PDF-1.4 fake"
        mock_blob.page_image_path.side_effect = (
            lambda ws, did, i: "/blobs/%s/%s/page_%04d.png" % (ws, did, i)
        )

        with patch("pdf2image.pdfinfo_from_bytes", return_value={"Pages": 2}), \
             patch.object(
                 isvc, "_rasterize_pdf_batch", return_value=[b"png-0", b"png-1"],
             ):
            pages = await service._render_pdf_pages(doc)

        pages[0].metadata["figures"] = [{"figure_no": 0}]
        assert "figures" not in pages[1].metadata

    @pytest.mark.asyncio
    async def test_a_document_with_no_vfs_ref_stores_no_null_key(
        self, service, mock_blob
    ):
        """Direct uploads have no catalog entry. Omit the key rather than
        persisting a null, so ``'vfs_ref' in metadata`` stays meaningful.
        """
        import memorylayer_saas.services.document.ingestion_service as isvc

        doc = make_document(document_type=DocumentType.PDF, source_vfs_ref=None)
        mock_blob.retrieve_file.return_value = b"%PDF-1.4 fake"
        mock_blob.page_image_path.side_effect = (
            lambda ws, did, i: "/blobs/%s/%s/page_%04d.png" % (ws, did, i)
        )

        with patch("pdf2image.pdfinfo_from_bytes", return_value={"Pages": 1}), \
             patch.object(isvc, "_rasterize_pdf_batch", return_value=[b"png-0"]):
            pages = await service._render_pdf_pages(doc)

        assert pages[0].metadata == {}

    def test_b64_encode_all_is_pure_and_matches_stdlib(self):
        import base64

        from memorylayer_saas.services.document.ingestion_service import (
            _b64_encode_all,
        )

        blobs = [b"\x00\x01two", b"three-bytes"]
        assert _b64_encode_all(blobs) == [
            base64.b64encode(b).decode("ascii") for b in blobs
        ]
        assert _b64_encode_all([]) == []


# ---------------------------------------------------------------------------
# office_convert helper (soffice subprocess wrapper)
# ---------------------------------------------------------------------------

class TestOfficeConvertHelper:
    """Tests for the office_convert soffice wrapper (subprocess mocked)."""

    @pytest.mark.asyncio
    async def test_convert_invokes_soffice_and_returns_pdf(self, tmp_path):
        """A successful soffice run returns the produced PDF bytes."""
        import memorylayer_saas.services.document.office_convert as oc

        def fake_run(args, cwd, **kwargs):
            # Simulate soffice writing input.pdf into the work dir.
            import os
            with open(os.path.join(cwd, "input.pdf"), "wb") as fh:
                fh.write(b"%PDF-1.4 converted")
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        with patch.object(oc, "_resolve_soffice_binary", return_value="/usr/bin/soffice"), \
             patch.object(oc, "subprocess_run", side_effect=fake_run):
            result = await oc.convert_office_bytes_to_pdf(b"docx-bytes", "docx")

        assert result == b"%PDF-1.4 converted"

    @pytest.mark.asyncio
    async def test_convert_raises_when_soffice_absent(self):
        """A clear OfficeConversionError is raised when soffice is unavailable."""
        import memorylayer_saas.services.document.office_convert as oc

        with patch.object(oc, "_resolve_soffice_binary", return_value=None):
            with pytest.raises(oc.OfficeConversionError, match="LibreOffice"):
                await oc.convert_office_bytes_to_pdf(b"docx-bytes", "docx")

    @pytest.mark.asyncio
    async def test_convert_raises_on_nonzero_exit(self):
        """A non-zero soffice exit raises OfficeConversionError with stderr."""
        import memorylayer_saas.services.document.office_convert as oc

        def fake_run(args, cwd, **kwargs):
            return MagicMock(returncode=1, stdout=b"", stderr=b"boom")

        with patch.object(oc, "_resolve_soffice_binary", return_value="/usr/bin/soffice"), \
             patch.object(oc, "subprocess_run", side_effect=fake_run):
            with pytest.raises(oc.OfficeConversionError, match="boom"):
                await oc.convert_office_bytes_to_pdf(b"docx-bytes", "docx")

    @pytest.mark.asyncio
    async def test_convert_raises_when_no_pdf_produced(self):
        """If soffice exits 0 but writes no PDF, an error is raised."""
        import memorylayer_saas.services.document.office_convert as oc

        def fake_run(args, cwd, **kwargs):
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        with patch.object(oc, "_resolve_soffice_binary", return_value="/usr/bin/soffice"), \
             patch.object(oc, "subprocess_run", side_effect=fake_run):
            with pytest.raises(oc.OfficeConversionError, match="no PDF"):
                await oc.convert_office_bytes_to_pdf(b"docx-bytes", "docx")

    # --- Correction 1: subprocess timeout ---

    @pytest.mark.asyncio
    async def test_convert_raises_on_timeout(self):
        """A hung soffice raises OfficeConversionError with a timeout message."""
        import subprocess
        import memorylayer_saas.services.document.office_convert as oc

        def fake_run(args, cwd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout", 120))

        with patch.object(oc, "_resolve_soffice_binary", return_value="/usr/bin/soffice"), \
             patch.object(oc, "subprocess_run", side_effect=fake_run):
            with pytest.raises(oc.OfficeConversionError, match="timed out"):
                await oc.convert_office_bytes_to_pdf(b"docx-bytes", "docx")

    @pytest.mark.asyncio
    async def test_convert_passes_timeout_to_subprocess_run(self):
        """The resolved timeout is forwarded to subprocess_run."""
        import memorylayer_saas.services.document.office_convert as oc

        captured: list[dict] = []

        def fake_run(args, cwd, **kwargs):
            captured.append(kwargs)
            import os
            with open(os.path.join(cwd, "input.pdf"), "wb") as fh:
                fh.write(b"%PDF-1.4 ok")
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        with patch.object(oc, "_resolve_soffice_binary", return_value="/usr/bin/soffice"), \
             patch.object(oc, "_resolve_timeout", return_value=30), \
             patch.object(oc, "subprocess_run", side_effect=fake_run):
            await oc.convert_office_bytes_to_pdf(b"docx-bytes", "docx")

        assert captured[0]["timeout"] == 30

    # --- Correction 2: per-invocation profile isolation ---

    @pytest.mark.asyncio
    async def test_convert_passes_isolated_profile_to_soffice(self):
        """Each conversion passes -env:UserInstallation pointing inside the temp dir."""
        import memorylayer_saas.services.document.office_convert as oc

        captured_args: list[list[str]] = []

        def fake_run(args, cwd, **kwargs):
            captured_args.append(list(args))
            import os
            with open(os.path.join(cwd, "input.pdf"), "wb") as fh:
                fh.write(b"%PDF-1.4 ok")
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        with patch.object(oc, "_resolve_soffice_binary", return_value="/usr/bin/soffice"), \
             patch.object(oc, "subprocess_run", side_effect=fake_run):
            await oc.convert_office_bytes_to_pdf(b"docx-bytes", "docx")

        assert len(captured_args) == 1
        profile_args = [a for a in captured_args[0] if a.startswith("-env:UserInstallation=")]
        assert len(profile_args) == 1, "Expected exactly one -env:UserInstallation= arg"
        # The profile must be inside a temp dir (file:// URL, path contains lo-profile).
        assert "lo-profile" in profile_args[0]
        assert profile_args[0].startswith("-env:UserInstallation=file://")

    # --- Correction 3: deterministic output path ---

    @pytest.mark.asyncio
    async def test_convert_uses_deterministic_pdf_path(self, tmp_path):
        """When soffice writes input.pdf the deterministic path is used (not glob)."""
        import memorylayer_saas.services.document.office_convert as oc

        def fake_run(args, cwd, **kwargs):
            import os
            # Write the deterministic name soffice normally produces.
            with open(os.path.join(cwd, "input.pdf"), "wb") as fh:
                fh.write(b"%PDF-1.4 deterministic")
            # Also write a decoy .pdf to confirm we don't pick the wrong file.
            with open(os.path.join(cwd, "decoy.pdf"), "wb") as fh:
                fh.write(b"%PDF-1.4 decoy")
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        with patch.object(oc, "_resolve_soffice_binary", return_value="/usr/bin/soffice"), \
             patch.object(oc, "subprocess_run", side_effect=fake_run):
            result = await oc.convert_office_bytes_to_pdf(b"docx-bytes", "docx")

        assert result == b"%PDF-1.4 deterministic"

    @pytest.mark.asyncio
    async def test_convert_falls_back_to_glob_when_stem_differs(self):
        """When soffice writes a non-default stem the glob fallback picks it up."""
        import memorylayer_saas.services.document.office_convert as oc

        def fake_run(args, cwd, **kwargs):
            import os
            # Simulate a platform that names the output differently.
            with open(os.path.join(cwd, "other.pdf"), "wb") as fh:
                fh.write(b"%PDF-1.4 fallback")
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        with patch.object(oc, "_resolve_soffice_binary", return_value="/usr/bin/soffice"), \
             patch.object(oc, "subprocess_run", side_effect=fake_run):
            result = await oc.convert_office_bytes_to_pdf(b"docx-bytes", "docx")

        assert result == b"%PDF-1.4 fallback"


# ---------------------------------------------------------------------------
# Ingestion-job coalescing + reconcile-on-complete (service layer)
# ---------------------------------------------------------------------------

def _cancelled_job_ids(mock_storage) -> list[str]:
    """Job ids the service transitioned to CANCELLED via update_job."""
    return [
        call.args[0]
        for call in mock_storage.update_job.call_args_list
        if call.kwargs.get("status") == JobStatus.CANCELLED.value
    ]


class TestJobCoalescing:
    """At-most-one in-flight job per document on create + reconcile-on-complete."""

    @pytest.mark.asyncio
    async def test_upload_supersedes_existing_inflight_job(self, service, mock_storage):
        """A 2nd upload-path job cancels the prior in-flight job for the document."""
        mock_storage.find_document_by_hash.return_value = None
        existing = make_job(job_id="job_old", status=JobStatus.RUNNING)
        mock_storage.list_active_jobs_for_documents.return_value = [existing]
        mock_storage.get_job.return_value = existing  # cancel_job re-reads it

        await service.upload_document(
            file_data=b"pdf-bytes", filename="a.pdf", workspace_id="ws_test",
        )

        # The prior in-flight job was superseded, and a fresh job was still created.
        assert "job_old" in _cancelled_job_ids(mock_storage)
        mock_storage.create_job.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reprocess_supersedes_then_creates_job(self, service, mock_storage):
        """Explicit reprocess supersedes any in-flight job AND still mints a new one."""
        mock_storage.get_document.return_value = make_document()
        existing = make_job(job_id="job_old", status=JobStatus.RUNNING)
        mock_storage.list_active_jobs_for_documents.return_value = [existing]
        mock_storage.get_job.return_value = existing

        job = await service.reprocess_document("doc_test000001", workspace_id="ws_test")

        assert "job_old" in _cancelled_job_ids(mock_storage)
        mock_storage.create_job.assert_awaited_once()
        assert job.id is not None  # reprocess is not broken by coalescing

    @pytest.mark.asyncio
    async def test_upload_noop_when_no_inflight_job(self, service, mock_storage):
        """With no prior in-flight job, nothing is cancelled."""
        mock_storage.find_document_by_hash.return_value = None
        mock_storage.list_active_jobs_for_documents.return_value = []

        await service.upload_document(
            file_data=b"pdf-bytes", filename="a.pdf", workspace_id="ws_test",
        )

        assert _cancelled_job_ids(mock_storage) == []
        mock_storage.create_job.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_finalize_completed_supersedes_other_jobs_but_keeps_driver(
        self, service, mock_storage
    ):
        """Document->COMPLETED cancels other in-flight jobs, keeping the driver job."""
        doc = make_document(memory_ids=["mem_001"])
        mock_storage.get_document.return_value = doc

        other = make_job(
            job_id="job_other", document_ids=["doc_test000001"], status=JobStatus.RUNNING,
        )
        driver = make_job(
            job_id="job_driver", document_ids=["doc_test000001"], status=JobStatus.RUNNING,
        )
        mock_storage.list_active_jobs_for_documents.return_value = [other, driver]
        mock_storage.get_job.side_effect = lambda jid: {
            "job_other": other, "job_driver": driver,
        }.get(jid)

        gap_mod = "memorylayer_saas.services.document.gap_analysis"
        with patch(
            f"{gap_mod}.analyze_document_gaps",
            new=AsyncMock(return_value=MagicMock(is_complete=True)),
        ), patch(f"{gap_mod}.resolve_effective_flags", return_value=MagicMock()):
            await service._finalize(doc, ["mem_001"], "job_driver")

        cancelled = _cancelled_job_ids(mock_storage)
        assert "job_other" in cancelled
        assert "job_driver" not in cancelled  # the completion driver is kept

    @pytest.mark.asyncio
    async def test_finalize_partial_does_not_supersede(self, service, mock_storage):
        """A non-COMPLETED finalize (PARTIAL) does not reconcile other jobs."""
        doc = make_document(memory_ids=["mem_001"])
        mock_storage.get_document.return_value = doc
        mock_storage.list_active_jobs_for_documents.return_value = [
            make_job(job_id="job_other", status=JobStatus.RUNNING),
        ]

        gap_mod = "memorylayer_saas.services.document.gap_analysis"
        with patch(
            f"{gap_mod}.analyze_document_gaps",
            new=AsyncMock(return_value=MagicMock(is_complete=False)),
        ), patch(f"{gap_mod}.resolve_effective_flags", return_value=MagicMock()):
            await service._finalize(doc, ["mem_001"], "job_driver")

        # memory_ids present but not complete -> PARTIAL, no reconcile pass.
        assert _cancelled_job_ids(mock_storage) == []


# ---------------------------------------------------------------------------
# Multivector spill (peak-memory control during ingestion)
# ---------------------------------------------------------------------------

class TestMultivectorSpillCodec:
    """The float32 codec backing the multivector spill."""

    def test_round_trip_preserves_shape_and_values(self):
        """Encode -> decode returns the same shape, within float32 precision."""
        mv = [[i * 0.001 + j for j in range(128)] for i in range(64)]

        back = isvc._decode_multivector(isvc._encode_multivector(mv))

        assert len(back) == 64
        assert all(len(row) == 128 for row in back)
        # float32 is what pgvector stores anyway, so this rounding is not a loss
        # relative to the destination column.
        assert max(
            abs(a - b) for ra, rb in zip(mv, back) for a, b in zip(ra, rb)
        ) < 1e-4

    def test_round_trip_empty(self):
        """An empty multivector survives the round trip."""
        assert isvc._decode_multivector(isvc._encode_multivector([])) == []

    def test_encoded_form_is_flat_float32(self):
        """The spill is 4 bytes per value plus the header — the whole point."""
        mv = [[0.5] * 128 for _ in range(1030)]

        blob = isvc._encode_multivector(mv)

        assert len(blob) == isvc._MULTIVECTOR_HEADER.size + 1030 * 128 * 4

    def test_decode_rejects_short_blob(self):
        """A blob smaller than the header is rejected, not indexed past."""
        with pytest.raises(ValueError, match="shorter than"):
            isvc._decode_multivector(b"\x00\x01")

    def test_decode_rejects_declared_shape_mismatch(self):
        """A garbage header fails fast instead of spinning.

        Regression: the decoder used to trust ``count`` from the header and
        build the result with ``range(count)``. A truncated write or a blob
        that is not a spill at all yields a huge ``count``, which hung the
        worker in a multi-billion-iteration loop rather than raising.
        """
        not_a_spill = b"%PDF-1.4 fake pdf bytes that are not a multivector"

        with pytest.raises(ValueError, match="declares"):
            isvc._decode_multivector(not_a_spill)


class TestMultivectorSpillPipeline:
    """The embed phase spills multivectors; later phases rehydrate one at a time."""

    @pytest.mark.asyncio
    async def test_embed_spills_multivector_and_clears_it(
        self, service, mock_blob, mock_embed,
    ):
        """After embed, the page carries a spill path and no in-memory vectors."""
        doc = make_document()
        page = DocumentPage(
            id="page_001", document_id=doc.id, workspace_id=doc.workspace_id,
            page_no=0, image_storage_path="/blobs/ws_test/page_0000.png",
        )
        mock_blob.retrieve_file.return_value = b"png-bytes"
        mock_blob.page_multivector_path.return_value = "/blobs/ws_test/mv_0000.f32"

        await service._embed_pages([page], doc)

        assert page.multivector is None, "vectors must not stay resident"
        assert page.multivector_storage_path == "/blobs/ws_test/mv_0000.f32"
        # The spilled payload is the encoded form of what the embed server sent.
        spilled = [
            c.args[1] for c in mock_blob.store_file.await_args_list
            if c.args[0] == "/blobs/ws_test/mv_0000.f32"
        ]
        assert spilled, "the multivector was never written to blob"
        assert isvc._decode_multivector(spilled[0]) == [[pytest.approx(0.1), pytest.approx(0.2)]]

    @pytest.mark.asyncio
    async def test_persist_rehydrates_spilled_multivector_for_page_row(
        self, service, mock_storage, mock_blob,
    ):
        """``create_page`` still receives the vectors, read back from blob."""
        doc = make_document()
        page = DocumentPage(
            id="page_001", document_id=doc.id, workspace_id=doc.workspace_id,
            page_no=0, transcript="text",
            multivector=None,
            multivector_storage_path="/blobs/ws_test/mv_0000.f32",
        )
        mock_blob.retrieve_file.return_value = isvc._encode_multivector([[0.25, 0.5]])

        # ``create_page`` is handed the page object itself, which is mutated
        # again the moment the write returns — so snapshot the value the
        # storage layer actually saw rather than reading it back afterwards.
        seen: list = []

        async def _capture(*, workspace_id, document_id, page):
            seen.append(page.multivector)
            return MagicMock(id="page_001")

        mock_storage.create_page.side_effect = _capture

        await service._persist_pages(doc, [page])

        assert seen == [[[pytest.approx(0.25), pytest.approx(0.5)]]]
        # ...and it is released again once the row is written.
        assert page.multivector is None

    @pytest.mark.asyncio
    async def test_visual_only_page_with_spill_still_creates_memory(
        self, service, mock_storage, mock_blob, mock_memory_service,
    ):
        """A transcript-less page whose multivector was spilled is not skipped.

        The skip-guard tests the in-memory ``multivector``, which the spill
        clears; without also honouring the spill path an OCR-free document
        would produce zero memories and be marked FAILED.
        """
        doc = make_document()
        service._memory_service = mock_memory_service
        page = DocumentPage(
            id="page_001", document_id=doc.id, workspace_id=doc.workspace_id,
            page_no=0, transcript=None,
            multivector=None,
            multivector_storage_path="/blobs/ws_test/mv_0000.f32",
        )
        mock_blob.retrieve_file.return_value = isvc._encode_multivector([[0.25, 0.5]])
        mock_storage.get_document_memories.return_value = []
        mock_storage.create_memory.return_value = MagicMock(id="mem_001")

        memory_ids = await service.create_memories_for_pages(doc, [page])

        assert memory_ids == ["mem_001"]
        assert mock_storage.create_memory.await_args.kwargs["multivector"] == [
            [pytest.approx(0.25), pytest.approx(0.5)]
        ]

    @pytest.mark.asyncio
    async def test_rehydrate_failure_is_non_fatal(
        self, service, mock_storage, mock_blob,
    ):
        """A bad spill blob degrades to no-multivector, it does not fail ingest."""
        doc = make_document()
        page = DocumentPage(
            id="page_001", document_id=doc.id, workspace_id=doc.workspace_id,
            page_no=0, transcript="text",
            multivector_storage_path="/blobs/ws_test/mv_0000.f32",
        )
        mock_blob.retrieve_file.return_value = b"corrupt"
        mock_storage.create_page.return_value = MagicMock(id="page_001")

        await service._persist_pages(doc, [page])

        written = mock_storage.create_page.await_args.kwargs["page"]
        assert written.multivector is None
