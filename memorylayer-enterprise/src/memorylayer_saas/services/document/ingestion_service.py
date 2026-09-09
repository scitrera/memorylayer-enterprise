"""Document ingestion service -- orchestrates the 6-phase pipeline.

Phases:
    1. Upload: validate, hash, dedup, store blob, create DB records, schedule task
    2. Render: convert document to page images (PDF -> PNG via pdf2image)
    3. Transcribe: OCR/VLM page images to markdown text via embed server
    4. Embed: generate single-vector and multi-vector embeddings via embed server
    5. Store: create memories from pages with pre-computed embeddings
    6. Finalize: update document and job status
"""
import asyncio
import base64
import hashlib
import io
import os
import struct
import sys
import uuid
from array import array
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from logging import Logger
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from scitrera_app_framework import Variables, ext_parse_bool, get_extension, get_logger

if TYPE_CHECKING:
    from memorylayer_server.services.memory import MemoryService

from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
from memorylayer_server.services.tasks import EXT_TASK_SERVICE
from memorylayer_server.services.memory import EXT_MEMORY_SERVICE
from memorylayer_server.services.ingest import (
    KNOWLEDGE_WORK_NORMALIZATION_KEY,
    normalize_connector_metadata,
)
from memorylayer_server.models.memory import RememberInput, MemoryType

from ..transcription import EXT_TRANSCRIPTION_SERVICE
from . import (
    DocumentIngestionPluginBase,
    EXT_EMBED_SERVER_CLIENT,
    EXT_BLOB_STORAGE_SERVICE,
)
from .embed_client import EmbedServerClient
from .blob_storage import BlobStorageService
from .image_embed import (
    generate_page_text_from_image_embeds,
    precompute_and_store_image_embeds,
)
from .inference_client import (
    default_inference_model,
    get_inference_client,
    slugify_model,
)
from .office_convert import convert_office_bytes_to_pdf
from .page_figures import PAGE_FIGURES_METADATA_KEY, store_page_figures
from ...config import (
    MEMORYLAYER_DOCUMENT_MAX_FILE_SIZE,
    DEFAULT_MEMORYLAYER_DOCUMENT_MAX_FILE_SIZE,
    DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED,
    DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_MAX_TOKENS,
    DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_PROMPT,
    DEFAULT_MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
    DEFAULT_MEMORYLAYER_VISUAL_TOKENIZER_ENABLED,
    MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED,
    MEMORYLAYER_DOCUMENT_CHAT_INGEST_MAX_TOKENS,
    MEMORYLAYER_DOCUMENT_CHAT_INGEST_PROMPT,
    MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
    MEMORYLAYER_VISUAL_TOKENIZER_ENABLED,
)
from ...models.document import (
    Document,
    DocumentEnrichmentStatus,
    IngestionJob,
    DocumentPage,
    DocumentStatus,
    JobStatus,
    DocumentType,
    DocumentExtractionOptions,
)

# Document types rendered by converting to PDF first (via LibreOffice) and
# then flowing through the shared PDF render path.  Maps the type to the
# source extension handed to soffice.
#
# TEXT/MARKDOWN use LibreOffice's plain-text import filter (the "txt" source
# extension) so a large text file becomes a paginated PDF -> bounded per-page
# page images/transcripts, instead of a single unbounded transcript that blows
# past the embed server's max_model_len.  Markdown renders as plain text, which
# is acceptable and bounded.
_OFFICE_DOCUMENT_TYPES: dict[DocumentType, str] = {
    DocumentType.HTML: "html",
    DocumentType.DOCX: "docx",
    DocumentType.PPTX: "pptx",
    DocumentType.TEXT: "txt",
    DocumentType.MARKDOWN: "txt",
}


# How many pages' worth of image bytes may be resident at once across the
# render -> transcribe -> embed phases. A ~100-page PDF OOM-killed the server
# because pdf2image rendered every page into memory at once (and the inline
# pipeline then held every page's base64 image through transcribe + embed). We
# process pages in fixed-size batches so peak memory is O(batch pages), not
# O(all pages). Guarded to >= 1 so a misconfigured 0/negative can't stall.
_RENDER_BATCH_PAGES = max(
    1, int(os.environ.get("MEMORYLAYER_RENDER_BATCH_PAGES", "8"))
)

# Rasterization DPI for page images. Peak render memory is
# O(_RENDER_BATCH_PAGES * page_area * dpi^2): ``convert_from_bytes`` returns the
# whole batch as decoded PIL images at once, so the batch — not one page — sets
# the high-water mark. At 200 DPI a US-Letter page decodes to ~10.7 MB, making a
# batch of 8 an ~86 MB spike; 150 DPI more than halves that to ~6.0 MB/page
# (~48 MB/batch) while staying well above the resolution the downstream vision
# and OCR models actually consume. Override when a corpus needs finer detail.
_RENDER_DPI = max(
    1, int(os.environ.get("MEMORYLAYER_RENDER_DPI", "150"))
)


def _rasterize_pdf_batch(pdf_bytes: bytes, first_page: int, last_page: int) -> list[bytes]:
    """Rasterize + PNG-encode a 1-indexed inclusive page range, OFF the event loop.

    Both legs are CPU-bound: ``convert_from_bytes`` rasterizes via poppler and
    ``img.save`` PNG-encodes each page. Doing the encode on the event loop (as the
    caller used to) blocked the loop for the whole document — starving the
    ``/livez`` probe and triggering liveness restarts. Run via
    ``asyncio.to_thread`` and return only the encoded bytes so no PIL image
    crosses back onto the loop.
    """
    from pdf2image import convert_from_bytes

    imgs = convert_from_bytes(
        pdf_bytes, dpi=_RENDER_DPI, fmt="png",
        first_page=first_page, last_page=last_page,
    )
    out: list[bytes] = []
    # Close + drop each PIL image as soon as it is encoded rather than holding
    # the whole decoded batch alive until the loop ends: ``imgs`` is the render
    # phase's peak allocation, and the encoded PNGs we return are an order of
    # magnitude smaller than the raw pixel buffers they came from.
    while imgs:
        img = imgs.pop(0)
        try:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append(buf.getvalue())
        finally:
            img.close()
    return out


def _b64_encode_all(blobs: list[bytes]) -> list[str]:
    """base64-encode a batch of image blobs OFF the event loop (CPU-bound for
    multi-MB page PNGs; previously encoded inline on the loop)."""
    return [base64.b64encode(b).decode("ascii") for b in blobs]


# Multivector spill codec. A ColPali page multivector is ~1030 vectors x 128
# dims. Held as a Python ``list[list[float]]`` that is ~5 MB per page (a float
# object is 24 bytes plus 8 for the list slot, vs 4 bytes as raw float32), so a
# 100-page document that kept every page's multivector resident until the
# persist phase carried ~500 MB of pure boxing overhead. We spill each page's
# vectors to blob storage as flat float32 and rehydrate one page at a time.
#
# Layout: little-endian ``<II`` header (vector count, dim) then count*dim
# float32 values, row-major. Explicitly little-endian so a blob written on one
# architecture stays readable on another.
_MULTIVECTOR_HEADER = struct.Struct("<II")


def _encode_multivector(multivector: list[list[float]]) -> bytes:
    """Pack a multivector into the flat float32 spill format."""
    count = len(multivector)
    dim = len(multivector[0]) if count else 0
    flat = array("f", (value for vec in multivector for value in vec))
    if sys.byteorder != "little":
        flat.byteswap()
    return _MULTIVECTOR_HEADER.pack(count, dim) + flat.tobytes()


def _decode_multivector(blob: bytes) -> list[list[float]]:
    """Unpack the flat float32 spill format back to ``list[list[float]]``.

    Returns the same shape the embed server handed us, so every downstream
    consumer (the pgvector codec in particular) is unchanged.

    Raises:
        ValueError: If the blob is not a well-formed spill. The header is
            attacker-irrelevant but not trustworthy — a truncated write or a
            path collision with some other blob yields a garbage ``count``, and
            allocating from it would spin on a multi-billion-iteration loop
            rather than fail. Validate the declared shape against the actual
            payload length so a bad blob fails fast and audibly instead.
    """
    if len(blob) < _MULTIVECTOR_HEADER.size:
        raise ValueError(
            "multivector blob is %d bytes, shorter than its %d-byte header"
            % (len(blob), _MULTIVECTOR_HEADER.size)
        )

    count, dim = _MULTIVECTOR_HEADER.unpack_from(blob, 0)
    payload = memoryview(blob)[_MULTIVECTOR_HEADER.size:]
    expected = count * dim * 4
    if len(payload) != expected:
        raise ValueError(
            "multivector blob declares %d x %d float32 (%d bytes) but carries %d"
            % (count, dim, expected, len(payload))
        )

    flat = array("f")
    flat.frombytes(payload)
    if sys.byteorder != "little":
        flat.byteswap()
    return [
        flat[i * dim:(i + 1) * dim].tolist()
        for i in range(count)
    ]


class DocumentIngestionService:
    """Orchestrates document ingestion: upload, render, transcribe, embed, store."""

    def __init__(
        self,
        v: Variables,
        storage_backend,
        blob_storage: BlobStorageService,
        embed_client: EmbedServerClient,
        task_service,
        memory_service: "MemoryService",
        max_file_size: int,
        logger: Logger,
    ):
        self._v = v
        self._storage = storage_backend
        self._blob = blob_storage
        self._embed = embed_client
        self._tasks = task_service
        self._memory_service = memory_service
        self._max_file_size = max_file_size
        self.logger = logger

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def upload_document(
        self,
        workspace_id: str,
        file_data: bytes,
        filename: str,
        document_type: DocumentType | None = None,
        extraction_options: DocumentExtractionOptions | None = None,
        metadata: dict | None = None,
    ) -> tuple[Document, IngestionJob]:
        """Upload a document and schedule ingestion.

        Phase 1: validate size, compute hash, check for duplicates, persist
        the original blob, create DB records, and schedule background processing.

        Args:
            workspace_id: Target workspace identifier.
            file_data: Raw document bytes.
            filename: Original filename (used for type detection).
            document_type: Explicit document type override (auto-detected if None).
            extraction_options: Extraction/chunking configuration.
            metadata: Arbitrary user metadata attached to the document.

        Returns:
            Tuple of (Document, IngestionJob) created records.

        Raises:
            ValueError: If file exceeds size limit, type is unsupported, or
                a duplicate content hash already exists in the workspace.
        """
        if len(file_data) > self._max_file_size:
            raise ValueError(
                "File size %d exceeds maximum %d" % (len(file_data), self._max_file_size)
            )

        if document_type is None:
            document_type = self._detect_document_type(filename)

        content_hash = hashlib.sha256(file_data).hexdigest()

        existing = await self._storage.find_document_by_hash(workspace_id, content_hash)
        if existing:
            raise ValueError("Duplicate document: %s has same content hash" % existing.id)

        if extraction_options is None:
            extraction_options = DocumentExtractionOptions()

        doc_id = "doc_%s" % uuid.uuid4().hex[:12]
        job_id = "job_%s" % uuid.uuid4().hex[:12]

        # Store original file in blob storage when configured to retain
        storage_path = None
        if extraction_options.retain_original:
            blob_path = self._blob.document_path(workspace_id, doc_id, filename)
            storage_path = await self._blob.store_file(blob_path, file_data)

        doc = Document(
            id=doc_id,
            workspace_id=workspace_id,
            filename=filename,
            document_type=document_type,
            content_hash=content_hash,
            size_bytes=len(file_data),
            status=DocumentStatus.PENDING,
            target_context_id=extraction_options.target_context_id,
            extraction_options=extraction_options,
            storage_path=storage_path,
            retain_original=extraction_options.retain_original,
            metadata=metadata or {},
        )
        doc = await self._storage.create_document(doc)

        job = IngestionJob(
            id=job_id,
            workspace_id=workspace_id,
            document_ids=[doc_id],
            status=JobStatus.QUEUED,
        )
        # Coalesce: at most one in-flight job per document. Supersede any prior
        # queued/running job for this document before minting the new one so
        # re-deliveries/replays don't leave orphaned in-flight jobs behind.
        await self._supersede_active_jobs([doc_id])
        job = await self._storage.create_job(job)

        await self._tasks.schedule_task(
            "document_render",
            {
                "document_id": doc_id,
                "job_id": job_id,
                "workspace_id": workspace_id,
            },
            priority=3,
        )

        self.logger.info(
            "Uploaded document %s (%s, %d bytes), job %s queued",
            doc_id, filename, len(file_data), job_id,
        )
        return doc, job

    async def store_upload_blob(
        self, doc: Document, file_data: bytes,
    ) -> Document:
        """Persist fetched upload bytes to blob storage for the chained pipeline.

        The VFS (``doc_added``) path creates the Document row and fetches bytes
        itself, then hands off to the chained ``document_render`` task (which
        re-reads the source bytes from ``doc.storage_path``). This stores the
        blob, stamps the real content_hash + size, and updates ``storage_path``
        so render can read it — the same blob handoff ``upload_document``
        performs before scheduling render, factored out so the by-bytes path
        produces identical state.

        Args:
            doc: Document domain model (already persisted, PENDING_FETCH).
            file_data: Raw document bytes fetched from data-connectors.

        Returns:
            The doc with ``storage_path``/``content_hash``/``size_bytes`` set on
            the in-memory object (and persisted).
        """
        updates: dict = {"size_bytes": len(file_data)}
        doc.size_bytes = len(file_data)

        # Compute the real content hash; replace the "pending" placeholder used
        # at row-creation time so dedup-by-hash works for re-deliveries.
        actual_hash = hashlib.sha256(file_data).hexdigest()
        if actual_hash != doc.content_hash:
            updates["content_hash"] = actual_hash
            doc.content_hash = actual_hash

        if doc.extraction_options.retain_original:
            blob_path = self._blob.document_path(
                doc.workspace_id, doc.id, doc.filename,
            )
            storage_path = await self._blob.store_file(blob_path, file_data)
            updates["storage_path"] = storage_path
            doc.storage_path = storage_path

        await self._storage.update_document(doc.id, **updates)
        return doc

    async def get_document(
        self, document_id: str, workspace_id: str = None,
    ) -> Document | None:
        """Retrieve a document record by ID.

        Args:
            document_id: Document identifier.
            workspace_id: Optional workspace scope.

        Returns:
            Document or None if not found.
        """
        return await self._storage.get_document(document_id, workspace_id)

    async def list_documents(
        self,
        workspace_id: str,
        status: str = None,
        document_type: str = None,
        created_after: str = None,
        created_before: str = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        """List documents in a workspace with optional filters.

        Args:
            workspace_id: Workspace identifier.
            status: Optional status filter (e.g. 'completed').
            document_type: Optional document type filter (e.g. 'pdf').
            created_after: Optional ISO datetime lower bound for created_at.
            created_before: Optional ISO datetime upper bound for created_at.
            limit: Maximum documents to return.
            offset: Pagination offset.

        Returns:
            Tuple of (documents_list, total_count).
        """
        return await self._storage.list_documents(
            workspace_id,
            status=status,
            document_type=document_type,
            created_after=created_after,
            created_before=created_before,
            limit=limit,
            offset=offset,
        )

    async def delete_document(
        self,
        document_id: str,
        workspace_id: str = None,
        delete_memories: bool = False,
    ) -> None:
        """Delete a document and optionally its extracted memories.

        Args:
            document_id: Document identifier.
            workspace_id: Optional workspace scope.
            delete_memories: If True, also delete memories created from this document.

        Raises:
            ValueError: If document not found.
        """
        doc = await self._storage.get_document(document_id, workspace_id)
        if not doc:
            raise ValueError("Document not found: %s" % document_id)

        if delete_memories and doc.memory_ids:
            for mem_id in doc.memory_ids:
                try:
                    await self._storage.delete_memory(doc.workspace_id, mem_id)
                except Exception as exc:
                    self.logger.warning("Failed to delete memory %s: %s", mem_id, exc)

        blob_prefix = self._blob.document_path(doc.workspace_id, doc.id, "")
        # Strip the trailing empty filename to get the directory
        blob_prefix = blob_prefix.rstrip("/")
        await self._blob.delete_tree(blob_prefix)

        await self._storage.delete_document(document_id)
        self.logger.info(
            "Deleted document %s (delete_memories=%s)", document_id, delete_memories,
        )

    async def reprocess_document(
        self,
        document_id: str,
        workspace_id: str = None,
        options: DocumentExtractionOptions = None,
        from_phase: str = "render",
    ) -> IngestionJob:
        """Re-run the extraction pipeline for an existing document.

        Args:
            document_id: Document identifier.
            workspace_id: Optional workspace scope.
            options: New extraction options to apply before reprocessing.
            from_phase: Pipeline phase to restart from. ``"render"`` (default)
                clears the prior derived state (page rows and derived blob
                subtrees, preserving the original upload) and re-renders from
                scratch. ``"embed"`` reuses the existing rendered pages and
                re-runs only the embed-onward stages.

        Returns:
            Newly created IngestionJob tracking the reprocessing.

        Raises:
            ValueError: If document not found, ``from_phase`` is invalid, or
                ``from_phase="embed"`` is requested but no pages exist.
        """
        if from_phase not in ("render", "embed"):
            raise ValueError(
                "Invalid from_phase %r (expected 'render' or 'embed')" % from_phase
            )

        doc = await self._storage.get_document(document_id, workspace_id)
        if not doc:
            raise ValueError("Document not found: %s" % document_id)

        if options:
            await self._storage.update_document(
                document_id, extraction_options=options.model_dump(),
            )

        if from_phase == "embed":
            # Re-embed only: pages must already exist; do NOT clear pages/blobs
            # or zero page_count.
            pages = await self._storage.get_pages(document_id, doc.workspace_id)
            if not pages:
                raise ValueError(
                    "No rendered pages for document %s; reprocess from render"
                    % document_id
                )
            await self._storage.update_document(
                document_id,
                status=DocumentStatus.PENDING.value,
                chunk_count=0,
                memory_ids=[],
                processing_started_at=None,
                processing_completed_at=None,
            )
            next_task = "document_embed"
        else:
            # Full reprocess from render: clear prior derived state so the
            # render phase can re-create pages without violating the
            # uq_document_page (document_id, page_no) constraint.
            await self._clear_derived_state(doc)
            await self._storage.update_document(
                document_id,
                status=DocumentStatus.PENDING.value,
                page_count=0,
                chunk_count=0,
                memory_ids=[],
                processing_started_at=None,
                processing_completed_at=None,
            )
            next_task = "document_render"

        job_id = "job_%s" % uuid.uuid4().hex[:12]
        job = IngestionJob(
            id=job_id,
            workspace_id=doc.workspace_id,
            document_ids=[document_id],
            status=JobStatus.QUEUED,
        )
        # Coalesce: supersede any in-flight job for this document so an explicit
        # reprocess replaces (rather than stacks on top of) a still-running one.
        await self._supersede_active_jobs([document_id])
        job = await self._storage.create_job(job)

        await self._tasks.schedule_task(
            next_task,
            {
                "document_id": document_id,
                "job_id": job_id,
                "workspace_id": doc.workspace_id,
            },
            priority=3,
        )

        self.logger.info(
            "Reprocessing document %s from %s phase, new job %s queued",
            document_id, from_phase, job_id,
        )
        return job

    async def _clear_derived_state(self, doc: Document) -> None:
        """Delete a document's derived page rows and blob subtrees.

        Clears everything the render/transcribe/embed phases produce so a full
        reprocess starts clean, while PRESERVING the original uploaded file
        (the doc-root blob the render phase re-reads from).

        Page rows are deleted from the registry; the derived blob subtrees
        (pages, image_embeds, legacy prompt_embeds, transcripts) are removed
        best-effort -- a missing subtree must not abort the reprocess.

        Args:
            doc: Document domain model.
        """
        deleted = await self._storage.delete_pages(doc.id, doc.workspace_id)
        self.logger.info(
            "Cleared %d page rows for document %s", deleted, doc.id,
        )

        derived_prefixes = (
            self._blob.document_pages_prefix(doc.workspace_id, doc.id),
            self._blob.document_image_embeds_prefix(doc.workspace_id, doc.id),
            self._blob.document_prompt_embeds_prefix(doc.workspace_id, doc.id),
            self._blob.document_transcripts_prefix(doc.workspace_id, doc.id),
        )
        for prefix in derived_prefixes:
            try:
                await self._blob.delete_tree(prefix)
            except Exception as exc:
                self.logger.warning(
                    "Failed to delete derived blob subtree %s: %s", prefix, exc,
                )

    async def get_job(self, job_id: str) -> IngestionJob | None:
        """Get an ingestion job by ID.

        Args:
            job_id: Job identifier.

        Returns:
            IngestionJob or None if not found.
        """
        return await self._storage.get_job(job_id)

    async def list_jobs(
        self, workspace_id: str, status: str = None, limit: int = 50,
    ) -> list[IngestionJob]:
        """List ingestion jobs for a workspace.

        Args:
            workspace_id: Workspace identifier.
            status: Optional status filter.
            limit: Maximum jobs to return.

        Returns:
            List of IngestionJob records.
        """
        return await self._storage.list_jobs(
            workspace_id, status=status, limit=limit,
        )

    async def cancel_job(self, job_id: str) -> None:
        """Cancel a queued or running job.

        Args:
            job_id: Job identifier.

        Raises:
            ValueError: If job not found or already in a terminal state.
        """
        job = await self._storage.get_job(job_id)
        if not job:
            raise ValueError("Job not found: %s" % job_id)
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            raise ValueError(
                "Job %s already in terminal state: %s" % (job_id, job.status.value)
            )
        await self._storage.update_job(
            job_id,
            status=JobStatus.CANCELLED.value,
            completed_at=datetime.now(timezone.utc),
        )
        self.logger.info("Cancelled job %s", job_id)

    async def _supersede_active_jobs(
        self, document_ids: list[str], keep_job_id: str | None = None,
    ) -> int:
        """Cancel in-flight (queued/running) jobs overlapping ``document_ids``.

        Enforces at-most-one active ingestion job per document. Every existing
        non-terminal job whose ``document_ids`` overlap the given set is
        superseded (transitioned to CANCELLED via the same mechanism as
        ``cancel_job``), EXCEPT ``keep_job_id`` — the job driving the current
        operation. The newest job therefore wins, so re-deliveries, reprocess
        requests, and Aether task replays stop accumulating orphaned jobs.

        Safe/idempotent: a job that races to a terminal state (or vanishes)
        between the list and the cancel is swallowed. Best-effort — a storage
        backend that does not implement ``list_active_jobs_for_documents`` is a
        no-op.

        Args:
            document_ids: Documents whose in-flight jobs should be superseded.
            keep_job_id: Job id to leave intact (the current driver), if any.

        Returns:
            Number of jobs superseded.
        """
        lister = getattr(self._storage, "list_active_jobs_for_documents", None)
        if lister is None:
            return 0
        try:
            active = await lister(list(document_ids))
        except NotImplementedError:
            return 0

        cancelled = 0
        for job in active:
            if keep_job_id is not None and job.id == keep_job_id:
                continue
            try:
                await self.cancel_job(job.id)
                cancelled += 1
            except ValueError:
                # Raced to terminal (or vanished) between list and cancel — fine.
                continue
        if cancelled:
            self.logger.info(
                "Superseded %d in-flight ingestion job(s) for documents %s",
                cancelled, list(document_ids),
            )
        return cancelled

    async def process_bytes(
        self,
        doc: Document,
        file_data: bytes,
        job_id: str,
        progress_cb: Optional[Callable[[str, float], Awaitable[None]]] = None,
    ) -> None:
        """Run the full ingestion pipeline against raw bytes.

        This is the entry point for the data-connectors VFS-based ingestion
        path.  The caller (``handle_doc_added``) has already created the
        Document row and fetched the bytes via JIT URL mint.  This method
        stores the blob (if retain_original), then runs render -> transcribe
        -> embed -> persist -> store memories -> finalize.

        Args:
            doc: Document domain model (already persisted with PENDING_FETCH).
            file_data: Raw document bytes fetched from data-connectors.
            job_id: Tracking job identifier.
            progress_cb: Optional async callback invoked after each pipeline
                phase as ``progress_cb(step_name, completion)``.  Best-effort:
                the caller is responsible for swallowing its own errors.

        Raises:
            Exception: On pipeline failure (caller decides retry vs terminal).
        """
        try:
            await self._storage.update_document(
                doc.id,
                status=DocumentStatus.PROCESSING.value,
                size_bytes=len(file_data),
                processing_started_at=datetime.now(timezone.utc),
            )
            await self._storage.update_job(
                job_id,
                status=JobStatus.RUNNING.value,
                started_at=datetime.now(timezone.utc),
            )

            # Store original file in blob storage when configured to retain
            if doc.extraction_options.retain_original:
                blob_path = self._blob.document_path(
                    doc.workspace_id, doc.id, doc.filename,
                )
                storage_path = await self._blob.store_file(blob_path, file_data)
                await self._storage.update_document(doc.id, storage_path=storage_path)
                doc.storage_path = storage_path

            # Compute content hash from actual bytes to update if placeholder was used
            actual_hash = hashlib.sha256(file_data).hexdigest()
            if actual_hash != doc.content_hash:
                await self._storage.update_document(doc.id, content_hash=actual_hash)
                doc.content_hash = actual_hash

            # Render pages (needs storage_path for the blob; for by-bytes path
            # we write the blob first so _render_pages can read it back)
            pages = await self._render_pages(doc)
            await self._storage.update_document(doc.id, page_count=len(pages))
            await self._storage.update_job(job_id, progress_percent=20)
            if progress_cb:
                await progress_cb("render", 0.25)

            # Transcribe (skipped when OCR transcription is disabled; document-chat
            # ingestion in the embed phase then sources page text from image-embeds)
            if self._transcribe_enabled():
                pages = await self._transcribe_pages(pages, doc)
            await self._storage.update_job(job_id, progress_percent=40)
            if progress_cb:
                await progress_cb("transcribe", 0.45)

            # Embed
            pages = await self._embed_pages(pages, doc)
            await self._storage.update_job(job_id, progress_percent=60)
            if progress_cb:
                await progress_cb("embed", 0.60)

            # Persist pages
            pages = await self._persist_pages(doc, pages)
            await self._storage.update_job(job_id, progress_percent=70)
            if progress_cb:
                await progress_cb("persist", 0.75)

            # Store as memories
            memory_ids = await self._store_as_memories(doc, pages, job_id=job_id)
            await self._storage.update_job(job_id, progress_percent=85)
            if progress_cb:
                await progress_cb("store", 0.90)

            # Finalize
            await self._finalize(doc, memory_ids, job_id)
            if progress_cb:
                await progress_cb("finalize", 0.97)

        except Exception as exc:
            self.logger.error(
                "process_bytes failed for document %s: %s",
                doc.id, exc, exc_info=True,
            )
            now = datetime.now(timezone.utc)
            await self._storage.update_document(
                doc.id,
                status=DocumentStatus.FAILED.value,
                processing_completed_at=now,
            )
            await self._storage.update_job(
                job_id,
                status=JobStatus.FAILED.value,
                completed_at=now,
                errors=[{"document_id": doc.id, "error": str(exc)}],
            )
            raise

    # ------------------------------------------------------------------ #
    # Pipeline (called from background task handler)
    # ------------------------------------------------------------------ #

    async def process_document(
        self, document_id: str, job_id: str, workspace_id: str,
    ) -> None:
        """Run the full processing pipeline for a document.

        Phases 2-6 are executed sequentially. On failure the document and job
        are moved to FAILED state with the error recorded.

        Args:
            document_id: Document to process.
            job_id: Tracking job identifier.
            workspace_id: Workspace scope.
        """
        try:
            await self._storage.update_document(
                document_id,
                status=DocumentStatus.PROCESSING.value,
                processing_started_at=datetime.now(timezone.utc),
            )
            await self._storage.update_job(
                job_id,
                status=JobStatus.RUNNING.value,
                started_at=datetime.now(timezone.utc),
            )

            doc = await self._storage.get_document(document_id, workspace_id)
            if not doc:
                raise ValueError("Document not found: %s" % document_id)

            # Phase 2: Render pages
            pages = await self._render_pages(doc)
            await self._storage.update_document(document_id, page_count=len(pages))
            await self._storage.update_job(job_id, progress_percent=20)

            # Phase 3: Transcribe pages (skipped when OCR transcription is
            # disabled; the embed phase's document-chat ingestion then sources
            # page text from image-embeds instead).
            if self._transcribe_enabled():
                pages = await self._transcribe_pages(pages, doc)
            await self._storage.update_job(job_id, progress_percent=40)

            # Phase 4: Embed pages
            pages = await self._embed_pages(pages, doc)
            await self._storage.update_job(job_id, progress_percent=60)

            # Phase 4.5: Persist pages to registry
            pages = await self._persist_pages(doc, pages)
            await self._storage.update_job(job_id, progress_percent=70)

            # Phase 5: Store as memories
            memory_ids = await self._store_as_memories(doc, pages, job_id=job_id)
            await self._storage.update_job(job_id, progress_percent=85)

            # Phase 6: Finalize
            await self._finalize(doc, memory_ids, job_id)

        except Exception as exc:
            self.logger.error(
                "Document processing failed for %s: %s",
                document_id, exc, exc_info=True,
            )
            now = datetime.now(timezone.utc)
            await self._storage.update_document(
                document_id,
                status=DocumentStatus.FAILED.value,
                processing_completed_at=now,
            )
            await self._storage.update_job(
                job_id,
                status=JobStatus.FAILED.value,
                completed_at=now,
                errors=[{"document_id": document_id, "error": str(exc)}],
            )

    # ------------------------------------------------------------------ #
    # Pipeline phases (private)
    # ------------------------------------------------------------------ #

    async def _render_pages(self, doc: Document) -> list[DocumentPage]:
        """Phase 2: convert document to page images.

        PDF documents are rendered to PNG images via pdf2image.  HTML, DOCX,
        PPTX, TEXT and MARKDOWN documents are first converted to PDF via
        LibreOffice and then flow through the same PDF render path, so text and
        markdown files become bounded per-page page images/transcripts rather
        than a single unbounded transcript.

        Args:
            doc: Document domain model.

        Returns:
            List of DocumentPage instances.

        Raises:
            ValueError: If the document has no storage path or its type is
                unsupported.
            OfficeConversionError: If LibreOffice conversion is unavailable or
                fails for an office/text document.
        """
        if doc.document_type == DocumentType.PDF:
            return await self._render_pdf_pages(doc)

        if doc.document_type in _OFFICE_DOCUMENT_TYPES:
            return await self._render_office_pages(doc)

        raise ValueError(
            "Rendering not supported for document type %s" % doc.document_type.value
        )

    async def _render_office_pages(self, doc: Document) -> list[DocumentPage]:
        """Render an HTML/DOCX/PPTX/TEXT/MARKDOWN document by converting it to PDF first.

        The uploaded bytes are converted to PDF via LibreOffice and then fed
        into the shared PDF render path so office/HTML/text documents flow
        through the same page/visual pipeline as native PDFs.

        Args:
            doc: Document domain model (HTML/DOCX/PPTX/TEXT/MARKDOWN, with a
                storage path).

        Returns:
            List of DocumentPage instances with image data and blob paths.

        Raises:
            ValueError: If the document has no storage path.
            OfficeConversionError: If LibreOffice conversion is unavailable or
                fails.
        """
        if not doc.storage_path:
            raise ValueError(
                "No storage path for %s document" % doc.document_type.value
            )

        source_bytes = await self._blob.retrieve_file(doc.storage_path)
        source_ext = _OFFICE_DOCUMENT_TYPES[doc.document_type]
        pdf_bytes = await convert_office_bytes_to_pdf(source_bytes, source_ext)

        self.logger.info(
            "Converted %s document %s to PDF (%d bytes)",
            doc.document_type.value, doc.id, len(pdf_bytes),
        )
        return await self._render_pdf_pages(doc, pdf_bytes=pdf_bytes)

    async def _render_pdf_pages(
        self, doc: Document, pdf_bytes: Optional[bytes] = None,
    ) -> list[DocumentPage]:
        """Render PDF pages to PNG images using pdf2image.

        Runs the CPU-bound pdf2image conversion in a thread pool to avoid
        blocking the event loop.

        Args:
            doc: Document domain model (must have a storage path unless
                ``pdf_bytes`` is supplied).
            pdf_bytes: Pre-converted PDF bytes (used by the office/HTML path).
                When ``None`` the bytes are retrieved from blob storage.

        Returns:
            List of DocumentPage instances with image data and blob paths.

        Raises:
            ValueError: If the document has no storage path and no bytes.
        """
        from pdf2image import pdfinfo_from_bytes

        if pdf_bytes is None:
            if not doc.storage_path:
                raise ValueError("No storage path for PDF document")
            pdf_bytes = await self._blob.retrieve_file(doc.storage_path)

        # Render in fixed-size page batches instead of rasterizing the whole PDF
        # at once — a ~100-page document rendered all-pages-at-once OOM-killed the
        # server. Get the page count up front, then rasterize only
        # ``_RENDER_BATCH_PAGES`` pages per pass so at most one batch of PIL
        # images / PNG bytes is resident at a time. The blob store is the source
        # of truth for page images: we persist each page's PNG and set
        # ``image_storage_path``, leaving ``image_b64`` None (transcribe/embed
        # re-read the bytes from blob per-batch), so nothing downstream needs the
        # inline base64 and it never accumulates here.
        info = await asyncio.to_thread(pdfinfo_from_bytes, pdf_bytes)
        total = int(info["Pages"])

        pages: list[DocumentPage] = []
        # Stamp the source VFS reference onto every page at ingest. This is the
        # only point where the owning Document is in hand; the page payload the
        # API returns carries page fields only, so a reader holding a page row
        # would otherwise have no route back to the source entry. Consumers
        # resolve a human-readable path from the VFS catalog via this ref —
        # ``vfs_ref`` is a handle, not a path, so it is deliberately kept out of
        # any path-shaped field. Omitted entirely when the document has no ref
        # (direct uploads), rather than persisting a null.
        base_metadata: dict[str, Any] = {}
        source_vfs_ref = getattr(doc, "source_vfs_ref", None)
        if source_vfs_ref:
            base_metadata["vfs_ref"] = source_vfs_ref
        # pdf2image first_page/last_page are 1-indexed and inclusive. Rasterize
        # AND PNG-encode each batch in ONE worker thread (_rasterize_pdf_batch),
        # returning just the encoded bytes — so the CPU-bound work stays OFF the
        # event loop and only the async store_file runs on it. (Encoding on the
        # loop previously blocked it for the whole document and starved /livez.)
        for start in range(1, total + 1, _RENDER_BATCH_PAGES):
            end = min(start + _RENDER_BATCH_PAGES - 1, total)
            png_batch = await asyncio.to_thread(
                _rasterize_pdf_batch, pdf_bytes, start, end,
            )
            for offset, img_bytes in enumerate(png_batch):
                # Global 0-based page number, preserved across batches.
                i = start - 1 + offset

                img_path = self._blob.page_image_path(doc.workspace_id, doc.id, i)
                await self._blob.store_file(img_path, img_bytes)

                pages.append(DocumentPage(
                    id="page_%s" % uuid.uuid4().hex[:12],
                    document_id=doc.id,
                    workspace_id=doc.workspace_id,
                    page_no=i,
                    image_storage_path=img_path,
                    image_b64=None,
                    metadata=dict(base_metadata),
                ))
            # Release this batch's PNG bytes before the next batch so peak memory
            # stays O(batch pages).
            del png_batch

        self.logger.info(
            "Rendered %d pages for document %s", len(pages), doc.id,
        )
        return pages

    async def _transcribe_pages(
        self, pages: list[DocumentPage], doc: Document,
    ) -> list[DocumentPage]:
        """Phase 3: transcribe page images to text.

        Routed through the configured transcription service, so this path does
        not care whether an embed-server or an in-process cascade against model
        endpoints is serving.

        Pages that already have a transcript (text/markdown) are skipped.

        Args:
            pages: List of document pages (some may already have transcripts).
            doc: Document domain model (for extraction options).

        Returns:
            The same list of pages, with transcripts populated.
        """
        # Images are no longer held inline (render sets image_b64=None and the
        # blob is authoritative), so select on image_storage_path and re-read the
        # PNG bytes from blob per batch.
        pages_needing_transcription = [
            p for p in pages if p.transcript is None and p.image_storage_path
        ]

        if not pages_needing_transcription:
            return pages

        system_prompt = doc.extraction_options.system_prompt

        transcription = get_extension(EXT_TRANSCRIPTION_SERVICE, self._v)

        try:
            await transcription.connect()
            # Transcribe in fixed-size batches so only ~_RENDER_BATCH_PAGES pages'
            # image bytes are resident at once (a whole-document image list is
            # what previously drove the memory spike). Each batch reads its images
            # from blob, transcribes, assigns transcripts back, then releases the
            # bytes before the next batch.
            for start in range(
                0, len(pages_needing_transcription), _RENDER_BATCH_PAGES
            ):
                batch = pages_needing_transcription[start:start + _RENDER_BATCH_PAGES]
                # Read the batch's image bytes (blob I/O), then base64-encode them
                # OFF the event loop — the encode of multi-MB page PNGs is CPU-bound
                # and previously ran inline on the loop.
                raw_images = [
                    await self._blob.retrieve_file(p.image_storage_path)
                    for p in batch
                ]
                images_b64 = await asyncio.to_thread(_b64_encode_all, raw_images)
                del raw_images
                page_results = await transcription.transcribe_pages(
                    images_b64, system_prompt=system_prompt,
                )

                # The embed server numbers pages by their position WITHIN the
                # request (0-based per call), so map each result back to this
                # batch by request-relative index — not by the page's global
                # page_no, which no longer equals the request index once we send
                # pages in batches.
                for page_result in page_results:
                    req_idx = page_result.request_index
                    if not (0 <= req_idx < len(batch)):
                        continue
                    p = batch[req_idx]
                    p.transcript = page_result.content
                    p.transcript_model = page_result.model
                    transcript_path = self._blob.page_transcript_path(
                        doc.workspace_id, doc.id, p.page_no,
                    )
                    await self._blob.store_file(
                        transcript_path, p.transcript.encode("utf-8"),
                    )
                    figures = await store_page_figures(
                        blob_storage=self._blob,
                        workspace_id=doc.workspace_id,
                        doc_id=doc.id,
                        page_no=p.page_no,
                        page_image_b64=images_b64[req_idx],
                        regions=page_result.regions,
                        captions=page_result.figure_captions,
                        logger=self.logger,
                    )
                    if figures:
                        # Records ride in page metadata so the stored crops are
                        # discoverable — an unenumerable blob is unfetchable.
                        # This path persists pages later (create_page), so
                        # mutating the in-memory page is enough.
                        p.metadata = {
                            **(p.metadata or {}),
                            PAGE_FIGURES_METADATA_KEY: figures,
                        }

                # Drop this batch's image bytes before rendering the next batch.
                del images_b64
        finally:
            await transcription.close()

        transcribed = sum(1 for p in pages if p.transcript)
        self.logger.info(
            "Transcribed %d/%d pages for document %s",
            transcribed, len(pages), doc.id,
        )
        return pages

    async def _embed_pages(
        self, pages: list[DocumentPage], doc: Document,
    ) -> list[DocumentPage]:
        """Phase 4: generate embeddings, image-embeds, and chat-ingest text.

        Mirrors the distributed ``document_embed`` task using the same shared
        building blocks, but operates on the in-memory pages (the inline
        pipeline persists pages in a later phase):

        - multi-vector (ColPali) embeddings from page images;
        - per-page visual-tokenizer image-embeds (when enabled), stored to blob
          now and to the page row by the later persist phase;
        - document-chat ingestion: when a page has no transcript (OCR
          transcription disabled), generate its text from its image-embeds via
          the inference LLM so it still becomes a memory;
        - single-vector text embeddings over every page that ends up with a
          transcript (original or chat-generated).

        Image-derived signals are additive and non-fatal: a per-signal failure
        is logged and skipped, never aborting the pipeline. The produced
        embeddings/text are set on the in-memory page objects so the subsequent
        persist + store-as-memories phases pick them up.

        Args:
            pages: List of document pages.
            doc: Document domain model (for options, filename, and flags).

        Returns:
            The same list of pages, with embeddings (and any chat-generated
            transcripts) populated.
        """
        pages_with_images = [p for p in pages if p.image_storage_path]

        try:
            await self._embed.connect()

            if pages_with_images:
                vt_enabled = self._v.environ(
                    MEMORYLAYER_VISUAL_TOKENIZER_ENABLED,
                    default=DEFAULT_MEMORYLAYER_VISUAL_TOKENIZER_ENABLED,
                    type_fn=ext_parse_bool,
                )
                precompute_embeds = vt_enabled or self._chat_ingest_enabled()

                # Process image-derived signals in fixed-size page batches so only
                # ~_RENDER_BATCH_PAGES pages' image bytes are resident at once,
                # instead of reading every page image into one big list (the
                # whole-document read was a per-signal memory spike). Each signal
                # stays independently non-fatal, matching the unbatched behavior.
                for start in range(0, len(pages_with_images), _RENDER_BATCH_PAGES):
                    batch = pages_with_images[start:start + _RENDER_BATCH_PAGES]

                    # Multi-vector (ColPali) embeddings — non-fatal.
                    try:
                        images_b64 = []
                        for p in batch:
                            img_bytes = await self._blob.retrieve_file(p.image_storage_path)
                            images_b64.append(base64.b64encode(img_bytes).decode("ascii"))
                        mv_results = await self._embed.embed_images_multivector(images_b64)
                        # Spill each page's multivector to blob and drop it from
                        # the page. Holding it here kept every page's ~5 MB of
                        # boxed floats alive until the persist phase, which is
                        # what pushed large documents past 1 GB. The persist
                        # phase rehydrates one page at a time.
                        for p, mv in zip(batch, mv_results):
                            mv_path = self._blob.page_multivector_path(
                                doc.workspace_id, doc.id, p.page_no,
                            )
                            await self._blob.store_file(
                                mv_path, _encode_multivector(mv["vectors"]),
                            )
                            p.multivector_storage_path = mv_path
                            p.multivector = None
                        del mv_results
                    except Exception as mv_exc:  # noqa: BLE001 - one signal; non-fatal
                        self.logger.warning(
                            "Multi-vector embedding failed for document %s (non-fatal): %s",
                            doc.id, mv_exc,
                        )

                    # Visual-tokenizer image-embeds — additive; persisted to blob
                    # now and to the page row by the later persist phase
                    # (persist=False). Chat-ingest depends on these existing, so it
                    # forces the precompute on even when the visual-tokenizer flag
                    # is off. Batched so it re-reads only this batch's images.
                    if precompute_embeds:
                        try:
                            await precompute_and_store_image_embeds(
                                embed_client=self._embed,
                                blob_storage=self._blob,
                                storage=self._storage,
                                pages=batch,
                                workspace_id=doc.workspace_id,
                                document_id=doc.id,
                                filename=doc.filename,
                                logger=self.logger,
                                source=getattr(doc, "source_vfs_ref", None),
                                persist=False,
                            )
                        except Exception as vt_exc:  # noqa: BLE001 - additive
                            self.logger.warning(
                                "Image-embed precompute failed for document %s "
                                "(non-fatal): %s", doc.id, vt_exc,
                            )

                    # Release this batch's image bytes before the next batch.
                    del images_b64

            # Document-chat ingestion: generate text for image pages lacking a
            # transcript (OCR transcription disabled). Additive; never fatal.
            if self._chat_ingest_enabled():
                pages_needing_text = [p for p in pages_with_images if not p.transcript]
                if pages_needing_text:
                    try:
                        inference_client = await get_inference_client(self._v, self.logger)
                        model = default_inference_model(self._v)
                        model_slug = slugify_model(model)
                        instruction = (
                            doc.extraction_options.system_prompt
                            or self._v.environ(
                                MEMORYLAYER_DOCUMENT_CHAT_INGEST_PROMPT,
                                default=DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_PROMPT,
                            )
                        )
                        max_tokens = int(self._v.environ(
                            MEMORYLAYER_DOCUMENT_CHAT_INGEST_MAX_TOKENS,
                            default=DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_MAX_TOKENS,
                        ))
                        for p in pages_needing_text:
                            text = await generate_page_text_from_image_embeds(
                                inference_client=inference_client,
                                blob_storage=self._blob,
                                page=p,
                                model=model,
                                model_slug=model_slug,
                                filename=doc.filename,
                                instruction=instruction,
                                max_tokens=max_tokens,
                                v=self._v,
                                logger=self.logger,
                            )
                            if text:
                                p.transcript = text
                                p.transcript_model = model
                    except Exception as ci_exc:  # noqa: BLE001 - additive
                        self.logger.warning(
                            "Document-chat ingest failed for document %s "
                            "(non-fatal): %s", doc.id, ci_exc,
                        )

            # Single-vector embeddings over all pages that now have a transcript.
            # Text is low-memory, but batch it too for embed-server request-size
            # safety and consistency with the image signals above.
            pages_with_text = [p for p in pages if p.transcript]
            for start in range(0, len(pages_with_text), _RENDER_BATCH_PAGES):
                batch = pages_with_text[start:start + _RENDER_BATCH_PAGES]
                texts = [p.transcript for p in batch]
                embeddings = await self._embed.embed_texts(texts)
                for p, emb in zip(batch, embeddings):
                    p.embedding = emb
        finally:
            await self._embed.close()

        self.logger.info(
            "Generated embeddings for %d pages",
            len([p for p in pages if p.transcript]),
        )
        return pages

    def _transcribe_enabled(self) -> bool:
        """Whether OCR/VLM transcription runs in the pipeline (default True)."""
        return self._v.environ(
            MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
            default=DEFAULT_MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
            type_fn=ext_parse_bool,
        )

    def _chat_ingest_enabled(self) -> bool:
        """Whether document-chat ingestion generates page text (default False)."""
        return self._v.environ(
            MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED,
            default=DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED,
            type_fn=ext_parse_bool,
        )

    @asynccontextmanager
    async def _rehydrated_multivector(self, page: DocumentPage):
        """Restore a page's spilled multivector for the body, then release it.

        The embed phase writes each page's ColPali multivector to blob storage
        and clears the in-memory field (see ``_encode_multivector``), so any
        consumer that needs the actual vectors — the page row write and the
        page memory write — has to pull it back for the duration of that write
        and drop it again afterwards. Keeping the window this narrow is what
        bounds peak residency to a single page instead of the whole document.

        A rehydrate failure is non-fatal and leaves ``multivector`` as ``None``:
        the vectors are an additive signal, and losing them is preferable to
        failing an otherwise-complete ingest. Pages that were never spilled pass
        through untouched, so this is safe to wrap around every page.
        """
        path = page.multivector_storage_path
        rehydrated = False
        if page.multivector is None and path:
            try:
                page.multivector = _decode_multivector(
                    await self._blob.retrieve_file(path)
                )
                rehydrated = True
            except Exception as exc:  # noqa: BLE001 - additive signal; non-fatal
                self.logger.warning(
                    "Multivector rehydrate failed for page %s of document %s "
                    "(non-fatal, proceeding without it): %s",
                    page.page_no, page.document_id, exc,
                )
        try:
            yield page
        finally:
            if rehydrated:
                page.multivector = None

    async def _persist_pages(
        self, doc: Document, pages: list[DocumentPage],
    ) -> list[DocumentPage]:
        """Phase 4.5: persist pages to the document pages registry.

        Stores each page with its embeddings, transcript, and image reference
        in the database for direct page-level queries and MaxSim search.

        Args:
            doc: Document domain model.
            pages: List of document pages with transcripts and embeddings.

        Returns:
            The same list of pages, now with persisted IDs.
        """
        persisted_pages = []
        for page in pages:
            # ``create_page`` reads ``page.multivector`` to write the page row's
            # MaxSim vectors, so the spilled vector has to be back in place for
            # the duration of the write — one page at a time.
            async with self._rehydrated_multivector(page):
                persisted = await self._storage.create_page(
                    workspace_id=doc.workspace_id,
                    document_id=doc.id,
                    page=page,
                )
            # Update the in-memory page with the persisted ID
            page.id = persisted.id
            persisted_pages.append(page)

        self.logger.info(
            "Persisted %d pages for document %s", len(persisted_pages), doc.id,
        )
        return persisted_pages

    @staticmethod
    def _build_page_memory_input(doc: Document, page: DocumentPage) -> RememberInput:
        """Build the per-page ``RememberInput`` for a page-derived memory.

        Single source of truth shared by the inline ``_store_as_memories`` path
        and the distributed ``document_finalize`` task so both produce identical
        memory inputs (content, type, tags, metadata, source linkage, and the
        ``context_id`` NULL-sentinel handling).

        In the OCR-free (visual-only) mode — transcription and chat-ingest both
        disabled — a page has no transcript, so the memory carries a minimal
        provenance placeholder as its textual content while the real retrieval
        signal is the page's ColPali ``multivector`` (see
        ``create_memories_for_pages``). Transcribed pages are unaffected: their
        transcript is used verbatim.

        Args:
            doc: Document domain model.
            page: Page with a transcript or (visual-only) a multivector.

        Returns:
            The ``RememberInput`` describing the page-derived memory.
        """
        # A transcript-less page is a visual-only stub: its ``content`` is only a
        # provenance placeholder and the real signal is the page image +
        # multivector. Flag it so the fact-decomposition handler reads facts from
        # the page IMAGE via a vision model instead of from the placeholder text.
        page_metadata = {
            "source_document_id": doc.id,
            "source_filename": doc.filename,
            "page_number": page.page_no,
            "document_type": doc.document_type.value,
        }
        connector_metadata = {
            **dict(doc.metadata or {}),
            "filename": doc.filename,
            "document_id": doc.id,
        }
        normalized = normalize_connector_metadata(
            connector_metadata,
            connector_type=str(connector_metadata.get("connector_type") or "document"),
            record_id=doc.source_vfs_ref or doc.id,
        ).metadata
        page_metadata.update(
            {
                key: normalized[key]
                for key in ("connector_type", "knowledge_work", KNOWLEDGE_WORK_NORMALIZATION_KEY)
                if key in normalized
            }
        )
        if not page.transcript:
            page_metadata["visual_only"] = True

        return RememberInput(
            content=page.transcript or (
                "[Visual page %d of %s]" % (page.page_no + 1, doc.filename)
            ),
            type=MemoryType.SEMANTIC,
            importance=doc.extraction_options.importance,
            tags=["document_ingestion", "doc:%s" % doc.id],
            metadata=page_metadata,
            # context_id is RESERVED / unused as a retrieval filter today, so
            # persist NULL for the "_default" sentinel: memories.context_id is
            # nullable (FK SET NULL), so NULL needs no contexts row. Only an
            # explicitly-set custom context is carried through (and the FK
            # would then require it to exist). See MemoryModel.context_id.
            context_id=(
                None if doc.target_context_id in (None, "_default")
                else doc.target_context_id
            ),
            source_document_id=doc.id,
            source_page_id=page.id,
        )

    async def create_memories_for_pages(
        self, doc: Document, pages: list[DocumentPage], job_id: str | None = None,
    ) -> list[str]:
        """Create memories from transcribed pages and enqueue post-store enrichment.

        Shared by the inline ingestion path (``_store_as_memories``) and the
        distributed ``document_finalize`` task. Each page with a transcript or a
        (visual-only) multivector is stored via the storage backend's
        ``create_memory()`` (preserving the pre-computed single-vector embedding
        and multivector so they are NOT re-generated), then routed through
        ``MemoryService.enqueue_post_store``
        so it receives the same decomposition + enrichment lifecycle as any
        memory created via ``remember()``.

        Args:
            doc: Document domain model.
            pages: Pages with transcripts and (restored) embeddings.
            job_id: Optional job correlation id threaded into the post-store
                decomposition payload.

        Returns:
            List of created memory IDs.
        """
        memory_ids: list[str] = []
        # Memories handed to background fact decomposition. Recorded on the
        # document so its knowledge phase can be resolved later from stored
        # state; see ``_record_enrichment_phase``.
        decompose_memory_ids: list[str] = []
        # Per-document opt-out, declared at upload and pinned in ingest_flags.
        # Suppresses only the fan-out: the composite page memory below is still
        # created, so retrieval and completeness are unaffected.
        from .gap_analysis import resolve_effective_flags

        decompose = resolve_effective_flags(self._v, doc).decompose

        # Idempotency: load the memories already created from this document once,
        # and skip pages already represented (upsert-by-source_page_id). This
        # prevents duplicate memories when the store/finalize phase re-runs on
        # retry or gap-fill. Existing memory ids are seeded into the returned set
        # so it reflects the FULL document (finalize writes doc.memory_ids from
        # it), not just the pages stored on this pass.
        existing_page_ids: set[str] = set()
        try:
            existing_memories = await self._storage.get_document_memories(
                doc.workspace_id, doc.id,
            )
            for mem in existing_memories:
                memory_ids.append(mem.id)
                if mem.source_page_id:
                    existing_page_ids.add(mem.source_page_id)
        except Exception as exc:  # noqa: BLE001 - defensive; treat as none-present
            self.logger.warning(
                "get_document_memories failed for doc %s; proceeding without "
                "skip-guard: %s", doc.id, exc,
            )

        for page in pages:
            # Store a memory for any page that carries a usable signal: either a
            # transcript (OCR/VLM text path) OR a ColPali ``multivector`` (the
            # OCR-free visual-only mode, where transcription and chat-ingest are
            # both disabled and the page has no transcript). Without the visual
            # fallback, an OCR-free document produces zero memories and finalize
            # marks it FAILED even though every page rendered and embedded
            # cleanly. A page with neither signal has nothing to store.
            # ``multivector_storage_path`` counts as the visual signal too: the
            # embed phase spills the vectors to blob and clears ``multivector``,
            # so testing the in-memory field alone would drop every visual-only
            # page and fail an OCR-free document.
            if (
                not page.transcript
                and page.multivector is None
                and page.multivector_storage_path is None
            ):
                continue

            # Skip pages that already have a composite memory (retry/gap-fill).
            if page.id is not None and page.id in existing_page_ids:
                self.logger.debug(
                    "Skipping page %s of doc %s: memory already exists",
                    page.id, doc.id,
                )
                continue

            # Rehydrate this page's spilled multivector just long enough to
            # write it, so at most one page's worth is resident at a time.
            async with self._rehydrated_multivector(page):
                input_data = self._build_page_memory_input(doc, page)

                memory = await self._storage.create_memory(
                    workspace_id=doc.workspace_id,
                    input=input_data,
                    embedding=page.embedding,
                    multivector=page.multivector,
                )
            memory_ids.append(memory.id)

            # Route the page memory through the shared post-store lifecycle so it
            # gets decomposition + associations + KG facts + contradiction +
            # tiering, identical to a memory created via remember(). Best-effort:
            # enrichment is additive, so one page's failure must not abort an
            # otherwise-successful multi-page ingest (the memory is already
            # created and counted above).
            # A visual-only page (no transcript) carries just a short placeholder
            # as ``content``, which fails the text-based decompose heuristic; force
            # decomposition so its facts are read from the page IMAGE by the
            # vision model. Transcript-backed pages use the normal heuristic.
            force_decompose = not page.transcript
            try:
                scheduled = await self._memory_service.enqueue_post_store(
                    doc.workspace_id,
                    memory,
                    embedding=page.embedding,
                    job_id=job_id,
                    force_decompose=force_decompose and decompose,
                    decompose=decompose,
                )
                if scheduled:
                    decompose_memory_ids.append(memory.id)
            except Exception as exc:  # noqa: BLE001 - additive enrichment; non-fatal
                self.logger.warning(
                    "Post-store enrichment enqueue failed for memory %s "
                    "(page %s, doc %s); memory created without it: %s",
                    memory.id, page.id, doc.id, exc,
                )

        self.logger.info(
            "Created %d memories for document %s", len(memory_ids), doc.id,
        )
        await self._record_enrichment_phase(doc, decompose_memory_ids)
        return memory_ids

    async def _record_enrichment_phase(
        self, doc: Document, decompose_memory_ids: list[str],
    ) -> None:
        """Record which memories the document is waiting on for its knowledge phase.

        Kept OUT of ``status``: that tracks retrieval readiness and goes
        COMPLETED as soon as pages, embeddings and memories are durable.
        Decomposition fans out to thousands of tasks and can trail it by a long
        way, so a caller that only searches must not be made to wait on it.

        The scheduled ids are UNIONED with whatever is already recorded rather
        than replaced. This path is idempotent and re-runs on gap-fill or retry,
        where pages already represented are skipped and schedule nothing — a
        straight overwrite would erase a still-outstanding set and report the
        document as enriched when it isn't.

        Best-effort: enrichment bookkeeping must never fail an ingest whose
        memories are already stored. ``doc_verify`` re-derives the phase anyway.
        """
        try:
            previously = list(getattr(doc, "enrichment_memory_ids", None) or [])
            pending = list(dict.fromkeys(previously + decompose_memory_ids))
            status = (
                DocumentEnrichmentStatus.PENDING
                if pending
                else DocumentEnrichmentStatus.NOT_APPLICABLE
            )
            await self._storage.update_document(
                doc.id,
                enrichment_status=status.value,
                enrichment_memory_ids=pending,
            )
            doc.enrichment_status = status
            doc.enrichment_memory_ids = pending
            self.logger.info(
                "Document %s knowledge phase: %s (%d memory/ies awaiting decomposition)",
                doc.id, status.value, len(pending),
            )
        except Exception as exc:  # noqa: BLE001 - bookkeeping is not the ingest
            self.logger.warning(
                "Failed to record enrichment phase for doc %s; doc_verify will "
                "re-derive it: %s", doc.id, exc,
            )

    async def _store_as_memories(
        self, doc: Document, pages: list[DocumentPage], job_id: str | None = None,
    ) -> list[str]:
        """Phase 5: create memories from pages with pre-computed embeddings.

        Thin wrapper over :meth:`create_memories_for_pages` (the shared create +
        post-store path). Calls the storage backend's ``create_memory()`` with
        pre-computed embedding and multivector data, bypassing the OSS
        MemoryService.remember() which would re-generate embeddings, then enqueues
        the post-store enrichment lifecycle for each page memory.

        Args:
            doc: Document domain model.
            pages: List of pages with transcripts and embeddings.
            job_id: Optional job correlation id for the post-store payload.

        Returns:
            List of created memory IDs.
        """
        return await self.create_memories_for_pages(doc, pages, job_id=job_id)

    async def _finalize(
        self, doc: Document, memory_ids: list[str], job_id: str,
    ) -> None:
        """Phase 6: update document and job status to reflect completion.

        Pins the effective ingest flag set onto ``doc.metadata['ingest_flags']``
        so later gap analysis is deterministic across flag flips, then derives
        the document status from a gap analysis instead of the old binary
        memories?-COMPLETED-else-FAILED:

        - COMPLETED when ``gaps.is_complete`` (all required artifacts present,
          ≥1 memory per transcribed page);
        - PARTIAL when memories exist but some pages are still missing artifacts
          (usable but incomplete — gives the dead PARTIAL enum its first use);
        - FAILED when no pages / zero memories.

        Args:
            doc: Document domain model.
            memory_ids: List of memory IDs representing the document.
            job_id: Job identifier.
        """
        # Delayed import to keep the module import graph flat (gap_analysis pulls
        # config + document models only; no service deps).
        from .gap_analysis import analyze_document_gaps, resolve_effective_flags

        now = datetime.now(timezone.utc)

        # Pin the effective flags onto the document metadata so completeness is
        # judged against the flags in force at ingest time, not a later flip.
        eff = resolve_effective_flags(self._v, doc)
        pinned_metadata = dict(doc.metadata or {})
        pinned_metadata["ingest_flags"] = eff.as_dict()
        await self._storage.update_document(doc.id, metadata=pinned_metadata)
        doc.metadata = pinned_metadata

        # Re-read so gap analysis sees the just-persisted pages/memories and the
        # pinned flags.
        refreshed = await self._storage.get_document(doc.id, doc.workspace_id)
        gap_doc = refreshed or doc

        try:
            gaps = await analyze_document_gaps(self._v, self._storage, gap_doc, flags=eff)
        except Exception as exc:  # noqa: BLE001 - fall back to the binary rule
            self.logger.warning(
                "Gap analysis failed for doc %s during finalize; falling back "
                "to binary status: %s", doc.id, exc,
            )
            gaps = None

        if gaps is not None and gaps.is_complete:
            status = DocumentStatus.COMPLETED
        elif memory_ids:
            status = DocumentStatus.PARTIAL
        else:
            status = DocumentStatus.FAILED

        await self._storage.update_document(
            doc.id,
            status=status.value,
            chunk_count=len(memory_ids),
            memory_ids=memory_ids,
            processing_completed_at=now,
        )

        await self._storage.update_job(
            job_id,
            status=JobStatus.COMPLETED.value,
            progress_percent=100,
            documents_processed=1,
            total_memories_created=len(memory_ids),
            completed_at=now,
        )

        # Reconcile-on-complete: once the document is COMPLETED, close any OTHER
        # in-flight jobs still referencing it (a late-superseded replay, or an
        # idempotent no-op completion driven by a different job). Keep the job
        # that drove this completion (already terminal above). Idempotent: a
        # no-op when no other in-flight jobs exist.
        if status == DocumentStatus.COMPLETED:
            await self._supersede_active_jobs([doc.id], keep_job_id=job_id)

        self.logger.info(
            "Finalized document %s: status=%s, memories=%d",
            doc.id, status.value, len(memory_ids),
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_document_type(filename: str) -> DocumentType:
        """Detect document type from the filename extension.

        Args:
            filename: Original filename.

        Returns:
            Matching DocumentType enum value.

        Raises:
            ValueError: If the extension is unsupported.
        """
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        mapping = {
            "pdf": DocumentType.PDF,
            "md": DocumentType.MARKDOWN,
            "markdown": DocumentType.MARKDOWN,
            "txt": DocumentType.TEXT,
            "text": DocumentType.TEXT,
            "html": DocumentType.HTML,
            "htm": DocumentType.HTML,
            "docx": DocumentType.DOCX,
            "pptx": DocumentType.PPTX,
        }
        if ext not in mapping:
            raise ValueError("Unsupported file type: .%s" % ext)
        return mapping[ext]


class DocumentIngestionServicePlugin(DocumentIngestionPluginBase):
    """Plugin for the default document ingestion service."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: Logger) -> DocumentIngestionService:
        """Build and return the ingestion service with all dependencies.

        Args:
            v: Variables instance for configuration access.
            logger: Logger instance.

        Returns:
            Fully wired DocumentIngestionService.
        """
        storage_backend = get_extension(EXT_STORAGE_BACKEND, v)
        embed_client = get_extension(EXT_EMBED_SERVER_CLIENT, v)
        blob_storage = get_extension(EXT_BLOB_STORAGE_SERVICE, v)
        task_service = get_extension(EXT_TASK_SERVICE, v)
        memory_service = get_extension(EXT_MEMORY_SERVICE, v)

        max_file_size = int(v.environ(
            MEMORYLAYER_DOCUMENT_MAX_FILE_SIZE,
            default=str(DEFAULT_MEMORYLAYER_DOCUMENT_MAX_FILE_SIZE),
        ))

        logger.info(
            "Initializing document ingestion service (max_file_size=%d)",
            max_file_size,
        )
        return DocumentIngestionService(
            v=v,
            storage_backend=storage_backend,
            blob_storage=blob_storage,
            embed_client=embed_client,
            task_service=task_service,
            memory_service=memory_service,
            max_file_size=max_file_size,
            logger=logger,
        )

    def get_dependencies(self, v: Variables):
        """Declare extension point dependencies."""
        return (
            EXT_EMBED_SERVER_CLIENT,
            EXT_BLOB_STORAGE_SERVICE,
            EXT_STORAGE_BACKEND,
            EXT_TASK_SERVICE,
            EXT_MEMORY_SERVICE,
        )
