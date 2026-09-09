"""Document ingestion API endpoints.

Endpoints:
- POST   /v1/documents               -- Upload document (multipart/form-data)
- GET    /v1/documents               -- List documents in workspace
- GET    /v1/documents/{id}          -- Get document metadata + status
- GET    /v1/documents/{id}/memories -- Get memories extracted from document
- DELETE /v1/documents/{id}          -- Delete document (+ optionally memories)
- POST   /v1/documents/{id}/reprocess -- Re-extract with different options

- GET    /v1/documents/jobs          -- List jobs in workspace
- GET    /v1/documents/jobs/{id}     -- Get job status + progress
- POST   /v1/documents/jobs/{id}/cancel -- Cancel running job
"""
from datetime import datetime
from logging import Logger
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field
from scitrera_app_framework import Plugin, Variables

from memorylayer_server.api import EXT_MULTI_API_ROUTERS
from memorylayer_server.api.v1.deps import (
    get_auth_service, get_authz_service, get_audit_service, get_task_service,
)
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep
from memorylayer_server.services.authentication import AuthenticationService, AuthenticationError
from memorylayer_server.services.authorization import AuthorizationService
from memorylayer_server.services.audit import AuditService, AuditEvent
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
from memorylayer_server.services.tasks import TaskService
from scitrera_app_framework import get_extension

from ...models.document import (
    DocumentExtractionOptions,
    DocumentStatus,
    DocumentType,
    JobStatus,
)
from ...services.document import get_document_ingestion_service, EXT_EMBED_SERVER_CLIENT, EXT_BLOB_STORAGE_SERVICE
from ...services.document.ingestion_service import DocumentIngestionService

router = APIRouter(prefix="/v1/documents", tags=["documents"])


# ------------------------------------------------------------------ #
# Response models
# ------------------------------------------------------------------ #

class DocumentResponse(BaseModel):
    """Document API response."""

    id: str
    workspace_id: str
    filename: str
    document_type: str
    content_hash: str
    size_bytes: int
    mime_type: Optional[str] = None
    source_vfs_ref: Optional[str] = None
    status: str
    #: Knowledge phase, independent of ``status``. ``status == "completed"``
    #: means the document is retrievable (pages, embeddings and memories are
    #: durable); this says whether fact extraction has finished, which can trail
    #: it by a long way. Poll this only if you need the extracted facts —
    #: search and page reads do not.
    enrichment_status: str = "not_applicable"
    target_context_id: str = "_default"
    extraction_options: DocumentExtractionOptions = Field(
        default_factory=DocumentExtractionOptions,
    )
    page_count: int = 0
    chunk_count: int = 0
    memory_ids: list[str] = Field(default_factory=list)
    storage_path: Optional[str] = None
    retain_original: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    documents: list[DocumentResponse]
    total_count: int


class JobResponse(BaseModel):
    """Ingestion job API response."""

    id: str
    workspace_id: str
    document_ids: list[str] = Field(default_factory=list)
    status: str
    progress_percent: int = 0
    documents_processed: int = 0
    total_memories_created: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class JobListResponse(BaseModel):
    """List of ingestion jobs."""

    jobs: list[JobResponse]


class UploadResponse(BaseModel):
    """Response for a successful document upload."""

    document: DocumentResponse
    job: JobResponse


class MemoryBriefResponse(BaseModel):
    """Abbreviated memory representation returned from the document memories endpoint."""

    id: str
    content: str
    type: str
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    created_at: datetime


class DocumentMemoriesResponse(BaseModel):
    """List of memories extracted from a document."""

    document_id: str
    memories: list[MemoryBriefResponse]
    total_count: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str


class PageResponse(BaseModel):
    """Document page API response."""

    id: str
    document_id: str
    workspace_id: str
    page_no: int
    image_storage_path: Optional[str] = None
    transcript: Optional[str] = None
    transcript_model: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    relevance_score: Optional[float] = None


def _page_to_response(p) -> "PageResponse":
    """Map a stored ``DocumentPage`` to the API shape.

    Shared by the page LIST and single-page GET so the two cannot drift in
    which fields they expose.
    """
    return PageResponse(
        id=p.id,
        document_id=p.document_id,
        workspace_id=p.workspace_id,
        page_no=p.page_no,
        image_storage_path=p.image_storage_path,
        transcript=p.transcript,
        transcript_model=p.transcript_model,
        metadata=p.metadata,
        created_at=p.created_at,
    )


class PageListResponse(BaseModel):
    """List of document pages."""

    document_id: str
    pages: list[PageResponse]
    total_count: int


class PageSearchRequest(BaseModel):
    """Request body for document page search."""

    query: str = Field(..., description="Natural language search query")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    doc_ids: Optional[list[str]] = Field(None, description="Optional document ID filter")


class PageSearchResponse(BaseModel):
    """Response for document page search."""

    pages: list[PageResponse]
    total_count: int
    query: str


class BatchPagesRequest(BaseModel):
    """Request body for fetching pages across several documents at once."""

    doc_ids: list[str] = Field(..., description="Document IDs to fetch pages for")
    workspace_id: Optional[str] = Field(
        None, description="Workspace scope; falls back to the caller's context",
    )
    include_transcript: bool = Field(
        True,
        description=(
            "Include page transcript text. Set false for an index-only view "
            "(page ids/numbers) at a fraction of the payload."
        ),
    )
    limit: Optional[int] = Field(
        None, ge=1, description="Max pages to return, for chunking large sets",
    )
    offset: int = Field(0, ge=0, description="Offset, for use with limit")


class BatchPagesResponse(BaseModel):
    """Pages for several documents, ordered by (document_id, page_no)."""

    pages: list[PageResponse]
    total_count: int


class PagesStatusRequest(BaseModel):
    """Request body for batch document ingestion status."""

    doc_ids: list[str] = Field(..., description="Document IDs to report on")
    workspace_id: Optional[str] = Field(
        None, description="Workspace scope; falls back to the caller's context",
    )


class DocumentPageStatus(BaseModel):
    """Per-document ingestion progress and completeness.

    ``is_complete`` is the gate-aware verdict — "should anything still run for
    this document?" — while ``page_count``/``pages_with_transcript`` are plain
    tallies. They disagree when transcription is disabled: no page carries a
    transcript, yet the document is legitimately complete and a caller waiting
    on it should stop waiting.
    """

    document_id: str
    is_complete: bool
    first_missing_phase: Optional[str] = None
    expected_page_count: int
    page_count: int
    pages_with_transcript: int
    flags: dict[str, Any] = Field(default_factory=dict)


class PagesStatusResponse(BaseModel):
    """Batch ingestion status with a rollup across the requested documents."""

    documents: list[DocumentPageStatus]
    total_count: int
    page_count: int
    pages_with_transcript: int
    is_complete: bool = Field(
        ..., description="True when every requested document is complete",
    )


# ------------------------------------------------------------------ #
# Dependencies
# ------------------------------------------------------------------ #

def get_document_service_dep(
    v: Variables = Depends(get_variables_dep),
) -> DocumentIngestionService:
    """FastAPI dependency wrapper for the document ingestion service."""
    return get_document_ingestion_service(v)


# ------------------------------------------------------------------ #
# Document endpoints
# ------------------------------------------------------------------ #

@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def upload_document(
    http_request: Request,
    file: UploadFile = File(...),
    target_context_id: str = Form("_default"),
    chunking_strategy: str = Form("page"),
    chunk_size: int = Form(4096),
    chunk_overlap: int = Form(200),
    importance: float = Form(0.5),
    retain_original: bool = Form(True),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DocumentIngestionService = Depends(get_document_service_dep),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> UploadResponse:
    """Upload a document for ingestion.

    Accepts multipart/form-data with the document file and optional
    extraction parameters.  Returns the created document record and
    the background ingestion job.

    Args:
        file: Document file upload.
        target_context_id: Memory context for extracted memories.
        chunking_strategy: Chunking strategy (page, semantic, fixed).
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between chunks in characters.
        importance: Default importance for extracted memories (0.0-1.0).
        retain_original: Whether to keep the original file in blob storage.
        workspace_id: Workspace ID from auth context.
        service: Document ingestion service instance.
        logger: Logger instance.

    Returns:
        UploadResponse with document and job details.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "documents", "write", workspace_id=ctx.workspace_id)

        file_data = await file.read()
        filename = file.filename or "unnamed"

        extraction_options = DocumentExtractionOptions(
            target_context_id=target_context_id,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            importance=importance,
            retain_original=retain_original,
        )

        logger.info(
            "Upload request: %s (%d bytes) for workspace %s",
            filename, len(file_data), ctx.workspace_id,
        )

        doc, job = await service.upload_document(
            workspace_id=ctx.workspace_id,
            file_data=file_data,
            filename=filename,
            extraction_options=extraction_options,
        )

        try:
            await audit_service.record(AuditEvent(
                event_type="document",
                action="upload",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="document",
                resource_id=doc.id,
            ))
        except Exception:
            logger.debug("Audit record failed for document upload")
        return UploadResponse(
            document=DocumentResponse(**doc.model_dump()),
            job=JobResponse(**job.model_dump()),
        )

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except ValueError as exc:
        logger.warning("Upload rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Upload failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document",
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_documents(
    http_request: Request,
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    document_type: Optional[str] = Query(None, description="Filter by document type (pdf, markdown, text, html, docx, pptx)"),
    created_after: Optional[str] = Query(None, description="Filter by created_at >= ISO datetime"),
    created_before: Optional[str] = Query(None, description="Filter by created_at <= ISO datetime"),
    workspace_id: Optional[str] = Query(None, description="Workspace to list (defaults to the caller's context workspace)"),
    limit: int = Query(50, ge=1, le=200, description="Maximum documents to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DocumentIngestionService = Depends(get_document_service_dep),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> DocumentListResponse:
    """List documents in the workspace.

    Args:
        status_filter: Optional status filter (pending, processing, completed, failed).
        limit: Maximum number of documents to return.
        offset: Pagination offset.
        workspace_id: Workspace ID from auth context.
        service: Document ingestion service instance.
        logger: Logger instance.

    Returns:
        DocumentListResponse with paginated results.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        ws = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ws)

        logger.debug(
            "Listing documents for workspace %s (status=%s, limit=%d, offset=%d)",
            ws, status_filter, limit, offset,
        )

        docs, total = await service.list_documents(
            workspace_id=ws,
            status=status_filter,
            document_type=document_type,
            created_after=created_after,
            created_before=created_before,
            limit=limit,
            offset=offset,
        )

        try:
            await audit_service.record(AuditEvent(
                event_type="document",
                action="list",
                tenant_id=ctx.tenant_id,
                workspace_id=ws,
                user_id=ctx.user_id,
                resource_type="document",
            ))
        except Exception:
            logger.debug("Audit record failed for document list")
        return DocumentListResponse(
            documents=[DocumentResponse(**d.model_dump()) for d in docs],
            total_count=total,
        )

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list documents: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list documents",
        )


@router.get(
    "/jobs",
    response_model=JobListResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_jobs(
    http_request: Request,
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by job status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum jobs to return"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DocumentIngestionService = Depends(get_document_service_dep),
    logger: Logger = Depends(get_logger),
) -> JobListResponse:
    """List ingestion jobs for the workspace.

    Args:
        status_filter: Optional job status filter.
        limit: Maximum number of jobs to return.
        workspace_id: Workspace ID from auth context.
        service: Document ingestion service instance.
        logger: Logger instance.

    Returns:
        JobListResponse with job records.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ctx.workspace_id)

        logger.debug(
            "Listing jobs for workspace %s (status=%s, limit=%d)",
            ctx.workspace_id, status_filter, limit,
        )

        jobs = await service.list_jobs(
            workspace_id=ctx.workspace_id,
            status=status_filter,
            limit=limit,
        )

        return JobListResponse(
            jobs=[JobResponse(**j.model_dump()) for j in jobs],
        )

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list jobs: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list jobs",
        )


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_job(
    http_request: Request,
    job_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DocumentIngestionService = Depends(get_document_service_dep),
    logger: Logger = Depends(get_logger),
) -> JobResponse:
    """Get ingestion job status and progress.

    Args:
        job_id: Job identifier.
        service: Document ingestion service instance.
        logger: Logger instance.

    Returns:
        JobResponse with current job state.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ctx.workspace_id)

        logger.debug("Fetching job: %s", job_id)

        job = await service.get_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found: %s" % job_id,
            )

        return JobResponse(**job.model_dump())

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get job %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job",
        )


@router.post(
    "/jobs/{job_id}/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def cancel_job(
    http_request: Request,
    job_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DocumentIngestionService = Depends(get_document_service_dep),
    logger: Logger = Depends(get_logger),
) -> None:
    """Cancel a queued or running ingestion job.

    Args:
        job_id: Job identifier.
        service: Document ingestion service instance.
        logger: Logger instance.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "documents", "write", workspace_id=ctx.workspace_id)

        logger.info("Cancel request for job: %s", job_id)
        await service.cancel_job(job_id)

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except ValueError as exc:
        logger.warning("Cancel rejected for job %s: %s", job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to cancel job %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel job",
        )


@router.post(
    "/search",
    response_model=PageSearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def search_document_pages(
    http_request: Request,
    request: PageSearchRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DocumentIngestionService = Depends(get_document_service_dep),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> PageSearchResponse:
    """Search document pages using ColPali MaxSim visual similarity.

    Embeds the query as a multi-vector and searches page embeddings
    using late interaction (MaxSim) scoring.

    Args:
        request: Search request with query and optional filters.
        workspace_id: Workspace ID from auth context.
        service: Document ingestion service instance.
        v: Variables for accessing storage and embed services.
        logger: Logger instance.

    Returns:
        PageSearchResponse with ranked pages.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ctx.workspace_id)

        logger.info(
            "Page search: query=%r, workspace=%s, limit=%d",
            request.query, ctx.workspace_id, request.limit,
        )

        # Get embed client and storage backend
        embed_client = get_extension(EXT_EMBED_SERVER_CLIENT, v)
        storage_backend = get_extension(EXT_STORAGE_BACKEND, v)

        # Embed query as multi-vector
        try:
            await embed_client.connect()
            mv_results = await embed_client.embed_texts_multivector([request.query])
            query_multivector = mv_results[0]["vectors"]
        finally:
            await embed_client.close()

        # Search pages by MaxSim
        results = await storage_backend.search_pages_by_maxsim(
            workspace_id=ctx.workspace_id,
            query_multivector=query_multivector,
            limit=request.limit,
            doc_ids=request.doc_ids,
        )

        pages = [
            PageResponse(
                id=page.id,
                document_id=page.document_id,
                workspace_id=page.workspace_id,
                page_no=page.page_no,
                image_storage_path=page.image_storage_path,
                transcript=page.transcript,
                transcript_model=page.transcript_model,
                metadata=page.metadata,
                created_at=page.created_at,
                relevance_score=score,
            )
            for page, score in results
        ]

        return PageSearchResponse(
            pages=pages,
            total_count=len(pages),
            query=request.query,
        )

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Page search failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search document pages",
        )


# NOTE: the /pages/* batch routes are registered BEFORE the /{document_id}
# block on purpose. FastAPI matches in registration order, so a literal path
# that could also be read as a document id must come first.

@router.post(
    "/pages/batch",
    response_model=BatchPagesResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_pages_for_documents(
    http_request: Request,
    request: BatchPagesRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> BatchPagesResponse:
    """Get pages for several documents in one request.

    The per-document ``GET /{document_id}/pages`` makes any caller working over
    a document set fan out one request per document; this collapses that to a
    single round trip. Pages come back ordered by ``(document_id, page_no)``.

    Set ``include_transcript=false`` when you only need the page index — the
    transcripts dominate the payload, and a caller polling for progress has no
    use for them.

    Args:
        request: Document IDs plus optional workspace scope and paging.

    Returns:
        BatchPagesResponse with the pages, ordered and grouped by document.
    """
    try:
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(
            ctx, "documents", "read", workspace_id=workspace_id,
        )

        if not request.doc_ids:
            return BatchPagesResponse(pages=[], total_count=0)

        storage_backend = get_extension(EXT_STORAGE_BACKEND, v)
        page_models = await storage_backend.get_pages_for_documents(
            request.doc_ids, workspace_id,
            limit=request.limit, offset=request.offset,
        )

        pages = [
            PageResponse(
                id=p.id,
                document_id=p.document_id,
                workspace_id=p.workspace_id,
                page_no=p.page_no,
                image_storage_path=p.image_storage_path,
                transcript=p.transcript if request.include_transcript else None,
                transcript_model=p.transcript_model,
                metadata=p.metadata,
                created_at=p.created_at,
            )
            for p in page_models
        ]

        return BatchPagesResponse(pages=pages, total_count=len(pages))

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get pages for documents %s: %s",
            request.doc_ids, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document pages",
        )


@router.post(
    "/pages/status",
    response_model=PagesStatusResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_pages_status(
    http_request: Request,
    request: PagesStatusRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> PagesStatusResponse:
    """Report ingestion progress and completeness for several documents.

    Exists so a caller waiting on ingestion can poll something cheap. The
    alternative — fetching every page and counting transcripts client-side —
    moves megabytes per poll to compute a single ratio.

    Backed by the same ``analyze_document_gaps`` the ingestion pipeline itself
    uses, so the verdict honours the effective per-document ingest flags. That
    matters: with the transcribe phase disabled, a document with zero
    transcripts is nonetheless COMPLETE, and a caller keyed on
    ``is_complete`` stops waiting instead of blocking until its timeout.

    Args:
        request: Document IDs plus optional workspace scope.

    Returns:
        PagesStatusResponse with per-document status and a rollup.
    """
    try:
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(
            ctx, "documents", "read", workspace_id=workspace_id,
        )

        if not request.doc_ids:
            return PagesStatusResponse(
                documents=[], total_count=0, page_count=0,
                pages_with_transcript=0, is_complete=True,
            )

        storage_backend = get_extension(EXT_STORAGE_BACKEND, v)

        # Delayed import: gap_analysis pulls the config surface in, and this
        # module is imported at router registration (mirrors ingestion_service).
        from ...services.document.gap_analysis import (
            PHASE_RENDER,
            analyze_documents_gaps,
        )

        get_documents = getattr(storage_backend, "get_documents", None)
        if get_documents is not None:
            docs = await get_documents(request.doc_ids, workspace_id)
        else:
            docs = [
                doc for doc in [
                    await storage_backend.get_document(doc_id, workspace_id)
                    for doc_id in request.doc_ids
                ] if doc is not None
            ]

        gaps = await analyze_documents_gaps(v, storage_backend, docs)

        documents = [
            DocumentPageStatus(
                document_id=g.document_id,
                is_complete=g.is_complete,
                first_missing_phase=g.first_missing_phase,
                expected_page_count=g.expected_page_count,
                page_count=g.page_count,
                pages_with_transcript=g.pages_with_transcript,
                flags=g.flags.as_dict() if g.flags else {},
            )
            for g in gaps
        ]

        # A doc_id that resolved to nothing is reported as incomplete rather
        # than silently dropped — otherwise a typo'd or deleted id contributes
        # nothing to the rollup, `all()` reads as complete, and a caller waiting
        # on it proceeds against a document that does not exist.
        found = {d.document_id for d in documents}
        for doc_id in request.doc_ids:
            if doc_id not in found:
                documents.append(DocumentPageStatus(
                    document_id=doc_id,
                    is_complete=False,
                    first_missing_phase=PHASE_RENDER,
                    expected_page_count=0,
                    page_count=0,
                    pages_with_transcript=0,
                ))

        return PagesStatusResponse(
            documents=documents,
            total_count=len(documents),
            page_count=sum(d.page_count for d in documents),
            pages_with_transcript=sum(d.pages_with_transcript for d in documents),
            is_complete=all(d.is_complete for d in documents) if documents else True,
        )

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get page status for documents %s: %s",
            request.doc_ids, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document page status",
        )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_document(
    http_request: Request,
    document_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DocumentIngestionService = Depends(get_document_service_dep),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> DocumentResponse:
    """Get document metadata and processing status.

    Args:
        document_id: Document identifier.
        workspace_id: Workspace ID from auth context.
        service: Document ingestion service instance.
        logger: Logger instance.

    Returns:
        DocumentResponse with full document details.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ctx.workspace_id)

        logger.debug("Fetching document: %s", document_id)

        doc = await service.get_document(document_id, ctx.workspace_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found: %s" % document_id,
            )

        try:
            await audit_service.record(AuditEvent(
                event_type="document",
                action="read",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="document",
                resource_id=document_id,
            ))
        except Exception:
            logger.debug("Audit record failed for document read")
        return DocumentResponse(**doc.model_dump())

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get document %s: %s", document_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document",
        )


@router.get(
    "/{document_id}/memories",
    response_model=DocumentMemoriesResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_document_memories(
    http_request: Request,
    document_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DocumentIngestionService = Depends(get_document_service_dep),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> DocumentMemoriesResponse:
    """Get memories extracted from a document.

    Retrieves all memories that were created during ingestion of
    the specified document.

    Args:
        document_id: Document identifier.
        workspace_id: Workspace ID from auth context.
        service: Document ingestion service instance.
        v: Variables for accessing storage backend.
        logger: Logger instance.

    Returns:
        DocumentMemoriesResponse with abbreviated memory records.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ctx.workspace_id)

        logger.debug("Fetching memories for document: %s", document_id)

        doc = await service.get_document(document_id, ctx.workspace_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found: %s" % document_id,
            )

        storage_backend = get_extension(EXT_STORAGE_BACKEND, v)
        memories: list[MemoryBriefResponse] = []

        for mem_id in doc.memory_ids:
            memory = await storage_backend.get_memory(
                doc.workspace_id, mem_id, track_access=False,
            )
            if memory:
                memories.append(MemoryBriefResponse(
                    id=memory.id,
                    content=memory.content,
                    type=memory.type.value if hasattr(memory.type, "value") else str(memory.type),
                    importance=memory.importance,
                    tags=memory.tags,
                    created_at=memory.created_at,
                ))

        return DocumentMemoriesResponse(
            document_id=document_id,
            memories=memories,
            total_count=len(memories),
        )

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get memories for document %s: %s",
            document_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document memories",
        )


@router.get(
    "/{document_id}/pages",
    response_model=PageListResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_document_pages(
    http_request: Request,
    document_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> PageListResponse:
    """Get all pages for a document.

    Args:
        document_id: Document identifier.
        workspace_id: Workspace ID from auth context.
        v: Variables for accessing storage backend.
        logger: Logger instance.

    Returns:
        PageListResponse with all pages ordered by page number.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ctx.workspace_id)

        logger.debug("Fetching pages for document: %s", document_id)

        storage_backend = get_extension(EXT_STORAGE_BACKEND, v)
        page_models = await storage_backend.get_pages(document_id, ctx.workspace_id)

        pages = [_page_to_response(p) for p in page_models]

        return PageListResponse(
            document_id=document_id,
            pages=pages,
            total_count=len(pages),
        )

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get pages for document %s: %s",
            document_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document pages",
        )


@router.get(
    "/{document_id}/pages/{page_id}",
    response_model=PageResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Page not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_document_page(
    http_request: Request,
    document_id: str,
    page_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> PageResponse:
    """Get a single page of a document by page ID.

    Ported from the OSS documents router, which ``_supersede_oss_routers``
    drops so the enterprise router can own ``/v1/documents`` — this route
    existed ONLY there, so it 404'd on any enterprise deployment.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ctx.workspace_id)

        storage_backend = get_extension(EXT_STORAGE_BACKEND, v)
        page = await storage_backend.get_page(page_id)

        # Also guards cross-document access: a page id from another document
        # must not be readable by naming any document the caller can see.
        if not page or page.document_id != document_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Page not found: {page_id}",
            )

        return _page_to_response(page)

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get page %s of document %s: %s",
            page_id, document_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document page",
        )


@router.get(
    "/{document_id}/pages/{page_id}/image",
    responses={
        404: {"model": ErrorResponse, "description": "Page not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_page_image(
    http_request: Request,
    document_id: str,
    page_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
):
    """Stream a page image from blob storage.

    Args:
        document_id: Document identifier (for URL structure).
        page_id: Page identifier.
        v: Variables for accessing storage services.
        logger: Logger instance.

    Returns:
        PNG image as streaming response.
    """
    from fastapi.responses import Response

    try:
        ctx = await auth_service.build_context(http_request)

        # Resolve the page FIRST, then authorize against the page's OWNING
        # workspace (not the caller's current ctx.workspace_id). Cross-workspace
        # reads are allowed when the user actually has documents:read on the
        # page's owning workspace; otherwise require_authorization raises 403.
        # Authorizing only ctx.workspace_id would let any holder of that one
        # workspace's grant read pages from arbitrary other workspaces, because
        # get_page() resolves purely by page_id with no rights check.
        storage_backend = get_extension(EXT_STORAGE_BACKEND, v)
        page = await storage_backend.get_page(page_id)

        if not page or page.document_id != document_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Page not found: %s" % page_id,
            )

        try:
            await authz_service.require_authorization(
                ctx, "documents", "read", workspace_id=page.workspace_id,
            )
        except HTTPException:
            # Collapse 403 → 404 so an unauthorized caller cannot distinguish
            # "page doesn't exist" from "page exists but you lack rights".
            # Re-raise authentication failures (4xx from AuthenticationError)
            # as-is; only authz denials are collapsed to 404.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Page not found: %s" % page_id,
            )

        if not page.image_storage_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No image available for page %s" % page_id,
            )

        blob_service = get_extension(EXT_BLOB_STORAGE_SERVICE, v)
        image_bytes = await blob_service.retrieve_file(page.image_storage_path)

        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get image for page %s: %s", page_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve page image",
        )


@router.get(
    "/{document_id}/pages/{page_id}/figures/{figure_no}",
    responses={
        404: {"model": ErrorResponse, "description": "Page or figure not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_page_figure(
    http_request: Request,
    document_id: str,
    page_id: str,
    figure_no: int,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
):
    """Stream one figure cropped from a page render.

    A grounded OCR model marks an illustration with a bounding box and emits no
    text for it, so the crop is the figure's only representation. ``figure_no``
    is 1-based and matches the ``[figure N]`` marker in the page transcript,
    which is how a caller gets from the text to the image.

    Args:
        document_id: Document identifier (for URL structure).
        page_id: Page identifier.
        figure_no: 1-based figure index within the page.

    Returns:
        PNG image as a response.
    """
    from fastapi.responses import Response

    from ...services.document.page_figures import PAGE_FIGURES_METADATA_KEY

    try:
        ctx = await auth_service.build_context(http_request)

        # Same ordering as get_page_image: resolve the page first, then
        # authorize against the page's OWNING workspace rather than the
        # caller's current one, since get_page() resolves by page_id alone.
        storage_backend = get_extension(EXT_STORAGE_BACKEND, v)
        page = await storage_backend.get_page(page_id)

        if not page or page.document_id != document_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Page not found: %s" % page_id,
            )

        try:
            await authz_service.require_authorization(
                ctx, "documents", "read", workspace_id=page.workspace_id,
            )
        except HTTPException:
            # Collapse 403 -> 404 so an unauthorized caller cannot distinguish
            # "page doesn't exist" from "page exists but you lack rights".
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Page not found: %s" % page_id,
            )

        figures = (page.metadata or {}).get(PAGE_FIGURES_METADATA_KEY) or []
        record = next(
            (f for f in figures if isinstance(f, dict) and f.get("figure_no") == figure_no),
            None,
        )
        if not record or not record.get("storage_path"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No figure %d on page %s" % (figure_no, page_id),
            )

        blob_service = get_extension(EXT_BLOB_STORAGE_SERVICE, v)
        image_bytes = await blob_service.retrieve_file(record["storage_path"])

        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get figure %d for page %s: %s", figure_no, page_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve page figure",
        )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def delete_document(
    http_request: Request,
    document_id: str,
    delete_memories: bool = Query(False, description="Also delete extracted memories"),
    workspace_id: str = Query(
        None,
        description=(
            "Workspace the document lives in. When provided it scopes both the "
            "authorization check and the document lookup; falls back to the "
            "auth-context workspace when omitted. Callers that reach a document "
            "outside the request's default workspace (e.g. the platform files "
            "library deleting a doc in a user's _private workspace) must set "
            "this, otherwise the lookup runs against the default workspace and "
            "404s."
        ),
    ),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DocumentIngestionService = Depends(get_document_service_dep),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> None:
    """Delete a document and optionally its extracted memories.

    Args:
        document_id: Document identifier.
        delete_memories: If True, also deletes all memories created from this document.
        workspace_id: Workspace ID from auth context.
        service: Document ingestion service instance.
        logger: Logger instance.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        # Explicit workspace_id (query) scopes both authz and lookup; fall back
        # to the auth-context workspace when the caller doesn't specify one.
        ws = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "documents", "delete", workspace_id=ws)

        logger.info(
            "Delete request for document %s (workspace=%s delete_memories=%s)",
            document_id, ws, delete_memories,
        )
        await service.delete_document(
            document_id, workspace_id=ws, delete_memories=delete_memories,
        )
        try:
            await audit_service.record(AuditEvent(
                event_type="document",
                action="delete",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="document",
                resource_id=document_id,
            ))
        except Exception:
            logger.debug("Audit record failed for document delete")

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except ValueError as exc:
        logger.warning("Delete rejected for document %s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to delete document %s: %s", document_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document",
        )


@router.post(
    "/{document_id}/reprocess",
    response_model=JobResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def reprocess_document(
    http_request: Request,
    document_id: str,
    target_context_id: Optional[str] = None,
    chunking_strategy: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    importance: Optional[float] = None,
    from_phase: str = Query("render", description="Phase to restart from: 'render' (full) or 'embed' (re-embed existing pages)"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DocumentIngestionService = Depends(get_document_service_dep),
    logger: Logger = Depends(get_logger),
) -> JobResponse:
    """Re-extract a document with optionally different extraction options.

    Creates a new ingestion job that reprocesses the document from its
    stored original file.

    Args:
        document_id: Document identifier.
        target_context_id: Override target context for memories.
        chunking_strategy: Override chunking strategy.
        chunk_size: Override chunk size.
        chunk_overlap: Override chunk overlap.
        importance: Override memory importance.
        from_phase: Pipeline phase to restart from. ``"render"`` (default)
            clears prior pages/derived blobs and re-renders; ``"embed"``
            re-embeds the existing rendered pages.
        workspace_id: Workspace ID from auth context.
        service: Document ingestion service instance.
        logger: Logger instance.

    Returns:
        JobResponse for the newly created reprocessing job.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "documents", "write", workspace_id=ctx.workspace_id)

        logger.info("Reprocess request for document %s", document_id)

        # Build extraction options only if any overrides are provided
        options = None
        overrides = {
            "target_context_id": target_context_id,
            "chunking_strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "importance": importance,
        }
        non_none = {k: v for k, v in overrides.items() if v is not None}
        if non_none:
            options = DocumentExtractionOptions(**non_none)

        job = await service.reprocess_document(
            document_id, workspace_id=ctx.workspace_id, options=options,
            from_phase=from_phase,
        )

        return JobResponse(**job.model_dump())

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except ValueError as exc:
        logger.warning("Reprocess rejected for document %s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to reprocess document %s: %s", document_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reprocess document",
        )


@router.post(
    "/{document_id}/verify",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def reverify_document(
    http_request: Request,
    document_id: str,
    workspace_id: Optional[str] = Query(
        None, description="Workspace of the document (defaults to the caller's context workspace)"
    ),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    task_service: TaskService = Depends(get_task_service),
    logger: Logger = Depends(get_logger),
) -> dict:
    """Re-verify / heal a SINGLE document — enqueues an on-demand ``doc_verify``
    that bypasses the reconcile attempt cap.

    Ported from the OSS documents router, which ``_supersede_oss_routers`` drops
    so the enterprise router can own ``/v1/documents``. This route existed ONLY
    in the OSS copy, so per-document reprocess 404'd on every enterprise
    deployment while the workspace/global sweep (``POST /v1/admin/documents/verify``,
    on the enterprise ADMIN router) worked — the superadmin dashboard calls both
    from one control, which is how the asymmetry surfaced.

    Distinct from ``POST /{document_id}/reprocess``: that re-runs extraction from
    a chosen phase, whereas this gap-analyzes and resumes only what is missing.

    Authorization: ``documents:write`` on the document's workspace.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        ws = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "documents", "write", workspace_id=ws)

        task_id = await task_service.schedule_task(
            "doc_verify",
            {"workspace_id": ws, "document_id": document_id},
            metadata={"bg_kind": "doc_verify", "visibility": "workspace", "title": "Reprocessing document"},
        )
        logger.info(
            "Enqueued doc_verify for document %s (workspace=%s task=%s)",
            document_id, ws, task_id,
        )
        return {"task_id": task_id, "mode": "document", "scheduled": task_id is not None}

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to enqueue doc_verify for document %s: %s", document_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enqueue document verification",
        )


# ------------------------------------------------------------------ #
# Plugin registration
# ------------------------------------------------------------------ #

class DocumentsAPIPlugin(Plugin):
    """Plugin to register document ingestion API routes."""

    def extension_point_name(self, v: Variables) -> str:
        """Return the multi-API-routers extension point."""
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        """Return the router for registration."""
        return router

    def is_enabled(self, v: Variables) -> bool:
        """Disable single-extension mode for a multi-extension plugin."""
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        """Mark this as a multi-extension plugin."""
        return True
