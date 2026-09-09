# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Document ingestion domain models."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PENDING_FETCH = "pending_fetch"  # awaiting byte-fetch from data-connectors
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class DocumentEnrichmentStatus(str, Enum):
    """Knowledge-phase status, tracked separately from ``DocumentStatus``.

    ``DocumentStatus`` answers "can this document be retrieved against?" — it
    goes COMPLETED once pages, transcripts, embeddings and composite memories
    are durable. Fact decomposition and the enrichment that follows it are
    deliberately NOT part of that: they fan out to thousands of background tasks
    (a decomposed page yields ~35-40 facts, each scheduling its own tiering and
    association work) and can run long after the document is usable. Callers
    that only search or read pages must not wait on them.

    This enum answers the second question — "is the knowledge extraction for
    this document finished?" — for the callers that do care.
    """

    #: Nothing was scheduled: decomposition is disabled, or no memory qualified.
    #: A terminal state, distinct from COMPLETE so "never ran" stays legible.
    NOT_APPLICABLE = "not_applicable"
    #: Decomposition was scheduled and at least one memory is still outstanding.
    PENDING = "pending"
    #: Every scheduled decomposition has finished.
    COMPLETE = "complete"


class JobStatus(str, Enum):
    """Ingestion job status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    DOCX = "docx"
    PPTX = "pptx"


class DocumentExtractionOptions(BaseModel):
    """Options for document extraction and chunking."""
    chunking_strategy: str = Field("page", description="Chunking strategy: page, semantic, fixed")
    chunk_size: int = Field(4096, description="Max chunk size in characters")
    chunk_overlap: int = Field(200, description="Overlap between chunks in characters")
    system_prompt: Optional[str] = Field(None, description="Custom transcription system prompt")
    # target_context_id sets each extracted memory's context_id. RESERVED /
    # unused as a retrieval filter today — see MemoryModel.context_id.
    target_context_id: str = Field("_default", description="Target context for memories")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Default importance for memories")
    retain_original: bool = Field(True, description="Keep original file in blob storage")


class Document(BaseModel):
    """Document domain model."""
    model_config = {"from_attributes": True}

    id: str
    workspace_id: str
    tenant_id: str = "default_tenant"
    filename: str
    document_type: DocumentType
    content_hash: str
    source_vfs_ref: Optional[str] = None
    size_bytes: int
    mime_type: Optional[str] = None
    status: DocumentStatus = DocumentStatus.PENDING
    enrichment_status: DocumentEnrichmentStatus = Field(
        DocumentEnrichmentStatus.NOT_APPLICABLE,
        description=(
            "Knowledge-phase status, independent of `status`. See "
            "DocumentEnrichmentStatus: `status` says the document is usable, "
            "this says whether fact extraction has finished."
        ),
    )
    enrichment_memory_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Memories scheduled for fact decomposition at ingest. Recorded so "
            "completion can be re-derived from stored state rather than counted "
            "from events, which keeps it correct across retries and restarts."
        ),
    )
    target_context_id: str = "_default"
    extraction_options: DocumentExtractionOptions = Field(default_factory=DocumentExtractionOptions)
    page_count: int = 0
    chunk_count: int = 0
    memory_ids: list[str] = Field(default_factory=list)
    deduplicated_count: int = 0
    storage_path: Optional[str] = None
    retain_original: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    extracted_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None


class IngestionJob(BaseModel):
    """Ingestion job domain model."""
    model_config = {"from_attributes": True}

    id: str
    workspace_id: str
    document_ids: list[str] = Field(default_factory=list)
    status: JobStatus = JobStatus.QUEUED
    progress_percent: int = 0
    documents_processed: int = 0
    total_memories_created: int = 0
    webhook_url: Optional[str] = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class DocumentPage(BaseModel):
    """Document page - persisted page from an ingested document."""
    model_config = {"from_attributes": True}

    id: Optional[str] = Field(None, description="Page identifier (generated on persist)")
    document_id: Optional[str] = Field(None, description="Parent document identifier")
    workspace_id: Optional[str] = Field(None, description="Workspace identifier")
    page_no: int = Field(..., description="Zero-indexed page number")
    image_storage_path: Optional[str] = Field(None, description="Blob storage path for rendered page image")
    image_b64: Optional[str] = Field(None, description="Transient base64 image data (not persisted)", exclude=True)
    transcript: Optional[str] = Field(None, description="VLM/OCR transcription as markdown")
    embedding: Optional[list[float]] = Field(None, description="Single-vector text embedding", exclude=True)
    multivector: Optional[list[list[float]]] = Field(None, description="ColPali multi-vector embedding")
    multivector_storage_path: Optional[str] = Field(
        None,
        description=(
            "Blob storage path for a spilled multivector (transient; set by the "
            "embed phase so the vector need not stay resident until persist)"
        ),
        exclude=True,
    )
    transcript_model: Optional[str] = Field(None, description="Model used for transcription")
    transcript_attempts: dict[str, Any] = Field(default_factory=dict, description="Transcription attempt log")
    visual_tokens: Optional[dict[str, Any]] = Field(None, description="Visual tokenizer output")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary page metadata")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class DocumentChunk(BaseModel):
    """Intermediate chunk during processing."""
    content: str
    page_no: int
    chunk_index: int = 0
    embedding: Optional[list[float]] = None
    multivector: Optional[list[list[float]]] = None
