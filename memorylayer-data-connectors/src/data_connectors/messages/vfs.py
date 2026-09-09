# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""VFS catalog entry request/response models."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class RegisterVfsEntryReq(BaseModel):
    """Register a new VFS entry in the catalog."""
    workspace_id: str = Field(..., description="Owning workspace")
    connector_id: str = Field(..., description="Provider/connector that sourced this entry")
    source_path: str = Field(..., description="Original path/key within the source")
    content_hash: str = Field(..., description="Content hash for dedup (e.g. SHA-256)")
    content_type: Optional[str] = Field(None, description="MIME type")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    metadata: dict = Field(default_factory=dict, description="Arbitrary metadata")
    initiated_by: Optional[str] = Field(
        None,
        description="Originating party for user-initiated ingest (e.g. 'us::{user}'). "
        "When set, the emitted doc_added task is classed BATCH; when None, BACKGROUND.",
    )
    visibility: str = Field(
        "workspace",
        description="Task-level visibility for Background Tasks: 'workspace' (default) or 'private'.",
    )


class UpdateVfsEntryReq(BaseModel):
    """Partial update to a VFS entry."""
    content_hash: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    metadata: Optional[dict] = None


class FinalizeVfsEntryReq(BaseModel):
    """Finalize a placeholder VFS entry after its blob has been uploaded.

    ``mint_upload_url`` pre-registers a placeholder VFS entry with an empty
    ``content_hash`` and ``size_bytes=None`` so callers have a stable
    ``vfs_ref`` before the upload completes. Once the browser has PUT/POSTed
    the blob to S3, the ws-server calls this route to (a) backfill
    ``content_hash`` / ``size_bytes`` (derived server-side from the uploaded
    blob when not supplied) and (b) emit the user-initiated (BATCH) doc_added
    ingest task. This is the user-upload analogue of connector-driven
    ``register_vfs_entry`` — it does NOT create a second VFS entry.
    """
    content_hash: Optional[str] = Field(
        None,
        description="Content hash for dedup. Derived from the uploaded blob "
        "server-side when omitted.",
    )
    content_type: Optional[str] = Field(None, description="MIME type override")
    size_bytes: Optional[int] = Field(
        None,
        description="File size in bytes. Derived from the uploaded blob "
        "server-side when omitted.",
    )
    metadata: Optional[dict] = Field(None, description="Metadata to merge into the entry")
    initiated_by: Optional[str] = Field(
        None,
        description="Originating party for user-initiated ingest (e.g. 'us::{user}'). "
        "When set, the emitted doc_added task is classed BATCH; when None, BACKGROUND.",
    )
    visibility: str = Field(
        "workspace",
        description="Task-level visibility for Background Tasks: 'workspace' (default) or 'private'.",
    )
    skip_ingest: bool = Field(
        False,
        description="Commit the blob (edge finalize -> content_hash/size backfill) but do "
        "NOT emit the doc_added ingest task. Used for agent-produced output artifacts "
        "(present_artifact) that must be downloadable/renderable via their vfs_ref but "
        "should not be ingested into the knowledge base.",
    )


class LinkMlDocumentReq(BaseModel):
    """Link a VFS entry to a MemoryLayer document."""
    ml_doc_id: str = Field(..., description="MemoryLayer document ID")
    ml_job_id: Optional[str] = Field(None, description="MemoryLayer ingestion job ID")


class VfsEntryResponse(BaseModel):
    """VFS catalog entry."""
    vfs_ref: str
    workspace_id: str
    connector_id: str
    source_path: str
    content_hash: str
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    blob_key: Optional[str] = None
    ml_doc_id: Optional[str] = None
    ml_job_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class VfsEntryListResponse(BaseModel):
    """Paginated list of VFS entries."""
    entries: list[VfsEntryResponse]
    total_count: int
