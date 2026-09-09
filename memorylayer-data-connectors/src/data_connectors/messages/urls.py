# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""URL minting request/response models."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class MintUploadURLReq(BaseModel):
    """Request a presigned upload URL."""
    workspace_id: str = Field(..., description="Target workspace")
    filename: str = Field(..., description="Original filename")
    source_path: Optional[str] = Field(
        None,
        description=(
            "Path recorded on the VFS entry, enabling folder-style grouping "
            "and source_path_prefix listing (e.g. '/Bids/acme/report.pdf'). "
            "Defaults to `filename` when omitted. Deliberately SEPARATE from "
            "filename: filename is also the display name downstream (it flows "
            "into the ingested document), so overloading it would render a "
            "full path everywhere a file name is shown."
        ),
    )
    content_type: Optional[str] = Field(None, description="MIME type")
    size_bytes: Optional[int] = Field(None, description="Expected file size")
    connector_id: Optional[str] = Field("manual_upload", description="Connector sourcing the upload")
    metadata: dict = Field(default_factory=dict, description="Arbitrary metadata")
    method: Literal["POST", "PUT"] = Field(
        "POST",
        description=(
            "Upload protocol. POST → S3 browser-form multipart upload "
            "(generate_presigned_post); response carries policy `fields`. "
            "PUT → raw-body upload (generate_presigned_url put_object); "
            "response carries required signed `headers`."
        ),
    )


class UploadURLResponse(BaseModel):
    """Presigned upload URL with instructions.

    Callers branch on ``method``:
      * POST: build a multipart FormData with every entry in ``fields``,
        append the file as ``"file"``, and POST to ``upload_url``.
      * PUT: send the raw file body to ``upload_url`` with the entries in
        ``headers`` as request headers (and matching Content-Type / -Length).
    """
    method: Literal["POST", "PUT"] = Field("POST", description="Upload protocol used")
    upload_url: str = Field(..., description="Presigned upload URL")
    blob_key: str = Field(..., description="Blob storage key")
    vfs_ref: Optional[str] = Field(None, description="Pre-allocated VFS reference (if applicable)")
    fields: dict[str, str] = Field(
        default_factory=dict,
        description="Multipart form fields for POST uploads (empty for PUT)",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Required signed headers for PUT uploads (empty for POST)",
    )
    expires_at: datetime = Field(..., description="URL expiration time")


class MintDownloadURLReq(BaseModel):
    """Request a presigned download URL for a VFS entry."""
    vfs_ref: str = Field(..., description="VFS reference")
    workspace_id: str = Field(..., description="Workspace scope")
    subject: Optional[str] = Field(
        None,
        description=(
            "End-user identity the download token should bind to (the value "
            "auth-go stamps as X-Scitrera-User, i.e. the user's email). When "
            "absent, the token is minted with the dc service-id subject "
            "(back-compat)."
        ),
    )


class MintFetchURLReq(BaseModel):
    """Request a just-in-time fetch URL (used by MemoryLayer workers)."""
    vfs_ref: str = Field(..., description="VFS reference to fetch")
    workspace_id: str = Field(..., description="Workspace scope")


class FetchURLResponse(BaseModel):
    """Fetch/download URL with optional auth headers."""
    url: str = Field(..., description="Presigned or OAuth-bearer URL")
    headers: dict[str, str] = Field(default_factory=dict, description="Headers to include in GET")
    expires_at: datetime = Field(..., description="URL expiration time")
