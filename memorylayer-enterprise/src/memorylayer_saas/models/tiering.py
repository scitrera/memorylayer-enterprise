from typing import Any, Optional

from pydantic import BaseModel, Field


# Request Schemas
class ArchiveRequest(BaseModel):
    """Request schema for archiving memories to cold tier."""

    memory_ids: Optional[list[str]] = Field(
        None,
        description="Specific memory IDs to archive. If not provided, auto_detect must be True."
    )
    auto_detect: bool = Field(
        False,
        description="Automatically detect archival candidates based on thresholds."
    )
    max_importance: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Maximum importance threshold for auto-detection."
    )
    max_access_count: Optional[int] = Field(
        None,
        ge=0,
        description="Maximum access count threshold for auto-detection."
    )
    older_than_days: Optional[int] = Field(
        None,
        ge=0,
        description="Minimum days since last access for auto-detection."
    )
    batch_size: int = Field(
        100,
        ge=1,
        le=1000,
        description="Maximum number of memories to archive in one operation."
    )


class RestoreRequest(BaseModel):
    """Request schema for restoring memories from cold tier."""

    memory_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Memory IDs to restore from cold tier."
    )
    regenerate_embeddings: bool = Field(
        True,
        description="Regenerate embeddings after restore."
    )


class TieringConfigUpdateRequest(BaseModel):
    """Request schema for updating tiering configuration."""

    cold_tier_enabled: Optional[bool] = Field(
        None,
        description="Enable or disable cold tier storage."
    )
    archival_age_days: Optional[int] = Field(
        None,
        ge=1,
        description="Minimum days since last access before archival."
    )
    min_importance_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Maximum importance score for archival."
    )
    min_access_count_threshold: Optional[int] = Field(
        None,
        ge=0,
        description="Maximum access count for archival."
    )
    archival_batch_size: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="Batch size for archival operations."
    )
    warmup_access_threshold: Optional[int] = Field(
        None,
        ge=1,
        description="Cold access count to trigger promotion."
    )
    warmup_batch_size: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="Batch size for warm-up operations."
    )
    cold_tier_search_enabled: Optional[bool] = Field(
        None,
        description="Enable cold tier search in recall."
    )


# Response Schemas
class TieringStatsResponse(BaseModel):
    """Response schema for tiering statistics."""

    hot_memory_count: int = Field(..., description="Number of memories in hot tier")
    cold_memory_count: int = Field(..., description="Number of memories in cold tier")
    hot_storage_bytes: int = Field(..., description="Storage bytes used by hot tier")
    cold_storage_bytes: int = Field(..., description="Storage bytes used by cold tier")
    compression_ratio: float = Field(..., description="Cold/hot storage ratio (lower is better)")
    estimated_savings_bytes: int = Field(..., description="Estimated storage savings from cold tier")
    archival_candidates_count: int = Field(..., description="Number of memories eligible for archival")
    # Actual on-disk size of the document tables (pg_total_relation_size: heap +
    # TOAST + indexes). Measured disk, unlike the hot/cold estimates. Populated
    # by the tenant-wide admin overview; 0 on the per-workspace endpoint.
    document_storage_bytes: int = Field(
        0, description="On-disk size of documents + document_pages tables (measured, not estimated)"
    )
    documents_table_bytes: int = Field(
        0, description="On-disk size of the documents table"
    )
    document_pages_bytes: int = Field(
        0, description="On-disk size of the document_pages table (transcripts + page embeddings + multivectors)"
    )


class AdminTieringRunRequest(BaseModel):
    """Request schema for an on-demand cross-workspace tiering sweep."""

    dry_run: bool = Field(
        True,
        description="Identify candidates only; archive nothing. Default True (safe).",
    )
    only_enabled: bool = Field(
        True,
        description="Only sweep workspaces whose cold_tier_enabled is set. False sweeps all.",
    )
    batch_size: int = Field(
        100, ge=1, le=1000,
        description="Max memories to archive per workspace this pass.",
    )
    max_workspaces: Optional[int] = Field(
        None, ge=1,
        description="Optional cap on the number of workspaces processed.",
    )


class AdminTieringRunResponse(BaseModel):
    """Response schema for an on-demand cross-workspace tiering sweep."""

    dry_run: bool
    only_enabled: bool
    workspaces_processed: int
    workspaces_skipped: int
    total_candidates: int
    total_archived: int
    total_failed: int
    per_workspace: list[dict[str, Any]] = Field(default_factory=list)


class ArchiveResponse(BaseModel):
    """Response schema for archive operation."""

    archived_count: int = Field(..., description="Number of memories successfully archived")
    failed_count: int = Field(..., description="Number of memories that failed to archive")
    archived_memory_ids: list[str] = Field(..., description="IDs of successfully archived memories")
    failed_memory_ids: list[str] = Field(..., description="IDs of memories that failed to archive")


class RestoreResponse(BaseModel):
    """Response schema for restore operation."""

    restored_count: int = Field(..., description="Number of memories successfully restored")
    failed_count: int = Field(..., description="Number of memories that failed to restore")
    restored_memory_ids: list[str] = Field(..., description="IDs of successfully restored memories")
    failed_memory_ids: list[str] = Field(..., description="IDs of memories that failed to restore")


class TieringConfigResponse(BaseModel):
    """Response schema for tiering configuration."""

    cold_tier_enabled: bool
    archival_age_days: int
    min_importance_threshold: float
    min_access_count_threshold: int
    archival_batch_size: int
    warmup_access_threshold: int
    warmup_batch_size: int
    cold_retrieval_latency_target_ms: int
    cold_tier_search_enabled: bool


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict[str, Any]] = Field(None, description="Additional error details")
