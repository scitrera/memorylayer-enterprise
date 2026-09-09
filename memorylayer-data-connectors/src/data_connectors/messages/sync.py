"""Sync engine request/response models."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TriggerSyncReq(BaseModel):
    """Trigger a sync for a specific provider."""
    provider_id: str = Field(..., description="Provider to sync")
    workspace_id: str = Field(..., description="Workspace scope")
    full_sync: bool = Field(False, description="Force full resync (ignore checkpoint)")


class SyncJobResponse(BaseModel):
    """Sync job status."""
    job_id: str
    provider_id: str
    workspace_id: str
    status: str = Field(..., description="pending, running, completed, failed")
    entries_discovered: int = 0
    entries_synced: int = 0
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
