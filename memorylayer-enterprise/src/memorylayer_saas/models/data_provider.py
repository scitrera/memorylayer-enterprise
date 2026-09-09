# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Data provider model for MemoryLayer Enterprise."""
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class DataProvider(BaseModel):
    """Data provider registry entry for document ingestion sources."""

    model_config = {"from_attributes": True}

    id: str = Field(..., description="Provider ID")
    tenant_id: str = Field(..., description="Tenant this provider belongs to")
    workspace_id: str = Field(..., description="Workspace scope")
    name: str = Field(..., description="Provider name")
    provider_type: str = Field(
        ..., description="Provider type (s3, gcs, azure_blob, sharepoint, confluence, web)"
    )
    description: Optional[str] = Field(None, description="Provider description")
    enabled: bool = Field(True, description="Whether the provider is active")
    connection_args: dict[str, Any] = Field(
        default_factory=dict, description="Connection arguments (non-sensitive)"
    )
    # encrypted_args stored in DB but never exposed in API responses
    encrypted_args: Optional[dict[str, Any]] = Field(
        None, description="Encrypted connection arguments (secrets)"
    )
    schedule: Optional[str] = Field(None, description="Cron schedule for auto-sync")
    last_sync_at: Optional[datetime] = Field(None, description="Last successful sync time")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Provider name cannot be empty")
        return v.strip()
