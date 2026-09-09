# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Provider (data source) request/response models."""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class CreateProviderReq(BaseModel):
    """Create a new data provider (connector source)."""
    name: str = Field(..., description="Human-readable provider name")
    provider_type: str = Field(..., description="Connector type: manual_upload, s3, web_scraper")
    description: Optional[str] = Field(None, description="Optional description")
    enabled: bool = Field(True, description="Whether the provider is active")
    connection_args: dict[str, Any] = Field(default_factory=dict, description="Connection configuration")
    schedule: Optional[str] = Field(None, description="Cron schedule for auto-sync")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class UpdateProviderReq(BaseModel):
    """Partial update to an existing provider."""
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    connection_args: Optional[dict[str, Any]] = None
    schedule: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class ProviderResponse(BaseModel):
    """Provider response (never exposes secrets)."""
    id: str
    workspace_id: str
    name: str
    provider_type: str
    description: Optional[str] = None
    enabled: bool = True
    connection_args: dict[str, Any] = Field(default_factory=dict)
    schedule: Optional[str] = None
    last_sync_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ProviderListResponse(BaseModel):
    """Paginated list of providers."""
    providers: list[ProviderResponse]
    total_count: int
