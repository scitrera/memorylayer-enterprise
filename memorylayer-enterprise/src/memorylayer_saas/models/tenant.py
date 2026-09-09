# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""
Tenant model for MemoryLayer.ai v2.

The top-level isolation boundary. OSS uses single tenant "default_tenant".
"""
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TenantSettings(BaseModel):
    """Tenant-level configuration."""

    max_workspaces: Optional[int] = Field(None, description="Max workspaces (None=unlimited)")
    default_retention_days: int = Field(365, ge=1, description="Default memory retention")
    enable_global_workspace: bool = Field(True, description="Enable _global workspace")
    session_auto_commit: bool = Field(True, description="Auto-commit sessions on close")


class Tenant(BaseModel):
    """Top-level tenant for multi-tenancy (single tenant in OSS)."""

    model_config = {"from_attributes": True}

    id: str = Field(..., description="Tenant ID (default_tenant for OSS)")
    name: str = Field(..., description="Tenant name")
    settings: TenantSettings = Field(default_factory=TenantSettings)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        """Validate that id is not empty."""
        if not v or not v.strip():
            raise ValueError("Tenant id cannot be empty")
        return v.strip()

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        """Validate that name is not empty."""
        if not v or not v.strip():
            raise ValueError("Tenant name cannot be empty")
        return v.strip()
