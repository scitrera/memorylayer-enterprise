"""Platform user model for MemoryLayer Enterprise."""
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class User(BaseModel):
    """Platform user within a tenant."""

    model_config = {"from_attributes": True}

    id: str = Field(..., description="User ID")
    tenant_id: str = Field(..., description="Tenant this user belongs to")
    email: str = Field(..., description="User email address")
    display_name: Optional[str] = Field(None, description="Display name")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    enabled: bool = Field(True, description="Whether the user account is enabled")
    licensed: bool = Field(False, description="Whether the user has a license seat")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("email")
    @classmethod
    def email_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Email cannot be empty")
        return v.strip().lower()
