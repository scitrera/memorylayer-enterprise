"""Collection item model for MemoryLayer Enterprise vector collections."""
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class CollectionItem(BaseModel):
    """An item in a vector collection (tools, skills, procedures, etc.)."""

    model_config = {"from_attributes": True}

    id: str = Field(..., description="Item ID")
    tenant_id: str = Field(..., description="Tenant this item belongs to")
    workspace_id: str = Field(..., description="Workspace scope")
    collection_name: str = Field(..., description="Collection name (e.g. 'tools', 'skills', 'procedures')")
    name: str = Field(..., description="Item name")
    content: str = Field(..., description="Item content (text to embed)")
    item_type: Optional[str] = Field(None, description="Optional type within the collection")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    enabled: bool = Field(True, description="Whether the item is active")
    embedding: Optional[list[float]] = Field(None, description="Vector embedding (populated by service)")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("collection_name")
    @classmethod
    def collection_name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Collection name cannot be empty")
        return v.strip().lower()
