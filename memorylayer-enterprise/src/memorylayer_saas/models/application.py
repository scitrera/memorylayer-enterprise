# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Application and workspace-application junction models for MemoryLayer Enterprise."""
import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Spec-mandated name rules shared with skills/mcp_servers: 1-64 chars, [a-z0-9-],
# no leading/trailing/consecutive hyphens. Mirrors validate_skill_name in OSS.
_CAPABILITY_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
OverrideMode = Literal["merge", "replace"]


def _validate_capability_name(name: str) -> str:
    if not name:
        raise ValueError("Capability name cannot be empty")
    if len(name) > 64:
        raise ValueError(f"Capability name must be 64 chars or fewer, got {len(name)}")
    if "--" in name:
        raise ValueError("Capability name cannot contain consecutive hyphens")
    if not _CAPABILITY_NAME_RE.match(name):
        raise ValueError(
            "Capability name must contain only lowercase letters, digits, and hyphens, "
            "and must not start or end with a hyphen"
        )
    return name


def _validate_name_list(values: list[str]) -> list[str]:
    return [_validate_capability_name(v) for v in values]


class Application(BaseModel):
    """Registered application in the platform."""

    model_config = {"from_attributes": True}

    id: str = Field(..., description="Application ID")
    tenant_id: str = Field(..., description="Tenant this application belongs to")
    name: str = Field(..., description="Application name")
    description: Optional[str] = Field(None, description="Application description")
    app_type: str = Field("generic", description="Application type (generic, chat, agent, tool)")
    enabled: bool = Field(True, description="Whether the application is enabled")
    config: dict[str, Any] = Field(default_factory=dict, description="Application configuration")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    # Tenant-level capability defaults. Resolved per workspace at bundle-load time.
    default_skill_names: list[str] = Field(
        default_factory=list, description="Default skill names attached to this application"
    )
    default_mcp_server_names: list[str] = Field(
        default_factory=list, description="Default MCP server names attached to this application"
    )
    default_tool_names: list[str] = Field(
        default_factory=list, description="Default tool names (plugin-resolved at runtime)"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Application name cannot be empty")
        return v.strip()

    @field_validator("default_skill_names", "default_mcp_server_names", "default_tool_names")
    @classmethod
    def validate_name_arrays(cls, v: list[str]) -> list[str]:
        return _validate_name_list(v)


class WorkspaceApplication(BaseModel):
    """Junction model linking an application to a workspace."""

    model_config = {"from_attributes": True}

    workspace_id: str = Field(..., description="Workspace ID")
    application_id: str = Field(..., description="Application ID")
    enabled: bool = Field(True, description="Whether the app is enabled in this workspace")
    config_overrides: dict[str, Any] = Field(
        default_factory=dict, description="Workspace-specific config overrides"
    )
    # Workspace-level capability overrides. Tool overrides live here as a list
    # of names (no tools table); skill/mcp overrides live in junction tables.
    tool_names: list[str] = Field(
        default_factory=list, description="Workspace-level tool name overrides"
    )
    skill_override_mode: OverrideMode = Field(
        "merge", description="How to combine binding skills with app defaults"
    )
    mcp_override_mode: OverrideMode = Field(
        "merge", description="How to combine binding MCP servers with app defaults"
    )
    tool_override_mode: OverrideMode = Field(
        "merge", description="How to combine binding tool_names with app default_tool_names"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("tool_names")
    @classmethod
    def validate_tool_names(cls, v: list[str]) -> list[str]:
        return _validate_name_list(v)


class ApplicationBundle(BaseModel):
    """Application + resolved capability set for a given workspace.

    Returned by GET /v1/workspaces/{ws}/applications/{app}. The ``skills``,
    ``mcp_servers``, and ``tool_names`` fields are populated based on the
    ``expand`` query (None = not requested). Skill/MCP rows are returned as
    dicts to keep this module independent of the OSS Skill/McpServer Pydantic
    classes — the caller already trusts the underlying ORM serialization.
    """

    model_config = {"from_attributes": True}

    application: Application
    workspace_binding: Optional[WorkspaceApplication] = None
    skills: Optional[list[dict[str, Any]]] = Field(
        None, description="Resolved skills when expand=skills was requested"
    )
    mcp_servers: Optional[list[dict[str, Any]]] = Field(
        None, description="Resolved MCP servers when expand=mcp was requested"
    )
    tool_names: Optional[list[str]] = Field(
        None, description="Resolved tool names when expand=tools was requested"
    )
