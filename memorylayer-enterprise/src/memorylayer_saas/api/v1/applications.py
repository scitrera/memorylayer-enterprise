"""Application management API endpoints.

Endpoints:
- POST   /v1/applications                                              - Create application
- GET    /v1/applications                                              - List applications
- GET    /v1/applications/{id}                                         - Get application
- PUT    /v1/applications/{id}                                         - Update application
- DELETE /v1/applications/{id}                                         - Delete application
- POST   /v1/applications/{id}/workspaces/{ws_id}                      - Associate with workspace
- DELETE /v1/applications/{id}/workspaces/{ws_id}                      - Disassociate from workspace
- GET    /v1/workspaces/{ws_id}/applications                           - List workspace applications
- GET    /v1/workspaces/{ws_id}/applications/{app_id}                  - App bundle (?expand=skills,mcp,tools)
- POST   /v1/workspaces/{ws_id}/applications/{app_id}/skills/{skill_id}
- DELETE /v1/workspaces/{ws_id}/applications/{app_id}/skills/{skill_id}
- GET    /v1/workspaces/{ws_id}/applications/{app_id}/skills
- POST   /v1/workspaces/{ws_id}/applications/{app_id}/mcp-servers/{mcp_id}
- DELETE /v1/workspaces/{ws_id}/applications/{app_id}/mcp-servers/{mcp_id}
- GET    /v1/workspaces/{ws_id}/applications/{app_id}/mcp-servers
- PUT    /v1/workspaces/{ws_id}/applications/{app_id}/tools            - Replace tool_names list
"""
from datetime import datetime
from logging import Logger
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from scitrera_app_framework import Plugin, Variables

from memorylayer_server.api import EXT_MULTI_API_ROUTERS
from memorylayer_server.api.v1.deps import get_auth_service, get_authz_service, get_audit_service
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep
from memorylayer_server.services.authentication import AuthenticationService
from memorylayer_server.services.authorization import AuthorizationService
from memorylayer_server.services.audit import AuditService, AuditEvent

from ...models.application import Application, ApplicationBundle, OverrideMode
from ...services.application import ApplicationService

router = APIRouter(tags=["applications"])

_app_service: Optional[ApplicationService] = None


def _get_app_service(v: Variables = Depends(get_variables_dep)) -> ApplicationService:
    global _app_service
    if _app_service is None:
        from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
        from scitrera_app_framework import get_extension
        storage = get_extension(EXT_STORAGE_BACKEND, v)
        _app_service = ApplicationService(session_factory=storage.session_factory)
    return _app_service


# ------------------------------------------------------------------ #
# Request/Response models
# ------------------------------------------------------------------ #

class AppCreateRequest(BaseModel):
    id: Optional[str] = Field(None, description="Optional application ID. If omitted, a unique ID is generated.")
    name: str = Field(..., description="Application name")
    description: Optional[str] = Field(None, description="Description")
    app_type: str = Field("generic", description="Application type")
    enabled: bool = Field(True, description="Enabled state")
    config: dict[str, Any] = Field(default_factory=dict, description="Configuration")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    default_skill_names: list[str] = Field(default_factory=list, description="Default skill names")
    default_mcp_server_names: list[str] = Field(default_factory=list, description="Default MCP server names")
    default_tool_names: list[str] = Field(default_factory=list, description="Default tool names")


class AppUpdateRequest(BaseModel):
    name: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    app_type: Optional[str] = Field(None)
    enabled: Optional[bool] = Field(None)
    config: Optional[dict[str, Any]] = Field(None)
    metadata: Optional[dict[str, Any]] = Field(None)
    default_skill_names: Optional[list[str]] = Field(None)
    default_mcp_server_names: Optional[list[str]] = Field(None)
    default_tool_names: Optional[list[str]] = Field(None)


class AppResponse(BaseModel):
    id: str
    tenant_id: str
    name: str
    description: Optional[str] = None
    app_type: str = "generic"
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    default_skill_names: list[str] = Field(default_factory=list)
    default_mcp_server_names: list[str] = Field(default_factory=list)
    default_tool_names: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class AppListResponse(BaseModel):
    applications: list[AppResponse]
    total_count: int


class WorkspaceAssociateRequest(BaseModel):
    enabled: bool = Field(True, description="Enable app in workspace")
    config_overrides: dict[str, Any] = Field(default_factory=dict, description="Config overrides")
    tool_names: Optional[list[str]] = Field(None, description="Workspace-level tool name overrides")
    skill_override_mode: Optional[OverrideMode] = Field(None)
    mcp_override_mode: Optional[OverrideMode] = Field(None)
    tool_override_mode: Optional[OverrideMode] = Field(None)


class BindingIdsResponse(BaseModel):
    """Lightweight list of capability IDs bound to a (workspace, app)."""
    ids: list[str]


class SetToolsRequest(BaseModel):
    tool_names: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str


_EXPAND_KEYS = {"skills", "mcp", "tools"}


def _parse_expand(expand: Optional[str]) -> set[str]:
    if not expand:
        return set()
    parts = {p.strip() for p in expand.split(",") if p.strip()}
    bad = parts - _EXPAND_KEYS
    if bad:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid expand keys: {sorted(bad)}. Allowed: {sorted(_EXPAND_KEYS)}",
        )
    return parts


# ------------------------------------------------------------------ #
# Application CRUD endpoints
# ------------------------------------------------------------------ #

@router.post(
    "/v1/applications",
    response_model=AppResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def create_application(
    http_request: Request,
    request: AppCreateRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> AppResponse:
    """Create a new application."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "write")

        app = Application(
            id=request.id or f"app_{uuid4().hex[:16]}",
            tenant_id=ctx.tenant_id,
            name=request.name,
            description=request.description,
            app_type=request.app_type,
            enabled=request.enabled,
            config=request.config,
            metadata=request.metadata,
            default_skill_names=request.default_skill_names,
            default_mcp_server_names=request.default_mcp_server_names,
            default_tool_names=request.default_tool_names,
        )
        created = await service.create_application(app)

        try:
            await audit_service.record(AuditEvent(
                event_type="application",
                action="create",
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                resource_type="application",
                resource_id=created.id,
            ))
        except Exception:
            logger.debug("Audit record failed for application create")
        return AppResponse(**created.model_dump())

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to create application: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create application")


@router.get(
    "/v1/applications",
    response_model=AppListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_applications(
    http_request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    logger: Logger = Depends(get_logger),
) -> AppListResponse:
    """List applications for the tenant."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "read")

        apps, total = await service.list_applications(ctx.tenant_id, limit=limit, offset=offset)
        return AppListResponse(
            applications=[AppResponse(**a.model_dump()) for a in apps],
            total_count=total,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list applications: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list applications")


@router.get(
    "/v1/applications/{app_id}",
    response_model=AppResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_application(
    http_request: Request,
    app_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    logger: Logger = Depends(get_logger),
) -> AppResponse:
    """Get an application by ID."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "read")

        app = await service.get_application(app_id, ctx.tenant_id)
        if not app:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Application not found: {app_id}")
        return AppResponse(**app.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get application %s: %s", app_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get application")


@router.put(
    "/v1/applications/{app_id}",
    response_model=AppResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def update_application(
    http_request: Request,
    app_id: str,
    request: AppUpdateRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> AppResponse:
    """Update an application."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "write")

        updates = request.model_dump(exclude_none=True)
        app = await service.update_application(app_id, ctx.tenant_id, **updates)
        if not app:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Application not found: {app_id}")

        try:
            await audit_service.record(AuditEvent(
                event_type="application",
                action="update",
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                resource_type="application",
                resource_id=app_id,
            ))
        except Exception:
            logger.debug("Audit record failed for application update")
        return AppResponse(**app.model_dump())
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to update application %s: %s", app_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update application")


@router.delete(
    "/v1/applications/{app_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def delete_application(
    http_request: Request,
    app_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> None:
    """Delete an application."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "delete")

        deleted = await service.delete_application(app_id, ctx.tenant_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Application not found: {app_id}")

        try:
            await audit_service.record(AuditEvent(
                event_type="application",
                action="delete",
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                resource_type="application",
                resource_id=app_id,
            ))
        except Exception:
            logger.debug("Audit record failed for application delete")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete application %s: %s", app_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete application")


# ------------------------------------------------------------------ #
# Workspace association endpoints
# ------------------------------------------------------------------ #

@router.post(
    "/v1/applications/{app_id}/workspaces/{workspace_id}",
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def associate_workspace(
    http_request: Request,
    app_id: str,
    workspace_id: str,
    request: WorkspaceAssociateRequest = None,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    logger: Logger = Depends(get_logger),
):
    """Associate an application with a workspace."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "write", workspace_id=workspace_id)

        req = request or WorkspaceAssociateRequest()
        assoc = await service.associate_workspace(
            workspace_id=workspace_id,
            app_id=app_id,
            enabled=req.enabled,
            config_overrides=req.config_overrides,
            tool_names=req.tool_names,
            skill_override_mode=req.skill_override_mode,
            mcp_override_mode=req.mcp_override_mode,
            tool_override_mode=req.tool_override_mode,
        )
        return assoc.model_dump()
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to associate app %s with workspace %s: %s", app_id, workspace_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to associate application")


@router.delete(
    "/v1/applications/{app_id}/workspaces/{workspace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def disassociate_workspace(
    http_request: Request,
    app_id: str,
    workspace_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    logger: Logger = Depends(get_logger),
) -> None:
    """Remove an application from a workspace."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "write", workspace_id=workspace_id)

        deleted = await service.disassociate_workspace(workspace_id, app_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Association not found: app={app_id}, workspace={workspace_id}",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to disassociate app %s from workspace %s: %s", app_id, workspace_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to disassociate application")


@router.get(
    "/v1/workspaces/{workspace_id}/applications",
    response_model=AppListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_workspace_applications(
    http_request: Request,
    workspace_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    logger: Logger = Depends(get_logger),
) -> AppListResponse:
    """List applications associated with a workspace."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "read", workspace_id=workspace_id)

        apps = await service.list_workspace_applications(workspace_id)
        return AppListResponse(
            applications=[AppResponse(**a.model_dump()) for a in apps],
            total_count=len(apps),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list workspace applications: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list workspace applications")


# ------------------------------------------------------------------ #
# Application bundle (workspace-scoped) + capability bindings
# ------------------------------------------------------------------ #

@router.get(
    "/v1/workspaces/{workspace_id}/applications/{app_id}",
    response_model=ApplicationBundle,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_workspace_application_bundle(
    http_request: Request,
    workspace_id: str,
    app_id: str,
    expand: Optional[str] = Query(
        None,
        description="Comma-separated list of {skills, mcp, tools} to expand in the response",
    ),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    logger: Logger = Depends(get_logger),
) -> ApplicationBundle:
    """Return the application bundle (app + binding + optional expanded capabilities)."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "read", workspace_id=workspace_id)

        expand_set = _parse_expand(expand)
        bundle = await service.get_application_bundle(
            workspace_id=workspace_id,
            app_id=app_id,
            tenant_id=ctx.tenant_id,
            expand=expand_set,
        )
        if bundle is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Application not found: {app_id}",
            )
        return bundle
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to load application bundle ws=%s app=%s: %s",
            workspace_id, app_id, e, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load application bundle",
        )


# ---- skill bindings ----

@router.post(
    "/v1/workspaces/{workspace_id}/applications/{app_id}/skills/{skill_id}",
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def bind_skill(
    http_request: Request,
    workspace_id: str,
    app_id: str,
    skill_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> dict[str, str]:
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "write", workspace_id=workspace_id)
        await service.add_skill_to_binding(workspace_id, app_id, skill_id)

        try:
            await audit_service.record(AuditEvent(
                event_type="application",
                action="bind_skill",
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                workspace_id=workspace_id,
                resource_type="application",
                resource_id=app_id,
                metadata={"skill_id": skill_id},
            ))
        except Exception:
            logger.debug("Audit record failed for bind_skill")

        return {"workspace_id": workspace_id, "application_id": app_id, "skill_id": skill_id}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to bind skill %s: %s", skill_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to bind skill")


@router.delete(
    "/v1/workspaces/{workspace_id}/applications/{app_id}/skills/{skill_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def unbind_skill(
    http_request: Request,
    workspace_id: str,
    app_id: str,
    skill_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> None:
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "write", workspace_id=workspace_id)
        deleted = await service.remove_skill_from_binding(workspace_id, app_id, skill_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill binding not found: app={app_id} skill={skill_id}",
            )
        try:
            await audit_service.record(AuditEvent(
                event_type="application",
                action="unbind_skill",
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                workspace_id=workspace_id,
                resource_type="application",
                resource_id=app_id,
                metadata={"skill_id": skill_id},
            ))
        except Exception:
            logger.debug("Audit record failed for unbind_skill")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to unbind skill %s: %s", skill_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to unbind skill")


@router.get(
    "/v1/workspaces/{workspace_id}/applications/{app_id}/skills",
    response_model=BindingIdsResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_binding_skills(
    http_request: Request,
    workspace_id: str,
    app_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    logger: Logger = Depends(get_logger),
) -> BindingIdsResponse:
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "read", workspace_id=workspace_id)
        ids = await service.list_binding_skill_ids(workspace_id, app_id)
        return BindingIdsResponse(ids=ids)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list binding skills: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list binding skills")


# ---- MCP-server bindings ----

@router.post(
    "/v1/workspaces/{workspace_id}/applications/{app_id}/mcp-servers/{mcp_id}",
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def bind_mcp_server(
    http_request: Request,
    workspace_id: str,
    app_id: str,
    mcp_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> dict[str, str]:
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "write", workspace_id=workspace_id)
        await service.add_mcp_server_to_binding(workspace_id, app_id, mcp_id)
        try:
            await audit_service.record(AuditEvent(
                event_type="application",
                action="bind_mcp_server",
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                workspace_id=workspace_id,
                resource_type="application",
                resource_id=app_id,
                metadata={"mcp_server_id": mcp_id},
            ))
        except Exception:
            logger.debug("Audit record failed for bind_mcp_server")
        return {"workspace_id": workspace_id, "application_id": app_id, "mcp_server_id": mcp_id}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to bind mcp server %s: %s", mcp_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to bind MCP server")


@router.delete(
    "/v1/workspaces/{workspace_id}/applications/{app_id}/mcp-servers/{mcp_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def unbind_mcp_server(
    http_request: Request,
    workspace_id: str,
    app_id: str,
    mcp_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> None:
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "write", workspace_id=workspace_id)
        deleted = await service.remove_mcp_server_from_binding(workspace_id, app_id, mcp_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server binding not found: app={app_id} mcp={mcp_id}",
            )
        try:
            await audit_service.record(AuditEvent(
                event_type="application",
                action="unbind_mcp_server",
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                workspace_id=workspace_id,
                resource_type="application",
                resource_id=app_id,
                metadata={"mcp_server_id": mcp_id},
            ))
        except Exception:
            logger.debug("Audit record failed for unbind_mcp_server")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to unbind mcp server %s: %s", mcp_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to unbind MCP server")


@router.get(
    "/v1/workspaces/{workspace_id}/applications/{app_id}/mcp-servers",
    response_model=BindingIdsResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_binding_mcp_servers(
    http_request: Request,
    workspace_id: str,
    app_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    logger: Logger = Depends(get_logger),
) -> BindingIdsResponse:
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "read", workspace_id=workspace_id)
        ids = await service.list_binding_mcp_server_ids(workspace_id, app_id)
        return BindingIdsResponse(ids=ids)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list binding mcp servers: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list binding MCP servers")


# ---- tools (replace whole list) ----

@router.put(
    "/v1/workspaces/{workspace_id}/applications/{app_id}/tools",
    response_model=SetToolsRequest,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def set_binding_tools(
    http_request: Request,
    workspace_id: str,
    app_id: str,
    request: SetToolsRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: ApplicationService = Depends(_get_app_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> SetToolsRequest:
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "applications", "write", workspace_id=workspace_id)

        # Validate names with the shared capability-name rule by piggy-backing
        # on the Application validator (defensive parse).
        from ...models.application import _validate_name_list  # noqa: WPS437 — local helper
        validated = _validate_name_list(request.tool_names)

        await service.set_binding_tool_names(workspace_id, app_id, validated)

        try:
            await audit_service.record(AuditEvent(
                event_type="application",
                action="set_tools",
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                workspace_id=workspace_id,
                resource_type="application",
                resource_id=app_id,
                metadata={"tool_count": len(validated)},
            ))
        except Exception:
            logger.debug("Audit record failed for set_tools")

        return SetToolsRequest(tool_names=validated)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to set binding tools: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to set binding tools")


# ------------------------------------------------------------------ #
# Plugin registration
# ------------------------------------------------------------------ #

class ApplicationsAPIPlugin(Plugin):
    """Plugin to register application API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True
