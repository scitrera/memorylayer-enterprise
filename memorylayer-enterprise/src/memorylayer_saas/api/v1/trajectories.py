# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Trajectory endpoints for retrieval observability."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from scitrera_app_framework import Plugin, get_extension
from scitrera_app_framework.api import Variables

from memorylayer_server.api import EXT_MULTI_API_ROUTERS
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep
from memorylayer_server.services.authentication import AuthenticationService, EXT_AUTHENTICATION_SERVICE
from memorylayer_server.services.authorization import AuthorizationService, EXT_AUTHORIZATION_SERVICE
from memorylayer_server.services.audit import AuditService, AuditEvent, EXT_AUDIT_SERVICE

from ...models.trajectory import Trajectory, TrajectoryListResponse
from ...services.trajectory import get_trajectory_service as _get_trajectory_service, TrajectoryService


router = APIRouter(prefix="/trajectories", tags=["trajectories"])


async def get_auth_service(v: Variables = Depends(get_variables_dep)) -> AuthenticationService:
    """Get authentication service instance."""
    return get_extension(EXT_AUTHENTICATION_SERVICE, v)


async def get_authz_service(v: Variables = Depends(get_variables_dep)) -> AuthorizationService:
    """Get authorization service instance."""
    return get_extension(EXT_AUTHORIZATION_SERVICE, v)


def get_audit_service(v: Variables = Depends(get_variables_dep)) -> AuditService:
    """Get audit service instance."""
    return get_extension(EXT_AUDIT_SERVICE, v)


def get_trajectory_service(v: Variables = Depends(get_variables_dep)) -> TrajectoryService:
    """FastAPI dependency wrapper for trajectory service."""
    return _get_trajectory_service(v)


@router.get(
    "/{trajectory_id}",
    response_model=Trajectory,
    responses={
        401: {"description": "Authentication failed"},
        403: {"description": "Authorization denied"},
        404: {"description": "Trajectory not found"},
    },
)
async def get_trajectory(
    http_request: Request,
    trajectory_id: str,
    workspace_id: str = Query(..., description="Workspace ID for authorization"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: TrajectoryService = Depends(get_trajectory_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger)
) -> Trajectory:
    """Get a trajectory by ID."""
    ctx = await auth_service.build_context(http_request, workspace_id)
    await authz_service.require_authorization(
        ctx, "trajectories", "read",
        resource_id=trajectory_id, workspace_id=workspace_id
    )

    logger.debug("Fetching trajectory: %s for workspace: %s", trajectory_id, workspace_id)

    trajectory = await service.get_trajectory(trajectory_id, workspace_id)

    if trajectory is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trajectory {trajectory_id} not found or expired"
        )

    try:
        await audit_service.record(AuditEvent(
            event_type="trajectory",
            action="read",
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            user_id=ctx.user_id,
            resource_type="trajectory",
            resource_id=trajectory_id,
        ))
    except Exception:
        logger.debug("Audit record failed for trajectory read")
    return trajectory


@router.get(
    "",
    response_model=TrajectoryListResponse,
    responses={
        401: {"description": "Authentication failed"},
        403: {"description": "Authorization denied"},
    },
)
async def list_trajectories(
    http_request: Request,
    workspace_id: str = Query(..., description="Workspace ID to filter by"),
    limit: int = Query(20, ge=1, le=100, description="Maximum trajectories to return"),
    offset: int = Query(0, ge=0, description="Number of trajectories to skip"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: TrajectoryService = Depends(get_trajectory_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger)
) -> TrajectoryListResponse:
    """List trajectories for a workspace."""
    ctx = await auth_service.build_context(http_request, workspace_id)
    await authz_service.require_authorization(
        ctx, "trajectories", "read", workspace_id=workspace_id
    )

    logger.debug(
        "Listing trajectories for workspace: %s (limit=%d, offset=%d)",
        workspace_id, limit, offset
    )

    result = await service.list_trajectories(
        workspace_id=workspace_id,
        limit=limit,
        offset=offset
    )
    try:
        await audit_service.record(AuditEvent(
            event_type="trajectory",
            action="list",
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            user_id=ctx.user_id,
            resource_type="trajectory",
        ))
    except Exception:
        logger.debug("Audit record failed for trajectory list")
    return result


class TrajectoriesAPIPlugin(Plugin):
    """Plugin to register trajectories API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
