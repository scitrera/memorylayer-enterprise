"""User management API endpoints.

Endpoints:
- POST   /v1/users          - Create a user
- GET    /v1/users          - List users
- GET    /v1/users/{id}     - Get user
- PUT    /v1/users/{id}     - Update user
- DELETE /v1/users/{id}     - Delete user
"""
from datetime import datetime
from logging import Logger
from typing import Any, Optional
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

from ...models.user import User
from ...services.user import UserService

router = APIRouter(prefix="/v1/users", tags=["users"])

# Singleton service instance
_user_service: Optional[UserService] = None


def _get_user_service(v: Variables = Depends(get_variables_dep)) -> UserService:
    global _user_service
    if _user_service is None:
        from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
        from scitrera_app_framework import get_extension
        storage = get_extension(EXT_STORAGE_BACKEND, v)
        _user_service = UserService(
            session_factory=storage.session_factory,
            logger=None,
        )
    return _user_service


# ------------------------------------------------------------------ #
# Request/Response models
# ------------------------------------------------------------------ #

class UserCreateRequest(BaseModel):
    email: str = Field(..., description="User email address")
    display_name: Optional[str] = Field(None, description="Display name")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    enabled: bool = Field(True, description="Whether the user is enabled")
    licensed: bool = Field(False, description="Whether the user has a license")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class UserUpdateRequest(BaseModel):
    email: Optional[str] = Field(None, description="Updated email")
    display_name: Optional[str] = Field(None, description="Updated display name")
    first_name: Optional[str] = Field(None, description="Updated first name")
    last_name: Optional[str] = Field(None, description="Updated last name")
    enabled: Optional[bool] = Field(None, description="Updated enabled state")
    licensed: Optional[bool] = Field(None, description="Updated license state")
    metadata: Optional[dict[str, Any]] = Field(None, description="Updated metadata")


class UserResponse(BaseModel):
    id: str
    tenant_id: str
    email: str
    display_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    enabled: bool = True
    licensed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class UserListResponse(BaseModel):
    users: list[UserResponse]
    total_count: int


class ErrorResponse(BaseModel):
    detail: str


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def create_user(
    http_request: Request,
    request: UserCreateRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: UserService = Depends(_get_user_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> UserResponse:
    """Create a new platform user."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "users", "write")

        user = User(
            id=f"usr_{uuid4().hex[:16]}",
            tenant_id=ctx.tenant_id,
            email=request.email,
            display_name=request.display_name,
            first_name=request.first_name,
            last_name=request.last_name,
            enabled=request.enabled,
            licensed=request.licensed,
            metadata=request.metadata,
        )

        created = await service.create_user(user)

        try:
            await audit_service.record(AuditEvent(
                event_type="user",
                action="create",
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                resource_type="user",
                resource_id=created.id,
            ))
        except Exception:
            logger.debug("Audit record failed for user create")
        return UserResponse(**created.model_dump())

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to create user: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create user")


@router.get(
    "",
    response_model=UserListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_users(
    http_request: Request,
    enabled: Optional[bool] = Query(None, description="Filter by enabled state"),
    licensed: Optional[bool] = Query(None, description="Filter by licensed state"),
    search: Optional[str] = Query(None, description="Case-insensitive substring over email + display_name"),
    ids: Optional[str] = Query(None, description="Comma-separated user ids to restrict to"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: UserService = Depends(_get_user_service),
    logger: Logger = Depends(get_logger),
) -> UserListResponse:
    """List platform users."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "users", "read")

        id_list = [i.strip() for i in ids.split(",") if i.strip()] if ids is not None else None
        users, total = await service.list_users(
            tenant_id=ctx.tenant_id,
            limit=limit,
            offset=offset,
            enabled=enabled,
            licensed=licensed,
            search=search,
            ids=id_list,
        )
        return UserListResponse(
            users=[UserResponse(**u.model_dump()) for u in users],
            total_count=total,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list users: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list users")


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_user(
    http_request: Request,
    user_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: UserService = Depends(_get_user_service),
    logger: Logger = Depends(get_logger),
) -> UserResponse:
    """Get a platform user by ID."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "users", "read")

        user = await service.get_user(user_id, ctx.tenant_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User not found: {user_id}")

        return UserResponse(**user.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get user %s: %s", user_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get user")


@router.put(
    "/{user_id}",
    response_model=UserResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def update_user(
    http_request: Request,
    user_id: str,
    request: UserUpdateRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: UserService = Depends(_get_user_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> UserResponse:
    """Update a platform user."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "users", "write")

        updates = request.model_dump(exclude_none=True)
        user = await service.update_user(user_id, ctx.tenant_id, **updates)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User not found: {user_id}")

        try:
            await audit_service.record(AuditEvent(
                event_type="user",
                action="update",
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                resource_type="user",
                resource_id=user_id,
            ))
        except Exception:
            logger.debug("Audit record failed for user update")
        return UserResponse(**user.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update user %s: %s", user_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update user")


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def delete_user(
    http_request: Request,
    user_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: UserService = Depends(_get_user_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> None:
    """Delete a platform user."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "users", "delete")

        deleted = await service.delete_user(user_id, ctx.tenant_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User not found: {user_id}")

        try:
            await audit_service.record(AuditEvent(
                event_type="user",
                action="delete",
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                resource_type="user",
                resource_id=user_id,
            ))
        except Exception:
            logger.debug("Audit record failed for user delete")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete user %s: %s", user_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete user")


# ------------------------------------------------------------------ #
# Plugin registration
# ------------------------------------------------------------------ #

class UsersAPIPlugin(Plugin):
    """Plugin to register users API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True
