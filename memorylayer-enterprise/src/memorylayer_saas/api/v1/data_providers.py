"""Data providers API endpoints.

Endpoints:
- POST   /v1/data-providers          - Create data provider
- GET    /v1/data-providers          - List data providers
- GET    /v1/data-providers/{id}     - Get data provider
- PUT    /v1/data-providers/{id}     - Update data provider
- DELETE /v1/data-providers/{id}     - Delete data provider
"""
from datetime import datetime
from logging import Logger
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from scitrera_app_framework import Plugin, Variables

from memorylayer_server.api import EXT_MULTI_API_ROUTERS
from memorylayer_server.api.v1.deps import get_auth_service, get_authz_service, get_audit_service
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep
from memorylayer_server.services.authentication import AuthenticationService
from memorylayer_server.services.authorization import AuthorizationService
from memorylayer_server.services.audit import AuditService, AuditEvent

from ...services.data_provider import DataProviderService

router = APIRouter(prefix="/v1/data-providers", tags=["data-providers"])

_dp_service: Optional[DataProviderService] = None


def _get_dp_service(v: Variables = Depends(get_variables_dep)) -> DataProviderService:
    global _dp_service
    if _dp_service is None:
        from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
        from scitrera_app_framework import get_extension
        storage = get_extension(EXT_STORAGE_BACKEND, v)
        _dp_service = DataProviderService(session_factory=storage.session_factory)
    return _dp_service


# ------------------------------------------------------------------ #
# Request/Response models
# ------------------------------------------------------------------ #

class DataProviderCreateRequest(BaseModel):
    name: str = Field(..., description="Provider name")
    provider_type: str = Field(..., description="Provider type (s3, gcs, azure_blob, sharepoint, confluence, web)")
    description: Optional[str] = Field(None, description="Description")
    enabled: bool = Field(True, description="Enabled state")
    connection_args: dict[str, Any] = Field(default_factory=dict, description="Connection arguments")
    encrypted_args: Optional[dict[str, Any]] = Field(None, description="Encrypted connection arguments (secrets)")
    schedule: Optional[str] = Field(None, description="Cron schedule for auto-sync")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class DataProviderUpdateRequest(BaseModel):
    name: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    enabled: Optional[bool] = Field(None)
    connection_args: Optional[dict[str, Any]] = Field(None)
    encrypted_args: Optional[dict[str, Any]] = Field(None)
    schedule: Optional[str] = Field(None)
    metadata: Optional[dict[str, Any]] = Field(None)


class DataProviderResponse(BaseModel):
    """Response model - never includes encrypted_args."""
    id: str
    tenant_id: str
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


class DataProviderListResponse(BaseModel):
    providers: list[DataProviderResponse]
    total_count: int


class ErrorResponse(BaseModel):
    detail: str


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@router.post(
    "",
    response_model=DataProviderResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def create_data_provider(
    http_request: Request,
    request: DataProviderCreateRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DataProviderService = Depends(_get_dp_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> DataProviderResponse:
    """Create a new data provider."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "data_providers", "write", workspace_id=ctx.workspace_id)

        from uuid import uuid4
        from ...models.data_provider import DataProvider

        provider = DataProvider(
            id=f"dp_{uuid4().hex[:16]}",
            tenant_id=ctx.tenant_id,
            workspace_id=ctx.workspace_id,
            name=request.name,
            provider_type=request.provider_type,
            description=request.description,
            enabled=request.enabled,
            connection_args=request.connection_args,
            encrypted_args=request.encrypted_args,
            schedule=request.schedule,
            metadata=request.metadata,
        )
        created = await service.create_provider(provider)

        try:
            await audit_service.record(AuditEvent(
                event_type="data_provider",
                action="create",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="data_provider",
                resource_id=created.id,
            ))
        except Exception:
            logger.debug("Audit record failed for data provider create")
        return _to_response(created)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to create data provider: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create data provider")


@router.get(
    "",
    response_model=DataProviderListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_data_providers(
    http_request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DataProviderService = Depends(_get_dp_service),
    logger: Logger = Depends(get_logger),
) -> DataProviderListResponse:
    """List data providers in workspace."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "data_providers", "read", workspace_id=ctx.workspace_id)

        providers, total = await service.list_providers(
            workspace_id=ctx.workspace_id,
            limit=limit,
            offset=offset,
        )
        return DataProviderListResponse(
            providers=[_to_response(p) for p in providers],
            total_count=total,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list data providers: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list data providers")


@router.get(
    "/by-name/{name}",
    response_model=DataProviderResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_data_provider_by_name(
    http_request: Request,
    name: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DataProviderService = Depends(_get_dp_service),
    logger: Logger = Depends(get_logger),
) -> DataProviderResponse:
    """Get a data provider by name within the current workspace."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "data_providers", "read", workspace_id=ctx.workspace_id)

        provider = await service.get_provider_by_name(name, ctx.workspace_id)
        if not provider:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Data provider not found: {name}")
        return _to_response(provider)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get data provider by name %s: %s", name, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get data provider")


@router.get(
    "/{provider_id}",
    response_model=DataProviderResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_data_provider(
    http_request: Request,
    provider_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DataProviderService = Depends(_get_dp_service),
    logger: Logger = Depends(get_logger),
) -> DataProviderResponse:
    """Get a data provider by ID."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "data_providers", "read", workspace_id=ctx.workspace_id)

        provider = await service.get_provider(provider_id, ctx.workspace_id)
        if not provider:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Data provider not found: {provider_id}")
        return _to_response(provider)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get data provider %s: %s", provider_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get data provider")


@router.put(
    "/{provider_id}",
    response_model=DataProviderResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def update_data_provider(
    http_request: Request,
    provider_id: str,
    request: DataProviderUpdateRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DataProviderService = Depends(_get_dp_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> DataProviderResponse:
    """Update a data provider."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "data_providers", "write", workspace_id=ctx.workspace_id)

        updates = request.model_dump(exclude_none=True)
        provider = await service.update_provider(provider_id, ctx.workspace_id, **updates)
        if not provider:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Data provider not found: {provider_id}")

        try:
            await audit_service.record(AuditEvent(
                event_type="data_provider",
                action="update",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="data_provider",
                resource_id=provider_id,
            ))
        except Exception:
            logger.debug("Audit record failed for data provider update")
        return _to_response(provider)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update data provider %s: %s", provider_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update data provider")


@router.delete(
    "/{provider_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def delete_data_provider(
    http_request: Request,
    provider_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DataProviderService = Depends(_get_dp_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> None:
    """Delete a data provider."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "data_providers", "delete", workspace_id=ctx.workspace_id)

        deleted = await service.delete_provider(provider_id, ctx.workspace_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Data provider not found: {provider_id}")

        try:
            await audit_service.record(AuditEvent(
                event_type="data_provider",
                action="delete",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="data_provider",
                resource_id=provider_id,
            ))
        except Exception:
            logger.debug("Audit record failed for data provider delete")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete data provider %s: %s", provider_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete data provider")


def _to_response(provider) -> DataProviderResponse:
    """Convert domain model to response, excluding encrypted_args."""
    data = provider.model_dump()
    data.pop("encrypted_args", None)
    return DataProviderResponse(**data)


# ------------------------------------------------------------------ #
# Plugin registration
# ------------------------------------------------------------------ #

class DataProvidersAPIPlugin(Plugin):
    """Plugin to register data providers API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True
