"""Vector collections API endpoints.

Endpoints:
- POST   /v1/collections          - Create collection item
- GET    /v1/collections          - List collection items
- GET    /v1/collections/{id}     - Get collection item
- PUT    /v1/collections/{id}     - Update collection item
- DELETE /v1/collections/{id}     - Delete collection item
- POST   /v1/collections/search   - Search by vector similarity
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

from ...services.collection import CollectionService

router = APIRouter(prefix="/v1/collections", tags=["collections"])

_collection_service: Optional[CollectionService] = None


def _get_collection_service(v: Variables = Depends(get_variables_dep)) -> CollectionService:
    global _collection_service
    if _collection_service is None:
        from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
        from scitrera_app_framework import get_extension
        storage = get_extension(EXT_STORAGE_BACKEND, v)
        _collection_service = CollectionService(session_factory=storage.session_factory)
    return _collection_service


# ------------------------------------------------------------------ #
# Request/Response models
# ------------------------------------------------------------------ #

class CollectionItemCreateRequest(BaseModel):
    collection_name: str = Field(..., description="Collection name")
    name: str = Field(..., description="Item name")
    content: str = Field(..., description="Item content")
    item_type: Optional[str] = Field(None, description="Item type")
    tags: list[str] = Field(default_factory=list, description="Tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    enabled: bool = Field(True, description="Enabled state")


class CollectionItemUpdateRequest(BaseModel):
    name: Optional[str] = Field(None)
    content: Optional[str] = Field(None)
    item_type: Optional[str] = Field(None)
    tags: Optional[list[str]] = Field(None)
    metadata: Optional[dict[str, Any]] = Field(None)
    enabled: Optional[bool] = Field(None)


class CollectionItemResponse(BaseModel):
    id: str
    tenant_id: str
    workspace_id: str
    collection_name: str
    name: str
    content: str
    item_type: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    created_at: datetime
    updated_at: datetime


class CollectionItemListResponse(BaseModel):
    items: list[CollectionItemResponse]
    total_count: int


class CollectionSearchRequest(BaseModel):
    query_embedding: list[float] = Field(..., description="Query embedding vector")
    collection_name: Optional[str] = Field(None, description="Filter by collection")
    limit: int = Field(10, ge=1, le=100, description="Max results")


class CollectionSearchResult(BaseModel):
    item: CollectionItemResponse
    similarity: float


class CollectionSearchResponse(BaseModel):
    results: list[CollectionSearchResult]
    total_count: int


class ErrorResponse(BaseModel):
    detail: str


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@router.post(
    "",
    response_model=CollectionItemResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def create_collection_item(
    http_request: Request,
    request: CollectionItemCreateRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: CollectionService = Depends(_get_collection_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> CollectionItemResponse:
    """Create a new collection item."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "collections", "write", workspace_id=ctx.workspace_id)

        from uuid import uuid4
        from ...models.collection import CollectionItem

        item = CollectionItem(
            id=f"col_{uuid4().hex[:16]}",
            tenant_id=ctx.tenant_id,
            workspace_id=ctx.workspace_id,
            collection_name=request.collection_name,
            name=request.name,
            content=request.content,
            item_type=request.item_type,
            tags=request.tags,
            metadata=request.metadata,
            enabled=request.enabled,
        )
        created = await service.create_item(item)

        try:
            await audit_service.record(AuditEvent(
                event_type="collection",
                action="create",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="collection_item",
                resource_id=created.id,
            ))
        except Exception:
            logger.debug("Audit record failed for collection item create")
        return CollectionItemResponse(**created.model_dump())

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to create collection item: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create collection item")


@router.get(
    "",
    response_model=CollectionItemListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_collection_items(
    http_request: Request,
    collection_name: Optional[str] = Query(None, description="Filter by collection"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: CollectionService = Depends(_get_collection_service),
    logger: Logger = Depends(get_logger),
) -> CollectionItemListResponse:
    """List collection items."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "collections", "read", workspace_id=ctx.workspace_id)

        items, total = await service.list_items(
            workspace_id=ctx.workspace_id,
            collection_name=collection_name,
            limit=limit,
            offset=offset,
        )
        return CollectionItemListResponse(
            items=[CollectionItemResponse(**i.model_dump()) for i in items],
            total_count=total,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list collection items: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list collection items")


@router.get(
    "/{item_id}",
    response_model=CollectionItemResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_collection_item(
    http_request: Request,
    item_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: CollectionService = Depends(_get_collection_service),
    logger: Logger = Depends(get_logger),
) -> CollectionItemResponse:
    """Get a collection item by ID."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "collections", "read", workspace_id=ctx.workspace_id)

        item = await service.get_item(item_id, ctx.workspace_id)
        if not item:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Collection item not found: {item_id}")
        return CollectionItemResponse(**item.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get collection item %s: %s", item_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get collection item")


@router.put(
    "/{item_id}",
    response_model=CollectionItemResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def update_collection_item(
    http_request: Request,
    item_id: str,
    request: CollectionItemUpdateRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: CollectionService = Depends(_get_collection_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> CollectionItemResponse:
    """Update a collection item."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "collections", "write", workspace_id=ctx.workspace_id)

        updates = request.model_dump(exclude_none=True)
        item = await service.update_item(item_id, ctx.workspace_id, **updates)
        if not item:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Collection item not found: {item_id}")

        try:
            await audit_service.record(AuditEvent(
                event_type="collection",
                action="update",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="collection_item",
                resource_id=item_id,
            ))
        except Exception:
            logger.debug("Audit record failed for collection item update")
        return CollectionItemResponse(**item.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update collection item %s: %s", item_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update collection item")


@router.delete(
    "/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def delete_collection_item(
    http_request: Request,
    item_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: CollectionService = Depends(_get_collection_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> None:
    """Delete a collection item."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "collections", "delete", workspace_id=ctx.workspace_id)

        deleted = await service.delete_item(item_id, ctx.workspace_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Collection item not found: {item_id}")

        try:
            await audit_service.record(AuditEvent(
                event_type="collection",
                action="delete",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="collection_item",
                resource_id=item_id,
            ))
        except Exception:
            logger.debug("Audit record failed for collection item delete")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete collection item %s: %s", item_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete collection item")


@router.post(
    "/search",
    response_model=CollectionSearchResponse,
    responses={500: {"model": ErrorResponse}},
)
async def search_collections(
    http_request: Request,
    request: CollectionSearchRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: CollectionService = Depends(_get_collection_service),
    logger: Logger = Depends(get_logger),
) -> CollectionSearchResponse:
    """Search collection items by vector similarity."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "collections", "read", workspace_id=ctx.workspace_id)

        results = await service.search_similar(
            workspace_id=ctx.workspace_id,
            query_embedding=request.query_embedding,
            collection_name=request.collection_name,
            limit=request.limit,
        )
        return CollectionSearchResponse(
            results=[
                CollectionSearchResult(
                    item=CollectionItemResponse(**item.model_dump()),
                    similarity=similarity,
                )
                for item, similarity in results
            ],
            total_count=len(results),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to search collections: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to search collections")


# ------------------------------------------------------------------ #
# Plugin registration
# ------------------------------------------------------------------ #

class CollectionsAPIPlugin(Plugin):
    """Plugin to register collections API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True
