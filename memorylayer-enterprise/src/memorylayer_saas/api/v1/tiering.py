"""
Cold storage tiering API endpoints.

Endpoints:
- GET /v1/tiering/stats - Get cold storage statistics
- POST /v1/tiering/archive - Manual archive memories to cold tier
- POST /v1/tiering/restore - Restore memories from cold tier
- GET /v1/tiering/config - Get tiering configuration
- PUT /v1/tiering/config - Update tiering configuration
"""
from logging import Logger
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request, status
from scitrera_app_framework import Variables, Plugin, get_extension

from memorylayer_server.api import EXT_MULTI_API_ROUTERS
from memorylayer_server.api.v1.deps import get_auth_service, get_authz_service, get_audit_service
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep
from memorylayer_server.services.authentication import AuthenticationService, AuthenticationError
from memorylayer_server.services.authorization import AuthorizationService
from memorylayer_server.services.audit import AuditService, AuditEvent
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND, StorageBackend

from ...services.tiering import TieringService, TieringStats, ArchivalResult, RestoreResult
from ...models.tiering import (
    TieringStatsResponse,
    ArchiveRequest,
    ArchiveResponse,
    RestoreRequest,
    RestoreResponse,
    TieringConfigResponse,
    TieringConfigUpdateRequest,
    ErrorResponse,
)

router = APIRouter(prefix="/v1/tiering", tags=["tiering"])

# Key under workspace.settings (JSONB) where per-workspace tiering overrides live.
# This is the same location the enterprise recall path reads
# (EnterpriseMemoryService.recall -> workspace.settings["tiering"]), so GET/PUT
# here and the live cold-tier behaviour stay consistent.
TIERING_SETTINGS_KEY = "tiering"


# Dependency to get tiering service
def get_tiering_service_dep(v: Variables = Depends(get_variables_dep)) -> TieringService:
    """
    Get tiering service instance from dependency injection.

    Uses the plugin system to retrieve the configured tiering service.
    """
    from ...services.tiering import get_tiering_service
    return get_tiering_service(v)


@router.get(
    "/stats",
    response_model=TieringStatsResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def get_tiering_stats(
        http_request: Request,
        include_candidates: bool = True,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        tiering_service: TieringService = Depends(get_tiering_service_dep),
        audit_service: AuditService = Depends(get_audit_service),
        logger: Logger = Depends(get_logger),
) -> TieringStatsResponse:
    """
    Get cold storage tiering statistics.

    Returns comprehensive statistics about hot and cold tier distribution,
    storage usage, compression ratio, and archival candidates.

    Args:
        include_candidates: If True, count archival candidates (may be slower).
        workspace_id: Workspace ID from auth context.
        tiering_service: Tiering service instance.

    Returns:
        TieringStatsResponse with storage metrics.

    Raises:
        HTTPException: If stats retrieval fails.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        logger.info(
            "Getting tiering stats for workspace: %s, include_candidates: %s",
            ctx.workspace_id,
            include_candidates
        )

        stats: TieringStats = await tiering_service.get_tiering_stats(
            workspace_id=ctx.workspace_id,
            include_candidates=include_candidates,
        )

        logger.info(
            "Retrieved tiering stats: hot=%d, cold=%d, compression=%.2f",
            stats.hot_memory_count,
            stats.cold_memory_count,
            stats.compression_ratio
        )

        response = TieringStatsResponse(
            hot_memory_count=stats.hot_memory_count,
            cold_memory_count=stats.cold_memory_count,
            hot_storage_bytes=stats.hot_storage_bytes,
            cold_storage_bytes=stats.cold_storage_bytes,
            compression_ratio=stats.compression_ratio,
            estimated_savings_bytes=stats.estimated_savings_bytes,
            archival_candidates_count=stats.archival_candidates_count,
        )
        try:
            await audit_service.record(AuditEvent(
                event_type="tiering",
                action="read",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="tier",
            ))
        except Exception:
            logger.debug("Audit record failed for tiering stats read")
        return response

    except AuthenticationError as e:
        logger.warning("Authentication failed: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get tiering stats: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tiering statistics"
        )


@router.post(
    "/archive",
    response_model=ArchiveResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def archive_memories(
        http_request: Request,
        request: ArchiveRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        tiering_service: TieringService = Depends(get_tiering_service_dep),
        audit_service: AuditService = Depends(get_audit_service),
        logger: Logger = Depends(get_logger),
) -> ArchiveResponse:
    """
    Archive memories from hot tier to cold tier.

    Can either archive specific memories by ID or automatically detect
    candidates based on importance, access count, and age thresholds.

    Args:
        request: Archive request with memory IDs or auto-detect settings.
        workspace_id: Workspace ID from auth context.
        tiering_service: Tiering service instance.

    Returns:
        ArchiveResponse with counts and IDs of archived/failed memories.

    Raises:
        HTTPException: If archive operation fails.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "write", workspace_id=ctx.workspace_id)

        # Validate request
        if not request.memory_ids and not request.auto_detect:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either memory_ids or auto_detect=True must be provided"
            )

        logger.info(
            "Archiving memories in workspace: %s, auto_detect: %s, count: %s",
            ctx.workspace_id,
            request.auto_detect,
            len(request.memory_ids) if request.memory_ids else "auto"
        )

        result: ArchivalResult = await tiering_service.archive_memories(
            workspace_id=ctx.workspace_id,
            memory_ids=request.memory_ids,
            auto_detect=request.auto_detect,
            max_importance=request.max_importance,
            max_access_count=request.max_access_count,
            older_than_days=request.older_than_days,
            batch_size=request.batch_size,
        )

        logger.info(
            "Archived %d memories (%d failed) in workspace: %s",
            result.archived_count,
            result.failed_count,
            ctx.workspace_id
        )

        response = ArchiveResponse(
            archived_count=result.archived_count,
            failed_count=result.failed_count,
            archived_memory_ids=result.archived_memory_ids,
            failed_memory_ids=result.failed_memory_ids,
        )
        try:
            await audit_service.record(AuditEvent(
                event_type="tiering",
                action="update",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="tier",
                metadata={"archived_count": result.archived_count},
            ))
        except Exception:
            logger.debug("Audit record failed for tiering archive")
        return response

    except AuthenticationError as e:
        logger.warning("Authentication failed: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Invalid archive request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to archive memories: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to archive memories"
        )


@router.post(
    "/restore",
    response_model=RestoreResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def restore_memories(
        http_request: Request,
        request: RestoreRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        tiering_service: TieringService = Depends(get_tiering_service_dep),
        audit_service: AuditService = Depends(get_audit_service),
        logger: Logger = Depends(get_logger),
) -> RestoreResponse:
    """
    Restore memories from cold tier back to hot tier.

    Restores specified memories and optionally regenerates their embeddings
    for full hot tier functionality.

    Args:
        request: Restore request with memory IDs.
        workspace_id: Workspace ID from auth context.
        tiering_service: Tiering service instance.

    Returns:
        RestoreResponse with counts and IDs of restored/failed memories.

    Raises:
        HTTPException: If restore operation fails.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "write", workspace_id=ctx.workspace_id)

        logger.info(
            "Restoring %d memories in workspace: %s",
            len(request.memory_ids),
            ctx.workspace_id
        )

        result: RestoreResult = await tiering_service.restore_memories(
            workspace_id=ctx.workspace_id,
            memory_ids=request.memory_ids,
            regenerate_embeddings=request.regenerate_embeddings,
        )

        logger.info(
            "Restored %d memories (%d failed) in workspace: %s",
            result.restored_count,
            result.failed_count,
            ctx.workspace_id
        )

        response = RestoreResponse(
            restored_count=result.restored_count,
            failed_count=result.failed_count,
            restored_memory_ids=result.restored_memory_ids,
            failed_memory_ids=result.failed_memory_ids,
        )
        try:
            await audit_service.record(AuditEvent(
                event_type="tiering",
                action="update",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="tier",
                metadata={"restored_count": result.restored_count},
            ))
        except Exception:
            logger.debug("Audit record failed for tiering restore")
        return response

    except AuthenticationError as e:
        logger.warning("Authentication failed: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Invalid restore request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to restore memories: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to restore memories"
        )


def _build_tiering_config_response(
        tiering_service: TieringService,
        stored: Optional[dict] = None,
) -> TieringConfigResponse:
    """Build a TieringConfigResponse from defaults overlaid with stored per-workspace overrides.

    The service instance supplies process-global defaults (env-var backed). Any
    per-workspace overrides previously persisted to ``workspace.settings["tiering"]``
    take precedence, so the response reflects the effective configuration.

    Args:
        tiering_service: Service instance providing default thresholds.
        stored: Optional dict of persisted overrides from workspace.settings["tiering"].
    """
    stored = stored or {}
    return TieringConfigResponse(
        # Defaults match EnterpriseMemoryService.recall's absent-key defaults
        # (workspace.settings["tiering"].get("cold_tier_enabled", False) and
        # .get("cold_tier_search_enabled", False)) so GET truthfully reports
        # the effective cold-tier state before any override is persisted via PUT.
        cold_tier_enabled=stored.get("cold_tier_enabled", False),
        archival_age_days=stored.get("archival_age_days", tiering_service.DEFAULT_OLDER_THAN_DAYS),
        min_importance_threshold=stored.get("min_importance_threshold", tiering_service.DEFAULT_MAX_IMPORTANCE),
        min_access_count_threshold=stored.get("min_access_count_threshold", tiering_service.DEFAULT_MAX_ACCESS_COUNT),
        archival_batch_size=stored.get("archival_batch_size", 100),
        warmup_access_threshold=stored.get("warmup_access_threshold", tiering_service.DEFAULT_WARMUP_ACCESS_THRESHOLD),
        warmup_batch_size=stored.get("warmup_batch_size", 100),
        cold_retrieval_latency_target_ms=stored.get("cold_retrieval_latency_target_ms", 500),
        cold_tier_search_enabled=stored.get("cold_tier_search_enabled", False),
    )


async def _read_tiering_overrides(storage: StorageBackend, workspace_id: str) -> dict:
    """Read persisted tiering overrides from workspace.settings["tiering"].

    Returns an empty dict when the workspace or the tiering settings are absent.
    """
    workspace = await storage.get_workspace(workspace_id)
    if not workspace or not workspace.settings:
        return {}
    overrides = workspace.settings.get(TIERING_SETTINGS_KEY)
    return dict(overrides) if isinstance(overrides, dict) else {}


@router.get(
    "/config",
    response_model=TieringConfigResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_tiering_config_endpoint(
        http_request: Request,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        tiering_service: TieringService = Depends(get_tiering_service_dep),
        audit_service: AuditService = Depends(get_audit_service),
        v: Variables = Depends(get_variables_dep),
        logger: Logger = Depends(get_logger),
) -> TieringConfigResponse:
    """
    Get current tiering configuration.

    Returns the effective tiering configuration for the workspace: process-global
    defaults overlaid with any per-workspace overrides persisted via PUT.

    Args:
        workspace_id: Workspace ID from auth context.

    Returns:
        TieringConfigResponse with current configuration.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        logger.debug("Getting tiering config for workspace: %s", ctx.workspace_id)

        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        stored = await _read_tiering_overrides(storage, ctx.workspace_id)
        response = _build_tiering_config_response(tiering_service, stored)
        try:
            await audit_service.record(AuditEvent(
                event_type="tiering",
                action="read",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="tier",
            ))
        except Exception:
            logger.debug("Audit record failed for tiering config read")
        return response

    except AuthenticationError as e:
        logger.warning("Authentication failed: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get tiering config: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tiering configuration"
        )


@router.put(
    "/config",
    response_model=TieringConfigResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def update_tiering_config(
        http_request: Request,
        request: TieringConfigUpdateRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        tiering_service: TieringService = Depends(get_tiering_service_dep),
        audit_service: AuditService = Depends(get_audit_service),
        v: Variables = Depends(get_variables_dep),
        logger: Logger = Depends(get_logger),
) -> TieringConfigResponse:
    """
    Update tiering configuration.

    Persists workspace-specific tiering overrides to ``workspace.settings["tiering"]``
    (the same location the enterprise recall path reads). Only provided (non-null)
    fields are written; others retain their current persisted/default values. The
    response is re-read from storage so it reflects the persisted state
    (read-after-write consistency with GET).

    Args:
        request: Configuration update request.
        workspace_id: Workspace ID from auth context.

    Returns:
        TieringConfigResponse with updated configuration.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "write", workspace_id=ctx.workspace_id)

        logger.info("Updating tiering config for workspace: %s", ctx.workspace_id)

        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)

        # Only persist explicitly provided (non-null) fields.
        update_fields = request.model_dump(exclude_none=True)

        workspace = await storage.get_workspace(ctx.workspace_id)
        if workspace is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace not found: {ctx.workspace_id}",
            )

        # Build a NEW settings dict (avoids in-place JSONB mutation) and merge the
        # provided overrides into the "tiering" sub-dict.
        settings = dict(workspace.settings or {})
        existing = settings.get(TIERING_SETTINGS_KEY)
        existing = dict(existing) if isinstance(existing, dict) else {}
        merged = {**existing, **update_fields}

        changed = merged != existing
        if changed:
            settings[TIERING_SETTINGS_KEY] = merged
            await storage.update_workspace(ctx.workspace_id, settings=settings)
            logger.info(
                "Persisted tiering config fields for workspace %s: %s",
                ctx.workspace_id,
                list(update_fields.keys()),
            )
        else:
            logger.debug(
                "No tiering config changes to persist for workspace %s",
                ctx.workspace_id,
            )

        # Re-read from storage so the response reflects the persisted state.
        stored = await _read_tiering_overrides(storage, ctx.workspace_id)
        response = _build_tiering_config_response(tiering_service, stored)

        # Only record an audit "update" event when something actually changed.
        if changed:
            try:
                await audit_service.record(AuditEvent(
                    event_type="tiering",
                    action="update",
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                    resource_type="tier",
                    metadata={"updated_fields": list(update_fields.keys())},
                ))
            except Exception:
                logger.debug("Audit record failed for tiering config update")
        return response

    except AuthenticationError as e:
        logger.warning("Authentication failed: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update tiering config: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update tiering configuration"
        )


@router.post(
    "/warmup",
    response_model=RestoreResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def run_warmup_cycle(
        http_request: Request,
        access_threshold: Optional[int] = None,
        limit: int = 100,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        tiering_service: TieringService = Depends(get_tiering_service_dep),
        audit_service: AuditService = Depends(get_audit_service),
        logger: Logger = Depends(get_logger),
) -> RestoreResponse:
    """
    Run a warm-up cycle to promote frequently accessed cold memories.

    Identifies cold tier memories that have been accessed frequently
    and promotes them back to hot tier for faster access.

    Args:
        access_threshold: Minimum cold access count for promotion.
        limit: Maximum number of memories to promote.
        workspace_id: Workspace ID from auth context.
        tiering_service: Tiering service instance.

    Returns:
        RestoreResponse with counts and IDs of promoted/failed memories.

    Raises:
        HTTPException: If warm-up operation fails.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "write", workspace_id=ctx.workspace_id)

        logger.info(
            "Running warm-up cycle in workspace: %s, threshold: %s, limit: %d",
            ctx.workspace_id,
            access_threshold,
            limit
        )

        result: RestoreResult = await tiering_service.run_warmup_cycle(
            workspace_id=ctx.workspace_id,
            access_threshold=access_threshold,
            limit=limit,
        )

        logger.info(
            "Warm-up promoted %d memories (%d failed) in workspace: %s",
            result.restored_count,
            result.failed_count,
            ctx.workspace_id
        )

        response = RestoreResponse(
            restored_count=result.restored_count,
            failed_count=result.failed_count,
            restored_memory_ids=result.restored_memory_ids,
            failed_memory_ids=result.failed_memory_ids,
        )
        try:
            await audit_service.record(AuditEvent(
                event_type="tiering",
                action="update",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="tier",
                metadata={"promoted_count": result.restored_count},
            ))
        except Exception:
            logger.debug("Audit record failed for tiering warmup")
        return response

    except AuthenticationError as e:
        logger.warning("Authentication failed: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to run warm-up cycle: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run warm-up cycle"
        )


class TieringAPIPlugin(Plugin):
    """Plugin to register tiering API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def is_multi_extension(self, v: Variables) -> bool:
        return True
