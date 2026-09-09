"""Dataset API endpoints.

Endpoints:
- POST   /v1/datasets                    -- Upload dataset (multipart/form-data)
- GET    /v1/datasets                    -- List datasets in workspace
- GET    /v1/datasets/{id}               -- Get dataset metadata + schema + profile
- GET    /v1/datasets/{id}/memories      -- Get memories extracted from dataset
- DELETE /v1/datasets/{id}               -- Delete dataset (+ optionally memories)
- POST   /v1/datasets/{id}/slice         -- Query a slice of data (DuckDB)

- GET    /v1/datasets/jobs               -- List dataset jobs in workspace
- GET    /v1/datasets/jobs/{id}          -- Get job status + progress
- POST   /v1/datasets/jobs/{id}/cancel   -- Cancel running job
"""
from datetime import datetime
from logging import Logger
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field
from scitrera_app_framework import Plugin, Variables, get_extension

from memorylayer_server.api import EXT_MULTI_API_ROUTERS
from memorylayer_server.api.v1.deps import get_auth_service, get_authz_service, get_audit_service
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep
from memorylayer_server.services.authentication import AuthenticationService, AuthenticationError
from memorylayer_server.services.authorization import AuthorizationService
from memorylayer_server.services.audit import AuditService, AuditEvent
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

from ...models.dataset import (
    DatasetColumn,
    DatasetProfilingOptions,
    DatasetSliceRequest,
    DatasetSliceResult,
    DatasetStatus,
    DatasetFormat,
)
from ...services.dataset import get_dataset_service
from ...services.dataset.dataset_service import DatasetService

router = APIRouter(prefix="/v1/datasets", tags=["datasets"])


# ------------------------------------------------------------------ #
# Response models
# ------------------------------------------------------------------ #

class DatasetResponse(BaseModel):
    """Dataset API response."""
    id: str
    workspace_id: str
    name: str
    filename: str
    format: str
    content_hash: str
    size_bytes: int
    status: str
    target_context_id: str = "_default"
    profiling_options: DatasetProfilingOptions = Field(default_factory=DatasetProfilingOptions)
    row_count: int = 0
    column_count: int = 0
    columns: list[DatasetColumn] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    profile_summary: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    profiling_started_at: Optional[datetime] = None
    profiling_completed_at: Optional[datetime] = None


class DatasetListResponse(BaseModel):
    """Paginated list of datasets."""
    datasets: list[DatasetResponse]
    total_count: int


class DatasetJobResponse(BaseModel):
    """Dataset job API response."""
    id: str
    workspace_id: str
    dataset_ids: list[str] = Field(default_factory=list)
    status: str
    progress_percent: int = 0
    datasets_processed: int = 0
    total_memories_created: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class DatasetJobListResponse(BaseModel):
    """List of dataset jobs."""
    jobs: list[DatasetJobResponse]


class UploadDatasetResponse(BaseModel):
    """Response for a successful dataset upload."""
    dataset: DatasetResponse
    job: DatasetJobResponse


class MemoryBriefResponse(BaseModel):
    """Abbreviated memory representation."""
    id: str
    content: str
    type: str
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    created_at: datetime


class DatasetMemoriesResponse(BaseModel):
    """List of memories extracted from a dataset."""
    dataset_id: str
    memories: list[MemoryBriefResponse]
    total_count: int


class SliceResponse(BaseModel):
    """Response for a dataset slice query."""
    dataset_id: str
    columns: list[str]
    dtypes: list[str] = Field(default_factory=list)
    rows: list[list[Any]]
    total_matching: int
    returned_count: int
    sql_executed: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str


# ------------------------------------------------------------------ #
# Dependencies
# ------------------------------------------------------------------ #

def get_dataset_service_dep(
    v: Variables = Depends(get_variables_dep),
) -> DatasetService:
    """FastAPI dependency wrapper for the dataset service."""
    return get_dataset_service(v)


# ------------------------------------------------------------------ #
# Dataset endpoints
# ------------------------------------------------------------------ #

@router.post(
    "",
    response_model=UploadDatasetResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def upload_dataset(
    http_request: Request,
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    target_context_id: str = Form("_default"),
    importance: float = Form(0.5),
    sample_rows: int = Form(1000),
    detect_time_series: bool = Form(True),
    generate_summaries: bool = Form(True),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DatasetService = Depends(get_dataset_service_dep),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> UploadDatasetResponse:
    """Upload a dataset for profiling and memory extraction.

    Accepts multipart/form-data with the dataset file and optional
    profiling parameters. Returns the created dataset record and
    the background profiling job.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "datasets", "write", workspace_id=ctx.workspace_id)

        file_data = await file.read()
        filename = file.filename or "unnamed.csv"

        profiling_options = DatasetProfilingOptions(
            target_context_id=target_context_id,
            importance=importance,
            sample_rows=sample_rows,
            detect_time_series=detect_time_series,
            generate_summaries=generate_summaries,
        )

        logger.info(
            "Dataset upload request: %s (%d bytes) for workspace %s",
            filename, len(file_data), ctx.workspace_id,
        )

        ds, job = await service.upload_dataset(
            workspace_id=ctx.workspace_id,
            file_data=file_data,
            filename=filename,
            name=name,
            profiling_options=profiling_options,
        )

        try:
            await audit_service.record(AuditEvent(
                event_type="dataset",
                action="upload",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="dataset",
                resource_id=ds.id,
            ))
        except Exception:
            logger.debug("Audit record failed for dataset upload")
        return UploadDatasetResponse(
            dataset=DatasetResponse(**ds.model_dump()),
            job=DatasetJobResponse(**job.model_dump()),
        )

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except ValueError as exc:
        logger.warning("Upload rejected: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Dataset upload failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload dataset",
        )


@router.get(
    "",
    response_model=DatasetListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_datasets(
    http_request: Request,
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DatasetService = Depends(get_dataset_service_dep),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> DatasetListResponse:
    """List datasets in the workspace."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "datasets", "read", workspace_id=ctx.workspace_id)

        datasets, total = await service.list_datasets(
            workspace_id=ctx.workspace_id,
            status=status_filter,
            limit=limit,
            offset=offset,
        )

        try:
            await audit_service.record(AuditEvent(
                event_type="dataset",
                action="list",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="dataset",
            ))
        except Exception:
            logger.debug("Audit record failed for dataset list")
        return DatasetListResponse(
            datasets=[DatasetResponse(**d.model_dump()) for d in datasets],
            total_count=total,
        )

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list datasets: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list datasets",
        )


@router.get(
    "/jobs",
    response_model=DatasetJobListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_dataset_jobs(
    http_request: Request,
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DatasetService = Depends(get_dataset_service_dep),
    logger: Logger = Depends(get_logger),
) -> DatasetJobListResponse:
    """List dataset processing jobs for the workspace."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "datasets", "read", workspace_id=ctx.workspace_id)

        jobs = await service.list_jobs(
            workspace_id=ctx.workspace_id,
            status=status_filter,
            limit=limit,
        )

        return DatasetJobListResponse(
            jobs=[DatasetJobResponse(**j.model_dump()) for j in jobs],
        )

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list dataset jobs: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list dataset jobs",
        )


@router.get(
    "/jobs/{job_id}",
    response_model=DatasetJobResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_dataset_job(
    http_request: Request,
    job_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DatasetService = Depends(get_dataset_service_dep),
    logger: Logger = Depends(get_logger),
) -> DatasetJobResponse:
    """Get dataset job status and progress."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "datasets", "read", workspace_id=ctx.workspace_id)

        job = await service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found: %s" % job_id)

        return DatasetJobResponse(**job.model_dump())

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get dataset job %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dataset job",
        )


@router.post(
    "/jobs/{job_id}/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def cancel_dataset_job(
    http_request: Request,
    job_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DatasetService = Depends(get_dataset_service_dep),
    logger: Logger = Depends(get_logger),
) -> None:
    """Cancel a queued or running dataset job."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "datasets", "write", workspace_id=ctx.workspace_id)

        logger.info("Cancel request for dataset job: %s", job_id)
        await service.cancel_job(job_id)

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except ValueError as exc:
        logger.warning("Cancel rejected for dataset job %s: %s", job_id, exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to cancel dataset job %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel dataset job",
        )


@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_dataset(
    http_request: Request,
    dataset_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DatasetService = Depends(get_dataset_service_dep),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> DatasetResponse:
    """Get dataset metadata, schema, and profile."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "datasets", "read", workspace_id=ctx.workspace_id)

        ds = await service.get_dataset(dataset_id, ctx.workspace_id)
        if not ds:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found: %s" % dataset_id,
            )

        try:
            await audit_service.record(AuditEvent(
                event_type="dataset",
                action="read",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="dataset",
                resource_id=dataset_id,
            ))
        except Exception:
            logger.debug("Audit record failed for dataset read")
        return DatasetResponse(**ds.model_dump())

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get dataset %s: %s", dataset_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dataset",
        )


@router.get(
    "/{dataset_id}/memories",
    response_model=DatasetMemoriesResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_dataset_memories(
    http_request: Request,
    dataset_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DatasetService = Depends(get_dataset_service_dep),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> DatasetMemoriesResponse:
    """Get memories extracted from a dataset."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "datasets", "read", workspace_id=ctx.workspace_id)

        ds = await service.get_dataset(dataset_id, ctx.workspace_id)
        if not ds:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found: %s" % dataset_id,
            )

        storage_backend = get_extension(EXT_STORAGE_BACKEND, v)
        memories: list[MemoryBriefResponse] = []

        for mem_id in ds.memory_ids:
            memory = await storage_backend.get_memory(
                ds.workspace_id, mem_id, track_access=False,
            )
            if memory:
                memories.append(MemoryBriefResponse(
                    id=memory.id,
                    content=memory.content,
                    type=memory.type.value if hasattr(memory.type, "value") else str(memory.type),
                    importance=memory.importance,
                    tags=memory.tags,
                    created_at=memory.created_at,
                ))

        return DatasetMemoriesResponse(
            dataset_id=dataset_id,
            memories=memories,
            total_count=len(memories),
        )

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get memories for dataset %s: %s", dataset_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dataset memories",
        )


@router.post(
    "/{dataset_id}/slice",
    response_model=SliceResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def query_dataset_slice(
    http_request: Request,
    dataset_id: str,
    request: DatasetSliceRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DatasetService = Depends(get_dataset_service_dep),
    logger: Logger = Depends(get_logger),
) -> SliceResponse:
    """Query a slice of the dataset using DuckDB.

    Supports both structured filters and raw SQL (SELECT only).
    The dataset is queried as a table named 'data'.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "datasets", "read", workspace_id=ctx.workspace_id)

        logger.info(
            "Dataset slice query: dataset=%s, workspace=%s",
            dataset_id, ctx.workspace_id,
        )

        result = await service.query_slice(
            dataset_id=dataset_id,
            workspace_id=ctx.workspace_id,
            request=request,
        )

        return SliceResponse(**result.model_dump())

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except ValueError as exc:
        logger.warning("Slice query rejected: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Dataset slice query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query dataset",
        )


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def delete_dataset(
    http_request: Request,
    dataset_id: str,
    delete_memories: bool = Query(False, description="Also delete extracted memories"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: DatasetService = Depends(get_dataset_service_dep),
    audit_service: AuditService = Depends(get_audit_service),
    logger: Logger = Depends(get_logger),
) -> None:
    """Delete a dataset and optionally its extracted memories."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "datasets", "delete", workspace_id=ctx.workspace_id)

        logger.info(
            "Delete request for dataset %s (delete_memories=%s)",
            dataset_id, delete_memories,
        )
        await service.delete_dataset(
            dataset_id, workspace_id=ctx.workspace_id, delete_memories=delete_memories,
        )
        try:
            await audit_service.record(AuditEvent(
                event_type="dataset",
                action="delete",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="dataset",
                resource_id=dataset_id,
            ))
        except Exception:
            logger.debug("Audit record failed for dataset delete")

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except ValueError as exc:
        logger.warning("Delete rejected for dataset %s: %s", dataset_id, exc)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete dataset %s: %s", dataset_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete dataset",
        )


# ------------------------------------------------------------------ #
# Plugin registration
# ------------------------------------------------------------------ #

class DatasetsAPIPlugin(Plugin):
    """Plugin to register dataset API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True
