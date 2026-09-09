# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Admin cross-workspace overview endpoints.

Endpoints:
- GET /v1/admin/stats       -- Aggregate counts across all workspaces
- GET /v1/admin/memories     -- List memories across all workspaces
- GET /v1/admin/sessions     -- List sessions across all workspaces
- GET /v1/admin/documents    -- List documents across all workspaces
- GET /v1/admin/datasets     -- List datasets across all workspaces
- GET /v1/admin/jobs         -- List jobs across all workspaces
"""
from logging import Logger
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from scitrera_app_framework import Plugin, Variables, get_extension

from memorylayer_server.api import EXT_MULTI_API_ROUTERS
from memorylayer_server.api.v1.deps import (
    get_auth_service,
    get_authz_service,
    get_audit_service,
    get_task_service,
)
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep
from memorylayer_server.services.authentication import AuthenticationService, AuthenticationError
from memorylayer_server.services.authorization import AuthorizationService
from memorylayer_server.services.audit import AuditService, AuditEvent
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
from memorylayer_server.services.tasks import TaskService

from ...models.tiering import (
    TieringStatsResponse,
    AdminTieringRunRequest,
    AdminTieringRunResponse,
)

router = APIRouter(prefix="/v1/admin", tags=["admin"])


# ------------------------------------------------------------------ #
# Response models
# ------------------------------------------------------------------ #

class AdminStatsResponse(BaseModel):
    workspace_count: int = 0
    memory_count: int = 0
    session_count: int = 0
    document_count: int = 0
    dataset_count: int = 0
    token_count: int = 0


class PaginatedResponse(BaseModel):
    items: list = Field(default_factory=list)
    total: int = 0
    limit: int = 100
    offset: int = 0


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #

def _get_storage(v: Variables):
    return get_extension(EXT_STORAGE_BACKEND, v)


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    http_request: Request,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    audit_service: AuditService = Depends(get_audit_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> AdminStatsResponse:
    """Get aggregate counts across all workspaces."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        stats = await storage.get_admin_stats()

        try:
            await audit_service.record(AuditEvent(
                event_type="admin",
                action="read",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="stats",
            ))
        except Exception:
            logger.debug("Audit record failed for admin stats read")

        return AdminStatsResponse(**stats)

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get admin stats: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve admin statistics",
        )


@router.get("/tiering", response_model=TieringStatsResponse)
async def get_admin_tiering_stats(
    http_request: Request,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    audit_service: AuditService = Depends(get_audit_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> TieringStatsResponse:
    """Tenant-wide (cross-workspace) tiering statistics.

    The admin analogue of ``GET /v1/tiering/stats`` (which is per-workspace).
    Lets the admin console show one hot/cold picture for the whole tenant.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        # Lazy import mirrors api/v1/tiering.py's get_tiering_service_dep to
        # avoid pulling the tiering service into this module's import graph.
        from ...services.tiering import get_tiering_service
        tiering_service = get_tiering_service(v)
        stats = await tiering_service.get_admin_tiering_stats()

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
            logger.debug("Audit record failed for admin tiering stats read")

        return TieringStatsResponse(
            hot_memory_count=stats.hot_memory_count,
            cold_memory_count=stats.cold_memory_count,
            hot_storage_bytes=stats.hot_storage_bytes,
            cold_storage_bytes=stats.cold_storage_bytes,
            compression_ratio=stats.compression_ratio,
            estimated_savings_bytes=stats.estimated_savings_bytes,
            archival_candidates_count=stats.archival_candidates_count,
            document_storage_bytes=stats.document_storage_bytes,
            documents_table_bytes=stats.documents_table_bytes,
            document_pages_bytes=stats.document_pages_bytes,
        )

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get admin tiering stats: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve admin tiering statistics",
        )


@router.get("/storage/usage")
async def get_admin_storage_usage(
    http_request: Request,
    force: bool = Query(False, description="Bypass the data-connectors cache and recompute."),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
):
    """Tenant storage-usage rollup (effective/physical + deduped-logical).

    A thin proxy over data-connectors' ``/v1/admin/storage/usage``, which caches
    blobgw's per-domain pack accounting (blobgw owns the pack tables; only it can
    produce the physical/dedup numbers). MemoryLayer holds no storage DSNs — DC
    is the storage bridge. The dedup domain IS the tenant id. See
    ``.slop/tenant-storage-usage-spec.md``.
    """
    import json as _json
    from urllib.parse import urlencode

    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        # The dedup/storage domain is always the tenant id.
        resolved_domain = (getattr(ctx, "tenant_id", "") or "").strip()
        if not resolved_domain:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="no tenant in request context",
            )

        # Lazy imports mirror tasks/doc_added.py: keep the aether proxy + client
        # off this module's import path.
        from scitrera_aether_client.proxy import proxy_http_async
        from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION
        from ...config import (
            MEMORYLAYER_DATA_CONNECTORS_TOPIC,
            DEFAULT_MEMORYLAYER_DATA_CONNECTORS_TOPIC,
        )

        dc_topic = v.environ(
            MEMORYLAYER_DATA_CONNECTORS_TOPIC, DEFAULT_MEMORYLAYER_DATA_CONNECTORS_TOPIC
        )
        agent_service = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
        client = agent_service.client if agent_service is not None else None
        if client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="aether connection unavailable",
            )

        qs = urlencode({"domain": resolved_domain, "force": "true" if force else "false"})
        response = await proxy_http_async(
            client,
            target_topic=dc_topic,
            method="GET",
            path="/v1/admin/storage/usage?" + qs,
            timeout=30.0,
        )
        if response.status_code != 200:
            body = getattr(response, "body", b"") or b""
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="data-connectors storage usage returned %d: %s"
                % (response.status_code, body[:200]),
            )
        return _json.loads(response.body)

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get admin storage usage: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve storage usage",
        )


@router.post("/tiering/run", response_model=AdminTieringRunResponse)
async def run_admin_tiering(
    http_request: Request,
    request: AdminTieringRunRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    audit_service: AuditService = Depends(get_audit_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> AdminTieringRunResponse:
    """Run an on-demand cross-workspace tiering sweep.

    Defaults to ``dry_run=true`` so it reports how many memories *would* be
    archived (per workspace + totals) without changing anything — the intended
    way to gauge tiering impact before enabling the recurring job. Set
    ``dry_run=false`` to actually archive.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "write", workspace_id=ctx.workspace_id)

        from ...services.tiering import get_tiering_service
        tiering_service = get_tiering_service(v)

        logger.info(
            "Admin tiering run: dry_run=%s only_enabled=%s batch_size=%d",
            request.dry_run, request.only_enabled, request.batch_size,
        )
        summary = await tiering_service.run_archival_sweep(
            batch_size=request.batch_size,
            only_enabled=request.only_enabled,
            dry_run=request.dry_run,
            max_workspaces=request.max_workspaces,
        )

        try:
            await audit_service.record(AuditEvent(
                event_type="tiering",
                action="read" if request.dry_run else "update",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                resource_type="tier",
                metadata={
                    "dry_run": request.dry_run,
                    "archived": summary.get("total_archived", 0),
                    "candidates": summary.get("total_candidates", 0),
                },
            ))
        except Exception:
            logger.debug("Audit record failed for admin tiering run")

        return AdminTieringRunResponse(**summary)

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to run admin tiering sweep: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run tiering sweep",
        )


@router.get("/memories", response_model=PaginatedResponse)
async def list_admin_memories(
    http_request: Request,
    workspace_id: Optional[str] = Query(None),
    memory_status: Optional[str] = Query(None, alias="status"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> PaginatedResponse:
    """List memories across all workspaces (or filtered by workspace_id)."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        items, total = await storage.admin_list_memories(
            workspace_id=workspace_id,
            status=memory_status,
            limit=limit,
            offset=offset,
        )
        return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list admin memories: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list memories",
        )


@router.get("/sessions", response_model=PaginatedResponse)
async def list_admin_sessions(
    http_request: Request,
    workspace_id: Optional[str] = Query(None),
    include_expired: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> PaginatedResponse:
    """List sessions across all workspaces (or filtered by workspace_id)."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        items, total = await storage.admin_list_sessions(
            workspace_id=workspace_id,
            include_expired=include_expired,
            limit=limit,
            offset=offset,
        )
        return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list admin sessions: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list sessions",
        )


@router.get("/documents", response_model=PaginatedResponse)
async def list_admin_documents(
    http_request: Request,
    workspace_id: Optional[str] = Query(None),
    doc_status: Optional[str] = Query(None, alias="status"),
    enrichment_status: Optional[str] = Query(
        None,
        description=(
            "Knowledge-phase filter (not_applicable|pending|complete), "
            "independent of `status`. Combine the two to isolate e.g. "
            "ingested-but-never-decomposed: status=completed&"
            "enrichment_status=pending."
        ),
    ),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> PaginatedResponse:
    """List documents across all workspaces (or filtered by workspace_id).

    Each item carries BOTH phase fields: ``status`` (retrieval readiness) and
    ``enrichment_status`` (fact decomposition). They advance independently — a
    usable document is routinely ``completed``/``pending`` — so there is no
    single "is it done?" reading and callers should surface the two separately.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        items, total = await storage.admin_list_documents(
            workspace_id=workspace_id,
            status=doc_status,
            enrichment_status=enrichment_status,
            limit=limit,
            offset=offset,
        )
        return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list admin documents: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list documents",
        )


def _sample(ids, cap: int = 25) -> dict:
    """Summarize a page-id list as ``{count, sample}``.

    A stuck 1,000-page document would otherwise put 1,000 ids in the response
    for every phase; the count is what diagnoses it and the sample is enough to
    go look at a page.
    """
    ids = list(ids or ())
    return {"count": len(ids), "sample": ids[:cap], "truncated": len(ids) > cap}


@router.get("/documents/{document_id}")
async def get_admin_document(
    http_request: Request,
    document_id: str,
    workspace_id: Optional[str] = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> dict:
    """Full diagnostic detail for ONE document (admin, cross-workspace).

    Aggregates what is needed to answer "why is this document not done?", which
    spans two independent phases:

    * ``ingestion`` — the render/transcribe/embed/store gap analysis, including
      ``first_missing_phase``: the phase a resume would start at.
    * ``enrichment`` — the knowledge phase (fact decomposition), with the count
      still missing facts re-derived live rather than read from the stored
      status, so it is correct even if no sweep has run since.
    * ``jobs`` — every ingestion job that touched the document, ANY status. The
      failed and superseded attempts are usually the informative ones.

    Authorization: ``admin:read``.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        doc = await storage.get_document(document_id, workspace_id=workspace_id)
        if doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found",
            )

        from ...services.document.gap_analysis import (
            analyze_document_gaps,
            analyze_fact_gaps,
            resolve_effective_flags,
            resolve_enrichment_status,
        )

        # Each block is independently best-effort: a document is often opened
        # BECAUSE something about it is broken, and a failure in one diagnostic
        # must not blank the others (which are frequently the ones that explain
        # it). Failures are reported in-band rather than swallowed.
        gaps_out: dict = {}
        try:
            gaps = await analyze_document_gaps(v, storage, doc)
            gaps_out = {
                "is_complete": gaps.is_complete,
                "first_missing_phase": gaps.first_missing_phase,
                "has_pages": gaps.has_pages,
                "expected_page_count": gaps.expected_page_count,
                "page_count": gaps.page_count,
                "missing_render": gaps.missing_render,
                "pages_with_transcript": gaps.pages_with_transcript,
                "missing_transcript": _sample(gaps.pages_missing_transcript),
                "missing_image_embed": _sample(gaps.pages_missing_image_embed),
                "missing_text_embedding": _sample(gaps.pages_missing_text_embedding),
                "missing_multivector": _sample(gaps.pages_missing_multivector),
                "missing_memory": _sample(gaps.pages_missing_memory),
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("admin doc %s: gap analysis failed: %s", document_id, e)
            gaps_out = {"error": str(e)}

        scheduled = list(getattr(doc, "enrichment_memory_ids", None) or [])
        stored_status = getattr(doc.enrichment_status, "value", doc.enrichment_status)
        enrichment_out: dict = {
            "status": stored_status,
            # NOT the outstanding count: doc_verify never rewrites this list, so
            # it stays the set handed to decomposition at ingest.
            "scheduled_count": len(scheduled),
        }
        try:
            outstanding = await analyze_fact_gaps(storage, doc)
            # Same derivation doc_verify converges on, so "live" is exactly what
            # the next sweep would store. When it disagrees with the stored
            # status the document is simply awaiting a sweep, which is worth
            # seeing before anyone re-drives it by hand.
            live, still = resolve_enrichment_status(doc, outstanding)
            enrichment_out["live_status"] = live.value
            enrichment_out["stale"] = live.value != stored_status
            enrichment_out["outstanding_count"] = len(still)
            enrichment_out["outstanding_sample"] = still[:25]
        except Exception as e:  # noqa: BLE001
            logger.warning("admin doc %s: fact-gap analysis failed: %s", document_id, e)
            enrichment_out["error"] = str(e)

        try:
            flags = resolve_effective_flags(v, doc)
            enrichment_out["effective_flags"] = flags.as_dict()
        except Exception as e:  # noqa: BLE001
            logger.warning("admin doc %s: flag resolution failed: %s", document_id, e)

        jobs_out: list = []
        lister = getattr(storage, "list_jobs_for_documents", None)
        if lister is not None:
            try:
                jobs = await lister([document_id], limit=25)
                jobs_out = [
                    {
                        "id": j.id,
                        "status": getattr(j.status, "value", j.status),
                        "progress_percent": j.progress_percent,
                        "documents_processed": j.documents_processed,
                        "total_memories_created": j.total_memories_created,
                        "errors": j.errors,
                        "created_at": j.created_at.isoformat() if j.created_at else None,
                        "started_at": j.started_at.isoformat() if j.started_at else None,
                        "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                    }
                    for j in jobs
                ]
            except Exception as e:  # noqa: BLE001
                logger.warning("admin doc %s: job lookup failed: %s", document_id, e)

        return {
            "document": {
                "id": doc.id,
                "workspace_id": doc.workspace_id,
                "filename": doc.filename,
                "document_type": doc.document_type,
                "mime_type": doc.mime_type,
                "status": getattr(doc.status, "value", doc.status),
                "size_bytes": doc.size_bytes,
                "page_count": doc.page_count,
                "source_vfs_ref": getattr(doc, "source_vfs_ref", None),
                "metadata": getattr(doc, "metadata", None) or {},
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "processing_started_at": (
                    doc.processing_started_at.isoformat() if doc.processing_started_at else None
                ),
                "processing_completed_at": (
                    doc.processing_completed_at.isoformat() if doc.processing_completed_at else None
                ),
            },
            "ingestion": gaps_out,
            "enrichment": enrichment_out,
            "jobs": jobs_out,
        }

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get admin document %s: %s", document_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get document",
        )


class DocVerifySweepRequest(BaseModel):
    """Body for the admin ``doc_verify`` sweep. ``workspace_id`` omitted → GLOBAL
    sweep (every workspace); present → that workspace only. Per-DOCUMENT re-verify
    is the self-service, non-admin ``POST /v1/documents/{id}/verify``."""

    workspace_id: Optional[str] = Field(None)


@router.post("/documents/verify", status_code=status.HTTP_202_ACCEPTED)
async def trigger_doc_verify_sweep(
    http_request: Request,
    request: DocVerifySweepRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    task_service: TaskService = Depends(get_task_service),
    logger: Logger = Depends(get_logger),
) -> dict:
    """Enqueue a WORKSPACE-wide or SYSTEM-wide ``doc_verify`` reconcile sweep
    (admin-only): ``{}`` sweeps every workspace, ``{workspace_id}`` sweeps one.

    Per-document re-verify is the self-service ``POST /v1/documents/{id}/verify``.
    Authorization: ``admin:write``.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "write", workspace_id=ctx.workspace_id)

        payload: dict = {}
        if request.workspace_id:
            payload["workspace_id"] = request.workspace_id
        mode = "workspace" if request.workspace_id else "global"
        title = (
            "Reconciling workspace documents" if request.workspace_id
            else "Reconciling all documents"
        )
        task_id = await task_service.schedule_task(
            "doc_verify",
            payload,
            metadata={"bg_kind": "doc_verify", "visibility": "workspace", "title": title},
        )
        logger.info(
            "Admin doc_verify sweep enqueued (mode=%s workspace=%s task=%s)",
            mode, request.workspace_id or "*", task_id,
        )
        return {"task_id": task_id, "mode": mode, "scheduled": task_id is not None}

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to enqueue admin doc_verify sweep: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enqueue doc_verify sweep",
        )


@router.get("/datasets", response_model=PaginatedResponse)
async def list_admin_datasets(
    http_request: Request,
    workspace_id: Optional[str] = Query(None),
    ds_status: Optional[str] = Query(None, alias="status"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> PaginatedResponse:
    """List datasets across all workspaces (or filtered by workspace_id)."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        items, total = await storage.admin_list_datasets(
            workspace_id=workspace_id,
            status=ds_status,
            limit=limit,
            offset=offset,
        )
        return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list admin datasets: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list datasets",
        )


@router.get("/jobs", response_model=PaginatedResponse)
async def list_admin_jobs(
    http_request: Request,
    workspace_id: Optional[str] = Query(None),
    job_status: Optional[str] = Query(None, alias="status"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> PaginatedResponse:
    """List all jobs across all workspaces (or filtered by workspace_id)."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        items, total = await storage.admin_list_jobs(
            workspace_id=workspace_id,
            status=job_status,
            limit=limit,
            offset=offset,
        )
        return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list admin jobs: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list jobs",
        )


@router.get("/chat-threads", response_model=PaginatedResponse)
async def list_admin_chat_threads(
    http_request: Request,
    workspace_id: Optional[str] = Query(None),
    ownership: Optional[str] = Query(None),
    include_hidden: bool = Query(True),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> PaginatedResponse:
    """List chat threads across all workspaces (or filtered by workspace_id)."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        items, total = await storage.admin_list_chat_threads(
            workspace_id=workspace_id,
            ownership=ownership,
            include_hidden=include_hidden,
            limit=limit,
            offset=offset,
        )
        return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list admin chat threads: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list chat threads",
        )


@router.get("/chat-threads/{thread_id}")
async def get_admin_chat_thread(
    http_request: Request,
    thread_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> dict:
    """Full chat-thread row for the admin detail drawer."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        row = await storage.admin_get_chat_thread(thread_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat thread not found")
        return row

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get admin chat thread: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get chat thread",
        )


@router.get("/skills", response_model=PaginatedResponse)
async def list_admin_skills(
    http_request: Request,
    workspace_id: Optional[str] = Query(None),
    enabled: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> PaginatedResponse:
    """List skills across all workspaces (or filtered by workspace_id)."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        items, total = await storage.admin_list_skills(
            workspace_id=workspace_id,
            enabled=enabled,
            limit=limit,
            offset=offset,
        )
        return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list admin skills: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list skills",
        )


@router.get("/skills/{skill_id}")
async def get_admin_skill(
    http_request: Request,
    skill_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> dict:
    """Full skill row for the admin detail drawer."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        row = await storage.admin_get_skill(skill_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill not found")
        return row

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get admin skill: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get skill",
        )


@router.get("/mcp-servers", response_model=PaginatedResponse)
async def list_admin_mcp_servers(
    http_request: Request,
    workspace_id: Optional[str] = Query(None),
    transport: Optional[str] = Query(None),
    enabled: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> PaginatedResponse:
    """List MCP servers across all workspaces (or filtered by workspace_id)."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        items, total = await storage.admin_list_mcp_servers(
            workspace_id=workspace_id,
            transport=transport,
            enabled=enabled,
            limit=limit,
            offset=offset,
        )
        return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list admin MCP servers: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list MCP servers",
        )


@router.get("/mcp-servers/{mcp_id}")
async def get_admin_mcp_server(
    http_request: Request,
    mcp_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> dict:
    """Full MCP-server row for the admin detail drawer."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        row = await storage.admin_get_mcp_server(mcp_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found")
        return row

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get admin MCP server: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCP server",
        )


@router.get("/applications", response_model=PaginatedResponse)
async def list_admin_applications(
    http_request: Request,
    enabled: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> PaginatedResponse:
    """List registered applications (tenant-scoped)."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        items, total = await storage.admin_list_applications(
            enabled=enabled,
            limit=limit,
            offset=offset,
        )
        return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list admin applications: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list applications",
        )


@router.get("/applications/{app_id}")
async def get_admin_application(
    http_request: Request,
    app_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: Logger = Depends(get_logger),
) -> dict:
    """Full application row for the admin detail drawer."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        storage = _get_storage(v)
        row = await storage.admin_get_application(app_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
        return row

    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get admin application: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get application",
        )


class AdminAPIPlugin(Plugin):
    """Plugin to register admin API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True
