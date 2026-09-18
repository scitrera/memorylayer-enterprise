# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""FastAPI application for the data-connectors service.

Exposes the HTTP surface described in the plan:
- /v1/providers/* — connector/provider lifecycle (admin)
- /v1/sync/* — sync execution
- /v1/urls/* — URL minting (upload/download/fetch)
- /v1/vfs/entries/* — VFS catalog CRUD + link subresource

The app is designed to run both standalone (uvicorn) and behind Aether's
``proxy_http_async`` via the ``ProxyHttpTerminator`` registered in
``aether_service.py``.
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx
from contextlib import asynccontextmanager
from typing import Any, Optional
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response, status

from data_connectors.messages.providers import (
    CreateProviderReq,
    ProviderListResponse,
    ProviderResponse,
    UpdateProviderReq,
)
from data_connectors.messages.vfs import (
    FinalizeVfsEntryReq,
    LinkMlDocumentReq,
    RegisterVfsEntryReq,
    UpdateVfsEntryReq,
    VfsEntryListResponse,
    VfsEntryResponse,
)
from data_connectors.messages.urls import (
    FetchURLResponse,
    MintDownloadURLReq,
    MintFetchURLReq,
    MintUploadURLReq,
    UploadURLResponse,
)
from data_connectors.messages.sync import SyncJobResponse, TriggerSyncReq
from data_connectors.vfs.blob_store import BlobStore
from data_connectors.vfs.catalog import VfsCatalog
from data_connectors.vfs.deletion import delete_entry_with_blob
from data_connectors.services.url_minter import UrlMinter
from data_connectors.services.sync_engine import SyncEngine
from data_connectors.services.vfs_gc import (
    DEFAULT_BATCH_LIMIT as DEFAULT_GC_BATCH_LIMIT,
    DEFAULT_MAX_AGE_SECONDS as DEFAULT_GC_MAX_AGE_SECONDS,
    gc_enabled,
    run_gc_loop,
    sweep_abandoned_uploads,
)
from data_connectors.connectors import ConnectorRegistry, import_all_connectors
from data_connectors.connectors.manual_upload import ManualUploadConnector
from data_connectors.db.provider_store import InMemoryProviderStore
from data_connectors.server.aether_service import AetherServiceRegistration
from data_connectors.server.vfs_authorization import (
    ACCESS_READ,
    ACCESS_READ_WRITE,
    TrustedVfsAuthority,
    VfsAuthorizationError,
    require_vfs_access,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state (module-level singletons, initialized during lifespan)
# ---------------------------------------------------------------------------

_blob_store: Optional[BlobStore] = None
_catalog: Optional[Any] = None  # VfsCatalog (in-memory) or PgVfsCatalog
_url_minter: Optional[UrlMinter] = None
_sync_engine: Optional[SyncEngine] = None
_manual_upload: Optional[ManualUploadConnector] = None
_aether_service: Optional[AetherServiceRegistration] = None
# Background sweep for uploads that were minted but never finalized.
_gc_task: Optional[asyncio.Task] = None

# Provider store: PgProviderStore when DC_POSTGRESQL_URL is set, else in-memory.
_provider_store: Any = None
# Per-tenant storage-usage rollup (lazy TTL cache over blobgw's /admin/usage).
_storage_usage: Any = None  # StorageUsageService
# In-memory sync job store (transient run state; not persisted)
_sync_jobs: dict[str, dict] = {}

# Built-in providers seeded at startup. ``manual_upload`` is a built-in connector
# (direct user uploads), not a user-configured provider, but uploads tag
# vfs_entries.connector_id='manual_upload' which FKs to providers.id — so it must
# exist as a row. A single global row (id == connector id) satisfies the FK for
# every workspace's uploads; it is not workspace-scoped and won't show in any
# workspace's provider list.
MANUAL_UPLOAD_PROVIDER_ID = "manual_upload"
# sahara_artifact: like manual_upload, a built-in connector (not a user-configured
# provider) whose only job is to be a providers row so vfs_entries tagged
# connector_id='sahara_artifact' satisfy the FK. Used for artifacts the sahara
# agent harness produces (present_artifacts etc.) and lands in the VFS directly.
SAHARA_ARTIFACT_PROVIDER_ID = "sahara_artifact"
# agent_generated: the third built-in, and the DEFAULT connector_id of
# TenantInterface2.put_artifact — so it is what every app-generated artifact is
# tagged with (JGL bid summaries, contract reviews, generated .docx). It was
# missing, so the FK rejected the insert and mint_upload_url returned 500: an
# agent could do all of its work and then fail on the last step, writing its
# output. tenant_interface2.ARTIFACT_CONNECTOR_IDS already lists this alongside
# sahara_artifact, i.e. the caller side always expected both to exist.
AGENT_GENERATED_PROVIDER_ID = "agent_generated"
_BUILTIN_PROVIDER_WORKSPACE = "_global"


def _builtin_manual_upload_provider() -> dict:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return {
        "id": MANUAL_UPLOAD_PROVIDER_ID,
        "workspace_id": _BUILTIN_PROVIDER_WORKSPACE,
        "name": "Manual Upload",
        "provider_type": "manual_upload",
        "description": "Built-in connector for direct user file uploads.",
        "enabled": True,
        "connection_args": {},
        "schedule": None,
        "last_sync_at": None,
        "metadata": {"builtin": True},
        "created_at": now,
        "updated_at": now,
    }


def _builtin_sahara_artifact_provider() -> dict:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return {
        "id": SAHARA_ARTIFACT_PROVIDER_ID,
        "workspace_id": _BUILTIN_PROVIDER_WORKSPACE,
        "name": "Sahara Artifact",
        "provider_type": "sahara_artifact",
        "description": "Built-in connector for artifacts produced by the sahara agent harness.",
        "enabled": True,
        "connection_args": {},
        "schedule": None,
        "last_sync_at": None,
        "metadata": {"builtin": True},
        "created_at": now,
        "updated_at": now,
    }


def _builtin_agent_generated_provider() -> dict:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return {
        "id": AGENT_GENERATED_PROVIDER_ID,
        "workspace_id": _BUILTIN_PROVIDER_WORKSPACE,
        "name": "Agent Generated",
        "provider_type": "agent_generated",
        "description": "Built-in connector for artifacts produced by platform apps and agents.",
        "enabled": True,
        "connection_args": {},
        "schedule": None,
        "last_sync_at": None,
        "metadata": {"builtin": True},
        "created_at": now,
        "updated_at": now,
    }


def _init_services() -> None:
    """Initialize shared services from environment configuration.

    The VFS catalog + provider store are PostgreSQL-backed when
    ``DC_POSTGRESQL_URL`` is set (persists across restarts), and fall back to
    in-memory implementations otherwise (dev/test).
    """
    global _blob_store, _catalog, _url_minter, _sync_engine, _manual_upload, _provider_store, _storage_usage

    # Backend selection (DC_BLOB_TYPE: 's3' default, or 'blobgw'). The blobgw
    # backend routes through the object gateway + external edge; the S3 backend
    # is constructed via the module-level ``BlobStore`` symbol so existing tests
    # that patch ``app.BlobStore`` keep working.
    from data_connectors.vfs.blob_store_factory import is_blobgw_backend

    if is_blobgw_backend():
        from data_connectors.vfs.blob_store_factory import create_blob_store

        _blob_store = create_blob_store()
    else:
        prefix = os.environ.get("DC_BLOB_PREFIX", "")
        bucket = os.environ.get("DC_BLOB_BUCKET", "data-connectors-dev")
        endpoint_url = os.environ.get("DC_BLOB_ENDPOINT_URL")
        region = os.environ.get("DC_BLOB_REGION", "us-east-1")
        # When the access key / secret are unset the BlobStore uses boto3's default
        # credential chain (IRSA web-identity role in-cluster) — no static keys.
        access_key = os.environ.get("DC_BLOB_ACCESS_KEY_ID")
        secret_key = os.environ.get("DC_BLOB_SECRET_ACCESS_KEY")

        _blob_store = BlobStore(
            bucket=bucket,
            prefix=prefix,
            endpoint_url=endpoint_url,
            region=region,
            access_key_id=access_key,
            secret_access_key=secret_key,
        )

    # Storage-usage rollup calls blobgw directly (owns the pack tables) regardless
    # of which blob-I/O backend is selected above, so it reads DC_BLOBGW_URL on its
    # own. Cache TTL is configurable; force bypasses it at request time.
    from data_connectors.vfs.storage_usage import DEFAULT_TTL_S, StorageUsageService

    _storage_usage = StorageUsageService(
        blobgw_url=os.environ.get("DC_BLOBGW_URL", ""),
        ttl_s=int(os.environ.get("DC_STORAGE_USAGE_TTL_S", str(DEFAULT_TTL_S))),
    )

    # Select the persistence backend based on DC_POSTGRESQL_URL. Delayed import
    # of the engine keeps SQLAlchemy engine creation off the import path.
    from data_connectors.db.engine import get_database_url

    if get_database_url() is not None:
        from data_connectors.db.engine import session_scope
        from data_connectors.db.provider_store import PgProviderStore
        from data_connectors.vfs.catalog import PgVfsCatalog

        _catalog = PgVfsCatalog(session_scope)
        _provider_store = PgProviderStore(session_scope)
        logger.info("Persistence backend: PostgreSQL (DC_POSTGRESQL_URL set)")
    else:
        _catalog = VfsCatalog()
        _provider_store = InMemoryProviderStore()
        logger.info("Persistence backend: in-memory (DC_POSTGRESQL_URL unset)")

    _url_minter = UrlMinter(_blob_store, _catalog)

    # Aether client for task emission is set after connection
    _sync_engine = SyncEngine(task_client=None, catalog=_catalog)
    _manual_upload = ManualUploadConnector(_blob_store, _catalog)

    # Populate the connector registry so providers of any type can be synced.
    registered = import_all_connectors()
    logger.info("Registered connectors: %s", ", ".join(registered))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize services and Aether connection."""
    global _aether_service, _sync_engine, _gc_task

    # Initialize OpenTelemetry tracing + instrument this app. No-op unless the
    # OTel SDK is installed AND MEMORYLAYER_OTEL_ENABLED is truthy; fully
    # fail-soft so a telemetry problem never breaks ingest. Done first so spans
    # cover startup (migrations, service init) as well as request handling.
    from data_connectors.server.otel import init_otel

    _otel_shutdown = init_otel(app)

    # Run DB migrations before initializing services when persisting to Postgres.
    # alembic drives its own event loop, so run it in a worker thread to avoid
    # "asyncio.run() inside a running loop" under uvicorn's lifespan.
    from data_connectors.db.engine import get_database_url

    if get_database_url() is not None:
        from data_connectors.db.engine import run_migrations
        await asyncio.to_thread(run_migrations)

    _init_services()

    # Seed the built-in providers so entries that tag vfs_entries.connector_id with
    # a built-in id ('manual_upload' for direct uploads, 'sahara_artifact' for
    # harness-produced artifacts, 'agent_generated' for app/agent output) satisfy
    # the providers FK. Runs after migrations (providers table exists) and is
    # idempotent across restarts.
    for _builtin in (
        _builtin_manual_upload_provider,
        _builtin_sahara_artifact_provider,
        _builtin_agent_generated_provider,
    ):
        try:
            await _provider_store.ensure(_builtin())
        except Exception:
            logger.warning("Failed to seed built-in provider %s", _builtin.__name__, exc_info=True)

    # Reclaim entries from uploads that were never completed. Runs in-process
    # rather than as a CronJob because it is per-tenant state and this service
    # already owns it; the deployment is single-replica, and the sweep is a
    # delete-by-predicate, so a second replica would at worst duplicate work
    # rather than corrupt anything.
    if gc_enabled():
        _gc_task = asyncio.create_task(run_gc_loop(_catalog, _blob_store))
    else:
        logger.info("Abandoned-upload GC disabled via DC_UPLOAD_GC_ENABLED")

    # Connect to Aether (non-fatal if unavailable)
    _aether_service = AetherServiceRegistration()
    try:
        await _aether_service.connect(app)
        # Wire the Aether client into the sync engine for task emission
        if _aether_service.client and _sync_engine:
            _sync_engine._task_client = _aether_service.client
    except Exception:
        logger.warning("Aether connection failed; running in standalone mode", exc_info=True)

    yield

    # Shutdown
    if _gc_task is not None:
        _gc_task.cancel()
        try:
            await _gc_task
        except asyncio.CancelledError:
            pass

    if _aether_service:
        await _aether_service.disconnect()

    # Dispose the Postgres engine if one was created.
    from data_connectors.db.engine import get_database_url

    if get_database_url() is not None:
        from data_connectors.db.engine import close_engine
        await close_engine()

    # Flush + close the OTel TracerProvider (no-op when tracing was disabled).
    if _otel_shutdown is not None:
        _otel_shutdown()


app = FastAPI(
    title="data-connectors",
    description="Universal VFS, connector, and URL minting service for the Scitrera AI platform",
    version="0.0.1",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/livez")
async def livez():
    """Connection-gated k8s liveness probe (readiness stays on /healthz).

    Returns 200 while the Aether service connection is healthy — live now, or
    down-but-recently-up within the SDK reconnect grace (AETHER_MAX_RECONNECT_
    ATTEMPTS=0 retries forever, so a routine gateway roll heals on its own).
    Returns 503 only when the connection has been down past the grace (a stuck
    gateway), so kubelet restarts the pod. Before the Aether client connects
    (standalone mode / startup) the probe stays 200 — the data-plane is HTTP and
    the SDK's pre-connect grace governs boot, mirroring connection_healthy().
    """
    svc = _aether_service
    client = svc.client if svc is not None else None
    if client is None or client.connection_healthy():
        return {"status": "ok"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


# ---------------------------------------------------------------------------
# Storage usage (admin)
# ---------------------------------------------------------------------------

@app.get("/v1/admin/storage/usage")
async def storage_usage(domain: str = Query(...), force: bool = Query(False)):
    """Per-tenant storage-usage rollup for ``domain`` (the dedup/tenant domain).

    data-connectors is the shared aggregation point: it fetches blobgw's
    ``/admin/usage`` (physical/effective + deduped-logical) and caches per domain
    with a TTL. ``force=true`` bypasses the cache. MemoryLayer's admin API calls
    this over Aether and surfaces it in the admin console.
    """
    if _storage_usage is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="storage usage service not initialized")
    try:
        return await _storage_usage.get(domain, force=force)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"blobgw usage error: {e.response.status_code}")
    except Exception as e:  # noqa: BLE001 — surface any blobgw/transport failure as 502
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"blobgw usage failed: {e}")


# ---------------------------------------------------------------------------
# Provider lifecycle (admin)
# ---------------------------------------------------------------------------

@app.post("/v1/providers", response_model=ProviderResponse, status_code=status.HTTP_201_CREATED)
async def create_provider(request: CreateProviderReq) -> ProviderResponse:
    """Create a new data provider."""
    from datetime import datetime, timezone
    provider_id = f"dp_{uuid4().hex[:16]}"
    now = datetime.now(timezone.utc)
    provider = {
        "id": provider_id,
        "workspace_id": request.metadata.get("workspace_id", "_default"),
        "name": request.name,
        "provider_type": request.provider_type,
        "description": request.description,
        "enabled": request.enabled,
        "connection_args": request.connection_args,
        "schedule": request.schedule,
        "last_sync_at": None,
        "metadata": request.metadata,
        "created_at": now,
        "updated_at": now,
    }
    assert _provider_store is not None
    await _provider_store.create(provider)
    return ProviderResponse(**provider)


@app.get("/v1/providers", response_model=ProviderListResponse)
async def list_providers(
    workspace_id: str = Query("_default"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> ProviderListResponse:
    """List data providers."""
    assert _provider_store is not None
    page, total = await _provider_store.list(workspace_id=workspace_id, limit=limit, offset=offset)
    return ProviderListResponse(
        providers=[ProviderResponse(**p) for p in page],
        total_count=total,
    )


@app.patch("/v1/providers/{provider_id}", response_model=ProviderResponse)
async def update_provider(provider_id: str, request: UpdateProviderReq) -> ProviderResponse:
    """Update a data provider."""
    assert _provider_store is not None
    updates = request.model_dump(exclude_none=True)
    provider = await _provider_store.update(provider_id, updates)
    if provider is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Provider not found: {provider_id}")
    return ProviderResponse(**provider)


@app.delete("/v1/providers/{provider_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_provider(provider_id: str) -> None:
    """Delete a data provider."""
    assert _provider_store is not None
    deleted = await _provider_store.delete(provider_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Provider not found: {provider_id}")


# ---------------------------------------------------------------------------
# Sync execution
# ---------------------------------------------------------------------------

def _instantiate_connector(provider: dict) -> Any:
    """Instantiate a connector from a provider record.

    Looks up the connector class via ``ConnectorRegistry``, then passes
    ``connection_args`` plus the shared ``catalog`` / ``blob_store`` where
    the connector's ``__init__`` accepts them.

    Raises:
        HTTPException 400 if the connector type is unknown or instantiation fails.
    """
    provider_type = provider.get("provider_type", "")
    connection_args: dict = dict(provider.get("connection_args", {}))

    # Ensure all connector modules are imported so the registry is populated.
    # (Also done at startup via _init_services; idempotent here as a safety net.)
    import_all_connectors()

    cls = ConnectorRegistry.get(provider_type)
    if cls is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown connector type: {provider_type}",
        )

    # Inject catalog / blob_store for connectors that accept them
    import inspect
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    if "catalog" in params and _catalog is not None:
        connection_args["catalog"] = _catalog
    if "blob_store" in params and _blob_store is not None:
        connection_args["blob_store"] = _blob_store

    try:
        return cls(**connection_args)
    except TypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to instantiate {provider_type} connector: {exc}",
        )


async def _run_sync_job(
    job_id: str,
    provider_id: str,
    workspace_id: str,
    connector: Any,
    full_sync: bool,
) -> None:
    """Execute a sync job in the background, updating the job record on completion."""
    from datetime import datetime, timezone

    job = _sync_jobs.get(job_id)
    if job is None:
        return

    job["status"] = "running"
    job["started_at"] = datetime.now(timezone.utc)

    try:
        await connector.load()
        assert _sync_engine is not None
        result = await _sync_engine.sync_provider(
            provider_id=provider_id,
            workspace_id=workspace_id,
            connector=connector,
            full_sync=full_sync,
        )
        job["entries_discovered"] = result.get("discovered", 0)
        job["entries_synced"] = result.get("synced", 0)
        job["status"] = "completed"
    except Exception as exc:
        logger.error("Sync job %s failed: %s", job_id, exc, exc_info=True)
        job["status"] = "failed"
        job["error"] = str(exc)
    finally:
        job["completed_at"] = datetime.now(timezone.utc)


@app.post("/v1/sync/trigger", response_model=SyncJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_sync(request: TriggerSyncReq, background_tasks: BackgroundTasks) -> SyncJobResponse:
    """Trigger a sync for a provider.

    Instantiates the appropriate connector, creates a sync job record,
    and runs the sync in a background task.  Returns 202 with the job
    ID immediately; poll ``GET /v1/sync/jobs/{job_id}`` for status.
    """
    from datetime import datetime, timezone

    assert _provider_store is not None
    provider = await _provider_store.get(request.provider_id)
    if provider is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Provider not found: {request.provider_id}")

    # Instantiate the connector (validates provider_type + connection_args)
    connector = _instantiate_connector(provider)

    job_id = f"syncjob_{uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)
    job = {
        "job_id": job_id,
        "provider_id": request.provider_id,
        "workspace_id": request.workspace_id,
        "status": "pending",
        "entries_discovered": 0,
        "entries_synced": 0,
        "error": None,
        "started_at": None,
        "completed_at": None,
        "created_at": now,
    }
    _sync_jobs[job_id] = job

    # Run the sync in a background task so we return 202 immediately
    background_tasks.add_task(
        _run_sync_job,
        job_id=job_id,
        provider_id=request.provider_id,
        workspace_id=request.workspace_id,
        connector=connector,
        full_sync=request.full_sync,
    )

    return SyncJobResponse(**job)


@app.get("/v1/sync/jobs/{job_id}", response_model=SyncJobResponse)
async def get_sync_job(job_id: str) -> SyncJobResponse:
    """Get sync job status."""
    job = _sync_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Sync job not found: {job_id}")
    return SyncJobResponse(**job)


# ---------------------------------------------------------------------------
# URL minting
# ---------------------------------------------------------------------------


def _authorize_vfs(
    http_request: Request,
    *,
    workspace_id: str,
    operation: str,
    required_access_level: int,
    vfs_ref: Optional[str] = None,
) -> TrustedVfsAuthority:
    try:
        return require_vfs_access(
            http_request,
            workspace_id=workspace_id,
            operation=operation,
            required_access_level=required_access_level,
            vfs_ref=vfs_ref,
        )
    except VfsAuthorizationError as exc:
        logger.warning(
            "VFS authorization denied workspace=%s operation=%s vfs_ref=%s: %s",
            workspace_id,
            operation,
            vfs_ref,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="VFS access denied",
        ) from exc


async def _vfs_entry_or_404(vfs_ref: str, claimed_workspace: Optional[str] = None) -> Any:
    assert _catalog is not None
    entry = await _catalog.get(vfs_ref)
    if entry is None or (
        claimed_workspace is not None and entry.workspace_id != claimed_workspace
    ):
        # Deliberately use the same response for a missing ref and a workspace
        # mismatch so callers cannot use this endpoint as a cross-workspace
        # existence oracle.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"VFS entry not found: {vfs_ref}",
        )
    return entry


@app.post("/v1/urls/upload", response_model=UploadURLResponse)
async def mint_upload_url(request: MintUploadURLReq, http_request: Request) -> UploadURLResponse:
    """Mint a presigned upload URL (POST policy by default, PUT on request)."""
    assert _url_minter is not None
    _authorize_vfs(
        http_request,
        workspace_id=request.workspace_id,
        operation="write",
        required_access_level=ACCESS_READ_WRITE,
    )
    result = await _url_minter.mint_upload_url(
        workspace_id=request.workspace_id,
        filename=request.filename,
        content_type=request.content_type,
        connector_id=request.connector_id or "manual_upload",
        method=request.method,
        # source_path/size_bytes/metadata were accepted by the request model
        # and then dropped here, so a caller's metadata silently never reached
        # the entry. Forwarded now.
        source_path=request.source_path,
        size_bytes=request.size_bytes,
        metadata=request.metadata,
    )
    return UploadURLResponse(
        method=result["method"],
        upload_url=result["url"],
        upload_internal_url=result.get("internal_url"),
        blob_key=result["blob_key"],
        vfs_ref=result.get("vfs_ref"),
        fields=result.get("fields", {}),
        headers=result.get("headers", {}),
        expires_at=result["expires_at"],
    )


@app.post("/v1/urls/download", response_model=FetchURLResponse)
async def mint_download_url(request: MintDownloadURLReq, http_request: Request) -> FetchURLResponse:
    """Mint a presigned download URL for a VFS entry."""
    assert _url_minter is not None
    await _vfs_entry_or_404(request.vfs_ref, request.workspace_id)
    authority = _authorize_vfs(
        http_request,
        workspace_id=request.workspace_id,
        operation="read",
        required_access_level=ACCESS_READ,
        vfs_ref=request.vfs_ref,
    )
    try:
        url, headers, expires_at = await _url_minter.mint_download_url(
            vfs_ref=request.vfs_ref,
            workspace_id=request.workspace_id,
            subject=authority.user_subject if authority.enforced else request.subject,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return FetchURLResponse(url=url, headers=headers, expires_at=expires_at)


@app.post("/v1/urls/fetch", response_model=FetchURLResponse)
async def mint_fetch_url(request: MintFetchURLReq, http_request: Request) -> FetchURLResponse:
    """Mint a JIT fetch URL (used by MemoryLayer workers)."""
    assert _url_minter is not None
    await _vfs_entry_or_404(request.vfs_ref, request.workspace_id)
    _authorize_vfs(
        http_request,
        workspace_id=request.workspace_id,
        operation="read",
        required_access_level=ACCESS_READ,
        vfs_ref=request.vfs_ref,
    )
    try:
        url, headers, expires_at = await _url_minter.mint_fetch_url(
            vfs_ref=request.vfs_ref,
            workspace_id=request.workspace_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return FetchURLResponse(url=url, headers=headers, expires_at=expires_at)


# ---------------------------------------------------------------------------
# VFS catalog
# ---------------------------------------------------------------------------

@app.post("/v1/vfs/entries", response_model=VfsEntryResponse, status_code=status.HTTP_201_CREATED)
async def register_vfs_entry(request: RegisterVfsEntryReq, http_request: Request) -> VfsEntryResponse:
    """Register a new VFS entry and emit a doc_added task."""
    assert _catalog is not None
    authority = _authorize_vfs(
        http_request,
        workspace_id=request.workspace_id,
        operation="write",
        required_access_level=ACCESS_READ_WRITE,
    )
    entry = await _catalog.register(
        workspace_id=request.workspace_id,
        connector_id=request.connector_id,
        source_path=request.source_path,
        content_hash=request.content_hash,
        content_type=request.content_type,
        size_bytes=request.size_bytes,
        metadata=request.metadata,
    )
    # Emit doc_added task for MemoryLayer ingestion.
    #
    # This route is the integration point for user-initiated ingest: the
    # ws-server's dc_client calls POST /v1/vfs/entries after an upload
    # completes. When the caller supplies ``initiated_by`` the task is classed
    # BATCH (long-running user-initiated job) and carries initiated_by +
    # visibility; otherwise it is treated as a BACKGROUND ingest.
    if _sync_engine is not None:
        # Delayed import: keeps the Aether SDK off the module-load path.
        from data_connectors.services.sync_engine import (
            TASK_CLASS_BACKGROUND,
            TASK_CLASS_BATCH,
        )
        filename = request.source_path.split("/")[-1] if request.source_path else "unknown"
        initiated_by = authority.initiated_by if authority.enforced else request.initiated_by
        if initiated_by is not None:
            task_class = TASK_CLASS_BATCH
        else:
            task_class = TASK_CLASS_BACKGROUND
        await _sync_engine.emit_doc_added(
            workspace_id=request.workspace_id,
            vfs_ref=entry.vfs_ref,
            content_hash=request.content_hash,
            connector_id=request.connector_id,
            filename_hint=filename,
            task_class=task_class,
            initiated_by=initiated_by,
            visibility=request.visibility,
        )
    return VfsEntryResponse(**entry.model_dump())


@app.post("/v1/vfs/entries/{vfs_ref}/finalize", response_model=VfsEntryResponse)
async def finalize_vfs_entry(vfs_ref: str, request: FinalizeVfsEntryReq, http_request: Request) -> VfsEntryResponse:
    """Finalize a placeholder VFS entry post-upload and emit a doc_added task.

    ``mint_upload_url`` pre-registers a placeholder entry (empty content_hash,
    null size) and hands the caller a stable ``vfs_ref``. After the browser
    finishes uploading the blob to S3, the ws-server calls this route. It:

      1. Backfills ``content_hash`` / ``size_bytes`` — derived server-side from
         the uploaded blob (``head_object`` for size, streaming SHA-256 for the
         hash) when the caller does not supply them.
      2. Emits the doc_added ingest task. When ``initiated_by`` is set the task
         is classed BATCH (user-initiated); otherwise BACKGROUND.

    Unlike ``register_vfs_entry`` this does NOT create a new entry — it operates
    on the existing placeholder, so user uploads do not double-register.

    Idempotency: the placeholder created by ``mint_upload_url`` starts with an
    empty ``content_hash``. We capture ``was_finalized`` from that signal at the
    top of the handler; a repeated FILE_UPLOAD_COMPLETE (browser retry, double
    click, ws reconnect replay) therefore skips blob re-derivation and the
    doc_added emit, returning the existing entry with 200 — so no duplicate
    ingest task appears in the Background Tasks panel.
    """
    assert _catalog is not None
    entry = await _vfs_entry_or_404(vfs_ref)
    authority = _authorize_vfs(
        http_request,
        workspace_id=entry.workspace_id,
        operation="write",
        required_access_level=ACCESS_READ_WRITE,
        vfs_ref=vfs_ref,
    )

    # Capture the finalize state BEFORE any backfill/update. A non-empty
    # content_hash means a prior finalize already populated this entry.
    was_finalized = bool(entry.content_hash)

    # Re-finalize: already populated, so skip blob re-derivation and the emit.
    # Return the existing entry to keep the endpoint 200 + idempotent.
    if was_finalized:
        return VfsEntryResponse(**entry.model_dump())

    content_hash = request.content_hash
    size_bytes = request.size_bytes
    content_type = request.content_type or entry.content_type

    # Backend-conditional finalize.
    #
    # blobgw: finalize_blob COMMITS the staged upload (moves it S3-staging ->
    # blobgw) AND returns content_hash + size. The commit is REQUIRED for the blob
    # to be retrievable, so it must run UNCONDITIONALLY — even when the caller
    # supplied both content_hash and size_bytes. Gating it on "metadata missing"
    # (as this once did) leaves the blob orphaned in staging and every later
    # GET /blob/{ref} 404s. Caller-supplied values are preserved; only missing
    # fields are filled from the finalize response.
    #
    # S3: the browser PUT already landed the object (no staging/commit step), so
    # finalize is metadata-only — derive size (head_object) / hash (streamed
    # SHA-256) only when the caller didn't supply them.
    if _blob_store is not None and entry.blob_key:
        from data_connectors.vfs.blob_store_factory import is_blobgw_backend

        if is_blobgw_backend():
            try:
                info = await _blob_store.finalize_blob(
                    entry.blob_key, content_type=content_type,
                )
                if content_hash is None:
                    content_hash = info.get("content_hash") or None
                if size_bytes is None:
                    size_val = info.get("size")
                    size_bytes = int(size_val) if size_val is not None else None
                if content_type is None:
                    content_type = info.get("content_type") or None
            except Exception:
                logger.warning(
                    "blobgw finalize failed for vfs_ref=%s (blob_key=%s); "
                    "leaving the placeholder pending for retry",
                    vfs_ref, entry.blob_key, exc_info=True,
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Blob commit failed; retry finalization",
                ) from None
        elif content_hash is None or size_bytes is None:
            try:
                if size_bytes is None:
                    head = await _blob_store.head_object(entry.blob_key)
                    if head is not None:
                        size_bytes = head.get("ContentLength")
                        if content_type is None:
                            content_type = head.get("ContentType")
                if content_hash is None:
                    import hashlib
                    data = await _blob_store.get_object(entry.blob_key)
                    content_hash = hashlib.sha256(data).hexdigest()
                    if size_bytes is None:
                        size_bytes = len(data)
            except Exception:
                logger.warning(
                    "Failed to derive blob metadata for vfs_ref=%s (blob_key=%s); "
                    "finalizing with available values",
                    vfs_ref, entry.blob_key, exc_info=True,
                )

    # Fall back to the existing entry hash if still unresolved (keeps the task
    # emittable; downstream dedup tolerates a recomputed hash later).
    if content_hash is None:
        content_hash = entry.content_hash or ""

    merged_metadata = dict(entry.metadata or {})
    if request.metadata:
        merged_metadata.update(request.metadata)

    updated = await _catalog.update(
        vfs_ref,
        content_hash=content_hash,
        content_type=content_type,
        size_bytes=size_bytes,
        metadata=merged_metadata,
    )
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"VFS entry not found: {vfs_ref}")

    # skip_ingest commits the blob (the edge finalize + content_hash/size backfill above
    # already ran) but suppresses the doc_added emit — for agent output artifacts that
    # must resolve by vfs_ref (download/render) without ingesting into the knowledge base.
    if _sync_engine is not None and not request.skip_ingest:
        # Delayed import: keeps the Aether SDK off the module-load path.
        from data_connectors.services.sync_engine import (
            TASK_CLASS_BACKGROUND,
            TASK_CLASS_BATCH,
        )
        filename = updated.source_path.split("/")[-1] if updated.source_path else "unknown"
        initiated_by = authority.initiated_by if authority.enforced else request.initiated_by
        task_class = TASK_CLASS_BATCH if initiated_by is not None else TASK_CLASS_BACKGROUND
        await _sync_engine.emit_doc_added(
            workspace_id=updated.workspace_id,
            vfs_ref=updated.vfs_ref,
            content_hash=content_hash,
            connector_id=updated.connector_id,
            filename_hint=filename,
            task_class=task_class,
            initiated_by=initiated_by,
            visibility=request.visibility,
        )
    return VfsEntryResponse(**updated.model_dump())


@app.get("/v1/vfs/entries/{vfs_ref}", response_model=VfsEntryResponse)
async def get_vfs_entry(vfs_ref: str, http_request: Request) -> VfsEntryResponse:
    """Get a VFS entry by reference."""
    assert _catalog is not None
    entry = await _vfs_entry_or_404(vfs_ref)
    _authorize_vfs(
        http_request,
        workspace_id=entry.workspace_id,
        operation="read",
        required_access_level=ACCESS_READ,
        vfs_ref=vfs_ref,
    )
    return VfsEntryResponse(**entry.model_dump())


@app.get("/v1/vfs/entries", response_model=VfsEntryListResponse)
async def list_vfs_entries(
    http_request: Request,
    workspace_id: str = Query(...),
    connector_id: Optional[str] = Query(None),
    connector_ids: Optional[str] = Query(
        None,
        description="Comma-separated connector allow-list (e.g. "
                    "'manual_upload,gdrive'). Applied after connector_id.",
    ),
    exclude_connector_ids: Optional[str] = Query(
        None,
        description="Comma-separated connector deny-list (e.g. "
                    "'sahara_artifact,agent_generated'). Applied last, so it "
                    "overrides the allow-lists.",
    ),
    source_path_prefix: Optional[str] = Query(
        None,
        description="Return only entries whose source_path starts with this "
                    "prefix (e.g. '/Bids/acme/'), for folder-style grouping.",
    ),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> VfsEntryListResponse:
    """List VFS entries with optional filtering.

    ``source_path_prefix`` exists because source_path is the only field
    carrying an application's OWN organisation of its files (e.g. grouping
    uploads under /Bids/{bidder}/). Without a server-side filter the caller
    must page the whole workspace and filter client-side — and a caller that
    passes an unsupported filter gets it silently dropped by FastAPI, then
    receives EVERY entry in the workspace while believing the query was scoped.
    """
    def _split(v: Optional[str]) -> Optional[list[str]]:
        if not v:
            return None
        items = [s.strip() for s in v.split(',') if s.strip()]
        return items or None

    assert _catalog is not None
    _authorize_vfs(
        http_request,
        workspace_id=workspace_id,
        operation="read",
        required_access_level=ACCESS_READ,
    )
    entries, total = await _catalog.list_entries(
        workspace_id=workspace_id,
        connector_id=connector_id,
        connector_ids=_split(connector_ids),
        exclude_connector_ids=_split(exclude_connector_ids),
        source_path_prefix=source_path_prefix,
        limit=limit,
        offset=offset,
    )
    return VfsEntryListResponse(
        entries=[VfsEntryResponse(**e.model_dump()) for e in entries],
        total_count=total,
    )


@app.patch("/v1/vfs/entries/{vfs_ref}", response_model=VfsEntryResponse)
async def update_vfs_entry(vfs_ref: str, request: UpdateVfsEntryReq, http_request: Request) -> VfsEntryResponse:
    """Update a VFS entry."""
    assert _catalog is not None
    existing = await _vfs_entry_or_404(vfs_ref)
    _authorize_vfs(
        http_request,
        workspace_id=existing.workspace_id,
        operation="write",
        required_access_level=ACCESS_READ_WRITE,
        vfs_ref=vfs_ref,
    )
    updates = request.model_dump(exclude_none=True)
    entry = await _catalog.update(vfs_ref, **updates)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"VFS entry not found: {vfs_ref}")
    return VfsEntryResponse(**entry.model_dump())


@app.post("/v1/vfs/entries/{vfs_ref}/link", response_model=VfsEntryResponse)
async def link_ml_document(vfs_ref: str, request: LinkMlDocumentReq, http_request: Request) -> VfsEntryResponse:
    """Link a VFS entry to a MemoryLayer document."""
    assert _catalog is not None
    existing = await _vfs_entry_or_404(vfs_ref)
    _authorize_vfs(
        http_request,
        workspace_id=existing.workspace_id,
        operation="write",
        required_access_level=ACCESS_READ_WRITE,
        vfs_ref=vfs_ref,
    )
    entry = await _catalog.link_ml_document(
        vfs_ref=vfs_ref,
        ml_doc_id=request.ml_doc_id,
        ml_job_id=request.ml_job_id,
    )
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"VFS entry not found: {vfs_ref}")
    return VfsEntryResponse(**entry.model_dump())


@app.delete("/v1/vfs/entries/{vfs_ref}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_vfs_entry(vfs_ref: str, http_request: Request) -> None:
    """Delete a VFS entry and the blob it points at."""
    assert _catalog is not None
    existing = await _vfs_entry_or_404(vfs_ref)
    _authorize_vfs(
        http_request,
        workspace_id=existing.workspace_id,
        operation="write",
        required_access_level=ACCESS_READ_WRITE,
        vfs_ref=vfs_ref,
    )
    outcome = await delete_entry_with_blob(_catalog, _blob_store, vfs_ref)
    if not outcome.found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"VFS entry not found: {vfs_ref}")


@app.post("/v1/admin/vfs/gc-abandoned-uploads")
async def gc_abandoned_uploads(
    dry_run: bool = False,
    max_age_seconds: int = DEFAULT_GC_MAX_AGE_SECONDS,
    limit: int = DEFAULT_GC_BATCH_LIMIT,
) -> dict:
    """Run the abandoned-upload sweep now instead of waiting for the timer.

    Exists for two reasons: draining a backlog that accumulated before the
    sweep existed, and letting an operator SEE what would go (``dry_run``)
    before anything is deleted.
    """
    assert _catalog is not None
    if max_age_seconds <= 0 or limit <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="max_age_seconds and limit must be positive",
        )
    return await sweep_abandoned_uploads(
        _catalog, _blob_store,
        max_age_seconds=max_age_seconds, limit=limit, dry_run=dry_run,
    )


# NOTE: there is deliberately no direct blob-upload endpoint. One existed
# ("for MemoryLayer page-image writes") and was never called by anything:
# MemoryLayer writes page images through its own BlobStorageService, not
# through here. It wrote blobs under a `blobs/` prefix with NO catalog entry,
# so anything stored that way was unreachable and uncollectable -- invisible
# to listings and to the abandoned-upload sweep alike. Blobs enter through
# mint-upload -> PUT -> finalize so that every one of them has an owning VFS
# entry, which is what makes it findable, deletable, and collectable.
