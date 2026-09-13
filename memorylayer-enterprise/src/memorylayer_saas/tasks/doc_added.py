# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Task handler for ``memorylayer-task.doc_added`` pool tasks.

Triggered by ``data-connectors`` when a new VFS entry is registered.  The
handler CLASSIFIES against any existing document (vfs_ref → content_hash) and
acts idempotently:

1. Resolve existing: by ``vfs_ref`` first, then ``(workspace, content_hash)``.
   Same bytes via a new vfs_ref → LINK the vfs_ref to the existing doc.
2. NO-OP if the existing doc is in-flight (PROCESSING/PENDING* within the
   freshness TTL) or already complete (gap analysis ``is_complete``; the
   complete case re-emits ``ingest_complete`` in case the KB signal was lost).
3. RESUME (gap-fill) otherwise: claim the doc (re-stamp
   ``processing_started_at``) and schedule the chained task for the first
   missing phase via ``resume_at`` (render/transcribe/embed/finalize).
4. FRESH (no existing): create a Document (PENDING_FETCH), JIT-mint a fetch URL
   from ``data-connectors`` and HTTP GET the bytes, store the source blob via
   ``DocumentIngestionService.store_upload_blob()``, link the VFS entry, and
   schedule the chained ``document_render`` pipeline. ``ingest_complete`` is
   emitted by the chained ``document_finalize`` (NOT inline here).

On terminal failure: emit ``memorylayer.ingest_failed`` and do NOT re-raise so
Aether marks the task complete.  On retryable failure: raise so Aether's
pool-task disconnect-reaper requeues.

Configuration
-------------
``MEMORYLAYER_DATA_CONNECTORS_TOPIC``
    Aether topic for the data-connectors service.  Default:
    ``sv::data-connectors`` (no specifier — gateway routes to any replica).
"""
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote

from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
from memorylayer_server.services.tasks.base import EXT_TASK_SERVICE, TaskSchedule
from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
from scitrera_app_framework import Variables, ext_parse_bool, get_extension, get_logger

# Guarded imports: these are optional runtime deps that may not be present
# in all deployment configurations.  Imported at module level so tests can
# trivially patch ``memorylayer_saas.tasks.doc_added.proxy_http_async`` etc.
try:
    from scitrera_aether_client.proxy import proxy_http_async
except ImportError:  # pragma: no cover
    proxy_http_async = None  # type: ignore[assignment]

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from memorylayer_saas.config import (
    DEFAULT_MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
    DEFAULT_MEMORYLAYER_INGEST_INFLIGHT_TTL,
    MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
    MEMORYLAYER_INGEST_INFLIGHT_TTL,
)
from memorylayer_saas.models.document import (
    Document,
    DocumentExtractionOptions,
    DocumentStatus,
    DocumentType,
    IngestionJob,
    JobStatus,
)
from memorylayer_saas.services.document import (
    EXT_DOCUMENT_INGESTION_SERVICE,
    get_document_ingestion_service,
)
from memorylayer_saas.services.document.gap_analysis import (
    PHASE_EMBED,
    PHASE_RENDER,
    PHASE_STORE,
    PHASE_TRANSCRIBE,
    analyze_document_gaps,
)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

# Aether ProgressKind.PROGRESS_KIND_APP — classifies these reports for the
# Background Tasks ("app") UI surface rather than chat/task surfaces.
_PROGRESS_KIND_APP = 2

# Total number of forward milestones the happy path emits: queued, fetch, the
# six pipeline phases (render/transcribe/embed/persist/store/finalize), and the
# terminal complete.  Used to compute the step fraction (step_sequence/total).
_PROGRESS_TOTAL_STEPS = 8

DATA_CONNECTORS_TOPIC = "MEMORYLAYER_DATA_CONNECTORS_TOPIC"
# Implementation-only address — see config.py:DEFAULT_MEMORYLAYER_DATA_CONNECTORS_TOPIC
# for why we don't include a specifier here.
DEFAULT_DATA_CONNECTORS_TOPIC = "sv::data-connectors"

_VFS_RESOURCE_TYPE = "vfs"
_ACCESS_READ = 10
_ACCESS_READ_WRITE = 20


def _vfs_access_request(
    workspace_id: str,
    vfs_ref: str,
    *,
    operation: str,
    required_access_level: int,
):
    """Build the exact Aether resource check enforced by data-connectors."""
    from scitrera_aether_client.proto import aether_pb2

    return aether_pb2.ResourceAccessRequest(
        resource_type=_VFS_RESOURCE_TYPE,
        resource_id=(
            f"workspaces/{quote(workspace_id, safe='')}/entries/"
            f"{quote(vfs_ref, safe='')}"
        ),
        operation=operation,
        workspace=workspace_id,
        required_access_level=required_access_level,
    )


# ---------------------------------------------------------------------------
# Retryable vs terminal exception classification
# ---------------------------------------------------------------------------

# Network errors and timeouts are retryable; everything else is terminal.
_RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


class DocAddedTaskHandler(TaskHandlerPlugin):
    """Task handler for ``doc_added`` — triggered by data-connectors.

    Claims pool tasks of type ``memorylayer-task.doc_added`` and drives the
    ingestion pipeline for VFS-referenced documents.
    """

    def get_task_type(self) -> str:
        return "doc_added"

    async def handle(self, v: Variables, payload: dict) -> None:
        """Execute the doc_added handler.

        Args:
            v: Variables instance.
            payload: Dict with workspace_id, vfs_ref, content_hash,
                     connector_id (optional), filename_hint (optional).
        """
        logger = get_logger(v, name="DocAddedTaskHandler")

        workspace_id = payload.get("workspace_id", "")
        vfs_ref = payload.get("vfs_ref", "")
        content_hash = payload.get("content_hash", "")
        filename_hint = payload.get("filename_hint", "document")
        connector_id = payload.get("connector_id", "")

        # Reserved keys injected by the worker runner for the native (POOL)
        # task path: the Aether task_id (snapshot correlation key) and the
        # task's string metadata map (title, visibility, task_class, ...).
        aether_task_id = payload.get("_aether_task_id", "")
        task_metadata = payload.get("_task_metadata") or {}
        progress = _ProgressEmitter(v, aether_task_id, task_metadata, logger)

        if not workspace_id or not vfs_ref:
            logger.error(
                "doc_added payload missing workspace_id or vfs_ref: %s", payload,
            )
            return  # terminal — malformed payload, don't requeue

        storage = get_extension(EXT_STORAGE_BACKEND, v)
        task_service = get_extension(EXT_TASK_SERVICE, v)

        # ---- Classify: resolve any existing document -------------------
        # Identity precedence: vfs_ref first, then (workspace_id, content_hash).
        existing = None
        if hasattr(storage, "find_document_by_vfs_ref"):
            existing = await storage.find_document_by_vfs_ref(vfs_ref)

        linked_new_vfs_ref = False
        if existing is None and content_hash:
            by_hash = await storage.find_document_by_hash(workspace_id, content_hash)
            if by_hash is not None:
                # Same bytes arriving via a NEW vfs_ref: link the vfs_ref to the
                # existing doc instead of re-ingesting (decision #2 — LINK).
                existing = by_hash
                linked_new_vfs_ref = True

        if existing is not None:
            await self._handle_existing(
                v, storage, task_service, existing, vfs_ref, workspace_id,
                content_hash, link_new_vfs_ref=linked_new_vfs_ref,
                progress=progress, filename_hint=filename_hint, logger=logger,
            )
            return

        # ---- FRESH: no existing document -------------------------------
        await self._handle_fresh(
            v, storage, task_service, vfs_ref, workspace_id, content_hash,
            connector_id, filename_hint, progress, logger,
        )

    async def _handle_existing(
        self, v, storage, task_service, existing, vfs_ref, workspace_id,
        content_hash, *, link_new_vfs_ref, progress, filename_hint, logger,
    ) -> None:
        """Classify an existing document and either no-op or resume gap-fill."""
        # Link the new vfs_ref to the existing doc (same bytes, new ref).
        if link_new_vfs_ref and not existing.source_vfs_ref:
            try:
                await storage.update_document(existing.id, source_vfs_ref=vfs_ref)
                existing.source_vfs_ref = vfs_ref
            except Exception:  # noqa: BLE001 - linkage is best-effort
                logger.warning(
                    "doc_added: failed to link vfs_ref %s to existing doc %s",
                    vfs_ref, existing.id, exc_info=True,
                )

        # Every arriving VFS entry needs its own reverse link, even when the
        # content hash resolves to a document with a different source_vfs_ref.
        # Retry failed acknowledgements instead of reporting ingestion success.
        await _link_vfs_entry(
            v, vfs_ref, workspace_id, existing.id, "", logger, require_success=True,
        )

        # In-flight protection: a fresh PROCESSING/PENDING* doc is owned by
        # another worker — NO-OP. Past the freshness TTL it is treated as
        # orphaned and re-driven via gap-fill below.
        inflight_statuses = {
            DocumentStatus.PROCESSING,
            DocumentStatus.PENDING,
            DocumentStatus.PENDING_FETCH,
        }
        ttl = int(v.environ(
            MEMORYLAYER_INGEST_INFLIGHT_TTL,
            default=DEFAULT_MEMORYLAYER_INGEST_INFLIGHT_TTL,
        ))
        if existing.status in inflight_statuses and _is_fresh(existing, ttl):
            logger.info(
                "doc_added: doc %s in-flight (status=%s, fresh) — NO-OP",
                existing.id, existing.status,
            )
            return

        # Analyze gaps to decide NO-OP (complete) vs resume.
        gaps = await analyze_document_gaps(v, storage, existing)
        if gaps.is_complete:
            logger.info(
                "doc_added: doc %s already complete — NO-OP", existing.id,
            )
            # Re-emit ingest_complete in case the KB signal was lost.
            await _emit_event(v, workspace_id, "memorylayer.ingest_complete", {
                "ml_doc_id": existing.id,
                "vfs_ref": vfs_ref,
            }, logger)
            return

        # Gap-fill: atomically CLAIM the doc (conditional UPDATE to PROCESSING +
        # re-stamp processing_started_at). The CAS closes the read-then-write race
        # the freshness check leaves open: if a concurrent worker already claimed
        # it (fresh PROCESSING), we lose and NO-OP. Then resume at the first
        # missing phase via the chained pipeline.
        claimed = await storage.try_claim_document(existing.id, workspace_id, ttl)
        if not claimed:
            logger.info(
                "doc_added: lost claim race for doc %s (owned by another worker) "
                "— NO-OP", existing.id,
            )
            return

        job_id = "job_%s" % uuid.uuid4().hex[:12]
        job = IngestionJob(
            id=job_id,
            workspace_id=workspace_id,
            document_ids=[existing.id],
            status=JobStatus.QUEUED,
        )
        await storage.create_job(job)

        await progress.emit(
            "running",
            step_name="resume",
            step_detail="Resuming ingestion for %s at %s"
            % (filename_hint, gaps.first_missing_phase or "finalize"),
            step_sequence=1,
            completion=0.10,
        )

        await resume_at(
            v, task_service, gaps.first_missing_phase, existing, job_id, logger,
        )
        logger.info(
            "doc_added: resumed doc %s at phase=%s (job %s)",
            existing.id, gaps.first_missing_phase, job_id,
        )

    async def _handle_fresh(
        self, v, storage, task_service, vfs_ref, workspace_id, content_hash,
        connector_id, filename_hint, progress, logger,
    ) -> None:
        """Fresh ingest: create the doc, fetch bytes, store the blob, then
        schedule the chained ``document_render`` pipeline (no inline run)."""
        doc_id = "doc_%s" % uuid.uuid4().hex[:12]
        job_id = "job_%s" % uuid.uuid4().hex[:12]

        doc_type = _detect_type_from_filename(filename_hint)
        extraction_options = DocumentExtractionOptions()

        # Seeded at CREATION, not finalize: create_memories_for_pages resolves
        # flags long before finalize pins them, so a decompose opt-out has to be
        # on the document from the start or the first pass decomposes anyway.
        entry_metadata = await _fetch_vfs_entry_metadata(
            v, vfs_ref, workspace_id, logger,
        )
        doc_metadata: dict = _document_source_metadata(entry_metadata)
        if connector_id:
            doc_metadata["connector_id"] = connector_id
        requested_flags = _requested_ingest_flags(entry_metadata)
        if requested_flags:
            doc_metadata["requested_ingest_flags"] = requested_flags
            logger.info(
                "doc_added: %s requested ingest flags %s", doc_id, requested_flags,
            )

        doc = Document(
            id=doc_id,
            workspace_id=workspace_id,
            filename=filename_hint,
            document_type=doc_type,
            content_hash=content_hash or "pending",
            source_vfs_ref=vfs_ref,
            size_bytes=0,  # updated after fetch
            status=DocumentStatus.PENDING_FETCH,
            target_context_id=extraction_options.target_context_id,
            extraction_options=extraction_options,
            metadata=doc_metadata,
        )
        doc = await storage.create_document(doc)

        job = IngestionJob(
            id=job_id,
            workspace_id=workspace_id,
            document_ids=[doc_id],
            status=JobStatus.QUEUED,
        )
        await storage.create_job(job)

        logger.info(
            "doc_added: created document %s (vfs_ref=%s), job %s",
            doc_id, vfs_ref, job_id,
        )

        # Milestone 1: claimed -> running.
        await progress.emit(
            "running",
            step_name="queued",
            step_detail="Ingestion started for %s" % filename_hint,
            step_sequence=1,
            completion=0.0,
        )

        # ---- JIT URL mint + fetch bytes --------------------------------
        await progress.emit(
            "running",
            step_name="fetch",
            step_detail="Fetching content for %s" % filename_hint,
            step_sequence=2,
            completion=0.10,
        )
        try:
            file_bytes = await _fetch_bytes_via_data_connectors(
                v, vfs_ref, workspace_id, logger,
            )
        except _RETRYABLE_EXCEPTIONS as exc:
            logger.warning(
                "doc_added: retryable fetch failure for %s: %s, raising for requeue",
                vfs_ref, exc,
            )
            await progress.emit(
                "running",
                step_name="retrying",
                step_detail="Fetch failed, retrying: %s" % exc,
                step_sequence=2,
                completion=-1.0,
            )
            raise
        except Exception as exc:
            logger.error(
                "doc_added: terminal fetch failure for %s: %s",
                vfs_ref, exc, exc_info=True,
            )
            await progress.emit(
                "failed",
                step_name="fetch",
                step_detail="Fetch failed: %s" % exc,
                step_sequence=2,
                completion=-1.0,
                summary=str(exc),
            )
            now = datetime.now(timezone.utc)
            await storage.update_document(
                doc_id,
                status=DocumentStatus.FAILED.value,
                processing_completed_at=now,
            )
            await storage.update_job(
                job_id,
                status=JobStatus.FAILED.value,
                completed_at=now,
                errors=[{"document_id": doc_id, "error": str(exc)}],
            )
            await _emit_event(v, workspace_id, "memorylayer.ingest_failed", {
                "ml_doc_id": doc_id,
                "vfs_ref": vfs_ref,
                "error": str(exc),
            }, logger)
            return  # terminal — don't re-raise

        # ---- Store blob + hand off to the chained pipeline -------------
        # Converged path: fresh ingests now go through the chained document_*
        # tasks (render -> transcribe -> embed -> finalize), the same path
        # upload_document uses. We store the fetched bytes as the source blob
        # (so document_render can re-read them) and schedule render; finalize
        # emits ingest_complete + links the VFS entry is best-effort here now.
        try:
            ingestion_service = get_document_ingestion_service(v)
            await ingestion_service.store_upload_blob(doc, file_bytes)
        except _RETRYABLE_EXCEPTIONS as exc:
            logger.warning(
                "doc_added: retryable blob-store failure for %s: %s, raising",
                doc_id, exc,
            )
            await progress.emit(
                "running",
                step_name="retrying",
                step_detail="Blob store failed, retrying: %s" % exc,
                step_sequence=3,
                completion=-1.0,
            )
            raise
        except Exception as exc:
            logger.error(
                "doc_added: terminal blob-store failure for %s: %s",
                doc_id, exc, exc_info=True,
            )
            await progress.emit(
                "failed",
                step_name="ingest",
                step_detail="Blob store failed: %s" % exc,
                step_sequence=3,
                completion=-1.0,
                summary=str(exc),
            )
            now = datetime.now(timezone.utc)
            await storage.update_document(
                doc_id,
                status=DocumentStatus.FAILED.value,
                processing_completed_at=now,
            )
            await storage.update_job(
                job_id,
                status=JobStatus.FAILED.value,
                completed_at=now,
                errors=[{"document_id": doc_id, "error": str(exc)}],
            )
            await _emit_event(v, workspace_id, "memorylayer.ingest_failed", {
                "ml_doc_id": doc_id,
                "vfs_ref": vfs_ref,
                "error": str(exc),
            }, logger)
            return  # terminal — don't re-raise

        # Link the ML doc back to the VFS entry now that the row exists. The
        # chained finalize emits ingest_complete on completion.
        await _link_vfs_entry(
            v, vfs_ref, workspace_id, doc_id, job_id, logger,
        )

        await progress.emit(
            "running",
            step_name="ingest",
            step_detail="Queued processing for %s" % filename_hint,
            step_sequence=3,
            completion=0.50,
        )

        await task_service.schedule_task(
            "document_render",
            {
                "document_id": doc_id,
                "job_id": job_id,
                "workspace_id": workspace_id,
            },
            priority=3,
        )
        logger.info(
            "doc_added: scheduled document_render for %s (vfs_ref=%s, job %s)",
            doc_id, vfs_ref, job_id,
        )

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        return None


# ---------------------------------------------------------------------------
# Progress emission (Background Tasks UI — Phase 2)
# ---------------------------------------------------------------------------

class _ProgressEmitter:
    """Best-effort emitter of ingest progress for the Background Tasks UI.

    Captures the **Aether task_id** (the id the worker claimed, as surfaced by
    the runner under ``payload["_aether_task_id"]``) so live progress can be
    merged with the ``query_tasks`` snapshot, which is keyed by that same id.

    All emission is best-effort: any failure is logged and swallowed so it can
    never abort ingestion or change the task lifecycle.  When no Aether task_id
    is available (e.g. legacy/scheduler dispatch paths) the emitter is a no-op.
    """

    __slots__ = ("_client", "_task_id", "_recipient", "_base_metadata", "_logger")

    def __init__(self, v: Variables, aether_task_id: str, task_metadata: dict, logger) -> None:
        self._logger = logger
        self._task_id = aether_task_id or ""

        client = None
        try:
            agent_service = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
            client = getattr(agent_service, "client", None)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Progress: could not resolve Aether client", exc_info=True)
        self._client = client

        # Workspace-broadcast by default; target the initiating user only when
        # the ingest is private AND we know who started it.
        recipient = ""
        if task_metadata.get("visibility") == "private":
            initiated_by = task_metadata.get("initiated_by")
            if initiated_by:
                recipient = initiated_by
        self._recipient = recipient

        # Carried on every report so the relay/frontend can filter + label.
        self._base_metadata = {
            "title": task_metadata.get("title", ""),
            "bg_kind": task_metadata.get("bg_kind", "ingest"),
            "task_class": task_metadata.get("task_class", ""),
        }

    @property
    def enabled(self) -> bool:
        return bool(self._task_id) and self._client is not None

    async def emit(
        self,
        state: str,
        *,
        step_name: str,
        step_detail: str = "",
        step_sequence: int = 0,
        completion: float = -1.0,
        summary: str = "",
    ) -> None:
        """Emit one progress milestone (best-effort)."""
        if not self.enabled:
            return

        metadata = dict(self._base_metadata)
        if completion >= 0.0:
            metadata["completion"] = "%.3f" % completion

        try:
            await self._client.report_progress(
                self._task_id,
                state=state,
                completion=completion,
                summary=summary or step_detail or step_name,
                step_name=step_name,
                step_detail=step_detail,
                step_sequence=step_sequence,
                step_total=_PROGRESS_TOTAL_STEPS,
                step_type="processing",
                recipient=self._recipient,
                metadata=metadata,
                kind=_PROGRESS_KIND_APP,
            )
        except Exception:
            self._logger.warning(
                "Progress: report_progress failed for task %s (best-effort)",
                self._task_id, exc_info=True,
            )

        # Mirror progress onto the task's own metadata so the snapshot
        # (reconstructed from task metadata) stays consistent with the live
        # stream.  ``update_task`` is not yet exposed by the Aether SDK; guard
        # for it so this engages automatically once it lands.
        update_task = getattr(self._client, "update_task", None)
        if update_task is None:
            return
        try:
            step_blob = json.dumps({
                "name": step_name,
                "detail": step_detail,
                "sequence": step_sequence,
                "total": _PROGRESS_TOTAL_STEPS,
                "state": state,
            })
            await update_task(self._task_id, metadata={
                "completion": "%.3f" % completion if completion >= 0.0 else "-1",
                "step": step_blob,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            self._logger.warning(
                "Progress: update_task failed for task %s (best-effort)",
                self._task_id, exc_info=True,
            )


# ---------------------------------------------------------------------------
# Classification / resume helpers
# ---------------------------------------------------------------------------

def _is_fresh(doc, ttl_seconds: int) -> bool:
    """Whether a doc's ``processing_started_at`` is within the freshness window.

    A doc with no ``processing_started_at`` is treated as NOT fresh (re-drivable)
    so a wedged PENDING/PENDING_FETCH that never started is not blocked forever.
    """
    started = getattr(doc, "processing_started_at", None)
    if started is None:
        return False
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - started) < timedelta(seconds=ttl_seconds)


async def resume_at(
    v: Variables, task_service, phase: Optional[str], doc, job_id: str, logger,
) -> None:
    """Schedule the chained task that resumes ingestion at ``phase``.

    Every chained ``document_*`` task loads its state from storage, so resuming
    at the first missing phase is just a matter of scheduling the right task:

    - render    -> ``document_render``
    - transcribe-> ``document_transcribe`` (or ``document_embed`` if transcribe
                   is disabled, mirroring document_render's own skip)
    - embed     -> ``document_embed``
    - store/None-> ``document_finalize`` (recompute memories + status)
    """
    if phase == PHASE_RENDER:
        next_task = "document_render"
    elif phase == PHASE_TRANSCRIBE:
        transcribe_enabled = v.environ(
            MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
            default=DEFAULT_MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
            type_fn=ext_parse_bool,
        )
        next_task = "document_transcribe" if transcribe_enabled else "document_embed"
    elif phase == PHASE_EMBED:
        next_task = "document_embed"
    else:
        # PHASE_STORE or None: recompute memories + finalize status.
        next_task = "document_finalize"

    # A resumed embed phase carries the same multi-minute retry policy as the
    # forward pipeline so a transient embed-backend outage is retried in-task
    # (imported lazily to avoid a tasks-package import cycle at module load).
    retry_policy = None
    if next_task == "document_embed":
        from memorylayer_saas.tasks.document_embed import build_embed_retry_policy
        retry_policy = build_embed_retry_policy()

    await task_service.schedule_task(
        next_task,
        {
            "document_id": doc.id,
            "job_id": job_id,
            "workspace_id": doc.workspace_id,
        },
        priority=3,
        retry_policy=retry_policy,
    )
    logger.info(
        "resume_at: scheduled %s for doc %s (phase=%s)",
        next_task, doc.id, phase,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _fetch_bytes_via_data_connectors(
    v: Variables, vfs_ref: str, workspace_id: str, logger,
) -> bytes:
    """Mint a JIT fetch URL from data-connectors and HTTP GET the bytes.

    Uses ``proxy_http_async`` to reach data-connectors over Aether, then
    performs a direct HTTP GET against the returned upstream URL.
    """
    dc_topic = v.environ(DATA_CONNECTORS_TOPIC, DEFAULT_DATA_CONNECTORS_TOPIC)

    agent_service = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
    client = agent_service.client

    # Mint fetch URL
    fetch_body = json.dumps({
        "vfs_ref": vfs_ref,
        "workspace_id": workspace_id,
    }).encode("utf-8")

    response = await proxy_http_async(
        client,
        target_topic=dc_topic,
        method="POST",
        path="/v1/urls/fetch",
        headers={"Content-Type": "application/json"},
        body=fetch_body,
        app_workspace=workspace_id,
        checked_access=_vfs_access_request(
            workspace_id,
            vfs_ref,
            operation="read",
            required_access_level=_ACCESS_READ,
        ),
        timeout=30.0,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "data-connectors /v1/urls/fetch returned %d: %s"
            % (response.status_code, response.body[:200])
        )

    url_info = json.loads(response.body)
    upstream_url = url_info["url"]
    upstream_headers = url_info.get("headers", {})

    # HTTP GET the actual bytes
    async with httpx.AsyncClient(timeout=120.0) as http_client:
        resp = await http_client.get(upstream_url, headers=upstream_headers)
        resp.raise_for_status()
        return resp.content


#: Ingest-flag keys accepted from a VFS entry. An allow-list, not a passthrough:
#: entry metadata is written by whoever minted the upload, so anything not named
#: here — including flags a future build might add — is ignored rather than
#: seeded onto the document.
_REQUESTABLE_INGEST_FLAGS = ("decompose",)
_SOURCE_METADATA_SECRET_FRAGMENTS = ("authorization", "password", "secret", "token")
_SOURCE_METADATA_BULK_KEYS = ("slack_content", "teams_content", "discord_content", "_raw_content_b64")


def _requested_ingest_flags(entry_metadata: dict) -> dict:
    requested = entry_metadata.get("ingest_flags")
    if not isinstance(requested, dict):
        return {}
    return {
        key: bool(requested[key])
        for key in _REQUESTABLE_INGEST_FLAGS
        if key in requested
    }


def _document_source_metadata(entry_metadata: dict) -> dict:
    """Bounded VFS metadata safe to retain on the enterprise Document row."""
    return {
        str(key): value
        for key, value in entry_metadata.items()
        if str(key) not in _SOURCE_METADATA_BULK_KEYS
        and not any(fragment in str(key).casefold() for fragment in _SOURCE_METADATA_SECRET_FRAGMENTS)
    }


async def _fetch_vfs_entry_metadata(
    v: Variables, vfs_ref: str, workspace_id: str, logger,
) -> dict:
    """Read connector metadata from the authoritative VFS entry, best-effort."""
    try:
        dc_topic = v.environ(DATA_CONNECTORS_TOPIC, DEFAULT_DATA_CONNECTORS_TOPIC)
        agent_service = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
        response = await proxy_http_async(
            agent_service.client,
            target_topic=dc_topic,
            method="GET",
            path=f"/v1/vfs/entries/{quote(vfs_ref, safe='')}",
            app_workspace=workspace_id,
            checked_access=_vfs_access_request(
                workspace_id,
                vfs_ref,
                operation="read",
                required_access_level=_ACCESS_READ,
            ),
            timeout=10.0,
        )
        if response.status_code != 200:
            logger.debug("VFS entry %s lookup returned %d", vfs_ref, response.status_code)
            return {}
        body = json.loads(response.body) or {}
        metadata = body.get("metadata") or {}
        return metadata if isinstance(metadata, dict) else {}
    except Exception:
        logger.warning(
            "Failed to read metadata from VFS entry %s (best-effort)",
            vfs_ref,
            exc_info=True,
        )
        return {}


async def _fetch_requested_ingest_flags(
    v: Variables, vfs_ref: str, workspace_id: str, logger,
) -> dict:
    """Read the upload-time ingest flags off the VFS entry.

    The ``doc_added`` payload carries only vfs_ref / content_hash /
    connector_id / filename_hint, so flags the uploader asked for have to be
    read back from the entry itself.

    Best-effort: ingesting with default flags is a far better outcome than
    failing to ingest because data-connectors was briefly unreachable, so every
    failure degrades to "no request expressed".
    """
    return _requested_ingest_flags(
        await _fetch_vfs_entry_metadata(v, vfs_ref, workspace_id, logger)
    )


async def _link_vfs_entry(
    v: Variables,
    vfs_ref: str,
    workspace_id: str,
    doc_id: str,
    job_id: str,
    logger,
    *,
    require_success: bool = False,
) -> None:
    """Link the VFS entry back to the ML document via data-connectors.

    Existing-document callers require acknowledgement before skipping ingestion.
    Other callers retain best-effort behavior.
    """
    try:
        dc_topic = v.environ(DATA_CONNECTORS_TOPIC, DEFAULT_DATA_CONNECTORS_TOPIC)
        agent_service = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
        client = agent_service.client

        link = {"ml_doc_id": doc_id}
        if job_id:
            link["ml_job_id"] = job_id
        link_body = json.dumps(link).encode("utf-8")

        response = await proxy_http_async(
            client,
            target_topic=dc_topic,
            method="POST",
            path=f"/v1/vfs/entries/{quote(vfs_ref, safe='')}/link",
            headers={"Content-Type": "application/json"},
            body=link_body,
            app_workspace=workspace_id,
            checked_access=_vfs_access_request(
                workspace_id,
                vfs_ref,
                operation="write",
                required_access_level=_ACCESS_READ_WRITE,
            ),
            timeout=10.0,
        )
        if not 200 <= response.status_code < 300:
            raise ConnectionError("VFS link was not acknowledged: HTTP %s" % response.status_code)
    except Exception as exc:
        if require_success:
            raise ConnectionError("VFS document link is unconfirmed") from exc
        logger.warning(
            "Failed to link VFS entry %s to document %s (best-effort)",
            vfs_ref, doc_id, exc_info=True,
        )


async def _emit_event(
    v: Variables, workspace_id: str, event_name: str, data: dict, logger,
) -> None:
    """Emit an event via Aether's ``send_event`` to the ``event::*`` plane.

    Best-effort; failure is logged but not raised.

    The payload is JSON (UTF-8), NOT msgpack: Aether's native workflow engine
    consumes the default ``event::*`` plane and ``json.Unmarshal``s each event
    into ``EventPayload{source_agent, event_names[], data, workspace}``.  We
    deliberately send to the broad (un-scoped) ``event::*`` topic so the
    workflow engine receives it; a workspace-scoped variant would not be seen.
    """
    try:
        agent_service = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
        client = agent_service.client

        if not hasattr(client, "send_event"):
            logger.debug("Aether client does not support send_event, skipping event emission")
            return

        envelope = {
            "source_agent": "memorylayer",
            "workspace": workspace_id,
            "event_names": [event_name],
            "data": {"workspace_id": workspace_id, **data},
        }
        payload_bytes = json.dumps(envelope).encode("utf-8")
        await client.send_event(payload_bytes)
    except Exception:
        logger.warning(
            "Failed to emit event %s for workspace %s (best-effort)",
            event_name, workspace_id,
            exc_info=True,
        )


def _detect_type_from_filename(filename: str) -> DocumentType:
    """Best-effort document type detection from filename.

    Falls back to TEXT for unknown extensions.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mapping = {
        "pdf": DocumentType.PDF,
        "md": DocumentType.MARKDOWN,
        "markdown": DocumentType.MARKDOWN,
        "txt": DocumentType.TEXT,
        "text": DocumentType.TEXT,
        "html": DocumentType.HTML,
        "htm": DocumentType.HTML,
        "docx": DocumentType.DOCX,
        "pptx": DocumentType.PPTX,
    }
    return mapping.get(ext, DocumentType.TEXT)
