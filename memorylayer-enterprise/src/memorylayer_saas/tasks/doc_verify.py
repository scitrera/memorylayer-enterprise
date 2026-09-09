"""Task handler for ``memorylayer-task.doc_verify`` — reconcile / gap-fill sweep.

The Phase-3 backstop of the idempotent-ingestion design (§8). It reuses the
SAME primitives as the ``doc_added`` entry point — ``analyze_document_gaps`` +
``resume_at`` — to heal documents that ended up incomplete, plus
``analyze_fact_gaps`` to re-drive fact decomposition that never ran.

Two modes, by payload:

1. **On-demand** ``{workspace_id, document_id}`` — load the doc; if it is not
   in-flight (fresh PROCESSING/PENDING* within ``MEMORYLAYER_INGEST_INFLIGHT_TTL``)
   and ``analyze_document_gaps`` finds gaps, claim it (PROCESSING +
   ``processing_started_at=now``) and ``resume_at`` the first missing phase. Then
   re-schedule ``decompose_facts`` for any page composite memory missing facts.
   A complete, fact-complete doc is a NO-OP. The manual "heal this doc."

2. **Scheduled sweep** ``{workspace_id}`` or ``{}`` (global) — over each target
   workspace, list ``{PARTIAL, FAILED}`` documents PLUS stale-``PROCESSING`` ones
   (``processing_started_at`` older than the freshness TTL — the crashed-worker
   backstop), bounded per status/workspace by
   ``MEMORYLAYER_DOC_VERIFY_BATCH_LIMIT``. Gap-analyze each and resume the
   fixable ones; re-drive fact gaps. Logs swept / resumed / fact-resumed /
   skipped counts (no silent truncation).

**Attempt cap + terminal flag (anti-churn guard):** each time the sweep resumes
a doc it increments ``doc.metadata['reprocess_attempts']`` and persists it.  At
or above ``MEMORYLAYER_DOC_VERIFY_MAX_ATTEMPTS`` the doc is instead marked
terminally ``FAILED`` (``metadata['reconcile_terminal']=True``) and excluded
from all future sweeps.  This prevents permanently-broken docs (e.g. a corrupt
file failing render every run) from churning indefinitely.  The on-demand path
(document_id present) bypasses the cap to let an operator force a re-try.

**is_complete self-heal:** if gap-analysis finds a doc complete but its DB
status is not ``COMPLETED``, the sweep promotes it to ``COMPLETED`` rather than
resuming (heals a stale status left by a crash between finalize and the status
write).

Registered as a recurring schedule (``MEMORYLAYER_DOC_VERIFY_INTERVAL``,
``max_concurrent`` is implied by the single recurring registration) mirroring
``tasks/blob_gc.py``. A global ({} payload) sweep iterates all workspaces, the
same scoping ``decay_memories`` uses for its all-workspaces pass.
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from scitrera_app_framework import Variables, ext_parse_bool, get_extension, get_logger

from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
from memorylayer_server.services.tasks.base import TaskSchedule, EXT_TASK_SERVICE
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

from memorylayer_saas.config import (
    DEFAULT_MEMORYLAYER_DOC_VERIFY_BATCH_LIMIT,
    DEFAULT_MEMORYLAYER_DOC_VERIFY_ENABLED,
    DEFAULT_MEMORYLAYER_DOC_VERIFY_INTERVAL,
    DEFAULT_MEMORYLAYER_DOC_VERIFY_MAX_ATTEMPTS,
    DEFAULT_MEMORYLAYER_INGEST_INFLIGHT_TTL,
    MEMORYLAYER_DOC_VERIFY_BATCH_LIMIT,
    MEMORYLAYER_DOC_VERIFY_ENABLED,
    MEMORYLAYER_DOC_VERIFY_INTERVAL,
    MEMORYLAYER_DOC_VERIFY_MAX_ATTEMPTS,
    MEMORYLAYER_INGEST_INFLIGHT_TTL,
)
from memorylayer_saas.models.document import (
    DocumentStatus,
    IngestionJob,
    JobStatus,
)
from memorylayer_saas.services.document.gap_analysis import (
    analyze_document_gaps,
    analyze_fact_gaps,
    resolve_enrichment_status,
)
from memorylayer_saas.tasks.doc_added import _is_fresh, resume_at

# Statuses the scheduled sweep inspects for incomplete documents. Stale
# PROCESSING is handled separately (freshness-gated) since a FRESH PROCESSING
# doc is owned by an in-flight worker and must not be touched.
_SWEEP_TERMINAL_STATUSES = (
    DocumentStatus.PARTIAL,
    DocumentStatus.FAILED,
)

# A doc is considered in-flight (owned by another worker) when its status is one
# of these AND its processing_started_at is within the freshness window.
_INFLIGHT_STATUSES = {
    DocumentStatus.PROCESSING,
    DocumentStatus.PENDING,
    DocumentStatus.PENDING_FETCH,
}


class DocVerifyTaskHandler(TaskHandlerPlugin):
    """Task handler for ``doc_verify`` — on-demand heal + scheduled reconcile."""

    def get_task_type(self) -> str:
        return "doc_verify"

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        enabled: bool = v.environ(
            MEMORYLAYER_DOC_VERIFY_ENABLED,
            default=DEFAULT_MEMORYLAYER_DOC_VERIFY_ENABLED,
            type_fn=ext_parse_bool,
        )
        if not enabled:
            return None
        interval: int = v.environ(
            MEMORYLAYER_DOC_VERIFY_INTERVAL,
            default=DEFAULT_MEMORYLAYER_DOC_VERIFY_INTERVAL,
            type_fn=int,
        )
        # Empty payload => global sweep over all workspaces (mirrors how
        # decay_memories runs its all-workspaces pass with an empty payload).
        return TaskSchedule(interval_seconds=interval, default_payload={})

    async def handle(self, v: Variables, payload: dict) -> None:
        """Dispatch on-demand (document_id present) vs sweep (no document_id)."""
        logger = get_logger(name=self.get_task_type(), v=v)
        storage = get_extension(EXT_STORAGE_BACKEND, v)
        task_service = get_extension(EXT_TASK_SERVICE, v)

        payload = payload or {}
        workspace_id = payload.get("workspace_id", "")
        document_id = payload.get("document_id", "")

        ttl = int(v.environ(
            MEMORYLAYER_INGEST_INFLIGHT_TTL,
            default=DEFAULT_MEMORYLAYER_INGEST_INFLIGHT_TTL,
        ))

        if document_id:
            if not workspace_id:
                logger.error(
                    "doc_verify on-demand payload missing workspace_id: %s", payload,
                )
                return
            doc = await self._get_document(storage, document_id, workspace_id)
            if doc is None:
                logger.info(
                    "doc_verify: doc %s (ws=%s) not found — NO-OP",
                    document_id, workspace_id,
                )
                return
            outcome, fact_resumed = await self._verify_one(
                v, storage, task_service, doc, ttl, 0, logger,
            )
            logger.info(
                "doc_verify: on-demand doc %s — outcome=%s fact_resumed=%d",
                document_id, outcome, fact_resumed,
            )
            return

        # ---- Scheduled sweep -------------------------------------------
        if workspace_id:
            workspace_ids = [workspace_id]
        else:
            workspace_ids = await self._list_workspace_ids(storage, logger)

        limit: int = int(v.environ(
            MEMORYLAYER_DOC_VERIFY_BATCH_LIMIT,
            default=DEFAULT_MEMORYLAYER_DOC_VERIFY_BATCH_LIMIT,
        ))
        max_attempts: int = int(v.environ(
            MEMORYLAYER_DOC_VERIFY_MAX_ATTEMPTS,
            default=DEFAULT_MEMORYLAYER_DOC_VERIFY_MAX_ATTEMPTS,
        ))

        swept = resumed = fact_resumed = skipped = healed = terminal = 0
        for ws in workspace_ids:
            try:
                candidates = await self._collect_sweep_candidates(
                    storage, ws, ttl, limit, logger,
                )
            except Exception as exc:  # noqa: BLE001 - one ws must not abort sweep
                logger.error(
                    "doc_verify: failed to collect candidates for ws=%s; "
                    "skipping: %s", ws, exc, exc_info=True,
                )
                continue

            for doc in candidates:
                swept += 1
                try:
                    outcome, n_facts = await self._verify_one(
                        v, storage, task_service, doc, ttl, max_attempts, logger,
                    )
                    if outcome == "resumed":
                        resumed += 1
                    elif outcome == "healed":
                        healed += 1
                    elif outcome == "terminal":
                        terminal += 1
                    else:
                        skipped += 1
                    fact_resumed += n_facts
                except Exception as exc:  # noqa: BLE001 - non-fatal per doc
                    skipped += 1
                    logger.error(
                        "doc_verify: error verifying doc %s (ws=%s); skipping: %s",
                        doc.id, ws, exc, exc_info=True,
                    )

        logger.info(
            "doc_verify: sweep complete over %d workspace(s): %d swept, %d resumed, "
            "%d healed, %d terminal, %d fact-decompositions re-scheduled, "
            "%d skipped (limit=%d/status/ws, max_attempts=%d)",
            len(workspace_ids), swept, resumed, healed, terminal,
            fact_resumed, skipped, limit, max_attempts,
        )

    # ------------------------------------------------------------------
    # Core: verify a single document
    # ------------------------------------------------------------------

    async def _verify_one(
        self, v, storage, task_service, doc, ttl: int, max_attempts: int, logger,
    ) -> tuple[str, int]:
        """Heal one document: resume gap-fill if needed; re-drive fact gaps.

        Returns ``(outcome, fact_resumed_count)`` where ``outcome`` is one of:
        ``"resumed"`` (claimed + phase scheduled), ``"healed"`` (mislabeled doc
        promoted to COMPLETED), ``"terminal"`` (doc capped + marked FAILED),
        ``"skip"`` (nothing to do, claim lost, or in-flight).  A fresh in-flight
        doc is left untouched (returns ``("skip", 0)``).

        The on-demand path does NOT enforce the attempt cap (an operator calling
        doc_verify on-demand explicitly wants to force a re-try).  The sweep path
        passes ``max_attempts`` > 0; the on-demand path passes ``max_attempts=0``
        to opt out.
        """
        # In-flight protection: a fresh PROCESSING/PENDING* doc is owned by
        # another worker — leave it alone. Past the TTL it is an orphan we heal.
        if doc.status in _INFLIGHT_STATUSES and _is_fresh(doc, ttl):
            logger.debug(
                "doc_verify: doc %s in-flight (status=%s, fresh) — skip",
                doc.id, doc.status,
            )
            return "skip", 0

        meta = dict(doc.metadata or {})

        # Sweep-only: enforce per-doc cross-run reprocess cap.  An operator
        # calling on-demand (max_attempts=0) bypasses this so a manual re-try
        # always proceeds regardless of the counter.
        if max_attempts > 0:
            if meta.get("reconcile_terminal"):
                logger.debug(
                    "doc_verify: doc %s already marked terminal — skip", doc.id,
                )
                return "skip", 0
            attempts = int(meta.get("reprocess_attempts", 0) or 0)
            if attempts >= max_attempts:
                await self._mark_terminal(
                    storage, doc, meta,
                    reason="exceeded reprocess cap (%d attempts)" % attempts,
                    logger=logger,
                )
                return "terminal", 0

        gaps = await analyze_document_gaps(v, storage, doc)

        # Self-heal: gap-analysis finds the doc complete but its status is not
        # COMPLETED (e.g. a crash between finalize and the status write left a
        # stale FAILED/PARTIAL).  Promote to COMPLETED rather than re-resuming.
        if gaps.is_complete and doc.status != DocumentStatus.COMPLETED:
            await storage.update_document(
                doc.id,
                status=DocumentStatus.COMPLETED.value,
                processing_completed_at=datetime.now(timezone.utc),
            )
            logger.info(
                "doc_verify: doc %s gap-complete but status=%s — healed to COMPLETED",
                doc.id, getattr(doc.status, "value", doc.status),
            )
            return "healed", 0

        outcome = "skip"
        if not gaps.is_complete:
            # Atomically CLAIM the doc (conditional UPDATE to PROCESSING +
            # re-stamp processing_started_at) before resuming. The CAS closes the
            # read-then-write race the freshness check leaves open: a concurrent
            # worker (another sweep, or doc_added) that already claimed this doc
            # wins and we skip the resume (NO-OP for the phase). Fact-gap re-drive
            # below still runs — it is additive and independent of the claim.
            claimed = await storage.try_claim_document(doc.id, doc.workspace_id, ttl)
            if claimed:
                # Increment and persist the cross-run attempt counter so the cap
                # check above eventually fires for a permanently-broken doc.
                meta["reprocess_attempts"] = int(meta.get("reprocess_attempts", 0) or 0) + 1
                await storage.update_document(doc.id, metadata=meta)

                job_id = "job_%s" % uuid.uuid4().hex[:12]
                job = IngestionJob(
                    id=job_id,
                    workspace_id=doc.workspace_id,
                    document_ids=[doc.id],
                    status=JobStatus.QUEUED,
                )
                await storage.create_job(job)

                # resume_at attaches the embed RetryPolicy itself when the
                # resumed phase is document_embed, so a re-driven embed retries
                # across a transient outage (consistent with Part A3).
                await resume_at(
                    v, task_service, gaps.first_missing_phase, doc, job_id, logger,
                )
                outcome = "resumed"
                logger.info(
                    "doc_verify: resumed doc %s at phase=%s (job %s, attempt %d)",
                    doc.id, gaps.first_missing_phase, job_id,
                    meta["reprocess_attempts"],
                )
            else:
                logger.debug(
                    "doc_verify: lost claim race for doc %s (owned by another "
                    "worker) — skip resume", doc.id,
                )

        # Fact-gap re-drive is additive / non-blocking: it never claims the doc
        # or affects its status, and runs whether or not a phase was resumed (the
        # store gap and the fact gap are independent). Best-effort.
        fact_resumed = 0
        try:
            fact_resumed = await self._redrive_fact_gaps(
                storage, task_service, doc, logger,
            )
        except Exception as exc:  # noqa: BLE001 - fact gaps must never fail a doc
            logger.warning(
                "doc_verify: fact-gap re-drive failed for doc %s (best-effort): %s",
                doc.id, exc, exc_info=True,
            )

        return outcome, fact_resumed

    async def _mark_terminal(
        self, storage, doc, meta: dict, reason: str, logger,
    ) -> None:
        """Mark a doc terminally FAILED so it is excluded from all future sweeps."""
        meta = dict(meta)
        meta["reconcile_terminal"] = True
        meta["reconcile_terminal_reason"] = reason
        await storage.update_document(
            doc.id,
            status=DocumentStatus.FAILED.value,
            processing_completed_at=datetime.now(timezone.utc),
            metadata=meta,
        )
        logger.warning(
            "doc_verify: doc %s marked terminally FAILED (%s)", doc.id, reason,
        )

    async def _converge_enrichment_status(
        self, storage, doc, missing_parents, logger,
    ) -> None:
        """Advance the document's knowledge phase toward its true value.

        This is where PENDING becomes COMPLETE. Nothing emits a "document fully
        enriched" event — decomposition fans out per memory and each task only
        knows about itself — so the phase is re-derived here from stored state
        on every sweep. That makes it self-healing: a crashed worker, a lost
        event or a replay all converge on the next pass instead of stranding a
        document as permanently PENDING.

        Writes only on change, so a settled document costs one comparison.
        Best-effort: this is reporting, and it must not break the sweep's actual
        job of re-driving decomposition.
        """
        try:
            previous = getattr(doc, "enrichment_status", None)
            status, still = resolve_enrichment_status(doc, missing_parents)
            if status == previous:
                return
            await storage.update_document(doc.id, enrichment_status=status.value)
            doc.enrichment_status = status
            logger.info(
                "doc_verify: doc %s knowledge phase %s -> %s (%d outstanding)",
                doc.id, getattr(previous, "value", previous), status.value, len(still),
            )
        except Exception as exc:  # noqa: BLE001 - reporting, not the sweep's job
            logger.warning(
                "doc_verify: could not converge knowledge phase for doc %s: %s",
                doc.id, exc,
            )

    async def _redrive_fact_gaps(
        self, storage, task_service, doc, logger,
    ) -> int:
        """Re-schedule ``decompose_facts`` for page composites missing facts.

        Mirrors the producer call ``enqueue_post_store`` makes:
        ``schedule_task("decompose_facts", {memory_id, workspace_id, job_id})``.
        Idempotent — the handler early-returns on an already-atomic or
        already-archived parent — so re-driving is safe.
        """
        missing_parents = await analyze_fact_gaps(storage, doc)
        await self._converge_enrichment_status(storage, doc, missing_parents, logger)
        for memory_id in missing_parents:
            await task_service.schedule_task(
                "decompose_facts",
                {
                    "memory_id": memory_id,
                    "workspace_id": doc.workspace_id,
                    "job_id": None,
                },
            )
        if missing_parents:
            logger.info(
                "doc_verify: re-scheduled decompose_facts for %d composite(s) of "
                "doc %s", len(missing_parents), doc.id,
            )
        return len(missing_parents)

    # ------------------------------------------------------------------
    # Sweep candidate collection
    # ------------------------------------------------------------------

    async def _collect_sweep_candidates(
        self, storage, workspace_id: str, ttl: int, limit: int, logger,
    ) -> list:
        """Gather PARTIAL/FAILED + stale-PROCESSING docs for a workspace.

        Bounded by ``limit`` per status (no silent truncation — a full batch is
        logged so an operator can tell the sweep is saturated). De-dupes by id.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl)

        seen: set[str] = set()
        candidates: list = []

        statuses = list(_SWEEP_TERMINAL_STATUSES) + [DocumentStatus.PROCESSING]
        for st in statuses:
            docs, total = await storage.list_documents(
                workspace_id, status=st.value, limit=limit,
            )
            if total > len(docs):
                logger.info(
                    "doc_verify: ws=%s status=%s has %d docs; processing first %d "
                    "(MEMORYLAYER_DOC_VERIFY_BATCH_LIMIT) this run",
                    workspace_id, st.value, total, len(docs),
                )
            for doc in docs:
                # Stale-PROCESSING only: a FRESH PROCESSING doc is in-flight.
                if st == DocumentStatus.PROCESSING and _is_fresh(doc, ttl):
                    continue
                # Never re-scan a doc already marked terminally failed.
                if (doc.metadata or {}).get("reconcile_terminal"):
                    continue
                if doc.id in seen:
                    continue
                seen.add(doc.id)
                candidates.append(doc)
        return candidates

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _list_workspace_ids(storage, logger) -> list:
        """All workspace ids for the global sweep (mirrors decay_all_workspaces)."""
        lister = getattr(storage, "list_all_workspace_ids", None)
        if lister is None:
            logger.warning(
                "doc_verify: storage has no list_all_workspace_ids; global sweep "
                "is a no-op (schedule a per-workspace payload instead)",
            )
            return []
        return await lister()

    @staticmethod
    async def _get_document(storage, doc_id: str, workspace_id: str):
        """Resolve a Document, treating not-found as absence."""
        try:
            return await storage.get_document(doc_id, workspace_id)
        except FileNotFoundError:
            return None
        except KeyError:
            return None
