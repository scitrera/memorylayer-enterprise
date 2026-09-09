"""Task handler for the ingestion-job orphan-reconcile sweep.

A periodic safety net for the ingestion-job lifecycle. Create-time coalescing
(``ingestion_service._supersede_active_jobs`` on the upload/reprocess paths) and
reconcile-on-complete (``ingestion_service._finalize``) are the primary defenses
against ``ingestion_jobs`` rows accumulating and orphaning. This sweep is the
backstop for anything that slips past both: Aether task replays that mint a
fresh job, or a worker that marked a job ``running`` then died at 0% forever.

Each run cancels every ``queued``/``running`` ingestion job whose referenced
documents are ALL already ``completed`` (delegated to the storage backend as a
single set-based operation -- see ``cancel_orphaned_ingestion_jobs``). A job
that references a still-processing document, or a document that no longer
exists, is left untouched.

Registered as a recurring schedule (``MEMORYLAYER_JOB_RECONCILE_INTERVAL_SEC``)
mirroring ``tasks/blob_gc.py``. It is workspace-agnostic: the backing UPDATE is
keyed off the documents table directly, so no per-workspace fan-out is needed.
"""
from typing import Optional

from scitrera_app_framework import Variables, ext_parse_bool, get_extension, get_logger

from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
from memorylayer_server.services.tasks.base import TaskSchedule
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

from memorylayer_saas.config import (
    MEMORYLAYER_JOB_RECONCILE_ENABLED,
    DEFAULT_MEMORYLAYER_JOB_RECONCILE_ENABLED,
    MEMORYLAYER_JOB_RECONCILE_INTERVAL_SEC,
    DEFAULT_MEMORYLAYER_JOB_RECONCILE_INTERVAL_SEC,
)


class JobReconcileTaskHandler(TaskHandlerPlugin):
    """Periodic reconcile sweep that cancels doc-complete orphaned ingestion jobs."""

    def get_task_type(self) -> str:
        return "job_reconcile"

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        enabled: bool = v.environ(
            MEMORYLAYER_JOB_RECONCILE_ENABLED,
            default=DEFAULT_MEMORYLAYER_JOB_RECONCILE_ENABLED,
            type_fn=ext_parse_bool,
        )
        if not enabled:
            return None
        interval: int = v.environ(
            MEMORYLAYER_JOB_RECONCILE_INTERVAL_SEC,
            default=DEFAULT_MEMORYLAYER_JOB_RECONCILE_INTERVAL_SEC,
            type_fn=int,
        )
        return TaskSchedule(interval_seconds=interval, default_payload={})

    async def handle(self, v: Variables, payload: dict) -> None:
        """Run one reconcile sweep, cancelling doc-complete orphaned jobs.

        Args:
            v: Variables instance.
            payload: Unused (the sweep is global/workspace-agnostic).
        """
        logger = get_logger(name=self.get_task_type(), v=v)
        storage = get_extension(EXT_STORAGE_BACKEND, v)

        canceller = getattr(storage, "cancel_orphaned_ingestion_jobs", None)
        if canceller is None:
            logger.warning(
                "job_reconcile: storage backend has no cancel_orphaned_ingestion_jobs; "
                "NO-OP",
            )
            return

        try:
            cancelled = await canceller()
        except NotImplementedError:
            logger.warning(
                "job_reconcile: storage backend does not implement orphan reconcile; "
                "NO-OP",
            )
            return

        logger.info(
            "job_reconcile: sweep complete -- cancelled %d orphaned ingestion job(s)",
            cancelled,
        )
