"""Task handler for document render phase.

Phase 1 of the distributed ingestion pipeline:
- Mark document as PROCESSING
- Render pages from source document (PDF -> PNG images, text -> text pages)
- Persist page records to DB via storage.create_page()
- Schedule document_transcribe task
- Update job progress to 20%
"""
from datetime import datetime, timezone
from typing import Optional

from scitrera_app_framework import Variables, ext_parse_bool, get_extension

from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
from memorylayer_server.services.tasks.base import TaskSchedule, EXT_TASK_SERVICE
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

from memorylayer_saas.config import (
    DEFAULT_MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
    MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
)
from memorylayer_saas.models.document import DocumentStatus, JobStatus
from memorylayer_saas.services.document import get_document_ingestion_service
# Reuse the single-source-of-truth embed RetryPolicy so the document_embed task
# scheduled here (transcribe-disabled skip path) retries across a multi-minute
# embed-backend outage instead of Aether's useless default re-pend.
from memorylayer_saas.tasks.document_embed import build_embed_retry_policy


class DocumentRenderTaskHandler(TaskHandlerPlugin):
    """Task handler for rendering document pages from source files.

    Triggered once per uploaded/reprocessed document. Marks the document as
    PROCESSING, renders pages, persists page records, and schedules the
    transcription phase.
    """

    def get_task_type(self) -> str:
        return "document_render"

    async def handle(self, v: Variables, payload: dict) -> None:
        """Execute the render phase of document ingestion.

        Args:
            v: Variables instance.
            payload: Dict with document_id, job_id, workspace_id.
        """
        document_id = payload["document_id"]
        job_id = payload["job_id"]
        workspace_id = payload["workspace_id"]

        storage = get_extension(EXT_STORAGE_BACKEND, v)
        task_service = get_extension(EXT_TASK_SERVICE, v)
        ingestion_service = get_document_ingestion_service(v)

        try:
            now = datetime.now(timezone.utc)
            await storage.update_document(
                document_id,
                status=DocumentStatus.PROCESSING.value,
                processing_started_at=now,
            )
            await storage.update_job(
                job_id,
                status=JobStatus.RUNNING.value,
                started_at=now,
            )

            doc = await storage.get_document(document_id, workspace_id)
            if not doc:
                raise ValueError("Document not found: %s" % document_id)

            pages = await ingestion_service._render_pages(doc)

            await storage.update_document(document_id, page_count=len(pages))

            # Idempotency: on a re-run / gap-fill, some page rows may already
            # exist. Creating them again violates uq_document_page
            # (document_id, page_no), so create only the MISSING page_nos.
            existing_pages = await storage.get_pages(document_id, workspace_id)
            existing_page_nos = {p.page_no for p in existing_pages}

            # Persist page records with image paths (no transcript/embedding yet)
            persisted_pages = []
            for page in pages:
                if page.page_no in existing_page_nos:
                    continue
                persisted = await storage.create_page(
                    workspace_id=workspace_id,
                    document_id=document_id,
                    page=page,
                )
                persisted_pages.append(persisted)

            await storage.update_job(job_id, progress_percent=20)

            ingestion_service.logger.info(
                "Render phase complete for document %s: %d pages rendered",
                document_id, len(persisted_pages),
            )

            # Skip the transcribe phase when disabled — go straight to embed.
            transcribe_enabled = v.environ(
                MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
                default=DEFAULT_MEMORYLAYER_DOCUMENT_TRANSCRIBE_ENABLED,
                type_fn=ext_parse_bool,
            )
            next_phase = "document_transcribe" if transcribe_enabled else "document_embed"
            if not transcribe_enabled:
                ingestion_service.logger.info(
                    "Transcribe phase disabled; scheduling document_embed for %s", document_id,
                )
            # Attach the embed RetryPolicy only when scheduling document_embed
            # directly (transcribe-disabled skip); the transcribe phase attaches
            # it itself when it schedules embed.
            retry_policy = build_embed_retry_policy() if next_phase == "document_embed" else None
            await task_service.schedule_task(
                next_phase,
                {
                    "document_id": document_id,
                    "job_id": job_id,
                    "workspace_id": workspace_id,
                },
                priority=3,
                retry_policy=retry_policy,
            )

        except Exception as exc:
            ingestion_service.logger.error(
                "Render phase failed for document %s: %s",
                document_id, exc, exc_info=True,
            )
            now = datetime.now(timezone.utc)
            await storage.update_document(
                document_id,
                status=DocumentStatus.FAILED.value,
                processing_completed_at=now,
            )
            await storage.update_job(
                job_id,
                status=JobStatus.FAILED.value,
                completed_at=now,
                errors=[{"document_id": document_id, "error": str(exc)}],
            )

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        return None
