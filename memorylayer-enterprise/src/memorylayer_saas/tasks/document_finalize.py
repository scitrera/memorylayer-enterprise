"""Task handler for document finalization phase.

Phase 4 (final) of the distributed ingestion pipeline:
- Load completed pages from DB (with transcripts and multivector embeddings)
- Create memories from pages using pre-computed embeddings stored in page metadata
- Update document status to COMPLETED
- Update job to 100% progress and COMPLETED status
"""
from datetime import datetime, timezone
from typing import Optional

from scitrera_app_framework import Variables, get_extension

from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
from memorylayer_server.services.tasks.base import TaskSchedule
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

from memorylayer_saas.models.document import DocumentStatus, JobStatus
from memorylayer_saas.services.document import get_document_ingestion_service
from memorylayer_saas.tasks.doc_added import _emit_event


class DocumentFinalizeTaskHandler(TaskHandlerPlugin):
    """Task handler for the final phase of document ingestion.

    Loads completed pages from DB, creates memories with pre-computed embeddings,
    and updates document + job status to reflect completion.

    Single-vector embeddings are read from the dedicated ``page.embedding``
    column (populated by the embed phase via ``_page_model_to_domain``).
    """

    def get_task_type(self) -> str:
        return "document_finalize"

    async def handle(self, v: Variables, payload: dict) -> None:
        """Execute the finalization phase of document ingestion.

        Args:
            v: Variables instance.
            payload: Dict with document_id, job_id, workspace_id.
        """
        document_id = payload["document_id"]
        job_id = payload["job_id"]
        workspace_id = payload["workspace_id"]

        storage = get_extension(EXT_STORAGE_BACKEND, v)
        ingestion_service = get_document_ingestion_service(v)

        try:
            doc = await storage.get_document(document_id, workspace_id)
            if not doc:
                raise ValueError("Document not found: %s" % document_id)

            pages = await storage.get_pages(document_id, workspace_id)
            if not pages:
                raise ValueError("No pages found for document: %s" % document_id)

            # The single-vector embedding now lives in the dedicated
            # ``page.embedding`` column (populated by ``_page_model_to_domain``),
            # so it is already set when ``create_memories_for_pages`` runs.
            # Transition fallback: for any not-yet-migrated row, restore it from
            # the legacy ``metadata['_embedding']`` stash. Removable once the P2.4
            # migration is universally applied.
            for page in pages:
                if page.embedding is None:
                    emb = page.metadata.get("_embedding")
                    if emb is not None:
                        page.embedding = emb

            # Delegate to the shared create + post-store path so this task and the
            # inline ingestion path build identical RememberInputs (fixing the
            # latent context_id sentinel bug) AND enqueue the same post-store
            # enrichment lifecycle (decompose + associations + KG + tiering).
            memory_ids = await ingestion_service.create_memories_for_pages(
                doc, pages, job_id=job_id,
            )

            ingestion_service.logger.info(
                "Finalize phase complete for document %s: %d memories created",
                document_id, len(memory_ids),
            )

            await ingestion_service._finalize(doc, memory_ids, job_id)

            # Signal KB / downstream coalesce that ingestion finished. The inline
            # doc_added path emits this itself; the chained + gap-fill paths land
            # here, so emit it here too or the per-workspace kb-coalesce never
            # fires a kb_update for chained ingests. Best-effort (never raises).
            await _emit_event(
                v, workspace_id, "memorylayer.ingest_complete",
                {
                    "ml_doc_id": document_id,
                    "vfs_ref": getattr(doc, "source_vfs_ref", None) or "",
                },
                ingestion_service.logger,
            )

        except Exception as exc:
            ingestion_service.logger.error(
                "Finalize phase failed for document %s: %s",
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
