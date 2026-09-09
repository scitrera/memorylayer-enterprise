"""Task handler for document transcription phase.

Phase 2 of the distributed ingestion pipeline:
- Load page records from DB
- Reload image data from blob storage for pages that need transcription
- Call the configured transcription service for VLM/OCR (embed-server or the
  in-process cascade against model endpoints)
- Update page records with transcripts via storage.update_page()
- Schedule document_embed task
- Update job progress to 40%
"""
import base64
from datetime import datetime, timezone
from typing import Optional

from scitrera_app_framework import Variables, get_extension

from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
from memorylayer_server.services.tasks.base import TaskSchedule, EXT_TASK_SERVICE
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

from memorylayer_saas.models.document import DocumentPage, DocumentStatus, JobStatus
from memorylayer_saas.services.document import (
    get_document_ingestion_service,
    EXT_BLOB_STORAGE_SERVICE,
)
from memorylayer_saas.services.document.page_figures import (
    PAGE_FIGURES_METADATA_KEY,
    store_page_figures,
)
from memorylayer_saas.services.transcription import EXT_TRANSCRIPTION_SERVICE
# Shared page-batch size: process image-derived transcription in fixed-size page
# batches so peak memory is O(batch pages), not O(all pages). Same constant used
# by document_embed.py — single source of truth in ingestion_service.
from memorylayer_saas.services.document.ingestion_service import _RENDER_BATCH_PAGES

# Reuse the single-source-of-truth embed RetryPolicy so the document_embed task
# scheduled here retries across a multi-minute embed-backend outage.
from memorylayer_saas.tasks.document_embed import build_embed_retry_policy


class DocumentTranscribeTaskHandler(TaskHandlerPlugin):
    """Task handler for transcribing document page images to text.

    Loads pages from DB, fetches image data from blob storage, calls the
    embed server for transcription, and persists results via update_page().
    """

    def get_task_type(self) -> str:
        return "document_transcribe"

    async def handle(self, v: Variables, payload: dict) -> None:
        """Execute the transcription phase of document ingestion.

        Args:
            v: Variables instance.
            payload: Dict with document_id, job_id, workspace_id.
        """
        document_id = payload["document_id"]
        job_id = payload["job_id"]
        workspace_id = payload["workspace_id"]

        storage = get_extension(EXT_STORAGE_BACKEND, v)
        task_service = get_extension(EXT_TASK_SERVICE, v)
        transcription = get_extension(EXT_TRANSCRIPTION_SERVICE, v)
        blob_storage = get_extension(EXT_BLOB_STORAGE_SERVICE, v)
        ingestion_service = get_document_ingestion_service(v)

        try:
            doc = await storage.get_document(document_id, workspace_id)
            if not doc:
                raise ValueError("Document not found: %s" % document_id)

            pages = await storage.get_pages(document_id, workspace_id)
            if not pages:
                raise ValueError("No pages found for document: %s" % document_id)

            # Identify pages needing transcription (PDF images without transcripts)
            pages_needing_transcription = [
                p for p in pages if p.transcript is None and p.image_storage_path
            ]

            if pages_needing_transcription:
                system_prompt = doc.extraction_options.system_prompt

                # NOTE: `transcription` is a PROCESS-WIDE extension
                # (get_extension above), shared by every concurrent
                # document_transcribe task. This block used to wrap the work
                # in try/finally: transcription.close(), which cascades to
                # provider.aclose() and nulls the shared httpx client -- so
                # the first task to finish killed in-flight requests belonging
                # to its siblings (surfacing as ReadError with no message,
                # against a server logging clean 200s) and left the client
                # unusable for everything after. connect() stays: it is a
                # no-op when connected and heals a client someone else closed.
                await transcription.connect()
                # Transcribe in fixed-size batches so only ~_RENDER_BATCH_PAGES
                # pages' image bytes are resident at once. Building the full
                # images_b64 list upfront was an O(all-pages) memory spike; we
                # re-read each batch's PNGs from blob and release before the
                # next batch. The embed server numbers pages by their position
                # WITHIN the request (0-based per call), so results are mapped
                # back by request-relative index (batch[req_idx]), not by the
                # page's global page_no — which no longer equals the request
                # index once pages are sent in batches.
                for b_start in range(
                    0, len(pages_needing_transcription), _RENDER_BATCH_PAGES
                ):
                    batch = pages_needing_transcription[
                        b_start:b_start + _RENDER_BATCH_PAGES
                    ]

                    # Reload base64 image data from blob storage for this batch
                    images_b64 = []
                    for page in batch:
                        img_bytes = await blob_storage.retrieve_file(
                            page.image_storage_path,
                        )
                        images_b64.append(
                            base64.b64encode(img_bytes).decode("ascii")
                        )

                    page_results = await transcription.transcribe_pages(
                        images_b64, system_prompt=system_prompt,
                    )

                    for page_result in page_results:
                        req_idx = page_result.request_index
                        if not (0 <= req_idx < len(batch)):
                            continue
                        page = batch[req_idx]
                        page.transcript = page_result.content
                        page.transcript_model = page_result.model

                        # Store transcript to blob storage
                        transcript_path = blob_storage.page_transcript_path(
                            doc.workspace_id, doc.id, page.page_no,
                        )
                        await blob_storage.store_file(
                            transcript_path,
                            page.transcript.encode("utf-8"),
                        )

                        # Figures carry no text, so the crop is the only
                        # representation of them; the [figure N] markers in
                        # the transcript above index these records.
                        figures = await store_page_figures(
                            blob_storage=blob_storage,
                            workspace_id=doc.workspace_id,
                            doc_id=doc.id,
                            page_no=page.page_no,
                            page_image_b64=images_b64[req_idx],
                            regions=page_result.regions,
                            captions=page_result.figure_captions,
                            logger=ingestion_service.logger,
                        )

                        # Update page record in DB. The figure records ride
                        # in metadata so the stored crops are discoverable —
                        # a blob nobody can enumerate is unfetchable.
                        page_updates = {
                            "transcript": page.transcript,
                            "transcript_model": page.transcript_model,
                        }
                        if figures:
                            page.metadata = {
                                **(page.metadata or {}),
                                PAGE_FIGURES_METADATA_KEY: figures,
                            }
                            page_updates["metadata"] = page.metadata
                        await storage.update_page(page.id, **page_updates)

                    # Release this batch's image bytes before the next batch.
                    del images_b64

            # Also update text/markdown pages that already have transcripts
            # (persisted in render phase without transcripts for these types)
            for page in pages:
                if page.transcript and page not in pages_needing_transcription:
                    await storage.update_page(
                        page.id,
                        transcript=page.transcript,
                        transcript_model=page.transcript_model,
                    )

            transcribed = sum(1 for p in pages if p.transcript)
            ingestion_service.logger.info(
                "Transcribe phase complete for document %s: %d/%d pages transcribed",
                document_id, transcribed, len(pages),
            )

            await storage.update_job(job_id, progress_percent=40)

            await task_service.schedule_task(
                "document_embed",
                {
                    "document_id": document_id,
                    "job_id": job_id,
                    "workspace_id": workspace_id,
                },
                priority=3,
                retry_policy=build_embed_retry_policy(),
            )

        except Exception as exc:
            ingestion_service.logger.error(
                "Transcribe phase failed for document %s: %s",
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
