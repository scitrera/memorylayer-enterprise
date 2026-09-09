"""Task handler for document embedding phase.

Phase 3 of the distributed ingestion pipeline:
- Load page records from DB (now with transcripts)
- Generate single-vector and multi-vector embeddings via embed server
- Update page records with embeddings via storage.update_page()
- Schedule document_finalize task
- Update job progress to 70%

Single-vector embeddings are written to the dedicated ``document_pages.embedding``
column via ``storage.update_page(page.id, embedding=...)``; the finalize phase
reads them back off ``page.embedding`` (populated by ``_page_model_to_domain``)
before creating memories. This replaces the brittle ``metadata['_embedding']``
JSON stash (idempotent-ingestion design P2.4).

The text, multi-vector (ColPali) and visual-tokenizer image-embed signals are
additive: a per-signal failure must not abort the (additive) phase or mark the
job FAILED. To keep partial/silent degradation observable, the phase tracks a
per-signal outcome (attempted / succeeded counts + any error), emits a single
greppable INFO summary line, and appends non-fatal entries to the ingestion
job's ``errors`` list for any signal that failed or produced 0/N when pages were
eligible.
"""
import base64
from datetime import datetime, timezone
from typing import Optional

from scitrera_app_framework import Variables, ext_parse_bool, get_extension

from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
from memorylayer_server.services.tasks.base import TaskSchedule, EXT_TASK_SERVICE
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

from memorylayer_saas.config import (
    DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED,
    DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_MAX_TOKENS,
    DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_PROMPT,
    DEFAULT_MEMORYLAYER_VISUAL_TOKENIZER_ENABLED,
    MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED,
    MEMORYLAYER_DOCUMENT_CHAT_INGEST_MAX_TOKENS,
    MEMORYLAYER_DOCUMENT_CHAT_INGEST_PROMPT,
    MEMORYLAYER_VISUAL_TOKENIZER_ENABLED,
)
from memorylayer_saas.models.document import DocumentStatus, JobStatus
from memorylayer_saas.services.document import (
    get_document_ingestion_service,
    EXT_EMBED_SERVER_CLIENT,
    EXT_BLOB_STORAGE_SERVICE,
)
# Shared page-batch size: process image-derived embed signals in fixed-size page
# batches so peak memory is O(batch pages), not O(all pages). See the render
# phase (the whole-PDF render OOM) — this is the embed-side counterpart.
from memorylayer_saas.services.document.ingestion_service import _RENDER_BATCH_PAGES
from memorylayer_saas.services.document.image_embed import (
    generate_page_text_from_image_embeds,
    precompute_and_store_image_embeds,
)
from memorylayer_saas.services.document.inference_client import (
    default_inference_model,
    get_inference_client,
    slugify_model,
)


class RetryableEmbedError(Exception):
    """Raised when the embed phase attempted work but EVERY signal failed.

    Signals the ``all-signals-failed`` transient case (e.g. the embed backend is
    down → multivector=0/N, text=0/N, image_embeds=0/N): the outage is almost
    certainly temporary, so the phase must NOT finalize the document as failed.
    Raising propagates to ``AetherTaskService._handle_task_assignment`` →
    ``fail_task()`` → the Aether server auto-reschedules the ``document_embed``
    task per the ``EMBED_RETRY_POLICY`` attached at scheduling time, spanning a
    multi-minute backend outage. A PARTIAL success (some signal produced >0) does
    NOT raise — it proceeds to finalize as before.
    """


def build_embed_retry_policy():
    """Build the ``document_embed`` Aether RetryPolicy (or ``None`` if the SDK lacks it).

    EXPONENTIAL backoff, ``max_attempts=6``, ``initial_delay_ms=5000``,
    ``max_delay_ms=300000`` (5 min), ``jitter_factor=0.2`` — so the automatic
    within-task retries span a multi-minute embed-backend outage (Aether's
    built-in default of a few immediate re-pends is useless for a transient
    backend outage). Defined here as the single source of truth so the doc_verify
    sweep (via resume_at) can reuse the exact same policy when it re-drives an
    embed phase.

    Returns ``None`` when the installed ``scitrera-aether-client`` predates the
    ``RetryPolicy`` proto (version drift); callers thread the ``None`` through
    unchanged (schedule_task's retry_policy=None == today's behavior).
    """
    try:
        from scitrera_aether_client.proto import aether_pb2
    except Exception:  # noqa: BLE001 - SDK layout / import guard
        return None
    if not hasattr(aether_pb2, "RetryPolicy") or not hasattr(aether_pb2, "BackoffStrategy"):
        return None
    return aether_pb2.RetryPolicy(
        max_attempts=6,
        backoff=aether_pb2.BackoffStrategy.BACKOFF_STRATEGY_EXPONENTIAL,
        initial_delay_ms=5000,
        max_delay_ms=300000,
        jitter_factor=0.2,
    )


class DocumentEmbedTaskHandler(TaskHandlerPlugin):
    """Task handler for generating embeddings for transcribed pages.

    Loads pages from DB, calls the embed server for single-vector text
    embeddings and multi-vector image embeddings, and persists via update_page().
    Single-vector embeddings are written to the dedicated ``embedding`` column
    for use in the finalize phase.
    """

    def get_task_type(self) -> str:
        return "document_embed"

    async def handle(self, v: Variables, payload: dict) -> None:
        """Execute the embedding phase of document ingestion.

        Args:
            v: Variables instance.
            payload: Dict with document_id, job_id, workspace_id.
        """
        document_id = payload["document_id"]
        job_id = payload["job_id"]
        workspace_id = payload["workspace_id"]

        storage = get_extension(EXT_STORAGE_BACKEND, v)
        task_service = get_extension(EXT_TASK_SERVICE, v)
        embed_client = get_extension(EXT_EMBED_SERVER_CLIENT, v)
        blob_storage = get_extension(EXT_BLOB_STORAGE_SERVICE, v)
        ingestion_service = get_document_ingestion_service(v)

        try:
            pages = await storage.get_pages(document_id, workspace_id)
            if not pages:
                raise ValueError("No pages found for document: %s" % document_id)

            # Text embeddings need transcripts; image-derived signals (multi-vector
            # + visual-tokenizer image-embeds) do NOT — they run whenever pages
            # have rendered images, independent of whether transcription ran.
            # Image-derived signals run FIRST because document-chat ingestion
            # (generating per-page memory text from image_embeds when OCR
            # transcription is disabled) depends on the image_embeds existing, and
            # the resulting text must then be single-vector embedded below.
            pages_with_images = [p for p in pages if p.image_storage_path]

            # Per-signal outcome tracking. Success counts reflect what actually
            # persisted (not merely "no exception"); ``error`` holds the failure
            # string when a step raised. Used for the summary log + non-fatal
            # job-error entries below.
            # text_attempted is finalized after document-chat ingestion may add
            # transcripts; the others are sized up-front.
            text_attempted = 0
            text_succeeded = 0
            text_error: Optional[str] = None

            # Skip-guard (re-run/gap-fill perf): only (re)embed pages that do
            # not already carry a multivector. On a fresh ingest no page has one,
            # so all are attempted and behavior is unchanged; on a re-run the
            # already-embedded pages are skipped (the recompute overwrote with the
            # same value before — correctness-neutral, pure perf).
            pages_needing_mv = [p for p in pages_with_images if p.multivector is None]
            mv_attempted = len(pages_needing_mv)
            mv_succeeded = 0
            mv_error: Optional[str] = None

            # image_embeds / chat_ingest are only attempted when their feature is
            # enabled; left at attempted=0 otherwise so they are reported as
            # skipped (not failed) signals.
            ie_attempted = 0
            ie_succeeded = 0
            ie_error: Optional[str] = None

            ci_attempted = 0
            ci_succeeded = 0
            ci_error: Optional[str] = None

            doc = None

            # Document-chat ingestion needs per-page image_embeds to exist, so it
            # forces the precompute below even when the visual tokenizer flag is
            # off (read here so the precompute gate can see it).
            chat_ingest_enabled = v.environ(
                MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED,
                default=DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_ENABLED,
                type_fn=ext_parse_bool,
            )

            # NOTE: embed_client is a PROCESS-WIDE extension (get_extension
            # above), shared by every concurrent document_embed task and
            # connected by the framework at startup. This block used to wrap
            # the work in try/finally: embed_client.close() -- which tore the
            # shared client down for every OTHER in-flight task, and for all
            # future ones, since nothing reconnects it. connect() is kept
            # because it is a no-op when already connected and self-heals a
            # client someone else closed; close() is the framework's job.
            await embed_client.connect()

            # Image-derived signals (independent of transcripts).
            if pages_with_images:
                # Multi-vector (ColPali) embeddings — non-fatal: one signal's
                # failure must not abort the embed phase or block the
                # visual-tokenizer image-embeds below. Pages that already
                # carry a multivector (gap-fill/re-run) are skipped.
                if pages_needing_mv:
                    try:
                        # Batch so only ~_RENDER_BATCH_PAGES pages' image
                        # bytes are resident at once (reading every page into
                        # one images_b64 list was a memory spike).
                        for b_start in range(
                            0, len(pages_needing_mv), _RENDER_BATCH_PAGES
                        ):
                            mv_batch = pages_needing_mv[
                                b_start:b_start + _RENDER_BATCH_PAGES
                            ]
                            images_b64 = []
                            for page in mv_batch:
                                img_bytes = await blob_storage.retrieve_file(page.image_storage_path)
                                images_b64.append(base64.b64encode(img_bytes).decode("ascii"))

                            mv_results = await embed_client.embed_images_multivector(images_b64)
                            for page, mv in zip(mv_batch, mv_results):
                                await storage.update_page(page.id, multivector=mv["vectors"])
                                mv_succeeded += 1
                            # Release this batch's image bytes.
                            del images_b64
                    except Exception as mv_exc:  # noqa: BLE001 - one signal; non-fatal
                        mv_error = str(mv_exc)
                        ingestion_service.logger.warning(
                            "Multi-vector embedding failed for document %s (non-fatal): %s",
                            document_id, mv_exc,
                        )

                # Precompute per-page image-embeds (raw vision-tower output +
                # grid) for the visual-tokenizer / vLLM image_embeds chat
                # path. Additive; never fatal. Also the prerequisite for the
                # document-chat ingestion step below, so chat-ingest forces it
                # on even when the visual-tokenizer flag is off.
                vt_enabled = v.environ(
                    MEMORYLAYER_VISUAL_TOKENIZER_ENABLED,
                    default=DEFAULT_MEMORYLAYER_VISUAL_TOKENIZER_ENABLED,
                    type_fn=ext_parse_bool,
                )
                if vt_enabled or chat_ingest_enabled:
                    ie_attempted = len(pages_with_images)
                    try:
                        doc = await storage.get_document(document_id, workspace_id)
                        # Batch the precompute so it re-reads only one batch of
                        # page images at a time; sum the per-batch stored counts
                        # so ie_succeeded still reflects the whole document.
                        for b_start in range(
                            0, len(pages_with_images), _RENDER_BATCH_PAGES
                        ):
                            ie_batch = pages_with_images[
                                b_start:b_start + _RENDER_BATCH_PAGES
                            ]
                            ie_succeeded += await precompute_and_store_image_embeds(
                                embed_client=embed_client,
                                blob_storage=blob_storage,
                                storage=storage,
                                pages=ie_batch,
                                workspace_id=workspace_id,
                                document_id=document_id,
                                filename=getattr(doc, "filename", None),
                                logger=ingestion_service.logger,
                                source=getattr(doc, "source_vfs_ref", None),
                            )
                    except Exception as vt_exc:  # noqa: BLE001 - additive
                        ie_error = str(vt_exc)
                        ingestion_service.logger.warning(
                            "Image-embed precompute failed for document %s "
                            "(non-fatal): %s", document_id, vt_exc,
                        )

            # Document-chat ingestion: for image pages that still have no
            # transcript (OCR transcription disabled), generate the page's
            # memory text from its freshly-computed image_embeds via the
            # prompt-embeds inference LLM — the same mechanism as
            # /v1/documents/chat. The generated text is stored on the page so
            # the single-vector embedding (below) and memory creation (finalize
            # phase) proceed unchanged. Additive; never fatal. Pages without
            # image_embeds for the inference model slug are skipped.
            if chat_ingest_enabled:
                pages_needing_text = [p for p in pages_with_images if not p.transcript]
                if pages_needing_text:
                    ci_attempted = len(pages_needing_text)
                    try:
                        if doc is None:
                            doc = await storage.get_document(document_id, workspace_id)
                        inference_client = await get_inference_client(
                            v, ingestion_service.logger,
                        )
                        model = default_inference_model(v)
                        model_slug = slugify_model(model)
                        instruction = (
                            doc.extraction_options.system_prompt
                            or v.environ(
                                MEMORYLAYER_DOCUMENT_CHAT_INGEST_PROMPT,
                                default=DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_PROMPT,
                            )
                        )
                        max_tokens = int(v.environ(
                            MEMORYLAYER_DOCUMENT_CHAT_INGEST_MAX_TOKENS,
                            default=DEFAULT_MEMORYLAYER_DOCUMENT_CHAT_INGEST_MAX_TOKENS,
                        ))
                        for page in pages_needing_text:
                            text = await generate_page_text_from_image_embeds(
                                inference_client=inference_client,
                                blob_storage=blob_storage,
                                page=page,
                                model=model,
                                model_slug=model_slug,
                                filename=getattr(doc, "filename", None),
                                instruction=instruction,
                                max_tokens=max_tokens,
                                v=v,
                                logger=ingestion_service.logger,
                            )
                            if not text:
                                continue
                            page.transcript = text
                            await storage.update_page(
                                page.id, transcript=text, transcript_model=model,
                            )
                            ci_succeeded += 1
                    except Exception as ci_exc:  # noqa: BLE001 - additive
                        ci_error = str(ci_exc)
                        ingestion_service.logger.warning(
                            "Document-chat ingest failed for document %s "
                            "(non-fatal): %s", document_id, ci_exc,
                        )

            # Single-vector embeddings over pages that now have a transcript
            # (original transcripts + any chat-generated text) — written to the
            # dedicated 'embedding' column for retrieval in the finalize phase.
            # Skip-guard (re-run/gap-fill perf): a page that already carries an
            # 'embedding' is skipped (the recompute overwrote the same value
            # before — correctness-neutral). Fresh ingests have none, so all
            # transcribed pages are embedded and behavior is unchanged.
            pages_with_text = [
                p for p in pages
                if p.transcript and p.embedding is None
            ]
            text_attempted = len(pages_with_text)
            if pages_with_text:
                # Text is low-memory, but batch it too for embed-server
                # request-size safety and consistency with the image signals.
                for b_start in range(
                    0, len(pages_with_text), _RENDER_BATCH_PAGES
                ):
                    text_batch = pages_with_text[
                        b_start:b_start + _RENDER_BATCH_PAGES
                    ]
                    texts = [p.transcript for p in text_batch]
                    embeddings = await embed_client.embed_texts(texts)
                    for page, emb in zip(text_batch, embeddings):
                        await storage.update_page(page.id, embedding=emb)
                        text_succeeded += 1
            else:
                ingestion_service.logger.info(
                    "No transcripts for document %s; skipping text embeddings",
                    document_id,
                )

            # Single greppable summary of every signal's persisted outcome.
            ingestion_service.logger.info(
                "embed summary doc=%s: text=%d/%d multivector=%d/%d image_embeds=%d/%d chat_ingest=%d/%d",
                document_id,
                text_succeeded, text_attempted,
                mv_succeeded, mv_attempted,
                ie_succeeded, ie_attempted,
                ci_succeeded, ci_attempted,
            )

            # Transient-outage guard: if embedding work was ATTEMPTED but ALL of
            # it failed (total_attempted > 0 and total_succeeded == 0), the embed
            # backend is almost certainly down (the observed multivector=0/26
            # case). Do NOT proceed to finalize (which would freeze status=failed
            # + memories=0 and leave broken artifacts stuck). Raise
            # RetryableEmbedError so AetherTaskService.fail_task() lets Aether
            # auto-reschedule this task per EMBED_RETRY_POLICY, spanning a
            # multi-minute outage. A PARTIAL success (any signal produced >0)
            # falls through to finalize as before. Signals with attempted==0
            # (e.g. text=0/0 when transcription is disabled, or a disabled
            # feature) are NOT failures — they simply do not contribute to
            # total_attempted, so an all-skipped document is not treated as a
            # transient failure.
            total_attempted = mv_attempted + ie_attempted + text_attempted + ci_attempted
            total_succeeded = mv_succeeded + ie_succeeded + text_succeeded + ci_succeeded
            if total_attempted > 0 and total_succeeded == 0:
                ingestion_service.logger.warning(
                    "All embed signals failed for document %s (attempted=%d, "
                    "succeeded=0) — embed backend likely unavailable; will retry "
                    "via Aether RetryPolicy (not finalizing as failed)",
                    document_id, total_attempted,
                )
                raise RetryableEmbedError(
                    "all embed signals failed for %s (attempted=%d, succeeded=0) — "
                    "embed backend likely unavailable; retrying"
                    % (document_id, total_attempted)
                )

            # Record non-fatal degradation on the job so it is queryable: any
            # signal that raised, or that produced 0/N when pages were eligible,
            # is appended to the job's existing ``errors`` list. These do NOT
            # change job status (the signals are additive) and do NOT abort the
            # phase; the all-success path adds nothing.
            non_fatal_errors = self._build_non_fatal_errors(
                document_id,
                text_attempted=text_attempted, text_succeeded=text_succeeded,
                text_error=text_error,
                mv_attempted=mv_attempted, mv_succeeded=mv_succeeded,
                mv_error=mv_error,
                ie_attempted=ie_attempted, ie_succeeded=ie_succeeded,
                ie_error=ie_error,
                ci_attempted=ci_attempted, ci_succeeded=ci_succeeded,
                ci_error=ci_error,
            )
            if non_fatal_errors:
                job = await storage.get_job(job_id)
                existing_errors = list(job.errors) if job and job.errors else []
                await storage.update_job(
                    job_id, errors=existing_errors + non_fatal_errors,
                )

            await storage.update_job(job_id, progress_percent=70)

            await task_service.schedule_task(
                "document_finalize",
                {
                    "document_id": document_id,
                    "job_id": job_id,
                    "workspace_id": workspace_id,
                },
                priority=3,
            )

        except RetryableEmbedError:
            # Transient all-signals-failed: do NOT mark the doc/job FAILED (that
            # would freeze the broken state and swallow the retry). Re-raise so
            # AetherTaskService.fail_task() lets Aether auto-reschedule this task
            # per the attached EMBED_RETRY_POLICY. The doc stays PROCESSING and is
            # retried; if every retry is exhausted the doc becomes stale-PROCESSING
            # and the doc_verify sweep re-drives (attempt-capped) it.
            raise
        except Exception as exc:
            ingestion_service.logger.error(
                "Embed phase failed for document %s: %s",
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

    @staticmethod
    def _build_non_fatal_errors(
        document_id: str,
        *,
        text_attempted: int, text_succeeded: int, text_error: Optional[str],
        mv_attempted: int, mv_succeeded: int, mv_error: Optional[str],
        ie_attempted: int, ie_succeeded: int, ie_error: Optional[str],
        ci_attempted: int, ci_succeeded: int, ci_error: Optional[str],
    ) -> list[dict]:
        """Build non-fatal job-error entries for degraded additive signals.

        A signal contributes an entry when it raised (``error`` set) or when it
        had eligible pages but persisted none (``attempted > 0`` and
        ``succeeded == 0``). ``chat_ingest`` additionally reports PARTIAL loss
        (``succeeded < attempted``): unlike the additive embedding signals,
        chat-ingest is the transcription replacement, so a page that fails to
        produce text silently never becomes a memory — that data loss must be
        queryable. Each entry is shaped
        ``{"document_id", "signal", "error", "non_fatal": True}``. A signal that
        was skipped (``attempted == 0``) or fully succeeded contributes nothing.
        """
        entries: list[dict] = []
        for signal, attempted, succeeded, error, report_partial in (
            ("text", text_attempted, text_succeeded, text_error, False),
            ("multivector", mv_attempted, mv_succeeded, mv_error, False),
            ("image_embeds", ie_attempted, ie_succeeded, ie_error, False),
            ("chat_ingest", ci_attempted, ci_succeeded, ci_error, True),
        ):
            if error is not None:
                message = error
            elif attempted > 0 and succeeded == 0:
                message = "produced 0/%d %s embeddings" % (attempted, signal)
            elif report_partial and succeeded < attempted:
                message = "produced %d/%d %s embeddings" % (succeeded, attempted, signal)
            else:
                continue
            entries.append({
                "document_id": document_id,
                "signal": signal,
                "error": message,
                "non_fatal": True,
            })
        return entries

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        return None
