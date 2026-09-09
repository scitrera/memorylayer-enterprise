# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Task handler for the document blob-store orphan garbage-collection sweep.

A periodic, low-priority reconciliation pass over the document blob store. It
complements the delete-time cleanup (``ingestion_service.delete_document`` ->
``blob_storage.delete_tree``) by reclaiming artifacts that slipped through:
failed deletes, crashed ingestions, and pre-existing stale data such as the
now-obsolete ``prompt_embeds/`` subdirs from the old pipeline.

Two reclamation modes per document directory ``{base}/{ws}/documents/{doc}``:

1. Orphan doc dir: no live ``Document`` row AND the newest mtime in the tree is
   older than the grace window -> ``delete_tree`` the whole dir.
2. Stale subtree under a LIVE doc: an obsolete ``prompt_embeds/`` subdir exists
   -> ``delete_tree`` only that subdir (never pages/image_embeds/transcripts).

The sweep is deliberately conservative. Anything ambiguous (cannot confirm
liveness, unknown mtime) is skipped and logged. Each document is processed
under its own try/except so one failure never aborts the sweep.
"""
import time
from typing import Optional

from scitrera_app_framework import Variables, ext_parse_bool, get_extension, get_logger

from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
from memorylayer_server.services.tasks.base import TaskSchedule
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

from memorylayer_saas.config import (
    MEMORYLAYER_DOCUMENT_BLOB_GC_ENABLED,
    DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_GC_ENABLED,
    MEMORYLAYER_DOCUMENT_BLOB_GC_INTERVAL_SEC,
    DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_GC_INTERVAL_SEC,
    MEMORYLAYER_DOCUMENT_BLOB_GC_GRACE_SEC,
    DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_GC_GRACE_SEC,
)
from memorylayer_saas.services.document import get_blob_storage_service

# Obsolete subdir produced by the old prompt-embeds pipeline. Superseded by
# image_embeds/. Safe to reclaim even under a live document.
_OBSOLETE_SUBDIR = "prompt_embeds"


class BlobGarbageCollectionTaskHandler(TaskHandlerPlugin):
    """Periodic reclamation sweep for orphaned document blob artifacts."""

    def get_task_type(self) -> str:
        return "blob_gc"

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        enabled: bool = v.environ(
            MEMORYLAYER_DOCUMENT_BLOB_GC_ENABLED,
            default=DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_GC_ENABLED,
            type_fn=ext_parse_bool,
        )
        if not enabled:
            return None
        interval: int = v.environ(
            MEMORYLAYER_DOCUMENT_BLOB_GC_INTERVAL_SEC,
            default=DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_GC_INTERVAL_SEC,
            type_fn=int,
        )
        grace: int = v.environ(
            MEMORYLAYER_DOCUMENT_BLOB_GC_GRACE_SEC,
            default=DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_GC_GRACE_SEC,
            type_fn=int,
        )
        # NOTE: TaskSchedule exposes only interval_seconds + default_payload; the
        # recurring-task API (schedule_recurring) carries no priority field, so the
        # sweep's low-priority intent is conveyed by its long interval rather than a
        # numeric priority. The grace window rides along in the payload so handle()
        # uses the configured value without re-reading config off the hot path.
        return TaskSchedule(
            interval_seconds=interval,
            default_payload={"grace_seconds": grace},
        )

    async def handle(self, v: Variables, payload: dict) -> None:
        """Run one reconciliation sweep over the document blob store.

        Args:
            v: Variables instance.
            payload: Optional dict; ``grace_seconds`` overrides the configured
                grace window (falls back to config, then the default).
        """
        logger = get_logger(name=self.get_task_type(), v=v)
        storage = get_extension(EXT_STORAGE_BACKEND, v)
        blob_storage = get_blob_storage_service(v)

        grace_seconds: int = payload.get("grace_seconds") if payload else None
        if grace_seconds is None:
            grace_seconds = v.environ(
                MEMORYLAYER_DOCUMENT_BLOB_GC_GRACE_SEC,
                default=DEFAULT_MEMORYLAYER_DOCUMENT_BLOB_GC_GRACE_SEC,
                type_fn=int,
            )

        now = time.time()
        grace_cutoff = now - grace_seconds

        orphan_dirs_reclaimed = 0
        stale_subdirs_reclaimed = 0
        skipped = 0

        try:
            document_dirs = await blob_storage.iter_document_dirs()
        except Exception as exc:
            logger.error(
                "blob_gc: failed to enumerate document dirs; aborting sweep: %s",
                exc, exc_info=True,
            )
            return

        logger.info("blob_gc: starting sweep over %d document dirs", len(document_dirs))

        for workspace_id, doc_id, doc_dir in document_dirs:
            try:
                doc = await self._get_document(storage, doc_id, workspace_id)

                if doc is None:
                    # Mode 1: candidate orphan. Require positive confirmation of
                    # absence (handled above) AND an old-enough mtime.
                    newest = await blob_storage.newest_mtime(doc_dir)
                    if newest is None:
                        logger.info(
                            "blob_gc: skipping orphan candidate %s (ws=%s): mtime "
                            "undeterminable, too uncertain to delete",
                            doc_dir, workspace_id,
                        )
                        skipped += 1
                        continue
                    if newest > grace_cutoff:
                        logger.info(
                            "blob_gc: skipping orphan candidate %s (ws=%s): modified "
                            "%.0fs ago, within %ds grace window",
                            doc_dir, workspace_id, now - newest, grace_seconds,
                        )
                        skipped += 1
                        continue
                    await blob_storage.delete_tree(doc_dir)
                    orphan_dirs_reclaimed += 1
                    logger.info(
                        "blob_gc: reclaimed orphan doc dir %s (ws=%s, doc=%s): no live "
                        "Document row and newest mtime %.0fs old (> %ds grace)",
                        doc_dir, workspace_id, doc_id, now - newest, grace_seconds,
                    )
                    continue

                # Mode 2: live doc. Only the obsolete prompt_embeds/ subdir is
                # eligible for reclamation. Everything else is left untouched.
                obsolete_dir = f"{doc_dir}/{_OBSOLETE_SUBDIR}"
                if await blob_storage.exists(obsolete_dir):
                    await blob_storage.delete_tree(obsolete_dir)
                    stale_subdirs_reclaimed += 1
                    logger.info(
                        "blob_gc: reclaimed obsolete %s/ under live doc %s (ws=%s, "
                        "doc=%s): superseded by image_embeds/",
                        _OBSOLETE_SUBDIR, obsolete_dir, workspace_id, doc_id,
                    )

            except Exception as exc:
                # Non-fatal: log and continue with the next document.
                skipped += 1
                logger.error(
                    "blob_gc: error processing doc dir %s (ws=%s, doc=%s); skipping: %s",
                    doc_dir, workspace_id, doc_id, exc, exc_info=True,
                )

        logger.info(
            "blob_gc: sweep complete: %d orphan dirs reclaimed, %d obsolete subdirs "
            "reclaimed, %d skipped (of %d scanned)",
            orphan_dirs_reclaimed, stale_subdirs_reclaimed, skipped, len(document_dirs),
        )

    @staticmethod
    async def _get_document(storage, doc_id: str, workspace_id: str):
        """Resolve a Document, treating not-found as absence.

        ``get_document`` may return ``None`` or raise a not-found error
        depending on the storage backend; both mean the doc is absent. Any
        OTHER error propagates so the caller skips the dir rather than risk
        deleting blobs for a doc whose liveness we could not determine.
        """
        try:
            return await storage.get_document(doc_id, workspace_id)
        except FileNotFoundError:
            return None
        except KeyError:
            return None
