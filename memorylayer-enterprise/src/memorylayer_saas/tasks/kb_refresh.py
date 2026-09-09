# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Recurring knowledgebase-refresh reconciler.

The KB is normally refreshed event-driven: ``memorylayer.ingest_complete`` /
``memorylayer.decompose_complete`` feed a per-workspace ``kb-coalesce`` join that
fires a ``kb_update`` task. Those events are **best-effort** (see
``memorylayer_server.services.events.emit_event`` — a send failure is logged and
swallowed, never retried). A transient Aether failure during fact decomposition
can therefore drop the KB-refresh signal, leaving a workspace's KB permanently
stale relative to its graph (the graph view shows hundreds of nodes while the
articles view still shows the empty first-generation index).

This reconciler is the KB analogue of ``doc_verify`` / ``blob_gc``: a periodic
pass that walks workspaces which already have a KB and regenerates any whose KB
fell behind. Staleness is detected cheaply by ``KnowledgebaseService.generate``
itself — it short-circuits on an unchanged workspace change-watermark (no
analyze, no LLM), so idle / up-to-date KBs cost almost nothing. Only KBs that
actually drifted do real work.

Defaults ON (``MEMORYLAYER_KB_REFRESH_ENABLED`` = True), matching the sibling
reconcilers ``doc_verify`` / ``blob_gc``: a stale KB is a correctness gap worth
healing without an operator opt-in, and the watermark short-circuit keeps idle
sweeps cheap. Set the env var to false to disable (``get_schedule`` then returns
``None`` and no recurring task is registered).
"""
from typing import Optional

from scitrera_app_framework import Variables, ext_parse_bool, get_extension, get_logger

from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
from memorylayer_server.services.tasks.base import TaskSchedule
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
from memorylayer_server.services.knowledgebase import get_knowledgebase_service

from memorylayer_saas.config import (
    MEMORYLAYER_KB_REFRESH_ENABLED,
    DEFAULT_MEMORYLAYER_KB_REFRESH_ENABLED,
    MEMORYLAYER_KB_REFRESH_INTERVAL_SEC,
    DEFAULT_MEMORYLAYER_KB_REFRESH_INTERVAL_SEC,
    MEMORYLAYER_KB_REFRESH_MAX_WORKSPACES,
    DEFAULT_MEMORYLAYER_KB_REFRESH_MAX_WORKSPACES,
)


class KBRefreshTaskHandler(TaskHandlerPlugin):
    """Periodic reconciler that regenerates stale knowledgebases."""

    def get_task_type(self) -> str:
        return "kb_refresh"

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        enabled: bool = v.environ(
            MEMORYLAYER_KB_REFRESH_ENABLED,
            default=DEFAULT_MEMORYLAYER_KB_REFRESH_ENABLED,
            type_fn=ext_parse_bool,
        )
        if not enabled:
            return None
        interval: int = v.environ(
            MEMORYLAYER_KB_REFRESH_INTERVAL_SEC,
            default=DEFAULT_MEMORYLAYER_KB_REFRESH_INTERVAL_SEC,
            type_fn=int,
        )
        max_workspaces: int = v.environ(
            MEMORYLAYER_KB_REFRESH_MAX_WORKSPACES,
            default=DEFAULT_MEMORYLAYER_KB_REFRESH_MAX_WORKSPACES,
            type_fn=int,
        )
        return TaskSchedule(
            interval_seconds=interval,
            default_payload={"max_workspaces": max_workspaces},
        )

    async def handle(self, v: Variables, payload: dict) -> None:
        """Run one reconciliation sweep over existing knowledgebases."""
        logger = get_logger(v, name=self.get_task_type())
        storage = get_extension(EXT_STORAGE_BACKEND, v)
        kb_service = get_knowledgebase_service(v)

        max_workspaces = (payload or {}).get("max_workspaces")
        if max_workspaces is None:
            max_workspaces = v.environ(
                MEMORYLAYER_KB_REFRESH_MAX_WORKSPACES,
                default=DEFAULT_MEMORYLAYER_KB_REFRESH_MAX_WORKSPACES,
                type_fn=int,
            )

        try:
            workspaces = await storage.list_workspaces()
        except Exception as exc:
            logger.error(
                "kb_refresh: failed to list workspaces; aborting sweep: %s",
                exc, exc_info=True,
            )
            return

        logger.info("kb_refresh: starting sweep over %d workspace(s)", len(workspaces))

        processed = 0
        checked = 0
        skipped_no_kb = 0
        errors = 0

        for ws in workspaces:
            if max_workspaces and processed >= max_workspaces:
                break
            processed += 1
            try:
                # Only refresh KBs that already exist. First-time KB creation is
                # the event-driven path's job (e.g. after a document upload); the
                # reconciler exists to recover KBs that drifted, not to eagerly
                # generate one for every workspace.
                existing = await kb_service.get_knowledgebase(ws.id)
                if existing is None:
                    skipped_no_kb += 1
                    continue
                # generate() with default options (regenerate=False)
                # short-circuits internally when the change-watermark is
                # unchanged (no analyze / no LLM); it only regenerates a KB that
                # actually fell behind its graph.
                await kb_service.generate(ws.id)
                checked += 1
            except Exception as exc:
                # One workspace failing must not abort the sweep.
                errors += 1
                logger.error(
                    "kb_refresh: failed for workspace %s; skipping: %s",
                    ws.id, exc, exc_info=True,
                )

        logger.info(
            "kb_refresh: sweep complete: processed=%d checked=%d no_kb_skipped=%d errors=%d",
            processed, checked, skipped_no_kb, errors,
        )
