"""Task handler for ``memorylayer-task.kb_update`` pool tasks.

Created by Aether's native workflow engine when a per-workspace **coalesce
join** (``kb-coalesce``) fires — i.e. after a burst of
``memorylayer.ingest_complete`` / ``memorylayer.decompose_complete`` events for
a workspace has settled.  The handler folds the workspace's new memories into
its knowledgebase by calling ``KnowledgebaseService.generate(workspace)`` and
emits a ``memorylayer.kb_updated`` event so downstream surfaces (e.g. the live
UI) can reload.

Debounce / coalescing is now owned entirely by the server-side coalesce join,
which replaced the hand-rolled per-workspace KV lease this handler used to
carry.  The handler is therefore a thin, idempotent regenerate-and-announce.

A ``degraded`` flag may be present in the payload (set by the join's
``on_timeout`` path, when the deadline sweep fires the task rather than a clean
completion); it is informational only — the regen runs identically.
"""
from datetime import datetime, timezone
from typing import Optional

from scitrera_app_framework import Variables, get_logger

from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
from memorylayer_server.services.tasks.base import TaskSchedule
from memorylayer_server.services.knowledgebase import get_knowledgebase_service
from memorylayer_server.services.knowledgebase.base import KBGenerateOptions

from .doc_added import _ProgressEmitter, _emit_event


class KBUpdateTaskHandler(TaskHandlerPlugin):
    """Task handler for ``kb_update`` — triggered by the workflow engine.

    Claims pool tasks of type ``memorylayer-task.kb_update`` and folds new
    documents into the workspace knowledgebase.  Debounce/coalescing is owned
    by the upstream coalesce join, so this handler simply regenerates once and
    announces the refreshed KB.
    """

    def get_task_type(self) -> str:
        return "kb_update"

    async def handle(self, v: Variables, payload: dict) -> None:
        """Execute the kb_update handler.

        Args:
            v: Variables instance.
            payload: Dict with workspace_id (plus runner-injected reserved
                     keys ``_aether_task_id`` / ``_task_metadata``, and an
                     optional ``degraded`` flag from the join on_timeout path).
        """
        logger = get_logger(v, name="KBUpdateTaskHandler")

        workspace_id = payload.get("workspace_id", "")

        aether_task_id = payload.get("_aether_task_id", "")
        task_metadata = payload.get("_task_metadata") or {}
        progress = _ProgressEmitter(v, aether_task_id, task_metadata, logger)

        if not workspace_id:
            logger.error("kb_update payload missing workspace_id: %s", payload)
            return  # terminal — malformed payload, don't requeue

        # The join's on_timeout path stamps degraded=true (deadline sweep rather
        # than a clean coalesce completion).  Informational only.
        if payload.get("degraded"):
            logger.info(
                "kb_update: workspace %s fired via join timeout (degraded)", workspace_id,
            )

        kb_service = get_knowledgebase_service(v)

        await progress.emit(
            "running",
            step_name="generating",
            step_detail="Updating knowledgebase for %s" % workspace_id,
            step_sequence=1,
            completion=0.5,
        )

        try:
            # Honor an optional regenerate flag from the payload (set by the
            # on-demand background regenerate path); absent/false = incremental
            # fold-in (the default). generate's signature is
            # (workspace_id, context_id, options) — there is no `regenerate` kwarg.
            regenerate = bool(payload.get("regenerate", False))
            await kb_service.generate(
                workspace_id,
                options=KBGenerateOptions(regenerate=regenerate),
            )
        except Exception as exc:
            logger.error(
                "kb_update: knowledgebase generation failed for %s: %s",
                workspace_id, exc, exc_info=True,
            )
            await progress.emit(
                "failed",
                step_name="generating",
                step_detail="Knowledgebase update failed: %s" % exc,
                step_sequence=1,
                completion=-1.0,
                summary=str(exc),
            )
            # Surface the failure to downstream surfaces (e.g. the live UI) by
            # mirroring the kb_updated event shape on the failure event. With the
            # AGE->NetworkX fallback in place generate() rarely hard-fails, but a
            # NetworkX-path or DB error can still occur — this event is the UI's
            # signal. Terminal/no-retry, same as the progress "failed" above.
            await _emit_event(v, workspace_id, "memorylayer.kb_update_failed", {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }, logger)
            return  # terminal — don't re-raise

        # Success: announce the refreshed KB.
        await _emit_event(v, workspace_id, "memorylayer.kb_updated", {
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, logger)

        await progress.emit(
            "completed",
            step_name="complete",
            step_detail="Knowledgebase updated for %s" % workspace_id,
            step_sequence=2,
            completion=1.0,
        )
        logger.info("kb_update: knowledgebase updated for workspace %s", workspace_id)

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        return None
