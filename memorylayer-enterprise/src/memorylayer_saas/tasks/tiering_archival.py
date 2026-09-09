"""Recurring cold-tier archival sweep.

Periodically moves eligible hot-tier memories to the LEANN cold tier across all
workspaces that have opted in (``settings["tiering"].cold_tier_enabled``). This
is the missing scheduler that makes the tiering subsystem actually *do*
something — ``TieringService.run_archival_cycle`` existed but nothing invoked it.

Ships DARK: ``MEMORYLAYER_TIERING_ARCHIVAL_ENABLED`` defaults False, so
``get_schedule`` returns ``None`` and no recurring task is registered until an
operator turns it on. Even when enabled, only per-workspace-opted-in workspaces
are touched. The same underlying sweep is exposed on-demand (with a dry-run
mode) via ``POST /v1/admin/tiering/run`` for impact assessment before enabling.
"""
from typing import Optional

from scitrera_app_framework import Variables, ext_parse_bool, get_logger

from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
from memorylayer_server.services.tasks.base import TaskSchedule

from memorylayer_saas.config import (
    MEMORYLAYER_TIERING_ARCHIVAL_ENABLED,
    DEFAULT_MEMORYLAYER_TIERING_ARCHIVAL_ENABLED,
    MEMORYLAYER_TIERING_ARCHIVAL_INTERVAL_SEC,
    DEFAULT_MEMORYLAYER_TIERING_ARCHIVAL_INTERVAL_SEC,
    MEMORYLAYER_TIERING_ARCHIVAL_BATCH_SIZE,
    DEFAULT_MEMORYLAYER_TIERING_ARCHIVAL_BATCH_SIZE,
)


class TieringArchivalTaskHandler(TaskHandlerPlugin):
    """Periodic archival sweep across cold-tier-enabled workspaces."""

    def get_task_type(self) -> str:
        return "tiering_archival"

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        enabled: bool = v.environ(
            MEMORYLAYER_TIERING_ARCHIVAL_ENABLED,
            default=DEFAULT_MEMORYLAYER_TIERING_ARCHIVAL_ENABLED,
            type_fn=ext_parse_bool,
        )
        if not enabled:
            return None
        interval: int = v.environ(
            MEMORYLAYER_TIERING_ARCHIVAL_INTERVAL_SEC,
            default=DEFAULT_MEMORYLAYER_TIERING_ARCHIVAL_INTERVAL_SEC,
            type_fn=int,
        )
        batch_size: int = v.environ(
            MEMORYLAYER_TIERING_ARCHIVAL_BATCH_SIZE,
            default=DEFAULT_MEMORYLAYER_TIERING_ARCHIVAL_BATCH_SIZE,
            type_fn=int,
        )
        # Payload carries the batch size so handle() doesn't re-read config on
        # the hot path (mirrors blob_gc's grace-window pattern).
        return TaskSchedule(
            interval_seconds=interval,
            default_payload={"batch_size": batch_size},
        )

    async def handle(self, v: Variables, payload: dict) -> None:
        """Run one archival sweep over all cold-tier-enabled workspaces."""
        logger = get_logger(name=self.get_task_type(), v=v)

        # Lazy import mirrors the tiering API's get_tiering_service_dep.
        from memorylayer_saas.services.tiering import get_tiering_service
        tiering_service = get_tiering_service(v)

        batch_size: Optional[int] = payload.get("batch_size") if payload else None
        if batch_size is None:
            batch_size = v.environ(
                MEMORYLAYER_TIERING_ARCHIVAL_BATCH_SIZE,
                default=DEFAULT_MEMORYLAYER_TIERING_ARCHIVAL_BATCH_SIZE,
                type_fn=int,
            )

        logger.info("tiering_archival: starting sweep (batch_size=%d)", batch_size)
        summary = await tiering_service.run_archival_sweep(
            batch_size=batch_size,
            only_enabled=True,
            dry_run=False,
        )
        logger.info(
            "tiering_archival: sweep complete: processed=%d skipped=%d archived=%d failed=%d",
            summary.get("workspaces_processed", 0),
            summary.get("workspaces_skipped", 0),
            summary.get("total_archived", 0),
            summary.get("total_failed", 0),
        )
