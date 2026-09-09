"""
PostgreSQL Audit Service for MemoryLayer Enterprise.

Provides buffered, async writes to the ``audit_events`` PostgreSQL table.
Events accumulate in an internal buffer and are flushed in batches either
when the batch size threshold is reached or when the flush interval elapses.
"""
import asyncio
import logging
from datetime import datetime, timezone
from logging import Logger
from typing import Optional

from scitrera_app_framework import Variables, get_extension, get_logger

from memorylayer_server.services.audit.base import (
    AuditEvent,
    AuditService,
    AuditServicePluginBase,
)
from memorylayer_server.services.storage.base import EXT_STORAGE_BACKEND

from memorylayer_saas.storage.models import AuditEventModel

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MEMORYLAYER_AUDIT_BATCH_SIZE = 'MEMORYLAYER_AUDIT_BATCH_SIZE'
DEFAULT_MEMORYLAYER_AUDIT_BATCH_SIZE = 50

MEMORYLAYER_AUDIT_FLUSH_INTERVAL = 'MEMORYLAYER_AUDIT_FLUSH_INTERVAL'
DEFAULT_MEMORYLAYER_AUDIT_FLUSH_INTERVAL = 5  # seconds


class PostgresAuditService(AuditService):
    """Audit service backed by PostgreSQL.

    Accumulates events in an in-memory buffer and flushes them to the
    ``audit_events`` table in batches.  Flushing is triggered either when
    the buffer reaches ``batch_size`` or when ``flush_interval`` seconds
    have elapsed since the last flush.

    The background flush task is started on the first call to :meth:`record`
    and runs until :meth:`close` is called.
    """

    def __init__(
            self,
            session_factory,
            batch_size: int = DEFAULT_MEMORYLAYER_AUDIT_BATCH_SIZE,
            flush_interval: float = DEFAULT_MEMORYLAYER_AUDIT_FLUSH_INTERVAL,
            logger: Optional[logging.Logger] = None,
    ):
        self._session_factory = session_factory
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self.logger = logger or logging.getLogger(__name__)

        self._buffer: list[AuditEvent] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_flush_task(self) -> None:
        """Start the background flush task if it is not already running."""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._periodic_flush())

    async def close(self) -> None:
        """Flush remaining buffered events and stop the background task."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush_buffer()

    # ------------------------------------------------------------------
    # AuditService ABC implementation
    # ------------------------------------------------------------------

    async def record(self, event: AuditEvent) -> None:
        """Buffer a single audit event, flushing if the batch size is reached."""
        async with self._lock:
            self._buffer.append(event)
            should_flush = len(self._buffer) >= self._batch_size

        self._ensure_flush_task()

        if should_flush:
            await self._flush_buffer()

    async def record_batch(self, events: list[AuditEvent]) -> None:
        """Buffer multiple audit events at once."""
        if not events:
            return

        async with self._lock:
            self._buffer.extend(events)
            should_flush = len(self._buffer) >= self._batch_size

        self._ensure_flush_task()

        if should_flush:
            await self._flush_buffer()

    async def query(
            self,
            tenant_id: str,
            workspace_id: Optional[str] = None,
            event_type: Optional[str] = None,
            since: Optional[datetime] = None,
            limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events from PostgreSQL.

        Args:
            tenant_id: Required tenant filter.
            workspace_id: Optional workspace filter.
            event_type: Optional event type filter.
            since: Optional lower-bound timestamp (inclusive).
            limit: Maximum events to return (default 100).

        Returns:
            List of :class:`AuditEvent` ordered by timestamp descending.
        """
        from sqlalchemy import select, and_

        filters = [AuditEventModel.tenant_id == tenant_id]
        if workspace_id is not None:
            filters.append(AuditEventModel.workspace_id == workspace_id)
        if event_type is not None:
            filters.append(AuditEventModel.event_type == event_type)
        if since is not None:
            filters.append(AuditEventModel.timestamp >= since)

        stmt = (
            select(AuditEventModel)
            .where(and_(*filters))
            .order_by(AuditEventModel.timestamp.desc())
            .limit(limit)
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()

        return [self._row_to_event(row) for row in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _flush_buffer(self) -> None:
        """Drain the buffer and persist all pending events to PostgreSQL."""
        async with self._lock:
            if not self._buffer:
                return
            batch, self._buffer = self._buffer, []

        try:
            async with self._session_factory() as session:
                async with session.begin():
                    session.add_all([self._event_to_row(e) for e in batch])
            self.logger.debug("Flushed %d audit events to PostgreSQL", len(batch))
        except Exception as exc:
            self.logger.error(
                "Failed to flush %d audit events: %s", len(batch), exc
            )
            # Re-buffer on failure to avoid silent data loss
            async with self._lock:
                self._buffer = batch + self._buffer

    async def _periodic_flush(self) -> None:
        """Background task: flush the buffer every ``flush_interval`` seconds."""
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                await self._flush_buffer()
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _event_to_row(event: AuditEvent) -> AuditEventModel:
        return AuditEventModel(
            id=event.id,
            event_type=event.event_type,
            action=event.action,
            tenant_id=event.tenant_id,
            workspace_id=event.workspace_id,
            user_id=event.user_id,
            resource_type=event.resource_type,
            resource_id=event.resource_id,
            meta=event.metadata or {},
            timestamp=event.timestamp,
        )

    @staticmethod
    def _row_to_event(row: AuditEventModel) -> AuditEvent:
        return AuditEvent(
            id=row.id,
            event_type=row.event_type,
            action=row.action,
            tenant_id=row.tenant_id,
            workspace_id=row.workspace_id,
            user_id=row.user_id,
            resource_type=row.resource_type,
            resource_id=row.resource_id,
            metadata=row.meta or {},
            timestamp=row.timestamp,
        )


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class PostgresAuditServicePlugin(AuditServicePluginBase):
    """Enterprise plugin that enables PostgreSQL-backed audit logging.

    Activated when ``MEMORYLAYER_AUDIT_SERVICE=postgresql``.

    Requires the PostgreSQL storage backend to be initialised first so
    that the async session factory is available.
    """
    PROVIDER_NAME = 'postgresql'

    def initialize(self, v: Variables, logger: Logger) -> PostgresAuditService:
        session_factory = lambda: storage.session_factory()

        batch_size = v.environ(
            MEMORYLAYER_AUDIT_BATCH_SIZE,
            default=DEFAULT_MEMORYLAYER_AUDIT_BATCH_SIZE,
            type_fn=int,
        )
        flush_interval = v.environ(
            MEMORYLAYER_AUDIT_FLUSH_INTERVAL,
            default=DEFAULT_MEMORYLAYER_AUDIT_FLUSH_INTERVAL,
            type_fn=float,
        )

        service = PostgresAuditService(
            session_factory=None,
            batch_size=batch_size,
            flush_interval=flush_interval,
            logger=logger,
        )
        logger.info(
            "PostgresAuditService initialised (batch_size=%d, flush_interval=%.1fs)",
            batch_size,
            flush_interval,
        )
        return service

    async def async_ready(self, v: Variables, logger: Logger, value: PostgresAuditService | None) -> None:
        # defer populating session factory until after storage backend is ready
        storage = get_extension(EXT_STORAGE_BACKEND, v)
        value._session_factory = storage.session_factory
        return

    def get_dependencies(self, v: Variables):
        return (EXT_STORAGE_BACKEND,)


__all__ = [
    "PostgresAuditService",
    "PostgresAuditServicePlugin",
    "MEMORYLAYER_AUDIT_BATCH_SIZE",
    "MEMORYLAYER_AUDIT_FLUSH_INTERVAL",
]
