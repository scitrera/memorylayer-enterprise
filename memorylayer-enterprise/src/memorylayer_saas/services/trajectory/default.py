"""Default trajectory service implementation.

Stores trajectories via the storage backend. Provides in-memory buffering
during active traces, then persists on save().
"""
from datetime import datetime, timezone
from logging import Logger
from typing import Optional

from scitrera_app_framework import Variables, get_logger

from ...models.trajectory import (
    Trajectory,
    TrajectoryEvent,
    TrajectoryEventType,
    TrajectoryListResponse,
)
from .base import (
    TrajectoryService,
    TrajectoryServicePluginBase,
    DEFAULT_TTL_SECONDS,
    EXT_TRAJECTORY_SERVICE,
)

from memorylayer_server.services.storage import EXT_STORAGE_BACKEND, StorageBackend
from memorylayer_server.utils import generate_id


class DefaultTrajectoryService(TrajectoryService):
    """Default trajectory service with storage backend persistence."""

    def __init__(self, v: Variables = None, storage: StorageBackend = None):
        super().__init__(v)
        self.storage = storage
        self.logger.info("Initialized DefaultTrajectoryService")

    def start_trace(self, workspace_id: str, query: str) -> Trajectory:
        """Start a new trajectory trace."""
        trajectory = Trajectory(
            id=generate_id("traj"),
            workspace_id=workspace_id,
            query=query,
            events=[],
            created_at=datetime.now(timezone.utc),
            ttl_seconds=DEFAULT_TTL_SECONDS,
        )
        self.logger.debug("Started trajectory trace: %s for query: %s", trajectory.id, query[:50])
        return trajectory

    def add_event(self, trajectory: Trajectory, event_type: TrajectoryEventType, data: dict) -> None:
        """Add an event to the trajectory."""
        event = TrajectoryEvent(
            type=event_type,
            timestamp=datetime.now(timezone.utc),
            data=data,
        )
        trajectory.events.append(event)
        self.logger.debug(
            "Added %s event to trajectory %s",
            event_type.value,
            trajectory.id
        )

    async def save(self, trajectory: Trajectory) -> None:
        """Persist trajectory to storage."""
        try:
            await self.storage.create_trajectory(trajectory)
            self.logger.debug(
                "Saved trajectory %s with %d events",
                trajectory.id,
                len(trajectory.events)
            )
        except AttributeError:
            self.logger.warning(
                "Storage backend does not support trajectory persistence"
            )
        except Exception as e:
            self.logger.error("Failed to save trajectory %s: %s", trajectory.id, e)

    async def get_trajectory(self, trajectory_id: str, workspace_id: str) -> Optional[Trajectory]:
        """Get trajectory by ID."""
        try:
            return await self.storage.get_trajectory(workspace_id, trajectory_id)
        except AttributeError:
            self.logger.warning("Storage backend does not support trajectory retrieval")
            return None

    async def list_trajectories(
        self, workspace_id: str, limit: int = 20, offset: int = 0
    ) -> TrajectoryListResponse:
        """List trajectories for workspace."""
        try:
            return await self.storage.list_trajectories(workspace_id, limit=limit, offset=offset)
        except AttributeError:
            self.logger.warning("Storage backend does not support trajectory listing")
            return TrajectoryListResponse(trajectories=[], total_count=0, limit=limit, offset=offset)

    async def delete_expired(self) -> int:
        """Delete expired trajectories."""
        try:
            return await self.storage.delete_expired_trajectories()
        except AttributeError:
            self.logger.warning("Storage backend does not support trajectory cleanup")
            return 0


class DefaultTrajectoryServicePlugin(TrajectoryServicePluginBase):
    """Plugin for default trajectory service."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> DefaultTrajectoryService:
        from memorylayer_server.services.storage import get_storage_backend
        storage = get_storage_backend(v)
        return DefaultTrajectoryService(v=v, storage=storage)
