"""Abstract trajectory service for retrieval observability.

Provides the interface for tracking recall pipeline decisions including
search, ranking, filtering, and fallback events.
"""
from abc import ABC, abstractmethod
from logging import Logger

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables, Plugin, enabled_option_pattern

from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

from ...models.trajectory import Trajectory, TrajectoryEvent, TrajectoryEventType, TrajectoryListResponse

# Extension point constant
EXT_TRAJECTORY_SERVICE = 'memorylayer-enterprise-trajectory-service'

# Configuration
MEMORYLAYER_TRAJECTORY_SERVICE = 'MEMORYLAYER_TRAJECTORY_SERVICE'
DEFAULT_MEMORYLAYER_TRAJECTORY_SERVICE = 'default'

# Default TTL for trajectories (24 hours)
DEFAULT_TTL_SECONDS = 86400


class TrajectoryService(ABC):
    """Abstract base class for trajectory services."""

    def __init__(self, v: Variables = None):
        self.logger = get_logger(v, name=self.__class__.__name__)

    @abstractmethod
    def start_trace(self, workspace_id: str, query: str) -> Trajectory:
        """Start a new trajectory trace for a recall operation."""
        pass

    @abstractmethod
    def add_event(self, trajectory: Trajectory, event_type: TrajectoryEventType, data: dict) -> None:
        """Add an event to an active trajectory."""
        pass

    @abstractmethod
    async def save(self, trajectory: Trajectory) -> None:
        """Persist a completed trajectory."""
        pass

    @abstractmethod
    async def get_trajectory(self, trajectory_id: str, workspace_id: str) -> Trajectory | None:
        """Retrieve a trajectory by ID."""
        pass

    @abstractmethod
    async def list_trajectories(
        self, workspace_id: str, limit: int = 20, offset: int = 0
    ) -> TrajectoryListResponse:
        """List trajectories for a workspace."""
        pass

    @abstractmethod
    async def delete_expired(self) -> int:
        """Delete expired trajectories. Returns count deleted."""
        pass


# noinspection PyAbstractClass
class TrajectoryServicePluginBase(Plugin):
    """Base plugin for trajectory service."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_TRAJECTORY_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_TRAJECTORY_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_TRAJECTORY_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_TRAJECTORY_SERVICE, DEFAULT_MEMORYLAYER_TRAJECTORY_SERVICE)

    def get_dependencies(self, v: Variables):
        return (EXT_STORAGE_BACKEND,)
