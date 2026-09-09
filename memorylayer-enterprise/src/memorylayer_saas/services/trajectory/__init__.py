"""Trajectory service package for retrieval observability."""
from .base import (
    TrajectoryService,
    TrajectoryServicePluginBase,
    EXT_TRAJECTORY_SERVICE,
    DEFAULT_TTL_SECONDS,
)
from .default import DefaultTrajectoryService, DefaultTrajectoryServicePlugin

from scitrera_app_framework import Variables, get_extension


def get_trajectory_service(v: Variables = None) -> TrajectoryService:
    """Get the trajectory service instance."""
    return get_extension(EXT_TRAJECTORY_SERVICE, v)


__all__ = (
    'TrajectoryService',
    'TrajectoryServicePluginBase',
    'DefaultTrajectoryService',
    'DefaultTrajectoryServicePlugin',
    'get_trajectory_service',
    'EXT_TRAJECTORY_SERVICE',
    'DEFAULT_TTL_SECONDS',
)
