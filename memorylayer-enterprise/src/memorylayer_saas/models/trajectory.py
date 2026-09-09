"""Trajectory models for retrieval observability.

Trajectories record the decision-making process during memory recall,
providing full observability into search, ranking, filtering, and
fallback decisions.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TrajectoryEventType(str, Enum):
    """Types of events that can occur during a recall trajectory."""
    SEARCH = "search"
    FILTER = "filter"
    RANK = "rank"
    THRESHOLD = "threshold"
    ASSOCIATION = "association"
    RERANK = "rerank"
    FALLBACK = "fallback"
    # Agentic recall (Phase 2 "Memora-Control") loop actions.
    EXPAND = "expand"
    RE_QUERY = "re_query"
    STOP = "stop"


class TrajectoryEvent(BaseModel):
    """A single event in a trajectory."""
    type: TrajectoryEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict = Field(default_factory=dict)


class Trajectory(BaseModel):
    """Full trajectory of a recall operation."""
    id: str
    workspace_id: str
    query: str
    events: list[TrajectoryEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = Field(default=86400, description="Time to live in seconds (default 24h)")

    @property
    def is_expired(self) -> bool:
        """Check if trajectory has expired based on TTL."""
        from datetime import timedelta
        expiry = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.now(timezone.utc) > expiry


class TrajectoryListResponse(BaseModel):
    """Response for listing trajectories."""
    trajectories: list[Trajectory] = Field(default_factory=list)
    total_count: int = 0
    limit: int = 20
    offset: int = 0
