# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""
Extended storage backend interface with cold tier support.

This extends the OSS StorageBackend with cold tier operations for
cost-optimized storage of less frequently accessed memories.
"""
from abc import abstractmethod
from typing import Optional, TYPE_CHECKING

from memorylayer_server.services.storage import StorageBackend

if TYPE_CHECKING:
    from ..models.trajectory import Trajectory, TrajectoryListResponse

# Use enterprise Memory model with multivector support
from memorylayer_saas.models.memory import Memory


class ColdTierStorageBackend(StorageBackend):
    """
    Extended storage backend with cold tier support.

    Adds methods for archiving memories to compressed cold storage
    and searching across both hot and cold tiers. The cold tier uses
    LEANN (Low Energy Approximate Nearest Neighbor) graph compression
    to reduce storage costs for infrequently accessed memories.

    Implementations:
    - PostgreSQLBackend: Enterprise profile with pgvector + LEANN cold tier
    """

    @abstractmethod
    async def archive_memory(
        self,
        workspace_id: str,
        memory_id: str,
    ) -> bool:
        """
        Archive a memory to cold tier storage.

        Moves the memory from hot tier (with full embeddings) to cold tier
        (compressed graph structure without embeddings). The memory's graph
        relationships are preserved in CSR (Compressed Sparse Row) format.

        Args:
            workspace_id: Workspace identifier
            memory_id: Memory to archive

        Returns:
            True if successfully archived, False if memory not found or already archived.
        """
        pass

    @abstractmethod
    async def restore_memory(
        self,
        workspace_id: str,
        memory_id: str,
    ) -> bool:
        """
        Restore a memory from cold tier to hot tier.

        Moves the memory back to hot tier. Embeddings will need to be
        regenerated separately by the embedding service.

        Args:
            workspace_id: Workspace identifier
            memory_id: Memory to restore

        Returns:
            True if successfully restored, False if memory not found or not in cold tier.
        """
        pass

    @abstractmethod
    async def search_cold_memories(
        self,
        workspace_id: str,
        query_embedding: list[float],
        limit: int = 10,
        min_relevance: float = 0.5,
    ) -> list[tuple[Memory, float]]:
        """
        Search memories in cold tier storage.

        Uses graph-guided search with on-demand embedding computation to find
        relevant memories in the cold tier. This may have higher latency than
        hot tier search due to on-demand computation.

        Args:
            workspace_id: Workspace identifier
            query_embedding: Query vector for similarity search
            limit: Maximum number of results
            min_relevance: Minimum relevance score threshold

        Returns:
            List of (memory, relevance_score) tuples
        """
        pass

    @abstractmethod
    async def get_archival_candidates(
        self,
        workspace_id: str,
        max_importance: float = 0.3,
        max_access_count: int = 5,
        older_than_days: int = 90,
        limit: int = 100,
    ) -> list[Memory]:
        """
        Get memories that are candidates for archival to cold tier.

        Returns memories that meet archival criteria based on:
        - importance score at or below threshold
        - access count at or below threshold
        - last accessed older than specified days

        These memories can be archived to reduce hot tier storage costs.

        Args:
            workspace_id: Workspace identifier
            max_importance: Maximum importance score for candidates
            max_access_count: Maximum access count for candidates
            older_than_days: Minimum days since last access
            limit: Maximum number of candidates to return

        Returns:
            List of memories eligible for archival
        """
        pass

    @abstractmethod
    async def get_hot_promotion_candidates(
        self,
        workspace_id: str,
        min_access_count: int = 10,
        limit: int = 100,
    ) -> list[Memory]:
        """
        Get cold tier memories that are frequently accessed and should be
        promoted back to hot tier.

        Uses cold tier access tracking (cold_access_count) rather than
        hot tier access_count to identify promotion candidates.

        Args:
            workspace_id: Workspace identifier
            min_access_count: Minimum cold access count threshold
            limit: Maximum number of candidates to return

        Returns:
            List of Memory objects eligible for hot promotion,
            sorted by cold access count descending.
        """
        pass

    @abstractmethod
    async def get_cold_storage_stats(self, workspace_id: str) -> dict:
        """
        Get cold tier storage statistics for workspace.

        Returns:
            Dict with:
            - cold_memory_count: number of memories in cold tier
            - cold_storage_bytes: total bytes used by cold tier
            - hot_memory_count: number of memories in hot tier
            - hot_storage_bytes: total bytes used by hot tier
            - compression_ratio: cold_storage_bytes / equivalent_hot_storage_bytes
            - estimated_savings_bytes: bytes saved by cold tier compression
        """
        pass

    # Trajectory storage methods (non-abstract with default no-op implementations)

    async def create_trajectory(self, trajectory: 'Trajectory') -> 'Trajectory':
        """Store a trajectory record. Override in subclasses for persistence."""
        return trajectory

    async def get_trajectory(self, workspace_id: str, trajectory_id: str) -> Optional['Trajectory']:
        """Get a trajectory by ID. Override in subclasses for persistence."""
        return None

    async def list_trajectories(
        self, workspace_id: str, limit: int = 20, offset: int = 0
    ) -> 'TrajectoryListResponse':
        """List trajectories for a workspace. Override in subclasses for persistence."""
        from ..models.trajectory import TrajectoryListResponse
        return TrajectoryListResponse(trajectories=[], total_count=0, limit=limit, offset=offset)

    async def delete_expired_trajectories(self) -> int:
        """Delete expired trajectories. Override in subclasses for persistence."""
        return 0

    # Dataset storage methods (non-abstract with default no-op implementations)

    async def create_dataset(self, dataset: 'Dataset') -> 'Dataset':
        """Store a dataset record. Override in subclasses for persistence."""
        return dataset

    async def get_dataset(self, dataset_id: str, workspace_id: str = None) -> Optional['Dataset']:
        """Get a dataset by ID. Override in subclasses for persistence."""
        return None

    async def find_dataset_by_hash(self, workspace_id: str, content_hash: str) -> Optional['Dataset']:
        """Find a dataset by content hash. Override in subclasses for persistence."""
        return None

    async def list_datasets(
        self, workspace_id: str, status: str = None, limit: int = 50, offset: int = 0,
    ) -> tuple[list, int]:
        """List datasets for a workspace. Override in subclasses for persistence."""
        return [], 0

    async def update_dataset(self, dataset_id: str, **kwargs) -> None:
        """Update dataset fields. Override in subclasses for persistence."""
        pass

    async def delete_dataset(self, dataset_id: str) -> None:
        """Delete a dataset record. Override in subclasses for persistence."""
        pass

    async def create_dataset_job(self, job: 'DatasetJob') -> 'DatasetJob':
        """Store a dataset job record. Override in subclasses for persistence."""
        return job

    async def get_dataset_job(self, job_id: str) -> Optional['DatasetJob']:
        """Get a dataset job by ID. Override in subclasses for persistence."""
        return None

    async def list_dataset_jobs(
        self, workspace_id: str, status: str = None, limit: int = 50,
    ) -> list:
        """List dataset jobs for a workspace. Override in subclasses for persistence."""
        return []

    async def update_dataset_job(self, job_id: str, **kwargs) -> None:
        """Update dataset job fields. Override in subclasses for persistence."""
        pass
