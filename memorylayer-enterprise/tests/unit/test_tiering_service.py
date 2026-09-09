"""
Unit tests for TieringService.

Tests:
- identify_archival_candidates: Finding memories eligible for cold storage
- archive_memories: Moving memories to cold tier
- restore_memories: Restoring memories from cold tier
- promote_hot_candidates: Automatic warm-up for frequently accessed cold memories
- get_tiering_stats: Cold/hot tier statistics
- run_archival_cycle/run_warmup_cycle: Scheduled cycle operations
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from typing import Optional

from memorylayer_server.models.memory import MemoryType
# Use enterprise Memory model with multivector support
from memorylayer_saas.models.memory import Memory
from memorylayer_server.services.embedding import EmbeddingService

from memorylayer_saas.storage.base import ColdTierStorageBackend

from memorylayer_saas.services.tiering.default import TieringService
from memorylayer_saas.services.tiering.base import (
    TieringStats,
    ArchivalResult,
    RestoreResult,
)
from memorylayer_saas.services.tiering.default import (
    DEFAULT_MAX_IMPORTANCE,
    DEFAULT_MAX_ACCESS_COUNT,
    DEFAULT_OLDER_THAN_DAYS,
    DEFAULT_WARMUP_ACCESS_THRESHOLD,
)


def create_test_memory(
    memory_id: str = "mem_test_001",
    workspace_id: str = "test_workspace",
    content: str = "Test memory content",
    importance: float = 0.5,
    access_count: int = 0,
    last_accessed_at: Optional[datetime] = None,
    created_at: Optional[datetime] = None,
) -> Memory:
    """Create a test Memory object."""
    now = datetime.now(timezone.utc)
    return Memory(
        id=memory_id,
        workspace_id=workspace_id,
        tenant_id="test_tenant",
        content=content,
        content_hash=f"hash_{memory_id}",
        type=MemoryType.SEMANTIC,
        importance=importance,
        access_count=access_count,
        last_accessed_at=last_accessed_at,
        created_at=created_at or now,
        updated_at=now,
    )


@pytest.fixture
def mock_storage() -> AsyncMock:
    """Create a mock storage backend with cold tier support."""
    storage = AsyncMock(spec=ColdTierStorageBackend)
    # Set up default return values
    storage.get_archival_candidates.return_value = []
    storage.archive_memory.return_value = True
    storage.restore_memory.return_value = True
    storage.get_memory.return_value = None
    storage.update_memory.return_value = None
    storage.get_hot_promotion_candidates.return_value = []
    storage.get_cold_storage_stats.return_value = {
        "hot_memory_count": 0,
        "cold_memory_count": 0,
        "hot_storage_bytes": 0,
        "cold_storage_bytes": 0,
        "compression_ratio": 0.0,
        "estimated_savings_bytes": 0,
    }
    return storage


@pytest.fixture
def mock_embedding_service() -> AsyncMock:
    """Create a mock embedding service."""
    service = AsyncMock(spec=EmbeddingService)
    service.embed.return_value = [0.1] * 1536
    return service


@pytest.fixture
def tiering_service(mock_storage: AsyncMock, v) -> TieringService:
    """Create TieringService with mock storage."""
    return TieringService(storage=mock_storage, v=v)


@pytest.fixture
def tiering_service_with_embedding(
    mock_storage: AsyncMock,
    mock_embedding_service: AsyncMock,
    v,
) -> TieringService:
    """Create TieringService with mock storage and embedding service."""
    return TieringService(
        storage=mock_storage,
        embedding_service=mock_embedding_service,
        v=v,
    )


@pytest.fixture
def workspace_id() -> str:
    """Test workspace ID."""
    return "test_workspace"


class TestIdentifyArchivalCandidates:
    """Tests for identify_archival_candidates operation."""

    @pytest.mark.asyncio
    async def test_identify_returns_empty_when_no_candidates(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test that empty list is returned when no candidates exist."""
        mock_storage.get_archival_candidates.return_value = []

        candidates = await tiering_service.identify_archival_candidates(workspace_id)

        assert candidates == []
        mock_storage.get_archival_candidates.assert_called_once()

    @pytest.mark.asyncio
    async def test_identify_returns_candidates(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test that candidates are returned when they exist."""
        test_memories = [
            create_test_memory(f"mem_{i}", workspace_id, importance=0.1, access_count=1)
            for i in range(3)
        ]
        mock_storage.get_archival_candidates.return_value = test_memories

        candidates = await tiering_service.identify_archival_candidates(workspace_id)

        assert len(candidates) == 3
        for memory in candidates:
            assert memory.importance == 0.1
            assert memory.access_count == 1

    @pytest.mark.asyncio
    async def test_identify_uses_default_thresholds(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test that default thresholds are used when not specified."""
        await tiering_service.identify_archival_candidates(workspace_id)

        mock_storage.get_archival_candidates.assert_called_once_with(
            workspace_id=workspace_id,
            max_importance=DEFAULT_MAX_IMPORTANCE,
            max_access_count=DEFAULT_MAX_ACCESS_COUNT,
            older_than_days=DEFAULT_OLDER_THAN_DAYS,
            limit=100,
        )

    @pytest.mark.asyncio
    async def test_identify_uses_custom_thresholds(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test that custom thresholds are used when specified."""
        await tiering_service.identify_archival_candidates(
            workspace_id,
            max_importance=0.5,
            max_access_count=10,
            older_than_days=30,
            limit=50,
        )

        mock_storage.get_archival_candidates.assert_called_once_with(
            workspace_id=workspace_id,
            max_importance=0.5,
            max_access_count=10,
            older_than_days=30,
            limit=50,
        )

    @pytest.mark.asyncio
    async def test_identify_respects_limit(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test that limit parameter is passed correctly."""
        await tiering_service.identify_archival_candidates(workspace_id, limit=25)

        call_args = mock_storage.get_archival_candidates.call_args
        assert call_args.kwargs["limit"] == 25

    @pytest.mark.asyncio
    async def test_identify_propagates_storage_errors(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test that storage errors are propagated."""
        mock_storage.get_archival_candidates.side_effect = RuntimeError("Storage error")

        with pytest.raises(RuntimeError, match="Storage error"):
            await tiering_service.identify_archival_candidates(workspace_id)


class TestArchiveMemories:
    """Tests for archive_memories operation."""

    @pytest.mark.asyncio
    async def test_archive_specific_memories_success(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test archiving specific memory IDs."""
        memory_ids = ["mem_1", "mem_2", "mem_3"]
        mock_storage.archive_memory.return_value = True

        result = await tiering_service.archive_memories(
            workspace_id,
            memory_ids=memory_ids,
        )

        assert isinstance(result, ArchivalResult)
        assert result.archived_count == 3
        assert result.failed_count == 0
        assert result.archived_memory_ids == memory_ids
        assert result.failed_memory_ids == []

    @pytest.mark.asyncio
    async def test_archive_handles_failures(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test handling of archive failures."""
        memory_ids = ["mem_1", "mem_2", "mem_3"]
        # Second memory fails
        mock_storage.archive_memory.side_effect = [True, False, True]

        result = await tiering_service.archive_memories(
            workspace_id,
            memory_ids=memory_ids,
        )

        assert result.archived_count == 2
        assert result.failed_count == 1
        assert "mem_1" in result.archived_memory_ids
        assert "mem_3" in result.archived_memory_ids
        assert "mem_2" in result.failed_memory_ids

    @pytest.mark.asyncio
    async def test_archive_handles_exceptions(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test handling of exceptions during archive."""
        memory_ids = ["mem_1", "mem_2"]
        mock_storage.archive_memory.side_effect = [
            True,
            RuntimeError("Archive error"),
        ]

        result = await tiering_service.archive_memories(
            workspace_id,
            memory_ids=memory_ids,
        )

        assert result.archived_count == 1
        assert result.failed_count == 1
        assert "mem_1" in result.archived_memory_ids
        assert "mem_2" in result.failed_memory_ids

    @pytest.mark.asyncio
    async def test_archive_auto_detect_uses_candidates(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test auto_detect mode finds and archives candidates."""
        candidates = [
            create_test_memory(f"mem_{i}", workspace_id, importance=0.1)
            for i in range(3)
        ]
        mock_storage.get_archival_candidates.return_value = candidates

        result = await tiering_service.archive_memories(
            workspace_id,
            auto_detect=True,
        )

        assert result.archived_count == 3
        mock_storage.get_archival_candidates.assert_called_once()

    @pytest.mark.asyncio
    async def test_archive_auto_detect_with_custom_thresholds(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test auto_detect with custom archival thresholds."""
        mock_storage.get_archival_candidates.return_value = []

        await tiering_service.archive_memories(
            workspace_id,
            auto_detect=True,
            max_importance=0.2,
            max_access_count=3,
            older_than_days=60,
        )

        call_args = mock_storage.get_archival_candidates.call_args
        assert call_args.kwargs["max_importance"] == 0.2
        assert call_args.kwargs["max_access_count"] == 3
        assert call_args.kwargs["older_than_days"] == 60

    @pytest.mark.asyncio
    async def test_archive_empty_call_returns_empty_result(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test that calling without memory_ids or auto_detect returns empty result."""
        result = await tiering_service.archive_memories(workspace_id)

        assert result.archived_count == 0
        assert result.failed_count == 0
        assert result.archived_memory_ids == []
        assert result.failed_memory_ids == []

    @pytest.mark.asyncio
    async def test_archive_respects_batch_size(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test that batch_size limits number of archived memories."""
        memory_ids = [f"mem_{i}" for i in range(10)]

        result = await tiering_service.archive_memories(
            workspace_id,
            memory_ids=memory_ids,
            batch_size=5,
        )

        # Should only archive first 5
        assert result.archived_count == 5
        assert mock_storage.archive_memory.call_count == 5


class TestRestoreMemories:
    """Tests for restore_memories operation."""

    @pytest.mark.asyncio
    async def test_restore_memories_success(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test successful memory restoration."""
        memory_ids = ["mem_1", "mem_2"]

        result = await tiering_service.restore_memories(
            workspace_id,
            memory_ids=memory_ids,
        )

        assert isinstance(result, RestoreResult)
        assert result.restored_count == 2
        assert result.failed_count == 0
        assert result.restored_memory_ids == memory_ids
        assert result.failed_memory_ids == []

    @pytest.mark.asyncio
    async def test_restore_handles_failures(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test handling of restore failures."""
        memory_ids = ["mem_1", "mem_2", "mem_3"]
        mock_storage.restore_memory.side_effect = [True, False, True]

        result = await tiering_service.restore_memories(
            workspace_id,
            memory_ids=memory_ids,
        )

        assert result.restored_count == 2
        assert result.failed_count == 1
        assert "mem_2" in result.failed_memory_ids

    @pytest.mark.asyncio
    async def test_restore_handles_exceptions(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test handling of exceptions during restore."""
        memory_ids = ["mem_1", "mem_2"]
        mock_storage.restore_memory.side_effect = [
            True,
            RuntimeError("Restore error"),
        ]

        result = await tiering_service.restore_memories(
            workspace_id,
            memory_ids=memory_ids,
        )

        assert result.restored_count == 1
        assert result.failed_count == 1
        assert "mem_2" in result.failed_memory_ids

    @pytest.mark.asyncio
    async def test_restore_with_embedding_regeneration(
        self,
        tiering_service_with_embedding: TieringService,
        mock_storage: AsyncMock,
        mock_embedding_service: AsyncMock,
        workspace_id: str,
    ):
        """Test that embeddings are regenerated after restore."""
        memory_ids = ["mem_1"]
        test_memory = create_test_memory("mem_1", workspace_id)
        mock_storage.get_memory.return_value = test_memory

        await tiering_service_with_embedding.restore_memories(
            workspace_id,
            memory_ids=memory_ids,
            regenerate_embeddings=True,
        )

        # Embedding should be regenerated
        mock_embedding_service.embed.assert_called_once_with(test_memory.content)
        mock_storage.update_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_without_embedding_regeneration(
        self,
        tiering_service_with_embedding: TieringService,
        mock_storage: AsyncMock,
        mock_embedding_service: AsyncMock,
        workspace_id: str,
    ):
        """Test that embeddings are not regenerated when disabled."""
        memory_ids = ["mem_1"]

        await tiering_service_with_embedding.restore_memories(
            workspace_id,
            memory_ids=memory_ids,
            regenerate_embeddings=False,
        )

        # Embedding should not be regenerated
        mock_embedding_service.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_restore_handles_missing_memory_for_embedding(
        self,
        tiering_service_with_embedding: TieringService,
        mock_storage: AsyncMock,
        mock_embedding_service: AsyncMock,
        workspace_id: str,
    ):
        """Test handling when memory not found for embedding regeneration."""
        memory_ids = ["mem_1"]
        mock_storage.get_memory.return_value = None

        result = await tiering_service_with_embedding.restore_memories(
            workspace_id,
            memory_ids=memory_ids,
            regenerate_embeddings=True,
        )

        # Restore should still succeed even if embedding regeneration fails
        assert result.restored_count == 1


class TestPromoteHotCandidates:
    """Tests for promote_hot_candidates (warm-up) operation."""

    @pytest.mark.asyncio
    async def test_promote_returns_empty_when_no_candidates(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test that empty result is returned when no hot candidates exist."""
        mock_storage.get_hot_promotion_candidates.return_value = []

        result = await tiering_service.promote_hot_candidates(workspace_id)

        assert result.restored_count == 0
        assert result.failed_count == 0

    @pytest.mark.asyncio
    async def test_promote_finds_frequently_accessed(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test promoting frequently accessed cold memories."""
        # Storage returns only memories that meet the threshold
        hot_memory = create_test_memory("mem_hot", workspace_id, access_count=15)

        mock_storage.get_hot_promotion_candidates.return_value = [hot_memory]

        result = await tiering_service.promote_hot_candidates(
            workspace_id,
            access_threshold=10,
        )

        assert result.restored_count == 1
        assert "mem_hot" in result.restored_memory_ids

    @pytest.mark.asyncio
    async def test_promote_uses_default_threshold(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test that default access threshold is used."""
        # Memory at exactly default threshold - storage returns it as a candidate
        memory = create_test_memory(
            "mem_test",
            workspace_id,
            access_count=DEFAULT_WARMUP_ACCESS_THRESHOLD,
        )
        mock_storage.get_hot_promotion_candidates.return_value = [memory]

        result = await tiering_service.promote_hot_candidates(workspace_id)

        # Should be promoted at exactly default threshold
        assert result.restored_count == 1

    @pytest.mark.asyncio
    async def test_promote_respects_limit(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test that limit parameter restricts promotions."""
        memories = [
            create_test_memory(f"mem_{i}", workspace_id, access_count=20)
            for i in range(10)
        ]
        mock_storage.get_hot_promotion_candidates.return_value = memories

        result = await tiering_service.promote_hot_candidates(
            workspace_id,
            limit=3,
        )

        # Should only restore up to limit
        assert result.restored_count <= 3

    @pytest.mark.asyncio
    async def test_promote_handles_no_cold_tier_support(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test handling when storage doesn't support cold tier."""
        mock_storage.get_cold_storage_stats.side_effect = AttributeError(
            "Cold tier not supported"
        )

        result = await tiering_service.promote_hot_candidates(workspace_id)

        assert result.restored_count == 0
        assert result.failed_count == 0


class TestGetTieringStats:
    """Tests for get_tiering_stats operation."""

    @pytest.mark.asyncio
    async def test_get_stats_returns_complete_stats(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test that complete tiering stats are returned."""
        mock_storage.get_cold_storage_stats.return_value = {
            "hot_memory_count": 100,
            "cold_memory_count": 50,
            "hot_storage_bytes": 1000000,
            "cold_storage_bytes": 50000,
            "compression_ratio": 0.05,
            "estimated_savings_bytes": 950000,
        }
        mock_storage.get_archival_candidates.return_value = [
            create_test_memory(f"mem_{i}", workspace_id)
            for i in range(10)
        ]

        stats = await tiering_service.get_tiering_stats(workspace_id)

        assert isinstance(stats, TieringStats)
        assert stats.hot_memory_count == 100
        assert stats.cold_memory_count == 50
        assert stats.hot_storage_bytes == 1000000
        assert stats.cold_storage_bytes == 50000
        assert stats.compression_ratio == 0.05
        assert stats.estimated_savings_bytes == 950000
        assert stats.archival_candidates_count == 10

    @pytest.mark.asyncio
    async def test_get_stats_without_candidates(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test getting stats without counting candidates."""
        mock_storage.get_cold_storage_stats.return_value = {
            "hot_memory_count": 100,
            "cold_memory_count": 50,
            "hot_storage_bytes": 1000000,
            "cold_storage_bytes": 50000,
            "compression_ratio": 0.05,
            "estimated_savings_bytes": 950000,
        }

        stats = await tiering_service.get_tiering_stats(
            workspace_id,
            include_candidates=False,
        )

        assert stats.archival_candidates_count == 0
        mock_storage.get_archival_candidates.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_stats_handles_missing_cold_tier(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test handling when cold tier stats are not available."""
        mock_storage.get_cold_storage_stats.side_effect = AttributeError(
            "No cold tier"
        )

        stats = await tiering_service.get_tiering_stats(workspace_id)

        assert stats.hot_memory_count == 0
        assert stats.cold_memory_count == 0
        assert stats.compression_ratio == 0.0


class TestRunArchivalCycle:
    """Tests for run_archival_cycle (scheduled job) operation."""

    @pytest.mark.asyncio
    async def test_archival_cycle_archives_candidates(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test that archival cycle archives eligible memories."""
        candidates = [
            create_test_memory(f"mem_{i}", workspace_id, importance=0.1)
            for i in range(5)
        ]
        mock_storage.get_archival_candidates.return_value = candidates

        result = await tiering_service.run_archival_cycle(workspace_id)

        assert result.archived_count == 5
        assert mock_storage.get_archival_candidates.called
        assert mock_storage.archive_memory.call_count == 5

    @pytest.mark.asyncio
    async def test_archival_cycle_uses_custom_params(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test archival cycle with custom parameters."""
        mock_storage.get_archival_candidates.return_value = []

        await tiering_service.run_archival_cycle(
            workspace_id,
            max_importance=0.4,
            max_access_count=8,
            older_than_days=45,
            batch_size=50,
        )

        call_args = mock_storage.get_archival_candidates.call_args
        assert call_args.kwargs["max_importance"] == 0.4
        assert call_args.kwargs["max_access_count"] == 8
        assert call_args.kwargs["older_than_days"] == 45
        assert call_args.kwargs["limit"] == 50


class TestRunWarmupCycle:
    """Tests for run_warmup_cycle (scheduled job) operation."""

    @pytest.mark.asyncio
    async def test_warmup_cycle_promotes_hot_memories(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test that warmup cycle promotes frequently accessed cold memories."""
        hot_memory = create_test_memory("mem_hot", workspace_id, access_count=20)
        mock_storage.get_hot_promotion_candidates.return_value = [hot_memory]

        result = await tiering_service.run_warmup_cycle(workspace_id)

        assert result.restored_count == 1

    @pytest.mark.asyncio
    async def test_warmup_cycle_uses_custom_params(
        self,
        tiering_service: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test warmup cycle with custom parameters."""
        mock_storage.get_hot_promotion_candidates.return_value = []

        await tiering_service.run_warmup_cycle(
            workspace_id,
            access_threshold=5,
            limit=20,
        )

        # Verify promotion candidates were queried (even though no results)
        mock_storage.get_hot_promotion_candidates.assert_called_once()


class TestDataclasses:
    """Tests for data classes (TieringStats, ArchivalResult, RestoreResult)."""

    def test_tiering_stats_creation(self):
        """Test TieringStats dataclass creation."""
        stats = TieringStats(
            hot_memory_count=100,
            cold_memory_count=50,
            hot_storage_bytes=1000000,
            cold_storage_bytes=50000,
            compression_ratio=0.05,
            estimated_savings_bytes=950000,
            archival_candidates_count=10,
        )

        assert stats.hot_memory_count == 100
        assert stats.cold_memory_count == 50
        assert stats.compression_ratio == 0.05

    def test_archival_result_creation(self):
        """Test ArchivalResult dataclass creation."""
        result = ArchivalResult(
            archived_count=5,
            failed_count=2,
            archived_memory_ids=["mem_1", "mem_2"],
            failed_memory_ids=["mem_3"],
        )

        assert result.archived_count == 5
        assert result.failed_count == 2
        assert len(result.archived_memory_ids) == 2

    def test_restore_result_creation(self):
        """Test RestoreResult dataclass creation."""
        result = RestoreResult(
            restored_count=3,
            failed_count=1,
            restored_memory_ids=["mem_1", "mem_2", "mem_3"],
            failed_memory_ids=["mem_4"],
        )

        assert result.restored_count == 3
        assert result.failed_count == 1
        assert len(result.restored_memory_ids) == 3


class TestServiceInitialization:
    """Tests for TieringService initialization."""

    def test_init_with_storage_only(self, mock_storage: AsyncMock, v):
        """Test initialization with storage only."""
        service = TieringService(storage=mock_storage, v=v)

        assert service.storage == mock_storage
        assert service.embedding is None

    def test_init_with_embedding_service(
        self,
        mock_storage: AsyncMock,
        mock_embedding_service: AsyncMock,
        v,
    ):
        """Test initialization with storage and embedding service."""
        service = TieringService(
            storage=mock_storage,
            embedding_service=mock_embedding_service,
            v=v,
        )

        assert service.storage == mock_storage
        assert service.embedding == mock_embedding_service

    def test_default_thresholds(self, mock_storage: AsyncMock, v):
        """Test that default thresholds are set correctly."""
        service = TieringService(storage=mock_storage, v=v)
        assert service.DEFAULT_MAX_IMPORTANCE == 0.3
        assert service.DEFAULT_MAX_ACCESS_COUNT == 5
        assert service.DEFAULT_OLDER_THAN_DAYS == 90
        assert service.DEFAULT_WARMUP_ACCESS_THRESHOLD == 10


class TestRegenerateEmbedding:
    """Tests for _regenerate_embedding helper method."""

    @pytest.mark.asyncio
    async def test_regenerate_embedding_success(
        self,
        tiering_service_with_embedding: TieringService,
        mock_storage: AsyncMock,
        mock_embedding_service: AsyncMock,
        workspace_id: str,
    ):
        """Test successful embedding regeneration."""
        test_memory = create_test_memory("mem_1", workspace_id)
        mock_storage.get_memory.return_value = test_memory

        result = await tiering_service_with_embedding._regenerate_embedding(
            workspace_id,
            "mem_1",
        )

        assert result is True
        mock_embedding_service.embed.assert_called_once_with(test_memory.content)

    @pytest.mark.asyncio
    async def test_regenerate_embedding_no_service(
        self,
        tiering_service: TieringService,
        workspace_id: str,
    ):
        """Test regeneration without embedding service returns False."""
        result = await tiering_service._regenerate_embedding(workspace_id, "mem_1")

        assert result is False

    @pytest.mark.asyncio
    async def test_regenerate_embedding_memory_not_found(
        self,
        tiering_service_with_embedding: TieringService,
        mock_storage: AsyncMock,
        workspace_id: str,
    ):
        """Test regeneration when memory not found."""
        mock_storage.get_memory.return_value = None

        result = await tiering_service_with_embedding._regenerate_embedding(
            workspace_id,
            "nonexistent_mem",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_regenerate_embedding_handles_error(
        self,
        tiering_service_with_embedding: TieringService,
        mock_storage: AsyncMock,
        mock_embedding_service: AsyncMock,
        workspace_id: str,
    ):
        """Test regeneration handles errors gracefully."""
        test_memory = create_test_memory("mem_1", workspace_id)
        mock_storage.get_memory.return_value = test_memory
        mock_embedding_service.embed.side_effect = RuntimeError("Embedding error")

        result = await tiering_service_with_embedding._regenerate_embedding(
            workspace_id,
            "mem_1",
        )

        assert result is False
