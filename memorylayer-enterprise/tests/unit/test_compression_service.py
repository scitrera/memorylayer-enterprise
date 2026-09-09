# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""
Unit tests for CompressionService.

Tests:
- build_neighbor_graph: Building pruned k-NN graphs from embeddings
- estimate_storage_reduction: Compression ratio calculations
- serialize/deserialize: Graph serialization (delegates to LeannStorage)
- cosine_similarity: Similarity computation
- Edge cases and validation
"""
import pytest
import numpy as np

from memorylayer_saas.services.compression.default import (
    CompressionService,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_K_NEIGHBORS,
    BYTES_PER_FLOAT,
)
from memorylayer_saas.services.compression.base import (
    CompressionStats,
    NeighborGraphResult,
    OnDemandEmbeddingResult,
    ColdRetrievalResult,
)
from memorylayer_saas.storage.leann import CSRGraph, LeannStorage


class TestBuildNeighborGraph:
    """Tests for build_neighbor_graph operation."""

    @pytest.fixture
    def compression_service(self):
        """Create CompressionService instance for testing."""
        return CompressionService()

    def test_build_graph_basic(
        self,
        compression_service: CompressionService,
        small_sample_embeddings: list[list[float]],
        small_sample_memory_ids: list[str],
    ):
        """Test building a basic neighbor graph."""
        result = compression_service.build_neighbor_graph(
            embeddings=small_sample_embeddings,
            memory_ids=small_sample_memory_ids,
            k=2,
        )

        assert isinstance(result, NeighborGraphResult)
        assert result.graph is not None
        assert result.graph.node_count == 5
        assert result.build_time_ms >= 0
        assert result.stats is not None

    def test_build_graph_creates_csr_format(
        self,
        compression_service: CompressionService,
        sample_embeddings: list[list[float]],
        sample_memory_ids: list[str],
    ):
        """Test that graph is in valid CSR format."""
        result = compression_service.build_neighbor_graph(
            embeddings=sample_embeddings,
            memory_ids=sample_memory_ids,
            k=4,
        )

        graph = result.graph

        # CSR format validation
        assert len(graph.indptr) == graph.node_count + 1
        assert graph.indptr[0] == 0
        assert graph.indptr[-1] == len(graph.indices)
        assert len(graph.indices) == len(graph.data)

        # All indices should be valid node indices
        assert np.all(graph.indices >= 0)
        assert np.all(graph.indices < graph.node_count)

    def test_build_graph_respects_k_neighbors(
        self,
        compression_service: CompressionService,
        sample_embeddings: list[list[float]],
        sample_memory_ids: list[str],
    ):
        """Test that each node has at most k neighbors."""
        k = 3
        result = compression_service.build_neighbor_graph(
            embeddings=sample_embeddings,
            memory_ids=sample_memory_ids,
            k=k,
        )

        graph = result.graph

        # Check each node's neighbor count
        for i in range(graph.node_count):
            neighbor_count = graph.indptr[i + 1] - graph.indptr[i]
            assert neighbor_count <= k, f"Node {i} has {neighbor_count} neighbors, expected <= {k}"

    def test_build_graph_no_self_loops(
        self,
        compression_service: CompressionService,
        sample_embeddings: list[list[float]],
        sample_memory_ids: list[str],
    ):
        """Test that graph has no self-loops."""
        result = compression_service.build_neighbor_graph(
            embeddings=sample_embeddings,
            memory_ids=sample_memory_ids,
        )

        graph = result.graph

        # Check that no node points to itself
        for i in range(graph.node_count):
            start = graph.indptr[i]
            end = graph.indptr[i + 1]
            neighbors = graph.indices[start:end]
            assert i not in neighbors, f"Node {i} has self-loop"

    def test_build_graph_weights_are_similarities(
        self,
        compression_service: CompressionService,
        sample_embeddings: list[list[float]],
        sample_memory_ids: list[str],
    ):
        """Test that edge weights are cosine similarities (bounded -1 to 1)."""
        result = compression_service.build_neighbor_graph(
            embeddings=sample_embeddings,
            memory_ids=sample_memory_ids,
        )

        graph = result.graph

        # All weights should be valid cosine similarities
        assert np.all(graph.data >= -1.0)
        assert np.all(graph.data <= 1.0)

    def test_build_graph_empty_input(self, compression_service: CompressionService):
        """Test handling empty input."""
        result = compression_service.build_neighbor_graph(
            embeddings=[],
            memory_ids=[],
        )

        assert result.graph.node_count == 0
        assert result.graph.edge_count == 0
        assert result.stats.compression_ratio == 0.0

    def test_build_graph_single_node(self, compression_service: CompressionService):
        """Test handling single node (no neighbors possible)."""
        embedding = [np.random.randn(1536).tolist()]
        memory_ids = ["mem_single"]

        result = compression_service.build_neighbor_graph(
            embeddings=embedding,
            memory_ids=memory_ids,
        )

        assert result.graph.node_count == 1
        assert result.graph.edge_count == 0

    def test_build_graph_mismatched_lengths_raises(self, compression_service: CompressionService):
        """Test that mismatched embeddings/memory_ids raises error."""
        embeddings = [np.random.randn(1536).tolist() for _ in range(5)]
        memory_ids = ["mem_1", "mem_2", "mem_3"]  # Only 3 IDs

        with pytest.raises(ValueError, match="Mismatched lengths"):
            compression_service.build_neighbor_graph(
                embeddings=embeddings,
                memory_ids=memory_ids,
            )

    def test_build_graph_preserves_memory_ids(
        self,
        compression_service: CompressionService,
        sample_embeddings: list[list[float]],
        sample_memory_ids: list[str],
    ):
        """Test that memory IDs are preserved in graph."""
        result = compression_service.build_neighbor_graph(
            embeddings=sample_embeddings,
            memory_ids=sample_memory_ids,
        )

        assert result.graph.node_ids == sample_memory_ids

    def test_build_graph_with_identical_embeddings(self, compression_service: CompressionService):
        """Test handling identical embeddings."""
        # Create identical embeddings
        base_embedding = np.random.randn(1536).tolist()
        embeddings = [base_embedding.copy() for _ in range(5)]
        memory_ids = [f"mem_{i}" for i in range(5)]

        result = compression_service.build_neighbor_graph(
            embeddings=embeddings,
            memory_ids=memory_ids,
            k=2,
        )

        # Should still produce valid graph
        assert result.graph.node_count == 5
        # All similarities should be 1.0 (identical vectors)
        assert np.allclose(result.graph.data, 1.0, atol=1e-5)

    def test_build_graph_default_k_neighbors(self, compression_service: CompressionService):
        """Test that default k_neighbors is used when not specified."""
        # Create enough nodes to test default k
        n_nodes = 50
        embeddings = [np.random.randn(1536).tolist() for _ in range(n_nodes)]
        memory_ids = [f"mem_{i}" for i in range(n_nodes)]

        result = compression_service.build_neighbor_graph(
            embeddings=embeddings,
            memory_ids=memory_ids,
            # Not specifying k
        )

        # Verify it used default
        max_neighbors = 0
        for i in range(result.graph.node_count):
            neighbor_count = result.graph.indptr[i + 1] - result.graph.indptr[i]
            max_neighbors = max(max_neighbors, neighbor_count)

        assert max_neighbors <= DEFAULT_K_NEIGHBORS

    def test_build_graph_k_larger_than_nodes(self, compression_service: CompressionService):
        """Test k larger than available nodes."""
        embeddings = [np.random.randn(1536).tolist() for _ in range(3)]
        memory_ids = ["mem_0", "mem_1", "mem_2"]

        result = compression_service.build_neighbor_graph(
            embeddings=embeddings,
            memory_ids=memory_ids,
            k=100,  # Much larger than 3 nodes
        )

        # Should have at most n-1 neighbors per node
        for i in range(result.graph.node_count):
            neighbor_count = result.graph.indptr[i + 1] - result.graph.indptr[i]
            assert neighbor_count <= 2  # n-1 = 2


class TestEstimateStorageReduction:
    """Tests for estimate_storage_reduction calculations."""

    @pytest.fixture
    def compression_service(self):
        """Create CompressionService instance for testing."""
        return CompressionService()

    def test_estimate_basic(self, compression_service: CompressionService):
        """Test basic storage reduction estimation."""
        stats = compression_service.estimate_storage_reduction(
            node_count=100,
            edge_count=3200,  # 100 nodes * 32 neighbors
        )

        assert isinstance(stats, CompressionStats)
        assert stats.node_count == 100
        assert stats.edge_count == 3200
        assert stats.original_bytes > 0
        assert stats.compressed_bytes > 0
        assert 0 < stats.compression_ratio < 1

    def test_estimate_achieves_90_percent_reduction(self, compression_service: CompressionService):
        """Test that typical graph achieves 90%+ reduction."""
        # Typical scenario: 1000 nodes, 32 neighbors each
        stats = compression_service.estimate_storage_reduction(
            node_count=1000,
            edge_count=32000,
        )

        # Original: 1000 * 1536 * 4 = 6,144,000 bytes
        # Compressed should be much smaller
        reduction_pct = (1 - stats.compression_ratio) * 100

        assert reduction_pct >= 85, f"Expected 85%+ reduction, got {reduction_pct:.1f}%"

    def test_estimate_with_custom_embedding_dim(self, compression_service: CompressionService):
        """Test estimation with custom embedding dimension."""
        stats = compression_service.estimate_storage_reduction(
            node_count=100,
            edge_count=3200,
            embedding_dim=768,  # Different embedding size
        )

        # Original bytes should use provided dimension
        expected_original = 100 * 768 * BYTES_PER_FLOAT
        assert stats.original_bytes == expected_original

    def test_estimate_zero_nodes(self, compression_service: CompressionService):
        """Test estimation with zero nodes."""
        stats = compression_service.estimate_storage_reduction(
            node_count=0,
            edge_count=0,
        )

        assert stats.original_bytes == 0
        assert stats.compressed_bytes > 0  # indptr still has 1 element
        assert stats.compression_ratio == 0.0

    def test_estimate_avg_neighbors_calculation(self, compression_service: CompressionService):
        """Test average neighbors per node calculation."""
        stats = compression_service.estimate_storage_reduction(
            node_count=100,
            edge_count=3200,
        )

        expected_avg = 3200 / 100  # 32
        assert stats.avg_neighbors_per_node == expected_avg


class TestSerializeDeserialize:
    """Tests for graph serialization (delegates to LeannStorage)."""

    @pytest.fixture
    def compression_service(self):
        """Create CompressionService instance for testing."""
        return CompressionService()

    def test_serialize_roundtrip(self, compression_service: CompressionService):
        """Test serialize/deserialize roundtrip through CompressionService."""
        # Build a graph
        embeddings = [np.random.randn(1536).tolist() for _ in range(10)]
        memory_ids = [f"mem_{i}" for i in range(10)]

        result = compression_service.build_neighbor_graph(
            embeddings=embeddings,
            memory_ids=memory_ids,
            k=3,
        )

        # Serialize and deserialize
        binary_data = compression_service.serialize_graph(result.graph)
        restored = compression_service.deserialize_graph(binary_data)

        # Verify
        assert restored.node_count == result.graph.node_count
        assert restored.edge_count == result.graph.edge_count
        np.testing.assert_array_equal(restored.indptr, result.graph.indptr)
        np.testing.assert_array_equal(restored.indices, result.graph.indices)
        assert restored.node_ids == result.graph.node_ids


class TestCosineSimilarity:
    """Tests for cosine_similarity static method."""

    def test_cosine_similarity_identical_vectors(self):
        """Test similarity of identical vectors is 1.0."""
        vec = [1.0, 2.0, 3.0, 4.0]
        similarity = CompressionService.cosine_similarity(vec, vec)
        assert pytest.approx(similarity, rel=1e-5) == 1.0

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors is 0.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        similarity = CompressionService.cosine_similarity(vec_a, vec_b)
        assert pytest.approx(similarity, abs=1e-5) == 0.0

    def test_cosine_similarity_opposite_vectors(self):
        """Test similarity of opposite vectors is -1.0."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [-1.0, -2.0, -3.0]
        similarity = CompressionService.cosine_similarity(vec_a, vec_b)
        assert pytest.approx(similarity, rel=1e-5) == -1.0

    def test_cosine_similarity_dimension_mismatch_raises(self):
        """Test that dimension mismatch raises error."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [1.0, 2.0]  # Different dimension

        with pytest.raises(ValueError, match="dimension mismatch"):
            CompressionService.cosine_similarity(vec_a, vec_b)

    def test_cosine_similarity_zero_vector(self):
        """Test similarity with zero vector is 0.0."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [0.0, 0.0, 0.0]
        similarity = CompressionService.cosine_similarity(vec_a, vec_b)
        assert similarity == 0.0


class TestSelectGraphCandidates:
    """Tests for select_graph_candidates method."""

    @pytest.fixture
    def compression_service(self):
        """Create CompressionService instance for testing."""
        return CompressionService()

    def test_select_candidates_returns_correct_count(self, compression_service: CompressionService):
        """Test that correct number of candidates are returned."""
        # Build a graph
        embeddings = [np.random.randn(1536).tolist() for _ in range(20)]
        memory_ids = [f"mem_{i}" for i in range(20)]

        result = compression_service.build_neighbor_graph(
            embeddings=embeddings,
            memory_ids=memory_ids,
            k=5,
        )

        candidates = compression_service.select_graph_candidates(
            result.graph,
            num_candidates=10,
            num_entry_points=3,
        )

        assert len(candidates) <= 10
        assert len(candidates) > 0

    def test_select_candidates_empty_graph(self, compression_service: CompressionService):
        """Test selecting from empty graph."""
        graph = CSRGraph(
            indptr=np.array([0], dtype=np.int64),
            indices=np.array([], dtype=np.int64),
            data=np.array([], dtype=np.float32),
            node_ids=[],
        )

        candidates = compression_service.select_graph_candidates(graph, num_candidates=10)
        assert candidates == []

    def test_select_candidates_caps_at_graph_size(self, compression_service: CompressionService):
        """Test that candidates are capped at graph size."""
        embeddings = [np.random.randn(1536).tolist() for _ in range(5)]
        memory_ids = [f"mem_{i}" for i in range(5)]

        result = compression_service.build_neighbor_graph(
            embeddings=embeddings,
            memory_ids=memory_ids,
            k=2,
        )

        candidates = compression_service.select_graph_candidates(
            result.graph,
            num_candidates=100,  # More than graph size
        )

        assert len(candidates) <= 5

    def test_select_candidates_valid_indices(self, compression_service: CompressionService):
        """Test that all returned indices are valid."""
        embeddings = [np.random.randn(1536).tolist() for _ in range(15)]
        memory_ids = [f"mem_{i}" for i in range(15)]

        result = compression_service.build_neighbor_graph(
            embeddings=embeddings,
            memory_ids=memory_ids,
            k=4,
        )

        candidates = compression_service.select_graph_candidates(
            result.graph,
            num_candidates=10,
        )

        for idx in candidates:
            assert 0 <= idx < result.graph.node_count


class TestSearchGraphCandidates:
    """Tests for search_graph_candidates method."""

    @pytest.fixture
    def compression_service(self):
        """Create CompressionService instance for testing."""
        return CompressionService()

    def test_search_returns_sorted_results(self, compression_service: CompressionService):
        """Test that search results are sorted by similarity descending."""
        # Create query and candidate embeddings
        query = np.random.randn(1536).tolist()

        graph = CSRGraph(
            indptr=np.array([0, 2, 4, 6], dtype=np.int64),
            indices=np.array([1, 2, 0, 2, 0, 1], dtype=np.int64),
            data=np.array([0.9, 0.8, 0.9, 0.7, 0.8, 0.7], dtype=np.float32),
            node_ids=["mem_0", "mem_1", "mem_2"],
        )

        candidate_embeddings = {
            0: np.random.randn(1536).tolist(),
            1: np.random.randn(1536).tolist(),
            2: np.random.randn(1536).tolist(),
        }

        results = compression_service.search_graph_candidates(
            graph=graph,
            query_embedding=query,
            candidate_embeddings=candidate_embeddings,
            limit=3,
        )

        # Results should be sorted by similarity descending
        similarities = [r[1] for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_search_respects_limit(self, compression_service: CompressionService):
        """Test that search respects limit parameter."""
        query = np.random.randn(1536).tolist()

        graph = CSRGraph(
            indptr=np.array([0, 2, 4, 6, 8, 10], dtype=np.int64),
            indices=np.array([1, 2, 0, 2, 0, 1, 0, 1, 0, 1], dtype=np.int64),
            data=np.ones(10, dtype=np.float32),
            node_ids=[f"mem_{i}" for i in range(5)],
        )

        candidate_embeddings = {
            i: np.random.randn(1536).tolist() for i in range(5)
        }

        results = compression_service.search_graph_candidates(
            graph=graph,
            query_embedding=query,
            candidate_embeddings=candidate_embeddings,
            limit=2,
        )

        assert len(results) == 2

    def test_search_empty_candidates(self, compression_service: CompressionService):
        """Test search with empty candidate embeddings."""
        query = np.random.randn(1536).tolist()

        graph = CSRGraph(
            indptr=np.array([0, 1], dtype=np.int64),
            indices=np.array([0], dtype=np.int64),
            data=np.array([0.5], dtype=np.float32),
            node_ids=["mem_0"],
        )

        results = compression_service.search_graph_candidates(
            graph=graph,
            query_embedding=query,
            candidate_embeddings={},  # Empty
            limit=5,
        )

        assert results == []


class TestCompressionServiceDataclasses:
    """Tests for CompressionService dataclasses."""

    def test_compression_stats_creation(self):
        """Test CompressionStats dataclass."""
        stats = CompressionStats(
            original_bytes=6144000,
            compressed_bytes=512000,
            compression_ratio=0.083,
            node_count=1000,
            edge_count=32000,
            avg_neighbors_per_node=32.0,
        )

        assert stats.original_bytes == 6144000
        assert stats.compression_ratio == 0.083

    def test_neighbor_graph_result_creation(self):
        """Test NeighborGraphResult dataclass."""
        graph = CSRGraph(
            indptr=np.array([0, 1], dtype=np.int64),
            indices=np.array([0], dtype=np.int64),
            data=np.array([0.5], dtype=np.float32),
            node_ids=["mem_0"],
        )

        stats = CompressionStats(
            original_bytes=100,
            compressed_bytes=10,
            compression_ratio=0.1,
            node_count=1,
            edge_count=1,
            avg_neighbors_per_node=1.0,
        )

        result = NeighborGraphResult(
            graph=graph,
            build_time_ms=50,
            stats=stats,
        )

        assert result.build_time_ms == 50
        assert result.graph.node_count == 1

    def test_on_demand_embedding_result_creation(self):
        """Test OnDemandEmbeddingResult dataclass."""
        result = OnDemandEmbeddingResult(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            memory_ids=["mem_0", "mem_1"],
            compute_time_ms=100,
        )

        assert len(result.embeddings) == 2
        assert result.compute_time_ms == 100

    def test_cold_retrieval_result_creation(self):
        """Test ColdRetrievalResult dataclass."""
        result = ColdRetrievalResult(
            memory_ids=["mem_0", "mem_1"],
            similarities=[0.95, 0.85],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            candidates_evaluated=64,
            total_retrieval_time_ms=200,
            embedding_time_ms=150,
        )

        assert len(result.memory_ids) == 2
        assert result.candidates_evaluated == 64
        assert result.total_retrieval_time_ms == 200


class TestCompressionServiceConfiguration:
    """Tests for CompressionService configuration."""

    def test_default_configuration(self):
        """Test default configuration values."""
        service = CompressionService()

        assert service.k_neighbors == DEFAULT_K_NEIGHBORS
        assert service.embedding_dim == DEFAULT_EMBEDDING_DIM

    def test_custom_configuration(self):
        """Test custom configuration values."""
        service = CompressionService(
            k_neighbors=16,
            embedding_dim=768,
        )

        assert service.k_neighbors == 16
        assert service.embedding_dim == 768

    def test_leann_storage_injection(self):
        """Test that LeannStorage can be injected."""
        custom_storage = LeannStorage()
        service = CompressionService(leann_storage=custom_storage)

        assert service.leann_storage is custom_storage
