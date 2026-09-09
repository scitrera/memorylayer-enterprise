"""
Unit tests for LEANN (Learning-Efficient Approximate Nearest Neighbor) storage.

Tests:
- CSRGraph dataclass operations
- Graph serialization/deserialization (binary CSR format)
- Neighbor lookup operations
- Edge cases and error handling
"""
import pytest
import numpy as np

from memorylayer_saas.storage.leann import (
    CSRGraph,
    LeannStorage,
    LeannDocument,
    LeannGraph,
    ColdStorageStats,
)


class TestCSRGraph:
    """Tests for CSRGraph dataclass."""

    def test_csr_graph_creation(self):
        """Test creating a CSRGraph with basic data."""
        indptr = np.array([0, 2, 4, 6], dtype=np.int64)
        indices = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        data = np.array([0.9, 0.8, 0.9, 0.7, 0.8, 0.7], dtype=np.float32)
        node_ids = ["mem_001", "mem_002", "mem_003"]

        graph = CSRGraph(
            indptr=indptr,
            indices=indices,
            data=data,
            node_ids=node_ids,
        )

        assert graph.node_count == 3
        assert graph.edge_count == 6
        assert len(graph.node_ids) == 3

    def test_csr_graph_node_count(self):
        """Test node_count property calculation."""
        # indptr has n_nodes + 1 elements
        indptr = np.array([0, 3, 6, 8, 10], dtype=np.int64)
        indices = np.array([1, 2, 3, 0, 2, 3, 0, 1, 0, 1], dtype=np.int64)
        data = np.ones(10, dtype=np.float32)
        node_ids = ["mem_001", "mem_002", "mem_003", "mem_004"]

        graph = CSRGraph(
            indptr=indptr,
            indices=indices,
            data=data,
            node_ids=node_ids,
        )

        assert graph.node_count == 4

    def test_csr_graph_edge_count(self):
        """Test edge_count property calculation."""
        indptr = np.array([0, 2, 5, 7], dtype=np.int64)
        indices = np.array([1, 2, 0, 2, 1, 0, 1], dtype=np.int64)
        data = np.ones(7, dtype=np.float32)
        node_ids = ["mem_001", "mem_002", "mem_003"]

        graph = CSRGraph(
            indptr=indptr,
            indices=indices,
            data=data,
            node_ids=node_ids,
        )

        assert graph.edge_count == 7

    def test_csr_graph_get_neighbors(self):
        """Test retrieving neighbors for a specific node."""
        # Graph: node 0 -> [1, 2], node 1 -> [0], node 2 -> [0, 1]
        indptr = np.array([0, 2, 3, 5], dtype=np.int64)
        indices = np.array([1, 2, 0, 0, 1], dtype=np.int64)
        data = np.array([0.9, 0.8, 0.9, 0.8, 0.7], dtype=np.float32)
        node_ids = ["mem_001", "mem_002", "mem_003"]

        graph = CSRGraph(
            indptr=indptr,
            indices=indices,
            data=data,
            node_ids=node_ids,
        )

        # Get neighbors of node 0
        neighbor_indices, weights = graph.get_neighbors(0)
        assert len(neighbor_indices) == 2
        assert list(neighbor_indices) == [1, 2]
        assert pytest.approx(weights[0], rel=1e-5) == 0.9
        assert pytest.approx(weights[1], rel=1e-5) == 0.8

        # Get neighbors of node 1
        neighbor_indices, weights = graph.get_neighbors(1)
        assert len(neighbor_indices) == 1
        assert list(neighbor_indices) == [0]

        # Get neighbors of node 2
        neighbor_indices, weights = graph.get_neighbors(2)
        assert len(neighbor_indices) == 2
        assert list(neighbor_indices) == [0, 1]

    def test_csr_graph_empty(self):
        """Test empty graph handling."""
        indptr = np.array([0], dtype=np.int64)
        indices = np.array([], dtype=np.int64)
        data = np.array([], dtype=np.float32)
        node_ids = []

        graph = CSRGraph(
            indptr=indptr,
            indices=indices,
            data=data,
            node_ids=node_ids,
        )

        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_csr_graph_single_node_no_edges(self):
        """Test graph with single node and no self-loops."""
        indptr = np.array([0, 0], dtype=np.int64)
        indices = np.array([], dtype=np.int64)
        data = np.array([], dtype=np.float32)
        node_ids = ["mem_001"]

        graph = CSRGraph(
            indptr=indptr,
            indices=indices,
            data=data,
            node_ids=node_ids,
        )

        assert graph.node_count == 1
        assert graph.edge_count == 0

        neighbor_indices, weights = graph.get_neighbors(0)
        assert len(neighbor_indices) == 0


class TestLeannStorageSerialization:
    """Tests for LEANN graph serialization/deserialization."""

    @pytest.fixture
    def storage(self):
        """Create LeannStorage instance for testing."""
        return LeannStorage()

    def test_serialize_deserialize_roundtrip(self, storage):
        """Test that serialize/deserialize is lossless."""
        # Create test graph
        indptr = np.array([0, 2, 4, 6], dtype=np.int64)
        indices = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        data = np.array([0.95, 0.85, 0.95, 0.75, 0.85, 0.75], dtype=np.float32)
        node_ids = ["mem_abc123", "mem_def456", "mem_ghi789"]

        original = CSRGraph(
            indptr=indptr,
            indices=indices,
            data=data,
            node_ids=node_ids,
        )

        # Serialize and deserialize
        binary_data = storage.serialize_graph(original)
        restored = storage.deserialize_graph(binary_data)

        # Verify all fields match
        assert restored.node_count == original.node_count
        assert restored.edge_count == original.edge_count
        np.testing.assert_array_equal(restored.indptr, original.indptr)
        np.testing.assert_array_equal(restored.indices, original.indices)
        np.testing.assert_array_almost_equal(restored.data, original.data, decimal=5)
        assert restored.node_ids == original.node_ids

    def test_serialize_large_graph(self, storage):
        """Test serialization of larger graph."""
        n_nodes = 100
        k_neighbors = 32

        # Create a graph with k neighbors per node
        indptr = [0]
        indices = []
        data = []

        for i in range(n_nodes):
            # Each node connects to k random other nodes
            neighbors = [j for j in range(n_nodes) if j != i][:k_neighbors]
            for j in neighbors:
                indices.append(j)
                data.append(0.5 + 0.5 * np.random.random())
            indptr.append(len(indices))

        graph = CSRGraph(
            indptr=np.array(indptr, dtype=np.int64),
            indices=np.array(indices, dtype=np.int64),
            data=np.array(data, dtype=np.float32),
            node_ids=[f"mem_{i:012d}" for i in range(n_nodes)],
        )

        # Serialize
        binary_data = storage.serialize_graph(graph)

        # Verify binary data is compact
        # Each edge: 8 bytes (index) + 4 bytes (weight) = 12 bytes
        # indptr: (n_nodes + 1) * 8 bytes
        # node_ids: variable but included
        assert len(binary_data) > 0

        # Deserialize and verify
        restored = storage.deserialize_graph(binary_data)
        assert restored.node_count == n_nodes
        assert restored.edge_count == graph.edge_count

    def test_serialize_empty_graph(self, storage):
        """Test serialization of empty graph."""
        graph = CSRGraph(
            indptr=np.array([0], dtype=np.int64),
            indices=np.array([], dtype=np.int64),
            data=np.array([], dtype=np.float32),
            node_ids=[],
        )

        binary_data = storage.serialize_graph(graph)
        restored = storage.deserialize_graph(binary_data)

        assert restored.node_count == 0
        assert restored.edge_count == 0
        assert restored.node_ids == []

    def test_serialize_preserves_node_ids_with_special_chars(self, storage):
        """Test that node IDs with special characters are preserved."""
        node_ids = ["mem_123", "mem_456-abc", "mem_789_xyz"]
        indptr = np.array([0, 1, 2, 3], dtype=np.int64)
        indices = np.array([1, 0, 0], dtype=np.int64)
        data = np.array([0.9, 0.9, 0.8], dtype=np.float32)

        graph = CSRGraph(
            indptr=indptr,
            indices=indices,
            data=data,
            node_ids=node_ids,
        )

        binary_data = storage.serialize_graph(graph)
        restored = storage.deserialize_graph(binary_data)

        assert restored.node_ids == node_ids

    def test_deserialize_invalid_magic_number(self, storage):
        """Test that invalid magic number raises error."""
        invalid_data = b"INVALID_MAGIC_NUMBER"

        with pytest.raises(ValueError, match="Invalid CSR magic number"):
            storage.deserialize_graph(invalid_data)

    def test_deserialize_invalid_version(self, storage):
        """Test that unsupported version raises error."""
        # Create valid header but with wrong version
        invalid_data = (
            b"CSR1"  # Magic
            + (999).to_bytes(4, "little")  # Invalid version
            + (0).to_bytes(4, "little")  # node_count
            + (0).to_bytes(4, "little")  # edge_count
            + (0).to_bytes(4, "little")  # num_node_ids
        )

        with pytest.raises(ValueError, match="Unsupported CSR version"):
            storage.deserialize_graph(invalid_data)

    def test_binary_format_header(self, storage):
        """Test binary format header structure."""
        graph = CSRGraph(
            indptr=np.array([0, 2, 4], dtype=np.int64),
            indices=np.array([1, 0, 0, 1], dtype=np.int64),
            data=np.array([0.9, 0.9, 0.8, 0.8], dtype=np.float32),
            node_ids=["mem_001", "mem_002"],
        )

        binary_data = storage.serialize_graph(graph)

        # Check magic number
        assert binary_data[:4] == b"CSR1"

        # Check version (should be 1)
        import struct
        version = struct.unpack("<I", binary_data[4:8])[0]
        assert version == 1

        # Check node count
        node_count = struct.unpack("<I", binary_data[8:12])[0]
        assert node_count == 2

        # Check edge count
        edge_count = struct.unpack("<I", binary_data[12:16])[0]
        assert edge_count == 4


class TestLeannDataclasses:
    """Tests for LEANN data model classes."""

    def test_leann_document_creation(self):
        """Test creating LeannDocument instance."""
        from datetime import datetime, timezone

        doc = LeannDocument(
            id="ldoc_abc123",
            graph_id="leann_xyz789",
            workspace_id="test_workspace",
            memory_id="mem_123456",
            content="Test memory content",
            position=0,
            cold_access_count=5,
            last_cold_access_at=datetime.now(timezone.utc),
            memory_type="semantic",
            memory_subtype="preference",
            importance=0.8,
            tags=["test", "unit"],
            metadata={"source": "test"},
            created_at=datetime.now(timezone.utc),
        )

        assert doc.id == "ldoc_abc123"
        assert doc.cold_access_count == 5
        assert doc.importance == 0.8
        assert "test" in doc.tags

    def test_leann_graph_creation(self):
        """Test creating LeannGraph instance."""
        from datetime import datetime, timezone

        graph = LeannGraph(
            id="leann_abc123",
            workspace_id="test_workspace",
            node_count=100,
            edge_count=3200,
            memory_ids=["mem_001", "mem_002", "mem_003"],
            metadata={"compression_ratio": 0.08},
            created_at=datetime.now(timezone.utc),
        )

        assert graph.id == "leann_abc123"
        assert graph.node_count == 100
        assert graph.edge_count == 3200
        assert len(graph.memory_ids) == 3

    def test_cold_storage_stats_creation(self):
        """Test creating ColdStorageStats instance."""
        stats = ColdStorageStats(
            total_graphs=10,
            total_documents=500,
            total_graph_bytes=102400,
            avg_nodes_per_graph=50.0,
            avg_edges_per_graph=1600.0,
        )

        assert stats.total_graphs == 10
        assert stats.total_documents == 500
        assert stats.total_graph_bytes == 102400
        assert stats.avg_nodes_per_graph == 50.0


class TestLeannStorageEdgeCases:
    """Tests for edge cases in LEANN storage."""

    @pytest.fixture
    def storage(self):
        """Create LeannStorage instance for testing."""
        return LeannStorage()

    def test_serialize_single_node_graph(self, storage):
        """Test serialization of single-node graph."""
        graph = CSRGraph(
            indptr=np.array([0, 0], dtype=np.int64),
            indices=np.array([], dtype=np.int64),
            data=np.array([], dtype=np.float32),
            node_ids=["mem_only_one"],
        )

        binary_data = storage.serialize_graph(graph)
        restored = storage.deserialize_graph(binary_data)

        assert restored.node_count == 1
        assert restored.edge_count == 0
        assert restored.node_ids == ["mem_only_one"]

    def test_serialize_dense_graph(self, storage):
        """Test serialization of fully connected graph."""
        n_nodes = 5

        # Fully connected (excluding self-loops)
        indptr = [0]
        indices = []
        data = []

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    indices.append(j)
                    data.append(0.9)
            indptr.append(len(indices))

        graph = CSRGraph(
            indptr=np.array(indptr, dtype=np.int64),
            indices=np.array(indices, dtype=np.int64),
            data=np.array(data, dtype=np.float32),
            node_ids=[f"mem_{i}" for i in range(n_nodes)],
        )

        # Should have n*(n-1) edges for fully connected
        assert graph.edge_count == n_nodes * (n_nodes - 1)

        binary_data = storage.serialize_graph(graph)
        restored = storage.deserialize_graph(binary_data)

        assert restored.node_count == n_nodes
        assert restored.edge_count == graph.edge_count

    def test_serialize_preserves_float_precision(self, storage):
        """Test that float32 precision is preserved."""
        # Use specific float values that might have precision issues
        data_values = [0.123456789, 0.987654321, 0.555555555]
        data = np.array(data_values, dtype=np.float32)

        graph = CSRGraph(
            indptr=np.array([0, 1, 2, 3], dtype=np.int64),
            indices=np.array([1, 2, 0], dtype=np.int64),
            data=data,
            node_ids=["a", "b", "c"],
        )

        binary_data = storage.serialize_graph(graph)
        restored = storage.deserialize_graph(binary_data)

        # Float32 has ~7 decimal digits of precision
        np.testing.assert_array_almost_equal(restored.data, graph.data, decimal=5)

    def test_serialize_negative_edge_weights(self, storage):
        """Test serialization handles negative similarity values."""
        graph = CSRGraph(
            indptr=np.array([0, 2, 4], dtype=np.int64),
            indices=np.array([1, 0, 0, 1], dtype=np.int64),
            data=np.array([-0.5, 0.5, 0.3, -0.3], dtype=np.float32),
            node_ids=["mem_001", "mem_002"],
        )

        binary_data = storage.serialize_graph(graph)
        restored = storage.deserialize_graph(binary_data)

        np.testing.assert_array_almost_equal(restored.data, graph.data, decimal=5)

    def test_serialize_long_node_ids(self, storage):
        """Test serialization handles long node IDs."""
        long_id = "mem_" + "a" * 100
        graph = CSRGraph(
            indptr=np.array([0, 1, 2], dtype=np.int64),
            indices=np.array([1, 0], dtype=np.int64),
            data=np.array([0.9, 0.9], dtype=np.float32),
            node_ids=[long_id, long_id + "_2"],
        )

        binary_data = storage.serialize_graph(graph)
        restored = storage.deserialize_graph(binary_data)

        assert restored.node_ids[0] == long_id


class TestLeannStorageCompression:
    """Tests for storage compression calculations."""

    @pytest.fixture
    def storage(self):
        """Create LeannStorage instance for testing."""
        return LeannStorage()

    def test_serialized_size_smaller_than_embeddings(self, storage):
        """Test that serialized graph is much smaller than raw embeddings."""
        n_nodes = 100
        k_neighbors = 32
        embedding_dim = 1536

        # Create graph
        indptr = [0]
        indices = []
        data = []

        for i in range(n_nodes):
            neighbors = [(i + j + 1) % n_nodes for j in range(k_neighbors)]
            for j in neighbors:
                indices.append(j)
                data.append(0.8)
            indptr.append(len(indices))

        graph = CSRGraph(
            indptr=np.array(indptr, dtype=np.int64),
            indices=np.array(indices, dtype=np.int64),
            data=np.array(data, dtype=np.float32),
            node_ids=[f"mem_{i:012d}" for i in range(n_nodes)],
        )

        binary_data = storage.serialize_graph(graph)

        # Original embedding storage: n_nodes * embedding_dim * 4 bytes
        original_size = n_nodes * embedding_dim * 4  # float32

        # Graph storage should be much smaller
        compression_ratio = len(binary_data) / original_size

        # Should achieve at least 80% compression (ratio < 0.2)
        assert compression_ratio < 0.2, f"Compression ratio {compression_ratio:.2%} is too high"
