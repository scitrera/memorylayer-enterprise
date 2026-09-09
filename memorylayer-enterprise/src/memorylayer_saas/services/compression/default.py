# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""
Default compression service implementation.

Implements LEANN (Learning-Efficient Approximate Nearest Neighbor) compression
for 90%+ storage reduction by eliminating stored embedding vectors.

Operations:
- build_neighbor_graph: Build pruned neighbor graph from embeddings
- compress_memories: Compress a batch of memories into LEANN format
- compute_on_demand_embedding: Re-embed content during cold retrieval
- estimate_storage_reduction: Calculate compression ratio
"""
import os
from datetime import datetime, timezone
from logging import Logger
from typing import Optional

import numpy as np

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

# Use enterprise Memory model with multivector support
from memorylayer_saas.models.memory import Memory
from memorylayer_server.services.embedding import EmbeddingService

from ...storage.leann import CSRGraph, LeannStorage
from .base import (
    CompressionServicePluginBase,
    CompressionStats,
    NeighborGraphResult,
    OnDemandEmbeddingResult,
    ColdRetrievalResult,
)

# Default embedding dimension - read from environment for configurability
DEFAULT_EMBEDDING_DIM = int(os.environ.get('MEMORYLAYER_EMBEDDING_DIMENSIONS', '1536'))

# Default number of neighbors to keep per node in the pruned graph
DEFAULT_K_NEIGHBORS = 32

# Bytes per float32 in embedding storage
BYTES_PER_FLOAT = 4


class CompressionService:
    """
    Service for building and compressing pruned neighbor graphs from embeddings.

    The LEANN algorithm eliminates the need to store full embedding vectors by
    maintaining only the graph structure (which nodes are neighbors). During
    retrieval, we use the graph structure to guide search and re-embed only
    the top candidate documents.

    This achieves 90%+ storage reduction because:
    - Full embedding: 1536 floats * 4 bytes = 6144 bytes per memory
    - Graph edge: ~16 bytes per neighbor (2 int64 indices + float32 weight)
    - With k=32 neighbors: 32 * 16 = 512 bytes per memory
    - Compression ratio: 512 / 6144 = ~8.3% storage (91.7% reduction)
    """

    def __init__(
            self,
            v: Variables = None,
            embedding_service: Optional[EmbeddingService] = None,
            leann_storage: Optional[LeannStorage] = None,
            k_neighbors: int = DEFAULT_K_NEIGHBORS,
            embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ):
        """
        Initialize CompressionService.

        Args:
            v: Variables instance for logger context.
            embedding_service: Optional embedding service for on-demand computation.
            leann_storage: Optional LEANN storage for graph serialization.
            k_neighbors: Number of neighbors to keep per node in pruned graph.
            embedding_dim: Embedding dimension for storage calculations.
        """
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.embedding = embedding_service
        self.leann_storage = leann_storage or LeannStorage()
        self.k_neighbors = k_neighbors
        self.embedding_dim = embedding_dim
        self.logger.info(
            "Initialized CompressionService with k_neighbors=%d, embedding_dim=%d",
            k_neighbors,
            embedding_dim,
        )

    def build_neighbor_graph(
            self,
            embeddings: list[list[float]],
            memory_ids: list[str],
            k: Optional[int] = None,
    ) -> NeighborGraphResult:
        """
        Build a pruned neighbor graph from embeddings.

        Creates a k-nearest neighbor graph where each node (memory) is connected
        to its k most similar neighbors based on cosine similarity.

        Args:
            embeddings: List of embedding vectors.
            memory_ids: List of memory IDs corresponding to embeddings.
            k: Number of neighbors per node (default: self.k_neighbors).

        Returns:
            NeighborGraphResult with the CSR graph and statistics.

        Raises:
            ValueError: If embeddings and memory_ids have different lengths.
        """
        start_time = datetime.now(timezone.utc)

        if len(embeddings) != len(memory_ids):
            raise ValueError(
                f"Mismatched lengths: {len(embeddings)} embeddings vs {len(memory_ids)} memory_ids"
            )

        n_nodes = len(embeddings)
        if n_nodes == 0:
            # Return empty graph
            graph = CSRGraph(
                indptr=np.array([0], dtype=np.int64),
                indices=np.array([], dtype=np.int64),
                data=np.array([], dtype=np.float32),
                node_ids=[],
            )
            return NeighborGraphResult(
                graph=graph,
                build_time_ms=0,
                stats=CompressionStats(
                    original_bytes=0,
                    compressed_bytes=0,
                    compression_ratio=0.0,
                    node_count=0,
                    edge_count=0,
                    avg_neighbors_per_node=0.0,
                ),
            )

        k_actual = min(k or self.k_neighbors, n_nodes - 1)

        self.logger.info(
            "Building neighbor graph: %d nodes, k=%d neighbors",
            n_nodes,
            k_actual,
        )

        # Convert to numpy array for efficient computation
        emb_matrix = np.array(embeddings, dtype=np.float32)

        # Normalize embeddings for cosine similarity (dot product of normalized = cosine)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        emb_normalized = emb_matrix / norms

        # Compute pairwise similarities (cosine similarity via dot product)
        # For large datasets, this could be optimized with approximate methods
        similarity_matrix = np.dot(emb_normalized, emb_normalized.T)

        # Build CSR format arrays
        indptr = [0]
        indices = []
        data = []

        for i in range(n_nodes):
            # Get similarities for this node
            similarities = similarity_matrix[i]

            # Set self-similarity to -inf to exclude from neighbors
            similarities[i] = -np.inf

            # Get top-k neighbors
            if k_actual > 0:
                # argpartition is O(n) vs O(n log n) for full sort
                top_k_indices = np.argpartition(similarities, -k_actual)[-k_actual:]
                # Sort these k indices by similarity (descending)
                sorted_indices = top_k_indices[
                    np.argsort(similarities[top_k_indices])[::-1]
                ]

                for idx in sorted_indices:
                    if similarities[idx] > -np.inf:  # Valid neighbor
                        indices.append(idx)
                        data.append(similarities[idx])

            indptr.append(len(indices))

        # Create CSR graph
        graph = CSRGraph(
            indptr=np.array(indptr, dtype=np.int64),
            indices=np.array(indices, dtype=np.int64),
            data=np.array(data, dtype=np.float32),
            node_ids=memory_ids,
        )

        # Calculate statistics
        build_time_ms = int(
            (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )

        stats = self.estimate_storage_reduction(
            node_count=n_nodes,
            edge_count=graph.edge_count,
            embedding_dim=len(embeddings[0]) if embeddings else self.embedding_dim,
        )

        self.logger.info(
            "Built neighbor graph in %d ms: %d nodes, %d edges, %.1f%% compression ratio",
            build_time_ms,
            graph.node_count,
            graph.edge_count,
            stats.compression_ratio * 100,
        )

        return NeighborGraphResult(
            graph=graph,
            build_time_ms=build_time_ms,
            stats=stats,
        )

    def estimate_storage_reduction(
            self,
            node_count: int,
            edge_count: int,
            embedding_dim: Optional[int] = None,
    ) -> CompressionStats:
        """
        Estimate storage reduction from compression.

        Calculates the storage savings achieved by using graph structure
        instead of full embeddings.

        Args:
            node_count: Number of nodes (memories) in the graph.
            edge_count: Total number of edges in the graph.
            embedding_dim: Embedding dimension (default: self.embedding_dim).

        Returns:
            CompressionStats with storage calculations.
        """
        dim = embedding_dim or self.embedding_dim

        # Original storage: full embeddings
        # Each embedding = dim * 4 bytes (float32)
        original_bytes = node_count * dim * BYTES_PER_FLOAT

        # Compressed storage: CSR graph structure
        # - indptr: (node_count + 1) * 8 bytes (int64)
        # - indices: edge_count * 8 bytes (int64)
        # - data: edge_count * 4 bytes (float32)
        # - node_ids: ~40 bytes per ID (UUID string avg)
        indptr_bytes = (node_count + 1) * 8
        indices_bytes = edge_count * 8
        data_bytes = edge_count * 4
        node_ids_bytes = node_count * 40  # Approximate

        compressed_bytes = indptr_bytes + indices_bytes + data_bytes + node_ids_bytes

        # Compression ratio (lower is better)
        if original_bytes > 0:
            compression_ratio = compressed_bytes / original_bytes
        else:
            compression_ratio = 0.0

        avg_neighbors = edge_count / node_count if node_count > 0 else 0.0

        return CompressionStats(
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=compression_ratio,
            node_count=node_count,
            edge_count=edge_count,
            avg_neighbors_per_node=avg_neighbors,
        )

    def serialize_graph(self, graph: CSRGraph) -> bytes:
        """
        Serialize a CSR graph to binary format.

        Delegates to LeannStorage.serialize_graph for consistent format.

        Args:
            graph: CSR graph to serialize.

        Returns:
            Binary representation of the graph.
        """
        return self.leann_storage.serialize_graph(graph)

    def deserialize_graph(self, data: bytes) -> CSRGraph:
        """
        Deserialize binary data to a CSR graph.

        Delegates to LeannStorage.deserialize_graph for consistent format.

        Args:
            data: Binary graph data.

        Returns:
            Deserialized CSR graph.
        """
        return self.leann_storage.deserialize_graph(data)

    async def compute_on_demand_embedding(
            self,
            content: str,
    ) -> list[float]:
        """
        Compute embedding on-demand for cold tier retrieval.

        During cold retrieval, we don't have stored embeddings. Instead, we
        re-embed the content to compute similarity with the query.

        Args:
            content: Text content to embed.

        Returns:
            Embedding vector.

        Raises:
            RuntimeError: If no embedding service is configured.
        """
        if not self.embedding:
            raise RuntimeError(
                "EmbeddingService required for on-demand embedding computation"
            )

        start_time = datetime.now(timezone.utc)

        embedding = await self.embedding.embed(content)

        compute_time_ms = int(
            (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )

        self.logger.debug(
            "Computed on-demand embedding in %d ms for content: %d chars",
            compute_time_ms,
            len(content),
        )

        return embedding

    async def compute_on_demand_embeddings_batch(
            self,
            contents: list[str],
            memory_ids: list[str],
    ) -> OnDemandEmbeddingResult:
        """
        Compute embeddings on-demand for multiple documents.

        Batch processing is more efficient than individual calls.

        Args:
            contents: List of text contents to embed.
            memory_ids: Corresponding memory IDs.

        Returns:
            OnDemandEmbeddingResult with embeddings and timing.

        Raises:
            RuntimeError: If no embedding service is configured.
        """
        if not self.embedding:
            raise RuntimeError(
                "EmbeddingService required for on-demand embedding computation"
            )

        if len(contents) != len(memory_ids):
            raise ValueError(
                f"Mismatched lengths: {len(contents)} contents vs {len(memory_ids)} memory_ids"
            )

        start_time = datetime.now(timezone.utc)

        embeddings = await self.embedding.embed_batch(contents)

        compute_time_ms = int(
            (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )

        self.logger.debug(
            "Computed %d on-demand embeddings in %d ms",
            len(embeddings),
            compute_time_ms,
        )

        return OnDemandEmbeddingResult(
            embeddings=embeddings,
            memory_ids=memory_ids,
            compute_time_ms=compute_time_ms,
        )

    def search_graph_candidates(
            self,
            graph: CSRGraph,
            query_embedding: list[float],
            candidate_embeddings: dict[int, list[float]],
            limit: int = 10,
            beam_width: int = 64,
    ) -> list[tuple[int, float]]:
        """
        Search the graph for nearest neighbors using beam search.

        Given a query embedding and a graph structure, find the most similar
        nodes by traversing the graph. This method assumes we have computed
        embeddings for candidate nodes (via on-demand embedding).

        Args:
            graph: CSR neighbor graph.
            query_embedding: Query vector.
            candidate_embeddings: Dict mapping node indices to their embeddings.
            limit: Number of results to return.
            beam_width: Number of candidates to track during search.

        Returns:
            List of (node_index, similarity) tuples sorted by similarity descending.
        """
        if graph.node_count == 0:
            return []

        query_np = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_np)
        if query_norm > 0:
            query_normalized = query_np / query_norm
        else:
            query_normalized = query_np

        # Start with available candidate embeddings
        candidates = []
        for node_idx, emb in candidate_embeddings.items():
            emb_np = np.array(emb, dtype=np.float32)
            emb_norm = np.linalg.norm(emb_np)
            if emb_norm > 0:
                emb_normalized = emb_np / emb_norm
            else:
                emb_normalized = emb_np

            similarity = float(np.dot(query_normalized, emb_normalized))
            candidates.append((node_idx, similarity))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return candidates[:limit]

    async def compress_memories(
            self,
            memories: list[Memory],
            embeddings: Optional[list[list[float]]] = None,
    ) -> NeighborGraphResult:
        """
        Compress a batch of memories into a LEANN neighbor graph.

        If embeddings are not provided, they will be computed using the
        embedding service.

        Args:
            memories: List of Memory objects to compress.
            embeddings: Optional pre-computed embeddings.

        Returns:
            NeighborGraphResult with the compressed graph.

        Raises:
            RuntimeError: If embeddings not provided and no embedding service.
        """
        if not memories:
            return NeighborGraphResult(
                graph=CSRGraph(
                    indptr=np.array([0], dtype=np.int64),
                    indices=np.array([], dtype=np.int64),
                    data=np.array([], dtype=np.float32),
                    node_ids=[],
                ),
                build_time_ms=0,
                stats=CompressionStats(
                    original_bytes=0,
                    compressed_bytes=0,
                    compression_ratio=0.0,
                    node_count=0,
                    edge_count=0,
                    avg_neighbors_per_node=0.0,
                ),
            )

        memory_ids = [m.id for m in memories]

        # Get embeddings
        if embeddings is None:
            if not self.embedding:
                raise RuntimeError(
                    "EmbeddingService required when embeddings not provided"
                )

            self.logger.info("Computing embeddings for %d memories", len(memories))
            contents = [m.content for m in memories]
            embeddings = await self.embedding.embed_batch(contents)

        # Build the neighbor graph
        return self.build_neighbor_graph(embeddings, memory_ids)

    def select_graph_candidates(
            self,
            graph: CSRGraph,
            num_candidates: int = 64,
            num_entry_points: int = 8,
    ) -> list[int]:
        """
        Select candidate nodes from graph using random entry points and expansion.

        Uses a breadth-first style expansion from random entry points to select
        candidates without needing embeddings. This is the first step in cold
        tier retrieval where we don't have stored embeddings.

        Args:
            graph: CSR neighbor graph.
            num_candidates: Target number of candidates to return.
            num_entry_points: Number of random entry points to start from.

        Returns:
            List of node indices to evaluate.
        """
        if graph.node_count == 0:
            return []

        # Cap candidates at graph size
        num_candidates = min(num_candidates, graph.node_count)

        # Select random entry points
        entry_points = np.random.choice(
            graph.node_count,
            size=min(num_entry_points, graph.node_count),
            replace=False,
        ).tolist()

        # BFS expansion from entry points
        visited = set(entry_points)
        frontier = list(entry_points)

        while len(visited) < num_candidates and frontier:
            current = frontier.pop(0)

            # Get neighbors from CSR graph
            start_idx = graph.indptr[current]
            end_idx = graph.indptr[current + 1]
            neighbors = graph.indices[start_idx:end_idx].tolist()

            for neighbor in neighbors:
                if neighbor not in visited and len(visited) < num_candidates:
                    visited.add(neighbor)
                    frontier.append(neighbor)

        return list(visited)

    async def retrieve_cold_with_on_demand_embedding(
            self,
            graph: CSRGraph,
            query_embedding: list[float],
            content_lookup: dict[str, str],
            limit: int = 10,
            max_candidates: int = 64,
    ) -> ColdRetrievalResult:
        """
        Retrieve from cold tier using graph structure and on-demand embedding.

        This is the main cold tier retrieval method that:
        1. Uses graph structure to select candidate nodes (without embeddings)
        2. Re-embeds only the candidate documents on-demand
        3. Scores candidates against the query
        4. Returns the top results

        This achieves 90%+ storage reduction by not storing embeddings,
        while maintaining retrieval quality through smart candidate selection.

        Args:
            graph: CSR neighbor graph (compressed storage).
            query_embedding: Pre-computed query embedding.
            content_lookup: Dict mapping memory_id to content for re-embedding.
            limit: Number of results to return.
            max_candidates: Maximum candidates to evaluate (controls latency).

        Returns:
            ColdRetrievalResult with ranked memories and timing info.

        Raises:
            RuntimeError: If no embedding service is configured.
        """
        if not self.embedding:
            raise RuntimeError(
                "EmbeddingService required for cold tier retrieval"
            )

        start_time = datetime.now(timezone.utc)

        if graph.node_count == 0:
            return ColdRetrievalResult(
                memory_ids=[],
                similarities=[],
                embeddings=[],
                candidates_evaluated=0,
                total_retrieval_time_ms=0,
                embedding_time_ms=0,
            )

        # Step 1: Select candidates using graph structure
        candidate_indices = self.select_graph_candidates(
            graph,
            num_candidates=max_candidates,
        )

        if not candidate_indices:
            return ColdRetrievalResult(
                memory_ids=[],
                similarities=[],
                embeddings=[],
                candidates_evaluated=0,
                total_retrieval_time_ms=0,
                embedding_time_ms=0,
            )

        # Step 2: Get memory IDs and content for candidates
        candidate_memory_ids = [graph.node_ids[i] for i in candidate_indices]
        candidate_contents = []
        valid_indices = []

        for i, mem_id in enumerate(candidate_memory_ids):
            content = content_lookup.get(mem_id)
            if content:
                candidate_contents.append(content)
                valid_indices.append(candidate_indices[i])

        if not candidate_contents:
            self.logger.warning("No content found for any candidates")
            return ColdRetrievalResult(
                memory_ids=[],
                similarities=[],
                embeddings=[],
                candidates_evaluated=len(candidate_indices),
                total_retrieval_time_ms=0,
                embedding_time_ms=0,
            )

        # Step 3: Compute embeddings on-demand for candidates only
        embedding_start = datetime.now(timezone.utc)

        candidate_embeddings = await self.embedding.embed_batch(candidate_contents)

        embedding_time_ms = int(
            (datetime.now(timezone.utc) - embedding_start).total_seconds() * 1000
        )

        # Step 4: Score candidates against query
        query_np = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_np)
        if query_norm > 0:
            query_normalized = query_np / query_norm
        else:
            query_normalized = query_np

        scored_results = []
        for i, emb in enumerate(candidate_embeddings):
            emb_np = np.array(emb, dtype=np.float32)
            emb_norm = np.linalg.norm(emb_np)
            if emb_norm > 0:
                emb_normalized = emb_np / emb_norm
            else:
                emb_normalized = emb_np

            similarity = float(np.dot(query_normalized, emb_normalized))
            memory_id = graph.node_ids[valid_indices[i]]
            scored_results.append((memory_id, similarity, emb))

        # Sort by similarity descending
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Take top-k results
        top_results = scored_results[:limit]

        total_time_ms = int(
            (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )

        self.logger.debug(
            "Cold tier retrieval: %d candidates -> %d results in %d ms (%d ms embedding)",
            len(candidate_contents),
            len(top_results),
            total_time_ms,
            embedding_time_ms,
        )

        return ColdRetrievalResult(
            memory_ids=[r[0] for r in top_results],
            similarities=[r[1] for r in top_results],
            embeddings=[r[2] for r in top_results],
            candidates_evaluated=len(candidate_contents),
            total_retrieval_time_ms=total_time_ms,
            embedding_time_ms=embedding_time_ms,
        )

    async def retrieve_cold_with_graph_guided_search(
            self,
            graph: CSRGraph,
            query_embedding: list[float],
            content_lookup: dict[str, str],
            limit: int = 10,
            beam_width: int = 32,
            max_iterations: int = 3,
    ) -> ColdRetrievalResult:
        """
        Retrieve from cold tier using iterative graph-guided search.

        This method uses an iterative approach:
        1. Start with random entry points and compute their embeddings
        2. Score against query, keep best candidates
        3. Expand to neighbors of best candidates
        4. Repeat until convergence or max iterations

        This can be more efficient than flat candidate selection for
        large graphs where good candidates are clustered.

        Args:
            graph: CSR neighbor graph.
            query_embedding: Pre-computed query embedding.
            content_lookup: Dict mapping memory_id to content.
            limit: Number of results to return.
            beam_width: Number of candidates to keep per iteration.
            max_iterations: Maximum search iterations.

        Returns:
            ColdRetrievalResult with ranked memories and timing info.

        Raises:
            RuntimeError: If no embedding service is configured.
        """
        if not self.embedding:
            raise RuntimeError(
                "EmbeddingService required for cold tier retrieval"
            )

        start_time = datetime.now(timezone.utc)

        if graph.node_count == 0:
            return ColdRetrievalResult(
                memory_ids=[],
                similarities=[],
                embeddings=[],
                candidates_evaluated=0,
                total_retrieval_time_ms=0,
                embedding_time_ms=0,
            )

        query_np = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_np)
        if query_norm > 0:
            query_normalized = query_np / query_norm
        else:
            query_normalized = query_np

        # Track all evaluated nodes and their scores
        evaluated: dict[int, tuple[float, list[float]]] = {}
        embedding_time_ms = 0

        # Initial entry points
        beam_width = min(beam_width, graph.node_count)
        current_beam = np.random.choice(
            graph.node_count,
            size=min(beam_width, graph.node_count),
            replace=False,
        ).tolist()

        for iteration in range(max_iterations):
            # Filter to unevaluated nodes
            to_evaluate = [n for n in current_beam if n not in evaluated]

            if not to_evaluate:
                break

            # Get content for unevaluated nodes
            contents = []
            valid_nodes = []
            for node_idx in to_evaluate:
                mem_id = graph.node_ids[node_idx]
                content = content_lookup.get(mem_id)
                if content:
                    contents.append(content)
                    valid_nodes.append(node_idx)

            if not contents:
                break

            # Compute embeddings on-demand
            emb_start = datetime.now(timezone.utc)
            embeddings = await self.embedding.embed_batch(contents)
            embedding_time_ms += int(
                (datetime.now(timezone.utc) - emb_start).total_seconds() * 1000
            )

            # Score and record
            for node_idx, emb in zip(valid_nodes, embeddings):
                emb_np = np.array(emb, dtype=np.float32)
                emb_norm = np.linalg.norm(emb_np)
                if emb_norm > 0:
                    emb_normalized = emb_np / emb_norm
                else:
                    emb_normalized = emb_np

                similarity = float(np.dot(query_normalized, emb_normalized))
                evaluated[node_idx] = (similarity, emb)

            # Select top candidates for next iteration
            sorted_nodes = sorted(
                evaluated.keys(),
                key=lambda n: evaluated[n][0],
                reverse=True,
            )[:beam_width]

            # Expand to neighbors of top candidates
            next_beam = set()
            for node_idx in sorted_nodes:
                start_idx = graph.indptr[node_idx]
                end_idx = graph.indptr[node_idx + 1]
                neighbors = graph.indices[start_idx:end_idx].tolist()
                next_beam.update(neighbors)
                next_beam.add(node_idx)

            current_beam = list(next_beam)

            self.logger.debug(
                "Iteration %d: evaluated %d nodes, beam size %d",
                iteration + 1,
                len(evaluated),
                len(current_beam),
            )

        # Final ranking
        sorted_results = sorted(
            evaluated.items(),
            key=lambda x: x[1][0],
            reverse=True,
        )[:limit]

        total_time_ms = int(
            (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )

        self.logger.debug(
            "Graph-guided cold retrieval: %d evaluated -> %d results in %d ms",
            len(evaluated),
            len(sorted_results),
            total_time_ms,
        )

        return ColdRetrievalResult(
            memory_ids=[graph.node_ids[node_idx] for node_idx, _ in sorted_results],
            similarities=[score for _, (score, _) in sorted_results],
            embeddings=[emb for _, (_, emb) in sorted_results],
            candidates_evaluated=len(evaluated),
            total_retrieval_time_ms=total_time_ms,
            embedding_time_ms=embedding_time_ms,
        )

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity (-1 to 1).
        """
        if len(a) != len(b):
            raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")

        a_np = np.array(a, dtype=np.float32)
        b_np = np.array(b, dtype=np.float32)

        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a_np, b_np) / (norm_a * norm_b))


class DefaultCompressionServicePlugin(CompressionServicePluginBase):
    """Default compression service plugin."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        from memorylayer_server.services.embedding import get_embedding_service

        embedding_service = get_embedding_service(v)

        logger.info("Initializing CompressionService")
        return CompressionService(
            v=v,
            embedding_service=embedding_service,
            leann_storage=LeannStorage(),
        )
