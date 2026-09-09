"""LEANN (Learning-Efficient Approximate Nearest Neighbor) cold tier storage.

Provides graph serialization/deserialization using CSR (Compressed Sparse Row) format
for 90%+ storage reduction by eliminating stored embeddings.
"""
import io
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from scitrera_app_framework import get_logger

from .database import session_scope
from .models import LeannDocumentModel, LeannGraphModel


@dataclass
class CSRGraph:
    """Compressed Sparse Row graph representation.

    CSR format stores a graph using three arrays:
    - indptr: Row pointer array (size n_nodes + 1)
    - indices: Column indices of non-zero entries
    - data: Edge weights/distances

    This format is memory-efficient for sparse graphs and enables
    fast row slicing for neighbor lookups.
    """

    indptr: np.ndarray  # Row pointers (n_nodes + 1,)
    indices: np.ndarray  # Column indices of neighbors
    data: np.ndarray  # Edge weights (cosine similarities)
    node_ids: list[str]  # Memory IDs corresponding to node indices

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return len(self.indptr) - 1

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return len(self.indices)

    def get_neighbors(self, node_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get neighbors and weights for a node.

        Args:
            node_idx: Index of the node.

        Returns:
            Tuple of (neighbor_indices, weights).
        """
        start = self.indptr[node_idx]
        end = self.indptr[node_idx + 1]
        return self.indices[start:end], self.data[start:end]


@dataclass
class LeannDocument:
    """Archived memory document in cold storage."""

    id: str
    graph_id: str
    workspace_id: str
    memory_id: Optional[str]
    content: str
    position: int
    cold_access_count: int
    last_cold_access_at: Optional[datetime]
    memory_type: Optional[str]
    memory_subtype: Optional[str]
    importance: float
    tags: list[str]
    metadata: dict[str, Any]
    created_at: datetime


@dataclass
class LeannGraph:
    """LEANN graph with metadata."""

    id: str
    workspace_id: str
    node_count: int
    edge_count: int
    memory_ids: list[str]
    metadata: dict[str, Any]
    created_at: datetime


@dataclass
class ColdStorageStats:
    """Statistics for cold tier storage."""

    total_graphs: int
    total_documents: int
    total_graph_bytes: int
    avg_nodes_per_graph: float
    avg_edges_per_graph: float


class LeannStorage:
    """LEANN cold tier storage for graph-based memory retrieval.

    Handles storing and retrieving pruned neighbor graphs in CSR format,
    enabling 90%+ storage reduction by eliminating embedding vectors.
    """

    # CSR format magic number and version for validation
    CSR_MAGIC = b"CSR1"
    CSR_VERSION = 1

    def __init__(self, session_factory: Optional[Any] = None):
        """Initialize LEANN storage.

        Args:
            session_factory: Optional SQLAlchemy async session factory.
                             If None, uses the global session factory.
        """
        self.logger = get_logger(name=self.__class__.__name__)
        self._session_factory = session_factory

    async def _get_session(self) -> AsyncSession:
        """Get a database session.

        Returns:
            AsyncSession from factory or global session_scope.
        """
        if self._session_factory:
            return self._session_factory()
        # Use session_scope for context management
        return session_scope()

    # -------------------------------------------------------------------------
    # Graph Serialization/Deserialization (CSR Format)
    # -------------------------------------------------------------------------

    def serialize_graph(self, graph: CSRGraph) -> bytes:
        """Serialize a CSR graph to binary format.

        Binary format:
        - 4 bytes: Magic number "CSR1"
        - 4 bytes: Version (uint32)
        - 4 bytes: Number of nodes (uint32)
        - 4 bytes: Number of edges (uint32)
        - 4 bytes: Number of node IDs (uint32)
        - (n_nodes + 1) * 8 bytes: indptr array (int64)
        - n_edges * 8 bytes: indices array (int64)
        - n_edges * 4 bytes: data array (float32)
        - Variable: JSON-encoded node_ids

        Args:
            graph: CSR graph to serialize.

        Returns:
            Binary representation of the graph.
        """
        buffer = io.BytesIO()

        # Header
        buffer.write(self.CSR_MAGIC)
        buffer.write(struct.pack("<I", self.CSR_VERSION))
        buffer.write(struct.pack("<I", graph.node_count))
        buffer.write(struct.pack("<I", graph.edge_count))
        buffer.write(struct.pack("<I", len(graph.node_ids)))

        # Arrays
        buffer.write(graph.indptr.astype(np.int64).tobytes())
        buffer.write(graph.indices.astype(np.int64).tobytes())
        buffer.write(graph.data.astype(np.float32).tobytes())

        # Node IDs (null-separated strings)
        node_ids_bytes = "\0".join(graph.node_ids).encode("utf-8")
        buffer.write(struct.pack("<I", len(node_ids_bytes)))
        buffer.write(node_ids_bytes)

        return buffer.getvalue()

    def deserialize_graph(self, data: bytes) -> CSRGraph:
        """Deserialize binary data to a CSR graph.

        Args:
            data: Binary graph data.

        Returns:
            Deserialized CSR graph.

        Raises:
            ValueError: If data is invalid or corrupted.
        """
        buffer = io.BytesIO(data)

        # Validate magic number
        magic = buffer.read(4)
        if magic != self.CSR_MAGIC:
            raise ValueError(f"Invalid CSR magic number: {magic!r}")

        # Read header
        version = struct.unpack("<I", buffer.read(4))[0]
        if version != self.CSR_VERSION:
            raise ValueError(f"Unsupported CSR version: {version}")

        node_count = struct.unpack("<I", buffer.read(4))[0]
        edge_count = struct.unpack("<I", buffer.read(4))[0]
        _ = struct.unpack("<I", buffer.read(4))[0]  # num_node_ids (reserved for future use)

        # Read arrays
        indptr_bytes = buffer.read((node_count + 1) * 8)
        indptr = np.frombuffer(indptr_bytes, dtype=np.int64)

        indices_bytes = buffer.read(edge_count * 8)
        indices = np.frombuffer(indices_bytes, dtype=np.int64)

        data_bytes = buffer.read(edge_count * 4)
        edge_data = np.frombuffer(data_bytes, dtype=np.float32)

        # Read node IDs
        node_ids_len = struct.unpack("<I", buffer.read(4))[0]
        node_ids_bytes = buffer.read(node_ids_len)
        node_ids = node_ids_bytes.decode("utf-8").split("\0") if node_ids_bytes else []

        return CSRGraph(
            indptr=indptr,
            indices=indices,
            data=edge_data,
            node_ids=node_ids,
        )

    # -------------------------------------------------------------------------
    # Graph Storage Operations
    # -------------------------------------------------------------------------

    async def store_graph(
        self,
        workspace_id: str,
        graph: CSRGraph,
        metadata: Optional[dict[str, Any]] = None,
    ) -> LeannGraph:
        """Store a LEANN graph in cold storage.

        Args:
            workspace_id: Workspace identifier.
            graph: CSR graph to store.
            metadata: Optional graph metadata.

        Returns:
            Stored graph with generated ID.
        """
        graph_id = f"leann_{uuid.uuid4().hex[:12]}"
        graph_data = self.serialize_graph(graph)

        async with self._get_session() as session:
            graph_model = LeannGraphModel(
                id=graph_id,
                workspace_id=workspace_id,
                graph_data=graph_data,
                node_count=graph.node_count,
                edge_count=graph.edge_count,
                memory_ids=graph.node_ids,
                metadata=metadata or {},
            )

            session.add(graph_model)
            await session.flush()
            await session.refresh(graph_model)

            self.logger.info(
                "Stored LEANN graph %s with %d nodes, %d edges",
                graph_id,
                graph.node_count,
                graph.edge_count,
            )

            return LeannGraph(
                id=graph_model.id,
                workspace_id=graph_model.workspace_id,
                node_count=graph_model.node_count,
                edge_count=graph_model.edge_count,
                memory_ids=graph_model.memory_ids,
                metadata=graph_model.metadata,
                created_at=graph_model.created_at,
            )

    async def get_graph(
        self,
        workspace_id: str,
        graph_id: str,
    ) -> Optional[tuple[LeannGraph, CSRGraph]]:
        """Get a LEANN graph by ID.

        Args:
            workspace_id: Workspace identifier.
            graph_id: Graph identifier.

        Returns:
            Tuple of (graph metadata, deserialized CSR graph) or None.
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(LeannGraphModel).where(
                    and_(
                        LeannGraphModel.id == graph_id,
                        LeannGraphModel.workspace_id == workspace_id,
                    )
                )
            )
            graph_model = result.scalar_one_or_none()

            if not graph_model:
                return None

            csr_graph = self.deserialize_graph(graph_model.graph_data)

            return (
                LeannGraph(
                    id=graph_model.id,
                    workspace_id=graph_model.workspace_id,
                    node_count=graph_model.node_count,
                    edge_count=graph_model.edge_count,
                    memory_ids=graph_model.memory_ids,
                    metadata=graph_model.metadata,
                    created_at=graph_model.created_at,
                ),
                csr_graph,
            )

    async def get_graphs_by_workspace(
        self,
        workspace_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LeannGraph]:
        """List LEANN graphs for a workspace.

        Args:
            workspace_id: Workspace identifier.
            limit: Maximum number of graphs to return.
            offset: Number of graphs to skip.

        Returns:
            List of graph metadata (without graph_data).
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(LeannGraphModel)
                .where(LeannGraphModel.workspace_id == workspace_id)
                .order_by(LeannGraphModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            graph_models = result.scalars().all()

            return [
                LeannGraph(
                    id=g.id,
                    workspace_id=g.workspace_id,
                    node_count=g.node_count,
                    edge_count=g.edge_count,
                    memory_ids=g.memory_ids,
                    metadata=g.metadata,
                    created_at=g.created_at,
                )
                for g in graph_models
            ]

    async def delete_graph(self, workspace_id: str, graph_id: str) -> bool:
        """Delete a LEANN graph and its associated documents.

        Args:
            workspace_id: Workspace identifier.
            graph_id: Graph identifier.

        Returns:
            True if graph was deleted, False if not found.
        """
        async with self._get_session() as session:
            # Delete documents first (cascade should handle this, but be explicit)
            await session.execute(
                delete(LeannDocumentModel).where(
                    and_(
                        LeannDocumentModel.graph_id == graph_id,
                        LeannDocumentModel.workspace_id == workspace_id,
                    )
                )
            )

            # Delete graph
            result = await session.execute(
                delete(LeannGraphModel).where(
                    and_(
                        LeannGraphModel.id == graph_id,
                        LeannGraphModel.workspace_id == workspace_id,
                    )
                )
            )

            deleted = result.rowcount > 0
            if deleted:
                self.logger.info("Deleted LEANN graph %s", graph_id)

            return deleted

    # -------------------------------------------------------------------------
    # Document Storage Operations
    # -------------------------------------------------------------------------

    async def store_document(
        self,
        workspace_id: str,
        graph_id: str,
        content: str,
        position: int,
        memory_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        memory_subtype: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> LeannDocument:
        """Store a document in cold storage associated with a graph.

        Args:
            workspace_id: Workspace identifier.
            graph_id: Parent graph identifier.
            content: Document content (original memory text).
            position: Position/index in the neighbor graph.
            memory_id: Original memory ID (if preserved).
            memory_type: Memory type (episodic, semantic, etc.).
            memory_subtype: Memory subtype.
            importance: Importance score (0-1).
            tags: Document tags.
            metadata: Additional metadata.

        Returns:
            Stored document.
        """
        doc_id = f"ldoc_{uuid.uuid4().hex[:12]}"

        async with self._get_session() as session:
            doc_model = LeannDocumentModel(
                id=doc_id,
                graph_id=graph_id,
                workspace_id=workspace_id,
                memory_id=memory_id,
                content=content,
                position=position,
                cold_access_count=0,
                memory_type=memory_type,
                memory_subtype=memory_subtype,
                importance=importance,
                tags=tags or [],
                metadata=metadata or {},
            )

            session.add(doc_model)
            await session.flush()
            await session.refresh(doc_model)

            return LeannDocument(
                id=doc_model.id,
                graph_id=doc_model.graph_id,
                workspace_id=doc_model.workspace_id,
                memory_id=doc_model.memory_id,
                content=doc_model.content,
                position=doc_model.position,
                cold_access_count=doc_model.cold_access_count,
                last_cold_access_at=doc_model.last_cold_access_at,
                memory_type=doc_model.memory_type,
                memory_subtype=doc_model.memory_subtype,
                importance=doc_model.importance,
                tags=doc_model.tags,
                metadata=doc_model.metadata,
                created_at=doc_model.created_at,
            )

    async def store_documents_batch(
        self,
        workspace_id: str,
        graph_id: str,
        documents: list[dict[str, Any]],
    ) -> list[LeannDocument]:
        """Store multiple documents in a batch.

        Args:
            workspace_id: Workspace identifier.
            graph_id: Parent graph identifier.
            documents: List of document dicts with keys:
                - content: str (required)
                - position: int (required)
                - memory_id: Optional[str]
                - memory_type: Optional[str]
                - memory_subtype: Optional[str]
                - importance: float (default 0.5)
                - tags: list[str] (default [])
                - metadata: dict (default {})

        Returns:
            List of stored documents.
        """
        async with self._get_session() as session:
            doc_models = []
            for doc in documents:
                doc_id = f"ldoc_{uuid.uuid4().hex[:12]}"
                doc_model = LeannDocumentModel(
                    id=doc_id,
                    graph_id=graph_id,
                    workspace_id=workspace_id,
                    memory_id=doc.get("memory_id"),
                    content=doc["content"],
                    position=doc["position"],
                    cold_access_count=0,
                    memory_type=doc.get("memory_type"),
                    memory_subtype=doc.get("memory_subtype"),
                    importance=doc.get("importance", 0.5),
                    tags=doc.get("tags", []),
                    metadata=doc.get("metadata", {}),
                )
                doc_models.append(doc_model)

            session.add_all(doc_models)
            await session.flush()

            # Refresh all to get created_at
            for model in doc_models:
                await session.refresh(model)

            self.logger.info(
                "Stored %d documents for graph %s",
                len(doc_models),
                graph_id,
            )

            return [
                LeannDocument(
                    id=m.id,
                    graph_id=m.graph_id,
                    workspace_id=m.workspace_id,
                    memory_id=m.memory_id,
                    content=m.content,
                    position=m.position,
                    cold_access_count=m.cold_access_count,
                    last_cold_access_at=m.last_cold_access_at,
                    memory_type=m.memory_type,
                    memory_subtype=m.memory_subtype,
                    importance=m.importance,
                    tags=m.tags,
                    metadata=m.metadata,
                    created_at=m.created_at,
                )
                for m in doc_models
            ]

    async def get_document(
        self,
        workspace_id: str,
        document_id: str,
        increment_access: bool = True,
    ) -> Optional[LeannDocument]:
        """Get a document by ID.

        Args:
            workspace_id: Workspace identifier.
            document_id: Document identifier.
            increment_access: Whether to increment access count.

        Returns:
            Document or None if not found.
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(LeannDocumentModel).where(
                    and_(
                        LeannDocumentModel.id == document_id,
                        LeannDocumentModel.workspace_id == workspace_id,
                    )
                )
            )
            doc_model = result.scalar_one_or_none()

            if not doc_model:
                return None

            if increment_access:
                doc_model.cold_access_count += 1
                doc_model.last_cold_access_at = datetime.now(timezone.utc)
                await session.flush()

            return LeannDocument(
                id=doc_model.id,
                graph_id=doc_model.graph_id,
                workspace_id=doc_model.workspace_id,
                memory_id=doc_model.memory_id,
                content=doc_model.content,
                position=doc_model.position,
                cold_access_count=doc_model.cold_access_count,
                last_cold_access_at=doc_model.last_cold_access_at,
                memory_type=doc_model.memory_type,
                memory_subtype=doc_model.memory_subtype,
                importance=doc_model.importance,
                tags=doc_model.tags,
                metadata=doc_model.metadata,
                created_at=doc_model.created_at,
            )

    async def get_documents_by_graph(
        self,
        workspace_id: str,
        graph_id: str,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[LeannDocument]:
        """Get all documents for a graph.

        Args:
            workspace_id: Workspace identifier.
            graph_id: Graph identifier.
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.

        Returns:
            List of documents ordered by position.
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(LeannDocumentModel)
                .where(
                    and_(
                        LeannDocumentModel.graph_id == graph_id,
                        LeannDocumentModel.workspace_id == workspace_id,
                    )
                )
                .order_by(LeannDocumentModel.position)
                .limit(limit)
                .offset(offset)
            )
            doc_models = result.scalars().all()

            return [
                LeannDocument(
                    id=m.id,
                    graph_id=m.graph_id,
                    workspace_id=m.workspace_id,
                    memory_id=m.memory_id,
                    content=m.content,
                    position=m.position,
                    cold_access_count=m.cold_access_count,
                    last_cold_access_at=m.last_cold_access_at,
                    memory_type=m.memory_type,
                    memory_subtype=m.memory_subtype,
                    importance=m.importance,
                    tags=m.tags,
                    metadata=m.metadata,
                    created_at=m.created_at,
                )
                for m in doc_models
            ]

    async def get_documents_by_positions(
        self,
        workspace_id: str,
        graph_id: str,
        positions: list[int],
    ) -> list[LeannDocument]:
        """Get documents at specific positions in a graph.

        Args:
            workspace_id: Workspace identifier.
            graph_id: Graph identifier.
            positions: List of positions to retrieve.

        Returns:
            List of documents at the specified positions.
        """
        if not positions:
            return []

        async with self._get_session() as session:
            result = await session.execute(
                select(LeannDocumentModel).where(
                    and_(
                        LeannDocumentModel.graph_id == graph_id,
                        LeannDocumentModel.workspace_id == workspace_id,
                        LeannDocumentModel.position.in_(positions),
                    )
                )
            )
            doc_models = result.scalars().all()

            return [
                LeannDocument(
                    id=m.id,
                    graph_id=m.graph_id,
                    workspace_id=m.workspace_id,
                    memory_id=m.memory_id,
                    content=m.content,
                    position=m.position,
                    cold_access_count=m.cold_access_count,
                    last_cold_access_at=m.last_cold_access_at,
                    memory_type=m.memory_type,
                    memory_subtype=m.memory_subtype,
                    importance=m.importance,
                    tags=m.tags,
                    metadata=m.metadata,
                    created_at=m.created_at,
                )
                for m in doc_models
            ]

    async def get_frequently_accessed_documents(
        self,
        workspace_id: str,
        min_access_count: int = 5,
        limit: int = 100,
    ) -> list[LeannDocument]:
        """Get frequently accessed cold tier documents for warm-up promotion.

        Args:
            workspace_id: Workspace identifier.
            min_access_count: Minimum cold access count threshold.
            limit: Maximum number of documents to return.

        Returns:
            List of documents sorted by access count descending.
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(LeannDocumentModel)
                .where(
                    and_(
                        LeannDocumentModel.workspace_id == workspace_id,
                        LeannDocumentModel.cold_access_count >= min_access_count,
                    )
                )
                .order_by(LeannDocumentModel.cold_access_count.desc())
                .limit(limit)
            )
            doc_models = result.scalars().all()

            return [
                LeannDocument(
                    id=m.id,
                    graph_id=m.graph_id,
                    workspace_id=m.workspace_id,
                    memory_id=m.memory_id,
                    content=m.content,
                    position=m.position,
                    cold_access_count=m.cold_access_count,
                    last_cold_access_at=m.last_cold_access_at,
                    memory_type=m.memory_type,
                    memory_subtype=m.memory_subtype,
                    importance=m.importance,
                    tags=m.tags,
                    metadata=m.metadata,
                    created_at=m.created_at,
                )
                for m in doc_models
            ]

    async def delete_document(self, workspace_id: str, document_id: str) -> bool:
        """Delete a document from cold storage.

        Args:
            workspace_id: Workspace identifier.
            document_id: Document identifier.

        Returns:
            True if document was deleted, False if not found.
        """
        async with self._get_session() as session:
            result = await session.execute(
                delete(LeannDocumentModel).where(
                    and_(
                        LeannDocumentModel.id == document_id,
                        LeannDocumentModel.workspace_id == workspace_id,
                    )
                )
            )

            return result.rowcount > 0

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    async def get_storage_stats(self, workspace_id: str) -> ColdStorageStats:
        """Get cold storage statistics for a workspace.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            Cold storage statistics.
        """
        async with self._get_session() as session:
            # Count graphs and total bytes
            graph_stats = await session.execute(
                select(
                    func.count(LeannGraphModel.id).label("count"),
                    func.coalesce(func.sum(func.length(LeannGraphModel.graph_data)), 0).label(
                        "total_bytes"
                    ),
                    func.coalesce(func.avg(LeannGraphModel.node_count), 0).label("avg_nodes"),
                    func.coalesce(func.avg(LeannGraphModel.edge_count), 0).label("avg_edges"),
                ).where(LeannGraphModel.workspace_id == workspace_id)
            )
            graph_row = graph_stats.one()

            # Count documents
            doc_count = await session.execute(
                select(func.count(LeannDocumentModel.id)).where(
                    LeannDocumentModel.workspace_id == workspace_id
                )
            )

            return ColdStorageStats(
                total_graphs=graph_row.count,
                total_documents=doc_count.scalar() or 0,
                total_graph_bytes=graph_row.total_bytes,
                avg_nodes_per_graph=float(graph_row.avg_nodes),
                avg_edges_per_graph=float(graph_row.avg_edges),
            )

    async def get_graph_containing_memory(
        self,
        workspace_id: str,
        memory_id: str,
    ) -> Optional[LeannGraph]:
        """Find the graph containing a specific memory ID.

        Args:
            workspace_id: Workspace identifier.
            memory_id: Memory identifier to search for.

        Returns:
            Graph containing the memory, or None.
        """
        async with self._get_session() as session:
            result = await session.execute(
                select(LeannGraphModel).where(
                    and_(
                        LeannGraphModel.workspace_id == workspace_id,
                        LeannGraphModel.memory_ids.contains([memory_id]),
                    )
                )
            )
            graph_model = result.scalar_one_or_none()

            if not graph_model:
                return None

            return LeannGraph(
                id=graph_model.id,
                workspace_id=graph_model.workspace_id,
                node_count=graph_model.node_count,
                edge_count=graph_model.edge_count,
                memory_ids=graph_model.memory_ids,
                metadata=graph_model.metadata,
                created_at=graph_model.created_at,
            )

    async def reset_access_counts(
        self,
        workspace_id: str,
        graph_id: Optional[str] = None,
    ) -> int:
        """Reset cold access counts for documents (after warm-up promotion).

        Args:
            workspace_id: Workspace identifier.
            graph_id: Optional graph ID to limit reset to.

        Returns:
            Number of documents reset.
        """
        async with self._get_session() as session:
            conditions = [LeannDocumentModel.workspace_id == workspace_id]
            if graph_id:
                conditions.append(LeannDocumentModel.graph_id == graph_id)

            result = await session.execute(
                update(LeannDocumentModel)
                .where(and_(*conditions))
                .values(cold_access_count=0, last_cold_access_at=None)
            )

            return result.rowcount
