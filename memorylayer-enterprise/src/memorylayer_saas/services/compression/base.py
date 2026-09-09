"""Compression service plugin base and data structures."""
from dataclasses import dataclass

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from memorylayer_server.services.embedding import EXT_EMBEDDING_SERVICE

from ...config import MEMORYLAYER_COMPRESSION_SERVICE, DEFAULT_MEMORYLAYER_COMPRESSION_SERVICE

# Extension point for compression service
EXT_COMPRESSION_SERVICE = 'memorylayer-enterprise-compression-service'


@dataclass
class CompressionStats:
    """Statistics for a compression operation."""

    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    node_count: int
    edge_count: int
    avg_neighbors_per_node: float


@dataclass
class NeighborGraphResult:
    """Result of building a neighbor graph."""

    graph: 'CSRGraph'
    build_time_ms: int
    stats: CompressionStats


@dataclass
class OnDemandEmbeddingResult:
    """Result of on-demand embedding computation."""

    embeddings: list[list[float]]
    memory_ids: list[str]
    compute_time_ms: int


@dataclass
class ColdRetrievalResult:
    """Result of cold tier retrieval with on-demand embedding."""

    memory_ids: list[str]
    similarities: list[float]
    embeddings: list[list[float]]
    candidates_evaluated: int
    total_retrieval_time_ms: int
    embedding_time_ms: int


# noinspection PyAbstractClass
class CompressionServicePluginBase(Plugin):
    """Base plugin for compression service."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_COMPRESSION_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_COMPRESSION_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_COMPRESSION_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_COMPRESSION_SERVICE, DEFAULT_MEMORYLAYER_COMPRESSION_SERVICE)

    def get_dependencies(self, v: Variables):
        return (EXT_EMBEDDING_SERVICE,)
