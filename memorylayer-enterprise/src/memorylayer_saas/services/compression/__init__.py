# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""
Compression Service - Build and compress neighbor graphs from embeddings.

Implements LEANN (Learning-Efficient Approximate Nearest Neighbor) compression
for 90%+ storage reduction by eliminating stored embedding vectors.

Operations:
- build_neighbor_graph: Build pruned neighbor graph from embeddings
- compress_memories: Compress a batch of memories into LEANN format
- compute_on_demand_embedding: Re-embed content during cold retrieval
- estimate_storage_reduction: Calculate compression ratio
"""
from scitrera_app_framework import Variables, get_extension

from .base import (
    EXT_COMPRESSION_SERVICE,
    CompressionServicePluginBase,
    CompressionStats,
    NeighborGraphResult,
    OnDemandEmbeddingResult,
    ColdRetrievalResult,
)
from .default import CompressionService


def get_compression_service(v: Variables = None):
    """Get the compression service instance."""
    return get_extension(EXT_COMPRESSION_SERVICE, v)


__all__ = (
    'CompressionService',
    'CompressionServicePluginBase',
    'get_compression_service',
    'EXT_COMPRESSION_SERVICE',
    'CompressionStats',
    'NeighborGraphResult',
    'OnDemandEmbeddingResult',
    'ColdRetrievalResult',
)
