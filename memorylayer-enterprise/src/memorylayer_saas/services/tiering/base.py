# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""
Tiering Service - Automatic memory tiering between hot and cold storage.

Manages the lifecycle of memories between hot tier (with full embeddings) and
cold tier (LEANN compressed storage) based on:
- Memory importance score
- Access frequency
- Time since last access

Operations:
- identify_archival_candidates: Find memories eligible for cold tier archival
- archive_memories: Move memories from hot to cold tier
- restore_memories_on_access: Restore frequently accessed cold memories
- get_tiering_stats: Get statistics about hot/cold tier distribution
"""

from dataclasses import dataclass

from scitrera_app_framework.api import Variables, Plugin, enabled_option_pattern

from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
from memorylayer_server.services.embedding import EXT_EMBEDDING_SERVICE

from ...config import MEMORYLAYER_TIERING_SERVICE, DEFAULT_MEMORYLAYER_TIERING_SERVICE

# Extension point for tiering service
EXT_TIERING_SERVICE = 'memorylayer-enterprise-tiering-service'


@dataclass
class TieringStats:
    """Statistics for memory tiering."""

    hot_memory_count: int
    cold_memory_count: int
    hot_storage_bytes: int
    cold_storage_bytes: int
    compression_ratio: float
    estimated_savings_bytes: int
    archival_candidates_count: int
    # Actual on-disk size (pg_total_relation_size) of the document tables. Only
    # populated by the tenant-wide admin stats; defaults keep the per-workspace
    # path valid. document_storage_bytes == documents_table_bytes + document_pages_bytes.
    document_storage_bytes: int = 0
    documents_table_bytes: int = 0
    document_pages_bytes: int = 0


@dataclass
class ArchivalResult:
    """Result of an archival operation."""

    archived_count: int
    failed_count: int
    archived_memory_ids: list[str]
    failed_memory_ids: list[str]


@dataclass
class RestoreResult:
    """Result of a restore operation."""

    restored_count: int
    failed_count: int
    restored_memory_ids: list[str]
    failed_memory_ids: list[str]


# noinspection PyAbstractClass
class TieringServicePluginBase(Plugin):
    """Base plugin for tiering service."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_TIERING_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_TIERING_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_TIERING_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_TIERING_SERVICE, DEFAULT_MEMORYLAYER_TIERING_SERVICE)

    def get_dependencies(self, v: Variables):
        return EXT_STORAGE_BACKEND, EXT_EMBEDDING_SERVICE
