# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

from datetime import datetime, timezone
from logging import Logger
from typing import Optional

from scitrera_app_framework import Variables, get_logger

from memorylayer_server.models import Memory

from memorylayer_server.services.storage.base import EXT_STORAGE_BACKEND
from memorylayer_server.services.embedding import EXT_EMBEDDING_SERVICE, EmbeddingService

from .base import (
    TieringServicePluginBase, ArchivalResult, RestoreResult, TieringStats
)
from ...storage import ColdTierStorageBackend

MEMORYLAYER_TIERING_MAX_IMPORTANCE = 'MEMORYLAYER_TIERING_MAX_IMPORTANCE'
MEMORYLAYER_TIERING_MAX_ACCESS_COUNT = 'MEMORYLAYER_TIERING_MAX_ACCESS_COUNT'
MEMORYLAYER_TIERING_OLDER_THAN_DAYS = 'MEMORYLAYER_TIERING_OLDER_THAN_DAYS'
MEMORYLAYER_TIERING_WARMUP_ACCESS_THRESHOLD = 'MEMORYLAYER_TIERING_WARMUP_ACCESS_THRESHOLD'

# Default thresholds for archival candidates
DEFAULT_MAX_IMPORTANCE = 0.3
DEFAULT_MAX_ACCESS_COUNT = 5
DEFAULT_OLDER_THAN_DAYS = 90
DEFAULT_WARMUP_ACCESS_THRESHOLD = 10


class TieringService:
    """
    Service for automatic memory tiering between hot and cold storage.

    This service coordinates between:
    - Storage backend (for accessing memories and cold storage operations)
    - Embedding service (for regenerating embeddings on restore)

    Tiering decisions are based on:
    - importance: Memories with low importance are archived first
    - access_count: Infrequently accessed memories are archived
    - last_accessed_at: Old memories that haven't been accessed recently
    """

    def __init__(
            self,
            storage: ColdTierStorageBackend,
            embedding_service: Optional[EmbeddingService] = None,
            v: Variables = None,
    ):
        """
        Initialize TieringService.

        Args:
            storage: Storage backend with cold tier support.
            embedding_service: Optional embedding service for regenerating
                               embeddings on memory restoration.
        """
        self.storage = storage
        self.embedding = embedding_service

        # configuration parameters -- fall back to module defaults when v is None (e.g. in tests)
        if v is not None:
            self.DEFAULT_MAX_IMPORTANCE = v.environ(MEMORYLAYER_TIERING_MAX_IMPORTANCE, default=DEFAULT_MAX_IMPORTANCE, type_fn=float)
            self.DEFAULT_MAX_ACCESS_COUNT = v.environ(MEMORYLAYER_TIERING_MAX_ACCESS_COUNT, default=DEFAULT_MAX_ACCESS_COUNT, type_fn=int)
            self.DEFAULT_OLDER_THAN_DAYS = v.environ(MEMORYLAYER_TIERING_OLDER_THAN_DAYS, default=DEFAULT_OLDER_THAN_DAYS, type_fn=int)
            self.DEFAULT_WARMUP_ACCESS_THRESHOLD = v.environ(MEMORYLAYER_TIERING_WARMUP_ACCESS_THRESHOLD,
                                                             default=DEFAULT_WARMUP_ACCESS_THRESHOLD, type_fn=int)
        else:
            self.DEFAULT_MAX_IMPORTANCE = DEFAULT_MAX_IMPORTANCE
            self.DEFAULT_MAX_ACCESS_COUNT = DEFAULT_MAX_ACCESS_COUNT
            self.DEFAULT_OLDER_THAN_DAYS = DEFAULT_OLDER_THAN_DAYS
            self.DEFAULT_WARMUP_ACCESS_THRESHOLD = DEFAULT_WARMUP_ACCESS_THRESHOLD

        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized TieringService")

    async def identify_archival_candidates(
            self,
            workspace_id: str,
            max_importance: Optional[float] = None,
            max_access_count: Optional[int] = None,
            older_than_days: Optional[int] = None,
            limit: int = 100,
    ) -> list[Memory]:
        """
        Identify memories that are candidates for archival to cold tier.

        Finds memories that meet ALL of the following criteria:
        - importance score <= max_importance threshold
        - access count <= max_access_count threshold
        - last accessed older than older_than_days days (or never accessed)
        - not already archived

        Args:
            workspace_id: Workspace identifier.
            max_importance: Maximum importance threshold (default: 0.3).
            max_access_count: Maximum access count threshold (default: 5).
            older_than_days: Minimum days since last access (default: 90).
            limit: Maximum number of candidates to return.

        Returns:
            List of Memory objects eligible for archival.
        """
        # Use defaults if not specified
        importance_threshold = max_importance if max_importance is not None else self.DEFAULT_MAX_IMPORTANCE
        access_threshold = max_access_count if max_access_count is not None else self.DEFAULT_MAX_ACCESS_COUNT
        age_threshold = older_than_days if older_than_days is not None else self.DEFAULT_OLDER_THAN_DAYS

        self.logger.info(
            "Identifying archival candidates in workspace %s: "
            "importance <= %.2f, access_count <= %d, older_than_days >= %d",
            workspace_id,
            importance_threshold,
            access_threshold,
            age_threshold,
        )

        try:
            candidates = await self.storage.get_archival_candidates(
                workspace_id=workspace_id,
                max_importance=importance_threshold,
                max_access_count=access_threshold,
                older_than_days=age_threshold,
                limit=limit,
            )

            self.logger.info(
                "Found %d archival candidates in workspace %s",
                len(candidates),
                workspace_id,
            )

            return candidates

        except Exception as e:
            self.logger.error(
                "Failed to identify archival candidates in workspace %s: %s",
                workspace_id,
                e,
            )
            raise

    async def archive_memories(
            self,
            workspace_id: str,
            memory_ids: Optional[list[str]] = None,
            auto_detect: bool = False,
            max_importance: Optional[float] = None,
            max_access_count: Optional[int] = None,
            older_than_days: Optional[int] = None,
            batch_size: int = 100,
    ) -> ArchivalResult:
        """
        Archive memories from hot tier to cold tier storage.

        Can either archive specific memories by ID or automatically detect
        and archive eligible candidates based on thresholds.

        Args:
            workspace_id: Workspace identifier.
            memory_ids: Specific memory IDs to archive (if not using auto_detect).
            auto_detect: If True, automatically detect candidates using thresholds.
            max_importance: Maximum importance threshold for auto-detection.
            max_access_count: Maximum access count threshold for auto-detection.
            older_than_days: Minimum days since last access for auto-detection.
            batch_size: Maximum number of memories to archive in one operation.

        Returns:
            ArchivalResult with counts and IDs of archived/failed memories.
        """
        start_time = datetime.now(timezone.utc)

        # Get memories to archive
        if auto_detect:
            candidates = await self.identify_archival_candidates(
                workspace_id=workspace_id,
                max_importance=max_importance,
                max_access_count=max_access_count,
                older_than_days=older_than_days,
                limit=batch_size,
            )
            memory_ids_to_archive = [m.id for m in candidates]
        elif memory_ids:
            memory_ids_to_archive = memory_ids[:batch_size]
        else:
            self.logger.warning(
                "archive_memories called without memory_ids or auto_detect=True"
            )
            return ArchivalResult(
                archived_count=0,
                failed_count=0,
                archived_memory_ids=[],
                failed_memory_ids=[],
            )

        self.logger.info(
            "Archiving %d memories in workspace %s",
            len(memory_ids_to_archive),
            workspace_id,
        )

        archived_ids: list[str] = []
        failed_ids: list[str] = []

        for memory_id in memory_ids_to_archive:
            try:
                success = await self.storage.archive_memory(
                    workspace_id=workspace_id,
                    memory_id=memory_id,
                )
                if success:
                    archived_ids.append(memory_id)
                else:
                    failed_ids.append(memory_id)
            except Exception as e:
                self.logger.error(
                    "Failed to archive memory %s: %s",
                    memory_id,
                    e,
                )
                failed_ids.append(memory_id)

        latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        self.logger.info(
            "Archived %d/%d memories in %d ms (workspace %s)",
            len(archived_ids),
            len(memory_ids_to_archive),
            latency_ms,
            workspace_id,
        )

        return ArchivalResult(
            archived_count=len(archived_ids),
            failed_count=len(failed_ids),
            archived_memory_ids=archived_ids,
            failed_memory_ids=failed_ids,
        )

    async def restore_memories(
            self,
            workspace_id: str,
            memory_ids: list[str],
            regenerate_embeddings: bool = True,
    ) -> RestoreResult:
        """
        Restore memories from cold tier back to hot tier.

        Restores specified memories and optionally regenerates their embeddings.

        Args:
            workspace_id: Workspace identifier.
            memory_ids: List of memory IDs to restore.
            regenerate_embeddings: If True and embedding service available,
                                   regenerate embeddings after restore.

        Returns:
            RestoreResult with counts and IDs of restored/failed memories.
        """
        start_time = datetime.now(timezone.utc)

        self.logger.info(
            "Restoring %d memories in workspace %s",
            len(memory_ids),
            workspace_id,
        )

        restored_ids: list[str] = []
        failed_ids: list[str] = []

        for memory_id in memory_ids:
            try:
                success = await self.storage.restore_memory(
                    workspace_id=workspace_id,
                    memory_id=memory_id,
                )
                if success:
                    restored_ids.append(memory_id)

                    # Regenerate embedding if service available and requested
                    if regenerate_embeddings and self.embedding:
                        await self._regenerate_embedding(
                            workspace_id=workspace_id,
                            memory_id=memory_id,
                        )
                else:
                    failed_ids.append(memory_id)

            except Exception as e:
                self.logger.error(
                    "Failed to restore memory %s: %s",
                    memory_id,
                    e,
                )
                failed_ids.append(memory_id)

        latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        self.logger.info(
            "Restored %d/%d memories in %d ms (workspace %s)",
            len(restored_ids),
            len(memory_ids),
            latency_ms,
            workspace_id,
        )

        return RestoreResult(
            restored_count=len(restored_ids),
            failed_count=len(failed_ids),
            restored_memory_ids=restored_ids,
            failed_memory_ids=failed_ids,
        )

    async def _regenerate_embedding(
            self,
            workspace_id: str,
            memory_id: str,
    ) -> bool:
        """
        Regenerate embedding for a restored memory.

        Args:
            workspace_id: Workspace identifier.
            memory_id: Memory identifier.

        Returns:
            True if embedding was regenerated successfully.
        """
        if not self.embedding:
            return False

        try:
            memory = await self.storage.get_memory(workspace_id, memory_id)
            if not memory:
                self.logger.warning(
                    "Memory %s not found for embedding regeneration",
                    memory_id,
                )
                return False

            # Generate new embedding
            embedding = await self.embedding.embed(memory.content)

            # Update memory with new embedding
            await self.storage.update_memory(
                workspace_id=workspace_id,
                memory_id=memory_id,
                embedding=embedding,
            )

            self.logger.debug("Regenerated embedding for memory %s", memory_id)
            return True

        except Exception as e:
            self.logger.error(
                "Failed to regenerate embedding for memory %s: %s",
                memory_id,
                e,
            )
            return False

    async def promote_hot_candidates(
            self,
            workspace_id: str,
            access_threshold: Optional[int] = None,
            limit: int = 100,
    ) -> RestoreResult:
        """
        Identify and restore frequently accessed cold tier memories.

        Memories in cold tier that are accessed frequently (above threshold)
        are automatically promoted back to hot tier. This implements the
        "warm-up" feature for cold storage.

        Args:
            workspace_id: Workspace identifier.
            access_threshold: Minimum cold access count to trigger promotion.
                              Default: DEFAULT_WARMUP_ACCESS_THRESHOLD.
            limit: Maximum number of memories to promote in one operation.

        Returns:
            RestoreResult with counts and IDs of promoted/failed memories.
        """
        threshold = access_threshold if access_threshold is not None else self.DEFAULT_WARMUP_ACCESS_THRESHOLD

        self.logger.info(
            "Checking for hot promotion candidates in workspace %s "
            "(access_threshold >= %d)",
            workspace_id,
            threshold,
        )

        # Get storage stats to check if cold tier methods are available
        try:
            # This method should be available on StorageBackend with cold tier support
            await self.storage.get_cold_storage_stats(workspace_id)
        except AttributeError:
            self.logger.warning(
                "Storage backend does not support cold tier operations"
            )
            return RestoreResult(
                restored_count=0,
                failed_count=0,
                restored_memory_ids=[],
                failed_memory_ids=[],
            )

        # Get frequently accessed cold tier documents using cold_access_count
        candidates = await self.storage.get_hot_promotion_candidates(
            workspace_id=workspace_id,
            min_access_count=threshold,
            limit=limit,
        )

        if not candidates:
            self.logger.info(
                "No hot promotion candidates found in workspace %s",
                workspace_id,
            )
            return RestoreResult(
                restored_count=0,
                failed_count=0,
                restored_memory_ids=[],
                failed_memory_ids=[],
            )

        # Limit candidates
        candidates = candidates[:limit]

        self.logger.info(
            "Found %d hot promotion candidates in workspace %s",
            len(candidates),
            workspace_id,
        )

        # Restore the frequently accessed memories
        return await self.restore_memories(
            workspace_id=workspace_id,
            memory_ids=[m.id for m in candidates],
            regenerate_embeddings=True,
        )

    async def get_tiering_stats(
            self,
            workspace_id: str,
            include_candidates: bool = True,
    ) -> TieringStats:
        """
        Get comprehensive tiering statistics for a workspace.

        Args:
            workspace_id: Workspace identifier.
            include_candidates: If True, also count archival candidates.

        Returns:
            TieringStats with hot/cold tier distribution information.
        """
        self.logger.debug("Getting tiering stats for workspace %s", workspace_id)

        try:
            # Get cold storage stats from backend
            cold_stats = await self.storage.get_cold_storage_stats(workspace_id)

            # Count archival candidates if requested
            candidates_count = 0
            if include_candidates:
                candidates = await self.identify_archival_candidates(
                    workspace_id=workspace_id,
                    limit=1000,  # Just for counting
                )
                candidates_count = len(candidates)

            return TieringStats(
                hot_memory_count=cold_stats.get("hot_memory_count", 0),
                cold_memory_count=cold_stats.get("cold_memory_count", 0),
                hot_storage_bytes=cold_stats.get("hot_storage_bytes", 0),
                cold_storage_bytes=cold_stats.get("cold_storage_bytes", 0),
                compression_ratio=cold_stats.get("compression_ratio", 0.0),
                estimated_savings_bytes=cold_stats.get("estimated_savings_bytes", 0),
                archival_candidates_count=candidates_count,
            )

        except AttributeError as e:
            self.logger.warning(
                "Storage backend does not fully support cold tier stats: %s",
                e,
            )
            return TieringStats(
                hot_memory_count=0,
                cold_memory_count=0,
                hot_storage_bytes=0,
                cold_storage_bytes=0,
                compression_ratio=0.0,
                estimated_savings_bytes=0,
                archival_candidates_count=0,
            )

    async def get_admin_tiering_stats(self) -> TieringStats:
        """Get tenant-wide (cross-workspace) tiering statistics.

        Aggregates hot/cold counts and bytes across ALL workspaces via
        ``storage.get_admin_tiering_stats``. Archival-candidate counting is
        skipped here (it is per-workspace and expensive); the per-workspace
        ``get_tiering_stats`` still surfaces candidates. Falls back to zeros if
        the storage backend doesn't implement the admin aggregate.
        """
        self.logger.debug("Getting admin (cross-workspace) tiering stats")
        try:
            stats = await self.storage.get_admin_tiering_stats()
        except AttributeError as e:
            self.logger.warning(
                "Storage backend does not support admin tiering stats: %s", e,
            )
            stats = {}

        return TieringStats(
            hot_memory_count=stats.get("hot_memory_count", 0),
            cold_memory_count=stats.get("cold_memory_count", 0),
            hot_storage_bytes=stats.get("hot_storage_bytes", 0),
            cold_storage_bytes=stats.get("cold_storage_bytes", 0),
            compression_ratio=stats.get("compression_ratio", 0.0),
            estimated_savings_bytes=stats.get("estimated_savings_bytes", 0),
            archival_candidates_count=0,
            document_storage_bytes=stats.get("document_storage_bytes", 0),
            documents_table_bytes=stats.get("documents_table_bytes", 0),
            document_pages_bytes=stats.get("document_pages_bytes", 0),
        )

    async def run_archival_cycle(
            self,
            workspace_id: str,
            max_importance: Optional[float] = None,
            max_access_count: Optional[int] = None,
            older_than_days: Optional[int] = None,
            batch_size: int = 100,
    ) -> ArchivalResult:
        """
        Run a complete archival cycle for a workspace.

        Convenience method that identifies candidates and archives them
        in a single operation. Suitable for scheduled background jobs.

        Args:
            workspace_id: Workspace identifier.
            max_importance: Maximum importance threshold.
            max_access_count: Maximum access count threshold.
            older_than_days: Minimum days since last access.
            batch_size: Maximum number of memories to archive.

        Returns:
            ArchivalResult with counts and IDs of archived/failed memories.
        """
        self.logger.info("Running archival cycle for workspace %s", workspace_id)

        return await self.archive_memories(
            workspace_id=workspace_id,
            auto_detect=True,
            max_importance=max_importance,
            max_access_count=max_access_count,
            older_than_days=older_than_days,
            batch_size=batch_size,
        )

    async def run_warmup_cycle(
            self,
            workspace_id: str,
            access_threshold: Optional[int] = None,
            limit: int = 100,
    ) -> RestoreResult:
        """
        Run a complete warm-up cycle for a workspace.

        Convenience method that identifies frequently accessed cold memories
        and promotes them to hot tier. Suitable for scheduled background jobs.

        Args:
            workspace_id: Workspace identifier.
            access_threshold: Minimum cold access count for promotion.
            limit: Maximum number of memories to promote.

        Returns:
            RestoreResult with counts and IDs of promoted/failed memories.
        """
        self.logger.info("Running warm-up cycle for workspace %s", workspace_id)

        return await self.promote_hot_candidates(
            workspace_id=workspace_id,
            access_threshold=access_threshold,
            limit=limit,
        )


    async def run_archival_sweep(
            self,
            *,
            batch_size: int = 100,
            only_enabled: bool = True,
            dry_run: bool = False,
            max_workspaces: Optional[int] = None,
    ) -> dict:
        """Run an archival pass across all workspaces.

        Shared by the periodic tiering task and the on-demand admin endpoint.

        Args:
            batch_size: Max memories to archive per workspace this pass.
            only_enabled: When True (default), only touch workspaces whose
                ``settings["tiering"].cold_tier_enabled`` is truthy. When False,
                sweep every workspace (used for ad-hoc admin runs).
            dry_run: When True, only *identify* candidates and report counts —
                nothing is archived. This is the safe way to gauge tiering
                impact before enabling it.
            max_workspaces: Optional cap on workspaces processed (safety valve).

        Returns:
            Summary dict: ``dry_run``, ``only_enabled``, ``workspaces_processed``,
            ``workspaces_skipped``, ``total_candidates``, ``total_archived``,
            ``total_failed``, and a ``per_workspace`` breakdown.
        """
        workspaces = await self.storage.list_workspaces()

        per_workspace: list[dict] = []
        processed = 0
        skipped = 0
        total_candidates = 0
        total_archived = 0
        total_failed = 0

        for ws in workspaces:
            if max_workspaces is not None and processed >= max_workspaces:
                break

            cfg = (getattr(ws, "settings", None) or {}).get("tiering") or {}
            enabled = bool(cfg.get("cold_tier_enabled", False))
            if only_enabled and not enabled:
                skipped += 1
                continue

            # Per-workspace overrides fall back to service defaults inside
            # identify/archive when None.
            max_importance = cfg.get("min_importance_threshold")
            max_access_count = cfg.get("min_access_count_threshold")
            older_than_days = cfg.get("archival_age_days")

            processed += 1
            try:
                if dry_run:
                    candidates = await self.identify_archival_candidates(
                        workspace_id=ws.id,
                        max_importance=max_importance,
                        max_access_count=max_access_count,
                        older_than_days=older_than_days,
                        limit=batch_size,
                    )
                    n = len(candidates)
                    total_candidates += n
                    per_workspace.append(
                        {"workspace_id": ws.id, "enabled": enabled, "candidates": n}
                    )
                else:
                    result = await self.run_archival_cycle(
                        workspace_id=ws.id,
                        max_importance=max_importance,
                        max_access_count=max_access_count,
                        older_than_days=older_than_days,
                        batch_size=batch_size,
                    )
                    total_archived += result.archived_count
                    total_failed += result.failed_count
                    per_workspace.append({
                        "workspace_id": ws.id,
                        "enabled": enabled,
                        "archived": result.archived_count,
                        "failed": result.failed_count,
                    })
            except Exception as e:
                # One workspace failing must not abort the sweep.
                self.logger.error(
                    "Tiering sweep failed for workspace %s: %s", ws.id, e,
                )
                per_workspace.append(
                    {"workspace_id": ws.id, "enabled": enabled, "error": str(e)}
                )

        self.logger.info(
            "Tiering sweep complete (dry_run=%s, only_enabled=%s): "
            "processed=%d skipped=%d candidates=%d archived=%d failed=%d",
            dry_run, only_enabled, processed, skipped,
            total_candidates, total_archived, total_failed,
        )

        return {
            "dry_run": dry_run,
            "only_enabled": only_enabled,
            "workspaces_processed": processed,
            "workspaces_skipped": skipped,
            "total_candidates": total_candidates,
            "total_archived": total_archived,
            "total_failed": total_failed,
            "per_workspace": per_workspace,
        }


class TieringServicePlugin(TieringServicePluginBase):
    """Plugin for tiering service."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        storage: ColdTierStorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        embedding_service = self.get_extension(EXT_EMBEDDING_SERVICE, v)

        # TODO: other configuration parameters...

        return TieringService(
            storage=storage,
            embedding_service=embedding_service,
            v=v,
        )

    def get_dependencies(self, v: Variables):
        return EXT_STORAGE_BACKEND, EXT_EMBEDDING_SERVICE
