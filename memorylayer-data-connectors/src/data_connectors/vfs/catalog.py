# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""VFS catalog — CRUD operations for virtual filesystem entries.

The catalog is the successor to backend-future's ``meta_files`` registry.
Each entry tracks a file/object known to the platform, its source connector,
content hash (for dedup), blob key, and optional MemoryLayer document linkage.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VfsEntry(BaseModel):
    """In-memory representation of a VFS catalog entry."""
    vfs_ref: str
    workspace_id: str
    connector_id: str
    source_path: str
    content_hash: str
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    blob_key: Optional[str] = None
    ml_doc_id: Optional[str] = None
    ml_job_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VfsCatalog:
    """VFS entry CRUD operations.

    This implementation uses an in-memory store for Phase 0b.  The PostgreSQL
    backend (via SQLAlchemy async sessions) will replace this once the PG
    schema is wired up in integration.

    The interface is stable — callers should not depend on in-memory semantics.
    """

    def __init__(self) -> None:
        self._entries: dict[str, VfsEntry] = {}

    @staticmethod
    def _generate_ref() -> str:
        """Generate a unique VFS reference."""
        return f"vfs_{uuid4().hex[:16]}"

    async def register(
        self,
        workspace_id: str,
        connector_id: str,
        source_path: str,
        content_hash: str,
        content_type: Optional[str] = None,
        size_bytes: Optional[int] = None,
        blob_key: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> VfsEntry:
        """Register a new VFS entry.

        Returns:
            The created VfsEntry with a generated vfs_ref.
        """
        vfs_ref = self._generate_ref()
        now = datetime.now(timezone.utc)
        entry = VfsEntry(
            vfs_ref=vfs_ref,
            workspace_id=workspace_id,
            connector_id=connector_id,
            source_path=source_path,
            content_hash=content_hash,
            content_type=content_type,
            size_bytes=size_bytes,
            blob_key=blob_key,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        self._entries[vfs_ref] = entry
        logger.debug("Registered VFS entry %s (workspace=%s, path=%s)", vfs_ref, workspace_id, source_path)
        return entry

    async def get(self, vfs_ref: str) -> Optional[VfsEntry]:
        """Get a VFS entry by reference."""
        return self._entries.get(vfs_ref)

    async def list_entries(
        self,
        workspace_id: str,
        connector_id: Optional[str] = None,
        connector_ids: Optional[list[str]] = None,
        exclude_connector_ids: Optional[list[str]] = None,
        source_path_prefix: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[VfsEntry], int]:
        """List VFS entries with optional filtering.

        ``source_path_prefix`` filters to entries under a source-path prefix
        (folder-style grouping). ``connector_ids`` allow-lists and
        ``exclude_connector_ids`` deny-lists sources; the deny-list is applied
        LAST so it always wins, which lets a caller say "everything except the
        agent-generated noise" without enumerating every connector that exists.

        Kept in lockstep with PgVfsCatalog below — divergence here shows up as
        dev/test passing while production silently returns the whole workspace.

        Returns:
            Tuple of (entries, total_count).
        """
        filtered = [
            e for e in self._entries.values()
            if e.workspace_id == workspace_id
            and (connector_id is None or e.connector_id == connector_id)
            and (not connector_ids or e.connector_id in connector_ids)
            and (not exclude_connector_ids
                 or e.connector_id not in exclude_connector_ids)
            and (source_path_prefix is None
                 or (e.source_path or '').startswith(source_path_prefix))
        ]
        total = len(filtered)
        # Sort by created_at descending
        filtered.sort(key=lambda e: e.created_at, reverse=True)
        page = filtered[offset:offset + limit]
        return page, total

    async def update(self, vfs_ref: str, **kwargs) -> Optional[VfsEntry]:
        """Update fields on a VFS entry.

        Returns:
            The updated entry, or None if not found.
        """
        entry = self._entries.get(vfs_ref)
        if entry is None:
            return None
        for key, value in kwargs.items():
            if value is not None and hasattr(entry, key):
                setattr(entry, key, value)
        entry.updated_at = datetime.now(timezone.utc)
        return entry

    async def link_ml_document(self, vfs_ref: str, ml_doc_id: str, ml_job_id: Optional[str] = None) -> Optional[VfsEntry]:
        """Link a VFS entry to a MemoryLayer document.

        Args:
            vfs_ref: VFS reference to link.
            ml_doc_id: MemoryLayer document ID.
            ml_job_id: Optional ingestion job ID.

        Returns:
            The updated entry, or None if not found.
        """
        return await self.update(vfs_ref, ml_doc_id=ml_doc_id, ml_job_id=ml_job_id)

    async def delete(self, vfs_ref: str) -> bool:
        """Delete a VFS entry.

        Returns:
            True if deleted, False if not found.
        """
        if vfs_ref in self._entries:
            del self._entries[vfs_ref]
            return True
        return False

    async def find_by_content_hash(self, workspace_id: str, content_hash: str) -> Optional[VfsEntry]:
        """Find a VFS entry by workspace and content hash (dedup lookup)."""
        for entry in self._entries.values():
            if entry.workspace_id == workspace_id and entry.content_hash == content_hash:
                return entry
        return None

    async def find_abandoned_uploads(self, older_than: datetime, limit: int = 500) -> list[VfsEntry]:
        """Find entries minted for an upload that never completed.

        An entry is created when an upload is minted and gains its content
        hash at finalize, so one that still has no hash long afterwards is an
        upload whose bytes never landed: the tab was closed, the request
        failed, the browser went away. Nothing will ever complete it, and it
        describes a file that does not exist.

        The ml_doc_id guard is belt-and-braces. A hashless entry should never
        have an ingested document; if one somehow does, the entry means
        something this doesn't understand, so it is left alone.

        Oldest first, so a capped sweep makes progress from the back of the
        backlog rather than revisiting the same recent rows.
        """
        found = [
            entry for entry in self._entries.values()
            if not entry.content_hash
            and entry.ml_doc_id is None
            and _as_utc(entry.created_at) < _as_utc(older_than)
        ]
        found.sort(key=lambda e: _as_utc(e.created_at))
        return found[:limit]


def _as_utc(value: datetime) -> datetime:
    """Treat a naive datetime as UTC so comparisons never raise."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


# Columns that ``update`` is allowed to set, mirroring VfsEntry's mutable fields.
_VFS_UPDATABLE_FIELDS = frozenset(
    {
        "content_hash",
        "content_type",
        "size_bytes",
        "blob_key",
        "ml_doc_id",
        "ml_job_id",
        "metadata",
        "source_path",
    }
)


class PgVfsCatalog:
    """PostgreSQL-backed VFS catalog.

    Drop-in replacement for :class:`VfsCatalog` that persists entries to the
    ``vfs_entries`` table (migration 001) so the catalog survives restarts. The
    public interface (method names, args, ``VfsEntry`` return type) matches the
    in-memory implementation exactly, so ``app.py`` / ``sync_engine.py`` callers
    are backend-agnostic.

    Selected at startup when ``DC_POSTGRESQL_URL`` is set; otherwise the
    in-memory :class:`VfsCatalog` is used (dev/test).
    """

    def __init__(self, session_scope) -> None:
        # session_scope is an async-context-manager factory yielding an
        # AsyncSession (see db.engine.session_scope). Injected to keep this
        # module free of a hard import on the engine singletons.
        self._session_scope = session_scope

    @staticmethod
    def _generate_ref() -> str:
        return f"vfs_{uuid4().hex[:16]}"

    @staticmethod
    def _row_to_entry(row) -> VfsEntry:
        """Map a ``vfs_entries`` row mapping to a :class:`VfsEntry`."""
        m = row._mapping if hasattr(row, "_mapping") else row
        return VfsEntry(
            vfs_ref=m["vfs_ref"],
            workspace_id=m["workspace_id"],
            connector_id=m["connector_id"] or "",
            source_path=m["source_path"],
            content_hash=m["content_hash"],
            content_type=m["content_type"],
            size_bytes=m["size_bytes"],
            blob_key=m["blob_key"],
            ml_doc_id=m["ml_doc_id"],
            ml_job_id=m["ml_job_id"],
            metadata=dict(m["metadata"] or {}),
            created_at=m["created_at"],
            updated_at=m["updated_at"],
        )

    async def register(
        self,
        workspace_id: str,
        connector_id: str,
        source_path: str,
        content_hash: str,
        content_type: Optional[str] = None,
        size_bytes: Optional[int] = None,
        blob_key: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> VfsEntry:
        from sqlalchemy import insert

        from data_connectors.db.tables import vfs_entries_table

        vfs_ref = self._generate_ref()
        now = datetime.now(timezone.utc)
        # connector_id maps to a nullable FK on providers.id; the sync path
        # passes a real provider id, but ad-hoc registers may pass a logical
        # connector name with no provider row. Store NULL in that case to avoid
        # FK violations while keeping the in-memory contract (empty string).
        connector_fk = connector_id or None
        values = {
            "vfs_ref": vfs_ref,
            "workspace_id": workspace_id,
            "connector_id": connector_fk,
            "source_path": source_path,
            "content_hash": content_hash,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "blob_key": blob_key,
            "ml_doc_id": None,
            "ml_job_id": None,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
        }
        async with self._session_scope() as session:
            await session.execute(insert(vfs_entries_table).values(**values))
        logger.debug("Registered VFS entry %s (workspace=%s, path=%s)", vfs_ref, workspace_id, source_path)
        return VfsEntry(
            vfs_ref=vfs_ref,
            workspace_id=workspace_id,
            connector_id=connector_id,
            source_path=source_path,
            content_hash=content_hash,
            content_type=content_type,
            size_bytes=size_bytes,
            blob_key=blob_key,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )

    async def get(self, vfs_ref: str) -> Optional[VfsEntry]:
        from sqlalchemy import select

        from data_connectors.db.tables import vfs_entries_table

        async with self._session_scope() as session:
            result = await session.execute(
                select(vfs_entries_table).where(vfs_entries_table.c.vfs_ref == vfs_ref)
            )
            row = result.first()
        return self._row_to_entry(row) if row is not None else None

    async def list_entries(
        self,
        workspace_id: str,
        connector_id: Optional[str] = None,
        connector_ids: Optional[list[str]] = None,
        exclude_connector_ids: Optional[list[str]] = None,
        source_path_prefix: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[VfsEntry], int]:
        """List VFS entries with optional filtering.

        Every filter here must behave identically to VfsCatalog's in-memory
        implementation above; a divergence passes the test suite and fails
        only in production.
        """
        from sqlalchemy import func, select

        from data_connectors.db.tables import vfs_entries_table

        t = vfs_entries_table
        where = [t.c.workspace_id == workspace_id]
        if connector_id is not None:
            where.append(t.c.connector_id == connector_id)
        if connector_ids:
            where.append(t.c.connector_id.in_(connector_ids))
        if exclude_connector_ids:
            # Applied last so the deny-list wins, matching the in-memory path.
            where.append(t.c.connector_id.notin_(exclude_connector_ids))
        if source_path_prefix is not None:
            # autoescape=True is REQUIRED: startswith() compiles to LIKE, and
            # without escaping, '%' and '_' in a path act as wildcards. '_' is
            # the dangerous one — a prefix '/Bids/acme_corp/' would also match
            # '/Bids/acmeXcorp/', i.e. leak a sibling entity's files, which is
            # the exact bug this filter exists to prevent.
            where.append(
                t.c.source_path.startswith(source_path_prefix, autoescape=True)
            )

        async with self._session_scope() as session:
            total = (
                await session.execute(
                    select(func.count()).select_from(t).where(*where)
                )
            ).scalar_one()
            rows = (
                await session.execute(
                    select(t)
                    .where(*where)
                    .order_by(t.c.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        return [self._row_to_entry(r) for r in rows], int(total)

    async def update(self, vfs_ref: str, **kwargs) -> Optional[VfsEntry]:
        from sqlalchemy import update as sa_update

        from data_connectors.db.tables import vfs_entries_table

        # Mirror the in-memory semantics: only apply non-None, known fields.
        updates = {
            k: v
            for k, v in kwargs.items()
            if v is not None and k in _VFS_UPDATABLE_FIELDS
        }
        if not updates:
            # Nothing to change: return the current entry (or None if missing).
            return await self.get(vfs_ref)
        updates["updated_at"] = datetime.now(timezone.utc)

        async with self._session_scope() as session:
            result = await session.execute(
                sa_update(vfs_entries_table)
                .where(vfs_entries_table.c.vfs_ref == vfs_ref)
                .values(**updates)
                .returning(vfs_entries_table)
            )
            row = result.first()
        return self._row_to_entry(row) if row is not None else None

    async def link_ml_document(self, vfs_ref: str, ml_doc_id: str, ml_job_id: Optional[str] = None) -> Optional[VfsEntry]:
        return await self.update(vfs_ref, ml_doc_id=ml_doc_id, ml_job_id=ml_job_id)

    async def delete(self, vfs_ref: str) -> bool:
        from sqlalchemy import delete as sa_delete

        from data_connectors.db.tables import vfs_entries_table

        async with self._session_scope() as session:
            result = await session.execute(
                sa_delete(vfs_entries_table).where(vfs_entries_table.c.vfs_ref == vfs_ref)
            )
        return result.rowcount > 0

    async def find_by_content_hash(self, workspace_id: str, content_hash: str) -> Optional[VfsEntry]:
        from sqlalchemy import select

        from data_connectors.db.tables import vfs_entries_table

        t = vfs_entries_table
        async with self._session_scope() as session:
            result = await session.execute(
                select(t)
                .where(t.c.workspace_id == workspace_id, t.c.content_hash == content_hash)
                .limit(1)
            )
            row = result.first()
        return self._row_to_entry(row) if row is not None else None

    async def find_abandoned_uploads(self, older_than: datetime, limit: int = 500) -> list[VfsEntry]:
        """See VfsCatalog.find_abandoned_uploads — must match it exactly."""
        from sqlalchemy import select

        from data_connectors.db.tables import vfs_entries_table

        t = vfs_entries_table
        async with self._session_scope() as session:
            result = await session.execute(
                select(t)
                .where(
                    t.c.content_hash == "",
                    t.c.ml_doc_id.is_(None),
                    t.c.created_at < older_than,
                )
                .order_by(t.c.created_at.asc())
                .limit(limit)
            )
            rows = result.fetchall()
        return [self._row_to_entry(row) for row in rows]
