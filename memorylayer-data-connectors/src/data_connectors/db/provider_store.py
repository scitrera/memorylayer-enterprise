"""Provider store — persistence for data-provider records.

Two interchangeable implementations:

- :class:`InMemoryProviderStore` — process-local dict (dev/test; the previous
  ``_providers`` module global in ``app.py``).
- :class:`PgProviderStore` — PostgreSQL-backed via the ``providers`` table
  (migration 001), so provider definitions survive restarts.

Both expose the same async interface and operate on plain ``dict`` records whose
keys match :class:`data_connectors.messages.providers.ProviderResponse`, so the
route handlers stay backend-agnostic and can keep calling ``ProviderResponse(**record)``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class InMemoryProviderStore:
    """Process-local provider store (dev/test fallback)."""

    def __init__(self) -> None:
        self._providers: dict[str, dict] = {}

    async def create(self, record: dict) -> dict:
        self._providers[record["id"]] = record
        return record

    async def ensure(self, record: dict) -> None:
        """Idempotently insert a provider (no-op if the id already exists)."""
        self._providers.setdefault(record["id"], record)

    async def get(self, provider_id: str) -> Optional[dict]:
        return self._providers.get(provider_id)

    async def list(self, workspace_id: str, limit: int, offset: int) -> tuple[list[dict], int]:
        filtered = [p for p in self._providers.values() if p["workspace_id"] == workspace_id]
        total = len(filtered)
        page = filtered[offset:offset + limit]
        return page, total

    async def update(self, provider_id: str, updates: dict) -> Optional[dict]:
        provider = self._providers.get(provider_id)
        if provider is None:
            return None
        provider.update(updates)
        provider["updated_at"] = datetime.now(timezone.utc)
        return provider

    async def delete(self, provider_id: str) -> bool:
        if provider_id in self._providers:
            del self._providers[provider_id]
            return True
        return False


# Provider record keys exposed to the API (ProviderResponse). The PG table also
# carries an ``encrypted_args`` column not surfaced here; it defaults to {}.
_PROVIDER_FIELDS = (
    "id",
    "workspace_id",
    "name",
    "provider_type",
    "description",
    "enabled",
    "connection_args",
    "schedule",
    "last_sync_at",
    "metadata",
    "created_at",
    "updated_at",
)


class PgProviderStore:
    """PostgreSQL-backed provider store using the ``providers`` table."""

    def __init__(self, session_scope) -> None:
        # session_scope: async-context-manager factory yielding an AsyncSession.
        self._session_scope = session_scope

    @staticmethod
    def _row_to_record(row) -> dict:
        m = row._mapping if hasattr(row, "_mapping") else row
        return {
            "id": m["id"],
            "workspace_id": m["workspace_id"],
            "name": m["name"],
            "provider_type": m["provider_type"],
            "description": m["description"],
            "enabled": m["enabled"],
            "connection_args": dict(m["connection_args"] or {}),
            "schedule": m["schedule"],
            "last_sync_at": m["last_sync_at"],
            "metadata": dict(m["metadata"] or {}),
            "created_at": m["created_at"],
            "updated_at": m["updated_at"],
        }

    async def create(self, record: dict) -> dict:
        from sqlalchemy import insert

        from data_connectors.db.tables import providers_table

        values = {k: record.get(k) for k in _PROVIDER_FIELDS}
        values["encrypted_args"] = {}
        async with self._session_scope() as session:
            await session.execute(insert(providers_table).values(**values))
        return record

    async def ensure(self, record: dict) -> None:
        """Idempotently insert a provider row (no-op if the id already exists).

        Used to seed built-in providers (e.g. ``manual_upload``) so vfs_entries
        referencing them satisfy the providers FK without clobbering an existing
        row on restart.
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from data_connectors.db.tables import providers_table

        values = {k: record.get(k) for k in _PROVIDER_FIELDS}
        values["encrypted_args"] = {}
        stmt = pg_insert(providers_table).values(**values).on_conflict_do_nothing(
            index_elements=["id"]
        )
        async with self._session_scope() as session:
            await session.execute(stmt)

    async def get(self, provider_id: str) -> Optional[dict]:
        from sqlalchemy import select

        from data_connectors.db.tables import providers_table

        async with self._session_scope() as session:
            row = (
                await session.execute(
                    select(providers_table).where(providers_table.c.id == provider_id)
                )
            ).first()
        return self._row_to_record(row) if row is not None else None

    async def list(self, workspace_id: str, limit: int, offset: int) -> tuple[list[dict], int]:
        from sqlalchemy import func, select

        from data_connectors.db.tables import providers_table

        t = providers_table
        async with self._session_scope() as session:
            total = (
                await session.execute(
                    select(func.count()).select_from(t).where(t.c.workspace_id == workspace_id)
                )
            ).scalar_one()
            rows = (
                await session.execute(
                    select(t)
                    .where(t.c.workspace_id == workspace_id)
                    .order_by(t.c.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        return [self._row_to_record(r) for r in rows], int(total)

    async def update(self, provider_id: str, updates: dict) -> Optional[dict]:
        from sqlalchemy import update as sa_update

        from data_connectors.db.tables import providers_table

        # Only persist columns that exist on the table (drop unknown keys).
        allowed = {k: v for k, v in updates.items() if k in _PROVIDER_FIELDS and k != "id"}
        allowed["updated_at"] = datetime.now(timezone.utc)
        async with self._session_scope() as session:
            row = (
                await session.execute(
                    sa_update(providers_table)
                    .where(providers_table.c.id == provider_id)
                    .values(**allowed)
                    .returning(providers_table)
                )
            ).first()
        return self._row_to_record(row) if row is not None else None

    async def delete(self, provider_id: str) -> bool:
        from sqlalchemy import delete as sa_delete

        from data_connectors.db.tables import providers_table

        async with self._session_scope() as session:
            result = await session.execute(
                sa_delete(providers_table).where(providers_table.c.id == provider_id)
            )
        return result.rowcount > 0
