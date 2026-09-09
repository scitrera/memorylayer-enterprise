# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Data provider service for MemoryLayer Enterprise."""
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import select, delete, func, and_
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..models.data_provider import DataProvider
from .encryption import decrypt_json, encrypt_json


class DataProviderService:
    """Async CRUD service for data providers."""

    def __init__(self, session_factory: async_sessionmaker, logger: logging.Logger = None):
        self._session_factory = session_factory
        self.logger = logger or logging.getLogger(__name__)

    async def create_provider(self, provider: DataProvider) -> DataProvider:
        """Create a new data provider."""
        from ..storage.models import DataProviderModel

        provider_id = provider.id or f"dp_{uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)

        async with self._session_factory() as session:
            model = DataProviderModel(
                id=provider_id,
                tenant_id=provider.tenant_id,
                workspace_id=provider.workspace_id,
                name=provider.name,
                provider_type=provider.provider_type,
                description=provider.description,
                enabled=provider.enabled,
                connection_args=provider.connection_args or {},
                encrypted_args=encrypt_json(provider.encrypted_args) if provider.encrypted_args else {},
                schedule=provider.schedule,
                last_sync_at=provider.last_sync_at,
                meta=provider.metadata or {},
                created_at=now,
                updated_at=now,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._to_domain(model)

    async def get_provider(self, provider_id: str, workspace_id: str) -> Optional[DataProvider]:
        """Get data provider by ID."""
        from ..storage.models import DataProviderModel

        async with self._session_factory() as session:
            result = await session.execute(
                select(DataProviderModel).where(
                    and_(
                        DataProviderModel.id == provider_id,
                        DataProviderModel.workspace_id == workspace_id,
                    )
                )
            )
            model = result.scalar_one_or_none()
            return self._to_domain(model) if model else None

    async def get_provider_by_name(self, name: str, workspace_id: str) -> Optional[DataProvider]:
        """Get data provider by name within a workspace."""
        from ..storage.models import DataProviderModel

        async with self._session_factory() as session:
            result = await session.execute(
                select(DataProviderModel).where(
                    and_(
                        DataProviderModel.name == name,
                        DataProviderModel.workspace_id == workspace_id,
                    )
                )
            )
            model = result.scalar_one_or_none()
            return self._to_domain(model) if model else None

    async def list_providers(
        self,
        workspace_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[DataProvider], int]:
        """List data providers for a workspace."""
        from ..storage.models import DataProviderModel

        async with self._session_factory() as session:
            conditions = [DataProviderModel.workspace_id == workspace_id]

            count_result = await session.execute(
                select(func.count(DataProviderModel.id)).where(and_(*conditions))
            )
            total = count_result.scalar_one()

            result = await session.execute(
                select(DataProviderModel)
                .where(and_(*conditions))
                .order_by(DataProviderModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            providers = [self._to_domain(m) for m in result.scalars().all()]
            return providers, total

    async def update_provider(
        self, provider_id: str, workspace_id: str, **updates
    ) -> Optional[DataProvider]:
        """Update data provider fields."""
        from ..storage.models import DataProviderModel

        if not updates:
            return await self.get_provider(provider_id, workspace_id)

        async with self._session_factory() as session:
            result = await session.execute(
                select(DataProviderModel).where(
                    and_(
                        DataProviderModel.id == provider_id,
                        DataProviderModel.workspace_id == workspace_id,
                    )
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None

            for key, value in updates.items():
                if key == "metadata":
                    setattr(model, "meta", value)
                elif key == "encrypted_args":
                    model.encrypted_args = encrypt_json(value) if value else {}
                elif hasattr(model, key):
                    setattr(model, key, value)
            model.updated_at = datetime.now(timezone.utc)

            await session.commit()
            await session.refresh(model)
            return self._to_domain(model)

    async def get_provider_credentials(
        self, provider_id: str, workspace_id: str
    ) -> Optional[dict[str, Any]]:
        """Get decrypted credentials for internal use (e.g., sync operations).

        Returns the decrypted ``encrypted_args`` dict, or ``None`` if the
        provider does not exist or has no credentials stored.
        """
        from ..storage.models import DataProviderModel

        async with self._session_factory() as session:
            result = await session.execute(
                select(DataProviderModel.encrypted_args).where(
                    and_(
                        DataProviderModel.id == provider_id,
                        DataProviderModel.workspace_id == workspace_id,
                    )
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            if not row or row == {}:
                return {}
            # decrypt_json handles both encrypted wrapper and legacy plaintext dicts
            return decrypt_json(row)

    async def delete_provider(self, provider_id: str, workspace_id: str) -> bool:
        """Delete a data provider."""
        from ..storage.models import DataProviderModel

        async with self._session_factory() as session:
            result = await session.execute(
                delete(DataProviderModel).where(
                    and_(
                        DataProviderModel.id == provider_id,
                        DataProviderModel.workspace_id == workspace_id,
                    )
                )
            )
            await session.commit()
            return result.rowcount > 0

    @staticmethod
    def _to_domain(model) -> DataProvider:
        return DataProvider(
            id=model.id,
            tenant_id=model.tenant_id,
            workspace_id=model.workspace_id,
            name=model.name,
            provider_type=model.provider_type,
            description=model.description,
            enabled=model.enabled,
            connection_args=model.connection_args or {},
            encrypted_args=None,  # Never expose encrypted args in domain model
            schedule=model.schedule,
            last_sync_at=model.last_sync_at,
            metadata=model.meta or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
