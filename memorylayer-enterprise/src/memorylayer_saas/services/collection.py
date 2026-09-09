# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Collection service for MemoryLayer Enterprise vector collections."""
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import select, delete, func, and_
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..models.collection import CollectionItem


class CollectionService:
    """Async CRUD service for vector collection items with similarity search."""

    def __init__(self, session_factory: async_sessionmaker, logger: logging.Logger = None):
        self._session_factory = session_factory
        self.logger = logger or logging.getLogger(__name__)

    async def create_item(self, item: CollectionItem) -> CollectionItem:
        """Create a new collection item."""
        from ..storage.models import CollectionItemModel

        item_id = item.id or f"col_{uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)

        async with self._session_factory() as session:
            model = CollectionItemModel(
                id=item_id,
                tenant_id=item.tenant_id,
                workspace_id=item.workspace_id,
                collection_name=item.collection_name,
                name=item.name,
                content=item.content,
                item_type=item.item_type,
                tags=item.tags or [],
                meta=item.metadata or {},
                enabled=item.enabled,
                embedding=item.embedding,
                created_at=now,
                updated_at=now,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._to_domain(model)

    async def get_item(self, item_id: str, workspace_id: str) -> Optional[CollectionItem]:
        """Get a collection item by ID."""
        from ..storage.models import CollectionItemModel

        async with self._session_factory() as session:
            result = await session.execute(
                select(CollectionItemModel).where(
                    and_(
                        CollectionItemModel.id == item_id,
                        CollectionItemModel.workspace_id == workspace_id,
                    )
                )
            )
            model = result.scalar_one_or_none()
            return self._to_domain(model) if model else None

    async def list_items(
        self,
        workspace_id: str,
        collection_name: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[CollectionItem], int]:
        """List collection items for a workspace."""
        from ..storage.models import CollectionItemModel

        async with self._session_factory() as session:
            conditions = [CollectionItemModel.workspace_id == workspace_id]
            if collection_name:
                conditions.append(CollectionItemModel.collection_name == collection_name)

            count_result = await session.execute(
                select(func.count(CollectionItemModel.id)).where(and_(*conditions))
            )
            total = count_result.scalar_one()

            result = await session.execute(
                select(CollectionItemModel)
                .where(and_(*conditions))
                .order_by(CollectionItemModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            items = [self._to_domain(m) for m in result.scalars().all()]
            return items, total

    async def update_item(self, item_id: str, workspace_id: str, **updates) -> Optional[CollectionItem]:
        """Update collection item fields."""
        from ..storage.models import CollectionItemModel

        if not updates:
            return await self.get_item(item_id, workspace_id)

        async with self._session_factory() as session:
            result = await session.execute(
                select(CollectionItemModel).where(
                    and_(
                        CollectionItemModel.id == item_id,
                        CollectionItemModel.workspace_id == workspace_id,
                    )
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None

            for key, value in updates.items():
                if key == "metadata":
                    setattr(model, "meta", value)
                elif hasattr(model, key):
                    setattr(model, key, value)
            model.updated_at = datetime.now(timezone.utc)

            await session.commit()
            await session.refresh(model)
            return self._to_domain(model)

    async def delete_item(self, item_id: str, workspace_id: str) -> bool:
        """Delete a collection item."""
        from ..storage.models import CollectionItemModel

        async with self._session_factory() as session:
            result = await session.execute(
                delete(CollectionItemModel).where(
                    and_(
                        CollectionItemModel.id == item_id,
                        CollectionItemModel.workspace_id == workspace_id,
                    )
                )
            )
            await session.commit()
            return result.rowcount > 0

    async def search_similar(
        self,
        workspace_id: str,
        query_embedding: list[float],
        collection_name: Optional[str] = None,
        limit: int = 10,
    ) -> list[tuple[CollectionItem, float]]:
        """Search for similar items using pgvector cosine distance."""
        from ..storage.models import CollectionItemModel

        async with self._session_factory() as session:
            # Build cosine distance expression
            distance = CollectionItemModel.embedding.cosine_distance(query_embedding)

            conditions = [
                CollectionItemModel.workspace_id == workspace_id,
                CollectionItemModel.enabled.is_(True),
                CollectionItemModel.embedding.is_not(None),
            ]
            if collection_name:
                conditions.append(CollectionItemModel.collection_name == collection_name)

            result = await session.execute(
                select(CollectionItemModel, distance.label("distance"))
                .where(and_(*conditions))
                .order_by(distance)
                .limit(limit)
            )

            items = []
            for row in result.all():
                model = row[0]
                dist = row[1]
                similarity = 1.0 - dist  # Convert distance to similarity
                items.append((self._to_domain(model), similarity))
            return items

    @staticmethod
    def _to_domain(model) -> CollectionItem:
        return CollectionItem(
            id=model.id,
            tenant_id=model.tenant_id,
            workspace_id=model.workspace_id,
            collection_name=model.collection_name,
            name=model.name,
            content=model.content,
            item_type=model.item_type,
            tags=model.tags or [],
            metadata=model.meta or {},
            enabled=model.enabled,
            embedding=None,  # Don't return embeddings in domain model by default
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
