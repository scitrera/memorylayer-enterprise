"""User service for MemoryLayer Enterprise platform user management."""
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..models.user import User


class UserService:
    """Async CRUD service for platform users."""

    def __init__(self, session_factory: async_sessionmaker, logger: logging.Logger = None):
        self._session_factory = session_factory
        self.logger = logger or logging.getLogger(__name__)

    async def create_user(self, user: User) -> User:
        """Create a new user."""
        # Delayed import to avoid circular dependency with storage models
        from ..storage.models import UserModel

        user_id = user.id or f"usr_{uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)

        async with self._session_factory() as session:
            # Upsert: create if new, update mutable fields if exists
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            stmt = pg_insert(UserModel).values(
                id=user_id,
                tenant_id=user.tenant_id,
                email=user.email,
                display_name=user.display_name,
                first_name=user.first_name,
                last_name=user.last_name,
                enabled=user.enabled,
                licensed=user.licensed,
                meta=user.metadata or {},
                created_at=now,
                updated_at=now,
            ).on_conflict_do_update(
                constraint='uq_users_tenant_email',
                set_={
                    'display_name': user.display_name,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'enabled': user.enabled,
                    'licensed': user.licensed,
                    'metadata': user.metadata or {},
                    'updated_at': now,
                },
            )
            await session.execute(stmt)
            await session.commit()
            result = await session.execute(
                select(UserModel).where(UserModel.id == user_id)
            )
            model = result.scalar_one()
            return self._to_domain(model)

    async def get_user(self, user_id: str, tenant_id: str) -> Optional[User]:
        """Get user by ID within a tenant."""
        from ..storage.models import UserModel

        async with self._session_factory() as session:
            result = await session.execute(
                select(UserModel).where(
                    and_(UserModel.id == user_id, UserModel.tenant_id == tenant_id)
                )
            )
            model = result.scalar_one_or_none()
            return self._to_domain(model) if model else None

    async def list_users(
        self,
        tenant_id: str,
        limit: int = 50,
        offset: int = 0,
        enabled: Optional[bool] = None,
        licensed: Optional[bool] = None,
        search: Optional[str] = None,
        ids: Optional[list[str]] = None,
    ) -> tuple[list[User], int]:
        """List users for a tenant with pagination + optional filters.

        ``licensed`` filters on the seat flag; ``ids`` restricts to a specific set
        of user ids (empty list => no rows); ``search`` is a case-insensitive
        substring match over email + display_name.
        """
        from ..storage.models import UserModel

        async with self._session_factory() as session:
            conditions = [UserModel.tenant_id == tenant_id]
            if enabled is not None:
                conditions.append(UserModel.enabled == enabled)
            if licensed is not None:
                conditions.append(UserModel.licensed == licensed)
            if ids is not None:
                conditions.append(UserModel.id.in_(ids))
            if search:
                like = f"%{search}%"
                conditions.append(or_(
                    UserModel.email.ilike(like),
                    UserModel.display_name.ilike(like),
                ))

            # Count
            count_result = await session.execute(
                select(func.count(UserModel.id)).where(and_(*conditions))
            )
            total = count_result.scalar_one()

            # Fetch
            result = await session.execute(
                select(UserModel)
                .where(and_(*conditions))
                .order_by(UserModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            users = [self._to_domain(m) for m in result.scalars().all()]
            return users, total

    async def update_user(self, user_id: str, tenant_id: str, **updates) -> Optional[User]:
        """Update user fields."""
        from ..storage.models import UserModel

        if not updates:
            return await self.get_user(user_id, tenant_id)

        async with self._session_factory() as session:
            result = await session.execute(
                select(UserModel).where(
                    and_(UserModel.id == user_id, UserModel.tenant_id == tenant_id)
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

    async def delete_user(self, user_id: str, tenant_id: str) -> bool:
        """Delete a user."""
        from ..storage.models import UserModel

        async with self._session_factory() as session:
            result = await session.execute(
                delete(UserModel).where(
                    and_(UserModel.id == user_id, UserModel.tenant_id == tenant_id)
                )
            )
            await session.commit()
            return result.rowcount > 0

    @staticmethod
    def _to_domain(model) -> User:
        """Convert SQLAlchemy model to Pydantic domain model."""
        return User(
            id=model.id,
            tenant_id=model.tenant_id,
            email=model.email,
            display_name=model.display_name,
            first_name=model.first_name,
            last_name=model.last_name,
            enabled=model.enabled,
            licensed=model.licensed,
            metadata=model.meta or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
