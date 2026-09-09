# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Database connection and session management for PostgreSQL."""
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

# Environment variable names for database configuration
MEMORYLAYER_POSTGRESQL_URL = 'MEMORYLAYER_POSTGRESQL_URL'
DEFAULT_MEMORYLAYER_POSTGRESQL_URL = 'postgresql+asyncpg://localhost:5432/memorylayer'

MEMORYLAYER_DATABASE_ECHO = 'MEMORYLAYER_DATABASE_ECHO'
MEMORYLAYER_DATABASE_POOL_CLASS = 'MEMORYLAYER_DATABASE_POOL_CLASS'


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""

    pass


# Global engine and session factory
_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_database_url() -> str:
    """Get database URL from environment."""
    return os.environ.get(MEMORYLAYER_POSTGRESQL_URL, DEFAULT_MEMORYLAYER_POSTGRESQL_URL)


def _get_database_echo() -> bool:
    """Get database echo setting from environment."""
    return os.environ.get(MEMORYLAYER_DATABASE_ECHO, 'false').lower() in ('true', '1', 'yes')


def _get_database_pool_class() -> str | None:
    """Get database pool class from environment."""
    return os.environ.get(MEMORYLAYER_DATABASE_POOL_CLASS)


def create_engine(database_url: str | None = None, **kwargs) -> AsyncEngine:
    """Create async SQLAlchemy engine.

    Args:
        database_url: Database connection URL. If None, uses environment variable.
        **kwargs: Additional engine parameters.

    Returns:
        Configured async engine.
    """
    url = database_url or _get_database_url()
    echo = _get_database_echo()
    pool_class = _get_database_pool_class()

    # Default engine parameters
    engine_kwargs = {
        "echo": echo,
        "pool_pre_ping": True,
        "pool_size": 10,
        "max_overflow": 20,
    }

    # Use NullPool for serverless (Neon) to avoid connection exhaustion
    if "neon" in url or pool_class == "NullPool":
        engine_kwargs["poolclass"] = NullPool
        engine_kwargs.pop("pool_size", None)
        engine_kwargs.pop("max_overflow", None)

    # Override with provided kwargs
    engine_kwargs.update(kwargs)

    return create_async_engine(url, **engine_kwargs)


def get_engine() -> AsyncEngine:
    """Get or create the global async engine.

    Returns:
        The global async engine instance.
    """
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the global async session factory.

    Returns:
        The global async session factory.
    """
    global _async_session_factory
    if _async_session_factory is None:
        engine = get_engine()
        _async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _async_session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session.

    This is intended to be used as a FastAPI dependency:

        @app.get("/items")
        async def list_items(session: AsyncSession = Depends(get_session)):
            result = await session.execute(select(Item))
            return result.scalars().all()

    Yields:
        AsyncSession: Database session that is automatically closed after use.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database sessions with automatic commit/rollback.

    Usage:
        async with session_scope() as session:
            item = Item(name="test")
            session.add(item)
            # Automatically commits on success, rolls back on exception

    Yields:
        AsyncSession: Database session.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_engine() -> None:
    """Close the global database engine.

    Should be called on application shutdown.
    """
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
