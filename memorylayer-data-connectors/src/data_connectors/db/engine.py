# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Async PostgreSQL engine / session management for data-connectors.

When ``DC_POSTGRESQL_URL`` is set the service persists the VFS catalog and
provider store to PostgreSQL (reusing the tenant's ml-postgres cluster via a
dedicated ``dataconnectors`` database). When it is unset the service falls back
to the in-memory stores (dev/test).

Mirrors the async-engine conventions in
``memorylayer-enterprise/.../storage/database.py`` (global engine + session
factory, ``session_scope`` context manager, NullPool for serverless URLs).
"""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Optional

from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

# Environment variable names for database configuration.
DC_POSTGRESQL_URL = "DC_POSTGRESQL_URL"
DC_DATABASE_ECHO = "DC_DATABASE_ECHO"
DC_DATABASE_POOL_CLASS = "DC_DATABASE_POOL_CLASS"

# Global engine + session factory (initialized lazily / at startup).
_engine: Optional[AsyncEngine] = None
_async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def _normalize_async_url(url: str) -> str:
    """Force the asyncpg driver. The chart/CNPG-provided URL is ``postgresql://…``
    (the bare scheme = the psycopg2 *sync* dialect), but BOTH our engines — the app's
    ``create_async_engine`` and the alembic ``env.py`` ``async_engine_from_config`` —
    are async and require asyncpg, else SQLAlchemy tries to import psycopg2 (not
    installed) and startup fails. Rewrite the bare scheme to ``postgresql+asyncpg://``;
    leave an already-qualified ``+driver`` URL untouched.
    """
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix):]
    return url


def get_database_url() -> Optional[str]:
    """Return the configured PostgreSQL URL (normalized to asyncpg), or None when unset.

    A None return is the signal used at startup to select the in-memory
    backend instead of the PostgreSQL backend.
    """
    raw = os.environ.get(DC_POSTGRESQL_URL) or None
    return _normalize_async_url(raw) if raw else None


def _get_database_echo() -> bool:
    return os.environ.get(DC_DATABASE_ECHO, "false").lower() in ("true", "1", "yes")


def create_engine(database_url: Optional[str] = None, **kwargs) -> AsyncEngine:
    """Create the async SQLAlchemy engine for the data-connectors DB."""
    url = database_url or get_database_url()
    if url is None:
        raise RuntimeError(
            f"{DC_POSTGRESQL_URL} is not set; cannot create a PostgreSQL engine"
        )

    pool_class = os.environ.get(DC_DATABASE_POOL_CLASS)
    engine_kwargs: dict = {
        "echo": _get_database_echo(),
        "pool_pre_ping": True,
        "pool_size": 10,
        "max_overflow": 20,
    }
    # Use NullPool for serverless Postgres (Neon) to avoid connection exhaustion.
    if "neon" in url or pool_class == "NullPool":
        engine_kwargs["poolclass"] = NullPool
        engine_kwargs.pop("pool_size", None)
        engine_kwargs.pop("max_overflow", None)
    engine_kwargs.update(kwargs)

    return create_async_engine(url, **engine_kwargs)


def get_engine() -> AsyncEngine:
    """Get or create the global async engine."""
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the global async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _async_session_factory


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    """Context manager yielding a session with commit/rollback semantics."""
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
    """Dispose of the global engine (call on shutdown)."""
    global _engine, _async_session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None


def run_migrations() -> None:
    """Run ``alembic upgrade head`` programmatically against ``DC_POSTGRESQL_URL``.

    Invoked at startup when a PostgreSQL URL is configured so the chart does not
    need a separate init container. The data-connectors alembic ``env.py`` reads
    ``DC_DATABASE_URL`` for the connection URL, so we mirror ``DC_POSTGRESQL_URL``
    into ``DC_DATABASE_URL`` here when the latter is unset.

    Runs synchronously (alembic drives its own asyncio loop internally), so this
    must be called from a worker thread when invoked inside an async context
    (see ``app.lifespan``).
    """
    url = get_database_url()
    if url is None:
        logger.info("%s unset; skipping alembic migrations", DC_POSTGRESQL_URL)
        return

    # Delayed import: alembic + its Config pull in mako/logging machinery.
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    # The data-connectors alembic env.py reads DC_DATABASE_URL; keep both in sync
    # so a single DC_POSTGRESQL_URL is sufficient for the deployment.
    os.environ.setdefault("DC_DATABASE_URL", url)

    # Resolve the alembic migrations dir RELATIVE TO THIS MODULE (it's the package
    # subdir data_connectors/db/migrations, right next to this file). This works both
    # in the src/ dev tree AND the pip-installed container layout (site-packages/
    # data_connectors/...), where the old `parents[3]/src/...` + repo-root alembic.ini
    # assumption breaks (no src/ layer, no alembic.ini installed). Build the Config
    # programmatically so we depend on neither. env.py reads DC_DATABASE_URL.
    migrations_dir = Path(__file__).resolve().parent / "migrations"
    cfg = Config()
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.set_main_option("script_location", str(migrations_dir))

    logger.info("Running alembic upgrade head for data-connectors")
    command.upgrade(cfg, "head")
    logger.info("Alembic migrations complete")
