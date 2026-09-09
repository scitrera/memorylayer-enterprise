"""Alembic environment for data-connectors.

Standard async-engine boilerplate. The DC_DATABASE_URL env var, if set,
overrides the placeholder sqlalchemy.url from alembic.ini — required for
real deployments since the .ini default points at localhost.

Migrations under ``versions/`` are pure-SQL (``op.create_table`` and
friends, no declarative Base import), so ``target_metadata`` stays None.
This keeps ``alembic upgrade head`` runnable without importing the rest
of the data-connectors package, which avoids a chicken-and-egg with the
runtime config loader. ``alembic revision --autogenerate`` is NOT
supported until a real metadata target is wired here.
"""
from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

config = context.config

# DC_DATABASE_URL is the alembic-specific override; DC_POSTGRESQL_URL is the
# service-wide URL. Accept either so a single DC_POSTGRESQL_URL is sufficient
# whether migrations run via the programmatic startup hook or `alembic` directly.
_db_url = os.environ.get("DC_DATABASE_URL") or os.environ.get("DC_POSTGRESQL_URL")
if _db_url:
    config.set_main_option("sqlalchemy.url", _db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting (used by ``alembic upgrade --sql``)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def _run_migrations_online() -> None:
    """Run migrations against a real DB connection using an asyncpg engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(_run_migrations_online())
