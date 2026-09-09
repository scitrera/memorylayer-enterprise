"""Alembic environment configuration with async support for PostgreSQL."""
import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import all models to ensure they're registered with Base.metadata
from memorylayer_saas.storage.database import Base
from memorylayer_saas.storage.models import (  # noqa: F401
    ContextModel,
    DocumentModel,
    IngestionJobModel,
    LeannDocumentModel,
    LeannGraphModel,
    MemoryAccessLogModel,
    MemoryAssociationModel,
    MemoryFragmentModel,
    MemoryModel,
    SessionContextModel,
    SessionModel,
    WorkspaceModel,
)

# This is the Alembic Config object
config = context.config

# Interpret the config file for Python logging.
# disable_existing_loggers=False is critical: this env.py also runs in-process
# (the server checks/applies migrations at startup), and fileConfig defaults to
# disabling every already-configured logger. That silently kills the app's own
# loggers for the life of the process — INFO milestones and even ERROR
# tracebacks vanish. Keeping existing loggers alive lets app logging survive the
# migration step while still applying alembic/sqlalchemy logger config.
#
# configure_logger: the in-process startup runner
# (PostgreSQLBackend._apply_alembic_migrations) sets this attribute to False so
# fileConfig is skipped entirely. Even with disable_existing_loggers=False,
# fileConfig still applies alembic.ini's [logger_root] level=WARN and installs
# alembic's console handler on the root logger — that clobbers the app's SAF
# INFO stderr handler and silences every app INFO log for the life of the
# server process. Standalone ``alembic`` CLI runs leave the attribute unset
# (default True) and keep their normal logging.
if config.attributes.get("configure_logger", True) and config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Set target metadata for 'autogenerate' support
target_metadata = Base.metadata

# Override sqlalchemy.url from environment variable if present
database_url = os.getenv("DATABASE_URL")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Execute migrations with the provided connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations in async mode."""
    # Get the Alembic config
    config_section = config.get_section(config.config_ini_section)
    if config_section is None:
        raise RuntimeError("Config section not found")

    # Override with environment variable if present
    if database_url:
        config_section["sqlalchemy.url"] = database_url

    # Create async engine
    connectable = async_engine_from_config(
        config_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    # Run migrations
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    # Dispose engine
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using async engine."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
