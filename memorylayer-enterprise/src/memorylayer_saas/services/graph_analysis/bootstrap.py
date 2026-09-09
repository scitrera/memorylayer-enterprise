# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Idempotent Apache-AGE bootstrap for the graph-analysis backend.

Bootstrap duties:
  1. ``CREATE EXTENSION IF NOT EXISTS age`` — enable AGE (no-op if present).
  2. ``LOAD 'age'`` — load the extension into the current session.
  3. ``SET search_path = ag_catalog, "$user", public`` — makes ``cypher(...)``
     and the AGE catalog visible. In the enterprise image the relational app
     tables also live in ``ag_catalog``, so this same path serves both.
  4. ``SELECT create_graph('memorylayer')`` — guarded by a Postgres advisory
     lock + ``NOT EXISTS`` so concurrent first-callers cannot race into a
     duplicate-graph error.

Concurrency safety
------------------
``_bootstrap_lock`` (``asyncio.Lock``) serialises concurrent Python callers
in the same process — the first acquires, runs, sets ``_bootstrapped``;
subsequent callers short-circuit in Python without hitting the DB. For the
edge case of multiple processes (or a race before the flag is set), the DB-
level ``pg_advisory_xact_lock(42)`` inside a transaction ensures only one
connection runs ``create_graph`` at a time. ``NOT EXISTS`` makes the check
idempotent so the second process simply skips the create.
"""

from __future__ import annotations

import asyncio

from ._cypher import GRAPH_NAME, wrap_cypher

# search_path that exposes BOTH the AGE catalog (for cypher) and the app
# tables (which live in ag_catalog in the enterprise image).
AGE_SEARCH_PATH = 'ag_catalog, "$user", public'

# asyncio.Lock — module-level (process-singleton). Serialises concurrent
# first-bootstrap calls within one Python process so only one connection
# races into the DB-level advisory-lock path.
_bootstrap_lock: asyncio.Lock | None = None


def _get_bootstrap_lock() -> asyncio.Lock:
    """Return the module-level asyncio.Lock, creating it on the running loop.

    The Lock is created lazily so it is always bound to the current event
    loop (important for test environments that recreate the loop per test).
    """
    global _bootstrap_lock  # noqa: PLW0603
    # Re-create if the existing lock was created on a different (now-closed)
    # loop — detected by checking if any lock state references a closed loop.
    try:
        if _bootstrap_lock is not None:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                _bootstrap_lock = None
    except RuntimeError:
        _bootstrap_lock = None
    if _bootstrap_lock is None:
        _bootstrap_lock = asyncio.Lock()
    return _bootstrap_lock


async def ensure_age_session(raw_conn) -> None:
    """Load AGE and set the search_path on a single raw asyncpg connection.

    Must be called on EVERY raw connection that will issue cypher — ``LOAD``
    and ``SET search_path`` are session-scoped, so a freshly checked-out
    pooled connection needs them re-applied each time.
    """
    await raw_conn.execute("LOAD 'age';")
    await raw_conn.execute("SET search_path = %s;" % AGE_SEARCH_PATH)


async def bootstrap_age(raw_conn) -> None:
    """Idempotently provision the AGE extension and the ``memorylayer`` graph.

    Args:
        raw_conn: a raw asyncpg connection (``driver_connection``), obtained
            the same way ``PostgreSQLBackend._run_migrations`` does it.

    Idempotency:
        * ``CREATE EXTENSION IF NOT EXISTS`` — no-op when already installed.
        * ``create_graph`` is protected by a Postgres advisory lock
          (``pg_advisory_xact_lock(42)``) acquired inside a transaction, so
          concurrent processes cannot both pass the ``NOT EXISTS`` guard and
          race into a "graph already exists" error.
    """
    # 1. Enable + load AGE, set search_path for this session.
    await raw_conn.execute("CREATE EXTENSION IF NOT EXISTS age;")
    await ensure_age_session(raw_conn)

    # 2. Create the graph under an advisory lock so concurrent processes
    #    don't both pass the NOT EXISTS check and collide on create_graph.
    await raw_conn.execute(
        """
        DO $$
        BEGIN
            PERFORM pg_advisory_xact_lock(42);
            IF NOT EXISTS (
                SELECT 1 FROM ag_catalog.ag_graph WHERE name = %(name)r
            ) THEN
                PERFORM create_graph(%(name)r);
            END IF;
        END;
        $$ LANGUAGE plpgsql;
        """ % {"name": GRAPH_NAME}
    )


def make_age_engine_kwargs() -> dict:
    """Return ``create_async_engine`` kwargs that set the AGE ``search_path``.

    Call this when building the engine used by ``AgeGraphAnalysisService``
    so that EVERY session-factory connection (both ORM reads and raw AGE
    connections) resolves tables in ``ag_catalog`` without needing a manual
    ``SET search_path`` call on each connection.

    Usage::

        engine = create_async_engine(url, **make_age_engine_kwargs())

    The ``server_settings`` key is asyncpg-specific and is passed through
    SQLAlchemy's ``connect_args``.
    """
    return {
        "connect_args": {
            "server_settings": {"search_path": AGE_SEARCH_PATH},
        }
    }
