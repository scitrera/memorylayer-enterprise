# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Single source of truth for the PostgreSQL embedding vector dimension.

Every single-vector ``vector(N)`` column (memories, fragments, cue anchors,
document pages, collection items, entity names) stores output from the same
embedding provider, so ``N`` must equal the provider's output dimension.

``MEMORYLAYER_EMBEDDING_DIMENSIONS`` wins when set. Otherwise the dimension
defaults to what the configured ``MEMORYLAYER_EMBEDDING_PROVIDER`` produces by
default (enterprise's default provider, ``embed_server``, is 384-d). Earlier
releases fell back to a hard-coded 1536 regardless of provider, so a default
install created 1536-d columns and then failed every insert of a 384-d vector.

Because an existing database may have been created under that old fallback,
:func:`find_embedding_dimension_mismatches` compares the live columns with the
resolved dimension so startup and migrations can refuse to run against a
mismatched schema instead of failing on every write. Operators repairing such a
schema (e.g. with a migration that alters ``memories.embedding``) can bypass the
check with ``MEMORYLAYER_SKIP_EMBEDDING_DIMENSION_CHECK=1``.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache

from sqlalchemy import MetaData, text
from sqlalchemy.engine import Connection

MEMORYLAYER_EMBEDDING_DIMENSIONS = "MEMORYLAYER_EMBEDDING_DIMENSIONS"
MEMORYLAYER_EMBEDDING_PROVIDER = "MEMORYLAYER_EMBEDDING_PROVIDER"
MEMORYLAYER_SKIP_EMBEDDING_DIMENSION_CHECK = "MEMORYLAYER_SKIP_EMBEDDING_DIMENSION_CHECK"

logger = logging.getLogger(__name__)

# Enterprise default provider (mirrors memorylayer_saas.config, which cannot be
# imported here without pulling in the OSS service registry at ORM import time).
DEFAULT_EMBEDDING_PROVIDER = "embed_server"

# Default output dimension of each OSS embedding provider when
# MEMORYLAYER_EMBEDDING_DIMENSIONS is unset. Kept in sync with the provider
# modules by tests/unit/test_embedding_dimensions.py.
PROVIDER_DEFAULT_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "embed_server": 384,
    "hash": 384,
    "mock": 384,
    "openai": 1536,
    "google": 768,
}

# The table every deployment writes to; a mismatch here means no memory can be
# stored, so it is always fatal.
PRIMARY_EMBEDDING_COLUMN = ("memories", "embedding")


def resolve_embedding_dimensions(environ: Mapping[str, str] | None = None) -> int:
    """Return the configured single-vector embedding dimension.

    Args:
        environ: Environment mapping to read; defaults to ``os.environ``.

    Raises:
        ValueError: If ``MEMORYLAYER_EMBEDDING_DIMENSIONS`` is not a positive integer.
    """
    env = os.environ if environ is None else environ
    raw = (env.get(MEMORYLAYER_EMBEDDING_DIMENSIONS) or "").strip()
    if raw:
        try:
            dimensions = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{MEMORYLAYER_EMBEDDING_DIMENSIONS} must be a positive integer, got {raw!r}"
            ) from exc
        if dimensions <= 0:
            raise ValueError(f"{MEMORYLAYER_EMBEDDING_DIMENSIONS} must be a positive integer, got {raw!r}")
        return dimensions
    provider = (env.get(MEMORYLAYER_EMBEDDING_PROVIDER) or DEFAULT_EMBEDDING_PROVIDER).strip().lower()
    if provider not in PROVIDER_DEFAULT_EMBEDDING_DIMENSIONS:
        fallback = PROVIDER_DEFAULT_EMBEDDING_DIMENSIONS[DEFAULT_EMBEDDING_PROVIDER]
        _warn_unknown_provider(provider, fallback)
        return fallback
    return PROVIDER_DEFAULT_EMBEDDING_DIMENSIONS[provider]


@cache
def _warn_unknown_provider(provider: str, fallback: int) -> None:
    """Warn once per provider: the models, migrations, and services all resolve the width."""
    logger.warning(
        "No default embedding dimension is known for %s=%r; assuming %d. Set %s to the "
        "dimension your embedding model produces.",
        MEMORYLAYER_EMBEDDING_PROVIDER, provider, fallback, MEMORYLAYER_EMBEDDING_DIMENSIONS,
    )


def embedding_dimension_check_skipped(
    log: logging.Logger | None = None, environ: Mapping[str, str] | None = None,
) -> bool:
    """Return True (and warn) when the operator disabled the dimension guard.

    Honored by server startup and ``migrations/env.py`` so an operator can run
    Alembic (downgrade, stamp, or a remediation migration) against a schema whose
    ``memories.embedding`` width differs from the configured dimension.
    """
    env = os.environ if environ is None else environ
    value = (env.get(MEMORYLAYER_SKIP_EMBEDDING_DIMENSION_CHECK) or "").strip().lower()
    if value not in {"1", "true", "yes", "on"}:
        return False
    (log or logger).warning(
        "%s is set: skipping the embedding vector width check. Writes fail if "
        "memories.embedding does not match the configured dimension; unset it once "
        "the schema is repaired.",
        MEMORYLAYER_SKIP_EMBEDDING_DIMENSION_CHECK,
    )
    return True


@dataclass(frozen=True)
class DimensionMismatch:
    """A ``vector(N)`` column whose database width differs from the ORM."""

    table: str
    column: str
    expected: int
    actual: int

    @property
    def is_primary(self) -> bool:
        return (self.table, self.column) == PRIMARY_EMBEDDING_COLUMN


def _expected_vector_columns(metadata: MetaData) -> dict[tuple[str, str], int]:
    from pgvector.sqlalchemy import Vector

    expected: dict[tuple[str, str], int] = {}
    for table in metadata.tables.values():
        for column in table.columns:
            if isinstance(column.type, Vector) and column.type.dim is not None:
                expected[(table.name, column.name)] = column.type.dim
    return expected


def find_embedding_dimension_mismatches(connection: Connection, metadata: MetaData) -> list[DimensionMismatch]:
    """Compare existing ``vector(N)`` columns in the current schema with ``metadata``.

    Columns or tables that do not exist yet are ignored (bootstrap creates them
    with the ORM dimension). Must be called with a synchronous connection, e.g.
    via ``AsyncConnection.run_sync``.
    """
    expected = _expected_vector_columns(metadata)
    if not expected:
        return []
    has_vector_type = connection.execute(text("SELECT to_regtype('vector') IS NOT NULL")).scalar()
    if not has_vector_type:
        return []
    rows = connection.execute(text(
        "SELECT c.relname, a.attname, a.atttypmod "
        "FROM pg_attribute a "
        "JOIN pg_class c ON c.oid = a.attrelid "
        "WHERE c.relnamespace = current_schema()::regnamespace "
        "AND c.relkind = 'r' AND a.attnum > 0 AND NOT a.attisdropped "
        "AND a.atttypid = 'vector'::regtype"
    )).all()
    mismatches = []
    for table, column, typmod in rows:
        want = expected.get((table, column))
        # pgvector stores the declared dimension directly in atttypmod (-1 if unset).
        if want is not None and typmod > 0 and typmod != want:
            mismatches.append(DimensionMismatch(table, column, want, typmod))
    return sorted(mismatches, key=lambda m: (not m.is_primary, m.table, m.column))


def describe_mismatches(mismatches: list[DimensionMismatch]) -> str:
    """Human-readable, actionable description of dimension mismatches."""
    columns = ", ".join(f"{m.table}.{m.column} is vector({m.actual})" for m in mismatches)
    expected = mismatches[0].expected
    actual = sorted({m.actual for m in mismatches})
    configured = os.environ.get(MEMORYLAYER_EMBEDDING_DIMENSIONS)
    source = (
        f"{MEMORYLAYER_EMBEDDING_DIMENSIONS}={configured}" if configured
        else f"{MEMORYLAYER_EMBEDDING_DIMENSIONS} is unset, so the embedding provider's default applies"
    )
    return (
        f"Embedding dimension mismatch: the server is configured for {expected}-d vectors "
        f"({source}), but {columns}. Set {MEMORYLAYER_EMBEDDING_DIMENSIONS} to the dimension "
        f"your embedding model produces and the existing columns use (e.g. "
        f"{MEMORYLAYER_EMBEDDING_DIMENSIONS}={actual[0]} for databases created by releases "
        f"that defaulted to 1536). Changing the dimension of a populated database requires "
        f"migrating the columns and re-embedding; set "
        f"{MEMORYLAYER_SKIP_EMBEDDING_DIMENSION_CHECK}=1 while running that migration."
    )


def verify_migration_embedding_dimensions(
    connection: Connection, metadata: MetaData, log: logging.Logger | None = None,
) -> None:
    """Alembic guard: raise when ``memories.embedding`` differs from the configured width.

    Skipped (with a warning) when ``MEMORYLAYER_SKIP_EMBEDDING_DIMENSION_CHECK`` is
    set. Ends the read-only transaction it starts, because Alembic's
    ``begin_transaction()`` would otherwise join it and never commit migration DDL.

    Raises:
        RuntimeError: On a primary-column mismatch.
    """
    if embedding_dimension_check_skipped(log):
        return
    mismatches = find_embedding_dimension_mismatches(connection, metadata)
    connection.rollback()
    if any(m.is_primary for m in mismatches):
        raise RuntimeError(describe_mismatches(mismatches))
