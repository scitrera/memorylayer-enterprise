# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the shared PostgreSQL embedding-dimension resolution and guard.

Regression: storage/models.py and the migrations fell back to 1536 when
MEMORYLAYER_EMBEDDING_DIMENSIONS was unset, while the default ``embed_server``
provider produces 384-d vectors, so a default install created vector(1536)
columns and every memory insert failed with "expected 1536 dimensions, not 384".
"""
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, MetaData, Table, Text

from memorylayer_saas.storage import embedding_dimensions
from memorylayer_saas.storage.embedding_dimensions import (
    PROVIDER_DEFAULT_EMBEDDING_DIMENSIONS,
    DimensionMismatch,
    describe_mismatches,
    embedding_dimension_check_skipped,
    find_embedding_dimension_mismatches,
    resolve_embedding_dimensions,
    verify_migration_embedding_dimensions,
)
from memorylayer_saas.storage.postgresql import PostgreSQLBackend

SKIP = "MEMORYLAYER_SKIP_EMBEDDING_DIMENSION_CHECK"


class TestResolveEmbeddingDimensions:
    def test_default_provider_matches_embed_server_default(self):
        from memorylayer_server.config import DEFAULT_EMBEDDING_DIMENSIONS_EMBED_SERVER

        from memorylayer_saas.config import DEFAULT_MEMORYLAYER_EMBEDDING_PROVIDER

        assert DEFAULT_MEMORYLAYER_EMBEDDING_PROVIDER == "embed_server"
        assert resolve_embedding_dimensions({}) == DEFAULT_EMBEDDING_DIMENSIONS_EMBED_SERVER == 384

    def test_explicit_dimension_wins(self):
        env = {"MEMORYLAYER_EMBEDDING_DIMENSIONS": "1920", "MEMORYLAYER_EMBEDDING_PROVIDER": "openai"}
        assert resolve_embedding_dimensions(env) == 1920

    def test_blank_dimension_falls_back_to_provider_default(self):
        env = {"MEMORYLAYER_EMBEDDING_DIMENSIONS": " ", "MEMORYLAYER_EMBEDDING_PROVIDER": "openai"}
        assert resolve_embedding_dimensions(env) == 1536

    def test_unknown_provider_uses_enterprise_default_and_warns(self, caplog):
        embedding_dimensions._warn_unknown_provider.cache_clear()
        with caplog.at_level("WARNING", logger=embedding_dimensions.__name__):
            assert resolve_embedding_dimensions({"MEMORYLAYER_EMBEDDING_PROVIDER": "custom"}) == 384
            resolve_embedding_dimensions({"MEMORYLAYER_EMBEDDING_PROVIDER": "custom"})
        warnings = [r.getMessage() for r in caplog.records]
        assert len(warnings) == 1, "warn once per provider, not on every resolution"
        assert "'custom'" in warnings[0] and "MEMORYLAYER_EMBEDDING_DIMENSIONS" in warnings[0]

    def test_known_provider_and_explicit_dimension_do_not_warn(self, caplog):
        embedding_dimensions._warn_unknown_provider.cache_clear()
        with caplog.at_level("WARNING", logger=embedding_dimensions.__name__):
            resolve_embedding_dimensions({"MEMORYLAYER_EMBEDDING_PROVIDER": "openai"})
            resolve_embedding_dimensions(
                {"MEMORYLAYER_EMBEDDING_PROVIDER": "custom", "MEMORYLAYER_EMBEDDING_DIMENSIONS": "512"}
            )
        assert caplog.records == []

    @pytest.mark.parametrize("raw", ["abc", "0", "-5"])
    def test_invalid_dimension_is_rejected(self, raw):
        with pytest.raises(ValueError, match="MEMORYLAYER_EMBEDDING_DIMENSIONS"):
            resolve_embedding_dimensions({"MEMORYLAYER_EMBEDDING_DIMENSIONS": raw})

    def test_provider_defaults_match_oss_providers(self):
        """The table must track each OSS provider's own fallback dimension."""
        from memorylayer_server.config import DEFAULT_EMBEDDING_DIMENSIONS_EMBED_SERVER
        from memorylayer_server.services.embedding import google, mock, openai
        from memorylayer_server.services.embedding import hash as hash_provider

        assert PROVIDER_DEFAULT_EMBEDDING_DIMENSIONS == {
            "embed_server": DEFAULT_EMBEDDING_DIMENSIONS_EMBED_SERVER,
            "hash": hash_provider.DEFAULT_EMBEDDING_DIMENSIONS,
            "mock": mock.DEFAULT_EMBEDDING_DIMENSIONS,
            "openai": openai.DEFAULT_EMBEDDING_DIMENSIONS,
            "google": google.DEFAULT_EMBEDDING_DIMENSIONS,
        }

    def test_orm_and_compression_share_the_resolved_dimension(self):
        from memorylayer_saas.services.compression.default import DEFAULT_EMBEDDING_DIM
        from memorylayer_saas.storage.models import _EMBEDDING_DIM, MemoryModel

        assert _EMBEDDING_DIM == DEFAULT_EMBEDDING_DIM == resolve_embedding_dimensions()
        assert MemoryModel.__table__.c.embedding.type.dim == _EMBEDDING_DIM


def _metadata(dim: int) -> MetaData:
    md = MetaData()
    Table("memories", md, Column("id", Text, primary_key=True), Column("embedding", Vector(dim)))
    Table("entities", md, Column("id", Text, primary_key=True), Column("name_embedding", Vector(dim)))
    return md


def _connection(rows, *, has_vector=True):
    conn = MagicMock()
    has_vector_result = MagicMock()
    has_vector_result.scalar.return_value = has_vector
    rows_result = MagicMock()
    rows_result.all.return_value = rows
    conn.execute.side_effect = [has_vector_result, rows_result]
    return conn


class TestFindMismatches:
    def test_matching_columns_report_nothing(self):
        conn = _connection([("memories", "embedding", 384), ("entities", "name_embedding", 384)])
        assert find_embedding_dimension_mismatches(conn, _metadata(384)) == []

    def test_legacy_1536_schema_is_reported_primary_first(self):
        conn = _connection([("entities", "name_embedding", 1536), ("memories", "embedding", 1536)])
        mismatches = find_embedding_dimension_mismatches(conn, _metadata(384))
        assert mismatches == [
            DimensionMismatch("memories", "embedding", 384, 1536),
            DimensionMismatch("entities", "name_embedding", 384, 1536),
        ]
        assert mismatches[0].is_primary and not mismatches[1].is_primary

    def test_unknown_and_unsized_columns_are_ignored(self):
        conn = _connection([("other", "embedding", 99), ("memories", "embedding", -1)])
        assert find_embedding_dimension_mismatches(conn, _metadata(384)) == []

    def test_database_without_pgvector_is_skipped(self):
        conn = _connection([], has_vector=False)
        assert find_embedding_dimension_mismatches(conn, _metadata(384)) == []
        assert conn.execute.call_count == 1

    def test_description_is_actionable(self, monkeypatch):
        monkeypatch.delenv("MEMORYLAYER_EMBEDDING_DIMENSIONS", raising=False)
        message = describe_mismatches([DimensionMismatch("memories", "embedding", 384, 1536)])
        assert "memories.embedding is vector(1536)" in message
        assert "configured for 384-d" in message
        assert "MEMORYLAYER_EMBEDDING_DIMENSIONS=1536" in message
        assert "is unset" in message
        assert f"{SKIP}=1" in message


class TestSkipFlag:
    @pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
    def test_enabled_values_skip_and_warn(self, value):
        log = MagicMock()
        assert embedding_dimension_check_skipped(log, {SKIP: value}) is True
        log.warning.assert_called_once()
        assert SKIP in log.warning.call_args.args[1]

    @pytest.mark.parametrize("env", [{}, {SKIP: ""}, {SKIP: "0"}, {SKIP: "false"}])
    def test_otherwise_not_skipped_and_silent(self, env):
        log = MagicMock()
        assert embedding_dimension_check_skipped(log, env) is False
        log.warning.assert_not_called()


class TestMigrationGuard:
    """The env.py guard, which runs for every online alembic command."""

    def test_primary_mismatch_blocks_migration(self, monkeypatch):
        monkeypatch.delenv(SKIP, raising=False)
        conn = _connection([("memories", "embedding", 1536)])
        with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
            verify_migration_embedding_dimensions(conn, _metadata(384))
        conn.rollback.assert_called_once()

    def test_matching_schema_migrates_and_ends_check_transaction(self, monkeypatch):
        monkeypatch.delenv(SKIP, raising=False)
        conn = _connection([("memories", "embedding", 384)])
        verify_migration_embedding_dimensions(conn, _metadata(384))
        conn.rollback.assert_called_once()

    def test_skip_flag_allows_repair_without_querying(self, monkeypatch):
        """Downgrade/stamp/a column-altering remediation must be runnable."""
        monkeypatch.setenv(SKIP, "1")
        conn = MagicMock()
        log = MagicMock()
        verify_migration_embedding_dimensions(conn, _metadata(384), log)
        conn.execute.assert_not_called()
        log.warning.assert_called_once()


class _FakeEngine:
    def __init__(self, mismatches):
        self._mismatches = mismatches

    @asynccontextmanager
    async def connect(self):
        conn = MagicMock()

        async def run_sync(fn, *args):
            return self._mismatches

        conn.run_sync = run_sync
        yield conn


def _backend(mismatches):
    backend = PostgreSQLBackend.__new__(PostgreSQLBackend)
    backend._engine = _FakeEngine(mismatches)
    backend.logger = MagicMock()
    return backend


class TestStartupGuard:
    @pytest.fixture(autouse=True)
    def _no_skip(self, monkeypatch):
        monkeypatch.delenv(SKIP, raising=False)

    @pytest.mark.asyncio
    async def test_skip_flag_bypasses_primary_mismatch_with_warning(self, monkeypatch):
        monkeypatch.setenv(SKIP, "1")
        backend = _backend([DimensionMismatch("memories", "embedding", 384, 1536)])
        await backend._verify_embedding_dimensions(log_secondary=True)
        backend.logger.warning.assert_called_once()
        backend.logger.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_primary_mismatch_refuses_startup(self):
        backend = _backend([DimensionMismatch("memories", "embedding", 384, 1536)])
        with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
            await backend._verify_embedding_dimensions(log_secondary=True)

    @pytest.mark.asyncio
    async def test_secondary_mismatch_is_logged_not_fatal(self):
        backend = _backend([DimensionMismatch("entities", "name_embedding", 384, 1536)])
        await backend._verify_embedding_dimensions(log_secondary=True)
        backend.logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_pre_migration_check_does_not_duplicate_secondary_logs(self):
        backend = _backend([DimensionMismatch("entities", "name_embedding", 384, 1536)])
        await backend._verify_embedding_dimensions(log_secondary=False)
        backend.logger.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_mismatch_is_silent(self):
        backend = _backend([])
        await backend._verify_embedding_dimensions(log_secondary=True)
        backend.logger.error.assert_not_called()
