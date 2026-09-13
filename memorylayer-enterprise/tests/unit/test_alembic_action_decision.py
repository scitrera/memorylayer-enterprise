# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the alembic stamp-vs-upgrade decision.

Guards the fix for the "stamp head on a pre-existing untracked DB" trap, where a
missing alembic_version row caused the startup auto-migrate to stamp head and
silently hide missing columns (e.g. chat_threads.idle_action). A pre-existing DB
with no alembic_version must REFUSE (raise) rather than stamp.

Requires the OSS server source on the path until the dep is bumped::

    PYTHONPATH=../../oss/memorylayer-core-python/src .venv/bin/python -m pytest \
        tests/unit/test_alembic_action_decision.py
"""

import pytest

from memorylayer_saas.storage.postgresql import PostgreSQLBackend

decide = PostgreSQLBackend._decide_alembic_action
order = PostgreSQLBackend._schema_bootstrap_order


def test_tracked_db_runs_migrations_before_create_all():
    """The 032/033 wedge: ``create_all`` bootstraps every ORM table, including
    ones a pending migration is about to ``create_table``. Going first made 032
    raise DuplicateTableError, rolling back the whole migration (its
    ``add_column``s included) and pinning alembic at 031 forever — a restart
    just re-created the tables and reproduced it."""
    assert order(db_was_empty=False) == ("alembic", "create_all")


def test_fresh_db_creates_schema_before_stamping():
    """Nothing to collide with, and ``stamp`` requires the schema to already be
    at head — so the ORM builds it first."""
    assert order(db_was_empty=True) == ("create_all", "alembic")


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", " 0 "])
def test_auto_migrate_can_be_disabled(monkeypatch, value):
    """Disabled when an out-of-band migrator (the PreSync Job) owns the schema,
    so N replicas don't race create_all + upgrade head."""
    monkeypatch.setenv("MEMORYLAYER_AUTO_MIGRATE", value)
    assert PostgreSQLBackend._auto_migrate_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "anything-else"])
def test_auto_migrate_enabled_otherwise(monkeypatch, value):
    monkeypatch.setenv("MEMORYLAYER_AUTO_MIGRATE", value)
    assert PostgreSQLBackend._auto_migrate_enabled() is True


def test_auto_migrate_defaults_on(monkeypatch):
    """Unset must preserve today's behaviour — a chart that doesn't know about
    the flag keeps migrating itself rather than silently refusing to start."""
    monkeypatch.delenv("MEMORYLAYER_AUTO_MIGRATE", raising=False)
    assert PostgreSQLBackend._auto_migrate_enabled() is True


def test_create_all_always_runs():
    """It is still the safety net for ORM models no migration covers; the fix
    reorders it, it does not drop it."""
    for empty in (True, False):
        assert "create_all" in order(db_was_empty=empty)
        assert "alembic" in order(db_was_empty=empty)
        assert len(order(db_was_empty=empty)) == 2


def test_version_present_upgrades():
    assert decide(version_present=True, db_was_empty=False, force_stamp=False) == "upgrade"
    # version present wins even if the empty-detection were somehow True
    assert decide(version_present=True, db_was_empty=True, force_stamp=False) == "upgrade"


def test_fresh_db_stamps():
    assert decide(version_present=False, db_was_empty=True, force_stamp=False) == "stamp"


def test_preexisting_untracked_refuses():
    # The bug: this previously stamped head and hid the drift. Now it refuses.
    assert decide(version_present=False, db_was_empty=False, force_stamp=False) == "refuse"


def test_force_stamp_overrides_refusal():
    assert decide(version_present=False, db_was_empty=False, force_stamp=True) == "stamp"


@pytest.mark.parametrize("db_was_empty", [True, False])
def test_force_stamp_always_stamps_when_untracked(db_was_empty):
    assert decide(version_present=False, db_was_empty=db_was_empty, force_stamp=True) == "stamp"


@pytest.mark.asyncio
@pytest.mark.parametrize("application_tables, expected_empty", [([], True), (["memories"], False)])
async def test_bootstrap_ignores_extension_tables_on_search_path(monkeypatch, application_tables, expected_empty):
    """AGE catalogs visible on search_path must not change the app migration decision."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock
    import sqlalchemy
    import memorylayer_saas.storage.postgresql as storage

    sync_conn = SimpleNamespace(dialect=SimpleNamespace(default_schema_name="public"))
    inspector = MagicMock()
    inspector.get_table_names.side_effect = lambda schema=None: (
        application_tables if schema == "public" else application_tables + ["ag_graph", "ag_label"]
    )
    monkeypatch.setattr(sqlalchemy, "inspect", lambda conn: inspector)

    connection = MagicMock()

    async def run_sync(fn):
        return fn(sync_conn)

    connection.run_sync = run_sync
    engine = MagicMock()
    engine.connect.return_value.__aenter__ = AsyncMock(return_value=connection)
    engine.begin.return_value.__aenter__ = AsyncMock(return_value=connection)
    monkeypatch.setattr(storage, "create_async_engine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(storage, "async_sessionmaker", MagicMock())
    monkeypatch.setattr(storage, "PostgreSQLVersionedResourceStore", MagicMock())
    monkeypatch.setattr(storage, "LeannStorage", MagicMock())
    monkeypatch.setattr(storage.Base.metadata, "create_all", MagicMock())
    monkeypatch.setenv("MEMORYLAYER_AUTO_MIGRATE", "1")

    backend = storage.PostgreSQLBackend(connection_string="postgresql+asyncpg://fixture")
    backend._apply_alembic_migrations = AsyncMock()
    backend._run_migrations = AsyncMock()
    await backend.connect()
    backend._apply_alembic_migrations.assert_awaited_once_with(db_was_empty=expected_empty)
