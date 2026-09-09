# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""DB-free checks for the chat-thread idle-policy schema (columns, indexes,
and migration 027 chain).

The PostgreSQL query behavior (list_idle_threads / list_hidden_threads /
hide_thread / grace purge) mirrors the SQLite backend, which is covered by the
OSS ``test_chat_idle_policy`` suite; a live-PG integration test would need a
database (gate on the PG test DSN). These checks guard the model + migration
without a database.

Requires the OSS server source on the path; until the dep is bumped, run with::

    PYTHONPATH=../../oss/memorylayer-core-python/src .venv/bin/python -m pytest \
        tests/unit/test_chat_idle_policy_model.py
"""

import importlib.util
import pathlib

from memorylayer_saas.storage.models import ChatThreadModel


def test_chat_thread_model_has_idle_policy_columns():
    cols = ChatThreadModel.__table__.columns
    assert "idle_action" in cols
    assert "hidden_at" in cols
    # both nullable (no backfill; existing threads stay permanent/visible)
    assert cols["idle_action"].nullable is True
    assert cols["hidden_at"].nullable is True


def test_chat_thread_model_has_scan_indexes():
    idx = {i.name for i in ChatThreadModel.__table__.indexes}
    assert "idx_chat_threads_idle" in idx
    assert "idx_chat_threads_hidden" in idx


def test_migration_027_chain():
    p = pathlib.Path(__file__).resolve().parents[2] / "migrations" / "versions" / "027_add_chat_thread_idle_policy.py"
    spec = importlib.util.spec_from_file_location("m027", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.revision == "027"
    assert m.down_revision == "026"
    assert callable(m.upgrade) and callable(m.downgrade)
