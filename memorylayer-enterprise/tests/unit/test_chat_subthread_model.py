# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""DB-free checks for the chat-thread sub-thread schema (parent_thread column,
child-lookup index, and migration 028 chain).

PostgreSQL query/cascade behavior mirrors the SQLite backend (covered by the OSS
``test_chat_subthreads`` suite); a live-PG test would need a database. Run with
the OSS server source on the path until the dep is bumped::

    PYTHONPATH=../../oss/memorylayer-core-python/src .venv/bin/python -m pytest \
        tests/unit/test_chat_subthread_model.py
"""

import importlib.util
import pathlib

from memorylayer_saas.storage.models import ChatThreadModel


def test_chat_thread_model_has_parent_column():
    cols = ChatThreadModel.__table__.columns
    assert "parent_thread" in cols
    assert cols["parent_thread"].nullable is True


def test_chat_thread_model_has_parent_index():
    idx = {i.name for i in ChatThreadModel.__table__.indexes}
    assert "idx_chat_threads_parent" in idx


def test_migration_028_chain():
    p = pathlib.Path(__file__).resolve().parents[2] / "migrations" / "versions" / "028_add_chat_thread_parent.py"
    spec = importlib.util.spec_from_file_location("m028", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.revision == "028"
    assert m.down_revision == "027"
    assert callable(m.upgrade) and callable(m.downgrade)
