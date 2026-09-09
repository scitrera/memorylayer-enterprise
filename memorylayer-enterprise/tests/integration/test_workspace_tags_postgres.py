# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Integration test for tag-based workspace lookup on the PostgreSQL backend.

The tag filter uses PostgreSQL array operators (``@>`` containment for match='all',
``&&`` overlap for match='any'), which need a real PostgreSQL, so this test is GATED
on ``ML_AGE_TEST_DATABASE_URL`` and SKIPPED otherwise:

    ML_AGE_TEST_DATABASE_URL=postgresql://memorylayer:memorylayer_dev@localhost:55432/memorylayer \\
        .venv/bin/python -m pytest tests/integration/test_workspace_tags_postgres.py

Isolation + cleanup: unique workspace ids per run; teardown deletes ONLY those
workspaces. NEVER a global truncate.
"""
import os
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from memorylayer_server.models.workspace import Workspace
from memorylayer_server.utils import generate_id

_DB_URL = os.getenv("ML_AGE_TEST_DATABASE_URL", "").strip()

pytestmark = pytest.mark.skipif(
    not _DB_URL,
    reason="Set ML_AGE_TEST_DATABASE_URL to a PostgreSQL database to run the workspace-tags lookup test.",
)


def _asyncpg_url(url: str) -> str:
    return (
        url.replace("postgres://", "postgresql://")
        .replace("postgresql+asyncpg://", "postgresql://")
        .replace("postgresql://", "postgresql+asyncpg://")
    )


@pytest_asyncio.fixture
async def backend():
    from scitrera_app_framework import Variables

    from memorylayer_saas.storage.models import Base, WorkspaceModel
    from memorylayer_saas.storage.postgresql import PostgreSQLBackend

    v = Variables()
    b = PostgreSQLBackend(v=v, connection_string=_asyncpg_url(_DB_URL))
    engine = create_async_engine(_asyncpg_url(_DB_URL), pool_pre_ping=True)
    b._engine = engine
    b._session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Idempotent, scoped to the workspaces table only.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, tables=[WorkspaceModel.__table__])

    created_ids: list[str] = []
    yield b, created_ids

    # Teardown: delete only the workspaces this test created.
    from sqlalchemy import delete

    async with b._session_factory() as session:
        await session.execute(delete(WorkspaceModel).where(WorkspaceModel.id.in_(created_ids)))
        await session.commit()
    await engine.dispose()


async def _mk(backend, ws_id, tags):
    b, created_ids = backend
    now = datetime.now(UTC)
    await b.create_workspace(
        Workspace(id=ws_id, tenant_id="t1", name=ws_id, tags=tags, created_at=now, updated_at=now)
    )
    created_ids.append(ws_id)


@pytest.mark.asyncio
async def test_pg_workspace_tag_lookup(backend):
    b, _ = backend
    suffix = generate_id()
    kb_fin = f"ws_{suffix}_kb_fin"
    kb_legal = f"ws_{suffix}_kb_legal"
    plain = f"ws_{suffix}_plain"

    # tags normalized on write (case/dupe/whitespace)
    await _mk(backend, kb_fin, ["  Knowledge ", "knowledge", "Topic:Finance"])
    await _mk(backend, kb_legal, ["knowledge", "topic:legal"])
    await _mk(backend, plain, ["project"])

    # round-trip normalization
    got = await b.get_workspace(kb_fin)
    assert got.tags == ["knowledge", "topic:finance"]

    def ids(workspaces):
        return {w.id for w in workspaces}

    # single tag (default match=all with one tag == containment)
    knowledge = ids(await b.list_workspaces(tags=["knowledge"]))
    assert {kb_fin, kb_legal} <= knowledge
    assert plain not in knowledge

    # match=all (case-insensitive)
    all_match = ids(await b.list_workspaces(tags=["Knowledge", "topic:finance"], match="all"))
    assert kb_fin in all_match
    assert kb_legal not in all_match

    # match=any
    any_match = ids(await b.list_workspaces(tags=["topic:finance", "project"], match="any"))
    assert {kb_fin, plain} <= any_match
    assert kb_legal not in any_match
