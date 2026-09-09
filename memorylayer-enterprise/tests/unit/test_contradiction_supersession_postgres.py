"""Enterprise parity for recall-side supersession.

`newer_memory_id` — which of two conflicting memories is the CURRENT one — was computed by
the detector on every store and persisted by NEITHER backend. Without it a contradiction
records that a pair conflicts but not which side is stale, so supersession at recall has
nothing to act on. OSS sqlite gained the column; this covers the Postgres side.

Uses an in-memory SQLite database with a mirrored contradictions schema, the same pattern
as the other enterprise storage unit tests.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
import sqlalchemy as _sa
from memorylayer_server.services.contradiction.base import ContradictionRecord
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from memorylayer_saas.storage.postgresql import PostgreSQLBackend


class _Base(DeclarativeBase):
    pass


class _ContradictionSQLite(_Base):
    """Mirrors ContradictionModel without PG-specific types."""

    __tablename__ = "contradictions"

    id: Mapped[str] = mapped_column(_sa.Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    memory_a_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    memory_b_id: Mapped[str] = mapped_column(_sa.Text, nullable=False)
    contradiction_type: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    confidence: Mapped[float] = mapped_column(_sa.Float, nullable=False, server_default="0.0")
    detection_method: Mapped[str] = mapped_column(_sa.Text, nullable=False, server_default="")
    detected_at: Mapped[datetime] = mapped_column(_sa.DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(_sa.DateTime(timezone=True), nullable=True)
    resolution: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    merged_content: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)
    newer_memory_id: Mapped[str | None] = mapped_column(_sa.Text, nullable=True)


def _mid() -> str:
    return f"mem_{uuid.uuid4().hex[:12]}"


@pytest_asyncio.fixture(scope="module")
async def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with eng.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(_Base.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def backend(engine):
    import memorylayer_saas.storage.postgresql as pg_module

    with patch.object(PostgreSQLBackend, "__init__", lambda self, *a, **kw: None):
        b = PostgreSQLBackend()
    b._session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    b.logger = MagicMock()

    with patch.object(pg_module, "ContradictionModel", _ContradictionSQLite):
        yield b


async def _record(backend, ws, newer_id, older_id, *, direction: bool = True):
    return await backend.create_contradiction(
        ContradictionRecord(
            workspace_id=ws,
            memory_a_id=newer_id,
            memory_b_id=older_id,
            contradiction_type="temporal_supersession",
            confidence=0.9,
            detection_method="llm_fused",
            detected_at=datetime.now(UTC),
            newer_memory_id=newer_id if direction else None,
        )
    )


@pytest.mark.asyncio
class TestSupersessionDirectionPersistence:
    async def test_newer_memory_id_round_trips(self, backend):
        """Regression: the direction was computed on every detection and dropped on write."""
        ws, old, new = "ws_rt", _mid(), _mid()

        stored = await _record(backend, ws, new, old)
        fetched = await backend.get_contradiction(ws, stored.id)

        assert stored.newer_memory_id == new
        assert fetched is not None
        assert fetched.newer_memory_id == new

    async def test_unresolved_listing_carries_the_direction(self, backend):
        ws, old, new = "ws_list", _mid(), _mid()
        await _record(backend, ws, new, old)

        records = await backend.get_unresolved_contradictions(ws, limit=10)

        assert records and records[0].newer_memory_id == new


@pytest.mark.asyncio
class TestGetSupersededMemoryIds:
    async def test_returns_only_the_stale_side(self, backend):
        ws, old, new = "ws_stale", _mid(), _mid()
        await _record(backend, ws, new, old)

        assert await backend.get_superseded_memory_ids(ws, [old, new]) == {old}

    async def test_record_without_direction_supersedes_nothing(self, backend):
        """NULL means we know the pair conflicts, not which is current — never guessed."""
        ws, a, b = "ws_nodir", _mid(), _mid()
        await _record(backend, ws, a, b, direction=False)

        assert await backend.get_superseded_memory_ids(ws, [a, b]) == set()

    async def test_resolved_contradiction_stops_superseding(self, backend):
        ws, old, new = "ws_resolved", _mid(), _mid()
        rec = await _record(backend, ws, new, old)
        assert await backend.get_superseded_memory_ids(ws, [old, new]) == {old}

        await backend.resolve_contradiction(ws, rec.id, "keep_a")

        assert await backend.get_superseded_memory_ids(ws, [old, new]) == set()

    async def test_scoped_to_workspace(self, backend):
        old, new = _mid(), _mid()
        await _record(backend, "ws_owner", new, old)

        assert await backend.get_superseded_memory_ids("ws_other", [old, new]) == set()

    async def test_ids_outside_the_query_are_not_returned(self, backend):
        """Only ids the caller asked about come back, even though the row names both."""
        ws, old, new = "ws_subset", _mid(), _mid()
        await _record(backend, ws, new, old)

        assert await backend.get_superseded_memory_ids(ws, [new]) == set()
        assert await backend.get_superseded_memory_ids(ws, [old]) == {old}

    async def test_empty_input_is_a_no_op(self, backend):
        assert await backend.get_superseded_memory_ids("ws_empty", []) == set()
