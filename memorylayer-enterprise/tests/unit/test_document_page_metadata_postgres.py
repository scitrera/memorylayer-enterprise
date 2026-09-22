# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise page update persistence, including SQLAlchemy's inherited metadata.

Use the existing async SQLite shim pattern to exercise the real ORM update and
fresh-session readback without requiring pgvector. The reserved metadata name
is inherited from DeclarativeBase in both this model and the production model.
"""
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from memorylayer_saas.storage import postgresql as pg


class Base(DeclarativeBase):
    pass


class Page(Base):
    __tablename__ = "document_pages"
    id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(sa.Text)
    workspace_id: Mapped[str] = mapped_column(sa.Text)
    page_no: Mapped[int] = mapped_column(sa.Integer)
    image_storage_path: Mapped[str | None] = mapped_column(sa.Text)
    transcript: Mapped[str | None] = mapped_column(sa.Text)
    transcript_model: Mapped[str | None] = mapped_column(sa.Text)
    transcript_attempts: Mapped[dict] = mapped_column(sa.JSON, default=dict)
    visual_tokens: Mapped[dict | None] = mapped_column(sa.JSON)
    embedding: Mapped[list | None] = mapped_column(sa.JSON)
    multivector: Mapped[list | None] = mapped_column(sa.JSON)
    meta: Mapped[dict] = mapped_column("metadata", sa.JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True))


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [
    {"ocr_layout": {"version": 1, "regions": [{"bbox": [.1, .2, .8, .4]}]}, "figures": []},
    {},
])
async def test_page_metadata_update_survives_a_fresh_session(monkeypatch, metadata):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            session.add(Page(id="page", document_id="doc", workspace_id="ws", page_no=0,
                transcript="old", transcript_model="synthetic", meta={"stale": True},
                created_at=datetime.now(timezone.utc)))
            await session.commit()
        monkeypatch.setattr(pg, "DocumentPageModel", Page)
        backend = pg.PostgreSQLBackend.__new__(pg.PostgreSQLBackend)
        backend._session_factory = factory
        backend.logger = Mock()
        result = await backend.update_page("page", transcript="updated", metadata=metadata)
        async with factory() as session:
            stored = await session.get(Page, "page")
            assert stored.meta == metadata
            assert stored.transcript == "updated"
        assert result.metadata == metadata
    finally:
        await engine.dispose()
