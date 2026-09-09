# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""PostgreSQL adapter conformance for OSS typed versioned resources."""

from datetime import datetime

import pytest
from memorylayer_server.models.versioned_resource import (
    VersionedResourceConflictError,
    VersionedResourcePreconditionFailedError,
)
from memorylayer_server.services.versioned_resources.base import VersionedResourceService
from sqlalchemy import JSON, DateTime, Integer, String, UniqueConstraint
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

import memorylayer_saas.storage.versioned_resources as versioned_store_module
from memorylayer_saas.storage.postgresql import PostgreSQLBackend


class _Base(DeclarativeBase):
    pass


class _Head(_Base):
    __tablename__ = "versioned_resource_heads"
    tenant_id: Mapped[str] = mapped_column(String, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String, primary_key=True)
    namespace: Mapped[str] = mapped_column(String, primary_key=True)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    resource_key: Mapped[str] = mapped_column(String, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[dict] = mapped_column(JSON, nullable=False)
    meta: Mapped[dict] = mapped_column("metadata", JSON, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    state_hash: Mapped[str] = mapped_column(String, nullable=False)
    etag: Mapped[str] = mapped_column(String, nullable=False)
    created_by: Mapped[str | None] = mapped_column(String)
    updated_by: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime)
    __table_args__ = (
        UniqueConstraint("tenant_id", "workspace_id", "namespace", "resource_key"),
    )


class _Revision(_Base):
    __tablename__ = "versioned_resource_revisions"
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    workspace_id: Mapped[str] = mapped_column(String, nullable=False)
    namespace: Mapped[str] = mapped_column(String, nullable=False)
    id: Mapped[str] = mapped_column(String, nullable=False)
    resource_key: Mapped[str] = mapped_column(String, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[dict] = mapped_column(JSON, nullable=False)
    meta: Mapped[dict] = mapped_column("metadata", JSON, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    state_hash: Mapped[str] = mapped_column(String, nullable=False)
    etag: Mapped[str] = mapped_column(String, nullable=False)
    created_by: Mapped[str | None] = mapped_column(String)
    updated_by: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime)
    action: Mapped[str] = mapped_column(String, nullable=False)
    operation_id: Mapped[str] = mapped_column(String, nullable=False)
    request_hash: Mapped[str] = mapped_column(String, nullable=False)
    __table_args__ = (
        UniqueConstraint("tenant_id", "workspace_id", "namespace", "id", "revision"),
    )


class _Operation(_Base):
    __tablename__ = "versioned_resource_operations"
    tenant_id: Mapped[str] = mapped_column(String, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String, primary_key=True)
    namespace: Mapped[str] = mapped_column(String, primary_key=True)
    operation_id: Mapped[str] = mapped_column(String, primary_key=True)
    request_hash: Mapped[str] = mapped_column(String, nullable=False)
    resource_id: Mapped[str] = mapped_column(String, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)


@pytest.fixture
async def backend(monkeypatch):
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as connection:
        await connection.run_sync(_Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(versioned_store_module, "VersionedResourceHeadModel", _Head)
    monkeypatch.setattr(versioned_store_module, "VersionedResourceRevisionModel", _Revision)
    monkeypatch.setattr(versioned_store_module, "VersionedResourceOperationModel", _Operation)
    instance = PostgreSQLBackend()
    instance._session_factory = factory
    instance._versioned_resource_store = None
    yield instance
    await engine.dispose()


def _content(title: str) -> dict:
    return {"title": title, "content": title, "enabled": True}


@pytest.mark.asyncio
async def test_postgres_adapter_cas_replay_history_and_tombstone(backend):
    service = VersionedResourceService(backend)
    created = await service.create(
        tenant_id="tenant_a",
        workspace_id="ws_a",
        namespace="agent.prompt-note.v1",
        resource_key="coding/style",
        schema_version=1,
        content=_content("initial"),
        metadata={},
        actor="user:alice",
        operation_id="create-1",
        expected_etag="*",
    )
    replay = await service.create(
        tenant_id="tenant_a",
        workspace_id="ws_a",
        namespace="agent.prompt-note.v1",
        resource_key="coding/style",
        schema_version=1,
        content=_content("initial"),
        metadata={},
        actor="user:alice",
        operation_id="create-1",
        expected_etag="*",
    )
    assert replay.replayed is True
    assert replay.resource.id == created.resource.id

    with pytest.raises(VersionedResourceConflictError):
        await service.create(
            tenant_id="tenant_a",
            workspace_id="ws_a",
            namespace="agent.prompt-note.v1",
            resource_key="different",
            schema_version=1,
            content=_content("different"),
            metadata={},
            actor="user:alice",
            operation_id="create-1",
            expected_etag="*",
        )

    replaced = await service.replace(
        tenant_id="tenant_a",
        workspace_id="ws_a",
        namespace="agent.prompt-note.v1",
        resource_id=created.resource.id,
        schema_version=2,
        content=_content("replacement"),
        metadata={"reviewed": True},
        actor="user:bob",
        operation_id="replace-1",
        expected_etag=created.resource.etag,
    )
    with pytest.raises(VersionedResourcePreconditionFailedError):
        await service.replace(
            tenant_id="tenant_a",
            workspace_id="ws_a",
            namespace="agent.prompt-note.v1",
            resource_id=created.resource.id,
            schema_version=2,
            content=_content("stale"),
            metadata={},
            actor="user:carol",
            operation_id="replace-stale",
            expected_etag=created.resource.etag,
        )

    deleted = await service.delete(
        tenant_id="tenant_a",
        workspace_id="ws_a",
        namespace="agent.prompt-note.v1",
        resource_id=created.resource.id,
        actor="user:bob",
        operation_id="delete-1",
        expected_etag=replaced.resource.etag,
    )
    assert deleted.resource.deleted_at is not None
    delete_replay = await service.delete(
        tenant_id="tenant_a",
        workspace_id="ws_a",
        namespace="agent.prompt-note.v1",
        resource_id=created.resource.id,
        actor="user:bob",
        operation_id="delete-1",
        expected_etag=replaced.resource.etag,
    )
    assert delete_replay.replayed is True

    restored = await service.restore(
        tenant_id="tenant_a",
        workspace_id="ws_a",
        namespace="agent.prompt-note.v1",
        resource_id=created.resource.id,
        actor="user:bob",
        operation_id="restore-1",
        expected_etag=deleted.resource.etag,
    )
    assert restored.resource.deleted_at is None
    assert restored.resource.id == created.resource.id
    assert restored.resource.content == replaced.resource.content
    restore_replay = await service.restore(
        tenant_id="tenant_a",
        workspace_id="ws_a",
        namespace="agent.prompt-note.v1",
        resource_id=created.resource.id,
        actor="user:bob",
        operation_id="restore-1",
        expected_etag=deleted.resource.etag,
    )
    assert restore_replay.replayed is True

    with pytest.raises(VersionedResourceConflictError, match="not deleted"):
        await service.restore(
            tenant_id="tenant_a",
            workspace_id="ws_a",
            namespace="agent.prompt-note.v1",
            resource_id=created.resource.id,
            actor="user:bob",
            operation_id="restore-active",
            expected_etag=restored.resource.etag,
        )

    revisions, _ = await service.list_revision_page(
        "tenant_a", "ws_a", "agent.prompt-note.v1", created.resource.id, limit=10
    )
    assert [item.action for item in revisions] == ["restore", "delete", "replace", "create"]
