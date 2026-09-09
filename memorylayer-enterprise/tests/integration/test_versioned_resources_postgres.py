"""Live PostgreSQL conformance for versioned-resource transactional races."""

import asyncio
import os
import uuid

import pytest
import pytest_asyncio
from memorylayer_server.models.versioned_resource import (
    VersionedResourcePreconditionFailedError,
)
from memorylayer_server.services.versioned_resources.base import VersionedResourceService
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from memorylayer_saas.storage.models import (
    Base,
    VersionedResourceHeadModel,
    VersionedResourceOperationModel,
    VersionedResourceRevisionModel,
)
from memorylayer_saas.storage.postgresql import PostgreSQLBackend

_DB_URL = os.getenv("ML_AGE_TEST_DATABASE_URL", "").strip()
pytestmark = pytest.mark.skipif(
    not _DB_URL,
    reason="Set ML_AGE_TEST_DATABASE_URL to run live PostgreSQL conformance.",
)


def _asyncpg_url(url: str) -> str:
    return (
        url.replace("postgres://", "postgresql://")
        .replace("postgresql+asyncpg://", "postgresql://")
        .replace("postgresql://", "postgresql+asyncpg://")
    )


@pytest_asyncio.fixture
async def backend():
    engine = create_async_engine(_asyncpg_url(_DB_URL), pool_pre_ping=True)
    async with engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all,
            tables=[
                VersionedResourceRevisionModel.__table__,
                VersionedResourceHeadModel.__table__,
                VersionedResourceOperationModel.__table__,
            ],
        )
    instance = PostgreSQLBackend(connection_string=_asyncpg_url(_DB_URL))
    instance._engine = engine
    instance._session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    suffix = uuid.uuid4().hex
    tenant_id = f"vr_tenant_{suffix}"
    workspace_id = f"vr_workspace_{suffix}"
    yield instance, tenant_id, workspace_id

    async with instance._session_factory() as session:
        for model in (
            VersionedResourceOperationModel,
            VersionedResourceHeadModel,
            VersionedResourceRevisionModel,
        ):
            await session.execute(
                delete(model).where(
                    model.tenant_id == tenant_id,
                    model.workspace_id == workspace_id,
                )
            )
        await session.commit()
    await engine.dispose()


@pytest.mark.asyncio
async def test_live_postgres_serializes_competing_etag_writers(backend):
    storage, tenant_id, workspace_id = backend
    service = VersionedResourceService(storage)
    created = await service.create(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        namespace="agent.prompt-note.v1",
        resource_key="race",
        schema_version=1,
        content={"title": "race", "content": "initial", "enabled": True},
        metadata={},
        actor="user:alice",
        operation_id="create-race",
        expected_etag="*",
    )

    async def replace(label: str):
        return await service.replace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace="agent.prompt-note.v1",
            resource_id=created.resource.id,
            schema_version=1,
            content={"title": label, "content": label, "enabled": True},
            metadata={},
            actor=f"user:{label}",
            operation_id=f"replace-{label}",
            expected_etag=created.resource.etag,
        )

    outcomes = await asyncio.gather(replace("one"), replace("two"), return_exceptions=True)
    assert sum(not isinstance(item, Exception) for item in outcomes) == 1
    assert sum(
        isinstance(item, VersionedResourcePreconditionFailedError)
        for item in outcomes
    ) == 1
