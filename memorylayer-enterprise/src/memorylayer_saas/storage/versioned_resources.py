"""PostgreSQL authority for OSS typed versioned-resource contracts."""

from __future__ import annotations

from memorylayer_server.models.versioned_resource import (
    VersionedResource,
    VersionedResourceConflictError,
    VersionedResourceMutation,
    VersionedResourceMutationResult,
    VersionedResourceNotFoundError,
    VersionedResourcePreconditionFailedError,
    VersionedResourceRevision,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .models import (
    VersionedResourceHeadModel,
    VersionedResourceOperationModel,
    VersionedResourceRevisionModel,
)


class PostgreSQLVersionedResourceStore:
    """Transactional PostgreSQL implementation shared by typed OSS APIs."""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    async def mutate(self, mutation: VersionedResourceMutation) -> VersionedResourceMutationResult:
        desired = mutation.resource
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    replay = await self._operation_revision(
                        session,
                        desired.tenant_id,
                        desired.workspace_id,
                        desired.namespace,
                        mutation.operation_id,
                    )
                    if replay is not None:
                        return self._validate_replay(replay, mutation.request_hash)

                    current = await self._select_head(
                        session,
                        desired.tenant_id,
                        desired.workspace_id,
                        desired.namespace,
                        desired.id,
                        for_update=True,
                    )
                    if mutation.action == "create":
                        return await self._create(session, mutation, current)

                    # A competing writer may have committed while this transaction
                    # waited on the head lock. Re-check the operation before
                    # classifying the now-stale ETag.
                    replay = await self._operation_revision(
                        session,
                        desired.tenant_id,
                        desired.workspace_id,
                        desired.namespace,
                        mutation.operation_id,
                    )
                    if replay is not None:
                        return self._validate_replay(replay, mutation.request_hash)
                    return await self._replace_or_delete(session, mutation, current)
        except IntegrityError as exc:
            replay = await self.get_operation_result(
                desired.tenant_id,
                desired.workspace_id,
                desired.namespace,
                mutation.operation_id,
                mutation.request_hash,
            )
            if replay is not None:
                return replay
            if mutation.action == "create":
                existing = await self.get(
                    desired.tenant_id,
                    desired.workspace_id,
                    desired.namespace,
                    desired.id,
                    include_deleted=True,
                )
                existing_key = await self.get_by_key(
                    desired.tenant_id,
                    desired.workspace_id,
                    desired.namespace,
                    desired.resource_key,
                    include_deleted=True,
                )
                if existing is not None or existing_key is not None:
                    raise VersionedResourceConflictError("resource id or key already exists") from exc
            raise

    async def _create(self, session, mutation, current) -> VersionedResourceMutationResult:
        desired = mutation.resource
        if mutation.expected_etag != "*":
            raise VersionedResourcePreconditionFailedError("create requires If-None-Match: *")
        existing_key = await self._select_head_by_key(
            session,
            desired.tenant_id,
            desired.workspace_id,
            desired.namespace,
            desired.resource_key,
            for_update=True,
        )
        if current is not None or existing_key is not None:
            raise VersionedResourceConflictError("resource id or key already exists")

        final = desired.model_copy(
            update={"revision": 1, "etag": _etag(1, desired.state_hash)}
        )
        revision = _revision_model(final, mutation)
        session.add(revision)
        await session.flush()
        final = final.model_copy(update={"sequence": revision.sequence})
        session.add(_head_model(final))
        session.add(_operation_model(final, mutation))
        return VersionedResourceMutationResult(resource=final)

    async def _replace_or_delete(self, session, mutation, current) -> VersionedResourceMutationResult:
        desired = mutation.resource
        if current is None:
            raise VersionedResourceNotFoundError("resource not found")
        if mutation.expected_etag != current.etag:
            raise VersionedResourcePreconditionFailedError("ETag does not match current revision")
        if mutation.action == "restore":
            if current.deleted_at is None:
                raise VersionedResourceConflictError("resource is not deleted")
        elif current.deleted_at is not None:
            raise VersionedResourceNotFoundError("resource not found")
        if desired.resource_key != current.resource_key:
            raise VersionedResourceConflictError("resource key is immutable")

        next_revision = current.revision + 1
        final = desired.model_copy(
            update={
                "created_at": current.created_at,
                "created_by": current.created_by,
                "revision": next_revision,
                "etag": _etag(next_revision, desired.state_hash),
            }
        )
        revision = _revision_model(final, mutation)
        session.add(revision)
        await session.flush()
        final = final.model_copy(update={"sequence": revision.sequence})
        _apply_head(current, final)
        session.add(_operation_model(final, mutation))
        return VersionedResourceMutationResult(resource=final)

    async def get_operation_result(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        operation_id: str,
        request_hash: str,
    ) -> VersionedResourceMutationResult | None:
        async with self._session_factory() as session:
            revision = await self._operation_revision(
                session, tenant_id, workspace_id, namespace, operation_id
            )
        if revision is None:
            return None
        return self._validate_replay(revision, request_hash)

    async def get(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        *,
        include_deleted: bool = False,
    ) -> VersionedResource | None:
        async with self._session_factory() as session:
            row = await self._select_head(
                session, tenant_id, workspace_id, namespace, resource_id
            )
        if row is None or (row.deleted_at is not None and not include_deleted):
            return None
        return _resource_from_head(row)

    async def get_by_key(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_key: str,
        *,
        include_deleted: bool = False,
    ) -> VersionedResource | None:
        async with self._session_factory() as session:
            row = await self._select_head_by_key(
                session, tenant_id, workspace_id, namespace, resource_key
            )
        if row is None or (row.deleted_at is not None and not include_deleted):
            return None
        return _resource_from_head(row)

    async def list(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        *,
        limit: int,
        before_sequence: int | None = None,
        include_deleted: bool = False,
    ) -> list[VersionedResource]:
        conditions = [
            VersionedResourceHeadModel.tenant_id == tenant_id,
            VersionedResourceHeadModel.workspace_id == workspace_id,
            VersionedResourceHeadModel.namespace == namespace,
        ]
        if not include_deleted:
            conditions.append(VersionedResourceHeadModel.deleted_at.is_(None))
        if before_sequence is not None:
            conditions.append(VersionedResourceHeadModel.sequence < before_sequence)
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(VersionedResourceHeadModel)
                    .where(*conditions)
                    .order_by(VersionedResourceHeadModel.sequence.desc())
                    .limit(limit)
                )
            ).scalars().all()
        return [_resource_from_head(row) for row in rows]

    async def list_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[VersionedResourceRevision]:
        conditions = [
            VersionedResourceRevisionModel.tenant_id == tenant_id,
            VersionedResourceRevisionModel.workspace_id == workspace_id,
            VersionedResourceRevisionModel.namespace == namespace,
            VersionedResourceRevisionModel.id == resource_id,
        ]
        if before_sequence is not None:
            conditions.append(VersionedResourceRevisionModel.sequence < before_sequence)
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(VersionedResourceRevisionModel)
                    .where(*conditions)
                    .order_by(VersionedResourceRevisionModel.sequence.desc())
                    .limit(limit)
                )
            ).scalars().all()
        return [_revision_from_model(row) for row in rows]

    async def _select_head(
        self,
        session,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        *,
        for_update: bool = False,
    ):
        statement = select(VersionedResourceHeadModel).where(
            VersionedResourceHeadModel.tenant_id == tenant_id,
            VersionedResourceHeadModel.workspace_id == workspace_id,
            VersionedResourceHeadModel.namespace == namespace,
            VersionedResourceHeadModel.id == resource_id,
        )
        if for_update:
            statement = statement.with_for_update()
        return (await session.execute(statement)).scalar_one_or_none()

    async def _select_head_by_key(
        self,
        session,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_key: str,
        *,
        for_update: bool = False,
    ):
        statement = select(VersionedResourceHeadModel).where(
            VersionedResourceHeadModel.tenant_id == tenant_id,
            VersionedResourceHeadModel.workspace_id == workspace_id,
            VersionedResourceHeadModel.namespace == namespace,
            VersionedResourceHeadModel.resource_key == resource_key,
        )
        if for_update:
            statement = statement.with_for_update()
        return (await session.execute(statement)).scalar_one_or_none()

    async def _operation_revision(
        self,
        session,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        operation_id: str,
    ) -> VersionedResourceRevision | None:
        operation = (
            await session.execute(
                select(VersionedResourceOperationModel).where(
                    VersionedResourceOperationModel.tenant_id == tenant_id,
                    VersionedResourceOperationModel.workspace_id == workspace_id,
                    VersionedResourceOperationModel.namespace == namespace,
                    VersionedResourceOperationModel.operation_id == operation_id,
                )
            )
        ).scalar_one_or_none()
        if operation is None:
            return None
        revision = (
            await session.execute(
                select(VersionedResourceRevisionModel).where(
                    VersionedResourceRevisionModel.tenant_id == tenant_id,
                    VersionedResourceRevisionModel.workspace_id == workspace_id,
                    VersionedResourceRevisionModel.namespace == namespace,
                    VersionedResourceRevisionModel.id == operation.resource_id,
                    VersionedResourceRevisionModel.revision == operation.revision,
                )
            )
        ).scalar_one()
        return _revision_from_model(revision)

    @staticmethod
    def _validate_replay(
        revision: VersionedResourceRevision,
        request_hash: str,
    ) -> VersionedResourceMutationResult:
        if revision.request_hash != request_hash:
            raise VersionedResourceConflictError(
                "idempotency key was already used for a different request"
            )
        return VersionedResourceMutationResult(
            resource=revision.as_resource(), replayed=True
        )


def _etag(revision: int, state_hash: str) -> str:
    return f'"vr-{revision}-{state_hash}"'


def _head_model(resource: VersionedResource):
    return VersionedResourceHeadModel(
        id=resource.id,
        tenant_id=resource.tenant_id,
        workspace_id=resource.workspace_id,
        namespace=resource.namespace,
        resource_key=resource.resource_key,
        schema_version=resource.schema_version,
        content=resource.content,
        meta=resource.metadata,
        revision=resource.revision,
        sequence=resource.sequence,
        state_hash=resource.state_hash,
        etag=resource.etag,
        created_by=resource.created_by,
        updated_by=resource.updated_by,
        created_at=resource.created_at,
        updated_at=resource.updated_at,
        deleted_at=resource.deleted_at,
    )


def _revision_model(resource: VersionedResource, mutation: VersionedResourceMutation):
    return VersionedResourceRevisionModel(
        id=resource.id,
        tenant_id=resource.tenant_id,
        workspace_id=resource.workspace_id,
        namespace=resource.namespace,
        resource_key=resource.resource_key,
        schema_version=resource.schema_version,
        content=resource.content,
        meta=resource.metadata,
        revision=resource.revision,
        state_hash=resource.state_hash,
        etag=resource.etag,
        created_by=resource.created_by,
        updated_by=resource.updated_by,
        created_at=resource.created_at,
        updated_at=resource.updated_at,
        deleted_at=resource.deleted_at,
        action=mutation.action,
        operation_id=mutation.operation_id,
        request_hash=mutation.request_hash,
    )


def _operation_model(resource: VersionedResource, mutation: VersionedResourceMutation):
    return VersionedResourceOperationModel(
        tenant_id=resource.tenant_id,
        workspace_id=resource.workspace_id,
        namespace=resource.namespace,
        operation_id=mutation.operation_id,
        request_hash=mutation.request_hash,
        resource_id=resource.id,
        revision=resource.revision,
    )


def _apply_head(head, resource: VersionedResource) -> None:
    head.schema_version = resource.schema_version
    head.content = resource.content
    head.meta = resource.metadata
    head.revision = resource.revision
    head.sequence = resource.sequence
    head.state_hash = resource.state_hash
    head.etag = resource.etag
    head.updated_by = resource.updated_by
    head.updated_at = resource.updated_at
    head.deleted_at = resource.deleted_at


def _resource_from_head(row) -> VersionedResource:
    return VersionedResource(
        id=row.id,
        tenant_id=row.tenant_id,
        workspace_id=row.workspace_id,
        namespace=row.namespace,
        resource_key=row.resource_key,
        schema_version=row.schema_version,
        content=dict(row.content),
        metadata=dict(row.meta),
        revision=row.revision,
        sequence=row.sequence,
        state_hash=row.state_hash,
        etag=row.etag,
        created_by=row.created_by,
        updated_by=row.updated_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
        deleted_at=row.deleted_at,
    )


def _revision_from_model(row) -> VersionedResourceRevision:
    resource = _resource_from_head(row)
    return VersionedResourceRevision(
        **resource.model_dump(),
        action=row.action,
        operation_id=row.operation_id,
        request_hash=row.request_hash,
    )
