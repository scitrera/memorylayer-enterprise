# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Collections REST API must embed item content so /v1/collections/search finds it.

Regression: create/update never computed an embedding, and search filters on
``embedding IS NOT NULL``, so items created through REST were never returned.
"""
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from memorylayer_server.models.auth import RequestContext
from pydantic import ValidationError

from memorylayer_saas.api.v1 import collections as api
from memorylayer_saas.models.collection import CollectionItem

EMBEDDING = [0.1, 0.2, 0.3]


def _ctx() -> RequestContext:
    return RequestContext(tenant_id="tenant-1", workspace_id="ws-1", user_id="user-1", metadata={})


def _auth():
    auth = AsyncMock()
    auth.build_context = AsyncMock(return_value=_ctx())
    authz = AsyncMock()
    authz.require_authorization = AsyncMock(return_value=None)
    return auth, authz


def _embedding_service():
    svc = MagicMock()
    svc.embed = AsyncMock(return_value=EMBEDDING)
    return svc


def _item(**overrides) -> CollectionItem:
    now = datetime.now(UTC)
    fields = dict(
        id="col_1", tenant_id="tenant-1", workspace_id="ws-1", collection_name="tools",
        name="grep", content="search text in files", created_at=now, updated_at=now,
    )
    fields.update(overrides)
    return CollectionItem(**fields)


@pytest.mark.asyncio
async def test_create_embeds_content():
    auth, authz = _auth()
    embedding = _embedding_service()
    service = MagicMock()
    service.create_item = AsyncMock(side_effect=lambda item: item)

    await api.create_collection_item(
        http_request=MagicMock(),
        request=api.CollectionItemCreateRequest(collection_name="tools", name="grep", content="search text in files"),
        auth_service=auth, authz_service=authz, service=service, embedding_service=embedding,
        audit_service=AsyncMock(), logger=MagicMock(),
    )

    embedding.embed.assert_awaited_once_with("search text in files")
    stored = service.create_item.await_args.args[0]
    assert stored.embedding == EMBEDDING


def _failing_embedding_service(error: Exception):
    svc = MagicMock()
    svc.embed = AsyncMock(side_effect=error)
    return svc


EMBED_FAILURES = [RuntimeError("embed server down"), ValueError("provider rejected input: secret-detail")]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", EMBED_FAILURES)
async def test_create_returns_503_and_stores_nothing_when_embedding_fails(error):
    auth, authz = _auth()
    service = MagicMock()
    service.create_item = AsyncMock()

    with pytest.raises(HTTPException) as exc:
        await api.create_collection_item(
            http_request=MagicMock(),
            request=api.CollectionItemCreateRequest(collection_name="tools", name="grep", content="x"),
            auth_service=auth, authz_service=authz, service=service,
            embedding_service=_failing_embedding_service(error),
            audit_service=AsyncMock(), logger=MagicMock(),
        )
    assert exc.value.status_code == 503
    assert exc.value.detail == api.EMBEDDING_UNAVAILABLE_DETAIL
    assert "secret-detail" not in exc.value.detail
    service.create_item.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_request_validation_error_stays_400():
    """A ValueError from the request itself (empty collection name) is still a 400."""
    auth, authz = _auth()
    service = MagicMock()
    service.create_item = AsyncMock()

    with pytest.raises(HTTPException) as exc:
        await api.create_collection_item(
            http_request=MagicMock(),
            request=api.CollectionItemCreateRequest(collection_name="  ", name="grep", content="x"),
            auth_service=auth, authz_service=authz, service=service,
            embedding_service=_embedding_service(), audit_service=AsyncMock(), logger=MagicMock(),
        )
    assert exc.value.status_code == 400
    service.create_item.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", EMBED_FAILURES)
async def test_update_returns_503_when_embedding_fails(error):
    auth, authz = _auth()
    service = MagicMock()
    service.get_item = AsyncMock(return_value=_item())
    service.update_item = AsyncMock()

    with pytest.raises(HTTPException) as exc:
        await api.update_collection_item(
            http_request=MagicMock(), item_id="col_1",
            request=api.CollectionItemUpdateRequest(content="new text"),
            auth_service=auth, authz_service=authz, service=service,
            embedding_service=_failing_embedding_service(error),
            audit_service=AsyncMock(), logger=MagicMock(),
        )
    assert exc.value.status_code == 503
    service.update_item.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_missing_item_is_404_before_embedding():
    auth, authz = _auth()
    embedding = _embedding_service()
    service = MagicMock()
    service.get_item = AsyncMock(return_value=None)
    service.update_item = AsyncMock()

    with pytest.raises(HTTPException) as exc:
        await api.update_collection_item(
            http_request=MagicMock(), item_id="col_missing",
            request=api.CollectionItemUpdateRequest(content="new text"),
            auth_service=auth, authz_service=authz, service=service, embedding_service=embedding,
            audit_service=AsyncMock(), logger=MagicMock(),
        )
    assert exc.value.status_code == 404
    embedding.embed.assert_not_awaited()
    service.update_item.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", EMBED_FAILURES)
async def test_text_search_returns_503_when_embedding_fails(error):
    auth, authz = _auth()
    service = MagicMock()
    service.search_similar = AsyncMock()

    with pytest.raises(HTTPException) as exc:
        await api.search_collections(
            http_request=MagicMock(),
            request=api.CollectionSearchRequest(query="find text"),
            auth_service=auth, authz_service=authz, service=service,
            embedding_service=_failing_embedding_service(error), logger=MagicMock(),
        )
    assert exc.value.status_code == 503
    service.search_similar.assert_not_awaited()


def test_create_request_rejects_empty_content():
    with pytest.raises(ValidationError):
        api.CollectionItemCreateRequest(collection_name="tools", name="grep", content="")


def test_update_request_rejects_empty_content_but_allows_omitting_it():
    with pytest.raises(ValidationError):
        api.CollectionItemUpdateRequest(content="")
    assert api.CollectionItemUpdateRequest(enabled=False).content is None


@pytest.mark.asyncio
async def test_update_reembeds_changed_content():
    auth, authz = _auth()
    embedding = _embedding_service()
    service = MagicMock()
    service.get_item = AsyncMock(return_value=_item())
    service.update_item = AsyncMock(return_value=_item(content="new text"))

    await api.update_collection_item(
        http_request=MagicMock(), item_id="col_1",
        request=api.CollectionItemUpdateRequest(content="new text"),
        auth_service=auth, authz_service=authz, service=service, embedding_service=embedding,
        audit_service=AsyncMock(), logger=MagicMock(),
    )

    embedding.embed.assert_awaited_once_with("new text")
    service.update_item.assert_awaited_once_with("col_1", "ws-1", content="new text", embedding=EMBEDDING)


@pytest.mark.asyncio
async def test_update_without_content_keeps_embedding():
    auth, authz = _auth()
    embedding = _embedding_service()
    service = MagicMock()
    service.update_item = AsyncMock(return_value=_item(enabled=False))

    await api.update_collection_item(
        http_request=MagicMock(), item_id="col_1",
        request=api.CollectionItemUpdateRequest(enabled=False),
        auth_service=auth, authz_service=authz, service=service, embedding_service=embedding,
        audit_service=AsyncMock(), logger=MagicMock(),
    )

    embedding.embed.assert_not_awaited()
    service.update_item.assert_awaited_once_with("col_1", "ws-1", enabled=False)


@pytest.mark.asyncio
async def test_search_embeds_query_text():
    auth, authz = _auth()
    embedding = _embedding_service()
    service = MagicMock()
    service.search_similar = AsyncMock(return_value=[(_item(), 0.9)])

    response = await api.search_collections(
        http_request=MagicMock(),
        request=api.CollectionSearchRequest(query="find text", collection_name="tools"),
        auth_service=auth, authz_service=authz, service=service, embedding_service=embedding,
        logger=MagicMock(),
    )

    embedding.embed.assert_awaited_once_with("find text")
    service.search_similar.assert_awaited_once_with(
        workspace_id="ws-1", query_embedding=EMBEDDING, collection_name="tools", limit=10,
    )
    assert response.total_count == 1 and response.results[0].item.name == "grep"


@pytest.mark.asyncio
async def test_search_accepts_precomputed_embedding():
    auth, authz = _auth()
    embedding = _embedding_service()
    service = MagicMock()
    service.search_similar = AsyncMock(return_value=[])

    await api.search_collections(
        http_request=MagicMock(),
        request=api.CollectionSearchRequest(query_embedding=[1.0, 0.0, 0.0]),
        auth_service=auth, authz_service=authz, service=service, embedding_service=embedding,
        logger=MagicMock(),
    )

    embedding.embed.assert_not_awaited()
    assert service.search_similar.await_args.kwargs["query_embedding"] == [1.0, 0.0, 0.0]


@pytest.mark.parametrize("body", [{}, {"query": "a", "query_embedding": [1.0]}, {"query": "  "}])
def test_search_request_requires_exactly_one_query(body):
    with pytest.raises(ValidationError):
        api.CollectionSearchRequest(**body)
