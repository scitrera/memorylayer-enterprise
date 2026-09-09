# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""
Unit tests for documents API authentication and authorization enforcement.

Verifies that each endpoint:
- Calls build_context to extract the caller's identity
- Calls require_authorization with the correct resource and action
- Propagates ctx.workspace_id to downstream service calls
- Returns 401/403 on auth failures
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi import FastAPI

from memorylayer_server.models.auth import RequestContext
from memorylayer_server.services.authentication import AuthenticationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(workspace_id: str = "ws-test") -> RequestContext:
    """Build a minimal RequestContext for testing."""
    return RequestContext(
        tenant_id="tenant-1",
        workspace_id=workspace_id,
        user_id="user-1",
        metadata={},
    )


def _make_auth_service(ctx: RequestContext | None = None) -> AsyncMock:
    """Create a mock AuthenticationService that returns the given context."""
    svc = AsyncMock()
    svc.build_context = AsyncMock(return_value=ctx or _make_ctx())
    return svc


def _make_authz_service(deny: bool = False) -> AsyncMock:
    """Create a mock AuthorizationService that allows or denies."""
    svc = AsyncMock()
    if deny:
        svc.require_authorization = AsyncMock(
            side_effect=HTTPException(status_code=403, detail="Access denied")
        )
    else:
        svc.require_authorization = AsyncMock(return_value=None)
    return svc


def _make_doc_service() -> AsyncMock:
    """Create a minimal mock DocumentIngestionService."""
    svc = AsyncMock()
    # Default implementations for common calls
    svc.list_documents = AsyncMock(return_value=([], 0))
    svc.list_jobs = AsyncMock(return_value=[])
    svc.get_job = AsyncMock(return_value=None)
    svc.cancel_job = AsyncMock(return_value=None)
    return svc


# ---------------------------------------------------------------------------
# Tests for build_context + require_authorization calls
# ---------------------------------------------------------------------------

class TestDocumentAuthCalls:
    """Verify auth wiring on each documents endpoint."""

    @pytest.mark.asyncio
    async def test_list_documents_calls_auth(self):
        """GET /v1/documents must call build_context and require_authorization."""
        from memorylayer_saas.api.v1.documents import list_documents

        ctx = _make_ctx("ws-abc")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        doc_svc = _make_doc_service()
        request = MagicMock()
        logger = MagicMock()

        await list_documents(
            http_request=request,
            status_filter=None,
            # Calling the endpoint directly bypasses FastAPI, which is what
            # would normally turn a ``Query(None)`` default into None. Query
            # params the assertions depend on must therefore be passed here, or
            # the raw Query sentinel object is what reaches the service.
            workspace_id=None,
            limit=10,
            offset=0,
            auth_service=auth_svc,
            authz_service=authz_svc,
            service=doc_svc,
            logger=logger,
        )

        auth_svc.build_context.assert_awaited_once_with(request)
        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "documents", "read", workspace_id="ws-abc"
        )
        doc_svc.list_documents.assert_awaited_once()
        call_kwargs = doc_svc.list_documents.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-abc"

    @pytest.mark.asyncio
    async def test_list_documents_auth_error_raises_401(self):
        """Authentication failure must surface as HTTP 401."""
        from memorylayer_saas.api.v1.documents import list_documents

        auth_svc = _make_auth_service()
        auth_svc.build_context = AsyncMock(
            side_effect=AuthenticationError("bad token", status_code=401)
        )
        authz_svc = _make_authz_service()
        doc_svc = _make_doc_service()
        logger = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await list_documents(
                http_request=MagicMock(),
                status_filter=None,
                limit=10,
                offset=0,
                auth_service=auth_svc,
                authz_service=authz_svc,
                service=doc_svc,
                logger=logger,
            )
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_list_jobs_calls_auth(self):
        """GET /v1/documents/jobs must call build_context and require_authorization."""
        from memorylayer_saas.api.v1.documents import list_jobs

        ctx = _make_ctx("ws-jobs")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        doc_svc = _make_doc_service()
        request = MagicMock()
        logger = MagicMock()

        await list_jobs(
            http_request=request,
            status_filter=None,
            limit=10,
            auth_service=auth_svc,
            authz_service=authz_svc,
            service=doc_svc,
            logger=logger,
        )

        auth_svc.build_context.assert_awaited_once_with(request)
        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "documents", "read", workspace_id="ws-jobs"
        )

    @pytest.mark.asyncio
    async def test_get_job_calls_auth(self):
        """GET /v1/documents/jobs/{id} must call build_context and require_authorization."""
        from memorylayer_saas.api.v1.documents import get_job

        ctx = _make_ctx("ws-job-read")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        doc_svc = _make_doc_service()
        request = MagicMock()
        logger = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await get_job(
                http_request=request,
                job_id="job-123",
                auth_service=auth_svc,
                authz_service=authz_svc,
                service=doc_svc,
                logger=logger,
            )
        # Job not found (404) means auth was called successfully first
        assert exc_info.value.status_code == 404

        auth_svc.build_context.assert_awaited_once_with(request)
        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "documents", "read", workspace_id="ws-job-read"
        )

    @pytest.mark.asyncio
    async def test_cancel_job_calls_auth_write(self):
        """POST /v1/documents/jobs/{id}/cancel must require 'write' action."""
        from memorylayer_saas.api.v1.documents import cancel_job

        ctx = _make_ctx("ws-cancel")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        doc_svc = _make_doc_service()
        request = MagicMock()
        logger = MagicMock()

        await cancel_job(
            http_request=request,
            job_id="job-xyz",
            auth_service=auth_svc,
            authz_service=authz_svc,
            service=doc_svc,
            logger=logger,
        )

        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "documents", "write", workspace_id="ws-cancel"
        )

    @pytest.mark.asyncio
    async def test_get_document_calls_auth(self):
        """GET /v1/documents/{id} must call build_context and require_authorization."""
        from memorylayer_saas.api.v1.documents import get_document

        ctx = _make_ctx("ws-get-doc")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        doc_svc = _make_doc_service()
        doc_svc.get_document = AsyncMock(return_value=None)
        request = MagicMock()
        logger = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await get_document(
                http_request=request,
                document_id="doc-abc",
                auth_service=auth_svc,
                authz_service=authz_svc,
                service=doc_svc,
                logger=logger,
            )
        assert exc_info.value.status_code == 404

        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "documents", "read", workspace_id="ws-get-doc"
        )

    @pytest.mark.asyncio
    async def test_delete_document_calls_auth_delete(self):
        """DELETE /v1/documents/{id} must require 'delete' action."""
        from memorylayer_saas.api.v1.documents import delete_document

        ctx = _make_ctx("ws-delete-doc")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        doc_svc = _make_doc_service()
        doc_svc.delete_document = AsyncMock(return_value=None)
        request = MagicMock()
        logger = MagicMock()

        await delete_document(
            http_request=request,
            document_id="doc-del",
            delete_memories=False,
            # Omitted here means "fall back to the auth-context workspace"; pass
            # None explicitly since FastAPI is not present to resolve the
            # Query(None) default (see test_list_documents_calls_auth).
            workspace_id=None,
            auth_service=auth_svc,
            authz_service=authz_svc,
            service=doc_svc,
            logger=logger,
        )

        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "documents", "delete", workspace_id="ws-delete-doc"
        )
        doc_svc.delete_document.assert_awaited_once()
        call_kwargs = doc_svc.delete_document.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-delete-doc"

    @pytest.mark.asyncio
    async def test_delete_document_authz_denied_raises_403(self):
        """Authorization failure on delete must raise 403."""
        from memorylayer_saas.api.v1.documents import delete_document

        ctx = _make_ctx("ws-denied")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service(deny=True)
        doc_svc = _make_doc_service()
        doc_svc.delete_document = AsyncMock(return_value=None)
        logger = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await delete_document(
                http_request=MagicMock(),
                document_id="doc-x",
                delete_memories=False,
                auth_service=auth_svc,
                authz_service=authz_svc,
                service=doc_svc,
                logger=logger,
            )
        assert exc_info.value.status_code == 403
        # Service should not be called when authz fails
        doc_svc.delete_document.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reprocess_document_calls_auth_write(self):
        """POST /v1/documents/{id}/reprocess must require 'write' action."""
        from memorylayer_saas.api.v1.documents import reprocess_document

        ctx = _make_ctx("ws-reprocess")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()

        mock_job = MagicMock()
        mock_job.model_dump.return_value = {
            "id": "job-1",
            "workspace_id": "ws-reprocess",
            "document_ids": [],
            "status": "pending",
            "progress_percent": 0,
            "documents_processed": 0,
            "total_memories_created": 0,
            "errors": [],
            "created_at": "2024-01-01T00:00:00",
            "started_at": None,
            "completed_at": None,
        }
        doc_svc = _make_doc_service()
        doc_svc.reprocess_document = AsyncMock(return_value=mock_job)
        request = MagicMock()
        logger = MagicMock()

        await reprocess_document(
            http_request=request,
            document_id="doc-rp",
            auth_service=auth_svc,
            authz_service=authz_svc,
            service=doc_svc,
            logger=logger,
        )

        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "documents", "write", workspace_id="ws-reprocess"
        )
        doc_svc.reprocess_document.assert_awaited_once()
        call_kwargs = doc_svc.reprocess_document.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-reprocess"


class TestPageImageAuthorization:
    """B4: page-image reads are authorized against the page's OWNING workspace.

    get_page(page_id) resolves a page purely by id with no rights check, so the
    endpoint must authorize documents:read against the page's owning workspace
    — NOT the caller's current ctx.workspace_id. Cross-workspace reads are fine
    when the user actually holds rights there; denied otherwise.
    """

    def _make_page(self, page_id="page-1", document_id="doc-1", workspace_id="ws-owner"):
        page = MagicMock()
        page.id = page_id
        page.document_id = document_id
        page.workspace_id = workspace_id
        page.image_storage_path = "blobs/page-1.png"
        return page

    @pytest.mark.asyncio
    async def test_authorizes_against_page_owning_workspace_not_ctx(self):
        """require_authorization is called with the page's workspace, not ctx's.

        Caller's ctx is in a DIFFERENT workspace than the page's owner; with
        rights granted, the read succeeds and authz used the OWNING workspace.
        """
        from memorylayer_saas.api.v1.documents import get_page_image

        ctx = _make_ctx("ws-caller-current")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()  # allows

        page = self._make_page(workspace_id="ws-owner-other")
        storage = MagicMock()
        storage.get_page = AsyncMock(return_value=page)
        blob = MagicMock()
        blob.retrieve_file = AsyncMock(return_value=b"\x89PNG")

        def ext_side_effect(ext_name, v=None):
            from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
            from memorylayer_saas.services.document import EXT_BLOB_STORAGE_SERVICE
            return {EXT_STORAGE_BACKEND: storage, EXT_BLOB_STORAGE_SERVICE: blob}[ext_name]

        with patch("memorylayer_saas.api.v1.documents.get_extension", side_effect=ext_side_effect):
            resp = await get_page_image(
                http_request=MagicMock(),
                document_id="doc-1",
                page_id="page-1",
                auth_service=auth_svc,
                authz_service=authz_svc,
                v=MagicMock(),
                logger=MagicMock(),
            )

        assert resp.body == b"\x89PNG"
        # Authorized against the OWNING workspace, NOT ctx.workspace_id.
        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "documents", "read", workspace_id="ws-owner-other"
        )

    @pytest.mark.asyncio
    async def test_denied_when_user_lacks_rights_to_owning_workspace(self):
        """User without rights to the page's owning workspace gets 403."""
        from memorylayer_saas.api.v1.documents import get_page_image

        ctx = _make_ctx("ws-caller-current")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service(deny=True)

        page = self._make_page(workspace_id="ws-owner-other")
        storage = MagicMock()
        storage.get_page = AsyncMock(return_value=page)
        blob = MagicMock()
        blob.retrieve_file = AsyncMock(return_value=b"\x89PNG")

        def ext_side_effect(ext_name, v=None):
            from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
            from memorylayer_saas.services.document import EXT_BLOB_STORAGE_SERVICE
            return {EXT_STORAGE_BACKEND: storage, EXT_BLOB_STORAGE_SERVICE: blob}[ext_name]

        with patch("memorylayer_saas.api.v1.documents.get_extension", side_effect=ext_side_effect):
            with pytest.raises(HTTPException) as exc_info:
                await get_page_image(
                    http_request=MagicMock(),
                    document_id="doc-1",
                    page_id="page-1",
                    auth_service=auth_svc,
                    authz_service=authz_svc,
                    v=MagicMock(),
                    logger=MagicMock(),
                )

        # 403 is collapsed to 404 — unauthorized is indistinguishable from nonexistent.
        assert exc_info.value.status_code == 404
        authz_svc.require_authorization.assert_awaited_once_with(
            ctx, "documents", "read", workspace_id="ws-owner-other"
        )
        # Blob never read when authz fails.
        blob.retrieve_file.assert_not_awaited()


class TestDocumentWorkspacePropagation:
    """Verify that ctx.workspace_id is passed to all service calls."""

    @pytest.mark.asyncio
    async def test_list_documents_uses_ctx_workspace_id(self):
        """list_documents must pass ctx.workspace_id, not a hardcoded value."""
        from memorylayer_saas.api.v1.documents import list_documents

        ctx = _make_ctx("ws-from-ctx-123")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        doc_svc = _make_doc_service()
        logger = MagicMock()

        await list_documents(
            http_request=MagicMock(),
            status_filter=None,
            # No explicit workspace: the context's workspace is what must reach
            # the service (see test_list_documents_calls_auth on why None is
            # passed rather than left to the Query default).
            workspace_id=None,
            limit=50,
            offset=0,
            auth_service=auth_svc,
            authz_service=authz_svc,
            service=doc_svc,
            logger=logger,
        )

        call_kwargs = doc_svc.list_documents.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-from-ctx-123"

    @pytest.mark.asyncio
    async def test_list_jobs_uses_ctx_workspace_id(self):
        """list_jobs must pass ctx.workspace_id."""
        from memorylayer_saas.api.v1.documents import list_jobs

        ctx = _make_ctx("ws-jobs-ctx")
        auth_svc = _make_auth_service(ctx)
        authz_svc = _make_authz_service()
        doc_svc = _make_doc_service()
        logger = MagicMock()

        await list_jobs(
            http_request=MagicMock(),
            status_filter=None,
            limit=50,
            auth_service=auth_svc,
            authz_service=authz_svc,
            service=doc_svc,
            logger=logger,
        )

        call_kwargs = doc_svc.list_jobs.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-jobs-ctx"
