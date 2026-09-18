# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the doc_added task handler.

Tests the ``DocAddedTaskHandler`` in isolation with mocked dependencies. The
entry point now CLASSIFIES (vfs_ref -> content_hash) and either NO-OPs (complete
or in-flight), RESUMES gap-fill at the first missing phase, or runs FRESH. Fresh
ingests converge on the chained ``document_*`` pipeline: doc_added stores the
fetched blob and schedules ``document_render`` (no inline ``process_bytes``); the
chained ``document_finalize`` emits ``ingest_complete``.

Covered:
- In-flight NO-OP by fresh PROCESSING doc
- Complete NO-OP via gap analysis (+ re-emit ingest_complete)
- Resume gap-fill schedules the right chained task
- Fresh: create doc -> mint URL -> fetch -> store blob -> schedule document_render
- Terminal fetch failure: marks FAILED, emits ingest_failed, doesn't re-raise
- Retryable fetch failure: raises for Aether requeue
- Malformed payload: returns cleanly without raising
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memorylayer_saas.models.document import (
    Document,
    DocumentExtractionOptions,
    DocumentPage,
    DocumentStatus,
    DocumentType,
    IngestionJob,
    JobStatus,
)
from memorylayer_saas.tasks.doc_added import DocAddedTaskHandler, _vfs_access_request


def _make_existing_doc(*, status, page_count=0, processing_started_at=None,
                       source_vfs_ref="vfs_abc123", metadata=None):
    """Build a persisted Document domain model for the existing-doc paths."""
    return Document(
        id="doc_existing",
        workspace_id="ws_test",
        filename="report.pdf",
        document_type=DocumentType.PDF,
        content_hash="deadbeef" * 8,
        source_vfs_ref=source_vfs_ref,
        size_bytes=123,
        status=status,
        page_count=page_count,
        metadata=metadata or {},
        processing_started_at=processing_started_at,
    )


def _make_complete_page(page_id):
    """A page with transcript + multivector + single-vec embedding (column)."""
    return DocumentPage(
        id=page_id,
        document_id="doc_existing",
        workspace_id="ws_test",
        page_no=0,
        image_storage_path="/blobs/page_0.png",
        transcript="hello",
        embedding=[0.1, 0.2],
        multivector=[[1.0, 2.0]],
        metadata={},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DOC_ADDED_PAYLOAD = {
    "workspace_id": "ws_test",
    "vfs_ref": "vfs_abc123",
    "content_hash": "deadbeef" * 8,
    "connector_id": "manual_upload",
    "filename_hint": "report.pdf",
}


@pytest.fixture()
def mock_variables():
    v = MagicMock()
    v.environ = MagicMock(side_effect=lambda key, default, **kw: default)
    return v


def _make_mock_storage():
    storage = AsyncMock()
    storage.find_document_by_vfs_ref = AsyncMock(return_value=None)
    storage.find_document_by_hash = AsyncMock(return_value=None)
    storage.create_document = AsyncMock(side_effect=lambda doc: doc)
    storage.create_job = AsyncMock(side_effect=lambda job: job)
    storage.update_document = AsyncMock()
    # CAS claim: default to winning the race (returns True) so resume paths
    # proceed; race-loss tests override this to return False.
    storage.try_claim_document = AsyncMock(return_value=True)
    storage.update_job = AsyncMock()
    # Gap analysis (used on the existing-doc path) reads these.
    storage.get_pages = AsyncMock(return_value=[])
    storage.get_memory_source_page_ids = AsyncMock(return_value=set())
    return storage


def _make_mock_task_service():
    ts = AsyncMock()
    ts.schedule_task = AsyncMock(return_value=None)
    return ts


def _make_mock_ingestion_service():
    svc = MagicMock()
    svc.process_bytes = AsyncMock()
    svc.store_upload_blob = AsyncMock(side_effect=lambda doc, file_data: doc)
    return svc


def _make_mock_agent_service():
    agent_svc = MagicMock()
    agent_svc.client = AsyncMock()
    agent_svc.client.send_event = AsyncMock()
    agent_svc.client.report_progress = AsyncMock()
    # update_task is not yet exposed by the SDK; emulate that by removing it so
    # the emitter's hasattr/getattr guard exercises the report_progress-only path.
    if hasattr(agent_svc.client, "update_task"):
        del agent_svc.client.update_task
    return agent_svc


# The Aether task_id is the snapshot correlation key; the dctask id lives in
# task metadata["task_id"] and must NOT be used as the progress task_id.
AETHER_TASK_ID = "atask_aether123"
DCTASK_ID = "dctask_deadbeef"

# Phase 1 (data-connectors) stamps these onto the task metadata string map;
# the worker runner threads task_id + metadata into the payload under reserved
# keys before calling the handler.
TASK_METADATA_WORKSPACE = {
    "task_id": DCTASK_ID,
    "connector_id": "manual_upload",
    "title": "report.pdf",
    "bg_kind": "ingest",
    "visibility": "workspace",
    "task_class": "background",
}

TASK_METADATA_PRIVATE = {
    "task_id": DCTASK_ID,
    "connector_id": "manual_upload",
    "title": "report.pdf",
    "bg_kind": "ingest",
    "visibility": "private",
    "task_class": "batch",
    "initiated_by": "us::alice",
}


def _payload_with_progress(task_metadata):
    """Build a doc_added payload with the runner-injected progress keys."""
    return {
        **DOC_ADDED_PAYLOAD,
        "_aether_task_id": AETHER_TASK_ID,
        "_task_metadata": task_metadata,
    }


def _make_proxy_response(status_code=200, body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.body = body or json.dumps({
        "url": "https://blob.example.com/report.pdf?token=abc",
        "headers": {"Authorization": "Bearer tok"},
        "expires_at": "2026-05-08T12:00:00Z",
    }).encode()
    return resp


@contextmanager
def patch_doc_added(storage=None, ingestion_service=None, agent_service=None,
                    task_service=None, proxy_response=None,
                    http_content=b"fake-pdf-bytes"):
    """Patch all dependencies for DocAddedTaskHandler."""
    storage = storage or _make_mock_storage()
    ingestion_service = ingestion_service or _make_mock_ingestion_service()
    agent_service = agent_service or _make_mock_agent_service()
    task_service = task_service or _make_mock_task_service()
    proxy_resp = proxy_response or _make_proxy_response()

    from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
    from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION
    from memorylayer_server.services.tasks.base import EXT_TASK_SERVICE
    from memorylayer_saas.services.document import EXT_DOCUMENT_INGESTION_SERVICE

    def ext_side_effect(ext_name, v=None):
        return {
            EXT_STORAGE_BACKEND: storage,
            EXT_AETHER_SERVICE_CONNECTION: agent_service,
            EXT_TASK_SERVICE: task_service,
            EXT_DOCUMENT_INGESTION_SERVICE: ingestion_service,
        }.get(ext_name)

    mock_http_response = AsyncMock()
    mock_http_response.status_code = 200
    mock_http_response.content = http_content
    mock_http_response.raise_for_status = MagicMock()

    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_http_response)
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    with patch("memorylayer_saas.tasks.doc_added.get_extension", side_effect=ext_side_effect), \
         patch("memorylayer_saas.tasks.doc_added.get_document_ingestion_service",
               return_value=ingestion_service), \
         patch("memorylayer_saas.tasks.doc_added.get_logger", return_value=MagicMock()), \
         patch("memorylayer_saas.tasks.doc_added.proxy_http_async",
               new_callable=AsyncMock, return_value=proxy_resp) as mock_proxy, \
         patch("memorylayer_saas.tasks.doc_added.httpx") as mock_httpx:
        # Wire the httpx mock so AsyncClient() returns our mock context manager
        mock_httpx.AsyncClient.return_value = mock_http_client
        yield storage, ingestion_service, agent_service, mock_proxy, mock_http_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDocAddedHandler:
    """Tests for DocAddedTaskHandler."""

    def test_get_task_type(self):
        handler = DocAddedTaskHandler()
        assert handler.get_task_type() == "doc_added"

    def test_get_schedule_is_none(self):
        handler = DocAddedTaskHandler()
        assert handler.get_schedule(MagicMock()) is None

    @pytest.mark.asyncio
    async def test_inflight_doc_noop(self, mock_variables):
        """A fresh in-flight (PROCESSING) doc on this vfs_ref is a NO-OP."""
        storage = _make_mock_storage()
        existing = _make_existing_doc(
            status=DocumentStatus.PROCESSING,
            processing_started_at=datetime.now(timezone.utc),
        )
        storage.find_document_by_vfs_ref.return_value = existing

        with patch_doc_added(storage=storage) as (storage, ingestion_svc, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, DOC_ADDED_PAYLOAD)

        storage.create_document.assert_not_called()
        ingestion_svc.process_bytes.assert_not_called()
        ingestion_svc.store_upload_blob.assert_not_called()

    @pytest.mark.asyncio
    async def test_complete_doc_noop_reemits_complete(self, mock_variables):
        """An already-complete existing doc is a NO-OP and re-emits ingest_complete."""
        storage = _make_mock_storage()
        existing = _make_existing_doc(status=DocumentStatus.COMPLETED, page_count=1)
        storage.find_document_by_hash.return_value = existing
        # One rendered page with all artifacts + a memory -> gaps.is_complete.
        page = _make_complete_page("page_a")
        storage.get_pages.return_value = [page]
        storage.get_memory_source_page_ids.return_value = {"page_a"}

        agent_svc = _make_mock_agent_service()
        with patch_doc_added(storage=storage, agent_service=agent_svc) as (storage, ingestion_svc, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, DOC_ADDED_PAYLOAD)

        storage.create_document.assert_not_called()
        ingestion_svc.store_upload_blob.assert_not_called()
        # Re-emits ingest_complete for KB.
        assert agent_svc.client.send_event.called

    @pytest.mark.asyncio
    async def test_resume_schedules_finalize_for_store_gap(self, mock_variables):
        """An existing doc missing only memories resumes at document_finalize."""
        storage = _make_mock_storage()
        existing = _make_existing_doc(status=DocumentStatus.FAILED, page_count=1)
        storage.find_document_by_hash.return_value = existing
        # Page is fully embedded/transcribed but has NO memory -> store gap.
        page = _make_complete_page("page_a")
        storage.get_pages.return_value = [page]
        storage.get_memory_source_page_ids.return_value = set()

        ts = _make_mock_task_service()
        with patch_doc_added(storage=storage, task_service=ts) as (storage, ingestion_svc, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, DOC_ADDED_PAYLOAD)

        # No new doc; claimed via CAS; scheduled finalize.
        storage.create_document.assert_not_called()
        ts.schedule_task.assert_called_once()
        assert ts.schedule_task.call_args.args[0] == "document_finalize"
        # Claim is now an atomic CAS (try_claim_document), not a read-then-write
        # update_document(status=PROCESSING).
        storage.try_claim_document.assert_awaited_once()
        assert storage.try_claim_document.await_args.args[0] == "doc_existing"

    @pytest.mark.asyncio
    async def test_resume_schedules_render_for_render_gap(self, mock_variables):
        """An existing doc with no pages resumes at document_render."""
        storage = _make_mock_storage()
        existing = _make_existing_doc(status=DocumentStatus.FAILED, page_count=0)
        storage.find_document_by_hash.return_value = existing
        storage.get_pages.return_value = []
        storage.get_memory_source_page_ids.return_value = set()

        ts = _make_mock_task_service()
        with patch_doc_added(storage=storage, task_service=ts) as (storage, _, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, DOC_ADDED_PAYLOAD)

        ts.schedule_task.assert_called_once()
        assert ts.schedule_task.call_args.args[0] == "document_render"

    @pytest.mark.asyncio
    async def test_lost_claim_race_noops(self, mock_variables):
        """A doc with gaps whose CAS claim is lost (another worker won) NO-OPs:
        no job created and no chained phase scheduled."""
        storage = _make_mock_storage()
        existing = _make_existing_doc(status=DocumentStatus.FAILED, page_count=1)
        storage.find_document_by_hash.return_value = existing
        # Store gap (page with no memory) -> would resume, but the CAS loses.
        page = _make_complete_page("page_a")
        storage.get_pages.return_value = [page]
        storage.get_memory_source_page_ids.return_value = set()
        storage.try_claim_document = AsyncMock(return_value=False)

        ts = _make_mock_task_service()
        with patch_doc_added(storage=storage, task_service=ts) as (storage, _, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, DOC_ADDED_PAYLOAD)

        storage.try_claim_document.assert_awaited_once()
        # Lost the race: no job, no chained task scheduled.
        storage.create_job.assert_not_called()
        ts.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_happy_path_fresh_stores_blob_schedules_render(self, mock_variables):
        """Fresh flow: classify passes -> create doc -> mint URL -> fetch ->
        store blob -> schedule document_render (no inline process_bytes)."""
        ts = _make_mock_task_service()
        with patch_doc_added(task_service=ts) as (storage, ingestion_svc, agent_svc, mock_proxy, mock_http):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, DOC_ADDED_PAYLOAD)

        # Document created
        storage.create_document.assert_called_once()
        created_doc = storage.create_document.call_args.args[0]
        assert created_doc.source_vfs_ref == "vfs_abc123"
        assert created_doc.status == DocumentStatus.PENDING_FETCH
        assert created_doc.filename == "report.pdf"
        assert created_doc.document_type == DocumentType.PDF

        # Job created
        storage.create_job.assert_called_once()

        # proxy_http_async called three times: read the entry's requested ingest
        # flags, mint the fetch URL, then link the ML document. Asserted by PATH
        # rather than index so adding a call does not silently shift what a
        # positional assertion is actually checking.
        assert mock_proxy.call_count == 3
        paths = [c.kwargs["path"] for c in mock_proxy.call_args_list]
        assert "/v1/urls/fetch" in paths
        assert any(p.startswith("/v1/vfs/entries/") and p.endswith("/link") for p in paths)
        assert any(
            p.startswith("/v1/vfs/entries/") and not p.endswith("/link") for p in paths
        ), "expected a plain entry GET for the upload-requested ingest flags"

        for call in mock_proxy.call_args_list:
            checked = call.kwargs["checked_access"]
            assert call.kwargs["app_workspace"] == "ws_test"
            assert checked.resource_type == "vfs"
            assert checked.resource_id == "workspaces/ws_test/entries/vfs_abc123"
            assert checked.workspace == "ws_test"
            if call.kwargs["path"].endswith("/link"):
                assert checked.operation == "write"
                assert checked.required_access_level == 20
            else:
                assert checked.operation == "read"
                assert checked.required_access_level == 10

        # HTTP GET for bytes
        mock_http.get.assert_called_once()

        # Converged path: blob stored, render scheduled, NO inline process_bytes.
        ingestion_svc.process_bytes.assert_not_called()
        ingestion_svc.store_upload_blob.assert_called_once()
        assert ingestion_svc.store_upload_blob.call_args.args[1] == b"fake-pdf-bytes"
        ts.schedule_task.assert_called_once()
        assert ts.schedule_task.call_args.args[0] == "document_render"

    @pytest.mark.asyncio
    async def test_creates_doc_with_pending_fetch_status(self, mock_variables):
        """Document is created with PENDING_FETCH status and source_vfs_ref."""
        with patch_doc_added() as (storage, _, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, DOC_ADDED_PAYLOAD)

        doc = storage.create_document.call_args.args[0]
        assert doc.status == DocumentStatus.PENDING_FETCH
        assert doc.source_vfs_ref == "vfs_abc123"
        assert doc.content_hash == "deadbeef" * 8

    @pytest.mark.asyncio
    async def test_terminal_fetch_failure_marks_failed_no_reraise(self, mock_variables):
        """Terminal fetch errors mark FAILED and emit ingest_failed, don't re-raise."""
        proxy_resp = _make_proxy_response(status_code=404, body=b"not found")

        with patch_doc_added(proxy_response=proxy_resp) as (storage, ingestion_svc, _, _, _):
            handler = DocAddedTaskHandler()
            # Should NOT raise
            await handler.handle(mock_variables, DOC_ADDED_PAYLOAD)

        # Document marked FAILED
        failed_calls = [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.FAILED.value
        ]
        assert len(failed_calls) == 1

        # Pipeline NOT invoked
        ingestion_svc.process_bytes.assert_not_called()

    @pytest.mark.asyncio
    async def test_retryable_fetch_failure_raises(self, mock_variables):
        """ConnectionError during fetch raises for Aether requeue."""
        with patch_doc_added() as (_, _, _, mock_proxy, _):
            mock_proxy.side_effect = ConnectionError("network blip")
            handler = DocAddedTaskHandler()

            with pytest.raises(ConnectionError):
                await handler.handle(mock_variables, DOC_ADDED_PAYLOAD)

    @pytest.mark.asyncio
    async def test_malformed_payload_returns_cleanly(self, mock_variables):
        """Missing required fields returns without raising."""
        with patch_doc_added() as (storage, ingestion_svc, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, {"some": "junk"})

        storage.create_document.assert_not_called()
        ingestion_svc.process_bytes.assert_not_called()

    @pytest.mark.asyncio
    async def test_fresh_does_not_emit_complete_inline(self, mock_variables):
        """On the FRESH (chained) path doc_added does NOT emit ingest_complete;
        the chained document_finalize emits it once memories are created."""
        agent_svc = _make_mock_agent_service()

        with patch_doc_added(agent_service=agent_svc) as (_, _, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, DOC_ADDED_PAYLOAD)

        # No ingest_complete (or ingest_failed) event on the fresh happy path.
        assert not agent_svc.client.send_event.called

    @pytest.mark.asyncio
    async def test_vfs_link_called_on_fresh(self, mock_variables):
        """Fresh ingest links the VFS entry back to the ML doc after blob store."""
        with patch_doc_added() as (_, _, _, mock_proxy, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, DOC_ADDED_PAYLOAD)

        # Entry-flags GET, URL mint, VFS link. Matched by path, not position.
        assert mock_proxy.call_count == 3
        link_calls = [
            c for c in mock_proxy.call_args_list if c.kwargs["path"].endswith("/link")
        ]
        assert len(link_calls) == 1

    @pytest.mark.asyncio
    async def test_retryable_blob_store_failure_raises(self, mock_variables):
        """ConnectionError during blob store raises for Aether requeue."""
        ingestion_svc = _make_mock_ingestion_service()
        ingestion_svc.store_upload_blob.side_effect = ConnectionError("blob down")

        with patch_doc_added(ingestion_service=ingestion_svc) as (_, _, _, _, _):
            handler = DocAddedTaskHandler()

            with pytest.raises(ConnectionError):
                await handler.handle(mock_variables, DOC_ADDED_PAYLOAD)

    @pytest.mark.asyncio
    async def test_terminal_blob_store_failure_marks_failed(self, mock_variables):
        """Non-retryable blob-store errors mark FAILED, emit event, don't re-raise."""
        ingestion_svc = _make_mock_ingestion_service()
        ingestion_svc.store_upload_blob.side_effect = ValueError("corrupt PDF")

        with patch_doc_added(ingestion_service=ingestion_svc) as (storage, _, _, _, _):
            handler = DocAddedTaskHandler()
            # Should NOT raise
            await handler.handle(mock_variables, DOC_ADDED_PAYLOAD)

        ingestion_svc.store_upload_blob.assert_called_once()
        failed_calls = [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.FAILED.value
        ]
        assert len(failed_calls) == 1

    @pytest.mark.asyncio
    async def test_filename_hint_detects_document_type(self, mock_variables):
        """filename_hint drives document_type detection."""
        payload = {**DOC_ADDED_PAYLOAD, "filename_hint": "slides.pptx"}

        with patch_doc_added() as (storage, _, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, payload)

        doc = storage.create_document.call_args.args[0]
        assert doc.document_type == DocumentType.PPTX

    @pytest.mark.asyncio
    async def test_unknown_extension_defaults_to_text(self, mock_variables):
        """Unknown file extensions default to TEXT document type."""
        payload = {**DOC_ADDED_PAYLOAD, "filename_hint": "data.xyz"}

        with patch_doc_added() as (storage, _, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, payload)

        doc = storage.create_document.call_args.args[0]
        assert doc.document_type == DocumentType.TEXT


# ---------------------------------------------------------------------------
# Test: dispatch registration compatibility
# ---------------------------------------------------------------------------

class TestDocAddedRegistration:
    """Verify DocAddedTaskHandler is compatible with the TaskHandlerPlugin interface."""

    def test_is_task_handler_plugin(self):
        from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
        handler = DocAddedTaskHandler()
        assert isinstance(handler, TaskHandlerPlugin)

    def test_task_type_matches_dispatch_convention(self):
        """Task type 'doc_added' maps to aether type 'memorylayer-task.doc_added'."""
        handler = DocAddedTaskHandler()
        assert handler.get_task_type() == "doc_added"

    def test_dc_topic_default_has_no_specifier(self):
        """The default topic must stay implementation-only (no ``::specifier``).

        data-connectors self-registers as ``sv::data-connectors:{hostname}``, so
        pinning the default to a specifier such as ``::default`` would never
        match a live replica and every fetch would fail to route.
        """
        from memorylayer_saas.config import DEFAULT_MEMORYLAYER_DATA_CONNECTORS_TOPIC

        assert DEFAULT_MEMORYLAYER_DATA_CONNECTORS_TOPIC == "sv::data-connectors"

    def test_vfs_access_resource_is_canonical_and_exact(self):
        checked = _vfs_access_request(
            "project one/+",
            "vfs/ref +one",
            operation="read",
            required_access_level=10,
        )

        assert checked.resource_id == (
            "workspaces/project%20one%2F%2B/entries/vfs%2Fref%20%2Bone"
        )
        assert checked.workspace == "project one/+"


# ---------------------------------------------------------------------------
# Phase 2: Background Tasks progress emission
# ---------------------------------------------------------------------------

class TestDocAddedProgress:
    """Tests for live progress emission (Background Tasks UI)."""

    @pytest.mark.asyncio
    async def test_progress_uses_aether_task_id_not_dctask(self, mock_variables):
        """report_progress must key on the Aether task_id, NOT metadata['task_id']."""
        agent_svc = _make_mock_agent_service()

        with patch_doc_added(agent_service=agent_svc) as (_, _, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, _payload_with_progress(TASK_METADATA_WORKSPACE))

        assert agent_svc.client.report_progress.called
        for call in agent_svc.client.report_progress.call_args_list:
            task_id_arg = call.args[0] if call.args else call.kwargs.get("task_id")
            assert task_id_arg == AETHER_TASK_ID
            assert task_id_arg != DCTASK_ID

    @pytest.mark.asyncio
    async def test_progress_emitted_with_kind_app(self, mock_variables):
        """Every progress report is tagged kind=APP (2)."""
        agent_svc = _make_mock_agent_service()

        with patch_doc_added(agent_service=agent_svc) as (_, _, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, _payload_with_progress(TASK_METADATA_WORKSPACE))

        assert agent_svc.client.report_progress.called
        for call in agent_svc.client.report_progress.call_args_list:
            assert call.kwargs["kind"] == 2

    @pytest.mark.asyncio
    async def test_progress_metadata_carries_title_kind_class(self, mock_variables):
        """Progress metadata carries title, bg_kind and task_class for filter/label."""
        agent_svc = _make_mock_agent_service()

        with patch_doc_added(agent_service=agent_svc) as (_, _, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, _payload_with_progress(TASK_METADATA_WORKSPACE))

        first = agent_svc.client.report_progress.call_args_list[0]
        md = first.kwargs["metadata"]
        assert md["title"] == "report.pdf"
        assert md["bg_kind"] == "ingest"
        assert md["task_class"] == "background"

    @pytest.mark.asyncio
    async def test_workspace_visibility_broadcasts(self, mock_variables):
        """visibility=workspace -> recipient='' (broadcast to workspace plane)."""
        agent_svc = _make_mock_agent_service()

        with patch_doc_added(agent_service=agent_svc) as (_, _, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, _payload_with_progress(TASK_METADATA_WORKSPACE))

        for call in agent_svc.client.report_progress.call_args_list:
            assert call.kwargs["recipient"] == ""

    @pytest.mark.asyncio
    async def test_private_visibility_targets_initiator(self, mock_variables):
        """visibility=private + initiated_by -> recipient=initiated_by."""
        agent_svc = _make_mock_agent_service()

        with patch_doc_added(agent_service=agent_svc) as (_, _, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, _payload_with_progress(TASK_METADATA_PRIVATE))

        assert agent_svc.client.report_progress.called
        for call in agent_svc.client.report_progress.call_args_list:
            assert call.kwargs["recipient"] == "us::alice"

    @pytest.mark.asyncio
    async def test_fresh_handoff_emits_running_queued(self, mock_variables):
        """Fresh ingest hands off to the chained pipeline; its last progress is a
        running 'ingest' (queued) milestone — terminal 'completed' now comes from
        the chained finalize, not doc_added."""
        agent_svc = _make_mock_agent_service()

        with patch_doc_added(agent_service=agent_svc) as (_, _, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, _payload_with_progress(TASK_METADATA_WORKSPACE))

        states = [c.kwargs["state"] for c in agent_svc.client.report_progress.call_args_list]
        assert "failed" not in states
        last = agent_svc.client.report_progress.call_args_list[-1]
        assert last.kwargs["state"] == "running"
        assert last.kwargs["step_name"] == "ingest"

    @pytest.mark.asyncio
    async def test_terminal_blob_failure_emits_failed(self, mock_variables):
        """A terminal blob-store error emits state=failed with the error summary."""
        agent_svc = _make_mock_agent_service()
        ingestion_svc = _make_mock_ingestion_service()
        ingestion_svc.store_upload_blob.side_effect = ValueError("corrupt PDF")

        with patch_doc_added(agent_service=agent_svc, ingestion_service=ingestion_svc) as (_, _, _, _, _):
            handler = DocAddedTaskHandler()
            await handler.handle(mock_variables, _payload_with_progress(TASK_METADATA_WORKSPACE))

        failed = [
            c for c in agent_svc.client.report_progress.call_args_list
            if c.kwargs["state"] == "failed"
        ]
        assert len(failed) == 1
        assert "corrupt PDF" in failed[0].kwargs["summary"]

    @pytest.mark.asyncio
    async def test_retryable_failure_emits_running_not_failed(self, mock_variables):
        """A retryable failure emits a running (retrying) state, never terminal 'failed'."""
        agent_svc = _make_mock_agent_service()
        ingestion_svc = _make_mock_ingestion_service()
        ingestion_svc.store_upload_blob.side_effect = ConnectionError("blob down")

        with patch_doc_added(agent_service=agent_svc, ingestion_service=ingestion_svc) as (_, _, _, _, _):
            handler = DocAddedTaskHandler()
            with pytest.raises(ConnectionError):
                await handler.handle(mock_variables, _payload_with_progress(TASK_METADATA_WORKSPACE))

        states = [c.kwargs["state"] for c in agent_svc.client.report_progress.call_args_list]
        assert "failed" not in states
        assert states[-1] == "running"

    @pytest.mark.asyncio
    async def test_progress_failure_does_not_break_handler(self, mock_variables):
        """A report_progress exception is swallowed; ingestion still hands off."""
        agent_svc = _make_mock_agent_service()
        agent_svc.client.report_progress.side_effect = RuntimeError("progress plane down")

        with patch_doc_added(agent_service=agent_svc) as (storage, ingestion_svc, _, _, _):
            handler = DocAddedTaskHandler()
            # Must NOT raise despite every report_progress failing.
            await handler.handle(mock_variables, _payload_with_progress(TASK_METADATA_WORKSPACE))

        # Blob stored + doc created despite progress failures.
        ingestion_svc.store_upload_blob.assert_called_once()
        storage.create_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_aether_task_id_skips_progress(self, mock_variables):
        """Without the runner-injected Aether task_id, no progress is emitted."""
        agent_svc = _make_mock_agent_service()

        with patch_doc_added(agent_service=agent_svc) as (_, ingestion_svc, _, _, _):
            handler = DocAddedTaskHandler()
            # Legacy/scheduler payload: no _aether_task_id / _task_metadata.
            await handler.handle(mock_variables, DOC_ADDED_PAYLOAD)

        agent_svc.client.report_progress.assert_not_called()
        # Ingestion is unaffected.
        ingestion_svc.store_upload_blob.assert_called_once()


@pytest.mark.asyncio
async def test_identical_reupload_links_new_vfs_without_reingesting(mock_variables):
    storage = _make_mock_storage()
    existing = _make_existing_doc(status=DocumentStatus.COMPLETED, page_count=1, source_vfs_ref="vfs_original")
    storage.find_document_by_hash.return_value = existing
    storage.get_pages.return_value = [_make_complete_page("page_a")]
    storage.get_memory_source_page_ids.return_value = {"page_a"}
    task_service = _make_mock_task_service()
    with patch_doc_added(storage=storage, task_service=task_service) as (_, _, agent, proxy, _):
        await DocAddedTaskHandler().handle(mock_variables, DOC_ADDED_PAYLOAD)
    links = [c for c in proxy.call_args_list if c.kwargs["path"].endswith("/link")]
    assert len(links) == 1
    assert links[0].kwargs["path"] == "/v1/vfs/entries/vfs_abc123/link"
    assert json.loads(links[0].kwargs["body"]) == {"ml_doc_id": existing.id}
    assert links[0].kwargs["app_workspace"] == "ws_test"
    assert existing.source_vfs_ref == "vfs_original"
    task_service.schedule_task.assert_not_called()
    event = json.loads(agent.client.send_event.call_args.args[0])
    assert event["data"]["vfs_ref"] == "vfs_abc123"


@pytest.mark.asyncio
async def test_existing_document_link_failure_is_retryable(mock_variables):
    storage = _make_mock_storage()
    storage.find_document_by_hash.return_value = _make_existing_doc(status=DocumentStatus.COMPLETED)
    with patch_doc_added(storage=storage, proxy_response=_make_proxy_response(503)) as (_, _, agent, _, _):
        with pytest.raises(ConnectionError, match="link is unconfirmed"):
            await DocAddedTaskHandler().handle(mock_variables, DOC_ADDED_PAYLOAD)
    agent.client.send_event.assert_not_called()
