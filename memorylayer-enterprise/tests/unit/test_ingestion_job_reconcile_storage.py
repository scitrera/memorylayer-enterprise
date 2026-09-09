# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Storage-level tests for the ingestion-job coalescing/reconcile primitives.

Exercises the two new storage methods against a real (OSS) SQLite backend, which
implements the same storage protocol the enterprise PostgreSQL backend does:

- ``list_active_jobs_for_documents`` -- the overlap query the create/finalize
  paths use to coalesce jobs (at most one in-flight job per document).
- ``cancel_orphaned_ingestion_jobs`` -- the periodic orphan-reconcile sweep that
  cancels queued/running jobs whose referenced documents are all completed.

The enterprise service targets PostgreSQL in production; PostgreSQL parity for
these methods is covered by the integration suite (needs a live database). These
tests pin the semantics against the SQLite implementation so both backends stay
in sync.
"""
import uuid

import pytest
import pytest_asyncio

from memorylayer_server.services.storage.sqlite import SQLiteStorageBackend
from memorylayer_server.models.document import (
    Document,
    DocumentStatus,
    DocumentType,
    IngestionJob,
    JobStatus,
)

WS = "ws_reconcile_test"


@pytest_asyncio.fixture
async def backend(tmp_path):
    """A connected, schema-initialized SQLite storage backend on a temp db."""
    be = SQLiteStorageBackend(str(tmp_path / "reconcile.db"))
    await be.connect()
    yield be
    await be.disconnect()


async def _make_doc(backend, doc_id: str, status: DocumentStatus) -> Document:
    doc = Document(
        id=doc_id,
        workspace_id=WS,
        filename=f"{doc_id}.pdf",
        document_type=DocumentType.PDF,
        content_hash=uuid.uuid4().hex,  # unique per doc (UNIQUE(ws, content_hash))
        size_bytes=10,
        status=status,
    )
    return await backend.create_document(WS, doc)


async def _make_job(backend, job_id: str, doc_ids: list[str], status: JobStatus) -> IngestionJob:
    job = IngestionJob(
        id=job_id,
        workspace_id=WS,
        document_ids=doc_ids,
        status=status,
    )
    return await backend.create_job(job)


class TestListActiveJobsForDocuments:
    """Coverage for the overlap lookup used to coalesce jobs."""

    @pytest.mark.asyncio
    async def test_returns_only_inflight_jobs_overlapping_the_documents(self, backend):
        await _make_doc(backend, "doc_a", DocumentStatus.PROCESSING)
        await _make_doc(backend, "doc_b", DocumentStatus.PROCESSING)

        running = await _make_job(backend, "job_running", ["doc_a"], JobStatus.RUNNING)
        await _make_job(backend, "job_queued_other", ["doc_b"], JobStatus.QUEUED)
        await _make_job(backend, "job_done", ["doc_a"], JobStatus.COMPLETED)
        await _make_job(backend, "job_cancelled", ["doc_a"], JobStatus.CANCELLED)

        active = await backend.list_active_jobs_for_documents(["doc_a"])

        active_ids = {j.id for j in active}
        assert active_ids == {"job_running"}, active_ids
        assert running.status == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_empty_document_ids_returns_empty(self, backend):
        await _make_job(backend, "job_x", ["doc_a"], JobStatus.RUNNING)
        assert await backend.list_active_jobs_for_documents([]) == []

    @pytest.mark.asyncio
    async def test_matches_when_document_is_one_of_a_batch(self, backend):
        await _make_job(backend, "job_batch", ["doc_a", "doc_b", "doc_c"], JobStatus.QUEUED)
        active = await backend.list_active_jobs_for_documents(["doc_c"])
        assert {j.id for j in active} == {"job_batch"}


class TestListJobsForDocuments:
    """Coverage for the unfiltered (any-status) form backing admin doc detail."""

    @pytest.mark.asyncio
    async def test_returns_every_status_not_just_inflight(self, backend):
        """The point of the diagnostic form: terminal attempts show up.

        ``list_active_jobs_for_documents`` deliberately hides these, so a
        regression that made this delegate with the default statuses would
        silently drop exactly the failed/superseded jobs an operator opened the
        panel to find.
        """
        await _make_doc(backend, "doc_a", DocumentStatus.PROCESSING)
        await _make_job(backend, "job_running", ["doc_a"], JobStatus.RUNNING)
        await _make_job(backend, "job_failed", ["doc_a"], JobStatus.FAILED)
        await _make_job(backend, "job_cancelled", ["doc_a"], JobStatus.CANCELLED)
        await _make_job(backend, "job_other_doc", ["doc_b"], JobStatus.FAILED)

        jobs = await backend.list_jobs_for_documents(["doc_a"])

        assert {j.id for j in jobs} == {"job_running", "job_failed", "job_cancelled"}

    @pytest.mark.asyncio
    async def test_status_filter_still_narrows(self, backend):
        await _make_job(backend, "job_running", ["doc_a"], JobStatus.RUNNING)
        await _make_job(backend, "job_failed", ["doc_a"], JobStatus.FAILED)

        jobs = await backend.list_jobs_for_documents(["doc_a"], statuses=("failed",))

        assert {j.id for j in jobs} == {"job_failed"}

    @pytest.mark.asyncio
    async def test_limit_counts_only_overlapping_jobs(self, backend):
        """A limit applied before the overlap filter would return nothing here.

        SQLite evaluates the overlap in Python, so a naive SQL LIMIT would be
        spent on the non-overlapping jobs and return fewer rows than asked for.
        """
        for i in range(5):
            await _make_job(backend, f"job_noise_{i}", ["doc_other"], JobStatus.COMPLETED)
        await _make_job(backend, "job_wanted", ["doc_a"], JobStatus.COMPLETED)

        jobs = await backend.list_jobs_for_documents(["doc_a"], limit=2)

        assert {j.id for j in jobs} == {"job_wanted"}

    @pytest.mark.asyncio
    async def test_empty_document_ids_returns_empty(self, backend):
        await _make_job(backend, "job_x", ["doc_a"], JobStatus.RUNNING)
        assert await backend.list_jobs_for_documents([]) == []


class TestCancelOrphanedIngestionJobs:
    """Coverage for the periodic orphan-reconcile sweep."""

    @pytest.mark.asyncio
    async def test_cancels_jobs_whose_docs_are_all_completed(self, backend):
        await _make_doc(backend, "doc_done", DocumentStatus.COMPLETED)
        await _make_doc(backend, "doc_busy", DocumentStatus.PROCESSING)

        # Orphan: sole doc is completed -> should be cancelled.
        await _make_job(backend, "job_orphan", ["doc_done"], JobStatus.RUNNING)
        # Still working: doc not completed -> untouched.
        await _make_job(backend, "job_active", ["doc_busy"], JobStatus.QUEUED)
        # Mixed batch: one doc not completed -> untouched.
        await _make_job(backend, "job_mixed", ["doc_done", "doc_busy"], JobStatus.RUNNING)
        # Missing doc: liveness undeterminable -> untouched.
        await _make_job(backend, "job_missing", ["doc_ghost"], JobStatus.RUNNING)
        # Already terminal -> not re-touched by the sweep.
        await _make_job(backend, "job_terminal", ["doc_done"], JobStatus.COMPLETED)

        cancelled = await backend.cancel_orphaned_ingestion_jobs()
        assert cancelled == 1

        assert (await backend.get_job("job_orphan")).status == JobStatus.CANCELLED
        assert (await backend.get_job("job_orphan")).completed_at is not None
        # Everything else keeps its prior status.
        assert (await backend.get_job("job_active")).status == JobStatus.QUEUED
        assert (await backend.get_job("job_mixed")).status == JobStatus.RUNNING
        assert (await backend.get_job("job_missing")).status == JobStatus.RUNNING
        assert (await backend.get_job("job_terminal")).status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_noop_when_nothing_orphaned(self, backend):
        await _make_doc(backend, "doc_busy", DocumentStatus.PROCESSING)
        await _make_job(backend, "job_active", ["doc_busy"], JobStatus.RUNNING)
        assert await backend.cancel_orphaned_ingestion_jobs() == 0
        assert (await backend.get_job("job_active")).status == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_empty_document_ids_job_is_left_untouched(self, backend):
        # A job with no document_ids has undeterminable liveness: an empty set
        # must NOT be treated as "all docs completed" and swept (parity with the
        # PostgreSQL `cardinality(document_ids) > 0` guard).
        await _make_job(backend, "job_empty", [], JobStatus.RUNNING)
        assert await backend.cancel_orphaned_ingestion_jobs() == 0
        assert (await backend.get_job("job_empty")).status == JobStatus.RUNNING
