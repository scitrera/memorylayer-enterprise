# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Two-phase document status: retrieval readiness vs knowledge extraction.

``DocumentStatus`` goes COMPLETED once pages, embeddings and composite memories
are durable. Fact decomposition fans out to thousands of background tasks and
can trail that by a long way, so it is tracked separately by
``DocumentEnrichmentStatus`` -- callers that only search or read pages are never
made to wait on it.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_saas.models.document import (
    Document,
    DocumentEnrichmentStatus,
    DocumentStatus,
    DocumentType,
)
from memorylayer_saas.services.document.gap_analysis import resolve_enrichment_status


def _doc(**overrides) -> Document:
    kwargs = dict(
        id="doc_1",
        workspace_id="ws_test",
        filename="a.pdf",
        document_type=DocumentType.PDF,
        content_hash="d" * 64,
        size_bytes=10,
        status=DocumentStatus.COMPLETED,
    )
    kwargs.update(overrides)
    return Document(**kwargs)


class TestResolveEnrichmentStatus:
    def test_nothing_scheduled_is_not_applicable(self):
        # Distinct from COMPLETE on purpose: "never ran" must stay legible
        # rather than masquerading as finished work.
        status, still = resolve_enrichment_status(_doc(), [])
        assert status == DocumentEnrichmentStatus.NOT_APPLICABLE
        assert still == []

    def test_all_scheduled_memories_produced_facts_is_complete(self):
        doc = _doc(enrichment_memory_ids=["mem_a", "mem_b"])
        status, still = resolve_enrichment_status(doc, [])
        assert status == DocumentEnrichmentStatus.COMPLETE
        assert still == []

    def test_outstanding_scheduled_memory_keeps_it_pending(self):
        doc = _doc(enrichment_memory_ids=["mem_a", "mem_b"])
        status, still = resolve_enrichment_status(doc, ["mem_b"])
        assert status == DocumentEnrichmentStatus.PENDING
        assert still == ["mem_b"]

    def test_unscheduled_factless_memory_does_not_block_completion(self):
        """The reason the scheduled set is recorded at all.

        A page whose content was already atomic is never handed to
        decomposition; it stays ACTIVE and factless forever, so fact-gap
        analysis reports it every sweep. Counting it as outstanding would leave
        such a document PENDING permanently.
        """
        doc = _doc(enrichment_memory_ids=["mem_a"])
        status, still = resolve_enrichment_status(doc, ["mem_a", "mem_atomic"])

        assert still == ["mem_a"]
        status, still = resolve_enrichment_status(doc, ["mem_atomic"])
        assert status == DocumentEnrichmentStatus.COMPLETE
        assert still == []

    def test_resolution_is_stable_when_called_repeatedly(self):
        # The sweep re-derives every pass; it must converge, not oscillate.
        doc = _doc(enrichment_memory_ids=["mem_a"])
        first = resolve_enrichment_status(doc, [])
        second = resolve_enrichment_status(doc, [])
        assert first == second == (DocumentEnrichmentStatus.COMPLETE, [])


class TestRecordEnrichmentPhase:
    """``_record_enrichment_phase`` runs on a path that re-executes on retry."""

    def _service(self):
        from memorylayer_saas.services.document.ingestion_service import (
            DocumentIngestionService,
        )

        storage = MagicMock()
        storage.update_document = AsyncMock()
        service = DocumentIngestionService.__new__(DocumentIngestionService)
        service._storage = storage
        service.logger = MagicMock()
        return service, storage

    @pytest.mark.asyncio
    async def test_scheduled_memories_are_recorded_as_pending(self):
        service, storage = self._service()
        doc = _doc()

        await service._record_enrichment_phase(doc, ["mem_a", "mem_b"])

        kwargs = storage.update_document.await_args.kwargs
        assert kwargs["enrichment_status"] == DocumentEnrichmentStatus.PENDING.value
        assert kwargs["enrichment_memory_ids"] == ["mem_a", "mem_b"]

    @pytest.mark.asyncio
    async def test_nothing_scheduled_records_not_applicable(self):
        service, storage = self._service()

        await service._record_enrichment_phase(_doc(), [])

        kwargs = storage.update_document.await_args.kwargs
        assert kwargs["enrichment_status"] == DocumentEnrichmentStatus.NOT_APPLICABLE.value

    @pytest.mark.asyncio
    async def test_a_gapfill_rerun_does_not_erase_outstanding_work(self):
        """The regression this union guards against.

        Re-running store on gap-fill skips pages that already have memories, so
        it schedules nothing. Overwriting would empty a still-outstanding set
        and report the document as enriched when it is not.
        """
        service, storage = self._service()
        doc = _doc(
            enrichment_memory_ids=["mem_a", "mem_b"],
            enrichment_status=DocumentEnrichmentStatus.PENDING,
        )

        await service._record_enrichment_phase(doc, [])

        kwargs = storage.update_document.await_args.kwargs
        assert kwargs["enrichment_memory_ids"] == ["mem_a", "mem_b"]
        assert kwargs["enrichment_status"] == DocumentEnrichmentStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_newly_scheduled_memories_merge_without_duplicates(self):
        service, storage = self._service()
        doc = _doc(enrichment_memory_ids=["mem_a"])

        await service._record_enrichment_phase(doc, ["mem_a", "mem_c"])

        assert storage.update_document.await_args.kwargs["enrichment_memory_ids"] == [
            "mem_a", "mem_c",
        ]

    @pytest.mark.asyncio
    async def test_bookkeeping_failure_does_not_propagate(self):
        """The memories are already stored; losing the phase label must not
        fail an otherwise-successful ingest. doc_verify re-derives it."""
        service, storage = self._service()
        storage.update_document.side_effect = RuntimeError("db down")

        await service._record_enrichment_phase(_doc(), ["mem_a"])

        assert service.logger.warning.called
