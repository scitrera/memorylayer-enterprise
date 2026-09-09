"""Unit tests for the doc_verify reconcile task handler (Phase 3).

Tests ``DocVerifyTaskHandler`` in isolation with lightweight fakes. The handler
reuses the Phase-1 primitives (``analyze_document_gaps`` + ``resume_at``) plus
``analyze_fact_gaps`` to heal documents left incomplete by crashed workers.

Covered:
- On-demand: heals a doc with a store gap (claims PROCESSING + schedules finalize)
- On-demand: no-op on a complete + fact-complete doc
- On-demand: missing document => NO-OP
- On-demand: in-flight (fresh PROCESSING) doc is left untouched
- Sweep: selects PARTIAL / FAILED / stale-PROCESSING, skips fresh-PROCESSING
- Sweep: global payload iterates all workspaces
- Fact-gap: a page composite with no derived fact re-schedules decompose_facts
- Fact-gap: an archived composite (already decomposed) is NOT re-scheduled
- get_schedule honors the enabled flag
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memorylayer_saas.models.document import (
    Document,
    DocumentPage,
    DocumentStatus,
    DocumentType,
)
from memorylayer_saas.tasks.doc_verify import DocVerifyTaskHandler


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

def _make_doc(*, doc_id="doc1", status, page_count=0, processing_started_at=None,
              workspace_id="ws_test", metadata=None):
    return Document(
        id=doc_id,
        workspace_id=workspace_id,
        filename="report.pdf",
        document_type=DocumentType.PDF,
        content_hash="h",
        size_bytes=1,
        status=status,
        page_count=page_count,
        metadata=metadata or {},
        processing_started_at=processing_started_at,
    )


def _complete_page(page_id="page_a"):
    return DocumentPage(
        id=page_id,
        document_id="doc1",
        workspace_id="ws_test",
        page_no=0,
        image_storage_path="/blobs/page_0.png",
        transcript="hello",
        embedding=[0.1, 0.2],
        multivector=[[1.0, 2.0]],
        metadata={},
    )


def _composite_mem(mem_id, *, source_page_id, subtype=None, status="active"):
    """Lightweight stand-in for a Memory (gap analysis reads only these attrs)."""
    return SimpleNamespace(
        id=mem_id,
        source_page_id=source_page_id,
        subtype=subtype,
        status=status,
    )


def _make_v():
    v = MagicMock()
    v.environ = MagicMock(side_effect=lambda key, default=None, **kw: default)
    return v


def _make_storage(*, pages=None, mem_page_ids=None, document_memories=None,
                  fact_parent_ids=None, list_documents_map=None,
                  workspace_ids=None, get_document=None, claim=True):
    storage = AsyncMock()
    storage.get_pages = AsyncMock(return_value=list(pages or []))
    storage.get_memory_source_page_ids = AsyncMock(return_value=set(mem_page_ids or set()))
    storage.get_document_memories = AsyncMock(return_value=list(document_memories or []))
    storage.get_fact_memory_parent_ids = AsyncMock(return_value=set(fact_parent_ids or set()))
    storage.update_document = AsyncMock()
    # CAS claim: default to winning the race; race-loss tests pass claim=False.
    storage.try_claim_document = AsyncMock(return_value=claim)
    storage.create_job = AsyncMock()
    storage.list_all_workspace_ids = AsyncMock(return_value=list(workspace_ids or []))

    # list_documents returns (docs, total) keyed by status value.
    _ld_map = list_documents_map or {}

    async def _list_documents(workspace_id, status=None, limit=50, **kw):
        docs = _ld_map.get(status, [])
        return list(docs), len(docs)

    storage.list_documents = AsyncMock(side_effect=_list_documents)

    if get_document is not None:
        storage.get_document = AsyncMock(side_effect=get_document)
    return storage


def _make_task_service():
    ts = AsyncMock()
    ts.schedule_task = AsyncMock(return_value=None)
    return ts


def _patch_exts(storage, task_service):
    from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
    from memorylayer_server.services.tasks.base import EXT_TASK_SERVICE

    def ext_side_effect(ext_name, v=None):
        return {
            EXT_STORAGE_BACKEND: storage,
            EXT_TASK_SERVICE: task_service,
        }.get(ext_name)

    return patch.multiple(
        "memorylayer_saas.tasks.doc_verify",
        get_extension=MagicMock(side_effect=ext_side_effect),
        get_logger=MagicMock(return_value=MagicMock()),
    )


# ---------------------------------------------------------------------------
# get_task_type / get_schedule
# ---------------------------------------------------------------------------

class TestDocVerifyRegistration:
    def test_task_type(self):
        assert DocVerifyTaskHandler().get_task_type() == "doc_verify"

    def test_is_task_handler_plugin(self):
        from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
        assert isinstance(DocVerifyTaskHandler(), TaskHandlerPlugin)

    def test_schedule_enabled_by_default(self):
        sched = DocVerifyTaskHandler().get_schedule(_make_v())
        assert sched is not None
        assert sched.interval_seconds == 21600
        assert sched.default_payload == {}

    def test_schedule_none_when_disabled(self):
        v = MagicMock()
        v.environ = MagicMock(
            side_effect=lambda key, default=None, **kw: False if "ENABLED" in key else default
        )
        assert DocVerifyTaskHandler().get_schedule(v) is None


# ---------------------------------------------------------------------------
# On-demand mode
# ---------------------------------------------------------------------------

class TestDocVerifyOnDemand:
    @pytest.mark.asyncio
    async def test_heals_store_gap_schedules_finalize(self):
        """A doc whose page lacks a memory is claimed PROCESSING + resumed at finalize."""
        doc = _make_doc(status=DocumentStatus.FAILED, page_count=1)
        page = _complete_page("page_a")
        storage = _make_storage(
            pages=[page], mem_page_ids=set(),  # store gap: no memory for page_a
            get_document=lambda doc_id, ws: doc,
        )
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(
                _make_v(), {"workspace_id": "ws_test", "document_id": "doc1"},
            )

        # Claimed via atomic CAS and scheduled finalize.
        storage.try_claim_document.assert_awaited_once()
        assert storage.try_claim_document.await_args.args[0] == "doc1"
        ts.schedule_task.assert_called_once()
        assert ts.schedule_task.call_args.args[0] == "document_finalize"

    @pytest.mark.asyncio
    async def test_complete_doc_is_noop(self):
        """A complete + fact-complete doc neither claims nor schedules anything."""
        doc = _make_doc(status=DocumentStatus.COMPLETED, page_count=1)
        page = _complete_page("page_a")
        # Page has a memory; that composite has a derived fact => fact-complete.
        composite = _composite_mem("mem_a", source_page_id="page_a")
        storage = _make_storage(
            pages=[page], mem_page_ids={"page_a"},
            document_memories=[composite], fact_parent_ids={"mem_a"},
            get_document=lambda doc_id, ws: doc,
        )
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(
                _make_v(), {"workspace_id": "ws_test", "document_id": "doc1"},
            )

        storage.update_document.assert_not_called()
        ts.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_document_is_noop(self):
        storage = _make_storage(get_document=lambda doc_id, ws: None)
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(
                _make_v(), {"workspace_id": "ws_test", "document_id": "missing"},
            )

        storage.update_document.assert_not_called()
        ts.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_inflight_fresh_processing_untouched(self):
        """A fresh PROCESSING doc is owned by another worker — left alone."""
        doc = _make_doc(
            status=DocumentStatus.PROCESSING,
            page_count=1,
            processing_started_at=datetime.now(timezone.utc),
        )
        storage = _make_storage(get_document=lambda doc_id, ws: doc)
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(
                _make_v(), {"workspace_id": "ws_test", "document_id": "doc1"},
            )

        storage.update_document.assert_not_called()
        ts.schedule_task.assert_not_called()
        # gap analysis never reached
        storage.get_pages.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_workspace_id_is_noop(self):
        storage = _make_storage()
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(_make_v(), {"document_id": "doc1"})

        storage.get_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_lost_claim_race_skips_resume(self):
        """A doc with a store gap whose CAS claim is lost is NOT resumed (no job,
        no phase scheduled) — another worker owns it."""
        doc = _make_doc(status=DocumentStatus.FAILED, page_count=1)
        page = _complete_page("page_a")
        storage = _make_storage(
            pages=[page], mem_page_ids=set(),  # store gap
            get_document=lambda doc_id, ws: doc,
            claim=False,  # lose the CAS
        )
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(
                _make_v(), {"workspace_id": "ws_test", "document_id": "doc1"},
            )

        storage.try_claim_document.assert_awaited_once()
        # Lost the race: no job created, no chained phase scheduled.
        storage.create_job.assert_not_called()
        assert not [
            c for c in ts.schedule_task.call_args_list
            if c.args and c.args[0] == "document_finalize"
        ]


# ---------------------------------------------------------------------------
# Fact-gap detection
# ---------------------------------------------------------------------------

class TestDocVerifyFactGaps:
    @pytest.mark.asyncio
    async def test_composite_without_fact_reschedules_decompose(self):
        """A complete doc whose page composite has no fact re-drives decompose_facts."""
        doc = _make_doc(status=DocumentStatus.COMPLETED, page_count=1)
        page = _complete_page("page_a")
        composite = _composite_mem("mem_a", source_page_id="page_a")
        storage = _make_storage(
            pages=[page], mem_page_ids={"page_a"},
            document_memories=[composite], fact_parent_ids=set(),  # no facts yet
            get_document=lambda doc_id, ws: doc,
        )
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(
                _make_v(), {"workspace_id": "ws_test", "document_id": "doc1"},
            )

        # Doc is complete => no phase resume; only decompose_facts re-scheduled.
        decompose_calls = [
            c for c in ts.schedule_task.call_args_list
            if c.args and c.args[0] == "decompose_facts"
        ]
        assert len(decompose_calls) == 1
        assert decompose_calls[0].args[1]["memory_id"] == "mem_a"
        assert decompose_calls[0].args[1]["workspace_id"] == "ws_test"
        # Did NOT claim PROCESSING (complete doc).
        assert not [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.PROCESSING.value
        ]

    @pytest.mark.asyncio
    async def test_archived_composite_not_rescheduled(self):
        """An archived composite (already decomposed) is skipped, not re-driven."""
        doc = _make_doc(status=DocumentStatus.COMPLETED, page_count=1)
        page = _complete_page("page_a")
        composite = _composite_mem("mem_a", source_page_id="page_a", status="archived")
        storage = _make_storage(
            pages=[page], mem_page_ids={"page_a"},
            document_memories=[composite], fact_parent_ids=set(),
            get_document=lambda doc_id, ws: doc,
        )
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(
                _make_v(), {"workspace_id": "ws_test", "document_id": "doc1"},
            )

        assert not [
            c for c in ts.schedule_task.call_args_list
            if c.args and c.args[0] == "decompose_facts"
        ]

    @pytest.mark.asyncio
    async def test_fact_subtype_memory_not_treated_as_composite(self):
        """A subtype='fact' memory is never itself a decompose target."""
        doc = _make_doc(status=DocumentStatus.COMPLETED, page_count=1)
        page = _complete_page("page_a")
        composite = _composite_mem("mem_a", source_page_id="page_a")
        fact = _composite_mem("fact_1", source_page_id="page_a", subtype="fact")
        storage = _make_storage(
            pages=[page], mem_page_ids={"page_a"},
            document_memories=[composite, fact], fact_parent_ids={"mem_a"},
            get_document=lambda doc_id, ws: doc,
        )
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(
                _make_v(), {"workspace_id": "ws_test", "document_id": "doc1"},
            )

        # mem_a has a fact => nothing to re-drive.
        assert not [
            c for c in ts.schedule_task.call_args_list
            if c.args and c.args[0] == "decompose_facts"
        ]


# ---------------------------------------------------------------------------
# Scheduled sweep
# ---------------------------------------------------------------------------

class TestDocVerifySweep:
    @pytest.mark.asyncio
    async def test_sweep_selects_partial_failed_stale_processing(self):
        """Sweep gap-fills PARTIAL/FAILED and stale-PROCESSING; skips fresh-PROCESSING."""
        partial = _make_doc(doc_id="d_partial", status=DocumentStatus.PARTIAL, page_count=1)
        failed = _make_doc(doc_id="d_failed", status=DocumentStatus.FAILED, page_count=1)
        stale = _make_doc(
            doc_id="d_stale", status=DocumentStatus.PROCESSING, page_count=1,
            processing_started_at=datetime.now(timezone.utc) - timedelta(seconds=10_000),
        )
        fresh = _make_doc(
            doc_id="d_fresh", status=DocumentStatus.PROCESSING, page_count=1,
            processing_started_at=datetime.now(timezone.utc),
        )

        # Each has a store gap (page with no memory) so each is fixable.
        page = _complete_page("page_a")
        storage = _make_storage(
            pages=[page], mem_page_ids=set(),
            list_documents_map={
                DocumentStatus.PARTIAL.value: [partial],
                DocumentStatus.FAILED.value: [failed],
                DocumentStatus.PROCESSING.value: [stale, fresh],
            },
        )
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(
                _make_v(), {"workspace_id": "ws_test"},  # per-workspace sweep
            )

        # 3 docs resumed (partial, failed, stale) — fresh skipped.
        resumed_finalize = [
            c for c in ts.schedule_task.call_args_list
            if c.args and c.args[0] == "document_finalize"
        ]
        assert len(resumed_finalize) == 3
        # 3 docs CAS-claimed (partial, failed, stale) — fresh skipped pre-claim.
        assert storage.try_claim_document.await_count == 3

    @pytest.mark.asyncio
    async def test_global_sweep_iterates_all_workspaces(self):
        """An empty payload sweeps every workspace from list_all_workspace_ids."""
        failed = _make_doc(doc_id="d1", status=DocumentStatus.FAILED, page_count=1)
        page = _complete_page("page_a")
        storage = _make_storage(
            pages=[page], mem_page_ids=set(),
            workspace_ids=["ws_a", "ws_b"],
            list_documents_map={DocumentStatus.FAILED.value: [failed]},
        )
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(_make_v(), {})

        storage.list_all_workspace_ids.assert_awaited_once()
        # list_documents called for each (workspace x status).
        swept_ws = {c.args[0] for c in storage.list_documents.call_args_list}
        assert swept_ws == {"ws_a", "ws_b"}


# ---------------------------------------------------------------------------
# Attempt-cap, terminal flag, and is_complete self-heal
# (anti-churn guard folded into doc_verify).
# ---------------------------------------------------------------------------

class TestDocVerifyCapAndHeal:
    """Tests for the attempt-cap / terminal-flag / is_complete self-heal additions."""

    @pytest.mark.asyncio
    async def test_under_cap_resume_increments_attempts(self):
        """A sweep-driven resume increments metadata['reprocess_attempts'] and persists."""
        doc = _make_doc(
            doc_id="d_cap", status=DocumentStatus.FAILED, page_count=1,
            metadata={"reprocess_attempts": 2},
        )
        page = _complete_page("page_a")
        storage = _make_storage(
            pages=[page], mem_page_ids=set(),
            list_documents_map={DocumentStatus.FAILED.value: [doc]},
        )
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(
                _make_v(), {"workspace_id": "ws_test"},
            )

        # Attempt counter persisted as 3.
        meta_updates = [
            c for c in storage.update_document.call_args_list
            if "metadata" in c.kwargs
            and c.kwargs["metadata"].get("reprocess_attempts") == 3
        ]
        assert len(meta_updates) == 1
        # Phase was resumed.
        storage.try_claim_document.assert_awaited_once()
        ts.schedule_task.assert_called()

    @pytest.mark.asyncio
    async def test_over_cap_marked_terminal_not_resumed(self):
        """A doc at/over the reprocess cap is marked terminally FAILED, not resumed."""
        doc = _make_doc(
            doc_id="d_overcap", status=DocumentStatus.FAILED, page_count=1,
            metadata={"reprocess_attempts": 5},  # == default cap
        )
        page = _complete_page("page_a")
        storage = _make_storage(
            pages=[page], mem_page_ids=set(),
            list_documents_map={DocumentStatus.FAILED.value: [doc]},
        )
        ts = _make_task_service()

        # Override default so max_attempts=5 is returned.
        def _v_environ(key, default=None, **kw):
            from memorylayer_saas.config import MEMORYLAYER_DOC_VERIFY_MAX_ATTEMPTS
            if key == MEMORYLAYER_DOC_VERIFY_MAX_ATTEMPTS:
                return 5
            return default

        v = MagicMock()
        v.environ = MagicMock(side_effect=_v_environ)

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(v, {"workspace_id": "ws_test"})

        # Marked terminal FAILED with the reconcile_terminal flag.
        terminal_updates = [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.FAILED.value
            and c.kwargs.get("metadata", {}).get("reconcile_terminal") is True
        ]
        assert len(terminal_updates) == 1
        # Must NOT have been claimed or had a phase scheduled.
        storage.try_claim_document.assert_not_awaited()
        ts.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_mislabeled_complete_doc_healed_to_completed(self):
        """A FAILED doc that gap-analysis finds complete is healed to COMPLETED."""
        doc = _make_doc(
            doc_id="d_heal", status=DocumentStatus.FAILED, page_count=1,
            metadata={"reprocess_attempts": 1},
        )
        page = _complete_page("page_a")
        # Storage says the page has a memory → gap-analysis is_complete.
        storage = _make_storage(
            pages=[page], mem_page_ids={"page_a"},
            document_memories=[], fact_parent_ids=set(),
            list_documents_map={DocumentStatus.FAILED.value: [doc]},
        )
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(
                _make_v(), {"workspace_id": "ws_test"},
            )

        # Healed to COMPLETED; not claimed, no task scheduled, not marked terminal.
        healed = [
            c for c in storage.update_document.call_args_list
            if c.kwargs.get("status") == DocumentStatus.COMPLETED.value
        ]
        assert len(healed) == 1
        storage.try_claim_document.assert_not_awaited()
        assert not [
            c for c in ts.schedule_task.call_args_list
            if c.args and c.args[0] not in ("decompose_facts",)
        ]

    @pytest.mark.asyncio
    async def test_terminal_doc_excluded_from_sweep(self):
        """A doc already carrying reconcile_terminal=True is filtered out as a candidate."""
        doc = _make_doc(
            doc_id="d_terminal", status=DocumentStatus.FAILED, page_count=1,
            metadata={"reconcile_terminal": True},
        )
        page = _complete_page("page_a")
        storage = _make_storage(
            pages=[page], mem_page_ids=set(),
            list_documents_map={DocumentStatus.FAILED.value: [doc]},
        )
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(
                _make_v(), {"workspace_id": "ws_test"},
            )

        # Terminal doc filtered in candidate collection — nothing touches it.
        storage.try_claim_document.assert_not_awaited()
        ts.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_ondemand_bypasses_cap(self):
        """On-demand (document_id present) bypasses the attempt cap and always resumes."""
        doc = _make_doc(
            doc_id="d_ondemand", status=DocumentStatus.FAILED, page_count=1,
            metadata={"reprocess_attempts": 99},  # far over any cap
        )
        page = _complete_page("page_a")
        storage = _make_storage(
            pages=[page], mem_page_ids=set(),
            get_document=lambda doc_id, ws: doc,
        )
        ts = _make_task_service()

        with _patch_exts(storage, ts):
            await DocVerifyTaskHandler().handle(
                _make_v(), {"workspace_id": "ws_test", "document_id": "d_ondemand"},
            )

        # On-demand always claims + resumes regardless of reprocess_attempts.
        storage.try_claim_document.assert_awaited_once()
        ts.schedule_task.assert_called()
