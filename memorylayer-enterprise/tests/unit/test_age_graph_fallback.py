# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the AGE -> NetworkX fail-open fallback (B3).

AGE is the enterprise default graph backend, but it must never fail the KB
pipeline closed. ``AgeGraphAnalysisService._build_graph`` wraps the AGE path
(``_build_graph_age``) so that ANY exception is loudly logged (so the fault is
detected, not swallowed) and the inherited OSS NetworkX ``_build_graph`` is used
instead — producing the same ``nx.Graph`` shape the callers expect.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import networkx as nx
import pytest

from memorylayer_saas.services.graph_analysis.age import AgeGraphAnalysisService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_memory(mem_id: str, memory_type: str = "fact", subtype: str | None = None):
    mem = MagicMock()
    mem.id = mem_id
    mem.memory_type = memory_type
    mem.subtype = subtype
    return mem


def _make_association(source_id: str, target_id: str, relationship="related_to", strength=0.5):
    assoc = MagicMock()
    assoc.source_id = source_id
    assoc.target_id = target_id
    assoc.relationship = relationship
    assoc.strength = strength
    return assoc


def _make_storage():
    """A storage backend mock sufficient for the inherited NetworkX path."""
    storage = MagicMock()
    storage.connection_string = "postgresql+asyncpg://stub/stub"
    storage.search_memories_by_filter = AsyncMock(
        return_value=[_make_memory("m1"), _make_memory("m2")]
    )
    storage.get_associations_batch = AsyncMock(
        return_value=[_make_association("m1", "m2")]
    )
    # Awaited by the densify entity-cooccurrence signal. Without it the bare
    # MagicMock returns a non-awaitable and densify logs its own warning, which
    # is noise here (and previously masked the warning under test by being the
    # most recent one).
    storage.list_workspace_entity_members = AsyncMock(return_value=[])
    return storage


def _make_service(storage) -> AgeGraphAnalysisService:
    # Pass a stub engine so __init__ never creates a real asyncpg engine.
    return AgeGraphAnalysisService(storage=storage, v=MagicMock(), age_engine=MagicMock())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgeNetworkXFallback:
    @pytest.mark.asyncio
    async def test_falls_back_to_networkx_on_age_failure(self):
        """When the AGE path raises, _build_graph falls back to NetworkX.

        Asserts (a) a valid nx.Graph is returned from the inherited path, and
        (b) a warning identifying the AGE fault + workspace is logged (not
        swallowed).
        """
        storage = _make_storage()
        service = _make_service(storage)
        # Replace the framework logger with a mock so the WARN call is captured
        # deterministically regardless of framework logging plumbing.
        service.logger = MagicMock()

        # Force the AGE path to raise (simulates a cypher/bootstrap fault).
        service._build_graph_age = AsyncMock(
            side_effect=RuntimeError("AGE bootstrap exploded")
        )

        g = await service._build_graph("ws-fallback")

        # (a) Valid graph from the inherited NetworkX path.
        assert isinstance(g, nx.Graph)
        assert set(g.nodes()) == {"m1", "m2"}
        assert g.has_edge("m1", "m2")
        # Built via the inherited NetworkX path (uses get_associations_batch).
        storage.get_associations_batch.assert_awaited_once()

        # (b) The AGE fault is loudly logged (NOT swallowed): the WARN call
        # mentions the workspace and carries the underlying error.
        assert service.logger.warning.called
        # Search ALL warnings rather than only the most recent: later phases of
        # the same call legitimately warn too, and keying on call_args made this
        # assert whichever warning happened to come last.
        rendered = [
            call.args[0] % tuple(call.args[1:])
            for call in service.logger.warning.call_args_list
            if call.args
        ]
        assert any(
            "ws-fallback" in line and "AGE bootstrap exploded" in line
            for line in rendered
        ), rendered

    @pytest.mark.asyncio
    async def test_age_success_does_not_fall_back(self):
        """When the AGE path succeeds, its result is returned untouched."""
        storage = _make_storage()
        service = _make_service(storage)

        age_graph = nx.Graph()
        age_graph.add_node("age-only")
        service._build_graph_age = AsyncMock(return_value=age_graph)

        g = await service._build_graph("ws-ok")

        assert g is age_graph
        # NetworkX fallback path was NOT taken.
        storage.get_associations_batch.assert_not_awaited()
