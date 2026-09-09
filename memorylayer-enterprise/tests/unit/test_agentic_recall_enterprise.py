# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Enterprise Phase 2: agentic EXPAND/RE_QUERY/STOP recall ("Memora-Control").

These tests cover the ENTERPRISE ``_recall_agentic`` controller loop and its
dispatch from ``recall``:

  * RE_QUERY (pointer answer): the seed says "Mike went to the same college as
    Sarah"; the controller RE_QUERIES for Sarah's college; both memories end up
    in the working set + result, and a RE_QUERY trajectory event is recorded.
  * STOP immediately: result equals the seed set, ``sufficiency_reached`` True,
    only one search hop.
  * max_steps bound: EXPAND that never adds anything terminates (dry-stop),
    never loops forever.
  * disabled fallback: ``agentic_recall_enabled=False`` + mode AGENTIC behaves as
    RAG (one ``_recall_rag`` call, no loop).
  * failure safety: ``llm_service.complete`` raising -> the seed RAG result is
    returned, no exception.

Everything is mocked (no network, no Postgres). ``_recall_rag`` is monkeypatched
to return controlled ``RecallResult`` objects keyed by query so the controller's
hop accounting is exercised end to end.
"""

import json

import pytest
from memorylayer_server.models.memory import (
    Memory,
    MemoryType,
    RecallInput,
    RecallMode,
    RecallResult,
)
from scitrera_app_framework import Variables

from memorylayer_saas.services.enterprise_memory.default import EnterpriseMemoryService


# --------------------------------------------------------------------------
# Lightweight isolated Variables (does not read the environment)
# --------------------------------------------------------------------------


@pytest.fixture
def v() -> Variables:
    return Variables()


# --------------------------------------------------------------------------
# Mocks
# --------------------------------------------------------------------------


class _MockLLMService:
    """Returns scripted decision JSON strings in order (or raises)."""

    def __init__(self, decisions=None, raises=False):
        # decisions: list of JSON strings, consumed one per complete() call.
        self._decisions = list(decisions or [])
        self._raises = raises
        self.call_count = 0
        self.last_profile = None

    async def complete(self, request, profile="default", **_generation_metadata):
        self.call_count += 1
        self.last_profile = profile
        if self._raises:
            raise RuntimeError("llm down")

        content = self._decisions.pop(0) if self._decisions else '{"action": "STOP"}'

        class _Resp:
            pass

        resp = _Resp()
        resp.content = content
        return resp


class _FakeEmbedding:
    """Deterministic no-op embedding."""

    async def embed(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]


class _FakeStorage:
    """Serves canned memories by id for EXPAND hydration."""

    def __init__(self, memories: dict[str, Memory] | None = None):
        self._by_id = dict(memories or {})

    async def get_memory(self, workspace_id: str, memory_id: str, track_access: bool = True):
        return self._by_id.get(memory_id)

    async def get_workspace(self, workspace_id: str):
        # No tiering settings -> cold-tier search stays disabled in ``recall``.
        return None


class _FakeGraphService:
    """Returns canned neighbor memory ids per seed memory id."""

    def __init__(self, neighbor_map: dict[str, list[str]] | None = None):
        self._map = dict(neighbor_map or {})
        self.calls: list[str] = []

    async def neighbors(self, workspace_id, memory_id, *, depth=1, direction="both", limit=50):
        self.calls.append(memory_id)

        class _Node:
            def __init__(self, mid):
                self.memory_id = mid

        class _Result:
            def __init__(self, nodes):
                self.nodes = nodes

        return _Result([_Node(mid) for mid in self._map.get(memory_id, [])])


def _mem(mem_id: str, content: str) -> Memory:
    return Memory(
        id=mem_id,
        workspace_id="ws-1",
        tenant_id="tenant-1",
        content=content,
        content_hash="hash-" + mem_id,
        type=MemoryType.SEMANTIC,
        importance=0.5,
    )


def _decision(action: str, rewrite: str = "", reason: str = "r") -> str:
    return json.dumps({"action": action, "rewrite": rewrite, "reason": reason})


def _make_service(v, llm=None, storage=None, graph=None, *, enabled=True, max_steps=4) -> EnterpriseMemoryService:
    svc = EnterpriseMemoryService(
        v=v,
        storage=storage or _FakeStorage(),
        embedding_service=_FakeEmbedding(),
        llm_service=llm,
        graph_query_service=graph,
    )
    svc.agentic_recall_enabled = enabled
    svc.agentic_max_steps = max_steps
    return svc


class _RagStub:
    """Monkeypatch target for ``_recall_rag``: returns memories keyed by query.

    Falls back to ``default`` for any unmapped query. Records every query it was
    asked, so tests can assert the exact hop sequence.
    """

    def __init__(self, by_query: dict[str, list[Memory]], default: list[Memory] | None = None):
        self._by_query = by_query
        self._default = default or []
        self.queries: list[str] = []

    async def __call__(self, workspace_id, input, relevance_threshold, **kwargs):
        self.queries.append(input.query)
        mems = self._by_query.get(input.query, self._default)
        return RecallResult(
            memories=list(mems),
            total_count=len(mems),
            query_tokens=0,
            search_latency_ms=0,
            mode_used=RecallMode.RAG,
        )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agentic_requery_pointer_answer(v):
    """Pointer question: RE_QUERY pulls the second entity's fact into W."""
    mike = _mem("mem-mike", "Mike went to the same college as Sarah.")
    sarah = _mem("mem-sarah", "Sarah studied at MIT.")

    rag = _RagStub(
        by_query={
            "Where did Mike go to college?": [mike],
            "Where did Sarah go to college?": [sarah],
        }
    )
    llm = _MockLLMService(
        decisions=[
            _decision("RE_QUERY", "Where did Sarah go to college?", "pointer to Sarah"),
            _decision("STOP", reason="have both facts"),
        ]
    )
    svc = _make_service(v, llm=llm)
    svc._recall_rag = rag  # monkeypatch

    # A tracing-enabled trajectory recorder so we can assert RE_QUERY was traced.
    events: list = []

    class _Traj:
        pass

    class _TrajSvc:
        def add_event(self, trajectory, event_type, data):
            events.append((event_type, data))

    svc.trajectory_service = _TrajSvc()
    trajectory = _Traj()

    result = await svc._recall_agentic("ws-1", RecallInput(query="Where did Mike go to college?", limit=10), trajectory=trajectory)

    ids = {m.id for m in result.memories}
    assert "mem-mike" in ids
    assert "mem-sarah" in ids
    # Seed hop + one RE_QUERY hop.
    assert rag.queries == ["Where did Mike go to college?", "Where did Sarah go to college?"]
    # A RE_QUERY trajectory event was recorded.
    from memorylayer_saas.models.trajectory import TrajectoryEventType

    assert any(evt == TrajectoryEventType.RE_QUERY for evt, _ in events)
    assert result.mode_used == RecallMode.AGENTIC
    assert result.sufficiency_reached is True


@pytest.mark.asyncio
async def test_agentic_stop_immediately_equals_seed(v):
    """STOP on the first decision -> result == seed set, sufficiency True, 1 hop."""
    seed = [_mem("mem-1", "Alice likes tea."), _mem("mem-2", "Bob likes coffee.")]
    rag = _RagStub(by_query={"q": seed})
    llm = _MockLLMService(decisions=[_decision("STOP", reason="enough")])
    svc = _make_service(v, llm=llm)
    svc._recall_rag = rag

    result = await svc._recall_agentic("ws-1", RecallInput(query="q", limit=10))

    assert {m.id for m in result.memories} == {"mem-1", "mem-2"}
    assert rag.queries == ["q"]  # only the seed hop
    assert result.sufficiency_reached is True
    assert result.mode_used == RecallMode.AGENTIC


@pytest.mark.asyncio
async def test_agentic_max_steps_bound_dry_expand(v):
    """EXPAND that never adds anything terminates (dry-stop), never loops forever."""
    seed = [_mem("mem-1", "seed fact")]
    # RAG fallback for EXPAND returns only the already-seen seed -> nothing new.
    rag = _RagStub(by_query={"q": seed}, default=seed)
    # Always EXPAND (more decisions than max_steps to prove the bound holds).
    llm = _MockLLMService(decisions=[_decision("EXPAND") for _ in range(20)])
    # No graph service -> EXPAND uses the RAG fallback, which adds nothing new.
    svc = _make_service(v, llm=llm, graph=None, max_steps=4)
    svc._recall_rag = rag

    result = await svc._recall_agentic("ws-1", RecallInput(query="q", limit=10))

    # Terminated: only the seed made it in, and the LLM was consulted a bounded
    # number of times (dry-stop kicks in after 2 empty expansions).
    assert {m.id for m in result.memories} == {"mem-1"}
    assert llm.call_count <= 4
    assert result.mode_used == RecallMode.AGENTIC


@pytest.mark.asyncio
async def test_agentic_expand_via_graph(v):
    """EXPAND uses the graph service to hydrate a linked memory into W."""
    seed = [_mem("mem-mike", "Mike is linked to a college fact.")]
    college = _mem("mem-college", "That college is Stanford.")
    storage = _FakeStorage({"mem-college": college})
    graph = _FakeGraphService({"mem-mike": ["mem-college"]})
    rag = _RagStub(by_query={"q": seed})
    llm = _MockLLMService(decisions=[_decision("EXPAND"), _decision("STOP")])
    svc = _make_service(v, llm=llm, storage=storage, graph=graph)
    svc._recall_rag = rag

    result = await svc._recall_agentic("ws-1", RecallInput(query="q", limit=10))

    ids = {m.id for m in result.memories}
    assert "mem-mike" in ids
    assert "mem-college" in ids
    assert graph.calls == ["mem-mike"]  # frontier seed expanded via graph


@pytest.mark.asyncio
async def test_agentic_disabled_falls_back_to_rag(v):
    """Disabled + mode AGENTIC -> behaves as RAG: one _recall_rag call, no loop."""
    seed = [_mem("mem-1", "fact")]
    rag = _RagStub(by_query={"q": seed})
    llm = _MockLLMService(decisions=[_decision("EXPAND")])
    svc = _make_service(v, llm=llm, enabled=False)
    svc._recall_rag = rag

    async def _noop_increment(workspace_id, memory_id):
        return None

    svc.increment_access = _noop_increment

    result = await svc.recall("ws-1", RecallInput(query="q", limit=10, mode=RecallMode.AGENTIC, trace=False))

    # Downgraded to RAG: exactly one RAG hop, controller LLM never consulted.
    assert rag.queries == ["q"]
    assert llm.call_count == 0
    assert result.mode_used == RecallMode.RAG


@pytest.mark.asyncio
async def test_agentic_llm_failure_returns_seed(v):
    """LLM.complete raising -> the seed RAG result is returned, no exception."""
    seed = [_mem("mem-1", "seed a"), _mem("mem-2", "seed b")]
    rag = _RagStub(by_query={"q": seed})
    llm = _MockLLMService(raises=True)
    svc = _make_service(v, llm=llm)
    svc._recall_rag = rag

    result = await svc._recall_agentic("ws-1", RecallInput(query="q", limit=10))

    # Never raised; working set == seed (a STOP from the fail-safe decision path).
    assert {m.id for m in result.memories} == {"mem-1", "mem-2"}
    assert rag.queries == ["q"]
    assert result.mode_used == RecallMode.AGENTIC
    assert result.sufficiency_reached is True
