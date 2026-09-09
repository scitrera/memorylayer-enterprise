"""Unit tests for the EnterpriseRepresentationService (P3 derived-beliefs layer).

The enterprise service subclasses the OSS ``DefaultRepresentationService`` and
adds ONE additive, fail-safe layer: ``Representation.derived_beliefs`` populated
by an LLM over the observer's leakage-safe scoped observations, reconciled
against known contradictions.

These tests MOCK the LLM (no real model calls) and reuse in-memory fakes for the
entity registry + storage (no SQLite, no recall). They assert:

  * happy path: 2 mock beliefs -> populated correctly (statements, clamped
    confidence, support_memory_ids ⊆ the scoped observation ids).
  * contradiction reconciliation: a belief whose support overlaps a known
    unresolved contradiction -> contradicted=True, still present (not dropped).
  * fail-safe: LLM raises / unparseable / empty -> derived_beliefs=[], with the
    observations + leakage guarantee IDENTICAL to the OSS super() result.
  * perspective integrity: the LLM prompt only ever receives the observer's
    scoped observations — never other-observer content (leakage-0 extends to
    derived beliefs); out-of-scope support ids the model emits are stripped.
"""

from datetime import UTC, datetime, timedelta

import pytest
from memorylayer_server.models.entity_registry import (
    Entity,
    EntityResolution,
    EntityType,
    Member,
)
from memorylayer_server.models.llm import LLMResponse
from memorylayer_server.models.memory import Memory, MemoryStatus, MemoryType
from memorylayer_server.services.contradiction.base import ContradictionRecord

from memorylayer_saas.services.representation.default import EnterpriseRepresentationService

WS = "ws-rep-ent-1"


# --------------------------------------------------------------------------- #
# Fakes — minimal registry + storage (no recall surface exists, which is itself
# evidence of the no-recall property), plus a recording mock LLM.
# --------------------------------------------------------------------------- #
class FakeRegistry:
    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.members: dict[str, list[Member]] = {}

    def add_entity(self, name: str, entity_id: str, entity_type=EntityType.PERSON) -> Entity:
        now = datetime.now(UTC)
        ent = Entity(
            id=entity_id,
            workspace_id=WS,
            entity_type=entity_type,
            canonical_name=name,
            normalized_name=name.lower(),
            created_at=now,
            updated_at=now,
        )
        self.entities[name.lower()] = ent
        return ent

    def add_member(self, entity_id: str, memory_id: str, role: str):
        self.members.setdefault(entity_id, []).append(
            Member(entity_id=entity_id, memory_id=memory_id, role=role)
        )

    async def resolve(self, workspace_id, name, entity_type, *, allow_create=True, **kw):
        ent = self.entities.get(name.lower())
        if ent is None:
            if not allow_create:
                raise LookupError(f"no entity {name!r}")
            raise AssertionError("representation service must never create entities")
        return EntityResolution(entity=ent, matched_via="exact", score=1.0)

    async def list_members(self, workspace_id, entity_id, *, role=None, limit=100):
        rows = self.members.get(entity_id, [])
        if role is not None:
            rows = [m for m in rows if m.role == role]
        return rows[:limit]


class FakeStorage:
    def __init__(self):
        self.memories: dict[str, Memory] = {}
        self.track_access_calls: list[bool] = []

    def add_memory(self, mem: Memory):
        self.memories[mem.id] = mem

    async def get_memory(self, workspace_id, memory_id, track_access=True):
        self.track_access_calls.append(track_access)
        return self.memories.get(memory_id)


class FakeLLM:
    """Recording mock LLM. Returns a canned content (or raises) and stores the
    requests it received so tests can assert exactly what was prompted."""

    def __init__(self, *, content: str | None = None, raises: bool = False):
        self._content = content
        self._raises = raises
        self.requests = []  # list[LLMRequest]

    async def complete(self, request, profile="default", **_generation_metadata):
        self.requests.append(request)
        if self._raises:
            raise RuntimeError("simulated LLM failure")
        return LLMResponse(
            content=self._content,
            model="mock",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            finish_reason="stop",
        )

    def prompt_text(self) -> str:
        """Concatenated content of all messages in the last request."""
        assert self.requests, "LLM was never called"
        return "\n".join(m.content for m in self.requests[-1].messages)


class FakeContradiction:
    """Minimal contradiction service exposing only ``get_unresolved``."""

    def __init__(self, records: list[ContradictionRecord] | None = None):
        self._records = records or []

    async def get_unresolved(self, workspace_id, limit=10):
        return list(self._records)


def _mem(mem_id: str, content: str, *, event_offset_days: int = 1,
         status: MemoryStatus = MemoryStatus.ACTIVE) -> Memory:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return Memory(
        id=mem_id,
        workspace_id=WS,
        tenant_id="_default",
        content=content,
        content_hash=f"hash-{mem_id}",
        type=MemoryType.SEMANTIC,
        status=status,
        event_time=base + timedelta(days=event_offset_days),
        created_at=base,
        updated_at=base,
    )


@pytest.fixture
def registry():
    return FakeRegistry()


@pytest.fixture
def storage():
    return FakeStorage()


def _self_setup(registry, storage):
    """Alice self-report: two self-authored observations m1, m2."""
    alice = registry.add_entity("Alice", "ent-alice")
    registry.add_member(alice.id, "m1", "self")
    registry.add_member(alice.id, "m2", "self")
    storage.add_memory(_mem("m1", "I shipped the release", event_offset_days=2))
    storage.add_memory(_mem("m2", "I reviewed the PR", event_offset_days=1))
    return alice


@pytest.mark.asyncio
class TestDerivedBeliefsHappyPath:
    async def test_two_beliefs_populated(self, registry, storage):
        _self_setup(registry, storage)
        llm = FakeLLM(content=(
            '{"beliefs": ['
            '{"statement": "Alice is a productive engineer", "confidence": 0.7, '
            '"support_memory_ids": ["m1", "m2"]},'
            '{"statement": "Alice cares about code quality", "confidence": 0.6, '
            '"support_memory_ids": ["m2"]}'
            ']}'
        ))
        svc = EnterpriseRepresentationService(
            registry=registry, storage=storage, v=None, llm_service=llm,
            contradiction_service=None,
        )

        rep = await svc.get_representation(WS, "Alice", "Alice")

        # OSS assembly unchanged.
        assert {o.memory_id for o in rep.observations} == {"m1", "m2"}
        assert rep.provenance["scoping_mode"] == "self"

        # Derived beliefs populated correctly.
        assert len(rep.derived_beliefs) == 2
        b0, b1 = rep.derived_beliefs
        assert b0.statement == "Alice is a productive engineer"
        assert b0.confidence == 0.7
        assert set(b0.support_memory_ids) == {"m1", "m2"}
        assert b0.contradicted is False
        assert b1.support_memory_ids == ["m2"]
        # Support ids are a subset of the scoped observation ids.
        scoped = {o.memory_id for o in rep.observations}
        for b in rep.derived_beliefs:
            assert set(b.support_memory_ids) <= scoped

    async def test_max_beliefs_bound(self, registry, storage):
        _self_setup(registry, storage)
        many = ",".join(
            f'{{"statement": "b{i}", "confidence": 0.5, "support_memory_ids": ["m1"]}}'
            for i in range(10)
        )
        llm = FakeLLM(content=f'{{"beliefs": [{many}]}}')
        svc = EnterpriseRepresentationService(
            registry=registry, storage=storage, v=None, llm_service=llm,
            contradiction_service=None, max_beliefs=3,
        )
        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert len(rep.derived_beliefs) == 3

    async def test_confidence_clamped_and_bad_ids_stripped(self, registry, storage):
        _self_setup(registry, storage)
        llm = FakeLLM(content=(
            '{"beliefs": [{"statement": "overconfident", "confidence": 1.5, '
            '"support_memory_ids": ["m1", "ghost-id", "m2"]}]}'
        ))
        svc = EnterpriseRepresentationService(
            registry=registry, storage=storage, v=None, llm_service=llm,
            contradiction_service=None,
        )
        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert len(rep.derived_beliefs) == 1
        b = rep.derived_beliefs[0]
        assert b.confidence == 1.0  # clamped
        # out-of-scope "ghost-id" stripped; only scoped ids remain.
        assert set(b.support_memory_ids) == {"m1", "m2"}


@pytest.mark.asyncio
class TestContradictionReconciliation:
    async def test_belief_flagged_contradicted_but_kept(self, registry, storage):
        _self_setup(registry, storage)
        llm = FakeLLM(content=(
            '{"beliefs": ['
            '{"statement": "rests on m1", "confidence": 0.8, "support_memory_ids": ["m1"]},'
            '{"statement": "rests on m2", "confidence": 0.8, "support_memory_ids": ["m2"]}'
            ']}'
        ))
        # m1 participates in an unresolved contradiction.
        contra = FakeContradiction([
            ContradictionRecord(workspace_id=WS, memory_a_id="m1", memory_b_id="other-mem")
        ])
        svc = EnterpriseRepresentationService(
            registry=registry, storage=storage, v=None, llm_service=llm,
            contradiction_service=contra,
        )
        rep = await svc.get_representation(WS, "Alice", "Alice")

        assert len(rep.derived_beliefs) == 2  # both KEPT
        by_stmt = {b.statement: b for b in rep.derived_beliefs}
        assert by_stmt["rests on m1"].contradicted is True
        assert by_stmt["rests on m2"].contradicted is False

    async def test_no_contradictions_leaves_all_unflagged(self, registry, storage):
        _self_setup(registry, storage)
        llm = FakeLLM(content=(
            '{"beliefs": [{"statement": "x", "confidence": 0.5, "support_memory_ids": ["m1"]}]}'
        ))
        contra = FakeContradiction([])  # none
        svc = EnterpriseRepresentationService(
            registry=registry, storage=storage, v=None, llm_service=llm,
            contradiction_service=contra,
        )
        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert rep.derived_beliefs[0].contradicted is False

    async def test_contradiction_service_error_does_not_break(self, registry, storage):
        _self_setup(registry, storage)
        llm = FakeLLM(content=(
            '{"beliefs": [{"statement": "x", "confidence": 0.5, "support_memory_ids": ["m1"]}]}'
        ))

        class BrokenContra:
            async def get_unresolved(self, *a, **k):
                raise RuntimeError("contradiction store down")

        svc = EnterpriseRepresentationService(
            registry=registry, storage=storage, v=None, llm_service=llm,
            contradiction_service=BrokenContra(),
        )
        rep = await svc.get_representation(WS, "Alice", "Alice")
        # Belief survives; just left unflagged.
        assert len(rep.derived_beliefs) == 1
        assert rep.derived_beliefs[0].contradicted is False


@pytest.mark.asyncio
class TestFailSafe:
    async def _oss_baseline(self, registry, storage):
        """The deterministic OSS result for the same inputs (no LLM)."""
        from memorylayer_server.services.representation.default import (
            DefaultRepresentationService,
        )
        oss = DefaultRepresentationService(registry=registry, storage=storage, v=None)
        return await oss.get_representation(WS, "Alice", "Alice")

    async def test_llm_raises_yields_empty_and_observations_unchanged(self, registry, storage):
        _self_setup(registry, storage)
        baseline = await self._oss_baseline(_copy_registry(registry), _copy_storage(storage))

        llm = FakeLLM(raises=True)
        svc = EnterpriseRepresentationService(
            registry=registry, storage=storage, v=None, llm_service=llm,
            contradiction_service=None,
        )
        rep = await svc.get_representation(WS, "Alice", "Alice")

        assert rep.derived_beliefs == []  # exactly OSS behavior
        # Observations identical to the OSS super() result.
        assert [o.memory_id for o in rep.observations] == [o.memory_id for o in baseline.observations]
        assert rep.provenance["scoping_mode"] == baseline.provenance["scoping_mode"]

    async def test_unparseable_yields_empty(self, registry, storage):
        _self_setup(registry, storage)
        llm = FakeLLM(content="this is not json at all {")
        svc = EnterpriseRepresentationService(
            registry=registry, storage=storage, v=None, llm_service=llm,
            contradiction_service=None,
        )
        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert rep.derived_beliefs == []
        assert {o.memory_id for o in rep.observations} == {"m1", "m2"}

    async def test_empty_content_yields_empty(self, registry, storage):
        _self_setup(registry, storage)
        for content in ("", "   ", '{"beliefs": []}'):
            r = FakeRegistry()
            s = FakeStorage()
            _self_setup(r, s)
            llm = FakeLLM(content=content)
            svc = EnterpriseRepresentationService(
                registry=r, storage=s, v=None, llm_service=llm, contradiction_service=None,
            )
            rep = await svc.get_representation(WS, "Alice", "Alice")
            assert rep.derived_beliefs == []

    async def test_no_llm_service_yields_empty(self, registry, storage):
        _self_setup(registry, storage)
        svc = EnterpriseRepresentationService(
            registry=registry, storage=storage, v=None, llm_service=None,
            contradiction_service=None,
        )
        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert rep.derived_beliefs == []
        assert {o.memory_id for o in rep.observations} == {"m1", "m2"}

    async def test_derive_flag_off_yields_empty(self, registry, storage):
        _self_setup(registry, storage)
        llm = FakeLLM(content='{"beliefs": [{"statement": "x", "confidence": 0.5, "support_memory_ids": ["m1"]}]}')
        svc = EnterpriseRepresentationService(
            registry=registry, storage=storage, v=None, llm_service=llm,
            contradiction_service=None, derive_beliefs=False,
        )
        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert rep.derived_beliefs == []
        # LLM never even called when the layer is off.
        assert llm.requests == []

    async def test_no_observations_skips_llm(self, registry, storage):
        # Resolvable entity but no members -> no observations -> no derivation.
        registry.add_entity("Alice", "ent-alice")
        llm = FakeLLM(content='{"beliefs": []}')
        svc = EnterpriseRepresentationService(
            registry=registry, storage=storage, v=None, llm_service=llm,
            contradiction_service=None,
        )
        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert rep.observations == []
        assert rep.derived_beliefs == []
        assert llm.requests == []  # no LLM call when nothing to derive from


@pytest.mark.asyncio
class TestPerspectiveIntegrity:
    async def test_prompt_only_sees_observer_scoped_observations(self, registry, storage):
        """THE leakage guarantee extended to beliefs: the LLM prompt contains ONLY
        the observer's scoped (intersection) observations — never another
        observer's content."""
        observer = registry.add_entity("Observer", "ent-o")
        subject = registry.add_entity("Subject", "ent-s")
        other = registry.add_entity("Other", "ent-other")

        # M1: O authored + S mentioned -> in scope.
        registry.add_member(observer.id, "M1", "self")
        registry.add_member(subject.id, "M1", "mention")
        # M2: S mentioned, authored by a DIFFERENT observer -> EXCLUDED.
        registry.add_member(subject.id, "M2", "mention")
        registry.add_member(other.id, "M2", "self")

        storage.add_memory(_mem("M1", "SUBJECT_CLOSED_THE_DEAL_TOKEN", event_offset_days=1))
        storage.add_memory(_mem("M2", "OTHER_OBSERVER_SECRET_TOKEN", event_offset_days=1))

        llm = FakeLLM(content='{"beliefs": []}')
        svc = EnterpriseRepresentationService(
            registry=registry, storage=storage, v=None, llm_service=llm,
            contradiction_service=None,
        )
        rep = await svc.get_representation(WS, "Observer", "Subject")

        # Only the in-scope observation reaches the LLM; the other-observer
        # content NEVER appears in the prompt.
        prompt = llm.prompt_text()
        assert "SUBJECT_CLOSED_THE_DEAL_TOKEN" in prompt
        assert "OTHER_OBSERVER_SECRET_TOKEN" not in prompt
        # And the observation set itself is leakage-safe (only M1).
        assert {o.memory_id for o in rep.observations} == {"M1"}

    async def test_untrusted_content_is_delimited(self, registry, storage):
        _self_setup(registry, storage)
        llm = FakeLLM(content='{"beliefs": []}')
        svc = EnterpriseRepresentationService(
            registry=registry, storage=storage, v=None, llm_service=llm,
            contradiction_service=None,
        )
        await svc.get_representation(WS, "Alice", "Alice")
        prompt = llm.prompt_text()
        # Observation content is wrapped in the untrusted delimiter (injection
        # hardening) and the system prompt labels it as data, not instructions.
        assert "<untrusted_observation" in prompt
        assert "never instructions" in prompt.lower()


# --------------------------------------------------------------------------- #
# Tiny deep-copy helpers so the fail-safe baseline runs over an equivalent but
# independent fake (the service consumes get_memory calls / track_access state).
# --------------------------------------------------------------------------- #
def _copy_registry(src: FakeRegistry) -> FakeRegistry:
    r = FakeRegistry()
    r.entities = dict(src.entities)
    r.members = {k: list(v) for k, v in src.members.items()}
    return r


def _copy_storage(src: FakeStorage) -> FakeStorage:
    s = FakeStorage()
    s.memories = dict(src.memories)
    return s


# --------------------------------------------------------------------------- #
# User-scope (cross-workspace) self-representation: enterprise derived beliefs
# over the user-global set. Reuses FakeLLM / FakeContradiction above.
# --------------------------------------------------------------------------- #
from memorylayer_server.config import GLOBAL_USER_WORKSPACE_ID  # noqa: E402


class FakeUserStorage:
    """Storage fake exposing ``search_memories_by_filter`` honouring the FORCED
    user_id filter (the cross-user boundary)."""

    def __init__(self):
        self.memories: list[Memory] = []

    def add(self, mem: Memory):
        self.memories.append(mem)

    async def search_memories_by_filter(
        self, workspace_id, *, user_id=None, status="active", limit=100, offset=0, **kw
    ):
        rows = [
            m
            for m in self.memories
            if m.workspace_id == workspace_id and (user_id is None or m.user_id == user_id)
        ]
        return rows[offset : offset + limit]


def _user_mem(mem_id: str, content: str, *, user_id: str, origin="ws-1", event_offset_days: int = 1) -> Memory:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return Memory(
        id=mem_id,
        workspace_id=GLOBAL_USER_WORKSPACE_ID,
        tenant_id="_default",
        user_id=user_id,
        content=content,
        content_hash=f"hash-{mem_id}",
        type=MemoryType.SEMANTIC,
        event_time=base + timedelta(days=event_offset_days),
        metadata={"origin_workspace_id": origin},
        created_at=base,
        updated_at=base,
    )


@pytest.mark.asyncio
class TestUserScopeDerivedBeliefs:
    async def test_beliefs_derived_over_user_global_set(self):
        storage = FakeUserStorage()
        storage.add(_user_mem("u1", "Prefers dark mode", user_id="A", origin="ws-1", event_offset_days=2))
        storage.add(_user_mem("u2", "Likes concise answers", user_id="A", origin="ws-2", event_offset_days=1))
        llm = FakeLLM(content=(
            '{"beliefs": [{"statement": "Prefers minimal UI", "confidence": 0.7, '
            '"support_memory_ids": ["u1", "u2"]}]}'
        ))
        svc = EnterpriseRepresentationService(
            registry=None, storage=storage, v=None, llm_service=llm, contradiction_service=None,
        )

        rep = await svc.get_user_representation("A")

        # OSS user-scope assembly unchanged (cross-workspace span).
        assert {o.memory_id for o in rep.observations} == {"u1", "u2"}
        assert rep.provenance["origin_workspace_ids"] == ["ws-1", "ws-2"]
        # Derived beliefs populated over the user-global set.
        assert len(rep.derived_beliefs) == 1
        b = rep.derived_beliefs[0]
        assert b.statement == "Prefers minimal UI"
        assert set(b.support_memory_ids) == {"u1", "u2"}

    async def test_cross_user_isolation_llm_never_sees_other_user(self):
        """The LLM prompt only ever contains user A's user-global observations —
        user B's content NEVER reaches it (cross-user leakage-0 over beliefs)."""
        storage = FakeUserStorage()
        storage.add(_user_mem("u1", "A_PREFERENCE_TOKEN", user_id="A", event_offset_days=1))
        storage.add(_user_mem("b1", "B_SECRET_TOKEN", user_id="B", event_offset_days=1))
        llm = FakeLLM(content='{"beliefs": []}')
        svc = EnterpriseRepresentationService(
            registry=None, storage=storage, v=None, llm_service=llm, contradiction_service=None,
        )
        rep = await svc.get_user_representation("A")

        assert {o.memory_id for o in rep.observations} == {"u1"}
        prompt = llm.prompt_text()
        assert "A_PREFERENCE_TOKEN" in prompt
        assert "B_SECRET_TOKEN" not in prompt

    async def test_llm_error_fails_safe_to_empty(self):
        storage = FakeUserStorage()
        storage.add(_user_mem("u1", "pref", user_id="A", event_offset_days=1))
        llm = FakeLLM(raises=True)
        svc = EnterpriseRepresentationService(
            registry=None, storage=storage, v=None, llm_service=llm, contradiction_service=None,
        )
        rep = await svc.get_user_representation("A")
        # Exactly OSS behavior: empty beliefs, observations intact.
        assert rep.derived_beliefs == []
        assert {o.memory_id for o in rep.observations} == {"u1"}

    async def test_no_llm_service_yields_empty(self):
        storage = FakeUserStorage()
        storage.add(_user_mem("u1", "pref", user_id="A", event_offset_days=1))
        svc = EnterpriseRepresentationService(
            registry=None, storage=storage, v=None, llm_service=None, contradiction_service=None,
        )
        rep = await svc.get_user_representation("A")
        assert rep.derived_beliefs == []
        assert {o.memory_id for o in rep.observations} == {"u1"}

    async def test_no_user_id_skips_llm(self):
        storage = FakeUserStorage()
        llm = FakeLLM(content='{"beliefs": [{"statement": "x", "confidence": 0.5, "support_memory_ids": []}]}')
        svc = EnterpriseRepresentationService(
            registry=None, storage=storage, v=None, llm_service=llm, contradiction_service=None,
        )
        rep = await svc.get_user_representation("")
        assert rep.observations == []
        assert rep.derived_beliefs == []
        assert llm.requests == []  # no observations -> no LLM call

    async def test_contradiction_flagged_over_user_global(self):
        storage = FakeUserStorage()
        storage.add(_user_mem("u1", "pref one", user_id="A", event_offset_days=1))
        llm = FakeLLM(content='{"beliefs": [{"statement": "rests on u1", "confidence": 0.8, "support_memory_ids": ["u1"]}]}')
        contra = FakeContradiction([
            ContradictionRecord(workspace_id=GLOBAL_USER_WORKSPACE_ID, memory_a_id="u1", memory_b_id="other")
        ])
        svc = EnterpriseRepresentationService(
            registry=None, storage=storage, v=None, llm_service=llm, contradiction_service=contra,
        )
        rep = await svc.get_user_representation("A")
        assert len(rep.derived_beliefs) == 1
        assert rep.derived_beliefs[0].contradicted is True
