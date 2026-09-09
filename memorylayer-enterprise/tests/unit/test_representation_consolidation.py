"""Unit tests for the P3 representation CONSOLIDATION layer (maintained profile).

The consolidation layer DERIVES + PERSISTS a maintained per-(observer, subject)
representation (profile + derived beliefs) so
``EnterpriseRepresentationService.get_representation`` can serve a maintained
profile cheaply (NO LLM) instead of deriving on every call.

These tests MOCK the LLM + cache + task service (no real model / network).  They
assert the cardinal properties:

  * coalesce: N concurrent triggers for the same subject -> ONE derivation run
    (the per-subject lease collapses the rest).
  * persist + serve: after consolidation, get_representation serves the
    maintained profile (with persisted derived_beliefs) WITHOUT calling the LLM
    again; stale/absent -> on-demand fallback (LLM IS called).
  * watermark: an unchanged subject -> consolidation is a near-no-op (no LLM).
  * fail-safe: persist / LLM / lease error -> get_representation still returns
    the on-demand result; no raise; lease released.
  * leakage: the persisted profile for (observer, subject) contains only
    observer-scoped content (no other-observer token).
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

from memorylayer_saas.services.representation.default import EnterpriseRepresentationService
from memorylayer_saas.services.representation import consolidation_store
from memorylayer_saas.tasks.representation_consolidation import (
    RepresentationConsolidationTaskHandler,
)

WS = "ws-rep-consol-1"

import logging  # noqa: E402


class _FakeVars:
    """Minimal Variables stub for the task handler.

    ``environ`` returns the flag values (dark gates) per the test; ``get`` /
    ``set`` back ``get_logger`` so the handler can log without a real framework.
    """

    def __init__(self, *, consolidation_enabled: bool = True):
        self._consolidation_enabled = consolidation_enabled
        self._store: dict = {}

    def environ(self, key, default, **kw):
        type_fn = kw.get("type_fn")
        if "CONSOLIDATION_ENABLED" in key:
            return self._consolidation_enabled
        if "ENABLED" in key:  # surface flag
            return True
        return type_fn(default) if type_fn else default

    def get(self, key, default=None):
        if key in self._store:
            return self._store[key]
        return logging.getLogger("test-representation-consolidation")

    def set(self, key, value, **kw):
        self._store[key] = value
        return value


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeRegistry:
    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.members: dict[str, list[Member]] = {}

    def add_entity(self, name: str, entity_id: str, entity_type=EntityType.PERSON) -> Entity:
        now = datetime.now(UTC)
        ent = Entity(
            id=entity_id, workspace_id=WS, entity_type=entity_type,
            canonical_name=name, normalized_name=name.lower(),
            created_at=now, updated_at=now,
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
    """In-memory storage + a settable workspace change watermark."""

    def __init__(self, watermark=("2026-01-01T00:00:00Z", "", 2, 0)):
        self.memories: dict[str, Memory] = {}
        self._watermark = watermark
        self.watermark_raises = False

    def add_memory(self, mem: Memory):
        self.memories[mem.id] = mem

    def set_watermark(self, wm):
        self._watermark = wm

    async def get_memory(self, workspace_id, memory_id, track_access=True):
        return self.memories.get(memory_id)

    async def get_workspace_change_watermark(self, workspace_id):
        if self.watermark_raises:
            raise RuntimeError("watermark store down")
        return self._watermark


class FakeLLM:
    """Recording mock LLM; counts calls so tests can assert 'served without LLM'."""

    def __init__(self, *, content: str | None = None, raises: bool = False):
        self._content = content
        self._raises = raises
        self.requests = []

    async def complete(self, request, profile="default", **_generation_metadata):
        self.requests.append(request)
        if self._raises:
            raise RuntimeError("simulated LLM failure")
        return LLMResponse(
            content=self._content, model="mock",
            prompt_tokens=0, completion_tokens=0, total_tokens=0, finish_reason="stop",
        )

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def prompt_text(self) -> str:
        assert self.requests, "LLM was never called"
        return "\n".join(m.content for m in self.requests[-1].messages)


class FakeTaskService:
    """Records scheduled tasks; never executes them."""

    def __init__(self):
        self.scheduled: list[tuple[str, dict]] = []

    async def schedule_task(self, task_type, payload, delay_seconds=0, priority=5):
        self.scheduled.append((task_type, payload))
        return "task-id"


# Concrete EnterpriseCacheService so the handler's isinstance guard passes.
from memorylayer_saas.services.cache.base import EnterpriseCacheService  # noqa: E402


class LeasingCache(EnterpriseCacheService):
    """In-memory enterprise cache with a counting, ceiling-1 lease.

    ``acquire_lock`` enforces the ceiling-1 lease semantics: the first holder
    wins; a concurrent acquire of an already-held key returns False until it is
    released.
    """

    def __init__(self, *, set_raises: bool = False):
        self.store: dict = {}
        self._locks: set[str] = set()
        self.acquire_calls = 0
        self.set_raises = set_raises

    # CacheService surface.
    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ttl_seconds=None):
        if self.set_raises:
            raise RuntimeError("simulated persist failure")
        self.store[key] = value
        return True

    async def delete(self, key):
        return self.store.pop(key, None) is not None

    async def exists(self, key):
        return key in self.store

    async def clear_prefix(self, prefix):
        return 0

    # EnterpriseCacheService lease surface.
    async def acquire_lock(self, lock_key, holder_id, ttl=30):
        self.acquire_calls += 1
        if lock_key in self._locks:
            return False
        self._locks.add(lock_key)
        return True

    async def release_lock(self, lock_key, holder_id):
        self._locks.discard(lock_key)
        return True


def _mem(mem_id: str, content: str, *, event_offset_days: int = 1,
         status: MemoryStatus = MemoryStatus.ACTIVE) -> Memory:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return Memory(
        id=mem_id, workspace_id=WS, tenant_id="_default", content=content,
        content_hash=f"hash-{mem_id}", type=MemoryType.SEMANTIC, status=status,
        event_time=base + timedelta(days=event_offset_days),
        created_at=base, updated_at=base,
    )


def _self_setup(registry, storage):
    alice = registry.add_entity("Alice", "ent-alice")
    registry.add_member(alice.id, "m1", "self")
    registry.add_member(alice.id, "m2", "self")
    storage.add_memory(_mem("m1", "I shipped the release", event_offset_days=2))
    storage.add_memory(_mem("m2", "I reviewed the PR", event_offset_days=1))
    return alice


_TWO_BELIEFS = (
    '{"beliefs": ['
    '{"statement": "Alice is a productive engineer", "confidence": 0.7, '
    '"support_memory_ids": ["m1", "m2"]},'
    '{"statement": "Alice cares about code quality", "confidence": 0.6, '
    '"support_memory_ids": ["m2"]}'
    ']}'
)


def _make_service(registry, storage, llm, cache, *, task_service=None,
                  consolidation_enabled=True):
    return EnterpriseRepresentationService(
        registry=registry, storage=storage, v=None,
        llm_service=llm, contradiction_service=None,
        cache_service=cache, task_service=task_service,
        consolidation_enabled=consolidation_enabled,
    )


# --------------------------------------------------------------------------- #
# Persist + serve
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestPersistAndServe:
    async def test_consolidate_then_serve_without_llm(self):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        # 1. Consolidate: derives (LLM called once) + persists.
        ok = await svc.consolidate(WS, "Alice", "Alice")
        assert ok is True
        assert llm.call_count == 1
        key = consolidation_store.record_key(WS, "Alice", "Alice")
        assert key in cache.store  # persisted

        # 2. Serve: get_representation returns the maintained record WITHOUT a
        #    second LLM call (still 1), with the persisted beliefs.
        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert llm.call_count == 1  # NOT called again
        assert len(rep.derived_beliefs) == 2
        assert {b.statement for b in rep.derived_beliefs} == {
            "Alice is a productive engineer", "Alice cares about code quality",
        }
        assert {o.memory_id for o in rep.observations} == {"m1", "m2"}
        # Maintained profile is marked derived.
        assert rep.profile is not None and rep.profile.derived is True

    async def test_absent_record_falls_back_to_on_demand(self):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        # No consolidate() first -> no maintained record -> on-demand derivation.
        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert llm.call_count == 1  # derived on-demand
        assert len(rep.derived_beliefs) == 2

    async def test_stale_watermark_falls_back_to_on_demand(self):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        await svc.consolidate(WS, "Alice", "Alice")
        assert llm.call_count == 1

        # Subject changed -> workspace watermark advances -> record is stale.
        storage.set_watermark(("2026-02-02T00:00:00Z", "", 3, 0))
        rep = await svc.get_representation(WS, "Alice", "Alice")
        # Stale -> on-demand derivation (LLM called again).
        assert llm.call_count == 2
        assert len(rep.derived_beliefs) == 2

    async def test_consolidation_disabled_never_serves_maintained(self):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache, consolidation_enabled=False)

        # Even with a (hypothetical) record present, the disabled flag means
        # get_representation always derives on-demand.
        cache.store[consolidation_store.record_key(WS, "Alice", "Alice")] = (
            consolidation_store.serialize_record(
                await svc.get_representation(WS, "Alice", "Alice"),
                ("2026-01-01T00:00:00Z", "", 2, 0),
                limit=20,
            )
        )
        calls_before = llm.call_count
        await svc.get_representation(WS, "Alice", "Alice")
        # Derived again (did not serve the maintained record).
        assert llm.call_count == calls_before + 1


# --------------------------------------------------------------------------- #
# Watermark no-op
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestWatermarkNoOp:
    async def test_unchanged_subject_is_near_noop(self):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        assert await svc.consolidate(WS, "Alice", "Alice") is True
        assert llm.call_count == 1

        # Re-run with an UNCHANGED watermark -> watermark no-op gate skips the
        # derivation (no second LLM call).
        assert await svc.consolidate(WS, "Alice", "Alice") is True
        assert llm.call_count == 1  # NOT re-derived


# --------------------------------------------------------------------------- #
# Fail-safe
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestFailSafe:
    async def test_persist_error_still_serves_on_demand(self):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache(set_raises=True)  # persist always fails
        svc = _make_service(registry, storage, llm, cache)

        # consolidate() swallows the persist error and returns False.
        assert await svc.consolidate(WS, "Alice", "Alice") is False
        # get_representation still works (on-demand), no raise.
        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert len(rep.derived_beliefs) == 2

    async def test_llm_error_during_consolidate_persists_empty_beliefs(self):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(raises=True)  # LLM down
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        # consolidate() derives on-demand (beliefs=[] fail-safe) and persists.
        assert await svc.consolidate(WS, "Alice", "Alice") is True
        rep = await svc.get_representation(WS, "Alice", "Alice")
        # Served maintained record with empty beliefs (OSS contract preserved).
        assert rep.derived_beliefs == []
        assert {o.memory_id for o in rep.observations} == {"m1", "m2"}

    async def test_watermark_uncomputable_serves_on_demand(self):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        await svc.consolidate(WS, "Alice", "Alice")
        # Now the watermark cannot be computed -> serve path treats as stale.
        storage.watermark_raises = True
        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert llm.call_count == 2  # re-derived on-demand
        assert len(rep.derived_beliefs) == 2

    async def test_no_cache_service_serves_on_demand(self):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        svc = _make_service(registry, storage, llm, cache=None)

        # consolidate() is a no-op (no cache) and get_representation derives.
        assert await svc.consolidate(WS, "Alice", "Alice") is False
        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert len(rep.derived_beliefs) == 2


# --------------------------------------------------------------------------- #
# MAJOR-1 — watermark=None short-circuit in consolidate()
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestWatermarkNoneShortCircuit:
    async def test_none_watermark_no_llm_no_persist(self):
        """When the watermark is uncomputable, consolidate() must NOT call the
        LLM and must NOT write a cache record.  Writing a record with
        watermark=None would create a permanently-dead entry (the serve-path
        watermarks_equal check always returns False for a None side), burning
        an LLM call on every subsequent trigger."""
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        # Make the watermark uncomputable.
        storage.watermark_raises = True

        result = await svc.consolidate(WS, "Alice", "Alice")

        assert result is False          # short-circuited
        assert llm.call_count == 0      # LLM NOT called
        key = consolidation_store.record_key(WS, "Alice", "Alice")
        assert key not in cache.store   # nothing persisted

    async def test_none_watermark_get_representation_still_works(self):
        """Even when the watermark is uncomputable and consolidate() is a no-op,
        get_representation falls back to on-demand derivation without raising."""
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        storage.watermark_raises = True

        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert len(rep.derived_beliefs) == 2  # on-demand result returned
        assert llm.call_count == 1            # derived on-demand


# --------------------------------------------------------------------------- #
# MINOR-1 — profile.derived=True unconditional on persist
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestProfileDerivedFlag:
    async def test_derived_true_when_beliefs_present(self):
        """Standard case: maintained profile with beliefs -> derived=True."""
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        await svc.consolidate(WS, "Alice", "Alice")
        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert len(rep.derived_beliefs) == 2
        assert rep.profile is not None
        assert rep.profile.derived is True

    async def test_derived_true_even_with_zero_beliefs(self):
        """LLM-down produces a maintained record with zero beliefs; the profile
        must still be marked derived=True — the record is a maintained artifact
        regardless of belief count."""
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(raises=True)  # LLM down -> beliefs=[]
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        assert await svc.consolidate(WS, "Alice", "Alice") is True
        rep = await svc.get_representation(WS, "Alice", "Alice")
        assert rep.derived_beliefs == []
        assert rep.profile is not None
        assert rep.profile.derived is True  # unconditional


# --------------------------------------------------------------------------- #
# MINOR-2 — limit persisted in blob; mismatch falls through to on-demand
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestLimitGate:
    async def test_caller_limit_exceeds_record_limit_falls_back(self):
        """When the caller requests more observations than the record was derived
        at, the maintained record cannot satisfy the request.  The serve path
        must fall through to on-demand derivation (LLM called again)."""
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        # Consolidate at limit=5 (small canonical limit).
        await svc.consolidate(WS, "Alice", "Alice", limit=5)
        assert llm.call_count == 1

        # Serve with caller limit=5 (equal) -> served from cache.
        rep = await svc.get_representation(WS, "Alice", "Alice", limit=5)
        assert llm.call_count == 1  # NOT re-derived

        # Serve with caller limit=50 (greater) -> on-demand fallback.
        rep = await svc.get_representation(WS, "Alice", "Alice", limit=50)
        assert llm.call_count == 2  # re-derived on-demand
        assert len(rep.derived_beliefs) == 2

    async def test_caller_limit_equal_to_record_limit_served(self):
        """Caller limit == record limit -> served from cache (no LLM)."""
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        await svc.consolidate(WS, "Alice", "Alice", limit=20)
        assert llm.call_count == 1

        rep = await svc.get_representation(WS, "Alice", "Alice", limit=20)
        assert llm.call_count == 1  # served from cache
        assert len(rep.derived_beliefs) == 2

    async def test_caller_limit_less_than_record_limit_served(self):
        """Caller limit < record limit -> the record holds at least as many
        observations as needed -> served from cache."""
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        await svc.consolidate(WS, "Alice", "Alice", limit=20)
        assert llm.call_count == 1

        rep = await svc.get_representation(WS, "Alice", "Alice", limit=5)
        assert llm.call_count == 1  # served from cache
        assert len(rep.derived_beliefs) == 2

    async def test_limit_persisted_in_blob(self):
        """The limit the record was derived at is round-tripped through the
        serialized blob so old records (missing the field) degrade to a miss."""
        import json
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        await svc.consolidate(WS, "Alice", "Alice", limit=15)

        key = consolidation_store.record_key(WS, "Alice", "Alice")
        blob = cache.store[key]
        assert blob["limit"] == 15

        # An old blob without a limit field deserializes with limit=0, causing a
        # mismatch for any real caller limit -> treated as a miss.
        old_blob = dict(blob)
        del old_blob["limit"]
        parsed = consolidation_store.deserialize_record(old_blob)
        # limit defaults to 0 -> any caller_limit > 0 -> fall through.
        assert parsed is not None
        _rep, _wm, rec_limit = parsed
        assert rec_limit == 0


# --------------------------------------------------------------------------- #
# Leakage — persisted profile is observer-scoped only
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestLeakage:
    async def test_persisted_record_only_observer_scoped(self):
        registry, storage = FakeRegistry(), FakeStorage()
        observer = registry.add_entity("Observer", "ent-o")
        subject = registry.add_entity("Subject", "ent-s")
        other = registry.add_entity("Other", "ent-other")

        # M1: O authored + S mentioned -> in scope.
        registry.add_member(observer.id, "M1", "self")
        registry.add_member(subject.id, "M1", "mention")
        # M2: S mentioned, authored by a DIFFERENT observer -> EXCLUDED.
        registry.add_member(subject.id, "M2", "mention")
        registry.add_member(other.id, "M2", "self")
        storage.add_memory(_mem("M1", "SUBJECT_CLOSED_THE_DEAL_TOKEN"))
        storage.add_memory(_mem("M2", "OTHER_OBSERVER_SECRET_TOKEN"))

        llm = FakeLLM(content='{"beliefs": []}')
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        await svc.consolidate(WS, "Observer", "Subject")

        # The persisted blob must contain ONLY the in-scope observation content.
        import json
        blob = cache.store[consolidation_store.record_key(WS, "Observer", "Subject")]
        serialized = json.dumps(blob)
        assert "SUBJECT_CLOSED_THE_DEAL_TOKEN" in serialized
        assert "OTHER_OBSERVER_SECRET_TOKEN" not in serialized

        # And the served maintained record carries only M1.
        rep = await svc.get_representation(WS, "Observer", "Subject")
        assert {o.memory_id for o in rep.observations} == {"M1"}


# --------------------------------------------------------------------------- #
# Coalesce — the task handler's lease collapses concurrent triggers
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestCoalesce:
    async def test_concurrent_triggers_collapse_to_one_run(self, monkeypatch):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache)

        handler = RepresentationConsolidationTaskHandler()

        # Stub Variables: dark flags ON; lease TTL default.
        v = _FakeVars()

        # Resolve the representation + cache services from the handler's
        # get_extension calls by monkeypatching the module-level lookups.
        import memorylayer_saas.tasks.representation_consolidation as mod

        def fake_resolve_rep(_v, _logger):
            return svc

        def fake_resolve_cache(_v, _logger):
            return cache

        monkeypatch.setattr(mod, "_resolve_representation_service", fake_resolve_rep)
        monkeypatch.setattr(mod, "_resolve_cache_service", fake_resolve_cache)

        payload = {"workspace_id": WS, "observer": "Alice", "subject": "Alice"}

        # Hold the lease as if a run is in flight, then fire 3 concurrent triggers.
        lkey = consolidation_store.lease_key(WS, "Alice", "Alice")
        await cache.acquire_lock(lkey, "other-holder")  # simulate in-flight holder

        # All 3 triggers fail to acquire -> coalesce away -> NO derivation.
        for _ in range(3):
            await handler.handle(v, dict(payload))
        assert llm.call_count == 0  # lease held -> all coalesced

        # Release; now exactly one trigger runs the derivation.
        await cache.release_lock(lkey, "other-holder")
        await handler.handle(v, dict(payload))
        assert llm.call_count == 1  # ONE derivation
        # And the maintained record is persisted.
        assert consolidation_store.record_key(WS, "Alice", "Alice") in cache.store

    async def test_lease_released_on_failure(self, monkeypatch):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        # Persist fails -> consolidate raises nothing (returns False), lease must
        # still be released by the handler's try/finally.
        cache = LeasingCache(set_raises=True)
        svc = _make_service(registry, storage, llm, cache)
        handler = RepresentationConsolidationTaskHandler()

        import memorylayer_saas.tasks.representation_consolidation as mod
        monkeypatch.setattr(mod, "_resolve_representation_service", lambda _v, _l: svc)
        monkeypatch.setattr(mod, "_resolve_cache_service", lambda _v, _l: cache)

        payload = {"workspace_id": WS, "observer": "Alice", "subject": "Alice"}
        await handler.handle(_FakeVars(), dict(payload))

        # Lease must be free afterward (a second acquire succeeds).
        lkey = consolidation_store.lease_key(WS, "Alice", "Alice")
        assert await cache.acquire_lock(lkey, "probe") is True

    async def test_dark_gate_makes_handler_noop(self, monkeypatch):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        svc = _make_service(registry, storage, llm, cache, consolidation_enabled=False)
        handler = RepresentationConsolidationTaskHandler()

        import memorylayer_saas.tasks.representation_consolidation as mod
        monkeypatch.setattr(mod, "_resolve_representation_service", lambda _v, _l: svc)
        monkeypatch.setattr(mod, "_resolve_cache_service", lambda _v, _l: cache)

        await handler.handle(
            _FakeVars(consolidation_enabled=False),
            {"workspace_id": WS, "observer": "Alice", "subject": "Alice"},
        )
        assert llm.call_count == 0  # dark -> no derivation


# --------------------------------------------------------------------------- #
# Trigger — enqueue_consolidation
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
class TestTrigger:
    async def test_enqueue_schedules_task_when_enabled(self):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        llm = FakeLLM(content=_TWO_BELIEFS)
        cache = LeasingCache()
        tasks = FakeTaskService()
        svc = _make_service(registry, storage, llm, cache, task_service=tasks)

        await svc.enqueue_consolidation(WS, "Alice", "Alice")
        assert len(tasks.scheduled) == 1
        task_type, payload = tasks.scheduled[0]
        assert task_type == "representation_consolidation"
        assert payload == {"workspace_id": WS, "observer": "Alice", "subject": "Alice"}

    async def test_enqueue_noop_when_disabled(self):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        tasks = FakeTaskService()
        svc = _make_service(
            registry, storage, FakeLLM(content=_TWO_BELIEFS), LeasingCache(),
            task_service=tasks, consolidation_enabled=False,
        )
        await svc.enqueue_consolidation(WS, "Alice", "Alice")
        assert tasks.scheduled == []

    async def test_enqueue_noop_when_no_task_service(self):
        registry, storage = FakeRegistry(), FakeStorage()
        _self_setup(registry, storage)
        svc = _make_service(registry, storage, FakeLLM(content=_TWO_BELIEFS), LeasingCache())
        # No task service wired -> silent no-op (no raise).
        await svc.enqueue_consolidation(WS, "Alice", "Alice")
