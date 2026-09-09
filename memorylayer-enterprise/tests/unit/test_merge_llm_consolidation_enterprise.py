"""Enterprise Phase 1c: LLM update-vs-add consolidation on the write path.

These tests cover the ENTERPRISE override of ``_merge_memories``:

  * merge_llm_enabled OFF (default) -> byte-identical to the OSS deterministic
    merge; the LLM is NEVER consulted.
  * LLM verdict ``update`` -> the reconciled ``merged_content`` is persisted and a
    single update-history entry (previous_content_hash + reason) is appended.
  * LLM verdict ``add`` -> deterministic fallback (content = new_content, no
    history entry from the LLM path).
  * LLM raising -> deterministic fallback, never raises.

Everything is mocked (no network, no Postgres). The fake storage records the
``update_memory`` calls and applies partial updates so the override's
persistence flow is exercised end to end.
"""

import json

import pytest
from memorylayer_server.models.memory import Memory, MemoryType
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
    """Captures the last request and returns a canned content (or raises)."""

    def __init__(self, content=None, raises=False):
        self._content = content
        self._raises = raises
        self.last_request = None
        self.last_profile = None
        self.call_count = 0

    async def complete(self, request, profile="default", **_generation_metadata):
        self.call_count += 1
        self.last_request = request
        self.last_profile = profile
        if self._raises:
            raise RuntimeError("llm down")

        class _Resp:
            content = self._content

        return _Resp()


class _FakeEmbedding:
    """Deterministic no-op embedding."""

    async def embed(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]


class _FakeStorage:
    """Holds a single memory and applies partial ``update_memory`` writes."""

    def __init__(self, memory: Memory):
        self._memory = memory
        self.update_calls: list[dict] = []

    async def update_memory(self, workspace_id: str, memory_id: str, **fields) -> Memory:
        self.update_calls.append({"memory_id": memory_id, **fields})
        update = {k: v for k, v in fields.items() if v is not None and k != "embedding"}
        if "content" in update:
            # Simulate the storage recomputing the dedup hash on a content change.
            update["content_hash"] = "hash:" + update["content"]
        self._memory = self._memory.model_copy(update=update)
        return self._memory

    async def reindex_memory(self, workspace_id: str, memory_id: str) -> None:
        return None


def _existing_memory() -> Memory:
    return Memory(
        id="mem-1",
        workspace_id="ws-1",
        tenant_id="tenant-1",
        content="Alice works at Acme Corp.",
        content_hash="hash-existing",
        type=MemoryType.SEMANTIC,
        importance=0.5,
        tags=["work"],
        metadata={"source": "manual"},
    )


def _decision_json(action: str, merged_content: str = "", reason: str = "r") -> str:
    return json.dumps(
        {"action": action, "merged_content": merged_content, "reason": reason}
    )


def _make_service(v, llm, storage, *, merge_enabled: bool) -> EnterpriseMemoryService:
    svc = EnterpriseMemoryService(
        v=v,
        storage=storage,
        embedding_service=_FakeEmbedding(),
        llm_service=llm,
    )
    svc.merge_llm_enabled = merge_enabled
    return svc


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_disabled_by_default_never_calls_llm(v):
    """Default OFF: _merge_memories == OSS deterministic merge; LLM untouched."""
    existing = _existing_memory()
    storage = _FakeStorage(existing)
    llm = _MockLLMService(content=_decision_json("update", "MERGED"))
    svc = _make_service(v, llm, storage, merge_enabled=False)
    assert svc.merge_llm_enabled is False  # code default

    updated = await svc._merge_memories(
        "ws-1", existing, "Alice moved to Globex.", ["moved"], {"k": "val"}, 0.6
    )

    # Deterministic behaviour: new content replaces existing; LLM never consulted.
    assert updated.content == "Alice moved to Globex."
    assert llm.call_count == 0
    assert llm.last_request is None
    assert "history" not in updated.metadata
    # Deterministic provenance is still recorded.
    assert updated.metadata["merged_from"] == "hash-existing"


@pytest.mark.asyncio
async def test_merge_llm_update_persists_reconciled_content_and_history(v):
    """LLM 'update' -> merged_content persisted + one history entry recorded."""
    existing = _existing_memory()
    storage = _FakeStorage(existing)
    llm = _MockLLMService(
        content=_decision_json("update", "Alice works at Globex.", "supersedes")
    )
    svc = _make_service(v, llm, storage, merge_enabled=True)

    updated = await svc._merge_memories(
        "ws-1", existing, "Alice now works at Globex.", ["moved"], {"k": "val"}, 0.6
    )

    # Reconciled LLM content — NOT the raw new_content — was persisted.
    assert updated.content == "Alice works at Globex."
    assert llm.call_count == 1
    assert llm.last_profile == "merge"

    # Exactly one update-history entry with previous hash + reason.
    history = updated.metadata["history"]
    assert isinstance(history, list) and len(history) == 1
    entry = history[0]
    assert entry["previous_content_hash"] == "hash-existing"
    assert entry["reason"] == "supersedes"
    assert entry["action"] == "llm_merge"
    assert entry["similarity"] is None
    assert "at" in entry and isinstance(entry["at"], str)

    # Two storage writes: the deterministic merge, then the history metadata patch.
    assert len(storage.update_calls) == 2
    assert storage.update_calls[0]["content"] == "Alice works at Globex."
    assert "content" not in storage.update_calls[1]  # metadata-only patch


@pytest.mark.asyncio
async def test_merge_llm_add_falls_back_to_deterministic(v):
    """LLM 'add' (distinct) -> deterministic merge; no history entry."""
    existing = _existing_memory()
    storage = _FakeStorage(existing)
    llm = _MockLLMService(content=_decision_json("add", "", "distinct"))
    svc = _make_service(v, llm, storage, merge_enabled=True)

    updated = await svc._merge_memories(
        "ws-1", existing, "Alice enjoys hiking.", ["hobby"], {"k": "val"}, 0.6
    )

    # Deterministic path: raw new_content persisted, no LLM-merge history.
    assert updated.content == "Alice enjoys hiking."
    assert llm.call_count == 1  # the decision call happened
    assert "history" not in updated.metadata
    assert updated.metadata["merged_from"] == "hash-existing"
    assert len(storage.update_calls) == 1  # only the deterministic write


@pytest.mark.asyncio
async def test_merge_llm_error_falls_back_and_does_not_raise(v):
    """LLM raising -> deterministic fallback; never raises."""
    existing = _existing_memory()
    storage = _FakeStorage(existing)
    llm = _MockLLMService(raises=True)
    svc = _make_service(v, llm, storage, merge_enabled=True)

    updated = await svc._merge_memories(
        "ws-1", existing, "Alice moved to Globex.", ["moved"], {}, 0.6
    )

    assert updated.content == "Alice moved to Globex."
    assert "history" not in updated.metadata
    assert len(storage.update_calls) == 1


@pytest.mark.asyncio
async def test_merge_llm_empty_merged_content_falls_back(v):
    """LLM 'update' with empty merged_content -> deterministic fallback."""
    existing = _existing_memory()
    storage = _FakeStorage(existing)
    llm = _MockLLMService(content=_decision_json("update", "   ", "oops"))
    svc = _make_service(v, llm, storage, merge_enabled=True)

    updated = await svc._merge_memories(
        "ws-1", existing, "Alice moved to Globex.", ["moved"], {}, 0.6
    )

    assert updated.content == "Alice moved to Globex."
    assert "history" not in updated.metadata
    assert len(storage.update_calls) == 1


@pytest.mark.asyncio
async def test_merge_no_llm_service_falls_back(v):
    """merge_enabled but no LLM service wired -> deterministic merge, no call."""
    existing = _existing_memory()
    storage = _FakeStorage(existing)
    svc = _make_service(v, None, storage, merge_enabled=True)

    updated = await svc._merge_memories(
        "ws-1", existing, "Alice moved to Globex.", ["moved"], {}, 0.6
    )

    assert updated.content == "Alice moved to Globex."
    assert "history" not in updated.metadata
    assert len(storage.update_calls) == 1
