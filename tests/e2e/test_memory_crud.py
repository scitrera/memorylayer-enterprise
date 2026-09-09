# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""E2E: Memory CRUD operations — remember, recall, get, forget."""

import time

from memorylayer import SyncMemoryLayerClient
from memorylayer.types import MemoryType


def test_remember_and_recall(client: SyncMemoryLayerClient):
    """Store a memory and retrieve it via semantic search."""
    memory = client.remember(
        content="The project uses Python 3.12 and FastAPI",
        type=MemoryType.SEMANTIC,
        importance=0.8,
    )
    assert memory.id is not None
    assert memory.content == "The project uses Python 3.12 and FastAPI"

    # Small delay for embedding indexing
    time.sleep(0.5)

    results = client.recall("what programming language does the project use")
    assert results.memories, "Expected at least one memory in recall results"
    contents = [m.content for m in results.memories]
    assert any("Python" in c for c in contents)


def test_remember_and_get(client: SyncMemoryLayerClient):
    """Store a memory and retrieve it by ID."""
    memory = client.remember(
        content="The database is PostgreSQL with pgvector",
        type=MemoryType.SEMANTIC,
    )
    fetched = client.get_memory(memory.id)
    assert fetched.id == memory.id
    assert fetched.content == memory.content


def test_forget_soft(client: SyncMemoryLayerClient):
    """Soft-delete a memory (mark as forgotten, not physically removed)."""
    memory = client.remember(content="Temporary note to forget")
    client.forget(memory.id, hard=False)

    # After soft forget, recall should not return it
    time.sleep(0.5)
    results = client.recall("Temporary note to forget")
    found_ids = [m.id for m in results.memories]
    assert memory.id not in found_ids


def test_forget_hard(client: SyncMemoryLayerClient):
    """Hard-delete a memory (physically removed)."""
    memory = client.remember(content="Secret to hard delete")
    client.forget(memory.id, hard=True)

    # After hard forget, get_memory should fail
    import pytest
    with pytest.raises(Exception):
        client.get_memory(memory.id)


def test_remember_multiple_and_recall(client: SyncMemoryLayerClient):
    """Store multiple memories and verify recall returns relevant ones."""
    client.remember(content="Alice prefers dark mode in her IDE")
    client.remember(content="Bob likes to use vim for editing")
    client.remember(content="The deployment target is Kubernetes")

    time.sleep(0.5)

    results = client.recall("editor preferences")
    assert results.memories, "Expected memories about editor preferences"


def test_remember_with_metadata(client: SyncMemoryLayerClient):
    """Store a memory with custom metadata and verify it persists."""
    memory = client.remember(
        content="Meeting notes from standup",
        type=MemoryType.EPISODIC,
        metadata={"source": "standup", "date": "2026-03-16"},
    )
    fetched = client.get_memory(memory.id)
    assert fetched.metadata.get("source") == "standup"
    assert fetched.metadata.get("date") == "2026-03-16"
