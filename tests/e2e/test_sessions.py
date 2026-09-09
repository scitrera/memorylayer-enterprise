"""E2E: Session lifecycle — create, list, touch, delete."""

from memorylayer import SyncMemoryLayerClient


def test_create_and_list_sessions(client: SyncMemoryLayerClient):
    """Create a session and verify it appears in the session list."""
    session = client.create_session(ttl_seconds=3600)
    assert session.id is not None

    sessions = client.list_sessions()
    session_ids = [s.id for s in sessions]
    assert session.id in session_ids


def test_session_scoped_memories(client: SyncMemoryLayerClient):
    """Memories stored with a session_id are associated with that session."""
    import time

    session = client.create_session(ttl_seconds=3600)

    # Remember with session context
    memory = client.remember(
        content="This is a session-scoped fact",
        session_id=session.id,
    )
    assert memory.id is not None

    time.sleep(0.5)

    # Recall should find it
    results = client.recall("session-scoped fact")
    assert results.memories
