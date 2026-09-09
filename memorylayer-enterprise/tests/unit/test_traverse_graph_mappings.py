"""Unit test for PostgreSQLBackend.traverse_graph row mapping.

Regression for the bug where traverse_graph indexed a SQLAlchemy 2.0 Core
``Row`` by column name (``row["path"]``, ``row["id"]``, ...). A plain Row does
NOT support string subscripting and raises TypeError, so traverse_graph threw
on any non-empty result. The fix switches to ``result.mappings().all()`` which
yields dict-like ``RowMapping`` objects that DO support string subscripting.

This test uses a mocked session whose ``execute(...).mappings().all()`` returns
a fake dict row, and whose ``execute(...).fetchall()`` returns a Row-like object
that raises on string subscripting. It asserts traverse_graph returns >= 1
GraphPath without error, proving the code reads via ``.mappings()`` (not
``.fetchall()``). A live-PG integration test is gated elsewhere and is not
reachable in this environment.
"""

from contextlib import asynccontextmanager

import pytest

from memorylayer_saas.storage.postgresql import PostgreSQLBackend


class _RowLike:
    """Mimics a SQLAlchemy Core Row: NO string subscripting (raises TypeError)."""

    def __getitem__(self, key):
        if isinstance(key, str):
            raise TypeError("Row indices must be integers, not str")
        raise IndexError(key)


class _FakeResult:
    def __init__(self, mapping_rows):
        self._mapping_rows = mapping_rows

    def mappings(self):
        return self

    def all(self):
        return self._mapping_rows

    def fetchall(self):
        # If the code (incorrectly) used fetchall + string subscripting, the
        # downstream row["..."] access would raise TypeError -> test fails.
        return [_RowLike() for _ in self._mapping_rows]


class _FakeSession:
    def __init__(self, result):
        self._result = result
        self.calls = []

    async def execute(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._result


@pytest.mark.asyncio
async def test_traverse_graph_uses_mappings_not_row_subscript():
    backend = object.__new__(PostgreSQLBackend)

    # One dict-like row mirroring the CTE's selected columns.
    mapping_rows = [
        {
            "id": "assoc_1",
            "source_id": "mem_a",
            "target_id": "mem_b",
            "relationship": "relates_to",
            "strength": 0.9,
            "metadata": {"k": "v"},
            "depth": 1,
            "path": ["mem_a", "mem_b"],
        }
    ]
    fake_result = _FakeResult(mapping_rows)

    @asynccontextmanager
    async def fake_session_factory():
        yield _FakeSession(fake_result)

    backend._session_factory = fake_session_factory

    result = await backend.traverse_graph(
        workspace_id="ws_1",
        start_id="mem_a",
        max_depth=3,
    )

    assert result.total_paths == 1
    assert len(result.paths) == 1
    path = result.paths[0]
    assert path.nodes == ["mem_a", "mem_b"]
    assert path.depth == 1
    assert path.total_strength == 0.9
    assert len(path.edges) == 1
    edge = path.edges[0]
    assert edge.id == "assoc_1"
    assert edge.source_id == "mem_a"
    assert edge.target_id == "mem_b"
    assert edge.relationship == "relates_to"
    # start_id plus traversed nodes are all present.
    assert set(result.unique_nodes) == {"mem_a", "mem_b"}


@pytest.mark.parametrize(
    ("direction", "base_clause", "recursive_clause"),
    [
        (
            "outgoing",
            "AND source_id = :start_id",
            "INNER JOIN graph_traverse gt ON a.source_id = gt.current_node",
        ),
        (
            "incoming",
            "AND target_id = :start_id",
            "INNER JOIN graph_traverse gt ON a.target_id = gt.current_node",
        ),
        (
            "both",
            "AND (source_id = :start_id OR target_id = :start_id)",
            "INNER JOIN graph_traverse gt ON (a.source_id = gt.current_node OR a.target_id = gt.current_node)",
        ),
    ],
)
@pytest.mark.asyncio
async def test_traverse_graph_sql_respects_direction(direction, base_clause, recursive_clause):
    backend = object.__new__(PostgreSQLBackend)
    session = _FakeSession(_FakeResult([]))

    @asynccontextmanager
    async def fake_session_factory():
        yield session

    backend._session_factory = fake_session_factory

    await backend.traverse_graph(
        workspace_id="ws_1",
        start_id="mem_a",
        max_depth=2,
        relationships=["contains", "imports"],
        direction=direction,
    )

    statement_args, statement_kwargs = session.calls[0]
    sql = " ".join(str(statement_args[0]).split())
    params = statement_args[1] if len(statement_args) > 1 else statement_kwargs["params"]

    assert base_clause in sql
    assert recursive_clause in sql
    assert "AND relationship IN (:rel_0, :rel_1)" in sql
    assert "AND a.relationship IN (:rel_0, :rel_1)" in sql
    assert "CAST(:start_id AS text)" in sql
    assert "= ANY(gt.path)" in sql
    assert params["rel_0"] == "contains"
    assert params["rel_1"] == "imports"
