"""Document phases must not close the clients they borrow.

``document_embed`` and ``document_transcribe`` fetch their clients with
``get_extension`` -- process-wide singletons connected by the framework at
startup and shared by every concurrent task. Both used to wrap their work in
``try/finally: client.close()``.

With one task at a time that merely churned connections. Once the task lanes
let several document phases run at once it became a correctness bug: the first
task to finish closed the shared client out from under its siblings' in-flight
requests (surfacing as ``httpx.ReadError`` with an empty message, against
servers logging clean 200s) and left it unusable for everything afterwards,
since nothing reconnects it -- ``'NoneType' object has no attribute 'post'``
and ``EmbedServerClient.connect() not called``.

These tests pin the source, not the behaviour: reproducing the race needs the
real task bodies and their whole dependency graph, whereas the invariant --
"a borrowed client is never closed here" -- is exactly what regressed and is
cheap to state.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TASKS = Path(__file__).resolve().parents[2] / "src" / "memorylayer_saas" / "tasks"

# (module, the name bound to the shared extension in that module's handler)
_BORROWERS = [
    ("document_embed.py", "embed_client"),
    ("document_transcribe.py", "transcription"),
]


def _closes_of(source: str, name: str) -> list[int]:
    """Line numbers of `await <name>.close()` in the module."""
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "close"
            and isinstance(func.value, ast.Name)
            and func.value.id == name
        ):
            found.append(node.lineno)
    return found


@pytest.mark.parametrize("module,name", _BORROWERS)
def test_a_borrowed_client_is_never_closed_by_the_task(module, name):
    source = (_TASKS / module).read_text()
    closes = _closes_of(source, name)

    assert not closes, (
        f"{module} closes the shared '{name}' extension at line(s) {closes}. "
        "It is a process-wide singleton shared by concurrent tasks -- closing it "
        "kills siblings' in-flight requests and leaves it unusable, since nothing "
        "reconnects it. Lifecycle belongs to the framework."
    )


@pytest.mark.parametrize("module,name", _BORROWERS)
def test_the_client_is_still_borrowed_not_constructed(module, name):
    # Guards the other direction: if a future change makes the task OWN a client
    # (constructing it per task), closing becomes correct and this whole
    # invariant needs revisiting rather than silently passing.
    source = (_TASKS / module).read_text()
    assert f"{name} = get_extension(" in source, (
        f"{module} no longer binds '{name}' via get_extension; if the task now "
        "owns its client, revisit test_a_borrowed_client_is_never_closed_by_the_task."
    )


@pytest.mark.parametrize("module,name", _BORROWERS)
def test_connect_is_retained_so_a_closed_client_self_heals(module, name):
    # connect() is a no-op when already connected, and recreates a client that
    # something else closed. Removing it alongside close() would leave the phase
    # unable to recover from a client torn down elsewhere.
    source = (_TASKS / module).read_text()
    assert f"await {name}.connect()" in source
