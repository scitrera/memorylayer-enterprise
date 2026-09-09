# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the KB entity-article "Related Entities" wiring.

Exercises the pure ranking/resolution seam (``_top_related``) that turns the
whole-batch co-occurrence map into per-article Connections, WITHOUT a live
registry or LLM: only neighbors that themselves have an article survive, ranked
by shared-memory count.
"""

from memorylayer_saas.services.knowledgebase.enterprise import EnterpriseKnowledgebaseService

# e1 co-occurs with e2 (3), e3 (1), and eX (5 — but eX has no article)
_COOC = {
    "e1": {"e2": 3, "e3": 1, "eX": 5},
    "e2": {"e1": 3},
    "e3": {"e1": 1},
}
_NAMES = {"e1": "Acme Corp", "e2": "Beta LLC", "e3": "Gamma Inc"}  # eX absent on purpose


def test_top_related_ranks_by_shared_and_resolves_names():
    out = EnterpriseKnowledgebaseService._top_related("e1", _COOC, _NAMES)
    # eX dropped (no article); rest ranked by shared desc
    assert out == [
        {"id": "e2", "name": "Beta LLC", "shared": 3},
        {"id": "e3", "name": "Gamma Inc", "shared": 1},
    ]


def test_top_related_carries_the_neighbor_id():
    """The id — not the name — is what resolves a Connections link to an article.

    Rendering slugs the *name* for display but looks the target up by id, so a row
    without an id would render as unlinked plain text.
    """
    out = EnterpriseKnowledgebaseService._top_related("e1", _COOC, _NAMES)
    assert [r["id"] for r in out] == ["e2", "e3"]


def test_top_related_honors_limit():
    out = EnterpriseKnowledgebaseService._top_related("e1", _COOC, _NAMES, limit=1)
    assert out == [{"id": "e2", "name": "Beta LLC", "shared": 3}]


def test_top_related_empty_for_unknown_or_isolated():
    assert EnterpriseKnowledgebaseService._top_related("zzz", _COOC, _NAMES) == []
    assert EnterpriseKnowledgebaseService._top_related("e1", {}, _NAMES) == []
