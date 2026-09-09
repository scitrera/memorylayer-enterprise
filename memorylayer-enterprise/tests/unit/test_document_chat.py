"""Unit tests for the grounded document-chat endpoint (POST /v1/documents/chat).

Handler logic only — storage / embed clients / authz are mocked. The
image-embed block building and blob I/O are covered in test_image_embed.py.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from memorylayer_server.api.v1.deps import get_auth_service, get_authz_service
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

import memorylayer_saas.api.v1.document_chat as dc
from memorylayer_saas.services.document import EXT_BLOB_STORAGE_SERVICE, EXT_EMBED_SERVER_CLIENT

MODEL = "Qwen/Qwen3.6-27B-FP8"
SLUG = "qwen--qwen3.6-27b-fp8"


def _page(pid, page_no, ws="ws_main", *, has_embeds=True, doc="doc_1"):
    vt = (
        {SLUG: {"embeds_blob_path": f"/blobs/{pid}.pt.zst",
                "grid_blob_path": f"/blobs/{pid}.grid.pt",
                "num_image_tokens": 200}}
        if has_embeds else {}
    )
    return SimpleNamespace(
        id=pid, document_id=doc, workspace_id=ws, page_no=page_no, visual_tokens=vt,
    )


def _completion(text="answer"):
    return {"id": "cmpl-1", "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": text}}]}


def _client(*, storage, blob=None, embed=None, inference=None, authz=None, ctx_ws="ws_main"):
    """Build a TestClient with the route mounted and all collaborators mocked."""
    app = FastAPI()
    app.include_router(dc.router)

    # Filename resolution for page markers is best-effort; default to a doc
    # without a filename so markers fall back to "[Page N]".
    if not hasattr(storage, "get_document") or not isinstance(storage.get_document, AsyncMock):
        storage.get_document = AsyncMock(return_value=SimpleNamespace(filename=None))

    auth = MagicMock()
    auth.build_context = AsyncMock(return_value=SimpleNamespace(workspace_id=ctx_ws))
    authz = authz or MagicMock(require_authorization=AsyncMock())
    inference = inference or MagicMock(chat_completions=AsyncMock(return_value=_completion()))
    blob = blob or MagicMock()
    embed = embed or MagicMock(connect=AsyncMock(), close=AsyncMock(),
                               embed_texts_multivector=AsyncMock(return_value=[{"vectors": [[0.1]]}]))

    ext_map = {
        EXT_STORAGE_BACKEND: storage,
        EXT_BLOB_STORAGE_SERVICE: blob,
        EXT_EMBED_SERVER_CLIENT: embed,
    }

    app.dependency_overrides[get_auth_service] = lambda: auth
    app.dependency_overrides[get_authz_service] = lambda: authz
    app.dependency_overrides[get_variables_dep] = lambda: MagicMock()
    app.dependency_overrides[get_logger] = lambda: __import__("logging").getLogger("test")

    cm = patch.multiple(
        dc,
        get_extension=MagicMock(side_effect=lambda key, v=None: ext_map[key]),
        _get_inference_client=AsyncMock(return_value=inference),
        # block per included page (those whose visual_tokens has SLUG)
        build_image_embeds_content_blocks=AsyncMock(),
        repair_missing_image_embeds=AsyncMock(return_value=0),
        _default_inference_model=MagicMock(return_value=MODEL),
    )
    return TestClient(app), cm, inference, authz


def _block():
    return {"type": "image_embeds",
            "image_embeds": {"image_embeds": "e", "image_grid_thw": "g"}}


def _blocks_for(pages):
    """One image_embeds block per page that has an embeds ref for SLUG."""
    return [
        _block() for p in pages
        if (p.visual_tokens or {}).get(SLUG, {}).get("embeds_blob_path")
    ]


class TestDocumentChat:
    def test_non_streaming_pages_item(self):
        pages = [_page("p1", 0), _page("p2", 1)]
        storage = MagicMock(get_pages_by_ids=AsyncMock(return_value=pages))
        client, cm, inference, _ = _client(storage=storage)
        with cm:
            dc.build_image_embeds_content_blocks.return_value = _blocks_for(pages)
            resp = client.post("/v1/documents/chat", json={
                "context": [{"type": "pages", "page_ids": ["p1", "p2"]}],
                "input": "What are the risks?",
                "model": MODEL,
            })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["completion"]["choices"][0]["message"]["content"] == "answer"
        assert body["context"]["model_slug"] == SLUG
        assert [p["page_id"] for p in body["context"]["pages"]] == ["p1", "p2"]
        assert body["context"]["total_image_embed_tokens"] == 400

        # The payload INTERLEAVES, in page order: [Page marker text][image_embeds]
        # x2, then the question text as the LAST part.
        payload = inference.chat_completions.await_args.args[0]
        content = payload["messages"][0]["content"]
        assert sum(1 for c in content if c["type"] == "image_embeds") == 2
        # text markers precede each image part
        marker_idxs = [i for i, c in enumerate(content) if c["type"] == "text" and c["text"].startswith("[Page ")]
        assert len(marker_idxs) == 2
        for i in marker_idxs:
            assert content[i + 1]["type"] == "image_embeds"
        # question is the last part
        assert content[-1] == {"type": "text", "text": "What are the risks?"}
        assert payload["model"] == MODEL

    def test_missing_embeds_repaired_before_generation(self):
        pages = [_page("p1", 0, has_embeds=True), _page("p2", 1, has_embeds=False)]
        storage = MagicMock(get_pages_by_ids=AsyncMock(return_value=pages))
        client, cm, inference, _ = _client(storage=storage)

        async def _repair(**_kwargs):
            pages[1].visual_tokens = {
                SLUG: {
                    "embeds_blob_path": "/blobs/p2.pt.zst",
                    "grid_blob_path": "/blobs/p2.grid.pt",
                    "num_image_tokens": 200,
                }
            }
            return 1

        with cm:
            dc.repair_missing_image_embeds.side_effect = _repair
            # Hold the mock itself: outside this block the patch is undone and
            # ``dc.repair_missing_image_embeds`` is the real function again, so
            # asserting on the module attribute after the fact checks nothing.
            repair = dc.repair_missing_image_embeds
            dc.build_image_embeds_content_blocks.side_effect = (
                lambda **kwargs: _blocks_for(kwargs["pages"])
            )
            resp = client.post("/v1/documents/chat", json={
                "context": [{"type": "pages", "page_ids": ["p1", "p2"]}],
                "input": "q", "model": MODEL,
            })
        assert resp.status_code == 200, resp.text
        repair.assert_awaited_once()
        pages_out = {p["page_id"]: p for p in resp.json()["context"]["pages"]}
        assert pages_out["p1"]["included"] is True
        assert pages_out["p2"]["included"] is True
        assert pages_out["p2"]["skip_reason"] is None
        content = inference.chat_completions.await_args.args[0]["messages"][0]["content"]
        assert sum(1 for c in content if c["type"] == "image_embeds") == 2

    def test_unrepaired_missing_embeds_returns_409(self):
        pages = [_page("p1", 0, has_embeds=False)]
        storage = MagicMock(get_pages_by_ids=AsyncMock(return_value=pages))
        client, cm, inference, _ = _client(storage=storage)
        with cm:
            resp = client.post("/v1/documents/chat", json={
                "context": [{"type": "pages", "page_ids": ["p1"]}],
                "input": "q", "model": MODEL,
            })
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["missing_page_ids"] == ["p1"]
        assert detail["repair_attempted"] is True
        inference.chat_completions.assert_not_awaited()

    def test_multi_workspace_authorizes_each(self):
        hits = [(_page("p1", 0, ws="ws_a"), 0.9)]
        storage = MagicMock(
            search_pages_by_maxsim=AsyncMock(return_value=hits),
        )
        client, cm, inference, authz = _client(storage=storage)
        with cm:
            dc.build_image_embeds_content_blocks.return_value = [_block()]
            resp = client.post("/v1/documents/chat", json={
                "context": [{"type": "maxsim", "query": "risk", "top_k": 3,
                             "workspace_ids": ["ws_a", "ws_b"]}],
                "input": "q", "model": MODEL,
            })
        assert resp.status_code == 200, resp.text
        authorized = {c.kwargs["workspace_id"] for c in authz.require_authorization.await_args_list}
        assert authorized == {"ws_a", "ws_b"}

    def test_default_model_used_when_omitted(self):
        pages = [_page("p1", 0)]
        storage = MagicMock(get_pages_by_ids=AsyncMock(return_value=pages))
        client, cm, inference, _ = _client(storage=storage)
        with cm:
            dc.build_image_embeds_content_blocks.return_value = _blocks_for(pages)
            resp = client.post("/v1/documents/chat", json={
                "context": [{"type": "pages", "page_ids": ["p1"]}],
                "input": "q",  # no model
            })
        assert resp.status_code == 200, resp.text
        assert resp.json()["context"]["model"] == MODEL

    def test_generation_extra_cannot_override_grounded_payload(self):
        pages = [_page("p1", 0)]
        storage = MagicMock(get_pages_by_ids=AsyncMock(return_value=pages))
        client, cm, inference, _ = _client(storage=storage)
        with cm:
            dc.build_image_embeds_content_blocks.return_value = _blocks_for(pages)
            resp = client.post("/v1/documents/chat", json={
                "context": [{"type": "pages", "page_ids": ["p1"]}],
                "input": "q",
                "model": MODEL,
                "generation": {"extra": {"messages": [], "model": "attacker/model"}},
            })
        assert resp.status_code == 400
        inference.chat_completions.assert_not_awaited()

    def test_streaming_emits_context_event_first(self):
        pages = [_page("p1", 0)]
        storage = MagicMock(get_pages_by_ids=AsyncMock(return_value=pages))

        async def _sse():
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        inference = MagicMock(chat_completions=AsyncMock(return_value=_sse()))
        client, cm, inference, _ = _client(storage=storage, inference=inference)
        with cm:
            dc.build_image_embeds_content_blocks.return_value = _blocks_for(pages)
            resp = client.post("/v1/documents/chat", json={
                "context": [{"type": "pages", "page_ids": ["p1"]}],
                "input": "q", "model": MODEL, "stream": True,
            })
        assert resp.status_code == 200, resp.text
        text = resp.text
        assert text.startswith("event: context\n")
        assert '"model_slug": "qwen--qwen3.6-27b-fp8"' in text
        assert "[DONE]" in text  # upstream chunks proxied after the descriptor


class TestStatefulContext:
    def setup_method(self):
        # The grounded-context store is process-global; isolate each test.
        dc._GROUNDED_CONTEXTS.clear()

    def teardown_method(self):
        dc._GROUNDED_CONTEXTS.clear()

    def test_new_conversation_returns_and_persists_context_id(self):
        pages = [_page("p1", 0), _page("p2", 1)]
        storage = MagicMock(get_pages_by_ids=AsyncMock(return_value=pages))
        client, cm, inference, _ = _client(storage=storage)
        with cm:
            dc.build_image_embeds_content_blocks.return_value = _blocks_for(pages)
            resp = client.post("/v1/documents/chat", json={
                "context": [{"type": "pages", "page_ids": ["p1", "p2"]}],
                "input": "What are the risks?", "model": MODEL,
            })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        cid = body["context_id"]
        assert cid.startswith("gctx_")

        # An entry was persisted with the included page ids + first history turn.
        entry = dc._GROUNDED_CONTEXTS[cid]
        assert entry["model"] == MODEL
        assert entry["model_slug"] == SLUG
        assert entry["page_ids"] == ["p1", "p2"]
        assert entry["workspaces"] == ["ws_main"]
        assert entry["history"] == [
            {"question": "What are the risks?", "answer": "answer"},
        ]

    def test_resume_reresolves_pages_and_appends_history(self):
        # Seed a stored context as if a prior turn had created it.
        dc._GROUNDED_CONTEXTS["gctx_seed"] = {
            "model": MODEL, "model_slug": SLUG,
            "page_ids": ["p1", "p2"], "workspaces": ["ws_main"],
            "history": [{"question": "First question?", "answer": "first answer"}],
        }
        pages = [_page("p1", 0), _page("p2", 1)]
        get_by_ids = AsyncMock(return_value=pages)
        storage = MagicMock(get_pages_by_ids=get_by_ids)
        inference = MagicMock(chat_completions=AsyncMock(return_value=_completion("second answer")))
        client, cm, inference, _ = _client(storage=storage, inference=inference)
        with cm:
            dc.build_image_embeds_content_blocks.return_value = _blocks_for(pages)
            resp = client.post("/v1/documents/chat", json={
                "context_id": "gctx_seed", "input": "Second question?",
            })
        assert resp.status_code == 200, resp.text
        assert resp.json()["context_id"] == "gctx_seed"

        # Re-resolution used the stored page ids in stored order.
        get_by_ids.assert_awaited_once_with(["p1", "p2"])

        messages = inference.chat_completions.await_args.args[0]["messages"]
        # FIRST user message carries the grounded image_embeds blocks + the
        # original (history[0]) question.
        first = messages[0]
        assert first["role"] == "user"
        assert sum(1 for c in first["content"] if c["type"] == "image_embeds") == 2
        assert first["content"][-1] == {"type": "text", "text": "First question?"}
        # The assistant's prior answer is replayed.
        assert {"role": "assistant", "content": "first answer"} in messages
        # LAST message is the new question.
        assert messages[-1] == {
            "role": "user", "content": [{"type": "text", "text": "Second question?"}],
        }

        # History grew to include the new turn.
        assert dc._GROUNDED_CONTEXTS["gctx_seed"]["history"] == [
            {"question": "First question?", "answer": "first answer"},
            {"question": "Second question?", "answer": "second answer"},
        ]

    def test_400_when_neither_context_nor_context_id(self):
        storage = MagicMock(get_pages_by_ids=AsyncMock(return_value=[]))
        client, cm, inference, _ = _client(storage=storage)
        with cm:
            resp = client.post("/v1/documents/chat", json={"input": "q"})
        assert resp.status_code == 400
        inference.chat_completions.assert_not_awaited()

    def test_400_when_both_context_and_context_id(self):
        storage = MagicMock(get_pages_by_ids=AsyncMock(return_value=[]))
        client, cm, inference, _ = _client(storage=storage)
        with cm:
            resp = client.post("/v1/documents/chat", json={
                "context": [{"type": "pages", "page_ids": ["p1"]}],
                "context_id": "gctx_x", "input": "q",
            })
        assert resp.status_code == 400
        inference.chat_completions.assert_not_awaited()

    def test_404_for_unknown_context_id(self):
        storage = MagicMock(get_pages_by_ids=AsyncMock(return_value=[]))
        client, cm, inference, _ = _client(storage=storage)
        with cm:
            resp = client.post("/v1/documents/chat", json={
                "context_id": "gctx_missing", "input": "q",
            })
        assert resp.status_code == 404
        inference.chat_completions.assert_not_awaited()
