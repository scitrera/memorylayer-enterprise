# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the proprietary TokenaryClient adapter.

Mocked-HTTP tests of the wire-shape translation in BOTH directions:

- multivec: a fake ``/embed/multi`` base64 ``.npy`` response decodes to the same
  ``{"vectors", "num_vectors"}`` structure the parent ``EmbedServerClient``
  returns.
- score: caller-provided float vectors are encoded to base64 ``.npy`` in the
  outgoing ``/score`` request body, and the flat ``scores`` response parses back
  into the parent's ``[{"index", "score"}]`` structure.
- image multivec: ``images_b64`` become ``messages`` groups with ``image_url``
  data URIs.
- single-vec: ``/v1/embeddings`` with the ``dimensions`` field, OpenAI shape.
- chat: a nested ``image_embeds`` content block is flattened to tokenary's flat
  shape.
- plugin selection: with ``MEMORYLAYER_EMBED_SERVER_SERVICE=tokenary`` the
  ``EXT_EMBED_SERVER_CLIENT`` extension resolves to a ``TokenaryClient`` (and to
  the default ``EmbedServerClient`` otherwise).
"""
import base64
import io
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from memorylayer_saas.services.document.tokenary_client import (
    TokenaryClient,
    _npy_b64_to_vectors,
    _vectors_to_npy_b64,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def client(mock_logger):
    """TokenaryClient whose per-concern children have mocked request_json."""
    c = TokenaryClient(
        base_url="http://tokenary:8000",
        timeout=30.0,
        logger=mock_logger,
        textvec_url="http://textvec:8000",
        multivec_url="http://multivec:8000",
        score_url="http://score:8000",
        visualtok_url="http://visualtok:8000",
        embedding_dimensions=1024,
    )
    c._textvec.request_json = AsyncMock()
    c._multivec.request_json = AsyncMock()
    c._score.request_json = AsyncMock()
    c._visualtok.request_json = AsyncMock()
    return c


def _npy_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# .npy encode/decode round-trip
# ---------------------------------------------------------------------------

class TestNpyRoundTrip:
    def test_vectors_npy_roundtrip(self):
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        decoded = _npy_b64_to_vectors(_vectors_to_npy_b64(vectors))
        assert np.allclose(decoded, vectors)
        assert len(decoded) == 2 and len(decoded[0]) == 3


# ---------------------------------------------------------------------------
# embed_texts (single-vec) -> /v1/embeddings
# ---------------------------------------------------------------------------

class TestEmbedTexts:
    @pytest.mark.asyncio
    async def test_embed_texts_passes_dimensions(self, client):
        client._textvec.request_json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2], "index": 0},
                {"embedding": [0.3, 0.4], "index": 1},
            ]
        }
        result = await client.embed_texts(["a", "b"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        method, path, payload = client._textvec.request_json.call_args.args
        assert (method, path) == ("POST", "/v1/embeddings")
        assert payload == {"input": ["a", "b"], "dimensions": 1024}

    @pytest.mark.asyncio
    async def test_embed_texts_sorts_by_index(self, client):
        client._textvec.request_json.return_value = {
            "data": [
                {"embedding": [0.9], "index": 1},
                {"embedding": [0.1], "index": 0},
            ]
        }
        result = await client.embed_texts(["a", "b"])
        assert result == [[0.1], [0.9]]


# ---------------------------------------------------------------------------
# embed_texts_multivector -> /embed/multi (base64 .npy decode)
# ---------------------------------------------------------------------------

class TestEmbedTextsMultivector:
    @pytest.mark.asyncio
    async def test_decodes_npy_to_parent_structure(self, client):
        v0 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32")
        v1 = np.array([[0.5, 0.6]], dtype="float32")
        client._multivec.request_json.return_value = {
            "data": [
                {"index": 0, "multivec": _npy_b64(v0), "num_tokens": 2},
                {"index": 1, "multivec": _npy_b64(v1), "num_tokens": 1},
            ]
        }
        result = await client.embed_texts_multivector(["doc one", "doc two"])

        assert len(result) == 2
        # Same structure the parent EmbedServerClient returns.
        assert set(result[0].keys()) == {"vectors", "num_vectors"}
        assert np.allclose(result[0]["vectors"], v0.tolist())
        assert result[0]["num_vectors"] == 2
        assert np.allclose(result[1]["vectors"], v1.tolist())
        assert result[1]["num_vectors"] == 1

        method, path, payload = client._multivec.request_json.call_args.args
        assert (method, path) == ("POST", "/embed/multi")
        assert payload == {"input": ["doc one", "doc two"], "pool_factor": 1}

    @pytest.mark.asyncio
    async def test_sorts_by_index(self, client):
        v = np.array([[1.0]], dtype="float32")
        client._multivec.request_json.return_value = {
            "data": [
                {"index": 1, "multivec": _npy_b64(v * 9), "num_tokens": 1},
                {"index": 0, "multivec": _npy_b64(v), "num_tokens": 1},
            ]
        }
        result = await client.embed_texts_multivector(["a", "b"])
        assert np.allclose(result[0]["vectors"], [[1.0]])
        assert np.allclose(result[1]["vectors"], [[9.0]])


# ---------------------------------------------------------------------------
# embed_images_multivector -> /embed/multi messages groups
# ---------------------------------------------------------------------------

class TestEmbedImagesMultivector:
    @pytest.mark.asyncio
    async def test_images_become_message_groups_with_data_uris(self, client):
        v = np.array([[0.1, 0.2]], dtype="float32")
        client._multivec.request_json.return_value = {
            "data": [
                {"index": 0, "multivec": _npy_b64(v), "num_tokens": 1},
                {"index": 1, "multivec": _npy_b64(v), "num_tokens": 1},
            ]
        }
        result = await client.embed_images_multivector(["aGVsbG8=", "d29ybGQ="])

        assert len(result) == 2
        assert np.allclose(result[0]["vectors"], v.tolist())

        _, path, payload = client._multivec.request_json.call_args.args
        assert path == "/embed/multi"
        groups = payload["input"]["messages"]
        assert len(groups) == 2
        # Each group is one user message with an image_url data-URI part.
        part = groups[0][0]["content"][0]
        assert part["type"] == "image_url"
        assert part["image_url"]["url"] == "data:image/png;base64,aGVsbG8="
        assert payload["pool_factor"] == 1

    @pytest.mark.asyncio
    async def test_data_uri_passthrough(self, client):
        v = np.array([[0.1]], dtype="float32")
        client._multivec.request_json.return_value = {
            "data": [{"index": 0, "multivec": _npy_b64(v), "num_tokens": 1}]
        }
        await client.embed_images_multivector(["data:image/jpeg;base64,ZZZ"])
        _, _, payload = client._multivec.request_json.call_args.args
        url = payload["input"]["messages"][0][0]["content"][0]["image_url"]["url"]
        assert url == "data:image/jpeg;base64,ZZZ"


# ---------------------------------------------------------------------------
# score_maxsim -> /score (Precomputed{multivec} encode)
# ---------------------------------------------------------------------------

class TestScoreMaxsim:
    @pytest.mark.asyncio
    async def test_encodes_vectors_to_npy_and_parses_scores(self, client):
        client._score.request_json.return_value = {"scores": [0.95, 0.72]}

        query_vectors = [[0.1, 0.2], [0.3, 0.4]]
        document_vectors = [
            [[0.5, 0.6], [0.7, 0.8]],
            [[0.9, 0.0]],
        ]
        result = await client.score_maxsim(query_vectors, document_vectors)

        # Parent contract: list of {index, score} dicts, in document order.
        assert result == [{"index": 0, "score": 0.95}, {"index": 1, "score": 0.72}]

        method, path, payload = client._score.request_json.call_args.args
        assert (method, path) == ("POST", "/score")
        # Outgoing request encodes vectors as base64 .npy (Precomputed form).
        assert "multivec" in payload["query"]
        assert len(payload["documents"]) == 2
        assert all("multivec" in d for d in payload["documents"])
        assert payload["pool_factor"] == 1

        # The encoded query .npy round-trips back to the provided vectors.
        decoded_q = _npy_b64_to_vectors(payload["query"]["multivec"])
        assert np.allclose(decoded_q, query_vectors)
        decoded_d0 = _npy_b64_to_vectors(payload["documents"][0]["multivec"])
        assert np.allclose(decoded_d0, document_vectors[0])

    @pytest.mark.asyncio
    async def test_empty_documents(self, client):
        client._score.request_json.return_value = {"scores": []}
        result = await client.score_maxsim([[0.1]], [])
        assert result == []
        _, _, payload = client._score.request_json.call_args.args
        assert payload["documents"] == []


# ---------------------------------------------------------------------------
# encode -> /encode (single image)
# ---------------------------------------------------------------------------

class TestEncode:
    @pytest.mark.asyncio
    async def test_encode_builds_image_url_message(self, client):
        client._visualtok.request_json.return_value = {
            "prompt_embeds": "p",
            "image_embeds": "ie",
            "image_grid_thw": [[1, 2, 3]],
            "num_image_tokens": 6,
            "hidden_size": 8,
        }
        resp = await client.encode("aGVsbG8=")

        assert resp["image_embeds"] == "ie"
        method, path, payload = client._visualtok.request_json.call_args.args
        assert (method, path) == ("POST", "/encode")
        part = payload["messages"][0]["content"][0]
        assert part["type"] == "image_url"
        assert part["image_url"]["url"] == "data:image/png;base64,aGVsbG8="
        assert payload["apply_chat_template"] is True


# ---------------------------------------------------------------------------
# chat_completions -> flatten nested image_embeds block
# ---------------------------------------------------------------------------

class TestChatFlatten:
    @pytest.mark.asyncio
    async def test_flattens_nested_image_embeds_block(self, client):
        # Capture the payload the parent forwards by mocking request_json on self.
        captured = {}

        async def _fake_request_json(method, path, payload):
            captured["method"], captured["path"], captured["payload"] = method, path, payload
            return {"choices": []}

        client.request_json = _fake_request_json  # parent chat path uses self.request_json

        grid_list = [[1, 14, 14]]
        nested = {
            "type": "image_embeds",
            "image_embeds": {"image_embeds": "EMB_B64", "image_grid_thw": grid_list},
        }
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        nested,
                    ],
                }
            ],
        }
        await client.chat_completions(payload, stream=False)

        block = captured["payload"]["messages"][0]["content"][1]
        # Flat tokenary shape: image_embeds is a string, grid alongside.
        assert block["type"] == "image_embeds"
        assert block["image_embeds"] == "EMB_B64"
        assert block["image_grid_thw"] == grid_list
        # text block untouched
        assert captured["payload"]["messages"][0]["content"][0] == {"type": "text", "text": "hello"}

    @pytest.mark.asyncio
    async def test_flatten_decodes_base64_npy_grid(self, client):
        captured = {}

        async def _fake_request_json(method, path, payload):
            captured["payload"] = payload
            return {"choices": []}

        client.request_json = _fake_request_json

        grid_arr = np.array([[1, 14, 14]], dtype="int64")
        nested = {
            "type": "image_embeds",
            "image_embeds": {"image_embeds": "EMB", "image_grid_thw": _npy_b64(grid_arr)},
        }
        payload = {"messages": [{"role": "user", "content": [nested]}]}
        await client.chat_completions(payload, stream=False)

        block = captured["payload"]["messages"][0]["content"][0]
        assert block["image_embeds"] == "EMB"
        assert block["image_grid_thw"] == [[1, 14, 14]]

    @pytest.mark.asyncio
    async def test_passthrough_when_no_image_embeds(self, client):
        captured = {}

        async def _fake_request_json(method, path, payload):
            captured["payload"] = payload
            return {"choices": []}

        client.request_json = _fake_request_json
        payload = {"messages": [{"role": "user", "content": "plain text"}]}
        await client.chat_completions(payload, stream=False)
        assert captured["payload"]["messages"][0]["content"] == "plain text"


# ---------------------------------------------------------------------------
# Plugin selection via EXT_EMBED_SERVER_CLIENT
# ---------------------------------------------------------------------------

class TestPluginSelection:
    def test_tokenary_plugin_provider_name(self):
        from memorylayer_saas.services.document import TokenaryClientPlugin

        assert TokenaryClientPlugin.PROVIDER_NAME == "tokenary"

    def test_tokenary_plugin_initialize_returns_tokenary_client(self, mock_logger):
        from scitrera_app_framework import Variables

        from memorylayer_saas.services.document import TokenaryClientPlugin

        v = Variables()
        v.set("MEMORYLAYER_EMBED_SERVER_URL", "http://tokenary:8000")
        v.set("MEMORYLAYER_TOKENARY_MULTIVEC_URL", "http://mv:8000")
        client = TokenaryClientPlugin().initialize(v, mock_logger)
        assert isinstance(client, TokenaryClient)
        # multivec child routes to its own configured URL.
        assert client._multivec._base_url == "http://mv:8000"
        # textvec child falls back to the shared base URL when unset.
        assert client._textvec._base_url == "http://tokenary:8000"

    def test_default_plugin_returns_plain_client(self, mock_logger):
        from memorylayer_server.services.document.embed_client import (
            EmbedServerClient,
            EmbedServerClientPlugin,
        )
        from scitrera_app_framework import Variables

        v = Variables()
        v.set("MEMORYLAYER_EMBED_SERVER_URL", "http://embed:61051")
        client = EmbedServerClientPlugin().initialize(v, mock_logger)
        assert isinstance(client, EmbedServerClient)
        assert not isinstance(client, TokenaryClient)
