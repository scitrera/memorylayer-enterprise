"""
Unit tests for EmbedServerClient.

Tests:
- transcribe_pages: OCR/VLM transcription via POST /v1/transcribe
- embed_texts: Single-vector embeddings via POST /v1/embeddings
- embed_texts_multivector: Multi-vector embeddings via POST /v1/embeddings/multi
- embed_images_multivector: Image multi-vector embeddings via POST /v1/embeddings/images
- score_maxsim: MaxSim scoring via POST /v1/score
- HTTP error propagation
- connect/close lifecycle
"""
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock

from memorylayer_saas.services.document.embed_client import EmbedServerClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_logger():
    """Create a no-op logger mock."""
    return MagicMock()


@pytest.fixture
def embed_client(mock_logger):
    """Create an EmbedServerClient with a test base URL (not yet connected)."""
    return EmbedServerClient(
        base_url="http://test-embed-server:61051",
        timeout=30.0,
        logger=mock_logger,
    )


def _make_mock_http_client():
    """Return an AsyncMock whose .post() method returns a fresh AsyncMock response."""
    client = AsyncMock()
    return client


def _make_ok_response(json_body: dict) -> MagicMock:
    """Build a mock HTTP response that succeeds with the given JSON body.

    The production code calls response.json() and response.raise_for_status()
    synchronously (no await), so the response must be a MagicMock, not AsyncMock.
    """
    response = MagicMock()
    response.json.return_value = json_body
    response.raise_for_status = MagicMock()  # no-op (sync)
    return response


# ---------------------------------------------------------------------------
# connect / close lifecycle
# ---------------------------------------------------------------------------

class TestConnectAndClose:
    """Tests for the connect() and close() lifecycle methods."""

    @pytest.mark.asyncio
    async def test_connect_creates_http_client(self, embed_client):
        """Test that connect() initialises an httpx.AsyncClient on _client."""
        assert embed_client._client is None
        await embed_client.connect()
        assert embed_client._client is not None
        await embed_client.close()

    @pytest.mark.asyncio
    async def test_close_clears_client(self, embed_client):
        """Test that close() sets _client back to None."""
        await embed_client.connect()
        assert embed_client._client is not None
        await embed_client.close()
        assert embed_client._client is None

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, embed_client):
        """Test that close() on an already-closed client does not raise."""
        # _client is already None – should succeed silently
        await embed_client.close()


# ---------------------------------------------------------------------------
# transcribe_pages
# ---------------------------------------------------------------------------

class TestTranscribePages:
    """Tests for transcribe_pages()."""

    @pytest.mark.asyncio
    async def test_transcribe_pages_success(self, embed_client):
        """Test successful transcription of page images."""
        # transcribe_pages is a passthrough, so the body is arbitrary as far as
        # this assertion goes -- but use the embed server's REAL
        # TranscriptionResponse shape so this test can't be read as documenting
        # a wire format the server does not emit.
        expected = {
            "results": [
                {"page_index": 0, "content": "# Page One\nSome text.", "success": True,
                 "model_used": "vlm-v1", "provider_used": "glm-ocr", "attempts": []},
                {"page_index": 1, "content": "# Page Two\nMore text.", "success": True,
                 "model_used": "vlm-v1", "provider_used": "glm-ocr", "attempts": []},
            ],
            "stats": {"total_pages": 2, "successful_pages": 2, "failed_pages": 0},
        }
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = _make_ok_response(expected)
        embed_client._client = mock_http

        images_b64 = ["aGVsbG8=", "d29ybGQ="]
        result = await embed_client.transcribe_pages(images_b64)

        assert result == expected
        mock_http.post.assert_called_once_with(
            "/v1/transcribe",
            json={"images": images_b64},
        )

    @pytest.mark.asyncio
    async def test_transcribe_pages_with_system_prompt(self, embed_client):
        """Test transcription passes optional system_prompt in payload."""
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = _make_ok_response({"results": [], "stats": {}})
        embed_client._client = mock_http

        await embed_client.transcribe_pages(
            ["aGVsbG8="],
            system_prompt="Extract tables only.",
        )

        _, call_kwargs = mock_http.post.call_args
        assert call_kwargs["json"]["system_prompt"] == "Extract tables only."

    @pytest.mark.asyncio
    async def test_transcribe_pages_with_max_tokens(self, embed_client):
        """Test transcription passes optional max_tokens in payload."""
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = _make_ok_response({"results": [], "stats": {}})
        embed_client._client = mock_http

        await embed_client.transcribe_pages(["aGVsbG8="], max_tokens=512)

        _, call_kwargs = mock_http.post.call_args
        assert call_kwargs["json"]["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_transcribe_pages_http_error(self, embed_client):
        """Test that non-2xx responses propagate as httpx.HTTPStatusError."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = mock_response
        embed_client._client = mock_http

        with pytest.raises(httpx.HTTPStatusError):
            await embed_client.transcribe_pages(["aGVsbG8="])


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------

class TestEmbedTexts:
    """Tests for embed_texts() single-vector embeddings."""

    @pytest.mark.asyncio
    async def test_embed_texts_success(self, embed_client):
        """Test successful single-vector embedding of text inputs."""
        json_body = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
            ],
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = _make_ok_response(json_body)
        embed_client._client = mock_http

        result = await embed_client.embed_texts(["hello", "world"])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_http.post.assert_called_once_with(
            "/v1/embeddings",
            json={"input": ["hello", "world"]},
        )

    @pytest.mark.asyncio
    async def test_embed_texts_out_of_order_indices_sorted(self, embed_client):
        """Test that embeddings are re-ordered by index before returning."""
        json_body = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.9, 0.9], "index": 1},
                {"object": "embedding", "embedding": [0.1, 0.1], "index": 0},
            ],
        }
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = _make_ok_response(json_body)
        embed_client._client = mock_http

        result = await embed_client.embed_texts(["a", "b"])

        # index 0 should come first
        assert result[0] == [0.1, 0.1]
        assert result[1] == [0.9, 0.9]

    @pytest.mark.asyncio
    async def test_embed_texts_http_error(self, embed_client):
        """Test that HTTP errors from the embeddings endpoint are propagated."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "422 Unprocessable Entity",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = mock_response
        embed_client._client = mock_http

        with pytest.raises(httpx.HTTPStatusError):
            await embed_client.embed_texts(["test"])


# ---------------------------------------------------------------------------
# embed_texts_multivector
# ---------------------------------------------------------------------------

class TestEmbedTextsMultivector:
    """Tests for embed_texts_multivector() late-interaction embeddings."""

    @pytest.mark.asyncio
    async def test_embed_texts_multivector_success(self, embed_client):
        """Test successful multi-vector embedding of text inputs."""
        vectors_0 = [[0.1, 0.2], [0.3, 0.4]]
        vectors_1 = [[0.5, 0.6]]
        json_body = {
            "object": "list",
            "data": [
                {"index": 0, "vectors": vectors_0, "num_vectors": 2},
                {"index": 1, "vectors": vectors_1, "num_vectors": 1},
            ],
        }
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = _make_ok_response(json_body)
        embed_client._client = mock_http

        result = await embed_client.embed_texts_multivector(["doc one", "doc two"])

        assert len(result) == 2
        assert result[0] == {"vectors": vectors_0, "num_vectors": 2}
        assert result[1] == {"vectors": vectors_1, "num_vectors": 1}
        mock_http.post.assert_called_once_with(
            "/v1/embeddings/multi",
            json={"input": ["doc one", "doc two"], "input_type": "document"},
        )

    @pytest.mark.asyncio
    async def test_embed_texts_multivector_query_type(self, embed_client):
        """Test that input_type='query' is forwarded correctly."""
        json_body = {"data": [{"index": 0, "vectors": [[0.1]], "num_vectors": 1}]}
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = _make_ok_response(json_body)
        embed_client._client = mock_http

        await embed_client.embed_texts_multivector(["query text"], input_type="query")

        _, call_kwargs = mock_http.post.call_args
        assert call_kwargs["json"]["input_type"] == "query"

    @pytest.mark.asyncio
    async def test_embed_texts_multivector_out_of_order_sorted(self, embed_client):
        """Test that multi-vector results are sorted by index."""
        json_body = {
            "data": [
                {"index": 1, "vectors": [[0.9]], "num_vectors": 1},
                {"index": 0, "vectors": [[0.1]], "num_vectors": 1},
            ]
        }
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = _make_ok_response(json_body)
        embed_client._client = mock_http

        result = await embed_client.embed_texts_multivector(["a", "b"])

        assert result[0]["vectors"] == [[0.1]]
        assert result[1]["vectors"] == [[0.9]]


# ---------------------------------------------------------------------------
# embed_images_multivector
# ---------------------------------------------------------------------------

class TestEmbedImagesMultivector:
    """Tests for embed_images_multivector() image-based multi-vector embeddings."""

    @pytest.mark.asyncio
    async def test_embed_images_multivector_success(self, embed_client):
        """Test successful multi-vector embedding from page images."""
        vectors_0 = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        json_body = {
            "data": [
                {"index": 0, "vectors": vectors_0, "num_vectors": 2},
            ]
        }
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = _make_ok_response(json_body)
        embed_client._client = mock_http

        result = await embed_client.embed_images_multivector(["aGVsbG8="])

        assert len(result) == 1
        assert result[0] == {"vectors": vectors_0, "num_vectors": 2}
        mock_http.post.assert_called_once_with(
            "/v1/embeddings/images",
            json={"images": ["aGVsbG8="], "mode": "multi"},
        )

    @pytest.mark.asyncio
    async def test_embed_images_multivector_multiple_images(self, embed_client):
        """Test multi-vector embedding of multiple images returns one entry per image."""
        json_body = {
            "data": [
                {"index": 0, "vectors": [[0.1]], "num_vectors": 1},
                {"index": 1, "vectors": [[0.2]], "num_vectors": 1},
                {"index": 2, "vectors": [[0.3]], "num_vectors": 1},
            ]
        }
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = _make_ok_response(json_body)
        embed_client._client = mock_http

        result = await embed_client.embed_images_multivector(["img1", "img2", "img3"])

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_embed_images_multivector_http_error(self, embed_client):
        """Test that HTTP errors from the images endpoint are propagated."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503 Service Unavailable",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = mock_response
        embed_client._client = mock_http

        with pytest.raises(httpx.HTTPStatusError):
            await embed_client.embed_images_multivector(["aGVsbG8="])


# ---------------------------------------------------------------------------
# score_maxsim
# ---------------------------------------------------------------------------

class TestScoreMaxsim:
    """Tests for score_maxsim() late-interaction scoring."""

    @pytest.mark.asyncio
    async def test_score_maxsim_success(self, embed_client):
        """Test successful MaxSim scoring of query against documents."""
        query_vectors = [[0.1, 0.2], [0.3, 0.4]]
        doc_vectors = [
            [[0.5, 0.6], [0.7, 0.8]],
            [[0.9, 0.0], [0.1, 0.2]],
        ]
        expected_scores = [
            {"index": 0, "score": 0.95},
            {"index": 1, "score": 0.72},
        ]
        json_body = {"scores": expected_scores}
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = _make_ok_response(json_body)
        embed_client._client = mock_http

        result = await embed_client.score_maxsim(query_vectors, doc_vectors)

        assert result == expected_scores
        mock_http.post.assert_called_once_with(
            "/v1/score",
            json={
                "query_vectors": query_vectors,
                "document_vectors": doc_vectors,
            },
        )

    @pytest.mark.asyncio
    async def test_score_maxsim_empty_documents(self, embed_client):
        """Test scoring against an empty document list returns empty scores."""
        json_body = {"scores": []}
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = _make_ok_response(json_body)
        embed_client._client = mock_http

        result = await embed_client.score_maxsim([[0.1, 0.2]], [])

        assert result == []

    @pytest.mark.asyncio
    async def test_score_maxsim_http_error(self, embed_client):
        """Test that HTTP errors from the scoring endpoint are propagated."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_http = _make_mock_http_client()
        mock_http.post.return_value = mock_response
        embed_client._client = mock_http

        with pytest.raises(httpx.HTTPStatusError):
            await embed_client.score_maxsim([[0.1]], [[[0.2]]])
