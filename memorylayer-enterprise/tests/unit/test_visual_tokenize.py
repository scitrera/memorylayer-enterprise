"""Unit tests for the enterprise visual-tokenizer client helper.

Verifies the helper builds the ``/v1/visual-tokenize`` payload exactly and
routes it through the OSS client's public ``request_json`` seam.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_saas.services.document.visual_tokenize import visual_tokenize


def _client(response=None):
    client = MagicMock()
    client.request_json = AsyncMock(
        return_value=response if response is not None else {"results": [], "stats": {}, "model": "m"}
    )
    return client


@pytest.mark.asyncio
async def test_posts_images_and_return_tensors_minimal():
    """Minimal call: only images + return_tensors in the payload."""
    client = _client()

    out = await visual_tokenize(client, ["aW1n"])

    client.request_json.assert_awaited_once_with(
        "POST",
        "/v1/visual-tokenize",
        {"images": ["aW1n"], "return_tensors": True},
    )
    assert out == {"results": [], "stats": {}, "model": "m"}


@pytest.mark.asyncio
async def test_includes_metadata_when_given():
    """metadata is added to the payload when provided."""
    client = _client()
    meta = [{"filename": "a.pdf", "page_no": 0}]

    await visual_tokenize(client, ["aW1n"], metadata=meta)

    client.request_json.assert_awaited_once_with(
        "POST",
        "/v1/visual-tokenize",
        {"images": ["aW1n"], "return_tensors": True, "metadata": meta},
    )


@pytest.mark.asyncio
async def test_force_recompute_and_batch_size_only_when_provided():
    """force_recompute/batch_size appear only when explicitly set."""
    client = _client()

    await visual_tokenize(
        client,
        ["aW1n"],
        return_tensors=False,
        force_recompute=True,
        batch_size=4,
    )

    client.request_json.assert_awaited_once_with(
        "POST",
        "/v1/visual-tokenize",
        {
            "images": ["aW1n"],
            "return_tensors": False,
            "force_recompute": True,
            "batch_size": 4,
        },
    )


@pytest.mark.asyncio
async def test_omits_force_recompute_when_false():
    """force_recompute=False (default) is not added to the payload."""
    client = _client()

    await visual_tokenize(client, ["aW1n"], force_recompute=False)

    _method, _path, payload = client.request_json.await_args.args
    assert "force_recompute" not in payload
    assert "batch_size" not in payload


@pytest.mark.asyncio
async def test_returns_client_response_dict():
    """The helper returns the client's response verbatim."""
    expected = {
        "results": [{"image_embeds_b64": "AAA=", "num_image_tokens": 5}],
        "stats": {"total_images": 1},
        "model": "Qwen/Qwen3.6-27B-FP8",
    }
    client = _client(expected)

    out = await visual_tokenize(client, ["aW1n"], metadata=[{"page_no": 0}])

    assert out is expected
