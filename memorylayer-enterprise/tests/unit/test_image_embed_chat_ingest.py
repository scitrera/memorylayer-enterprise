"""Unit tests for ``generate_page_text_from_image_embeds``.

This is the document-chat ingestion building block: it injects a page's
precomputed ``image_embeds`` into a single-turn chat completion on the inference
LLM and returns the generated Markdown, to be used as the page's memory content
instead of OCR transcription.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memorylayer_saas.models.document import DocumentPage
from memorylayer_saas.services.document import image_embed


def _page(page_id="p1", page_no=0, visual_tokens=None):
    return DocumentPage(
        id=page_id,
        document_id="doc_1",
        workspace_id="ws_1",
        page_no=page_no,
        image_storage_path="/blobs/ws_1/doc_1/pages/page_0000.png",
        visual_tokens=visual_tokens,
    )


_BLOCK = {
    "type": "image_embeds",
    "image_embeds": {"image_embeds": "ZW1i", "image_grid_thw": "Z3JpZA=="},
}


@pytest.mark.asyncio
async def test_happy_path_returns_text_and_interleaves_prompt():
    page = _page(visual_tokens={"qwen--m": {"embeds_blob_path": "x"}})
    inference = AsyncMock()
    inference.chat_completions.return_value = {
        "choices": [{"message": {"content": "# Heading\n\nBody text."}}]
    }

    with patch.object(
        image_embed, "build_image_embeds_content_blocks",
        new=AsyncMock(return_value=[_BLOCK]),
    ):
        text = await image_embed.generate_page_text_from_image_embeds(
            inference_client=inference,
            blob_storage=MagicMock(),
            page=page,
            model="qwen/m",
            model_slug="qwen--m",
            filename="report.pdf",
            instruction="Transcribe the page.",
            max_tokens=512,
            v=MagicMock(),
            logger=MagicMock(),
        )

    assert text == "# Heading\n\nBody text."

    # Payload interleaves: [page marker text][image_embeds block][instruction].
    payload = inference.chat_completions.call_args.args[0]
    assert payload["model"] == "qwen/m"
    assert payload["max_tokens"] == 512
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "text" and content[0]["text"].startswith("[Page 1")
    assert content[1] == _BLOCK
    assert content[-1] == {"type": "text", "text": "Transcribe the page."}


@pytest.mark.asyncio
async def test_no_embeds_returns_none_without_calling_llm():
    page = _page(visual_tokens=None)
    inference = AsyncMock()

    with patch.object(
        image_embed, "build_image_embeds_content_blocks",
        new=AsyncMock(return_value=[]),  # no image_embeds for this slug
    ):
        text = await image_embed.generate_page_text_from_image_embeds(
            inference_client=inference,
            blob_storage=MagicMock(),
            page=page,
            model="qwen/m",
            model_slug="qwen--m",
            filename=None,
            instruction="Transcribe the page.",
            max_tokens=512,
            v=MagicMock(),
            logger=MagicMock(),
        )

    assert text is None
    inference.chat_completions.assert_not_called()


@pytest.mark.asyncio
async def test_malformed_completion_returns_none():
    page = _page(visual_tokens={"qwen--m": {"embeds_blob_path": "x"}})
    inference = AsyncMock()
    inference.chat_completions.return_value = {"unexpected": "shape"}

    with patch.object(
        image_embed, "build_image_embeds_content_blocks",
        new=AsyncMock(return_value=[_BLOCK]),
    ):
        text = await image_embed.generate_page_text_from_image_embeds(
            inference_client=inference,
            blob_storage=MagicMock(),
            page=page,
            model="qwen/m",
            model_slug="qwen--m",
            filename=None,
            instruction="Transcribe the page.",
            max_tokens=512,
            v=MagicMock(),
            logger=MagicMock(),
        )

    assert text is None


@pytest.mark.asyncio
async def test_empty_content_returns_none():
    page = _page(visual_tokens={"qwen--m": {"embeds_blob_path": "x"}})
    inference = AsyncMock()
    inference.chat_completions.return_value = {
        "choices": [{"message": {"content": "   \n  "}}]
    }

    with patch.object(
        image_embed, "build_image_embeds_content_blocks",
        new=AsyncMock(return_value=[_BLOCK]),
    ):
        text = await image_embed.generate_page_text_from_image_embeds(
            inference_client=inference,
            blob_storage=MagicMock(),
            page=page,
            model="qwen/m",
            model_slug="qwen--m",
            filename=None,
            instruction="Transcribe the page.",
            max_tokens=512,
            v=MagicMock(),
            logger=MagicMock(),
        )

    assert text is None
