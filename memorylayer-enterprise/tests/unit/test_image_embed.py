"""Unit tests for the per-page image-embed precompute/persist + injection helpers."""

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import zstandard

import memorylayer_saas.services.document.blob_cache as blob_cache_mod
from memorylayer_saas.services.document.image_embed import (
    build_image_embeds_content_blocks,
    build_page_metadata,
    page_metadata_text,
    precompute_and_store_image_embeds,
)

MODEL_SLUG = "qwen--qwen3.6-27b-fp8"


@pytest.fixture
def v_no_cache():
    """A Variables with the blob cache disabled (MAX_GB=0) + fresh singleton.

    Resets the process-local blob-cache singleton so content-block tests assert
    on the real underlying ``retrieve_file`` call count without cross-test
    cache state.
    """
    from scitrera_app_framework import Variables

    blob_cache_mod._BLOB_CACHE = None
    v = Variables()
    v.set(blob_cache_mod.MEMORYLAYER_DOCUMENT_BLOB_CACHE_MAX_GB, "0")
    yield v
    blob_cache_mod._BLOB_CACHE = None

# Embeds blob is stored zstd-compressed (raw torch.save bytes -> zstd).
RAW_EMBEDS = b"fake-torch-save-vision-embeds"
COMPRESSED_EMBEDS = zstandard.ZstdCompressor(level=1).compress(RAW_EMBEDS)
EMBEDS_B64 = base64.b64encode(COMPRESSED_EMBEDS).decode("ascii")
# Grid blob is stored uncompressed (raw torch.save bytes).
GRID_BYTES = b"fake-torch-save-grid"
GRID_B64 = base64.b64encode(GRID_BYTES).decode("ascii")


def _page(page_id, page_no, *, image=True, visual_tokens=None):
    return SimpleNamespace(
        id=page_id,
        page_no=page_no,
        image_storage_path=f"/blobs/ws/doc/pages/page_{page_no:04d}.png" if image else None,
        visual_tokens=visual_tokens,
    )


def _blob_storage():
    blob = MagicMock()
    blob.retrieve_file = AsyncMock(return_value=b"image-bytes")
    blob.store_file = AsyncMock(side_effect=lambda path, data: path)
    blob.page_image_embeds_path = MagicMock(
        side_effect=lambda ws, doc, pno, slug: f"/blobs/{ws}/{doc}/image_embeds/{slug}/page_{pno:04d}.pt.zst"
    )
    blob.page_image_grid_path = MagicMock(
        side_effect=lambda ws, doc, pno, slug: f"/blobs/{ws}/{doc}/image_embeds/{slug}/page_{pno:04d}.grid.pt"
    )
    return blob


def _embed_client(results, model="Qwen/Qwen3.6-27B-FP8"):
    client = MagicMock()
    client.request_json = AsyncMock(
        return_value={"results": results, "stats": {}, "model": model}
    )
    return client


def _ok_result(image_index, *, model_slug=MODEL_SLUG):
    return {
        "image_index": image_index,
        "success": True,
        "image_embeds_b64": EMBEDS_B64,
        "image_grid_thw_b64": GRID_B64,
        "image_grid_thw": [1, 38, 28],
        "embed_kind": "image_embeds",
        "num_image_tokens": 100 + image_index,
        "hidden_dim": 5120,
        "model_slug": model_slug,
    }


class TestBuildPageMetadata:
    def test_minimal(self):
        meta = build_page_metadata(filename=None, page_no=2, document_id="doc1")
        assert meta == {"page_no": 2, "document_id": "doc1"}

    def test_full(self):
        meta = build_page_metadata(
            filename="a.pdf", page_no=0, document_id="doc1", source="vfs://x",
        )
        assert meta["filename"] == "a.pdf"
        assert meta["source"] == "vfs://x"


class TestPageMetadataText:
    def test_with_filename(self):
        page = _page("p0", 0)
        assert page_metadata_text(page, "a.pdf") == "[Page 1 of a.pdf]"

    def test_without_filename(self):
        page = _page("p0", 4)
        assert page_metadata_text(page, None) == "[Page 5]"


class TestPrecomputeAndStore:
    @pytest.mark.asyncio
    async def test_stores_blobs_and_jsonb_ref(self):
        blob = _blob_storage()
        storage = MagicMock()
        storage.update_page = AsyncMock()
        pages = [_page("p0", 0), _page("p1", 1)]
        client = _embed_client([_ok_result(0), _ok_result(1)])

        stored = await precompute_and_store_image_embeds(
            embed_client=client,
            blob_storage=blob,
            storage=storage,
            pages=pages,
            workspace_id="ws",
            document_id="doc",
            filename="a.pdf",
            logger=MagicMock(),
        )

        assert stored == 2
        # Two blobs per page: compressed embeds (as-is) + raw grid.
        stored_map = {c.args[0]: c.args[1] for c in blob.store_file.await_args_list}
        embeds_paths = [p for p in stored_map if p.endswith(".pt.zst")]
        grid_paths = [p for p in stored_map if p.endswith(".grid.pt")]
        assert len(embeds_paths) == 2 and len(grid_paths) == 2
        # Embeds bytes are stored verbatim (still zstd-compressed).
        assert all(stored_map[p] == COMPRESSED_EMBEDS for p in embeds_paths)
        assert all(stored_map[p] == GRID_BYTES for p in grid_paths)

        # visual_tokens written keyed by model slug.
        first_call = storage.update_page.await_args_list[0]
        vt = first_call.kwargs["visual_tokens"]
        assert MODEL_SLUG in vt
        assert vt[MODEL_SLUG]["embed_kind"] == "image_embeds"
        assert vt[MODEL_SLUG]["embeds_blob_path"].endswith("/page_0000.pt.zst")
        assert vt[MODEL_SLUG]["grid_blob_path"].endswith("/page_0000.grid.pt")
        assert vt[MODEL_SLUG]["hidden_dim"] == 5120
        assert vt[MODEL_SLUG]["image_grid_thw"] == [1, 38, 28]

    @pytest.mark.asyncio
    async def test_preserves_existing_model_entries(self):
        """A second model must not clobber an existing model's ref (no migration)."""
        blob = _blob_storage()
        storage = MagicMock()
        storage.update_page = AsyncMock()
        existing = {"other-model": {"embeds_blob_path": "/x", "embed_kind": "image_embeds"}}
        pages = [_page("p0", 0, visual_tokens=existing)]
        client = _embed_client([_ok_result(0)])

        await precompute_and_store_image_embeds(
            embed_client=client, blob_storage=blob, storage=storage, pages=pages,
            workspace_id="ws", document_id="doc", filename="a.pdf", logger=MagicMock(),
        )

        vt = storage.update_page.await_args_list[0].kwargs["visual_tokens"]
        assert "other-model" in vt and MODEL_SLUG in vt

    @pytest.mark.asyncio
    async def test_skips_pages_with_existing_slug_entry(self):
        """Re-run/gap-fill: a page already carrying an entry for THIS model slug is
        not re-stored (counted as stored, no blob writes); a fresh page IS stored."""
        blob = _blob_storage()
        storage = MagicMock()
        storage.update_page = AsyncMock()
        # p0 already has an entry for MODEL_SLUG (done); p1 has none (fresh).
        done_vt = {MODEL_SLUG: {"embeds_blob_path": "/old", "embed_kind": "image_embeds"}}
        pages = [_page("p0", 0, visual_tokens=done_vt), _page("p1", 1)]
        client = _embed_client([_ok_result(0), _ok_result(1)])

        stored = await precompute_and_store_image_embeds(
            embed_client=client, blob_storage=blob, storage=storage, pages=pages,
            workspace_id="ws", document_id="doc", filename="a.pdf", logger=MagicMock(),
        )

        # Both counted as stored (p0 already done, p1 freshly stored).
        assert stored == 2
        # Only p1 triggered blob writes + a persist.
        assert storage.update_page.await_count == 1
        assert storage.update_page.await_args.args[0] == "p1"
        # p0's blobs were NOT re-written (2 writes for p1 only: embeds + grid).
        assert blob.store_file.await_count == 2

    @pytest.mark.asyncio
    async def test_skips_failed_results(self):
        blob = _blob_storage()
        storage = MagicMock()
        storage.update_page = AsyncMock()
        pages = [_page("p0", 0), _page("p1", 1)]
        bad = {"image_index": 1, "success": False, "error": "OOM"}
        client = _embed_client([_ok_result(0), bad])

        stored = await precompute_and_store_image_embeds(
            embed_client=client, blob_storage=blob, storage=storage, pages=pages,
            workspace_id="ws", document_id="doc", filename="a.pdf", logger=MagicMock(),
        )
        assert stored == 1
        assert storage.update_page.await_count == 1

    @pytest.mark.asyncio
    async def test_empty_pages_noop(self):
        client = _embed_client([])
        stored = await precompute_and_store_image_embeds(
            embed_client=client, blob_storage=_blob_storage(), storage=MagicMock(),
            pages=[], workspace_id="ws", document_id="doc", filename=None, logger=MagicMock(),
        )
        assert stored == 0
        client.request_json.assert_not_called()


class TestBuildContentBlocks:
    @pytest.mark.asyncio
    async def test_builds_blocks_for_pages_with_ref(self, v_no_cache):
        blob = MagicMock()

        async def _retrieve(path):
            return COMPRESSED_EMBEDS if path.endswith(".pt.zst") else GRID_BYTES

        blob.retrieve_file = AsyncMock(side_effect=_retrieve)
        pages = [
            _page("p0", 0, visual_tokens={MODEL_SLUG: {
                "embeds_blob_path": "/blobs/p0.pt.zst",
                "grid_blob_path": "/blobs/p0.grid.pt",
            }}),
            _page("p1", 1, visual_tokens={"other": {"embeds_blob_path": "/blobs/other.pt.zst"}}),
            _page("p2", 2, visual_tokens=None),
        ]
        blocks = await build_image_embeds_content_blocks(
            blob_storage=blob, pages=pages, model_slug=MODEL_SLUG, v=v_no_cache,
        )
        # Only p0 has an entry for MODEL_SLUG.
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image_embeds"
        # Embeds were zstd-decompressed back to raw torch.save bytes for the wire.
        assert blocks[0]["image_embeds"]["image_embeds"] == base64.b64encode(RAW_EMBEDS).decode()
        assert blocks[0]["image_embeds"]["image_grid_thw"] == GRID_B64

    @pytest.mark.asyncio
    async def test_uncompressed_embeds_passed_through(self, v_no_cache):
        """If the embeds blob is not zstd-framed, it is base64'd as-is."""
        blob = MagicMock()

        async def _retrieve(path):
            return RAW_EMBEDS if path.endswith(".pt.zst") else GRID_BYTES

        blob.retrieve_file = AsyncMock(side_effect=_retrieve)
        pages = [_page("p0", 0, visual_tokens={MODEL_SLUG: {
            "embeds_blob_path": "/blobs/p0.pt.zst",
            "grid_blob_path": "/blobs/p0.grid.pt",
        }})]
        blocks = await build_image_embeds_content_blocks(
            blob_storage=blob, pages=pages, model_slug=MODEL_SLUG, v=v_no_cache,
        )
        assert blocks[0]["image_embeds"]["image_embeds"] == base64.b64encode(RAW_EMBEDS).decode()

    @pytest.mark.asyncio
    async def test_empty_when_no_refs(self, v_no_cache):
        blob = MagicMock()
        blob.retrieve_file = AsyncMock()
        pages = [_page("p0", 0, visual_tokens=None)]
        blocks = await build_image_embeds_content_blocks(
            blob_storage=blob, pages=pages, model_slug=MODEL_SLUG, v=v_no_cache,
        )
        assert blocks == []
        blob.retrieve_file.assert_not_called()
