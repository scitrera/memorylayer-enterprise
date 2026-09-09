"""Real-model integration test for the Qwen3.6 image-embed provider.

Loads ``Qwen/Qwen3.6-27B-FP8`` via HF Transformers and verifies the
vision-tower image-embed extraction end-to-end, plus the zstd codec round-trip
and the uncompressed grid payload.

Heavy (loads a 27B FP8 model on GPU). Deselected by default; run with:

    RUN_VT_INTEGRATION=1 HF_HUB_OFFLINE=1 \
        pytest tests/integration -m integration

Skips automatically unless ``RUN_VT_INTEGRATION=1`` and torch+CUDA are present.
"""
import io
import os

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from memorylayer_embed_server_enterprise.services.visual_tokenizer.codec import (  # noqa: E402
    b64_to_tensor,
    tensor_to_b64,
)
from memorylayer_embed_server_enterprise.services.visual_tokenizer.qwen35_provider import (  # noqa: E402
    Qwen35VisualTokenizer,
)

pytestmark = pytest.mark.integration

MODEL = os.environ.get("VT_INTEGRATION_MODEL", "Qwen/Qwen3.6-27B-FP8")
EXPECTED_HIDDEN = 5120

_skip = pytest.mark.skipif(
    os.environ.get("RUN_VT_INTEGRATION") != "1" or not torch.cuda.is_available(),
    reason="set RUN_VT_INTEGRATION=1 and provide a CUDA GPU to run the real-model test",
)


def _make_image(w: int = 448, h: int = 448) -> bytes:
    img = Image.new("RGB", (w, h), (200, 120, 60))
    for x in range(0, w, 32):
        for y in range(0, h, 32):
            if (x // 32 + y // 32) % 2 == 0:
                for i in range(x, min(x + 32, w)):
                    for j in range(y, min(y + 32, h)):
                        img.putpixel((i, j), (30, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def provider():
    import asyncio

    prov = Qwen35VisualTokenizer(model_name=MODEL, embed_kind="image_embeds")
    asyncio.run(prov.preload())
    return prov


@_skip
class TestRealImageEmbed:
    @pytest.mark.asyncio
    async def test_vision_tower_embeds(self, provider):
        meta = {"filename": "annual_report.pdf", "page_no": 3, "source": "vfs://demo"}
        res = await provider.extract_features(_make_image(), meta)
        assert res.success, res.error
        f = res.features

        # Projected per-image vision embeds in the model's text-hidden space.
        assert f.embed_kind == "image_embeds"
        assert f.image_embeds.dim() == 2
        assert f.hidden_dim == EXPECTED_HIDDEN
        assert f.image_embeds.shape[1] == EXPECTED_HIDDEN
        assert f.num_visual_tokens == f.image_embeds.shape[0] > 0
        assert len(f.image_grid_thw) == 3

    @pytest.mark.asyncio
    async def test_codec_roundtrip_compressed_and_grid(self, provider):
        res = await provider.extract_features(_make_image(), {"page_no": 0})
        assert res.success, res.error
        f = res.features

        # Compressed image-embeds round-trip via our codec.
        restored = b64_to_tensor(tensor_to_b64(f.image_embeds.float(), compress=True))
        assert restored.shape == f.image_embeds.shape
        assert restored.shape[1] == EXPECTED_HIDDEN

        # Uncompressed grid payload round-trips to the same [t,h,w].
        grid = torch.tensor(f.image_grid_thw, dtype=torch.int64)
        restored_grid = b64_to_tensor(tensor_to_b64(grid, compress=False))
        assert restored_grid.dtype == torch.int64
        assert restored_grid.tolist() == f.image_grid_thw

    @pytest.mark.asyncio
    async def test_metadata_does_not_change_embeds(self, provider):
        """Image embeds depend only on the image; metadata is informational."""
        img = _make_image()
        a = await provider.extract_features(img, {"page_no": 0})
        b = await provider.extract_features(img, {"page_no": 1})
        assert a.success and b.success
        assert a.features.image_embeds.shape == b.features.image_embeds.shape
        assert torch.allclose(
            a.features.image_embeds.float(), b.features.image_embeds.float()
        )
