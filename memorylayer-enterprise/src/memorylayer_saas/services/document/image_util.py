"""Image helpers for the document pipeline.

Currently: shaping a rendered page image for attachment to a multimodal LLM
(fact decomposition). The full-resolution render stays the authoritative stored
artifact (viewing, OCR/transcription); this only shapes the copy sent to the
model — downscaled + JPEG — so the request body stays under the AI-gateway
ingress ``client_max_body_size`` (the nginx 413) and the multimodal call is
cheaper/faster.

JPEG is much smaller than PNG for document page scans (photos-of-text). The LLM
image contract (``services/llm/openai.py``) uses a raw base64 string verbatim as
``data:image/png;base64,...`` but passes a full ``data:<mime>;base64,...`` URI
through unchanged — so we emit a JPEG data URI with an explicit ``image/jpeg``
MIME rather than a bare base64 string.
"""

from __future__ import annotations

import base64
import io
import logging

logger = logging.getLogger(__name__)


def to_llm_image(image_b64: str, max_dim: int = 1200, quality: int = 85) -> str:
    """Return a page image sized/encoded for a multimodal LLM attachment.

    Downscales so the longest side is ``<= max_dim`` (aspect preserved; never
    upscales) and re-encodes as JPEG, returned as a ``data:image/jpeg;base64,...``
    data URI. ``max_dim <= 0`` skips downscaling but STILL re-encodes to JPEG (the
    size win). ``image_b64`` may be a raw base64 string or an existing data URI; a
    data URI is returned unchanged (already shaped upstream).

    Best-effort: on any failure (Pillow missing, undecodable bytes) returns the
    input unchanged so ingestion is never blocked — the model still receives the
    original image, and the AI-gateway body limit is the backstop.
    """
    if not image_b64 or image_b64.startswith("data:"):
        return image_b64
    try:
        from PIL import Image
    except Exception:  # Pillow not available — leave the image untouched
        return image_b64
    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        if max_dim > 0 and max(img.size) > max_dim:
            resample = getattr(Image, "Resampling", Image).LANCZOS
            img.thumbnail((max_dim, max_dim), resample)
        # JPEG has no alpha channel; page scans are opaque, but flatten to RGB so
        # any RGBA/palette source (e.g. a screenshot) still encodes.
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:  # noqa: BLE001 - best-effort; fall back to the original
        logger.warning("to_llm_image failed (%s); sending original image", exc)
        return image_b64
