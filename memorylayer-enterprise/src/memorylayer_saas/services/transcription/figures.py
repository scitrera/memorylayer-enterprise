"""Crop figure regions out of a page render.

Unlimited-OCR marks every figure with an exact box and emits no text for it, so
without this the illustrations in a document transcribe to nothing. The boxes
are normalized to :data:`.regions.BBOX_SCALE`, not page pixels, so they scale to
whatever DPI the page was rendered at.

Verified against the 200-DPI renders in ``test_data/TulsaRFP_pages``: a page-8
``image`` box maps to a tight crop of the restaurant photo the adjacent text
describes.
"""

from __future__ import annotations

import io

from .regions import BBOX_SCALE, LABEL_IMAGE, PageRegion

#: Skip crops smaller than this on either axis. A degenerate or near-empty box
#: yields a few pixels of background, which is worse than nothing downstream
#: (it still costs an embed and a blob).
DEFAULT_MIN_CROP_PX = 16


def bbox_to_pixels(
    bbox: tuple[int, int, int, int], width: int, height: int,
) -> tuple[int, int, int, int]:
    """Map a normalized box onto a page of ``width`` x ``height`` pixels.

    Coordinates are clamped to the page so a slightly out-of-range box (the
    model occasionally emits one) crops rather than raising.
    """
    x0, y0, x1, y1 = bbox
    left = round(x0 / BBOX_SCALE * width)
    top = round(y0 / BBOX_SCALE * height)
    right = round(x1 / BBOX_SCALE * width)
    bottom = round(y1 / BBOX_SCALE * height)
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    return (
        max(0, min(left, width)),
        max(0, min(top, height)),
        max(0, min(right, width)),
        max(0, min(bottom, height)),
    )


def crop_figures(
    page_image: bytes,
    regions: list[PageRegion],
    *,
    labels: tuple[str, ...] = (LABEL_IMAGE,),
    min_crop_px: int = DEFAULT_MIN_CROP_PX,
    image_format: str = "PNG",
) -> list[tuple[PageRegion, bytes]]:
    """Return ``(region, cropped_png_bytes)`` for each figure on the page.

    Regions without a box, or whose box is degenerate, are skipped rather than
    raising: one bad box must not fail a document's ingestion.
    """
    wanted = [r for r in regions if r.label in labels and r.bbox is not None]
    if not wanted:
        return []

    from PIL import Image

    crops: list[tuple[PageRegion, bytes]] = []
    with Image.open(io.BytesIO(page_image)) as page:
        page.load()
        width, height = page.size
        for region in wanted:
            left, top, right, bottom = bbox_to_pixels(region.bbox, width, height)
            if (right - left) < min_crop_px or (bottom - top) < min_crop_px:
                continue
            buffer = io.BytesIO()
            page.crop((left, top, right, bottom)).save(buffer, format=image_format)
            crops.append((region, buffer.getvalue()))
    return crops
