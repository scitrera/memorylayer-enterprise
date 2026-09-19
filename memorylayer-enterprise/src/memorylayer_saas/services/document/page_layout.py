# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only
"""Durable OCR layout tied to the exact page pixels and rendered transcript."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json

from PIL import Image
from ..transcription.regions import BBOX_SCALE

PAGE_LAYOUT_METADATA_KEY = "ocr_layout"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def layout_metadata(page, image_data: bytes) -> dict:
    # Header-only inspection: do not decode another full raster during ingestion.
    with Image.open(io.BytesIO(image_data)) as image:
        width, height = image.size
    image_hash = sha(image_data)
    regions = []
    for index, region in enumerate(page.regions):
        box = region.bbox
        valid = (box is not None and len(box) == 4 and all(type(v) is int for v in box)
                 and 0 <= box[0] < box[2] <= BBOX_SCALE and 0 <= box[1] < box[3] <= BBOX_SCALE)
        item = {"index": index, "label": region.label, "text": region.content,
                "bbox": [v / BBOX_SCALE for v in box] if valid else None}
        item["id"] = "region_" + sha(json.dumps([image_hash, item], sort_keys=True,
                                                ensure_ascii=False).encode())[:32]
        regions.append(item)
    return {"version": 1, "coordinate_space": "normalized_0_1", "image_sha256": image_hash,
            "image_width": width, "image_height": height, "transcript_sha256": sha(page.content.encode()),
            "model": page.model, "provider": page.provider, "output_contract": page.output_contract,
            "regions": regions}


async def store_page_layout(*, blob_storage, workspace_id, doc_id, page_no, page_image_b64, page) -> dict | None:
    """Persist raw output; publish coordinates only with a matching image identity.

    No geometry is invented for ungrounded providers or malformed rectangles.
    A failed raw-output write fails ingestion so a successful page never silently
    loses the promised provenance. Legacy callers with no raw output/layout remain
    compatible and return None; callers must remove any stale layout on replacement.
    """
    if not page.raw_content and not page.regions:
        return None
    def prepare():
        return layout_metadata(page, base64.b64decode(page_image_b64, validate=True))
    result = await asyncio.to_thread(prepare)
    if page.raw_content:
        raw = page.raw_content.encode()
        digest = sha(raw)
        path = blob_storage.page_raw_ocr_path(workspace_id, doc_id, page_no, digest)
        await blob_storage.store_file(path, raw)
        result["raw_ocr"] = {"storage_path": path, "sha256": digest}
    return result
