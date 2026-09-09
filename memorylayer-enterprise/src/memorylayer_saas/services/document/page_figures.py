"""Persist figures cropped out of a page render.

A grounded OCR model marks an illustration with a bounding box and emits no
text for it, so without this step every photograph, chart, and diagram in a
document ingests as silence. The crop is the figure's only representation.

Stored blobs are linked to the transcript by convention: the Nth figure on a
page is written to ``page_figure_path(..., figure_no=N)`` and appears in that
page's transcript as ``[figure N]``.
"""

from __future__ import annotations

import base64
from logging import Logger

from ..transcription.figures import crop_figures
from ..transcription.regions import PageRegion

#: Key under ``DocumentPage.metadata`` holding the page's figure records.
#: Metadata rather than a dedicated column so figures need no migration; the
#: records are what makes a stored crop DISCOVERABLE (and therefore fetchable)
#: — a blob nobody can enumerate is the same as a blob that isn't there.
PAGE_FIGURES_METADATA_KEY = "figures"


async def store_page_figures(
    *,
    blob_storage,
    workspace_id: str,
    doc_id: str,
    page_no: int,
    page_image_b64: str,
    regions: tuple[PageRegion, ...] | list[PageRegion],
    logger: Logger,
    captions: dict[int, str] | None = None,
) -> list[dict]:
    """Crop and store this page's figures; return their persistable records.

    Each record is ``{figure_no, storage_path, bbox, caption}``:

    * ``figure_no`` is 1-based and matches the ``[figure N]`` marker in the
      page transcript, which is what ties the text to the image;
    * ``bbox`` is the normalized (0-1000) box, retained so a consumer can
      locate the figure on the page render without re-running OCR;
    * ``caption`` is the model's description when captioning ran, so a client
      can decide whether it needs the bytes without fetching them.

    Never raises: a document whose figures fail to crop is still worth
    ingesting, so problems are logged and the text path continues. Page bytes
    are decoded transiently from the base64 the caller already holds, so this
    costs one page's image in peak memory, not a second full batch.
    """
    figures = [r for r in regions if r.is_figure and r.bbox is not None]
    if not figures:
        return []

    try:
        crops = crop_figures(base64.b64decode(page_image_b64), list(regions))
    except Exception as e:  # noqa: BLE001 — figures must not fail an ingest
        logger.warning(
            "Cropping figures for document %s page %d failed: %s", doc_id, page_no, e,
        )
        return []

    captions = captions or {}
    stored: list[dict] = []
    for figure_no, (region, data) in enumerate(crops, start=1):
        path = blob_storage.page_figure_path(workspace_id, doc_id, page_no, figure_no)
        try:
            await blob_storage.store_file(path, data)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Storing figure %d for document %s page %d failed: %s",
                figure_no, doc_id, page_no, e,
            )
            continue
        stored.append({
            "figure_no": figure_no,
            "storage_path": path,
            "bbox": list(region.bbox) if region.bbox else None,
            "caption": captions.get(figure_no),
        })

    if stored:
        logger.info(
            "Stored %d figure(s) for document %s page %d", len(stored), doc_id, page_no,
        )
    return stored
