"""Image-embed repair helpers for grounded document chat."""

from __future__ import annotations

from collections.abc import Sequence
from logging import Logger
from typing import Protocol

from .image_embed import precompute_and_store_image_embeds


class DocumentChatPage(Protocol):
    """Subset of DocumentPage fields needed by the chat repair path."""

    id: str
    document_id: str
    workspace_id: str
    page_no: int
    image_storage_path: str | None
    visual_tokens: dict | None


def page_has_image_embeds(page: DocumentChatPage, model_slug: str) -> bool:
    """Return whether a page has a usable image-embeds ref for ``model_slug``."""
    return bool((page.visual_tokens or {}).get(model_slug, {}).get("embeds_blob_path"))


def missing_image_embed_pages(
    pages: Sequence[DocumentChatPage],
    model_slug: str,
) -> list[DocumentChatPage]:
    """Return selected pages that cannot yet be injected into document chat."""
    return [page for page in pages if not page_has_image_embeds(page, model_slug)]


async def repair_missing_image_embeds(
    *,
    embed_client,
    blob_storage,
    storage,
    pages: Sequence[DocumentChatPage],
    model_slug: str,
    logger: Logger,
) -> int:
    """Attempt synchronous image-embed repair for missing document-chat pages.

    The existing ingestion precompute path updates each page's in-memory
    ``visual_tokens`` and persists the page row. Pages without rendered images
    are not repairable here and remain missing for the caller to report.
    """
    repairable_pages = [
        page
        for page in missing_image_embed_pages(pages, model_slug)
        if page.image_storage_path
    ]
    if not repairable_pages:
        return 0

    await embed_client.connect()

    grouped: dict[tuple[str, str], list[DocumentChatPage]] = {}
    for page in repairable_pages:
        key = (page.workspace_id, page.document_id)
        grouped.setdefault(key, []).append(page)

    stored = 0
    for (workspace_id, document_id), group_pages in grouped.items():
        doc = await storage.get_document(document_id, workspace_id)
        filename = getattr(doc, "filename", None) if doc else None
        source = getattr(doc, "source_vfs_ref", None) if doc else None
        stored += await precompute_and_store_image_embeds(
            embed_client=embed_client,
            blob_storage=blob_storage,
            storage=storage,
            pages=group_pages,
            workspace_id=workspace_id,
            document_id=document_id,
            filename=filename,
            logger=logger,
            source=source,
        )

    return stored
