"""Storage-agnostic blob service for documents, page images, and transcripts.

Uses fsspec to abstract over local filesystem and S3-compatible storage.
All I/O is wrapped with asyncio.to_thread() for non-blocking async access.
"""
import asyncio
import os
from logging import Logger

import fsspec

from scitrera_app_framework import Variables, get_logger

from . import BlobStoragePluginBase
from ...config import (
    MEMORYLAYER_BLOB_STORAGE_TYPE,
    DEFAULT_MEMORYLAYER_BLOB_STORAGE_TYPE,
    MEMORYLAYER_BLOB_STORAGE_BASE_PATH,
    DEFAULT_MEMORYLAYER_BLOB_STORAGE_BASE_PATH,
    MEMORYLAYER_BLOB_S3_ENDPOINT_URL,
    MEMORYLAYER_BLOB_S3_ACCESS_KEY,
    MEMORYLAYER_BLOB_S3_SECRET_KEY,
    MEMORYLAYER_BLOB_S3_REGION,
)


class BlobStorageService:
    """Storage-agnostic blob service for documents, page images, and transcripts.

    Provides path conventions and async I/O operations backed by an fsspec
    filesystem. Supports local disk and S3-compatible object stores.
    """

    def __init__(self, fs: fsspec.AbstractFileSystem, base_path: str, logger: Logger):
        """Initialize the blob storage service.

        Args:
            fs: An fsspec filesystem instance (e.g., local or S3).
            base_path: Root path for all blob storage operations.
            logger: Logger instance.
        """
        self._fs = fs
        self._base_path = base_path.rstrip('/')
        self.logger = logger

    # === Path Conventions ===

    def document_path(self, workspace_id: str, doc_id: str, filename: str) -> str:
        """Build the storage path for an original document file.

        Args:
            workspace_id: Workspace identifier.
            doc_id: Document identifier.
            filename: Original filename.

        Returns:
            Full storage path string.
        """
        return f"{self._base_path}/{workspace_id}/documents/{doc_id}/{os.path.basename(filename)}"

    def page_image_path(self, workspace_id: str, doc_id: str, page_no: int) -> str:
        """Build the storage path for a rendered page image.

        Args:
            workspace_id: Workspace identifier.
            doc_id: Document identifier.
            page_no: Zero-indexed page number.

        Returns:
            Full storage path string.
        """
        return f"{self._base_path}/{workspace_id}/documents/{doc_id}/pages/page_{page_no:04d}.png"

    def page_image_embeds_path(
        self, workspace_id: str, doc_id: str, page_no: int, model_slug: str,
    ) -> str:
        """Build the storage path for a page's precomputed image-embeds tensor.

        The blob holds the zstd-compressed raw ``torch.save`` vision-tower
        output. Keyed by ``model_slug`` so multiple models can coexist for the
        same page without collision (and without a DB migration — the page's
        ``visual_tokens`` JSONB just gains another model-keyed entry).

        Args:
            workspace_id: Workspace identifier.
            doc_id: Document identifier.
            page_no: Zero-indexed page number.
            model_slug: Filesystem-safe model identifier (e.g.
                ``qwen--qwen3.6-27b-fp8``).

        Returns:
            Full storage path string.
        """
        return (
            f"{self._base_path}/{workspace_id}/documents/{doc_id}"
            f"/image_embeds/{model_slug}/page_{page_no:04d}.pt.zst"
        )

    def page_multivector_path(self, workspace_id: str, doc_id: str, page_no: int) -> str:
        """Build the storage path for a page's spilled ColPali multivector.

        The blob holds the float32 codec written by
        ``_encode_multivector`` in the ingestion service. This exists so the
        embed phase can release each page's multivector instead of holding it
        until the persist phase — a ColPali page is ~1030x128 floats, which as
        a Python ``list[list[float]]`` costs ~5 MB per page, so a 100-page
        document accumulated ~500 MB before it wrote a single memory row.

        Args:
            workspace_id: Workspace identifier.
            doc_id: Document identifier.
            page_no: Zero-indexed page number.

        Returns:
            Full storage path string.
        """
        return (
            f"{self._base_path}/{workspace_id}/documents/{doc_id}"
            f"/multivectors/page_{page_no:04d}.f32"
        )

    def page_image_grid_path(
        self, workspace_id: str, doc_id: str, page_no: int, model_slug: str,
    ) -> str:
        """Build the storage path for a page's ``image_grid_thw`` tensor.

        The blob holds the uncompressed raw ``torch.save`` grid tensor that
        pairs with the page's image-embeds. Keyed by ``model_slug`` alongside
        :meth:`page_image_embeds_path`.

        Args:
            workspace_id: Workspace identifier.
            doc_id: Document identifier.
            page_no: Zero-indexed page number.
            model_slug: Filesystem-safe model identifier (e.g.
                ``qwen--qwen3.6-27b-fp8``).

        Returns:
            Full storage path string.
        """
        return (
            f"{self._base_path}/{workspace_id}/documents/{doc_id}"
            f"/image_embeds/{model_slug}/page_{page_no:04d}.grid.pt"
        )

    def document_pages_prefix(self, workspace_id: str, doc_id: str) -> str:
        """Build the directory prefix holding a document's rendered page images.

        This is the parent directory of :meth:`page_image_path`. Used when
        reprocessing from the render phase to clear prior derived page images
        without touching the original uploaded file.

        Args:
            workspace_id: Workspace identifier.
            doc_id: Document identifier.

        Returns:
            Directory prefix string (no trailing slash).
        """
        return f"{self._base_path}/{workspace_id}/documents/{doc_id}/pages"

    def document_image_embeds_prefix(self, workspace_id: str, doc_id: str) -> str:
        """Build the directory prefix holding a document's image-embeds tensors.

        This is the parent-of-parent of :meth:`page_image_embeds_path` (which
        nests per ``model_slug``); deleting this prefix clears the precomputed
        image-embeds and grid tensors for every model.

        Args:
            workspace_id: Workspace identifier.
            doc_id: Document identifier.

        Returns:
            Directory prefix string (no trailing slash).
        """
        return f"{self._base_path}/{workspace_id}/documents/{doc_id}/image_embeds"

    def document_prompt_embeds_prefix(self, workspace_id: str, doc_id: str) -> str:
        """Build the legacy ``prompt_embeds`` directory prefix for a document.

        Retained so reprocessing can clear derived state written by older
        ingestion runs.

        Args:
            workspace_id: Workspace identifier.
            doc_id: Document identifier.

        Returns:
            Directory prefix string (no trailing slash).
        """
        return f"{self._base_path}/{workspace_id}/documents/{doc_id}/prompt_embeds"

    def document_transcripts_prefix(self, workspace_id: str, doc_id: str) -> str:
        """Build the directory prefix holding a document's page transcripts.

        This is the parent directory of :meth:`page_transcript_path`.

        Args:
            workspace_id: Workspace identifier.
            doc_id: Document identifier.

        Returns:
            Directory prefix string (no trailing slash).
        """
        return f"{self._base_path}/{workspace_id}/documents/{doc_id}/transcripts"

    def page_transcript_path(self, workspace_id: str, doc_id: str, page_no: int) -> str:
        """Build the storage path for a page transcript.

        Args:
            workspace_id: Workspace identifier.
            doc_id: Document identifier.
            page_no: Zero-indexed page number.

        Returns:
            Full storage path string.
        """
        return f"{self._base_path}/{workspace_id}/documents/{doc_id}/transcripts/page_{page_no:04d}.md"

    def page_figure_path(
        self, workspace_id: str, doc_id: str, page_no: int, figure_no: int,
    ) -> str:
        """Build the storage path for a figure cropped out of a page render.

        A grounded OCR model marks illustrations with a box but emits no text
        for them, so the crop is the only surviving representation of the
        figure. ``figure_no`` is 1-based and matches the ``[figure N]``
        placeholder in that page's transcript, which is what links the two.

        Args:
            workspace_id: Workspace identifier.
            doc_id: Document identifier.
            page_no: Zero-indexed page number.
            figure_no: 1-based figure index within the page.

        Returns:
            Full storage path string.
        """
        return (
            f"{self._base_path}/{workspace_id}/documents/{doc_id}"
            f"/figures/page_{page_no:04d}_fig_{figure_no:02d}.png"
        )

    # === Async I/O Operations ===

    async def store_file(self, path: str, data: bytes) -> str:
        """Store file data at the given path.

        Creates parent directories as needed. Uses asyncio.to_thread()
        to avoid blocking the event loop on synchronous fsspec I/O.

        Args:
            path: Full storage path.
            data: File content as bytes.

        Returns:
            The storage path where the file was written.

        Raises:
            OSError: If the write operation fails.
        """
        self.logger.debug("Storing %d bytes at %s", len(data), path)

        def _write():
            self._fs.makedirs(self._fs._parent(path), exist_ok=True)
            with self._fs.open(path, 'wb') as f:
                f.write(data)

        await asyncio.to_thread(_write)
        return path

    async def retrieve_file(self, path: str) -> bytes:
        """Retrieve file data from the given path.

        Args:
            path: Full storage path.

        Returns:
            File content as bytes.

        Raises:
            FileNotFoundError: If the path does not exist.
            OSError: If the read operation fails.
        """
        self.logger.debug("Retrieving file from %s", path)

        def _read():
            with self._fs.open(path, 'rb') as f:
                return f.read()

        return await asyncio.to_thread(_read)

    async def delete_tree(self, prefix: str) -> None:
        """Delete all files under the given prefix.

        Silently succeeds if the prefix does not exist.

        Args:
            prefix: Storage path prefix to delete recursively.
        """
        self.logger.debug("Deleting tree at %s", prefix)

        def _delete():
            if self._fs.exists(prefix):
                self._fs.rm(prefix, recursive=True)

        await asyncio.to_thread(_delete)

    async def exists(self, path: str) -> bool:
        """Check if a path exists in storage.

        Args:
            path: Full storage path to check.

        Returns:
            True if the path exists, False otherwise.
        """
        return await asyncio.to_thread(self._fs.exists, path)

    async def list_dir(self, path: str) -> list[str]:
        """List immediate child paths under a directory.

        Wraps the blocking fsspec ``ls`` in a thread, mirroring the other
        I/O methods. Returns an empty list when the path does not exist.

        Args:
            path: Full storage path of the directory to list.

        Returns:
            Sorted list of full child paths (files and subdirectories).
        """

        def _ls() -> list[str]:
            if not self._fs.exists(path):
                return []
            children = self._fs.ls(path, detail=False)
            # fsspec may include the directory itself; drop it and normalize.
            target = path.rstrip("/")
            return sorted(
                child.rstrip("/")
                for child in children
                if child.rstrip("/") != target
            )

        return await asyncio.to_thread(_ls)

    async def iter_document_dirs(self) -> list[tuple[str, str, str]]:
        """Enumerate every document directory under ``{base}/*/documents/*``.

        Walks one workspace level then one document level using ``ls`` so the
        scan stays shallow (no recursive listing of page/embedding blobs).

        Returns:
            List of ``(workspace_id, doc_id, full_path)`` tuples. Empty when
            the base path does not exist.
        """

        def _walk() -> list[tuple[str, str, str]]:
            base = self._base_path
            if not self._fs.exists(base):
                return []
            results: list[tuple[str, str, str]] = []
            for ws_path in self._fs.ls(base, detail=False):
                ws_path = ws_path.rstrip("/")
                if ws_path == base:
                    continue
                workspace_id = os.path.basename(ws_path)
                documents_path = f"{ws_path}/documents"
                if not self._fs.exists(documents_path):
                    continue
                for doc_path in self._fs.ls(documents_path, detail=False):
                    doc_path = doc_path.rstrip("/")
                    if doc_path == documents_path:
                        continue
                    doc_id = os.path.basename(doc_path)
                    results.append((workspace_id, doc_id, doc_path))
            return results

        return await asyncio.to_thread(_walk)

    async def newest_mtime(self, path: str) -> float | None:
        """Return the most recent modification time (epoch seconds) in a tree.

        Recursively inspects every file under ``path`` and returns the maximum
        modification timestamp. Returns ``None`` when the path is empty or no
        usable timestamp is available from the underlying filesystem (in which
        case callers should treat the directory as too uncertain to delete).

        Args:
            path: Full storage path to inspect.

        Returns:
            Newest mtime as epoch seconds, or ``None`` if undeterminable.
        """

        def _newest() -> float | None:
            if not self._fs.exists(path):
                return None
            try:
                infos = self._fs.find(path, detail=True)
            except (FileNotFoundError, OSError):
                return None
            if not infos:
                return None
            newest: float | None = None
            for info in infos.values():
                mtime = _extract_mtime(info)
                if mtime is None:
                    continue
                if newest is None or mtime > newest:
                    newest = mtime
            return newest

        return await asyncio.to_thread(_newest)


def _extract_mtime(info: dict) -> float | None:
    """Extract an epoch-seconds mtime from an fsspec info dict.

    Different fsspec backends report modification time under different keys
    (local: ``mtime`` float; S3: ``LastModified`` datetime). Returns ``None``
    when no recognizable timestamp is present.

    Args:
        info: An fsspec detail dict for a single entry.

    Returns:
        Modification time as epoch seconds, or ``None``.
    """
    raw = info.get("mtime")
    if raw is None:
        raw = info.get("LastModified")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    # datetime-like (S3 LastModified)
    timestamp = getattr(raw, "timestamp", None)
    if callable(timestamp):
        try:
            return float(timestamp())
        except (TypeError, ValueError, OSError):
            return None
    return None


class BlobStoragePlugin(BlobStoragePluginBase):
    """Plugin for default blob storage service."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> BlobStorageService:
        """Initialize blob storage from configuration.

        Reads storage type (local or s3) and connection parameters from
        environment/config, constructs the appropriate fsspec filesystem,
        and returns a BlobStorageService.

        Args:
            v: Variables instance for configuration access.
            logger: Logger instance.

        Returns:
            Configured BlobStorageService instance.
        """
        storage_type = v.environ(
            MEMORYLAYER_BLOB_STORAGE_TYPE,
            default=DEFAULT_MEMORYLAYER_BLOB_STORAGE_TYPE,
        )
        base_path = v.environ(
            MEMORYLAYER_BLOB_STORAGE_BASE_PATH,
            default=DEFAULT_MEMORYLAYER_BLOB_STORAGE_BASE_PATH,
        )

        if storage_type == 's3':
            endpoint_url = v.environ(MEMORYLAYER_BLOB_S3_ENDPOINT_URL, default=None)
            access_key = v.environ(MEMORYLAYER_BLOB_S3_ACCESS_KEY, default=None)
            secret_key = v.environ(MEMORYLAYER_BLOB_S3_SECRET_KEY, default=None)
            region = v.environ(MEMORYLAYER_BLOB_S3_REGION, default=None)

            client_kwargs = {}
            if endpoint_url:
                client_kwargs['endpoint_url'] = endpoint_url
            if region:
                client_kwargs['region_name'] = region

            fs = fsspec.filesystem(
                's3',
                key=access_key,
                secret=secret_key,
                client_kwargs=client_kwargs if client_kwargs else None,
            )
            logger.info(
                "Initialized S3 blob storage: base_path=%s, endpoint=%s",
                base_path,
                endpoint_url or 'default',
            )
        else:
            fs = fsspec.filesystem('file')
            logger.info("Initialized local blob storage: base_path=%s", base_path)

        return BlobStorageService(fs=fs, base_path=base_path, logger=logger)
