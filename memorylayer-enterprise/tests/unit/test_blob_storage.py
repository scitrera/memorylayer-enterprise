"""
Unit tests for BlobStorageService.

Tests:
- Path convention methods: document_path, page_image_path, page_transcript_path
- store_file: async write via fsspec
- retrieve_file: async read via fsspec
- delete_tree: recursive delete, silent on missing prefix
- exists: async path existence check
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, mock_open

from memorylayer_saas.services.document.blob_storage import BlobStorageService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_fs():
    """Create a mock fsspec AbstractFileSystem."""
    fs = MagicMock()
    fs.exists.return_value = True
    fs._parent.return_value = "/blobs/ws1/documents/doc1"
    return fs


@pytest.fixture
def blob_service(mock_fs):
    """Create a BlobStorageService backed by the mock filesystem."""
    return BlobStorageService(
        fs=mock_fs,
        base_path="/blobs",
        logger=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Path conventions
# ---------------------------------------------------------------------------

class TestPathConventions:
    """Tests for path-building helper methods."""

    def test_document_path(self, blob_service):
        """Test that document_path builds the correct storage path."""
        path = blob_service.document_path("ws1", "doc1", "report.pdf")
        assert path == "/blobs/ws1/documents/doc1/report.pdf"

    def test_document_path_strips_trailing_slash_from_base(self):
        """Test that a trailing slash on base_path is stripped correctly."""
        service = BlobStorageService(
            fs=MagicMock(),
            base_path="/blobs/",
            logger=MagicMock(),
        )
        path = service.document_path("ws1", "doc1", "file.txt")
        assert path == "/blobs/ws1/documents/doc1/file.txt"

    def test_page_image_path_zero_padded(self, blob_service):
        """Test that page_image_path zero-pads the page number to 4 digits."""
        path = blob_service.page_image_path("ws1", "doc1", 3)
        assert path == "/blobs/ws1/documents/doc1/pages/page_0003.png"

    def test_page_image_path_large_page_number(self, blob_service):
        """Test page_image_path with a 4-digit page number."""
        path = blob_service.page_image_path("ws1", "doc1", 1234)
        assert path == "/blobs/ws1/documents/doc1/pages/page_1234.png"

    def test_page_image_path_page_zero(self, blob_service):
        """Test page_image_path for the first page (index 0)."""
        path = blob_service.page_image_path("ws1", "doc1", 0)
        assert path == "/blobs/ws1/documents/doc1/pages/page_0000.png"

    def test_page_image_embeds_path_keyed_by_model(self, blob_service):
        """page_image_embeds_path nests under a model slug and zero-pads page."""
        path = blob_service.page_image_embeds_path(
            "ws1", "doc1", 3, "qwen--qwen3.6-27b-fp8",
        )
        assert path == (
            "/blobs/ws1/documents/doc1/image_embeds/"
            "qwen--qwen3.6-27b-fp8/page_0003.pt.zst"
        )

    def test_page_image_embeds_path_distinct_per_model(self, blob_service):
        """Different model slugs yield different paths for the same page."""
        a = blob_service.page_image_embeds_path("ws1", "doc1", 0, "model-a")
        b = blob_service.page_image_embeds_path("ws1", "doc1", 0, "model-b")
        assert a != b
        assert a.endswith("/image_embeds/model-a/page_0000.pt.zst")
        assert b.endswith("/image_embeds/model-b/page_0000.pt.zst")

    def test_page_image_grid_path_keyed_by_model(self, blob_service):
        """page_image_grid_path nests under a model slug and zero-pads page."""
        path = blob_service.page_image_grid_path(
            "ws1", "doc1", 3, "qwen--qwen3.6-27b-fp8",
        )
        assert path == (
            "/blobs/ws1/documents/doc1/image_embeds/"
            "qwen--qwen3.6-27b-fp8/page_0003.grid.pt"
        )

    def test_page_transcript_path(self, blob_service):
        """Test that page_transcript_path builds the correct storage path."""
        path = blob_service.page_transcript_path("ws1", "doc1", 2)
        assert path == "/blobs/ws1/documents/doc1/transcripts/page_0002.md"

    def test_page_transcript_path_zero_padded(self, blob_service):
        """Test zero-padding for transcript path page number."""
        path = blob_service.page_transcript_path("ws_abc", "doc_xyz", 7)
        assert path == "/blobs/ws_abc/documents/doc_xyz/transcripts/page_0007.md"

    def test_different_workspace_and_doc_ids(self, blob_service):
        """Test path methods with varied workspace and document identifiers."""
        assert blob_service.document_path("team_a", "doc_99", "data.csv") == \
            "/blobs/team_a/documents/doc_99/data.csv"
        assert blob_service.page_image_path("team_b", "doc_00", 10) == \
            "/blobs/team_b/documents/doc_00/pages/page_0010.png"


# ---------------------------------------------------------------------------
# store_file
# ---------------------------------------------------------------------------

class TestStoreFile:
    """Tests for the async store_file() I/O method."""

    @pytest.mark.asyncio
    async def test_store_file_success(self, blob_service, mock_fs):
        """Test that store_file writes data and returns the path."""
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        mock_fs.open.return_value = cm

        path = "/blobs/ws1/documents/doc1/test.pdf"
        result = await blob_service.store_file(path, b"fake pdf content")

        assert result == path
        mock_fs.makedirs.assert_called_once()
        mock_fs.open.assert_called_once_with(path, "wb")
        cm.write.assert_called_once_with(b"fake pdf content")

    @pytest.mark.asyncio
    async def test_store_file_calls_makedirs_on_parent(self, blob_service, mock_fs):
        """Test that store_file creates parent directories before writing."""
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        mock_fs.open.return_value = cm
        mock_fs._parent.return_value = "/blobs/ws1/documents/doc1"

        await blob_service.store_file("/blobs/ws1/documents/doc1/file.txt", b"data")

        mock_fs.makedirs.assert_called_once_with(
            "/blobs/ws1/documents/doc1", exist_ok=True
        )

    @pytest.mark.asyncio
    async def test_store_file_returns_path_unchanged(self, blob_service, mock_fs):
        """Test that the exact path passed in is returned."""
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        mock_fs.open.return_value = cm

        target_path = "/blobs/some/deep/nested/path.png"
        result = await blob_service.store_file(target_path, b"\x89PNG")

        assert result == target_path


# ---------------------------------------------------------------------------
# retrieve_file
# ---------------------------------------------------------------------------

class TestRetrieveFile:
    """Tests for the async retrieve_file() I/O method."""

    @pytest.mark.asyncio
    async def test_retrieve_file_success(self, blob_service, mock_fs):
        """Test that retrieve_file reads and returns file bytes."""
        expected_data = b"retrieved content"
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.read.return_value = expected_data
        mock_fs.open.return_value = cm

        result = await blob_service.retrieve_file("/blobs/ws1/documents/doc1/file.txt")

        assert result == expected_data
        mock_fs.open.assert_called_once_with(
            "/blobs/ws1/documents/doc1/file.txt", "rb"
        )

    @pytest.mark.asyncio
    async def test_retrieve_file_returns_bytes(self, blob_service, mock_fs):
        """Test that retrieve_file returns raw bytes."""
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.read.return_value = b"\x00\x01\x02\x03"
        mock_fs.open.return_value = cm

        result = await blob_service.retrieve_file("/blobs/test.bin")

        assert isinstance(result, bytes)
        assert result == b"\x00\x01\x02\x03"


# ---------------------------------------------------------------------------
# delete_tree
# ---------------------------------------------------------------------------

class TestDeleteTree:
    """Tests for the async delete_tree() I/O method."""

    @pytest.mark.asyncio
    async def test_delete_tree_success(self, blob_service, mock_fs):
        """Test that delete_tree calls fs.rm when the prefix exists."""
        mock_fs.exists.return_value = True

        await blob_service.delete_tree("/blobs/ws1/documents/doc1")

        mock_fs.rm.assert_called_once_with(
            "/blobs/ws1/documents/doc1", recursive=True
        )

    @pytest.mark.asyncio
    async def test_delete_tree_nonexistent_is_silent(self, blob_service, mock_fs):
        """Test that delete_tree succeeds silently when the prefix does not exist."""
        mock_fs.exists.return_value = False

        # Should not raise
        await blob_service.delete_tree("/blobs/nonexistent/path")

        mock_fs.rm.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_tree_checks_existence_first(self, blob_service, mock_fs):
        """Test that delete_tree checks existence before attempting removal."""
        mock_fs.exists.return_value = True
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        mock_fs.open.return_value = cm

        prefix = "/blobs/ws2/documents/doc_abc"
        await blob_service.delete_tree(prefix)

        mock_fs.exists.assert_called_once_with(prefix)


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------

class TestExists:
    """Tests for the async exists() method."""

    @pytest.mark.asyncio
    async def test_exists_returns_true_when_present(self, blob_service, mock_fs):
        """Test that exists() returns True for a path that exists."""
        mock_fs.exists.return_value = True

        result = await blob_service.exists("/blobs/ws1/documents/doc1/file.pdf")

        assert result is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_when_absent(self, blob_service, mock_fs):
        """Test that exists() returns False for a path that does not exist."""
        mock_fs.exists.return_value = False

        result = await blob_service.exists("/blobs/missing/path.txt")

        assert result is False

    @pytest.mark.asyncio
    async def test_exists_delegates_to_fs(self, blob_service, mock_fs):
        """Test that exists() calls fs.exists with the correct path."""
        mock_fs.exists.return_value = True
        target = "/blobs/ws1/documents/doc1/pages/page_0001.png"

        await blob_service.exists(target)

        mock_fs.exists.assert_called_with(target)
