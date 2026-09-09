"""Unit tests for the source_vfs_ref field on Document models.

Verifies:
- OSS Document model accepts source_vfs_ref
- Enterprise Document model accepts source_vfs_ref
- Default is None (backward-compatible)
- PENDING_FETCH status enum value exists
"""
import pytest

from memorylayer_server.models.document import (
    Document as OSSDocument,
    DocumentStatus as OSSDocumentStatus,
    DocumentType as OSSDocumentType,
)
from memorylayer_saas.models.document import (
    Document as EntDocument,
    DocumentStatus as EntDocumentStatus,
    DocumentType as EntDocumentType,
)


class TestOSSDocumentSourceVfsRef:
    """Tests for source_vfs_ref on the OSS Document model."""

    def test_source_vfs_ref_default_is_none(self):
        doc = OSSDocument(
            id="doc_test",
            workspace_id="ws_test",
            filename="test.pdf",
            document_type=OSSDocumentType.PDF,
            content_hash="abc123",
            size_bytes=1024,
        )
        assert doc.source_vfs_ref is None

    def test_source_vfs_ref_can_be_set(self):
        doc = OSSDocument(
            id="doc_test",
            workspace_id="ws_test",
            filename="test.pdf",
            document_type=OSSDocumentType.PDF,
            content_hash="abc123",
            source_vfs_ref="vfs_ref_123",
            size_bytes=1024,
        )
        assert doc.source_vfs_ref == "vfs_ref_123"

    def test_pending_fetch_status_exists(self):
        assert OSSDocumentStatus.PENDING_FETCH == "pending_fetch"

    def test_pending_fetch_is_valid_status(self):
        doc = OSSDocument(
            id="doc_test",
            workspace_id="ws_test",
            filename="test.pdf",
            document_type=OSSDocumentType.PDF,
            content_hash="abc123",
            size_bytes=1024,
            status=OSSDocumentStatus.PENDING_FETCH,
        )
        assert doc.status == OSSDocumentStatus.PENDING_FETCH


class TestEnterpriseDocumentSourceVfsRef:
    """Tests for source_vfs_ref on the Enterprise Document model."""

    def test_source_vfs_ref_default_is_none(self):
        doc = EntDocument(
            id="doc_test",
            workspace_id="ws_test",
            filename="test.pdf",
            document_type=EntDocumentType.PDF,
            content_hash="abc123",
            size_bytes=1024,
        )
        assert doc.source_vfs_ref is None

    def test_source_vfs_ref_can_be_set(self):
        doc = EntDocument(
            id="doc_test",
            workspace_id="ws_test",
            filename="test.pdf",
            document_type=EntDocumentType.PDF,
            content_hash="abc123",
            source_vfs_ref="vfs_ref_456",
            size_bytes=1024,
        )
        assert doc.source_vfs_ref == "vfs_ref_456"

    def test_pending_fetch_status_exists(self):
        assert EntDocumentStatus.PENDING_FETCH == "pending_fetch"

    def test_pending_fetch_is_valid_status(self):
        doc = EntDocument(
            id="doc_test",
            workspace_id="ws_test",
            filename="test.pdf",
            document_type=EntDocumentType.PDF,
            content_hash="abc123",
            size_bytes=1024,
            status=EntDocumentStatus.PENDING_FETCH,
        )
        assert doc.status == EntDocumentStatus.PENDING_FETCH


class TestMigration016Structure:
    """Verify the migration file is well-formed and importable."""

    def test_migration_revision_chain(self):
        """Migration 016 follows 015 in the revision chain."""
        import importlib
        m016 = importlib.import_module("migrations.versions.016_add_source_vfs_ref")
        assert m016.revision == "016"
        assert m016.down_revision == "015"

    def test_migration_has_upgrade_and_downgrade(self):
        """Migration has both upgrade() and downgrade() functions."""
        import importlib
        mod = importlib.import_module("migrations.versions.016_add_source_vfs_ref")
        assert callable(getattr(mod, "upgrade", None))
        assert callable(getattr(mod, "downgrade", None))
