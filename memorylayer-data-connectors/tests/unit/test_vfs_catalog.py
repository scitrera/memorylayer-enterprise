"""Unit tests for VFS catalog CRUD invariants."""
from __future__ import annotations

import pytest

from data_connectors.vfs.catalog import VfsCatalog


@pytest.fixture
def catalog():
    return VfsCatalog()


class TestVfsCatalogRegister:
    """Verify VFS entry registration."""

    @pytest.mark.asyncio
    async def test_register_creates_entry(self, catalog):
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="manual_upload",
            source_path="test.pdf",
            content_hash="abc123",
        )
        assert entry.vfs_ref.startswith("vfs_")
        assert entry.workspace_id == "ws-1"
        assert entry.connector_id == "manual_upload"
        assert entry.source_path == "test.pdf"
        assert entry.content_hash == "abc123"

    @pytest.mark.asyncio
    async def test_register_generates_unique_refs(self, catalog):
        e1 = await catalog.register(
            workspace_id="ws-1", connector_id="c1",
            source_path="a.txt", content_hash="h1",
        )
        e2 = await catalog.register(
            workspace_id="ws-1", connector_id="c1",
            source_path="b.txt", content_hash="h2",
        )
        assert e1.vfs_ref != e2.vfs_ref

    @pytest.mark.asyncio
    async def test_register_with_optional_fields(self, catalog):
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="s3",
            source_path="data/file.csv",
            content_hash="xyz789",
            content_type="text/csv",
            size_bytes=1024,
            blob_key="ws-1/s3/abc/file.csv",
            metadata={"source": "test"},
        )
        assert entry.content_type == "text/csv"
        assert entry.size_bytes == 1024
        assert entry.blob_key == "ws-1/s3/abc/file.csv"
        assert entry.metadata == {"source": "test"}


class TestVfsCatalogGet:
    """Verify VFS entry retrieval."""

    @pytest.mark.asyncio
    async def test_get_existing_entry(self, catalog):
        entry = await catalog.register(
            workspace_id="ws-1", connector_id="c1",
            source_path="test.pdf", content_hash="h1",
        )
        result = await catalog.get(entry.vfs_ref)
        assert result is not None
        assert result.vfs_ref == entry.vfs_ref

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, catalog):
        result = await catalog.get("vfs_doesnotexist")
        assert result is None


class TestVfsCatalogList:
    """Verify VFS entry listing and filtering."""

    @pytest.mark.asyncio
    async def test_list_filters_by_workspace(self, catalog):
        await catalog.register(workspace_id="ws-1", connector_id="c1", source_path="a.txt", content_hash="h1")
        await catalog.register(workspace_id="ws-2", connector_id="c1", source_path="b.txt", content_hash="h2")
        await catalog.register(workspace_id="ws-1", connector_id="c1", source_path="c.txt", content_hash="h3")

        entries, total = await catalog.list_entries(workspace_id="ws-1")
        assert total == 2
        assert len(entries) == 2
        assert all(e.workspace_id == "ws-1" for e in entries)

    @pytest.mark.asyncio
    async def test_list_filters_by_connector(self, catalog):
        await catalog.register(workspace_id="ws-1", connector_id="s3", source_path="a.txt", content_hash="h1")
        await catalog.register(workspace_id="ws-1", connector_id="manual_upload", source_path="b.txt", content_hash="h2")

        entries, total = await catalog.list_entries(workspace_id="ws-1", connector_id="s3")
        assert total == 1
        assert entries[0].connector_id == "s3"

    @pytest.mark.asyncio
    async def test_list_pagination(self, catalog):
        for i in range(5):
            await catalog.register(workspace_id="ws-1", connector_id="c1", source_path=f"f{i}.txt", content_hash=f"h{i}")

        entries, total = await catalog.list_entries(workspace_id="ws-1", limit=2, offset=0)
        assert total == 5
        assert len(entries) == 2

        entries2, _ = await catalog.list_entries(workspace_id="ws-1", limit=2, offset=2)
        assert len(entries2) == 2

    @pytest.mark.asyncio
    async def test_list_empty_workspace(self, catalog):
        entries, total = await catalog.list_entries(workspace_id="empty")
        assert total == 0
        assert entries == []


class TestVfsCatalogUpdate:
    """Verify VFS entry updates."""

    @pytest.mark.asyncio
    async def test_update_content_hash(self, catalog):
        entry = await catalog.register(workspace_id="ws-1", connector_id="c1", source_path="a.txt", content_hash="old")
        updated = await catalog.update(entry.vfs_ref, content_hash="new")
        assert updated is not None
        assert updated.content_hash == "new"
        assert updated.updated_at > entry.created_at

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_none(self, catalog):
        result = await catalog.update("vfs_nope", content_hash="new")
        assert result is None


class TestVfsCatalogLink:
    """Verify MemoryLayer document linking."""

    @pytest.mark.asyncio
    async def test_link_ml_document(self, catalog):
        entry = await catalog.register(workspace_id="ws-1", connector_id="c1", source_path="a.txt", content_hash="h1")
        linked = await catalog.link_ml_document(entry.vfs_ref, ml_doc_id="doc_123", ml_job_id="job_456")
        assert linked is not None
        assert linked.ml_doc_id == "doc_123"
        assert linked.ml_job_id == "job_456"

    @pytest.mark.asyncio
    async def test_link_nonexistent_returns_none(self, catalog):
        result = await catalog.link_ml_document("vfs_nope", ml_doc_id="doc_1")
        assert result is None


class TestVfsCatalogDelete:
    """Verify VFS entry deletion."""

    @pytest.mark.asyncio
    async def test_delete_existing(self, catalog):
        entry = await catalog.register(workspace_id="ws-1", connector_id="c1", source_path="a.txt", content_hash="h1")
        assert await catalog.delete(entry.vfs_ref) is True
        assert await catalog.get(entry.vfs_ref) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, catalog):
        assert await catalog.delete("vfs_nope") is False


class TestVfsCatalogDedup:
    """Verify content-hash-based dedup lookup."""

    @pytest.mark.asyncio
    async def test_find_by_content_hash(self, catalog):
        entry = await catalog.register(workspace_id="ws-1", connector_id="c1", source_path="a.txt", content_hash="unique_hash")
        found = await catalog.find_by_content_hash("ws-1", "unique_hash")
        assert found is not None
        assert found.vfs_ref == entry.vfs_ref

    @pytest.mark.asyncio
    async def test_find_by_content_hash_wrong_workspace(self, catalog):
        await catalog.register(workspace_id="ws-1", connector_id="c1", source_path="a.txt", content_hash="h1")
        found = await catalog.find_by_content_hash("ws-2", "h1")
        assert found is None

    @pytest.mark.asyncio
    async def test_find_by_content_hash_not_found(self, catalog):
        found = await catalog.find_by_content_hash("ws-1", "nonexistent")
        assert found is None


class TestVfsCatalogSourcePathPrefix:
    """Verify source_path prefix filtering (folder-style grouping).

    Without this filter a caller has to page an entire workspace and filter
    client-side. Worse, an unsupported filter param is silently dropped by
    FastAPI, so the caller receives EVERY entry while believing the query was
    scoped — which is exactly how one app ended up showing every workspace file
    under every one of its entities.
    """

    async def _seed(self, catalog):
        for path in (
            "/Bids/acme/proposal.pdf",
            "/Bids/acme/pricing.xlsx",
            "/Bids/globex/proposal.pdf",
            "/RFP/default/rfp.pdf",
            "bare-filename.pdf",
        ):
            await catalog.register(
                workspace_id="ws-1",
                connector_id="manual_upload",
                source_path=path,
                content_hash="h",
            )

    @pytest.mark.asyncio
    async def test_prefix_scopes_to_one_entity(self, catalog):
        await self._seed(catalog)
        entries, total = await catalog.list_entries(
            workspace_id="ws-1", source_path_prefix="/Bids/acme/",
        )
        assert total == 2
        assert {e.source_path for e in entries} == {
            "/Bids/acme/proposal.pdf", "/Bids/acme/pricing.xlsx",
        }

    @pytest.mark.asyncio
    async def test_prefix_does_not_leak_sibling_entities(self, catalog):
        """The bug this guards: /Bids/acme must not return globex's files."""
        await self._seed(catalog)
        entries, _ = await catalog.list_entries(
            workspace_id="ws-1", source_path_prefix="/Bids/acme/",
        )
        assert all("globex" not in e.source_path for e in entries)

    @pytest.mark.asyncio
    async def test_parent_prefix_returns_all_children(self, catalog):
        await self._seed(catalog)
        _, total = await catalog.list_entries(
            workspace_id="ws-1", source_path_prefix="/Bids/",
        )
        assert total == 3

    @pytest.mark.asyncio
    async def test_omitted_prefix_returns_everything(self, catalog):
        """Absent filter must stay a no-op — every existing caller relies on it."""
        await self._seed(catalog)
        _, total = await catalog.list_entries(workspace_id="ws-1")
        assert total == 5

    @pytest.mark.asyncio
    async def test_no_match_returns_empty_not_everything(self, catalog):
        """A non-matching prefix must return 0, never fall back to unfiltered."""
        await self._seed(catalog)
        entries, total = await catalog.list_entries(
            workspace_id="ws-1", source_path_prefix="/Nope/",
        )
        assert total == 0 and entries == []

    @pytest.mark.asyncio
    async def test_prefix_combines_with_connector_filter(self, catalog):
        await self._seed(catalog)
        await catalog.register(
            workspace_id="ws-1", connector_id="agent_generated",
            source_path="/Bids/acme/summary.docx", content_hash="h",
        )
        _, total = await catalog.list_entries(
            workspace_id="ws-1",
            connector_id="agent_generated",
            source_path_prefix="/Bids/acme/",
        )
        assert total == 1

    @pytest.mark.asyncio
    async def test_prefix_still_scoped_by_workspace(self, catalog):
        await self._seed(catalog)
        await catalog.register(
            workspace_id="ws-2", connector_id="manual_upload",
            source_path="/Bids/acme/other.pdf", content_hash="h",
        )
        _, total = await catalog.list_entries(
            workspace_id="ws-1", source_path_prefix="/Bids/acme/",
        )
        assert total == 2


class TestVfsCatalogConnectorFilters:
    """Verify connector allow/deny filtering.

    The deny-list exists so a general file view can say "everything except the
    agent-generated noise" without enumerating every connector that exists —
    including ones added later.
    """

    async def _seed(self, catalog):
        for connector, path in (
            ('manual_upload', 'report.pdf'),
            ('manual_upload', 'notes.md'),
            ('gdrive', 'sheet.xlsx'),
            ('sahara_artifact', 'scratch.csv'),
            ('agent_generated', 'summary.docx'),
        ):
            await catalog.register(
                workspace_id='ws-1', connector_id=connector,
                source_path=path, content_hash='h',
            )

    @pytest.mark.asyncio
    async def test_exclude_hides_named_connectors(self, catalog):
        await self._seed(catalog)
        entries, total = await catalog.list_entries(
            workspace_id='ws-1',
            exclude_connector_ids=['sahara_artifact', 'agent_generated'],
        )
        assert total == 3
        assert {e.connector_id for e in entries} == {'manual_upload', 'gdrive'}

    @pytest.mark.asyncio
    async def test_include_selects_only_named_connectors(self, catalog):
        """The source-picker case: show ONLY the agent artifacts."""
        await self._seed(catalog)
        entries, total = await catalog.list_entries(
            workspace_id='ws-1', connector_ids=['sahara_artifact'],
        )
        assert total == 1
        assert entries[0].connector_id == 'sahara_artifact'

    @pytest.mark.asyncio
    async def test_exclude_wins_over_include(self, catalog):
        """Deny is applied last, so it overrides an allow-list."""
        await self._seed(catalog)
        _, total = await catalog.list_entries(
            workspace_id='ws-1',
            connector_ids=['manual_upload', 'sahara_artifact'],
            exclude_connector_ids=['sahara_artifact'],
        )
        assert total == 2

    @pytest.mark.asyncio
    async def test_no_connector_filters_returns_everything(self, catalog):
        await self._seed(catalog)
        _, total = await catalog.list_entries(workspace_id='ws-1')
        assert total == 5

    @pytest.mark.asyncio
    async def test_connector_filters_combine_with_path_prefix(self, catalog):
        await self._seed(catalog)
        await catalog.register(
            workspace_id='ws-1', connector_id='manual_upload',
            source_path='/Bids/acme/bid.pdf', content_hash='h',
        )
        _, total = await catalog.list_entries(
            workspace_id='ws-1',
            source_path_prefix='/Bids/acme/',
            exclude_connector_ids=['sahara_artifact', 'agent_generated'],
        )
        assert total == 1
