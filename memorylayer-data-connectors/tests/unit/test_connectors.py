# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for connector implementations."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_connectors.vfs.catalog import VfsCatalog
from data_connectors.connectors import ConnectorRegistry
from data_connectors.connectors.manual_upload import ManualUploadConnector


@pytest.fixture
def catalog():
    return VfsCatalog()


@pytest.fixture
def mock_blob_store():
    store = AsyncMock()
    store.generate_download_url = AsyncMock(return_value=("https://s3.example.com/blob", {}, None))
    store.put_object = AsyncMock(return_value=100)
    return store


class TestManualUploadConnector:
    """Verify the manual upload connector contract."""

    @pytest.mark.asyncio
    async def test_load_is_noop(self, mock_blob_store, catalog):
        connector = ManualUploadConnector(mock_blob_store, catalog)
        await connector.load()  # Should not raise

    @pytest.mark.asyncio
    async def test_poll_returns_empty(self, mock_blob_store, catalog):
        connector = ManualUploadConnector(mock_blob_store, catalog)
        result = await connector.poll()
        assert result == []

    @pytest.mark.asyncio
    async def test_register_upload_complete(self, mock_blob_store, catalog):
        connector = ManualUploadConnector(mock_blob_store, catalog)
        vfs_ref = await connector.register_upload_complete(
            workspace_id="ws-1",
            blob_key="ws-1/manual_upload/abc/test.pdf",
            filename="test.pdf",
            content_hash="sha256_xyz",
            content_type="application/pdf",
            size_bytes=2048,
        )
        assert vfs_ref.startswith("vfs_")

        # Verify the entry was registered in the catalog
        entry = await catalog.get(vfs_ref)
        assert entry is not None
        assert entry.workspace_id == "ws-1"
        assert entry.blob_key == "ws-1/manual_upload/abc/test.pdf"
        assert entry.source_path == "test.pdf"
        assert entry.content_hash == "sha256_xyz"

    @pytest.mark.asyncio
    async def test_get_content_url_existing(self, mock_blob_store, catalog):
        connector = ManualUploadConnector(mock_blob_store, catalog)
        vfs_ref = await connector.register_upload_complete(
            workspace_id="ws-1",
            blob_key="ws-1/manual_upload/abc/test.pdf",
            filename="test.pdf",
            content_hash="h1",
        )
        url = await connector.get_content_url(vfs_ref)
        assert url == "https://s3.example.com/blob"

    @pytest.mark.asyncio
    async def test_get_content_url_nonexistent(self, mock_blob_store, catalog):
        connector = ManualUploadConnector(mock_blob_store, catalog)
        url = await connector.get_content_url("vfs_nope")
        assert url is None

    @pytest.mark.asyncio
    async def test_get_metadata(self, mock_blob_store, catalog):
        connector = ManualUploadConnector(mock_blob_store, catalog)
        vfs_ref = await connector.register_upload_complete(
            workspace_id="ws-1",
            blob_key="ws-1/manual_upload/abc/test.pdf",
            filename="test.pdf",
            content_hash="h1",
            content_type="application/pdf",
            size_bytes=1024,
        )
        meta = await connector.get_metadata(vfs_ref)
        assert meta["connector_type"] == "manual_upload"
        assert meta["source_path"] == "test.pdf"
        assert meta["content_type"] == "application/pdf"

    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent(self, mock_blob_store, catalog):
        connector = ManualUploadConnector(mock_blob_store, catalog)
        meta = await connector.get_metadata("vfs_nope")
        assert meta == {}


# ---------------------------------------------------------------------------
# ConnectorRegistry tests
# ---------------------------------------------------------------------------


class TestConnectorRegistry:
    """Verify the connector registry discovers all self-registered connectors."""

    def test_manual_upload_registered(self):
        """manual_upload should be registered at import time."""
        cls = ConnectorRegistry.get("manual_upload")
        assert cls is ManualUploadConnector

    def test_s3_registered(self):
        """s3 should be registered at import time."""
        # Force module import so registration fires
        import data_connectors.connectors.s3  # noqa: F401
        cls = ConnectorRegistry.get("s3")
        assert cls is not None
        assert cls.__name__ == "S3Connector"

    def test_web_scraper_registered(self):
        import data_connectors.connectors.web_scraper  # noqa: F401
        cls = ConnectorRegistry.get("web_scraper")
        assert cls is not None
        assert cls.__name__ == "WebScraperConnector"

    def test_gdrive_registered(self):
        import data_connectors.connectors.gdrive  # noqa: F401
        cls = ConnectorRegistry.get("gdrive")
        assert cls is not None
        assert cls.__name__ == "GoogleDriveConnector"

    def test_dropbox_registered(self):
        import data_connectors.connectors.dropbox  # noqa: F401
        cls = ConnectorRegistry.get("dropbox")
        assert cls is not None
        assert cls.__name__ == "DropboxConnector"

    def test_slack_registered(self):
        import data_connectors.connectors.slack  # noqa: F401
        cls = ConnectorRegistry.get("slack")
        assert cls is not None
        assert cls.__name__ == "SlackConnector"

    def test_teams_registered(self):
        import data_connectors.connectors.teams  # noqa: F401
        cls = ConnectorRegistry.get("teams")
        assert cls is not None
        assert cls.__name__ == "TeamsConnector"

    def test_discord_registered(self):
        import data_connectors.connectors.discord  # noqa: F401
        cls = ConnectorRegistry.get("discord")
        assert cls is not None
        assert cls.__name__ == "DiscordConnector"

    def test_github_registered(self):
        import data_connectors.connectors.github  # noqa: F401
        cls = ConnectorRegistry.get("github")
        assert cls is not None
        assert cls.__name__ == "GitHubConnector"

    def test_local_fs_registered(self):
        import data_connectors.connectors.local_fs  # noqa: F401
        cls = ConnectorRegistry.get("local_fs")
        assert cls is not None
        assert cls.__name__ == "LocalFsConnector"

    def test_list_connectors_includes_all(self):
        """All ten connectors should appear in list_connectors."""
        # Ensure all modules are imported
        import data_connectors.connectors.s3  # noqa: F401
        import data_connectors.connectors.web_scraper  # noqa: F401
        import data_connectors.connectors.gdrive  # noqa: F401
        import data_connectors.connectors.dropbox  # noqa: F401
        import data_connectors.connectors.slack  # noqa: F401
        import data_connectors.connectors.teams  # noqa: F401
        import data_connectors.connectors.discord  # noqa: F401
        import data_connectors.connectors.github  # noqa: F401
        import data_connectors.connectors.local_fs  # noqa: F401

        names = ConnectorRegistry.list_connectors()
        expected = {"discord", "dropbox", "gdrive", "github", "local_fs", "manual_upload", "s3", "slack", "teams", "web_scraper"}
        assert expected.issubset(set(names))

    def test_get_nonexistent(self):
        assert ConnectorRegistry.get("nonexistent_connector_xyz") is None


# ---------------------------------------------------------------------------
# Google Drive connector tests
# ---------------------------------------------------------------------------


class TestGoogleDriveConnector:
    """Tests for the Google Drive connector."""

    @pytest.mark.asyncio
    async def test_load_validates_credentials(self):
        from data_connectors.connectors.gdrive import GoogleDriveConnector

        # Mock the Google API service
        mock_service = MagicMock()
        mock_files = MagicMock()
        mock_list = MagicMock()
        mock_list.execute.return_value = {"files": []}
        mock_files.list.return_value = mock_list
        mock_service.files.return_value = mock_files

        with patch("data_connectors.connectors.gdrive._build_service", return_value=mock_service):
            connector = GoogleDriveConnector(
                credentials={"oauth_token": "test-token"},
                settings={"folder_id": "abc123"},
            )
            await connector.load()
            mock_files.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_returns_entries(self):
        from data_connectors.connectors.gdrive import GoogleDriveConnector

        mock_service = MagicMock()
        mock_files = MagicMock()
        mock_list = MagicMock()
        mock_list.execute.return_value = {
            "files": [
                {
                    "id": "file1",
                    "name": "doc.pdf",
                    "mimeType": "application/pdf",
                    "modifiedTime": "2025-01-15T10:30:00Z",
                    "md5Checksum": "abc123",
                    "size": "1024",
                    "webViewLink": "https://drive.google.com/file/d/file1",
                    "owners": [{"displayName": "Alice Owner", "emailAddress": "alice@example.test"}],
                },
            ],
        }
        mock_files.list.return_value = mock_list
        mock_service.files.return_value = mock_files

        with patch("data_connectors.connectors.gdrive._build_service", return_value=mock_service):
            connector = GoogleDriveConnector(
                credentials={"oauth_token": "test-token"},
            )
            entries = await connector.poll()
            assert len(entries) == 1
            assert entries[0]["source_path"] == "doc.pdf"
            assert entries[0]["content_type"] == "application/pdf"
            assert entries[0]["size_bytes"] == 1024
            assert entries[0]["metadata"]["gdrive_file_id"] == "file1"
            assert entries[0]["metadata"]["owners"][0]["displayName"] == "Alice Owner"

    @pytest.mark.asyncio
    async def test_poll_google_native_doc_uses_modified_time_hash(self):
        from data_connectors.connectors.gdrive import GoogleDriveConnector

        mock_service = MagicMock()
        mock_files = MagicMock()
        mock_list = MagicMock()
        mock_list.execute.return_value = {
            "files": [
                {
                    "id": "gdoc1",
                    "name": "My Document",
                    "mimeType": "application/vnd.google-apps.document",
                    "modifiedTime": "2025-02-01T08:00:00Z",
                    "webViewLink": "https://docs.google.com/document/d/gdoc1",
                },
            ],
        }
        mock_files.list.return_value = mock_list
        mock_service.files.return_value = mock_files

        with patch("data_connectors.connectors.gdrive._build_service", return_value=mock_service):
            connector = GoogleDriveConnector(credentials={"oauth_token": "t"})
            entries = await connector.poll()
            assert len(entries) == 1
            # No md5, so hash should be based on id:modifiedTime
            assert entries[0]["content_hash"]
            assert entries[0]["size_bytes"] is None

    @pytest.mark.asyncio
    async def test_get_content_url_binary_file(self, catalog):
        from data_connectors.connectors.gdrive import GoogleDriveConnector

        connector = GoogleDriveConnector(
            credentials={"oauth_token": "test-token"},
            catalog=catalog,
        )
        # Register a VFS entry for a binary file
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="gdrive",
            source_path="doc.pdf",
            content_hash="h1",
            metadata={"gdrive_file_id": "file1", "gdrive_mime_type": "application/pdf"},
        )
        url = await connector.get_content_url(entry.vfs_ref)
        assert url == "https://www.googleapis.com/drive/v3/files/file1?alt=media"

    @pytest.mark.asyncio
    async def test_get_content_url_google_doc(self, catalog):
        from data_connectors.connectors.gdrive import GoogleDriveConnector

        connector = GoogleDriveConnector(
            credentials={"oauth_token": "test-token"},
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="gdrive",
            source_path="My Doc",
            content_hash="h2",
            metadata={
                "gdrive_file_id": "gdoc1",
                "gdrive_mime_type": "application/vnd.google-apps.document",
            },
        )
        url = await connector.get_content_url(entry.vfs_ref)
        assert "export" in url
        assert "mimeType=text/plain" in url

    @pytest.mark.asyncio
    async def test_get_content_url_nonexistent(self, catalog):
        from data_connectors.connectors.gdrive import GoogleDriveConnector

        connector = GoogleDriveConnector(credentials={"oauth_token": "t"}, catalog=catalog)
        url = await connector.get_content_url("vfs_nonexistent")
        assert url is None

    @pytest.mark.asyncio
    async def test_get_metadata_includes_auth(self, catalog):
        from data_connectors.connectors.gdrive import GoogleDriveConnector

        connector = GoogleDriveConnector(
            credentials={"oauth_token": "my-token"},
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="gdrive",
            source_path="doc.pdf",
            content_hash="h1",
            metadata={"gdrive_file_id": "file1"},
        )
        meta = await connector.get_metadata(entry.vfs_ref)
        assert meta["connector_type"] == "gdrive"
        assert meta["authorization_header"] == "Bearer my-token"

    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent(self):
        from data_connectors.connectors.gdrive import GoogleDriveConnector

        connector = GoogleDriveConnector(credentials={"oauth_token": "t"})
        meta = await connector.get_metadata("vfs_nope")
        assert meta == {"connector_type": "gdrive"}


# ---------------------------------------------------------------------------
# Dropbox connector tests
# ---------------------------------------------------------------------------


class TestDropboxConnector:
    """Tests for the Dropbox connector."""

    @pytest.mark.asyncio
    async def test_load_validates_credentials(self):
        from data_connectors.connectors.dropbox import DropboxConnector

        mock_dbx = MagicMock()
        mock_dbx.users_get_current_account.return_value = MagicMock()

        with patch("data_connectors.connectors.dropbox._build_client", return_value=mock_dbx):
            connector = DropboxConnector(credentials={"access_token": "test"})
            await connector.load()
            mock_dbx.users_get_current_account.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_returns_entries(self):
        from data_connectors.connectors.dropbox import DropboxConnector

        mock_dbx = MagicMock()
        mock_file = MagicMock()
        mock_file.name = "report.pdf"
        mock_file.id = "id:abc"
        mock_file.path_display = "/Documents/report.pdf"
        mock_file.path_lower = "/documents/report.pdf"
        mock_file.content_hash = "deadbeef"
        mock_file.size = 2048
        mock_file.server_modified = MagicMock()
        mock_file.server_modified.timestamp.return_value = 1700000000.0

        # Make isinstance check work for FileMetadata
        import types
        mock_module = types.ModuleType("dropbox")
        mock_files = types.ModuleType("dropbox.files")
        mock_module.files = mock_files
        mock_files.FileMetadata = type(mock_file)

        mock_result = MagicMock()
        mock_result.entries = [mock_file]
        mock_result.has_more = False
        mock_dbx.files_list_folder.return_value = mock_result

        with (
            patch("data_connectors.connectors.dropbox._build_client", return_value=mock_dbx),
            patch.dict("sys.modules", {"dropbox": mock_module, "dropbox.files": mock_files}),
        ):
            connector = DropboxConnector(credentials={"access_token": "test"})
            entries = await connector.poll()
            assert len(entries) == 1
            assert entries[0]["source_path"] == "/Documents/report.pdf"
            assert entries[0]["size_bytes"] == 2048
            assert entries[0]["metadata"]["dropbox_id"] == "id:abc"

    @pytest.mark.asyncio
    async def test_get_content_url_returns_temp_link(self, catalog):
        from data_connectors.connectors.dropbox import DropboxConnector

        mock_dbx = MagicMock()
        mock_result = MagicMock()
        mock_result.link = "https://dl.dropboxusercontent.com/temp/abc123"
        mock_dbx.files_get_temporary_link.return_value = mock_result

        with patch("data_connectors.connectors.dropbox._build_client", return_value=mock_dbx):
            connector = DropboxConnector(
                credentials={"access_token": "test"},
                catalog=catalog,
            )
            entry = await catalog.register(
                workspace_id="ws-1",
                connector_id="dropbox",
                source_path="report.pdf",
                content_hash="h1",
                metadata={"dropbox_path": "/documents/report.pdf"},
            )
            url = await connector.get_content_url(entry.vfs_ref)
            assert url == "https://dl.dropboxusercontent.com/temp/abc123"

    @pytest.mark.asyncio
    async def test_get_content_url_nonexistent(self, catalog):
        from data_connectors.connectors.dropbox import DropboxConnector

        connector = DropboxConnector(credentials={"access_token": "test"}, catalog=catalog)
        url = await connector.get_content_url("vfs_nope")
        assert url is None

    @pytest.mark.asyncio
    async def test_get_metadata(self, catalog):
        from data_connectors.connectors.dropbox import DropboxConnector

        connector = DropboxConnector(
            credentials={"access_token": "test"},
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="dropbox",
            source_path="report.pdf",
            content_hash="h1",
            content_type="application/pdf",
            size_bytes=2048,
            metadata={"dropbox_id": "id:abc"},
        )
        meta = await connector.get_metadata(entry.vfs_ref)
        assert meta["connector_type"] == "dropbox"
        assert meta["source_path"] == "report.pdf"

    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent(self):
        from data_connectors.connectors.dropbox import DropboxConnector

        connector = DropboxConnector(credentials={"access_token": "test"})
        meta = await connector.get_metadata("vfs_nope")
        assert meta == {"connector_type": "dropbox"}


# ---------------------------------------------------------------------------
# Slack connector tests
# ---------------------------------------------------------------------------


class TestSlackConnector:
    """Tests for the Slack connector (materialize+presign)."""

    @pytest.mark.asyncio
    async def test_load_validates_credentials(self):
        from data_connectors.connectors.slack import SlackConnector

        mock_client = AsyncMock()
        mock_client.auth_test.return_value = {"ok": True, "team": "TestTeam"}

        with patch("data_connectors.connectors.slack._build_client", return_value=mock_client):
            connector = SlackConnector(credentials={"bot_token": "xoxb-test"})
            await connector.load()
            mock_client.auth_test.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_raises_on_auth_failure(self):
        from data_connectors.connectors.slack import SlackConnector

        mock_client = AsyncMock()
        mock_client.auth_test.return_value = {"ok": False, "error": "invalid_auth"}

        with patch("data_connectors.connectors.slack._build_client", return_value=mock_client):
            connector = SlackConnector(credentials={"bot_token": "xoxb-bad"})
            with pytest.raises(ValueError, match="auth.test failed"):
                await connector.load()

    @pytest.mark.asyncio
    async def test_poll_returns_entries(self):
        from data_connectors.connectors.slack import SlackConnector

        mock_client = AsyncMock()
        # Channel is already an ID
        mock_client.conversations_history.return_value = {
            "messages": [
                {"text": "Hello world", "ts": "1700000001.000", "user": "U123"},
                {"text": "", "ts": "1700000002.000", "user": "U456"},  # empty, should be skipped
            ],
            "response_metadata": {},
        }

        with patch("data_connectors.connectors.slack._build_client", return_value=mock_client):
            connector = SlackConnector(
                credentials={"bot_token": "xoxb-test"},
                settings={"channels": ["C12345"]},
            )
            entries = await connector.poll()
            assert len(entries) == 1
            assert entries[0]["source_path"] == "slack/C12345/1700000001.000"
            assert entries[0]["content_type"] == "text/plain"
            assert entries[0]["metadata"]["slack_channel_id"] == "C12345"
            assert entries[0]["metadata"]["slack_content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_get_content_url_materializes_to_blob(self, catalog, mock_blob_store):
        from data_connectors.connectors.slack import SlackConnector

        connector = SlackConnector(
            credentials={"bot_token": "xoxb-test"},
            blob_store=mock_blob_store,
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="slack",
            source_path="slack/general/123",
            content_hash="h1",
            metadata={"slack_content": "Hello world"},
        )
        url = await connector.get_content_url(entry.vfs_ref)
        assert url == "https://s3.example.com/blob"
        mock_blob_store.put_object.assert_awaited_once()
        # Verify blob_key was set on the entry
        updated = await catalog.get(entry.vfs_ref)
        assert updated.blob_key is not None

    @pytest.mark.asyncio
    async def test_get_content_url_reuses_existing_blob(self, catalog, mock_blob_store):
        from data_connectors.connectors.slack import SlackConnector

        connector = SlackConnector(
            credentials={"bot_token": "xoxb-test"},
            blob_store=mock_blob_store,
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="slack",
            source_path="slack/general/123",
            content_hash="h1",
            blob_key="ws-1/slack/existing.txt",
            metadata={"slack_content": "Hello"},
        )
        url = await connector.get_content_url(entry.vfs_ref)
        assert url == "https://s3.example.com/blob"
        # Should NOT have called put_object since blob_key already exists
        mock_blob_store.put_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_content_url_nonexistent(self, catalog, mock_blob_store):
        from data_connectors.connectors.slack import SlackConnector

        connector = SlackConnector(
            credentials={"bot_token": "xoxb-test"},
            blob_store=mock_blob_store,
            catalog=catalog,
        )
        url = await connector.get_content_url("vfs_nope")
        assert url is None

    @pytest.mark.asyncio
    async def test_get_metadata_excludes_content(self, catalog):
        from data_connectors.connectors.slack import SlackConnector

        connector = SlackConnector(
            credentials={"bot_token": "xoxb-test"},
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="slack",
            source_path="slack/general/123",
            content_hash="h1",
            metadata={
                "slack_channel": "general",
                "slack_user": "U123",
                "slack_content": "Hello world",
            },
        )
        meta = await connector.get_metadata(entry.vfs_ref)
        assert meta["connector_type"] == "slack"
        assert meta["slack_channel"] == "general"
        assert "slack_content" not in meta  # Raw content excluded

    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent(self):
        from data_connectors.connectors.slack import SlackConnector

        connector = SlackConnector(credentials={"bot_token": "xoxb-test"})
        meta = await connector.get_metadata("vfs_nope")
        assert meta == {"connector_type": "slack"}


# ---------------------------------------------------------------------------
# Teams connector tests
# ---------------------------------------------------------------------------


class TestTeamsConnector:
    """Tests for the Microsoft Teams connector (materialize+presign)."""

    @pytest.mark.asyncio
    async def test_load_validates_credentials(self):
        from data_connectors.connectors.teams import TeamsConnector

        with (
            patch("data_connectors.connectors.teams._get_access_token", return_value="mock-token"),
            patch("data_connectors.connectors.teams._graph_get", return_value={"value": []}),
        ):
            connector = TeamsConnector(
                credentials={"client_id": "c", "client_secret": "s", "tenant_id": "t"},
                settings={"team_id": "team1"},
            )
            await connector.load()

    @pytest.mark.asyncio
    async def test_load_raises_without_team_id(self):
        from data_connectors.connectors.teams import TeamsConnector

        with patch("data_connectors.connectors.teams._get_access_token", return_value="mock-token"):
            connector = TeamsConnector(
                credentials={"client_id": "c", "client_secret": "s", "tenant_id": "t"},
                settings={},
            )
            with pytest.raises(ValueError, match="team_id"):
                await connector.load()

    @pytest.mark.asyncio
    async def test_poll_returns_entries(self):
        from data_connectors.connectors.teams import TeamsConnector

        messages_response = {
            "value": [
                {
                    "id": "msg1",
                    "createdDateTime": "2025-01-15T10:00:00Z",
                    "body": {"content": "Hello Teams", "contentType": "text"},
                    "from": {"user": {"displayName": "Alice"}},
                },
                {
                    "id": "msg2",
                    "createdDateTime": "2025-01-15T11:00:00Z",
                    "body": {"content": "", "contentType": "text"},  # empty, skipped
                    "from": {"user": {"displayName": "Bob"}},
                },
            ],
        }

        async def mock_graph_get(http, path, params=None):
            if "channels" in path and "messages" not in path:
                return {"value": [{"id": "chan1"}]}
            return messages_response

        with (
            patch("data_connectors.connectors.teams._get_access_token", return_value="mock-token"),
            patch("data_connectors.connectors.teams._graph_get", side_effect=mock_graph_get),
        ):
            connector = TeamsConnector(
                credentials={"client_id": "c", "client_secret": "s", "tenant_id": "t"},
                settings={"team_id": "team1"},
            )
            entries = await connector.poll()
            assert len(entries) == 1
            assert entries[0]["source_path"] == "teams/team1/chan1/msg1"
            assert entries[0]["metadata"]["teams_sender"] == "Alice"
            assert entries[0]["metadata"]["teams_content"] == "Hello Teams"

    @pytest.mark.asyncio
    async def test_get_content_url_materializes(self, catalog, mock_blob_store):
        from data_connectors.connectors.teams import TeamsConnector

        connector = TeamsConnector(
            credentials={"client_id": "c", "client_secret": "s", "tenant_id": "t"},
            settings={"team_id": "team1"},
            blob_store=mock_blob_store,
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="teams",
            source_path="teams/team1/chan1/msg1",
            content_hash="h1",
            metadata={"teams_content": "Hello Teams"},
        )
        url = await connector.get_content_url(entry.vfs_ref)
        assert url == "https://s3.example.com/blob"
        mock_blob_store.put_object.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_content_url_nonexistent(self, catalog, mock_blob_store):
        from data_connectors.connectors.teams import TeamsConnector

        connector = TeamsConnector(
            credentials={"client_id": "c", "client_secret": "s", "tenant_id": "t"},
            blob_store=mock_blob_store,
            catalog=catalog,
        )
        url = await connector.get_content_url("vfs_nope")
        assert url is None

    @pytest.mark.asyncio
    async def test_get_metadata_excludes_content(self, catalog):
        from data_connectors.connectors.teams import TeamsConnector

        connector = TeamsConnector(
            credentials={"client_id": "c", "client_secret": "s", "tenant_id": "t"},
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="teams",
            source_path="teams/team1/chan1/msg1",
            content_hash="h1",
            metadata={"teams_sender": "Alice", "teams_content": "Hello"},
        )
        meta = await connector.get_metadata(entry.vfs_ref)
        assert meta["connector_type"] == "teams"
        assert meta["teams_sender"] == "Alice"
        assert "teams_content" not in meta

    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent(self):
        from data_connectors.connectors.teams import TeamsConnector

        connector = TeamsConnector(
            credentials={"client_id": "c", "client_secret": "s", "tenant_id": "t"},
        )
        meta = await connector.get_metadata("vfs_nope")
        assert meta == {"connector_type": "teams"}


# ---------------------------------------------------------------------------
# Discord connector tests
# ---------------------------------------------------------------------------


class TestDiscordConnector:
    """Tests for the Discord connector (materialize+presign)."""

    @pytest.mark.asyncio
    async def test_load_validates_credentials(self):
        from data_connectors.connectors.discord import DiscordConnector

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http):
            connector = DiscordConnector(credentials={"bot_token": "test-token"})
            await connector.load()

    @pytest.mark.asyncio
    async def test_load_raises_without_bot_token(self):
        from data_connectors.connectors.discord import DiscordConnector

        connector = DiscordConnector(credentials={})
        with pytest.raises(ValueError, match="bot_token"):
            await connector.load()

    @pytest.mark.asyncio
    async def test_poll_returns_entries(self):
        from data_connectors.connectors.discord import DiscordConnector

        # Create a mock response for messages
        messages = [
            {"id": "1234567890123456", "content": "Hello Discord", "author": {"username": "alice"}},
            {"id": "1234567890123457", "content": "", "author": {"username": "bob"}},  # empty, skipped
        ]

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=messages)

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http):
            connector = DiscordConnector(
                credentials={"bot_token": "test-token"},
                settings={"channel_ids": ["chan1"], "guild_id": "guild1"},
            )
            entries = await connector.poll()
            assert len(entries) == 1
            assert entries[0]["source_path"] == "discord/guild1/chan1/1234567890123456"
            assert entries[0]["metadata"]["discord_author"] == "alice"
            assert entries[0]["metadata"]["discord_content"] == "Hello Discord"

    @pytest.mark.asyncio
    async def test_get_content_url_materializes(self, catalog, mock_blob_store):
        from data_connectors.connectors.discord import DiscordConnector

        connector = DiscordConnector(
            credentials={"bot_token": "test-token"},
            blob_store=mock_blob_store,
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="discord",
            source_path="discord/guild1/chan1/msg1",
            content_hash="h1",
            metadata={"discord_content": "Hello Discord"},
        )
        url = await connector.get_content_url(entry.vfs_ref)
        assert url == "https://s3.example.com/blob"
        mock_blob_store.put_object.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_content_url_reuses_existing_blob(self, catalog, mock_blob_store):
        from data_connectors.connectors.discord import DiscordConnector

        connector = DiscordConnector(
            credentials={"bot_token": "test-token"},
            blob_store=mock_blob_store,
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="discord",
            source_path="discord/guild1/chan1/msg1",
            content_hash="h1",
            blob_key="ws-1/discord/existing.txt",
            metadata={"discord_content": "Hello"},
        )
        url = await connector.get_content_url(entry.vfs_ref)
        assert url == "https://s3.example.com/blob"
        mock_blob_store.put_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_content_url_nonexistent(self, catalog, mock_blob_store):
        from data_connectors.connectors.discord import DiscordConnector

        connector = DiscordConnector(
            credentials={"bot_token": "test-token"},
            blob_store=mock_blob_store,
            catalog=catalog,
        )
        url = await connector.get_content_url("vfs_nope")
        assert url is None

    @pytest.mark.asyncio
    async def test_get_metadata_excludes_content(self, catalog):
        from data_connectors.connectors.discord import DiscordConnector

        connector = DiscordConnector(
            credentials={"bot_token": "test-token"},
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="discord",
            source_path="discord/guild1/chan1/msg1",
            content_hash="h1",
            metadata={"discord_author": "alice", "discord_content": "Hello"},
        )
        meta = await connector.get_metadata(entry.vfs_ref)
        assert meta["connector_type"] == "discord"
        assert meta["discord_author"] == "alice"
        assert "discord_content" not in meta

    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent(self):
        from data_connectors.connectors.discord import DiscordConnector

        connector = DiscordConnector(credentials={"bot_token": "test-token"})
        meta = await connector.get_metadata("vfs_nope")
        assert meta == {"connector_type": "discord"}


# ---------------------------------------------------------------------------
# GitHub connector tests
# ---------------------------------------------------------------------------


class TestGitHubConnector:
    """Tests for the GitHub connector."""

    @pytest.mark.asyncio
    async def test_load_validates_credentials(self):
        from data_connectors.connectors.github import GitHubConnector

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http):
            connector = GitHubConnector(
                credentials={"token": "ghp_test"},
                settings={"repos": ["owner/repo"]},
            )
            await connector.load()

    @pytest.mark.asyncio
    async def test_poll_returns_readme_and_issues(self):
        from data_connectors.connectors.github import GitHubConnector
        import base64

        readme_content = base64.b64encode(b"# My Repo\nHello").decode()

        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "/readme" in url:
                resp.json.return_value = {
                    "content": readme_content,
                    "html_url": "https://github.com/owner/repo/blob/main/README.md",
                    "download_url": "https://raw.githubusercontent.com/owner/repo/main/README.md",
                }
            elif "/issues" in url:
                resp.json.return_value = [
                    {
                        "number": 1,
                        "title": "Bug fix",
                        "body": "Fix the bug",
                        "state": "open",
                        "updated_at": "2025-01-15T10:00:00Z",
                        "html_url": "https://github.com/owner/repo/issues/1",
                        "labels": [{"name": "bug"}],
                        "user": {"login": "alice"},
                        "assignees": [{"login": "bob"}],
                    },
                    {
                        "number": 2,
                        "title": "Add feature",
                        "body": "New feature",
                        "state": "closed",
                        "updated_at": "2025-01-14T10:00:00Z",
                        "html_url": "https://github.com/owner/repo/pull/2",
                        "labels": [],
                        "pull_request": {"url": "..."},  # This is a PR
                    },
                ]
            else:
                resp.json.return_value = {}
            return resp

        mock_http = MagicMock()
        mock_http.get = AsyncMock(side_effect=mock_get)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http):
            connector = GitHubConnector(
                credentials={"token": "ghp_test"},
                settings={"repos": ["owner/repo"]},
            )
            entries = await connector.poll()
            # Should have 1 README + 1 issue + 1 PR = 3
            assert len(entries) == 3

            readme = next(e for e in entries if "README" in e["source_path"])
            assert readme["content_type"] == "text/markdown"
            assert readme["metadata"]["github_download_url"] is not None

            issue = next(e for e in entries if "issue" in e["source_path"])
            assert issue["metadata"]["github_type"] == "issue"
            assert issue["metadata"]["github_number"] == 1
            assert issue["metadata"]["github_author"] == {"login": "alice"}
            assert issue["metadata"]["github_assignees"] == [{"login": "bob"}]

            pr = next(e for e in entries if "pr" in e["source_path"])
            assert pr["metadata"]["github_type"] == "pr"

    @pytest.mark.asyncio
    async def test_poll_skips_prs_when_disabled(self):
        from data_connectors.connectors.github import GitHubConnector
        import base64

        readme_content = base64.b64encode(b"# Repo").decode()

        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "/readme" in url:
                resp.json.return_value = {"content": readme_content, "html_url": "", "download_url": ""}
            elif "/issues" in url:
                resp.json.return_value = [
                    {"number": 1, "title": "PR", "body": "x", "state": "open",
                     "updated_at": "2025-01-15T10:00:00Z", "html_url": "", "labels": [],
                     "pull_request": {}},
                ]
            else:
                resp.json.return_value = {}
            return resp

        mock_http = MagicMock()
        mock_http.get = AsyncMock(side_effect=mock_get)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http):
            connector = GitHubConnector(
                credentials={"token": "ghp_test"},
                settings={"repos": ["owner/repo"], "include_prs": False},
            )
            entries = await connector.poll()
            # Should have README only (PR excluded)
            assert len(entries) == 1
            assert "README" in entries[0]["source_path"]

    @pytest.mark.asyncio
    async def test_get_content_url_readme(self, catalog):
        from data_connectors.connectors.github import GitHubConnector

        connector = GitHubConnector(
            credentials={"token": "ghp_test"},
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="github",
            source_path="github/owner/repo/README",
            content_hash="h1",
            metadata={
                "github_download_url": "https://raw.githubusercontent.com/owner/repo/main/README.md",
            },
        )
        url = await connector.get_content_url(entry.vfs_ref)
        assert url == "https://raw.githubusercontent.com/owner/repo/main/README.md"

    @pytest.mark.asyncio
    async def test_get_content_url_issue(self, catalog):
        from data_connectors.connectors.github import GitHubConnector

        connector = GitHubConnector(
            credentials={"token": "ghp_test"},
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="github",
            source_path="github/owner/repo/issue/1",
            content_hash="h1",
            metadata={
                "github_html_url": "https://github.com/owner/repo/issues/1",
            },
        )
        url = await connector.get_content_url(entry.vfs_ref)
        assert url == "https://github.com/owner/repo/issues/1"

    @pytest.mark.asyncio
    async def test_get_content_url_nonexistent(self, catalog):
        from data_connectors.connectors.github import GitHubConnector

        connector = GitHubConnector(credentials={"token": "ghp_test"}, catalog=catalog)
        url = await connector.get_content_url("vfs_nope")
        assert url is None

    @pytest.mark.asyncio
    async def test_get_metadata_includes_auth(self, catalog):
        from data_connectors.connectors.github import GitHubConnector

        connector = GitHubConnector(
            credentials={"token": "ghp_my_token"},
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="github",
            source_path="github/owner/repo/README",
            content_hash="h1",
            metadata={"github_repo": "owner/repo"},
        )
        meta = await connector.get_metadata(entry.vfs_ref)
        assert meta["connector_type"] == "github"
        assert meta["authorization_header"] == "Bearer ghp_my_token"

    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent(self):
        from data_connectors.connectors.github import GitHubConnector

        connector = GitHubConnector(credentials={"token": "ghp_test"})
        meta = await connector.get_metadata("vfs_nope")
        assert meta == {"connector_type": "github"}


# ---------------------------------------------------------------------------
# Snowflake utility tests (Discord)
# ---------------------------------------------------------------------------


class TestDiscordSnowflakeUtils:
    """Test the Discord snowflake conversion utilities."""

    def test_epoch_to_snowflake_and_back(self):
        from data_connectors.connectors.discord import _epoch_to_snowflake, _snowflake_to_epoch

        # Known epoch: 2024-01-01T00:00:00Z = 1704067200
        epoch = 1704067200.0
        snowflake = _epoch_to_snowflake(epoch)
        recovered = _snowflake_to_epoch(snowflake)
        # Snowflake conversion loses sub-second precision
        assert abs(recovered - epoch) < 1.0

    def test_epoch_to_snowflake_before_discord_epoch(self):
        from data_connectors.connectors.discord import _epoch_to_snowflake

        # Before Discord epoch (2015-01-01)
        snowflake = _epoch_to_snowflake(1000000000.0)
        assert snowflake == "0"


# ---------------------------------------------------------------------------
# Local filesystem connector tests
# ---------------------------------------------------------------------------


class TestLocalFsConnector:
    """Tests for the local filesystem connector (materialize+presign)."""

    @pytest.mark.asyncio
    async def test_load_validates_existing_directory(self, tmp_path):
        from data_connectors.connectors.local_fs import LocalFsConnector

        connector = LocalFsConnector(base_directory=str(tmp_path))
        await connector.load()  # Should not raise

    @pytest.mark.asyncio
    async def test_load_raises_for_missing_directory(self):
        from data_connectors.connectors.local_fs import LocalFsConnector

        connector = LocalFsConnector(base_directory="/nonexistent/path/xyz_test")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            await connector.load()

    @pytest.mark.asyncio
    async def test_load_raises_for_file_path(self, tmp_path):
        from data_connectors.connectors.local_fs import LocalFsConnector

        file_path = tmp_path / "somefile.txt"
        file_path.write_text("hello")
        connector = LocalFsConnector(base_directory=str(file_path))
        with pytest.raises(NotADirectoryError, match="not a directory"):
            await connector.load()

    @pytest.mark.asyncio
    async def test_poll_discovers_files(self, tmp_path):
        from data_connectors.connectors.local_fs import LocalFsConnector

        # Create test files
        (tmp_path / "hello.md").write_text("# Hello World")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "data.txt").write_text("some data")

        connector = LocalFsConnector(base_directory=str(tmp_path))
        entries = await connector.poll()

        assert len(entries) == 2
        paths = {e["source_path"] for e in entries}
        assert "hello.md" in paths
        assert "subdir/data.txt" in paths

        for entry in entries:
            assert entry["content_hash"]  # SHA256 hex digest
            assert entry["size_bytes"] > 0
            assert entry["content_type"]  # MIME type guessed
            assert entry["blob_key"] is None  # No blob store provided

    @pytest.mark.asyncio
    async def test_poll_uploads_to_blob_store(self, tmp_path, mock_blob_store):
        from data_connectors.connectors.local_fs import LocalFsConnector

        (tmp_path / "test.txt").write_text("blob content")

        connector = LocalFsConnector(
            base_directory=str(tmp_path),
            blob_store=mock_blob_store,
        )
        entries = await connector.poll()

        assert len(entries) == 1
        assert entries[0]["blob_key"] is not None
        assert entries[0]["blob_key"].startswith("local_fs/")
        mock_blob_store.put_object.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_poll_empty_directory(self, tmp_path):
        from data_connectors.connectors.local_fs import LocalFsConnector

        connector = LocalFsConnector(base_directory=str(tmp_path))
        entries = await connector.poll()
        assert entries == []

    @pytest.mark.asyncio
    async def test_poll_content_hash_is_sha256(self, tmp_path):
        import hashlib
        from data_connectors.connectors.local_fs import LocalFsConnector

        content = "test content for hashing"
        (tmp_path / "hash_test.txt").write_text(content)

        connector = LocalFsConnector(base_directory=str(tmp_path))
        entries = await connector.poll()

        expected_hash = hashlib.sha256(content.encode()).hexdigest()
        assert entries[0]["content_hash"] == expected_hash

    @pytest.mark.asyncio
    async def test_poll_content_type_detection(self, tmp_path):
        from data_connectors.connectors.local_fs import LocalFsConnector

        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "page.html").write_text("<html></html>")
        (tmp_path / "unknown.xyz123").write_bytes(b"\x00\x01")

        connector = LocalFsConnector(base_directory=str(tmp_path))
        entries = await connector.poll()

        type_map = {e["source_path"]: e["content_type"] for e in entries}
        assert type_map["doc.pdf"] == "application/pdf"
        assert "html" in type_map["page.html"]
        assert type_map["unknown.xyz123"] == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_get_content_url_existing(self, catalog, mock_blob_store):
        from data_connectors.connectors.local_fs import LocalFsConnector

        connector = LocalFsConnector(
            base_directory="/tmp/test",
            catalog=catalog,
            blob_store=mock_blob_store,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="local_fs",
            source_path="hello.md",
            content_hash="h1",
            blob_key="local_fs/abc/hello.md",
        )
        url = await connector.get_content_url(entry.vfs_ref)
        assert url == "https://s3.example.com/blob"

    @pytest.mark.asyncio
    async def test_get_content_url_no_blob_key(self, catalog, mock_blob_store):
        from data_connectors.connectors.local_fs import LocalFsConnector

        connector = LocalFsConnector(
            base_directory="/tmp/test",
            catalog=catalog,
            blob_store=mock_blob_store,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="local_fs",
            source_path="hello.md",
            content_hash="h1",
        )
        url = await connector.get_content_url(entry.vfs_ref)
        assert url is None

    @pytest.mark.asyncio
    async def test_get_content_url_nonexistent(self, catalog, mock_blob_store):
        from data_connectors.connectors.local_fs import LocalFsConnector

        connector = LocalFsConnector(
            base_directory="/tmp/test",
            catalog=catalog,
            blob_store=mock_blob_store,
        )
        url = await connector.get_content_url("vfs_nope")
        assert url is None

    @pytest.mark.asyncio
    async def test_get_metadata(self, catalog):
        from data_connectors.connectors.local_fs import LocalFsConnector

        connector = LocalFsConnector(
            base_directory="/tmp/test",
            catalog=catalog,
        )
        entry = await catalog.register(
            workspace_id="ws-1",
            connector_id="local_fs",
            source_path="hello.md",
            content_hash="h1",
            content_type="text/markdown",
            size_bytes=128,
            metadata={"local_fs_base": "/tmp/test"},
        )
        meta = await connector.get_metadata(entry.vfs_ref)
        assert meta["connector_type"] == "local_fs"
        assert meta["source_path"] == "hello.md"
        assert meta["content_type"] == "text/markdown"
        assert meta["size_bytes"] == 128
        assert meta["local_fs_base"] == "/tmp/test"

    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent(self):
        from data_connectors.connectors.local_fs import LocalFsConnector

        connector = LocalFsConnector(base_directory="/tmp/test")
        meta = await connector.get_metadata("vfs_nope")
        assert meta == {"connector_type": "local_fs"}
