# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for FastAPI routes.

Tests each route for happy-path behavior using FastAPI's TestClient.
The app uses in-memory stores, so no external dependencies are needed.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked Aether connection + mocked blob store.

    The blob_store mock is essential for sync tests that exercise connectors
    which materialize content to the blob (e.g., local_fs, slack). Without it,
    aioboto3 attempts real S3 calls and fails with NoCredentialsError.
    """
    # Build the blob-store mock first so the lifespan startup hook
    # (which calls BlobStore(...)) returns it instead of a real S3-backed instance.
    mock_blob = AsyncMock()
    mock_blob.put_object = AsyncMock(return_value=42)
    mock_blob.generate_download_url = AsyncMock(
        return_value=("https://blob.test/x", {}, None)
    )
    mock_blob.generate_upload_url = AsyncMock(
        return_value=("https://blob.test/upload", "ws/key", None)
    )

    with patch("data_connectors.server.app.AetherServiceRegistration") as MockAether, \
         patch("data_connectors.server.app.BlobStore", return_value=mock_blob):
        mock_instance = AsyncMock()
        mock_instance.client = None
        mock_instance.connect = AsyncMock()
        mock_instance.disconnect = AsyncMock()
        MockAether.return_value = mock_instance

        from data_connectors.server.app import app
        with TestClient(app) as c:
            yield c


class TestHealthz:
    def test_healthz(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestProviderRoutes:
    def test_create_provider(self, client):
        resp = client.post("/v1/providers", json={
            "name": "Test S3",
            "provider_type": "s3",
            "description": "Test bucket",
            "metadata": {"workspace_id": "ws-1"},
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test S3"
        assert data["provider_type"] == "s3"
        assert data["id"].startswith("dp_")

    def test_list_providers(self, client):
        # Create a provider first
        client.post("/v1/providers", json={
            "name": "P1", "provider_type": "s3",
            "metadata": {"workspace_id": "ws-test"},
        })
        resp = client.get("/v1/providers?workspace_id=ws-test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] >= 1

    def test_update_provider(self, client):
        create_resp = client.post("/v1/providers", json={
            "name": "Original", "provider_type": "s3",
            "metadata": {"workspace_id": "_default"},
        })
        pid = create_resp.json()["id"]
        resp = client.patch(f"/v1/providers/{pid}", json={"name": "Updated"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"

    def test_update_nonexistent_provider(self, client):
        resp = client.patch("/v1/providers/dp_doesnotexist", json={"name": "X"})
        assert resp.status_code == 404

    def test_delete_provider(self, client):
        create_resp = client.post("/v1/providers", json={
            "name": "ToDelete", "provider_type": "s3",
            "metadata": {"workspace_id": "_default"},
        })
        pid = create_resp.json()["id"]
        resp = client.delete(f"/v1/providers/{pid}")
        assert resp.status_code == 204

    def test_delete_nonexistent_provider(self, client):
        resp = client.delete("/v1/providers/dp_doesnotexist")
        assert resp.status_code == 404


class TestVfsRoutes:
    def test_aether_mode_rejects_direct_http_without_receipt(self, client, monkeypatch):
        monkeypatch.setenv("DC_VFS_AUTHORIZATION_MODE", "aether")
        resp = client.get("/v1/vfs/entries?workspace_id=ws-1")
        assert resp.status_code == 403
        assert resp.json() == {"detail": "VFS access denied"}

    def test_register_vfs_entry(self, client):
        resp = client.post("/v1/vfs/entries", json={
            "workspace_id": "ws-1",
            "connector_id": "manual_upload",
            "source_path": "test.pdf",
            "content_hash": "sha256_abc",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["vfs_ref"].startswith("vfs_")
        assert data["workspace_id"] == "ws-1"
        assert data["content_hash"] == "sha256_abc"

    def test_get_vfs_entry(self, client):
        create_resp = client.post("/v1/vfs/entries", json={
            "workspace_id": "ws-1",
            "connector_id": "c1",
            "source_path": "file.txt",
            "content_hash": "h1",
        })
        vfs_ref = create_resp.json()["vfs_ref"]
        resp = client.get(f"/v1/vfs/entries/{vfs_ref}")
        assert resp.status_code == 200
        assert resp.json()["vfs_ref"] == vfs_ref

    def test_get_nonexistent_vfs_entry(self, client):
        resp = client.get("/v1/vfs/entries/vfs_nope")
        assert resp.status_code == 404

    def test_list_vfs_entries(self, client):
        client.post("/v1/vfs/entries", json={
            "workspace_id": "ws-list",
            "connector_id": "c1",
            "source_path": "a.txt",
            "content_hash": "h1",
        })
        resp = client.get("/v1/vfs/entries?workspace_id=ws-list")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] >= 1

    def test_update_vfs_entry(self, client):
        create_resp = client.post("/v1/vfs/entries", json={
            "workspace_id": "ws-1",
            "connector_id": "c1",
            "source_path": "a.txt",
            "content_hash": "old_hash",
        })
        vfs_ref = create_resp.json()["vfs_ref"]
        resp = client.patch(f"/v1/vfs/entries/{vfs_ref}", json={"content_hash": "new_hash"})
        assert resp.status_code == 200
        assert resp.json()["content_hash"] == "new_hash"

    def test_finalize_vfs_entry_backfills_and_emits_batch(self, client):
        # Register a placeholder entry (empty hash, like mint_upload_url does).
        create_resp = client.post("/v1/vfs/entries", json={
            "workspace_id": "ws-fin",
            "connector_id": "manual_upload",
            "source_path": "doc.pdf",
            "content_hash": "",
        })
        vfs_ref = create_resp.json()["vfs_ref"]

        with patch("data_connectors.server.app._sync_engine.emit_doc_added",
                   new=AsyncMock(return_value="dctask_x")) as emit:
            resp = client.post(f"/v1/vfs/entries/{vfs_ref}/finalize", json={
                "content_hash": "client_hash",
                "size_bytes": 123,
                "initiated_by": "us::alice",
                "visibility": "private",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["content_hash"] == "client_hash"
        assert data["size_bytes"] == 123

        emit.assert_awaited_once()
        kwargs = emit.await_args.kwargs
        # BATCH task class (3) for user-initiated ingest + initiator/visibility.
        from data_connectors.services.sync_engine import TASK_CLASS_BATCH
        assert kwargs["task_class"] == TASK_CLASS_BATCH
        assert kwargs["initiated_by"] == "us::alice"
        assert kwargs["visibility"] == "private"
        assert kwargs["vfs_ref"] == vfs_ref

    def test_finalize_is_idempotent_emits_once(self, client):
        # Register a placeholder entry (empty hash, like mint_upload_url does).
        create_resp = client.post("/v1/vfs/entries", json={
            "workspace_id": "ws-idem",
            "connector_id": "manual_upload",
            "source_path": "dup.pdf",
            "content_hash": "",
        })
        vfs_ref = create_resp.json()["vfs_ref"]

        with patch("data_connectors.server.app._sync_engine.emit_doc_added",
                   new=AsyncMock(return_value="dctask_x")) as emit:
            body = {
                "content_hash": "client_hash",
                "size_bytes": 123,
                "initiated_by": "us::alice",
                "visibility": "private",
            }
            # A repeated FILE_UPLOAD_COMPLETE (browser retry / ws replay) finalizes twice.
            first = client.post(f"/v1/vfs/entries/{vfs_ref}/finalize", json=body)
            second = client.post(f"/v1/vfs/entries/{vfs_ref}/finalize", json=body)

        assert first.status_code == 200
        # Second call still succeeds (idempotent), no duplicate task.
        assert second.status_code == 200
        assert second.json()["content_hash"] == "client_hash"
        # doc_added task emitted exactly once across both finalize calls.
        assert emit.await_count == 1

    def test_finalize_blobgw_commits_even_when_metadata_supplied(self, client):
        # For the blobgw backend, finalize_blob COMMITS the staged upload
        # (S3-staging -> blobgw). It must run even when the caller supplies both
        # content_hash and size_bytes — otherwise the blob is orphaned in staging
        # and every later GET /blob/{ref} 404s. (This is the regression that the
        # broken download path used to mask.)
        create_resp = client.post("/v1/vfs/entries", json={
            "workspace_id": "ws-bg",
            "connector_id": "manual_upload",
            "source_path": "commit.pdf",
            "content_hash": "",
        })
        vfs_ref = create_resp.json()["vfs_ref"]

        # mint_upload_url would have stamped a blob_key; set one so the blobgw
        # finalize (commit) branch is reachable.
        from data_connectors.server import app as dc_app
        dc_app._catalog._entries[vfs_ref].blob_key = "ws-bg/manual_upload/abc/commit.pdf"

        fake_store = MagicMock()
        fake_store.finalize_blob = AsyncMock(
            return_value={"content_hash": "edge_hash", "size": 999})

        with patch("data_connectors.server.app._blob_store", fake_store), \
             patch("data_connectors.vfs.blob_store_factory.is_blobgw_backend",
                   return_value=True), \
             patch("data_connectors.server.app._sync_engine.emit_doc_added",
                   new=AsyncMock(return_value="dctask_x")):
            resp = client.post(f"/v1/vfs/entries/{vfs_ref}/finalize", json={
                "content_hash": "client_hash",
                "size_bytes": 123,
            })

        assert resp.status_code == 200
        # THE FIX: the commit ran despite both fields being supplied.
        fake_store.finalize_blob.assert_awaited_once()
        # Caller-supplied metadata is preserved (finalize only fills what's missing).
        data = resp.json()
        assert data["content_hash"] == "client_hash"
        assert data["size_bytes"] == 123

    def test_finalize_nonexistent_vfs_entry(self, client):
        resp = client.post("/v1/vfs/entries/vfs_nope/finalize", json={"content_hash": "h"})
        assert resp.status_code == 404

    def test_link_ml_document(self, client):
        create_resp = client.post("/v1/vfs/entries", json={
            "workspace_id": "ws-1",
            "connector_id": "c1",
            "source_path": "doc.pdf",
            "content_hash": "dochash",
        })
        vfs_ref = create_resp.json()["vfs_ref"]
        resp = client.post(f"/v1/vfs/entries/{vfs_ref}/link", json={
            "ml_doc_id": "doc_abc",
            "ml_job_id": "job_xyz",
        })
        assert resp.status_code == 200
        assert resp.json()["ml_doc_id"] == "doc_abc"
        assert resp.json()["ml_job_id"] == "job_xyz"

    def test_link_nonexistent_returns_404(self, client):
        resp = client.post("/v1/vfs/entries/vfs_nope/link", json={"ml_doc_id": "doc_1"})
        assert resp.status_code == 404

    def test_delete_vfs_entry(self, client):
        create_resp = client.post("/v1/vfs/entries", json={
            "workspace_id": "ws-1",
            "connector_id": "c1",
            "source_path": "bye.txt",
            "content_hash": "byehash",
        })
        vfs_ref = create_resp.json()["vfs_ref"]
        resp = client.delete(f"/v1/vfs/entries/{vfs_ref}")
        assert resp.status_code == 204

    def test_delete_nonexistent_vfs_entry(self, client):
        resp = client.delete("/v1/vfs/entries/vfs_nope")
        assert resp.status_code == 404


class TestSyncRoutes:
    def test_trigger_sync_requires_provider(self, client):
        resp = client.post("/v1/sync/trigger", json={
            "provider_id": "dp_nonexistent",
            "workspace_id": "ws-1",
        })
        assert resp.status_code == 404

    def test_trigger_sync_happy_path(self, client, tmp_path):
        create_resp = client.post("/v1/providers", json={
            "name": "SyncTest", "provider_type": "local_fs",
            "connection_args": {"base_directory": str(tmp_path)},
            "metadata": {"workspace_id": "_default"},
        })
        pid = create_resp.json()["id"]
        resp = client.post("/v1/sync/trigger", json={
            "provider_id": pid,
            "workspace_id": "ws-1",
        })
        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"].startswith("syncjob_")
        assert data["status"] == "pending"

    def test_get_sync_job(self, client, tmp_path):
        create_resp = client.post("/v1/providers", json={
            "name": "SJ", "provider_type": "local_fs",
            "connection_args": {"base_directory": str(tmp_path)},
            "metadata": {"workspace_id": "_default"},
        })
        pid = create_resp.json()["id"]
        trigger_resp = client.post("/v1/sync/trigger", json={
            "provider_id": pid,
            "workspace_id": "ws-1",
        })
        job_id = trigger_resp.json()["job_id"]
        resp = client.get(f"/v1/sync/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == job_id

    def test_get_nonexistent_sync_job(self, client):
        resp = client.get("/v1/sync/jobs/syncjob_nope")
        assert resp.status_code == 404

    def test_trigger_sync_unknown_connector_type(self, client):
        """Trigger sync with an unknown connector type should return 400."""
        create_resp = client.post("/v1/providers", json={
            "name": "BadType", "provider_type": "nonexistent_xyz",
            "metadata": {"workspace_id": "_default"},
        })
        pid = create_resp.json()["id"]
        resp = client.post("/v1/sync/trigger", json={
            "provider_id": pid,
            "workspace_id": "ws-1",
        })
        assert resp.status_code == 400
        assert "Unknown connector type" in resp.json()["detail"]

    def test_trigger_sync_local_fs_runs_connector(self, client, tmp_path):
        """Trigger sync for a local_fs provider should actually run the connector."""
        # Create a test file in a temp directory
        import os
        test_file = tmp_path / "test_sync.txt"
        test_file.write_text("sync test content")

        create_resp = client.post("/v1/providers", json={
            "name": "LocalFS", "provider_type": "local_fs",
            "connection_args": {"base_directory": str(tmp_path)},
            "metadata": {"workspace_id": "ws-sync"},
        })
        pid = create_resp.json()["id"]

        resp = client.post("/v1/sync/trigger", json={
            "provider_id": pid,
            "workspace_id": "ws-sync",
        })
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # TestClient runs background tasks synchronously, so the job
        # should already be completed by the time we check.
        job_resp = client.get(f"/v1/sync/jobs/{job_id}")
        assert job_resp.status_code == 200
        job = job_resp.json()
        assert job["status"] == "completed"
        assert job["entries_discovered"] == 1
        assert job["entries_synced"] == 1

        # Verify the VFS entry was registered
        vfs_resp = client.get("/v1/vfs/entries?workspace_id=ws-sync")
        assert vfs_resp.status_code == 200
        entries = vfs_resp.json()["entries"]
        assert len(entries) >= 1
        paths = {e["source_path"] for e in entries}
        assert "test_sync.txt" in paths

    def test_trigger_sync_local_fs_missing_dir_fails(self, client):
        """Trigger sync for a local_fs provider with a missing directory should fail the job."""
        create_resp = client.post("/v1/providers", json={
            "name": "MissingDir", "provider_type": "local_fs",
            "connection_args": {"base_directory": "/nonexistent/path/xyz_test_42"},
            "metadata": {"workspace_id": "ws-bad"},
        })
        pid = create_resp.json()["id"]

        resp = client.post("/v1/sync/trigger", json={
            "provider_id": pid,
            "workspace_id": "ws-bad",
        })
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        job_resp = client.get(f"/v1/sync/jobs/{job_id}")
        assert job_resp.status_code == 200
        job = job_resp.json()
        assert job["status"] == "failed"
        assert job["error"] is not None


class TestStorageUsage:
    """`/v1/admin/storage/usage` — cache + force + error mapping.

    Injects a StorageUsageService backed by an httpx.MockTransport so no real
    blobgw is needed; a call counter proves the TTL cache serves the second
    request without a second upstream fetch, and that force re-fetches.
    """

    def _service(self, status_code=200, payload=None, mlfs=None):
        import httpx

        from data_connectors.vfs.storage_usage import StorageUsageService

        calls = {"n": 0}
        body = payload if payload is not None else {
            "domain": "t1", "physical_bytes": 100, "logical_deduped_bytes": 250,
            "compression_ratio": 2.5, "object_apparent_bytes": 300,
            "computed_at": "2026-07-19T00:00:00Z",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(status_code, json=body)

        # Deterministic mlfs side: default None (substrate-only) so tests don't
        # depend on a live meta DB. Pass a dict to exercise the combine.
        async def mlfs_stats_fn(domain):
            return mlfs

        svc = StorageUsageService(
            blobgw_url="http://blobgw.test", ttl_s=300,
            transport=httpx.MockTransport(handler),
            mlfs_stats_fn=mlfs_stats_fn,
        )
        return svc, calls

    def test_usage_happy_and_cached(self, client):
        import data_connectors.server.app as appmod

        svc, calls = self._service()
        appmod._storage_usage = svc

        r1 = client.get("/v1/admin/storage/usage", params={"domain": "t1"})
        assert r1.status_code == 200
        assert r1.json()["physical_bytes"] == 100
        assert r1.json()["cached"] is False

        r2 = client.get("/v1/admin/storage/usage", params={"domain": "t1"})
        assert r2.status_code == 200
        assert r2.json()["cached"] is True
        assert calls["n"] == 1  # served from cache, no second upstream fetch

    def test_usage_force_refetches(self, client):
        import data_connectors.server.app as appmod

        svc, calls = self._service()
        appmod._storage_usage = svc

        client.get("/v1/admin/storage/usage", params={"domain": "t1"})
        r = client.get("/v1/admin/storage/usage", params={"domain": "t1", "force": "true"})
        assert r.status_code == 200
        assert r.json()["cached"] is False
        assert calls["n"] == 2  # force bypassed the cache

    def test_usage_blobgw_error_is_502(self, client):
        import data_connectors.server.app as appmod

        svc, _ = self._service(status_code=500)
        appmod._storage_usage = svc

        r = client.get("/v1/admin/storage/usage", params={"domain": "t1"})
        assert r.status_code == 502

    def test_usage_empty_domain_is_400(self, client):
        import data_connectors.server.app as appmod

        svc, _ = self._service()
        appmod._storage_usage = svc

        r = client.get("/v1/admin/storage/usage", params={"domain": ""})
        assert r.status_code == 400

    def test_usage_substrate_only_when_mlfs_absent(self, client):
        """No mlfs metadata → substrate-only rollup, no apparent/dedup fields."""
        import data_connectors.server.app as appmod

        svc, _ = self._service(mlfs=None)
        appmod._storage_usage = svc

        body = client.get("/v1/admin/storage/usage", params={"domain": "t1"}).json()
        assert body["mlfs"] is False
        assert body["object_apparent_bytes"] == 300
        # Apparent + dedup are withheld without the mlfs reference side.
        assert "apparent_bytes" not in body
        assert "dedup_ratio" not in body
        assert "file_logical_bytes" not in body

    def test_usage_combines_mlfs(self, client):
        """mlfs metadata present → apparent = object_apparent + slice_source,
        dedup_ratio = apparent / deduped, file_logical passed through."""
        import data_connectors.server.app as appmod

        svc, _ = self._service(mlfs={
            "slice_source_bytes": 700,
            "slice_live_bytes": 500,
            "file_logical_bytes": 640,
        })
        appmod._storage_usage = svc

        body = client.get("/v1/admin/storage/usage", params={"domain": "t1"}).json()
        assert body["mlfs"] is True
        assert body["object_apparent_bytes"] == 300
        assert body["mlfs_apparent_bytes"] == 700
        assert body["apparent_bytes"] == 1000  # 300 blobgw objects + 700 mlfs slices
        assert body["file_logical_bytes"] == 640
        assert body["dedup_ratio"] == 1000 / 250  # apparent / deduped
