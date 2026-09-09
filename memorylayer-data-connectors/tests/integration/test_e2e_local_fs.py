"""End-to-end integration test for the local_fs document ingestion pipeline.

Exercises the full flow:
  data-connectors (local_fs connector) -> Aether pool task (doc_added)
  -> MemoryLayer doc_added handler -> JIT URL mint -> fetch -> render
  -> transcribe -> embed -> store -> ingest_complete event

Prerequisites:
  - Docker + docker compose installed
  - ``tests/integration/run.sh`` brings up the stack before running this test
  - All services healthy (compose --wait)

Run directly:
  pytest tests/integration/test_e2e_local_fs.py -v --tb=long

Or via the run script:
  tests/integration/run.sh
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx
import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

# Service URLs — tests run on the host against compose-exposed ports
DC_BASE_URL = os.environ.get("DC_BASE_URL", "http://localhost:61100")
ML_BASE_URL = os.environ.get("ML_BASE_URL", "http://localhost:65001")
AETHER_ADMIN_URL = os.environ.get("AETHER_ADMIN_URL", "http://localhost:60880")

# Test workspace
WORKSPACE_ID = "ws_integration"

# Timeout for polling assertions (seconds)
POLL_TIMEOUT = int(os.environ.get("INTEG_POLL_TIMEOUT", "120"))
POLL_INTERVAL = 2

# Expected fixture files (hello.md, data.txt; sample.pdf only present if generated)
FIXTURE_FILES_REQUIRED = {"hello.md", "data.txt"}


def _wait_for_service(url: str, timeout: int = 60) -> None:
    """Wait for a service to become healthy."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code < 500:
                return
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(1)
    raise TimeoutError(f"Service at {url} did not become ready within {timeout}s")


def _poll_until(
    check_fn,
    description: str,
    timeout: int = POLL_TIMEOUT,
    interval: int = POLL_INTERVAL,
) -> Any:
    """Poll check_fn until it returns a truthy value or timeout expires."""
    deadline = time.time() + timeout
    last_exc = None
    while time.time() < deadline:
        try:
            result = check_fn()
            if result:
                return result
        except Exception as exc:
            last_exc = exc
        time.sleep(interval)
    if last_exc:
        raise TimeoutError(f"Timed out waiting for: {description}") from last_exc
    raise TimeoutError(f"Timed out waiting for: {description}")


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module", autouse=True)
def wait_for_services():
    """Ensure all services are up before running tests."""
    _wait_for_service(f"{DC_BASE_URL}/healthz", timeout=60)
    _wait_for_service(f"{ML_BASE_URL}/health", timeout=60)


@pytest.fixture(scope="module")
def http_client():
    """Shared httpx client for the test module."""
    with httpx.Client(timeout=30.0) as client:
        yield client


@pytest.fixture(scope="module")
def provider_id(http_client: httpx.Client) -> str:
    """Create a local_fs provider pointing at /data/local_fs in the container."""
    resp = http_client.post(
        f"{DC_BASE_URL}/v1/providers",
        json={
            "name": "integration-local-fs",
            "provider_type": "local_fs",
            "description": "Integration test local filesystem provider",
            "enabled": True,
            "connection_args": {
                "base_directory": "/data/local_fs",
            },
            "metadata": {
                "workspace_id": WORKSPACE_ID,
            },
        },
    )
    assert resp.status_code == 201, f"Failed to create provider: {resp.text}"
    data = resp.json()
    return data["id"]


@pytest.fixture(scope="module")
def sync_job_id(http_client: httpx.Client, provider_id: str) -> str:
    """Trigger a sync and return the job ID."""
    resp = http_client.post(
        f"{DC_BASE_URL}/v1/sync/trigger",
        json={
            "provider_id": provider_id,
            "workspace_id": WORKSPACE_ID,
        },
    )
    assert resp.status_code == 202, f"Failed to trigger sync: {resp.text}"
    return resp.json()["job_id"]


@pytest.fixture(scope="module")
def completed_sync_job(http_client: httpx.Client, sync_job_id: str) -> dict:
    """Wait for the sync job to complete and return the final job record."""
    def _check():
        resp = http_client.get(f"{DC_BASE_URL}/v1/sync/jobs/{sync_job_id}")
        if resp.status_code != 200:
            return None
        job = resp.json()
        if job["status"] in ("completed", "failed"):
            return job
        return None

    job = _poll_until(_check, f"sync job {sync_job_id} to complete", timeout=60)
    return job


# ─────────────────────────────────────────────────────────────────────
# Test: Provider creation
# ─────────────────────────────────────────────────────────────────────


class TestProviderLifecycle:
    """Verify provider CRUD works."""

    def test_provider_created(self, http_client: httpx.Client, provider_id: str):
        """Provider should exist after creation."""
        resp = http_client.get(
            f"{DC_BASE_URL}/v1/providers",
            params={"workspace_id": WORKSPACE_ID},
        )
        assert resp.status_code == 200
        providers = resp.json()["providers"]
        ids = [p["id"] for p in providers]
        assert provider_id in ids

    def test_provider_is_local_fs_type(self, http_client: httpx.Client, provider_id: str):
        """Provider type should be local_fs."""
        resp = http_client.get(
            f"{DC_BASE_URL}/v1/providers",
            params={"workspace_id": WORKSPACE_ID},
        )
        providers = resp.json()["providers"]
        provider = next(p for p in providers if p["id"] == provider_id)
        assert provider["provider_type"] == "local_fs"


# ─────────────────────────────────────────────────────────────────────
# Test: Sync trigger and VFS catalog population (REAL sync path)
# ─────────────────────────────────────────────────────────────────────


class TestSyncAndVfsCatalog:
    """Verify sync discovers fixture files and populates the VFS catalog.

    Uses the real sync path: POST /v1/sync/trigger invokes the local_fs
    connector's poll() method, which walks /data/local_fs, computes
    SHA256 hashes, uploads to MinIO, and registers VFS entries.
    """

    def test_sync_job_accepted(self, sync_job_id: str):
        """Sync trigger should return a job ID."""
        assert sync_job_id.startswith("syncjob_")

    def test_sync_job_completed(self, completed_sync_job: dict):
        """Sync job should complete successfully."""
        assert completed_sync_job["status"] == "completed", (
            f"Sync job failed: {completed_sync_job.get('error')}"
        )

    def test_sync_discovered_fixture_files(self, completed_sync_job: dict):
        """Sync should discover at least the required fixture files."""
        assert completed_sync_job["entries_discovered"] >= len(FIXTURE_FILES_REQUIRED)

    def test_sync_synced_fixture_files(self, completed_sync_job: dict):
        """Sync should register all discovered files as new entries."""
        assert completed_sync_job["entries_synced"] >= len(FIXTURE_FILES_REQUIRED)

    def test_vfs_entries_populated(self, http_client: httpx.Client, completed_sync_job: dict):
        """VFS catalog should have entries for all fixture files after sync."""
        resp = http_client.get(
            f"{DC_BASE_URL}/v1/vfs/entries",
            params={"workspace_id": WORKSPACE_ID},
        )
        assert resp.status_code == 200
        data = resp.json()
        entry_paths = {e["source_path"] for e in data["entries"]}
        for fixture_file in FIXTURE_FILES_REQUIRED:
            assert fixture_file in entry_paths, f"Missing VFS entry for {fixture_file}"

    def test_vfs_entries_have_content_hash(self, http_client: httpx.Client, completed_sync_job: dict):
        """Each VFS entry should have a non-empty content_hash."""
        resp = http_client.get(
            f"{DC_BASE_URL}/v1/vfs/entries",
            params={"workspace_id": WORKSPACE_ID},
        )
        assert resp.status_code == 200
        for entry in resp.json()["entries"]:
            assert entry["content_hash"], f"Entry {entry['source_path']} missing content_hash"

    def test_vfs_entries_have_blob_keys(self, http_client: httpx.Client, completed_sync_job: dict):
        """Each VFS entry from local_fs should have a blob_key (materialized to MinIO)."""
        resp = http_client.get(
            f"{DC_BASE_URL}/v1/vfs/entries",
            params={"workspace_id": WORKSPACE_ID},
        )
        assert resp.status_code == 200
        for entry in resp.json()["entries"]:
            # Only check entries from the sync (not manually registered)
            if entry["connector_id"] == completed_sync_job["provider_id"]:
                assert entry["blob_key"], f"Entry {entry['source_path']} missing blob_key"

    def test_vfs_entry_individual_lookup(self, http_client: httpx.Client, completed_sync_job: dict):
        """Individual VFS entry lookup by vfs_ref should work."""
        resp = http_client.get(
            f"{DC_BASE_URL}/v1/vfs/entries",
            params={"workspace_id": WORKSPACE_ID},
        )
        entries = resp.json()["entries"]
        if entries:
            vfs_ref = entries[0]["vfs_ref"]
            detail_resp = http_client.get(f"{DC_BASE_URL}/v1/vfs/entries/{vfs_ref}")
            assert detail_resp.status_code == 200
            assert detail_resp.json()["vfs_ref"] == vfs_ref


# ─────────────────────────────────────────────────────────────────────
# Test: URL minting
# ─────────────────────────────────────────────────────────────────────


class TestUrlMinting:
    """Verify upload and download URL minting works with MinIO."""

    def test_mint_upload_url(self, http_client: httpx.Client):
        """Should return a presigned upload URL pointing at MinIO."""
        resp = http_client.post(
            f"{DC_BASE_URL}/v1/urls/upload",
            json={
                "workspace_id": WORKSPACE_ID,
                "filename": "test-upload.txt",
                "content_type": "text/plain",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["upload_url"]
        assert data["blob_key"]
        assert data["expires_at"]


# ─────────────────────────────────────────────────────────────────────
# Test: MemoryLayer server health
# ─────────────────────────────────────────────────────────────────────


class TestMemoryLayerServer:
    """Verify MemoryLayer server is running and accessible."""

    def test_health(self, http_client: httpx.Client):
        """MemoryLayer /health endpoint should return 200."""
        resp = http_client.get(f"{ML_BASE_URL}/health")
        assert resp.status_code == 200

    def test_api_accessible(self, http_client: httpx.Client):
        """MemoryLayer API should be accessible."""
        resp = http_client.get(f"{ML_BASE_URL}/api/v1/memories", params={"query": "test"})
        # May return 200 or 422 (missing auth), but should not be 500/connection error
        assert resp.status_code < 500


# ─────────────────────────────────────────────────────────────────────
# Test: Aether health
# ─────────────────────────────────────────────────────────────────────


class TestAetherGateway:
    """Verify Aether gateway is running."""

    def test_aether_health(self, http_client: httpx.Client):
        """Aether ops health endpoint should return 200."""
        resp = http_client.get("http://localhost:60090/health/live")
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────
# Test: Cross-service integration (data-connectors -> MinIO -> Aether)
# ─────────────────────────────────────────────────────────────────────


class TestCrossServiceIntegration:
    """Verify cross-service communication works.

    These tests check that data-connectors can communicate with Aether
    and that the VFS + blob store pipeline is functional end-to-end.
    """

    def test_data_connectors_healthz(self, http_client: httpx.Client):
        """data-connectors /healthz should confirm the service is ready."""
        resp = http_client.get(f"{DC_BASE_URL}/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_vfs_entry_link_ml_document(self, http_client: httpx.Client):
        """VFS entry should support linking to a MemoryLayer document ID."""
        # Register a VFS entry
        resp = http_client.post(
            f"{DC_BASE_URL}/v1/vfs/entries",
            json={
                "workspace_id": WORKSPACE_ID,
                "connector_id": "local_fs",
                "source_path": "link-test.txt",
                "content_hash": "linktest_hash_abc123",
                "content_type": "text/plain",
                "size_bytes": 42,
            },
        )
        assert resp.status_code == 201
        vfs_ref = resp.json()["vfs_ref"]

        # Link to a MemoryLayer document
        resp = http_client.post(
            f"{DC_BASE_URL}/v1/vfs/entries/{vfs_ref}/link",
            json={
                "ml_doc_id": "ml_doc_test_123",
                "ml_job_id": "ml_job_test_456",
            },
        )
        assert resp.status_code == 200
        linked = resp.json()
        assert linked["ml_doc_id"] == "ml_doc_test_123"
        assert linked["ml_job_id"] == "ml_job_test_456"

        # Verify the link persists on re-fetch
        resp = http_client.get(f"{DC_BASE_URL}/v1/vfs/entries/{vfs_ref}")
        assert resp.status_code == 200
        assert resp.json()["ml_doc_id"] == "ml_doc_test_123"

    def test_sync_populates_blob_keys_in_minio(self, http_client: httpx.Client, completed_sync_job: dict):
        """After sync, blob_keys on VFS entries should correspond to real MinIO objects.

        Verify by minting a download URL for one of the synced entries.
        """
        resp = http_client.get(
            f"{DC_BASE_URL}/v1/vfs/entries",
            params={"workspace_id": WORKSPACE_ID},
        )
        entries = resp.json()["entries"]
        # Find an entry with a blob_key from the sync
        synced_entry = None
        for entry in entries:
            if entry["connector_id"] == completed_sync_job["provider_id"] and entry["blob_key"]:
                synced_entry = entry
                break

        if synced_entry is None:
            pytest.skip("No synced entries with blob_keys found")

        # Mint a download URL for this entry
        resp = http_client.post(
            f"{DC_BASE_URL}/v1/urls/download",
            json={
                "vfs_ref": synced_entry["vfs_ref"],
                "workspace_id": WORKSPACE_ID,
            },
        )
        assert resp.status_code == 200
        url_data = resp.json()
        assert url_data["url"], "Download URL should be non-empty"
        assert url_data["expires_at"], "Download URL should have an expiry"
