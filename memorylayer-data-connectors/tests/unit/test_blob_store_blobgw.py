"""Unit tests for the blobgw-backed BlobStore backend.

Mocks both edge and internal-gateway HTTP surfaces with ``httpx.MockTransport``
(no live edge/gateway needed). Covers:

* generate_upload_url resolves the two edge steps (STAGE cap -> POST /staged)
  server-side and returns a single presigned PUT.
* generate_download_url mints a GET capability and returns the absolute edge URL.
* finalize_blob returns size + content_hash from the edge (the round-trip the
  app.py finalize path skips for this backend).
* Direct I/O (put/get/delete/head) against the internal gateway, including the
  S3-shaped head_object dict.
* not-found mapping: internal 404 -> BlobgwNotFound (a FileNotFoundError);
  head_object -> None; edge 404 -> BlobgwNotFound.
* The blob-store factory selecting the blobgw backend from env.

These tests deliberately exercise the native async httpx path (blobgw_client is
not installed in the data-connectors environment), which is the code that runs
in that deployment.
"""
from __future__ import annotations

import json

import httpx
import pytest

from data_connectors.vfs.blob_store_blobgw import BlobgwBlobStore, BlobgwNotFound
from data_connectors.services.url_minter import UrlMinter
from data_connectors.vfs.catalog import VfsEntry

EDGE = "https://edge.test"
INTERNAL = "https://gw.test"
PUBLIC = "https://storage2.test"
DOMAIN = "tenant-xyz"
SERVICE_ID = "dc-svc-1"


# ---------------------------------------------------------------------------
# Mock transport: a single handler routes by host + path across both surfaces.
# ---------------------------------------------------------------------------

class _FakeBackend:
    """In-memory stand-in for the edge + internal gateway.

    Records requests for assertions and serves canned responses that match the
    documented edge/gateway JSON shapes.
    """

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.staged_tokens: list[str] = []
        self.requests: list[httpx.Request] = []
        self.presigned_url = "https://s3.test/staging/abc?sig=xyz"

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path

        # --- edge: mint capability ---
        if path == "/capabilities":
            # Identity headers must match the edge's HeaderIdentityProvider
            # contract (canonical X-Auth-Tenant-ID / X-Scitrera-User). The old
            # X-Tenant / X-Subject names are no longer recognized by the edge.
            assert request.headers.get("X-Auth-Tenant-ID") == DOMAIN, "mint missing/incorrect X-Auth-Tenant-ID"
            assert request.headers.get("X-Scitrera-User"), "mint missing X-Scitrera-User"
            assert "X-Tenant" not in request.headers, "legacy X-Tenant must not be sent"
            assert "X-Subject" not in request.headers, "legacy X-Subject must not be sent"
            body = json.loads(request.content or b"{}")
            op = body["op"]
            ref = body["ref"]
            if op == "STAGE":
                cap_path = f"/staged/{ref}"
            elif op == "FINALIZE":
                cap_path = f"/finalize/{ref}"
            else:
                cap_path = f"/blob/{ref}"
            return httpx.Response(200, json={
                "token": "cap-token-123",
                "capability_url": f"{cap_path}?cap=cap-token-123",
                "op": op,
                "ref": ref,
                "tenant": DOMAIN,
                "expires_at": "2030-01-01T00:00:00Z",
            })

        # --- edge: POST /staged/{ref} -> presigned S3 upload_url ---
        if path.startswith("/staged/"):
            assert request.url.params.get("cap") == "cap-token-123"
            ref = path[len("/staged/"):]
            return httpx.Response(200, json={
                "ref": ref,
                "upload_url": self.presigned_url,
                "staging_key": f"staging/{ref}",
                "expires_at": "2030-01-01T00:00:00Z",
            })

        # --- edge: POST /finalize/{ref} -> size + content_hash ---
        if path.startswith("/finalize/"):
            assert request.url.params.get("cap") == "cap-token-123"
            ref = path[len("/finalize/"):]
            return httpx.Response(200, json={
                "ref": ref,
                "domain": DOMAIN,
                "content_hash": "deadbeefhash",
                "size": 4096,
                "content_type": "application/pdf",
                "created_at": "2030-01-01T00:00:00Z",
            })

        # --- internal gateway: /v1/objects/{ref} ---
        if path.startswith("/v1/objects/"):
            ref = path[len("/v1/objects/"):]
            if request.method == "PUT":
                ct = request.headers.get("Content-Type", "application/octet-stream")
                self.objects[ref] = (request.content, ct)
                return httpx.Response(200, json={"ref": ref, "size": len(request.content)})
            if request.method == "GET":
                if ref not in self.objects:
                    return httpx.Response(404, json={"error": "not found"})
                data, ct = self.objects[ref]
                return httpx.Response(200, content=data, headers={"Content-Type": ct})
            if request.method == "HEAD":
                if ref not in self.objects:
                    return httpx.Response(404)
                data, ct = self.objects[ref]
                return httpx.Response(200, headers={
                    "Content-Length": str(len(data)),
                    "Content-Type": ct,
                    "ETag": '"abc123hash"',
                })
            if request.method == "DELETE":
                self.objects.pop(ref, None)
                return httpx.Response(200)

        return httpx.Response(500, json={"error": f"unhandled {request.method} {path}"})


@pytest.fixture
def backend() -> _FakeBackend:
    return _FakeBackend()


@pytest.fixture
def store(backend: _FakeBackend) -> BlobgwBlobStore:
    s = BlobgwBlobStore(
        internal_url=INTERNAL,
        edge_url=EDGE,
        domain=DOMAIN,
        service_id=SERVICE_ID,
        transport=httpx.MockTransport(backend.handler),
    )
    # Force the native httpx direct-I/O path even if blobgw_client happens to be
    # importable in some environment: the client would bypass the mock transport.
    s._client = None
    return s


# ---------------------------------------------------------------------------
# generate_upload_url
# ---------------------------------------------------------------------------

class TestGenerateUploadUrl:
    async def test_resolves_two_edge_steps_into_single_put(self, store, backend):
        result = await store.generate_upload_url(
            "ws1/manual_upload/deadbeef/report.pdf",
            content_type="application/pdf",
        )
        assert result["method"] == "PUT"
        assert result["url"] == backend.presigned_url
        assert result["fields"] == {}
        assert result["headers"] == {"Content-Type": "application/pdf"}
        assert result["expires_at"] is not None

        # Two edge round-trips happened server-side: capability mint + /staged.
        paths = [r.url.path for r in backend.requests]
        assert paths[0] == "/capabilities"
        assert paths[1].startswith("/staged/")

    async def test_capability_body_carries_stage_op(self, store, backend):
        await store.generate_upload_url("k/obj", content_type="text/plain")
        cap_req = backend.requests[0]
        body = json.loads(cap_req.content)
        assert body["op"] == "STAGE"
        assert body["ref"] == "k/obj"
        assert body["content_type"] == "text/plain"

    async def test_stage_stays_internal_and_not_auth_bound(self, backend):
        """Upload stays on the internal edge: STAGE mint + /staged both hit the
        internal edge (never the public base), with no require_auth, and returns
        the S3 upload_url unchanged."""
        s = BlobgwBlobStore(
            internal_url=INTERNAL, edge_url=EDGE, domain=DOMAIN, service_id=SERVICE_ID,
            public_url=PUBLIC,  # set != edge to prove upload ignores the public base
            transport=httpx.MockTransport(backend.handler),
        )
        s._client = None

        result = await s.generate_upload_url("k/obj", content_type="text/plain")
        # Presigned S3 upload_url is returned verbatim (no public base rewrite).
        assert result["url"] == backend.presigned_url

        # Both server-side round-trips went to the INTERNAL edge, not PUBLIC.
        mint_req, staged_req = backend.requests[0], backend.requests[1]
        assert str(mint_req.url).startswith(EDGE)
        assert str(staged_req.url).startswith(EDGE)
        # STAGE mint is not auth-bound.
        assert "require_auth" not in json.loads(mint_req.content)
        # STAGE asserts the service-id subject (no per-request subject override):
        # uploads are bearer/no-require_auth, so no end user is bound.
        assert mint_req.headers.get("X-Scitrera-User") == SERVICE_ID
        assert mint_req.headers.get("X-Auth-Tenant-ID") == DOMAIN


# ---------------------------------------------------------------------------
# generate_download_url
# ---------------------------------------------------------------------------

class TestGenerateDownloadUrl:
    async def test_default_public_url_uses_edge_base(self, store, backend):
        """Back-compat: unset public_url (None) -> download URL uses edge_url."""
        url, headers, expires_at = await store.generate_download_url("ws1/doc.pdf")
        assert url == f"{EDGE}/blob/ws1/doc.pdf?cap=cap-token-123"
        assert headers == {}
        assert expires_at is not None

        body = json.loads(backend.requests[0].content)
        assert body["op"] == "GET"
        assert body["ref"] == "ws1/doc.pdf"

    async def test_default_mint_is_auth_bound(self, store, backend):
        """download_require_auth defaults True -> mint body carries require_auth."""
        await store.generate_download_url("ws1/doc.pdf")
        body = json.loads(backend.requests[0].content)
        assert body["op"] == "GET"
        assert body["require_auth"] is True
        # match_mode is never sent (edge defaults mm=exact under require_auth).
        assert "match_mode" not in body

    async def test_public_base_and_internal_mint(self, backend):
        """public_url != edge_url: returned URL uses the public base, but the
        capability mint still hits the INTERNAL edge."""
        s = BlobgwBlobStore(
            internal_url=INTERNAL, edge_url=EDGE, domain=DOMAIN, service_id=SERVICE_ID,
            public_url=PUBLIC,
            transport=httpx.MockTransport(backend.handler),
        )
        s._client = None

        url, _, _ = await s.generate_download_url("ws1/doc.pdf")
        # Returned (browser-facing) URL uses the public base.
        assert url.startswith(PUBLIC)
        assert url == f"{PUBLIC}/blob/ws1/doc.pdf?cap=cap-token-123"

        # The mint POST went to the INTERNAL edge, not the public base.
        mint_req = backend.requests[0]
        assert mint_req.url.path == "/capabilities"
        assert str(mint_req.url).startswith(EDGE)

    async def test_require_auth_false_omits_field(self, backend):
        """download_require_auth=False -> no require_auth in the mint body."""
        s = BlobgwBlobStore(
            internal_url=INTERNAL, edge_url=EDGE, domain=DOMAIN, service_id=SERVICE_ID,
            public_url=PUBLIC, download_require_auth=False,
            transport=httpx.MockTransport(backend.handler),
        )
        s._client = None

        await s.generate_download_url("ws1/doc.pdf")
        body = json.loads(backend.requests[0].content)
        assert body["op"] == "GET"
        assert "require_auth" not in body

    async def test_subject_binds_mint_and_auth_bound(self, store, backend):
        """A download with a subject asserts that user as X-Scitrera-User in the
        mint AND (require_auth on by default) sets require_auth in the body, so
        the auth-bound token's claims.Subject == the end user."""
        await store.generate_download_url("ws1/doc.pdf", subject="alice@x")
        mint_req = backend.requests[0]
        assert mint_req.headers.get("X-Scitrera-User") == "alice@x"
        assert mint_req.headers.get("X-Auth-Tenant-ID") == DOMAIN
        body = json.loads(mint_req.content)
        assert body["op"] == "GET"
        assert body["require_auth"] is True

    async def test_no_subject_falls_back_to_service_id(self, store, backend):
        """A download without a subject asserts the dc service id (back-compat)."""
        await store.generate_download_url("ws1/doc.pdf")
        mint_req = backend.requests[0]
        assert mint_req.headers.get("X-Scitrera-User") == SERVICE_ID


# ---------------------------------------------------------------------------
# generate_fetch_url (INTERNAL, server-side, bearer)
# ---------------------------------------------------------------------------

class TestGenerateFetchUrl:
    async def test_returns_internal_edge_url_bearer(self, backend):
        """Server-side fetch returns the INTERNAL edge URL (never the public base)
        and mints a BEARER capability (no require_auth), even when public_url is set
        and download_require_auth defaults True."""
        s = BlobgwBlobStore(
            internal_url=INTERNAL, edge_url=EDGE, domain=DOMAIN, service_id=SERVICE_ID,
            public_url=PUBLIC,  # set != edge to prove fetch ignores the public base
            transport=httpx.MockTransport(backend.handler),
        )
        s._client = None

        url, headers, expires_at = await s.generate_fetch_url("ws1/doc.pdf")
        # Internal edge base, NOT the public/storage2 base.
        assert url == f"{EDGE}/blob/ws1/doc.pdf?cap=cap-token-123"
        assert not url.startswith(PUBLIC)
        assert headers == {}
        assert expires_at is not None

        # Bearer: no require_auth, no per-user subject bound (service-id subject).
        mint_req = backend.requests[0]
        assert str(mint_req.url).startswith(EDGE)
        body = json.loads(mint_req.content)
        assert body["op"] == "GET"
        assert "require_auth" not in body
        assert mint_req.headers.get("X-Scitrera-User") == SERVICE_ID


# ---------------------------------------------------------------------------
# finalize_blob (the size/hash-returning path)
# ---------------------------------------------------------------------------

class TestFinalizeBlob:
    async def test_returns_size_and_hash(self, store, backend):
        info = await store.finalize_blob("ws1/doc.pdf", content_type="application/pdf")
        assert info["size"] == 4096
        assert info["content_hash"] == "deadbeefhash"
        assert info["content_type"] == "application/pdf"

        paths = [r.url.path for r in backend.requests]
        assert paths[0] == "/capabilities"
        assert paths[1] == "/finalize/ws1/doc.pdf"
        assert json.loads(backend.requests[0].content)["op"] == "FINALIZE"


# ---------------------------------------------------------------------------
# direct server-side I/O
# ---------------------------------------------------------------------------

class TestDirectIo:
    async def test_put_then_get_roundtrip(self, store):
        n = await store.put_object("blobs/x/y.bin", b"hello world", "application/octet-stream")
        assert n == len(b"hello world")
        got = await store.get_object("blobs/x/y.bin")
        assert got == b"hello world"

    async def test_head_object_s3_shape(self, store):
        await store.put_object("blobs/h.txt", b"1234567", "text/plain")
        head = await store.head_object("blobs/h.txt")
        assert head is not None
        assert head["ContentLength"] == 7
        assert head["ContentType"] == "text/plain"
        assert head["ContentHash"] == "abc123hash"

    async def test_delete_object(self, store):
        await store.put_object("blobs/d.bin", b"gone", "application/octet-stream")
        await store.delete_object("blobs/d.bin")
        with pytest.raises(BlobgwNotFound):
            await store.get_object("blobs/d.bin")

    async def test_prefix_is_applied_to_refs(self, backend):
        s = BlobgwBlobStore(
            internal_url=INTERNAL, edge_url=EDGE, domain=DOMAIN, service_id=SERVICE_ID,
            prefix="dc/", transport=httpx.MockTransport(backend.handler),
        )
        s._client = None
        await s.put_object("obj.bin", b"data")
        assert "dc/obj.bin" in backend.objects


# ---------------------------------------------------------------------------
# not-found mapping
# ---------------------------------------------------------------------------

class TestNotFoundMapping:
    async def test_get_missing_raises_filenotfound(self, store):
        with pytest.raises(BlobgwNotFound):
            await store.get_object("blobs/does-not-exist")
        # Subclasses FileNotFoundError so S3-style callers behave the same.
        assert issubclass(BlobgwNotFound, FileNotFoundError)

    async def test_head_missing_returns_none(self, store):
        assert await store.head_object("blobs/does-not-exist") is None

    async def test_edge_404_raises_not_found(self, backend):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "no such ref"})

        s = BlobgwBlobStore(
            internal_url=INTERNAL, edge_url=EDGE, domain=DOMAIN, service_id=SERVICE_ID,
            transport=httpx.MockTransport(handler),
        )
        s._client = None
        with pytest.raises(BlobgwNotFound):
            await s.generate_download_url("missing/ref")


# ---------------------------------------------------------------------------
# factory selection
# ---------------------------------------------------------------------------

class TestFactory:
    def test_default_is_s3(self, monkeypatch):
        from data_connectors.vfs import blob_store_factory as f

        monkeypatch.delenv("DC_BLOB_TYPE", raising=False)
        assert not f.is_blobgw_backend()
        assert f.blob_backend_type() == "s3"

    def test_selects_blobgw(self, monkeypatch):
        from data_connectors.vfs import blob_store_factory as f

        monkeypatch.setenv("DC_BLOB_TYPE", "blobgw")
        monkeypatch.setenv("DC_BLOBGW_URL", INTERNAL)
        monkeypatch.setenv("DC_BLOBGW_EDGE_URL", EDGE)
        monkeypatch.setenv("DC_BLOBGW_DOMAIN", DOMAIN)
        monkeypatch.setenv("DC_BLOBGW_SERVICE_ID", SERVICE_ID)
        assert f.is_blobgw_backend()
        store = f.create_blob_store()
        assert isinstance(store, BlobgwBlobStore)
        assert store._domain == DOMAIN
        assert store._service_id == SERVICE_ID
        assert store._internal_url == INTERNAL
        assert store._edge_url == EDGE
        # Unset public URL defaults to the internal edge (back-compat / safe).
        assert store._public_url == EDGE
        # Download auth-binding defaults on.
        assert store._download_require_auth is True

    def test_public_url_and_require_auth_from_env(self, monkeypatch):
        from data_connectors.vfs import blob_store_factory as f

        monkeypatch.setenv("DC_BLOB_TYPE", "blobgw")
        monkeypatch.setenv("DC_BLOBGW_URL", INTERNAL)
        monkeypatch.setenv("DC_BLOBGW_EDGE_URL", EDGE)
        monkeypatch.setenv("DC_BLOBGW_EDGE_PUBLIC_URL", PUBLIC)
        monkeypatch.setenv("DC_BLOBGW_DOWNLOAD_REQUIRE_AUTH", "false")
        monkeypatch.setenv("DC_BLOBGW_DOMAIN", DOMAIN)
        store = f.create_blob_store()
        assert isinstance(store, BlobgwBlobStore)
        assert store._edge_url == EDGE
        assert store._public_url == PUBLIC
        assert store._download_require_auth is False


# ---------------------------------------------------------------------------
# app.py finalize conditionalization
# ---------------------------------------------------------------------------

class TestFinalizeRouteUsesBlobgwPath:
    """The finalize route, under the blobgw backend, backfills size/hash from
    ``finalize_blob`` (the edge) and NEVER calls head_object/get_object.
    """

    def test_finalize_backfills_from_edge_not_head_get(self, monkeypatch):
        from unittest.mock import AsyncMock, patch
        from fastapi.testclient import TestClient

        # Select the blobgw backend for is_blobgw_backend() during finalize.
        monkeypatch.setenv("DC_BLOB_TYPE", "blobgw")

        # A blob store whose finalize_blob returns edge-derived size/hash, and
        # whose head_object/get_object would blow up if the S3 path were taken.
        mock_blob = AsyncMock()
        mock_blob.finalize_blob = AsyncMock(
            return_value={"size": 4096, "content_hash": "edgehash", "content_type": "application/pdf"}
        )
        mock_blob.head_object = AsyncMock(side_effect=AssertionError("head_object must not be called"))
        mock_blob.get_object = AsyncMock(side_effect=AssertionError("get_object must not be called"))
        from datetime import datetime, timezone

        mock_blob.generate_upload_url = AsyncMock(return_value={
            "method": "PUT", "url": "https://s3.test/put", "fields": {}, "headers": {},
            "expires_at": datetime.now(timezone.utc),
        })

        with patch("data_connectors.server.app.AetherServiceRegistration") as MockAether, \
             patch("data_connectors.server.app.BlobStore", return_value=mock_blob), \
             patch("data_connectors.vfs.blob_store_factory.create_blob_store", return_value=mock_blob):
            mock_instance = AsyncMock()
            mock_instance.client = None
            mock_instance.connect = AsyncMock()
            mock_instance.disconnect = AsyncMock()
            MockAether.return_value = mock_instance

            from data_connectors.server.app import app
            with TestClient(app) as c:
                # Register a placeholder (empty hash) with a blob_key so finalize
                # attempts server-side derivation.
                mint = c.post("/v1/urls/upload", json={
                    "workspace_id": "ws-bg",
                    "filename": "doc.pdf",
                    "content_type": "application/pdf",
                    "method": "PUT",
                })
                assert mint.status_code == 200
                vfs_ref = mint.json()["vfs_ref"]

                with patch("data_connectors.server.app._sync_engine.emit_doc_added",
                           new=AsyncMock(return_value="dctask_x")):
                    # No content_hash / size in body -> forces server-side derivation
                    # via the blobgw finalize path.
                    resp = c.post(f"/v1/vfs/entries/{vfs_ref}/finalize", json={
                        "initiated_by": "us::alice",
                        "visibility": "private",
                    })

        assert resp.status_code == 200
        data = resp.json()
        assert data["content_hash"] == "edgehash"
        assert data["size_bytes"] == 4096
        mock_blob.finalize_blob.assert_awaited_once()


# ---------------------------------------------------------------------------
# UrlMinter.mint_download_url subject forwarding
# ---------------------------------------------------------------------------

class _RecordingBlobStore:
    """Records the ``subject`` passed to generate_download_url and whether the
    internal generate_fetch_url path was used."""

    def __init__(self) -> None:
        self.download_subject: object = "<unset>"
        self.fetch_called = False

    async def generate_download_url(self, key, ttl_seconds, subject=None):
        self.download_subject = subject
        from datetime import datetime, timezone
        return ("https://public.test/dl", {}, datetime.now(timezone.utc))

    async def generate_fetch_url(self, key, ttl_seconds):
        self.fetch_called = True
        from datetime import datetime, timezone
        return ("https://edge.test/fetch", {}, datetime.now(timezone.utc))


class _OneEntryCatalog:
    """A catalog stub that returns a single canned VFS entry for any ref."""

    def __init__(self, entry: VfsEntry) -> None:
        self._entry = entry

    async def get(self, vfs_ref: str):
        return self._entry


def _entry(workspace_id: str = "ws1") -> VfsEntry:
    return VfsEntry(
        vfs_ref="vfs::x",
        workspace_id=workspace_id,
        connector_id="manual_upload",
        source_path="doc.pdf",
        content_hash="h",
        blob_key="ws1/manual_upload/abc/doc.pdf",
    )


class TestUrlMinterSubjectForwarding:
    async def test_forwards_subject_to_blob_store(self):
        blob = _RecordingBlobStore()
        minter = UrlMinter(blob, _OneEntryCatalog(_entry()))
        await minter.mint_download_url("vfs::x", "ws1", subject="alice@x")
        assert blob.download_subject == "alice@x"

    async def test_absent_subject_forwards_none(self):
        blob = _RecordingBlobStore()
        minter = UrlMinter(blob, _OneEntryCatalog(_entry()))
        await minter.mint_download_url("vfs::x", "ws1")
        assert blob.download_subject is None

    async def test_fetch_url_uses_internal_bearer_path(self):
        """mint_fetch_url (worker/internal) routes to the bearer INTERNAL
        generate_fetch_url — never the public/auth-bound generate_download_url —
        and forwards no subject."""
        blob = _RecordingBlobStore()
        minter = UrlMinter(blob, _OneEntryCatalog(_entry()))
        await minter.mint_fetch_url("vfs::x", "ws1")
        assert blob.fetch_called is True
        assert blob.download_subject == "<unset>"  # download path untouched
