"""Source file authorization and descriptor-only HTTP export contract."""

import hashlib
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from PIL import Image

from memorylayer_saas.services.document import source_files


class Variables:
    def __init__(self, configured=True):
        self.values = (
            {
                "MEMORYLAYER_SOURCE_FILES_BLOB_URL": "http://blob.test",
                "MEMORYLAYER_SOURCE_FILES_EDGE_URL": "http://edge.test",
                "MEMORYLAYER_SOURCE_FILES_FETCH_URL": "http://download.test",
            }
            if configured
            else {}
        )

    def environ(self, key, default=None):
        return self.values.get(key, default)


def page():
    return SimpleNamespace(
        id="page", document_id="doc", workspace_id="owned", image_storage_path="image.png", transcript="Full transcript café"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "transcript"])
async def test_export_bytes_use_http_and_control_reply_is_small(kind):
    out = io.BytesIO()
    Image.new("RGB", (1000, 500), "white").save(out, format="PNG")
    data = out.getvalue() if kind == "image" else page().transcript.encode()
    requests = []

    def send(request):
        requests.append(request)
        if request.method == "PUT":
            assert request.content == data
            assert request.headers["X-Blobgw-Domain"] == "jgl"
            return httpx.Response(200, json={})
        assert request.headers["X-Auth-Tenant-ID"] == "jgl"
        body = json.loads(request.content)
        assert body["op"] == "GET" and body["ttl_seconds"] == 120
        return httpx.Response(200, json={"capability_url": "/blob/ref?cap=secret", "expires_at": "2099-01-01T00:00:00Z"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(send))
    with patch.object(source_files.httpx, "AsyncClient", return_value=client):
        result = await source_files.export_page(
            Variables(), SimpleNamespace(tenant_id="jgl"), page(), AsyncMock(retrieve_file=AsyncMock(return_value=data)), kind
        )
    assert len(requests) == 2
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["size_bytes"] == len(data)
    assert result["url"] == "http://download.test/blob/ref?cap=secret"
    assert len(json.dumps(result)) < 1024
    assert "data_base64" not in result


@pytest.mark.asyncio
async def test_export_requires_configuration_and_bounded_valid_image():
    blob = AsyncMock(retrieve_file=AsyncMock(return_value=b"not an image"))
    for variables, status in [(Variables(False), 503), (Variables(), 422)]:
        with pytest.raises(HTTPException) as exc:
            await source_files.export_page(variables, SimpleNamespace(tenant_id="jgl"), page(), blob, "image")
        assert exc.value.status_code == status
    blob.retrieve_file.return_value = b"x" * (source_files.MAX_IMAGE_BYTES + 1)
    with pytest.raises(HTTPException) as exc:
        await source_files.export_page(Variables(), SimpleNamespace(tenant_id="jgl"), page(), blob, "image")
    assert exc.value.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("denied,wrong_doc", [(False, False), (True, False), (False, True)])
async def test_endpoint_authorizes_owning_workspace_before_export(denied, wrong_doc):
    from memorylayer_saas.api.v1 import documents

    ctx = SimpleNamespace(tenant_id="jgl", workspace_id="unrelated")
    auth = SimpleNamespace(build_context=AsyncMock(return_value=ctx))
    authz = SimpleNamespace(require_authorization=AsyncMock(side_effect=HTTPException(403) if denied else None))
    stored = page()
    stored.document_id = "other" if wrong_doc else "doc"
    storage = SimpleNamespace(get_page=AsyncMock(return_value=stored))
    export = AsyncMock(return_value={"version": 1})
    with patch.object(documents, "get_extension", return_value=storage), patch.object(source_files, "export_page", export):
        call = documents.materialize_page_file(
            MagicMock(), "doc", "page", documents.SourceFileRequest(kind="image"), auth, authz, Variables()
        )
        if denied or wrong_doc:
            with pytest.raises(HTTPException) as exc:
                await call
            assert exc.value.status_code == 404
            export.assert_not_awaited()
        else:
            assert await call == {"version": 1}
            export.assert_awaited_once()
    if not wrong_doc:
        authz.require_authorization.assert_awaited_once_with(ctx, "documents", "read", workspace_id="owned")


@pytest.mark.asyncio
async def test_delete_exports_is_scoped_and_optional():
    await source_files.delete_document_files(Variables(False), "jgl", "workspace", "doc")

    def send(request):
        assert request.method == "DELETE"
        assert request.headers["X-Blobgw-Domain"] == "jgl"
        assert request.url.params["prefix"] == source_files.document_prefix("workspace", "doc")
        return httpx.Response(200, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(send))
    with patch.object(source_files.httpx, "AsyncClient", return_value=client):
        await source_files.delete_document_files(Variables(), "jgl", "workspace", "doc")
