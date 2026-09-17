"""Canonical HTTP file capabilities: authorization, scope and integrity."""

import hashlib
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from PIL import Image

from memorylayer_saas.services.document import source_files


class Variables:
    def __init__(self, configured=True):
        self.values = (
            {
                "MEMORYLAYER_SOURCE_FILES_FETCH_URL": "http://download.test",
                "MEMORYLAYER_TENANT_ID": "jgl",
            }
            if configured
            else {}
        )

    def environ(self, key, default=None):
        return self.values.get(key, default)


def page():
    return SimpleNamespace(
        id="page",
        document_id="doc",
        workspace_id="owned",
        image_storage_path="image.png",
        transcript="Full transcript café",
    )


def token(descriptor):
    return descriptor["url"].rsplit("/", 1)[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "transcript"])
async def test_capability_serves_canonical_bytes_and_small_descriptor(kind):
    out = io.BytesIO()
    Image.new("RGB", (1000, 500), "white").save(out, format="PNG")
    data = out.getvalue() if kind == "image" else page().transcript.encode()
    blob = SimpleNamespace(retrieve_file=AsyncMock(return_value=data))
    storage = SimpleNamespace(get_page=AsyncMock(return_value=page()))
    result = await source_files.export_page(Variables(), SimpleNamespace(tenant_id="jgl"), page(), blob, kind)
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["size_bytes"] == len(data)
    assert result["url"].startswith("http://download.test/blob/source-pages/jgl/")
    assert len(json.dumps(result)) < 1600
    assert "data_base64" not in result
    actual, mime = await source_files.redeem_page(Variables(), token(result), storage, blob)
    assert actual == data and mime == result["mime"]
    storage.get_page.assert_awaited_once_with("page")


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["signature", "expired", "restart", "tenant", "workspace", "document", "deleted", "content"])
async def test_stale_tampered_or_foreign_capability_fails_closed(change):
    variables = Variables()
    stored = page()
    blob = AsyncMock()
    storage = SimpleNamespace(get_page=AsyncMock(return_value=stored))
    descriptor = await source_files.export_page(variables, SimpleNamespace(tenant_id="jgl"), stored, blob, "transcript")
    ticket = token(descriptor)
    if change == "signature":
        ticket = ticket[:-1] + ("0" if ticket[-1] != "0" else "1")
    if change == "restart":
        variables.values["MEMORYLAYER_SOURCE_FILES_SIGNING_KEY"] = "new-key" * 8
    if change == "tenant":
        variables.values["MEMORYLAYER_TENANT_ID"] = "other"
    if change == "workspace":
        stored.workspace_id = "other"
    if change == "document":
        stored.document_id = "other"
    if change == "deleted":
        storage.get_page.return_value = None
    if change == "content":
        stored.transcript = "changed"
    now = source_files.time.time() + (source_files.CAP_TTL + 1 if change == "expired" else 0)
    with patch.object(source_files.time, "time", return_value=now), pytest.raises(HTTPException) as exc:
        await source_files.redeem_page(variables, ticket, storage, blob)
    assert exc.value.status_code == 404
    if change in {"signature", "expired", "restart", "tenant"}:
        storage.get_page.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("ticket", ["", "x" * 2049, "a.b.c", "nonsigned", "x." + "0" * 64])
async def test_invalid_tickets_do_not_touch_storage(ticket):
    storage = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await source_files.redeem_page(Variables(), ticket, storage, AsyncMock())
    assert exc.value.status_code == 404
    storage.get_page.assert_not_awaited()


@pytest.mark.asyncio
async def test_mint_requires_configuration_tenant_and_bounded_valid_content():
    blob = SimpleNamespace(retrieve_file=AsyncMock(return_value=b"not an image"))
    for variables, tenant, kind, status in [
        (Variables(False), "jgl", "image", 503),
        (Variables(), "other", "transcript", 403),
        (Variables(), "jgl", "image", 422),
        (Variables(), "jgl", "unknown", 422),
    ]:
        with pytest.raises(HTTPException) as exc:
            await source_files.export_page(variables, SimpleNamespace(tenant_id=tenant), page(), blob, kind)
        assert exc.value.status_code == status
    blob.retrieve_file.return_value = b"x" * (source_files.MAX_IMAGE_BYTES + 1)
    with pytest.raises(HTTPException) as exc:
        await source_files.export_page(Variables(), SimpleNamespace(tenant_id="jgl"), page(), blob, "image")
    assert exc.value.status_code == 422
    variables = Variables()
    variables.values["MEMORYLAYER_SOURCE_FILES_SIGNING_KEY"] = "short"
    with pytest.raises(HTTPException) as exc:
        await source_files.export_page(variables, SimpleNamespace(tenant_id="jgl"), page(), blob, "transcript")
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_download_uses_header_credentials_and_no_store():
    from memorylayer_saas.api.v1 import documents

    request = MagicMock(headers={"Authorization": "Bearer private-ticket"})
    redeem = AsyncMock(return_value=(b"source", "text/plain"))
    with patch.object(documents, "get_extension"), patch.object(source_files, "redeem_page", redeem):
        response = await documents.download_source_file(request, Variables())
        assert response.body == b"source"
        assert response.headers["cache-control"] == "private, no-store"
        assert redeem.call_args.args[1] == "private-ticket"
        request.headers = {}
        with pytest.raises(HTTPException) as exc:
            await documents.download_source_file(request, Variables())
        assert exc.value.status_code == 404


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
