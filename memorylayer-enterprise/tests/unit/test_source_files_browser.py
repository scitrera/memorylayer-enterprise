# SPDX-License-Identifier: AGPL-3.0-only
"""Browser source images use authenticated, hash-bound edge capabilities."""
import hashlib
import io
from logging import getLogger
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import httpx
import pytest
from fastapi import HTTPException
from PIL import Image
from memorylayer_saas.services.document.source_files import export_page
from memorylayer_saas.services.document.blob_storage_blobgw import BlobGWBlobStorageService, tenant_client

class Variables:
 def __init__(self, **kw): self.values={"MEMORYLAYER_TENANT_ID":"example", "MEMORYLAYER_BLOBGW_EDGE_URL":"http://edge",**kw}
 def environ(self,key,default=None): return self.values.get(key,default)

def fixture():
 stream=io.BytesIO();Image.new("RGB",(20,20)).save(stream,format="PNG"); data=stream.getvalue()
 blob=BlobGWBlobStorageService(None,FileNotFoundError,"/blobs",getLogger(),domain="example")
 blob.retrieve_file=AsyncMock(return_value=data)
 page=SimpleNamespace(id="page",document_id="doc",workspace_id="owned",image_storage_path="/blobs/owned/doc/page.png")
 return blob,page,data

@pytest.mark.asyncio
async def test_browser_capability_uses_checked_identity_and_exact_hash():
 blob,page,data=fixture()
 http=AsyncMock();http.post.return_value=httpx.Response(200,json={"capability_url":"/blob/blobs/owned/doc/page.png?cap=short-ticket","expires_at":"soon"})
 http.__aenter__.return_value=http
 with patch("httpx.AsyncClient",return_value=http):
  result=await export_page(Variables(),SimpleNamespace(tenant_id="example",user_id="alice"),page,blob,"image",delivery="browser")
 args=http.post.call_args.kwargs
 assert args["headers"]=={"X-Auth-Tenant-ID":"example","X-Scitrera-User":"alice"}
 assert args["json"]=={"op":"GET","ref":"blobs/owned/doc/page.png","content_hash":hashlib.sha256(data).hexdigest(),"ttl_seconds":120,"require_auth":True,"match_mode":"exact"}
 assert result["url"]=="/storage/example/blob/blobs/owned/doc/page.png?cap=short-ticket"
 assert "data_base64" not in result

@pytest.mark.asyncio
@pytest.mark.parametrize("change",["tenant","domain","user","storage","kind"])
async def test_browser_requires_scope_and_canonical_storage(change):
 blob,page,_=fixture(); ctx=SimpleNamespace(tenant_id="example",user_id="alice");kind="image"
 if change=="tenant":ctx.tenant_id="other"
 if change=="domain":blob.domain="other"
 if change=="user":ctx.user_id=None
 if change=="storage":blob=SimpleNamespace()
 if change=="kind":kind="transcript"
 with patch("httpx.AsyncClient") as http, pytest.raises(HTTPException):
  await export_page(Variables(),ctx,page,blob,kind,delivery="browser")
 http.assert_not_called()

def test_tenant_domain_on_both_released_client_request_hooks():
 from blobgw_client import BlobGWClient
 client=tenant_client("http://blobgw","example")
 with patch.object(BlobGWClient,"_request",return_value=b"data") as req:
  assert client.get("page")==b"data"
  assert req.call_args.kwargs["headers"]["X-Blobgw-Domain"]=="example"
 with patch.object(BlobGWClient,"_request_with_headers",return_value=(b"",{})) as req:
  assert client.exists("page") is True
  assert req.call_args.kwargs["headers"]["X-Blobgw-Domain"]=="example"

@pytest.mark.asyncio
async def test_single_page_cannot_be_read_using_another_workspaces_grant():
 from memorylayer_saas.api.v1 import documents
 ctx=SimpleNamespace(tenant_id="example",workspace_id="allowed")
 auth=SimpleNamespace(build_context=AsyncMock(return_value=ctx))
 authz=SimpleNamespace(require_authorization=AsyncMock())
 storage=SimpleNamespace(get_page=AsyncMock(return_value=SimpleNamespace(document_id="doc",workspace_id="other")))
 with patch.object(documents,"get_extension",return_value=storage), pytest.raises(HTTPException) as exc:
  await documents.get_document_page(SimpleNamespace(),"doc","page",auth,authz,Variables())
 assert exc.value.status_code==404


def test_migration_retains_local_data_and_refuses_different_destination(tmp_path):
 import importlib.util
 from pathlib import Path
 path=Path(__file__).resolve().parents[3]/"scripts/migrate_document_blobs.py"
 spec=importlib.util.spec_from_file_location("migration",path)
 migration=importlib.util.module_from_spec(spec);spec.loader.exec_module(migration)
 class Client:
  def __init__(self):self.files={};self.puts=0
  def exists(self,key):return key in self.files
  def get(self,key):return self.files[key]
  def put(self,key,data,mime):self.files[key]=data;self.puts+=1
 source=tmp_path/"page.png";source.write_bytes(b"image");client=Client()
 assert migration.migrate(tmp_path)["verified"] is False
 assert migration.migrate(tmp_path,client)["verified"] is True
 migration.migrate(tmp_path,client);assert client.puts==1
 client.files[next(iter(client.files))]=b"changed"
 with pytest.raises(ValueError,match="refusing"):
  migration.migrate(tmp_path,client)
 assert source.read_bytes()==b"image"
 (tmp_path/"link").symlink_to(source)
 with pytest.raises(ValueError,match="symlink"):
  migration.migrate(tmp_path)
