# Source page delivery

`POST /v1/documents/{document}/pages/{page}/file` authorizes the owning workspace
and returns metadata and a short-lived URL, without transferring image bytes
through Aether. `delivery: internal` (the default) retains the canonical source
file capability used by task-local materialization.

For an authenticated browser, send `delivery: browser, kind: image`. Configure:

```
MEMORYLAYER_BLOB_STORAGE_SERVICE=blobgw
MEMORYLAYER_BLOBGW_URL=http://blobgw:8080
MEMORYLAYER_BLOBGW_DOMAIN=example
MEMORYLAYER_BLOBGW_EDGE_URL=http://storage-edge:8090
MEMORYLAYER_TENANT_ID=example
```

The domain must match the tenant. The returned relative
`/storage/example/blob/...` URL uses the integration gateway's existing auth
route. It binds the checked user, tenant, object ref and SHA-256 for 120 seconds.
The edge must enforce GET content-hash claims (replacement returns HTTP 412).
Never expose the internal capability-minting endpoint or accept client-supplied
identity headers at the browser gateway. Current workspace authorization is
checked when minting; a previously minted URL can remain usable by that same
signed-in user until expiry, at most 120 seconds. Logout/tenant identity changes
fail at the browser route. Canonical blob deletion invalidates access immediately.

## Migrating existing local blobs

Stop ingestion and other blob writes first. Keep the same configured base path
and run `scripts/migrate_document_blobs.py` inside an environment with the source
volume mounted at its original path and the released blobgw client installed:

```
python scripts/migrate_document_blobs.py --source-root /data/blobs \
  --blobgw-url http://blobgw:8080 --tenant example
# After reviewing the inventory:
python scripts/migrate_document_blobs.py --source-root /data/blobs \
  --blobgw-url http://blobgw:8080 --tenant example --copy
```

The script copies missing refs, refuses differing destination content and reads
back every object to verify its hash. It never deletes the local files or changes
the active provider. Switch the provider only after successful verification.
Retain the old volume for rollback; if new writes have occurred after the switch,
copy those back and verify before reverting. There is deliberately no silent
fallback to stale local objects when a blobgw ref is missing.
