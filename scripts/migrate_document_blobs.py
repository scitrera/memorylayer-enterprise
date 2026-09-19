#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Copy local canonical document blobs to one tenant's blobgw domain.

Run with ingestion stopped. Defaults to inventory only. --copy is idempotent,
verifies every object by reading it back, refuses differing destination bytes,
and never deletes local files or changes the active storage provider. Keep the
same base path when switching providers so existing database paths remain valid.
"""
import argparse
import hashlib
import json
from pathlib import Path
from memorylayer_saas.services.document.blob_storage_blobgw import tenant_client, _content_type_for


def migrate(root, client=None, *, tenant=None):
    root = root.resolve(strict=True)
    files = sorted(p for p in root.rglob('*') if p.is_file() or p.is_symlink())
    total = 0
    if client:
        if not tenant:
            raise ValueError('An explicit destination tenant is required')
        # Read-only preflight. The scoped adapter verifies the response header,
        # including an empty listing, before this script can write any object.
        client.list('__migration_domain_check__/')
    for path in files:
        if path.is_symlink():
            raise ValueError('Source contains a symlink; migration stopped')
        before = path.stat()
        data = path.read_bytes()
        digest = hashlib.sha256(data).digest()
        if client:
            ref = str(path).lstrip('/')
            if client.exists(ref):
                if client.head(ref).domain != tenant:
                    raise ValueError('Destination tenant domain mismatch')
                if hashlib.sha256(client.get(ref)).digest() != digest:
                    raise ValueError('Destination differs; refusing to overwrite an existing blob')
            else:
                client.put(ref, data, _content_type_for(str(path)))
            if client.head(ref).domain != tenant:
                raise ValueError('Destination tenant domain mismatch')
            if hashlib.sha256(client.get(ref)).digest() != digest:
                raise ValueError('Destination read-back verification failed')
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed during migration; stop ingestion before retrying')
        total += len(data)
    after_files = sorted(p for p in root.rglob('*') if p.is_file() or p.is_symlink())
    if files != after_files:
        raise ValueError('Source inventory changed during migration')
    return {'files':len(files), 'bytes':total, 'verified': client is not None, 'local_files_retained':True, 'destination_domain':tenant if client else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--blobgw-url',required=True)
    parser.add_argument('--tenant',required=True)
    parser.add_argument('--copy',action='store_true')
    args=parser.parse_args()
    client=tenant_client(args.blobgw_url,args.tenant) if args.copy else None
    print(json.dumps(migrate(args.source_root,client,tenant=args.tenant)))

if __name__=='__main__':main()
