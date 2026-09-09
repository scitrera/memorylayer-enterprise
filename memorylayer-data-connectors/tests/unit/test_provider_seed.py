"""Tests for the built-in manual_upload provider seed.

Direct uploads tag vfs_entries.connector_id='manual_upload', which FKs to
providers.id — so a 'manual_upload' provider row must exist. These cover the
seed record shape and the idempotent ensure().
"""

import pytest

from data_connectors.db.provider_store import InMemoryProviderStore, _PROVIDER_FIELDS
from data_connectors.server.app import (
    MANUAL_UPLOAD_PROVIDER_ID,
    _builtin_manual_upload_provider,
)

# NOT NULL columns on the providers table (must be present + non-None in the seed).
_REQUIRED = ("id", "workspace_id", "name", "provider_type", "enabled",
             "connection_args", "metadata", "created_at", "updated_at")


def test_seed_record_id_matches_upload_connector_id():
    rec = _builtin_manual_upload_provider()
    assert rec["id"] == MANUAL_UPLOAD_PROVIDER_ID == "manual_upload"
    assert rec["provider_type"] == "manual_upload"


def test_seed_record_has_all_provider_fields_and_required_non_null():
    rec = _builtin_manual_upload_provider()
    for field in _PROVIDER_FIELDS:
        assert field in rec, f"missing provider field: {field}"
    for field in _REQUIRED:
        assert rec[field] is not None, f"required field is None: {field}"
    assert rec["enabled"] is True


@pytest.mark.asyncio
async def test_in_memory_ensure_is_idempotent():
    store = InMemoryProviderStore()
    rec = _builtin_manual_upload_provider()

    await store.ensure(rec)
    got = await store.get("manual_upload")
    assert got is not None and got["name"] == "Manual Upload"

    # A second ensure must NOT clobber the existing row.
    await store.ensure({**rec, "name": "CHANGED"})
    assert (await store.get("manual_upload"))["name"] == "Manual Upload"
