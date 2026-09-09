"""Blob-store backend selection.

Chooses the VFS blob backend from ``DC_BLOB_TYPE`` and constructs it from the
environment. Two backends exist:

* ``s3`` (default): S3-compatible :class:`BlobStore` (aioboto3). Uses the
  ``DC_BLOB_*`` knobs (bucket / endpoint / region / key / secret).
* ``blobgw``: :class:`BlobgwBlobStore` routing through the blobgw internal
  gateway + external edge. In this mode the S3 knobs are unused; instead:

  ============================ ===============================================
  Env var                      Meaning
  ============================ ===============================================
  ``DC_BLOBGW_URL``            Internal gateway base URL (direct object I/O)
  ``DC_BLOBGW_EDGE_URL``       Internal edge base URL (mint/stage/finalize)
  ``DC_BLOBGW_EDGE_PUBLIC_URL``Browser-facing download base (default: edge URL)
  ``DC_BLOBGW_DOWNLOAD_REQUIRE_AUTH`` Auth-bind download URLs (default true)
  ``DC_BLOBGW_DOMAIN``         Per-tenant domain / asserted tenant id
  ``DC_BLOBGW_SERVICE_ID``     dc service subject id (identity assertion)
  ``DC_BLOBGW_UPLOAD_TTL_S``   Upload capability/presign TTL (default 3600)
  ``DC_BLOBGW_DOWNLOAD_TTL_S`` Download capability/presign TTL (default 3600)
  ``DC_BLOB_PREFIX``           Optional ref prefix (shared with the S3 backend)
  ============================ ===============================================

Follows the package's config idiom: env parsing via ``os.environ`` directly (as
elsewhere in data-connectors, which does not depend on
``scitrera_app_framework`` in its own source).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Backend selector + its accepted values.
DC_BLOB_TYPE = "DC_BLOB_TYPE"
BLOB_TYPE_S3 = "s3"
BLOB_TYPE_BLOBGW = "blobgw"

# blobgw backend config keys.
DC_BLOBGW_URL = "DC_BLOBGW_URL"
DC_BLOBGW_EDGE_URL = "DC_BLOBGW_EDGE_URL"
DC_BLOBGW_EDGE_PUBLIC_URL = "DC_BLOBGW_EDGE_PUBLIC_URL"
DC_BLOBGW_DOWNLOAD_REQUIRE_AUTH = "DC_BLOBGW_DOWNLOAD_REQUIRE_AUTH"
DC_BLOBGW_DOMAIN = "DC_BLOBGW_DOMAIN"
DC_BLOBGW_SERVICE_ID = "DC_BLOBGW_SERVICE_ID"
DC_BLOBGW_UPLOAD_TTL_S = "DC_BLOBGW_UPLOAD_TTL_S"
DC_BLOBGW_DOWNLOAD_TTL_S = "DC_BLOBGW_DOWNLOAD_TTL_S"

_DEFAULT_INTERNAL_URL = "http://localhost:8080"
_DEFAULT_EDGE_URL = "http://localhost:8090"
# Placeholder tenant domain. PARAMETERIZED: the platform finalizes the exact
# string later; keep it config-driven and never hard-coded at a call site.
_DEFAULT_DOMAIN = "data-connectors"
_DEFAULT_SERVICE_ID = "data-connectors"
_DEFAULT_TTL_S = 3600


def blob_backend_type() -> str:
    """The configured backend selector, lowercased (default ``s3``)."""
    return os.environ.get(DC_BLOB_TYPE, BLOB_TYPE_S3).strip().lower()


def is_blobgw_backend() -> bool:
    """True when the blobgw backend is selected.

    app.py uses this to conditionalize the finalize path: only the blobgw
    backend returns size/hash from finalize (skipping head_object/get_object).
    """
    return blob_backend_type() == BLOB_TYPE_BLOBGW


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r; using default %d", name, raw, default)
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    logger.warning("Invalid bool for %s=%r; using default %s", name, raw, default)
    return default


def create_blob_store():
    """Construct the configured blob-store backend.

    Returns an object duck-compatible with the shared blob-store interface
    (:class:`BlobStore` for ``s3`` or :class:`BlobgwBlobStore` for ``blobgw``).
    """
    backend = blob_backend_type()
    prefix = os.environ.get("DC_BLOB_PREFIX", "")

    if backend == BLOB_TYPE_BLOBGW:
        from data_connectors.vfs.blob_store_blobgw import BlobgwBlobStore

        internal_url = os.environ.get(DC_BLOBGW_URL, _DEFAULT_INTERNAL_URL)
        edge_url = os.environ.get(DC_BLOBGW_EDGE_URL, _DEFAULT_EDGE_URL)
        # Browser-facing download base; empty -> internal edge_url (safe default).
        public_url = os.environ.get(DC_BLOBGW_EDGE_PUBLIC_URL, "").strip() or edge_url
        download_require_auth = _bool_env(DC_BLOBGW_DOWNLOAD_REQUIRE_AUTH, True)
        domain = os.environ.get(DC_BLOBGW_DOMAIN, _DEFAULT_DOMAIN)
        service_id = os.environ.get(DC_BLOBGW_SERVICE_ID, _DEFAULT_SERVICE_ID)
        upload_ttl = _int_env(DC_BLOBGW_UPLOAD_TTL_S, _DEFAULT_TTL_S)
        download_ttl = _int_env(DC_BLOBGW_DOWNLOAD_TTL_S, _DEFAULT_TTL_S)
        logger.info(
            "Blob backend: blobgw (internal=%s, edge=%s, public=%s, require_auth=%s, "
            "domain=%s, service_id=%s)",
            internal_url, edge_url, public_url, download_require_auth, domain, service_id,
        )
        return BlobgwBlobStore(
            internal_url=internal_url,
            edge_url=edge_url,
            domain=domain,
            service_id=service_id,
            prefix=prefix,
            upload_ttl_s=upload_ttl,
            download_ttl_s=download_ttl,
            public_url=public_url,
            download_require_auth=download_require_auth,
        )

    # Default: S3-compatible backend.
    from data_connectors.vfs.blob_store import BlobStore

    bucket = os.environ.get("DC_BLOB_BUCKET", "data-connectors-dev")
    endpoint_url = os.environ.get("DC_BLOB_ENDPOINT_URL")
    region = os.environ.get("DC_BLOB_REGION", "us-east-1")
    access_key = os.environ.get("DC_BLOB_ACCESS_KEY_ID")
    secret_key = os.environ.get("DC_BLOB_SECRET_ACCESS_KEY")
    logger.info("Blob backend: s3 (bucket=%s, endpoint=%s)", bucket, endpoint_url)
    return BlobStore(
        bucket=bucket,
        prefix=prefix,
        endpoint_url=endpoint_url,
        region=region,
        access_key_id=access_key,
        secret_access_key=secret_key,
    )
