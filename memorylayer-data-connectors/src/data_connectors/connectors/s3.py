"""S3 connector — syncs files from an external S3 bucket.

Discovers objects in a configured S3 prefix, computes content hashes,
and returns entries for the sync engine to register.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from data_connectors.connectors import ConnectorRegistry

logger = logging.getLogger(__name__)

CONNECTOR_TYPE = "s3"


class S3Connector:
    """Connector that syncs from an external S3-compatible bucket.

    Args:
        bucket: Source bucket name.
        prefix: Object prefix to scan.
        endpoint_url: S3-compatible endpoint (None for AWS).
        region: AWS region.
        access_key_id: Access key (None for instance role).
        secret_access_key: Secret key (None for instance role).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: Optional[str] = None,
        region: str = "us-east-1",
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix
        self._endpoint_url = endpoint_url
        self._region = region
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key

    def _get_session_kwargs(self) -> dict:
        """Build kwargs for aioboto3 session client creation."""
        kwargs: dict = {
            "service_name": "s3",
            "region_name": self._region,
        }
        if self._endpoint_url:
            kwargs["endpoint_url"] = self._endpoint_url
        if self._access_key_id and self._secret_access_key:
            kwargs["aws_access_key_id"] = self._access_key_id
            kwargs["aws_secret_access_key"] = self._secret_access_key
        return kwargs

    async def load(self) -> None:
        """Validate S3 credentials by listing the bucket."""
        # Delayed import: aioboto3 pulls in botocore
        import aioboto3

        session = aioboto3.Session()
        async with session.client(**self._get_session_kwargs()) as s3:
            await s3.head_bucket(Bucket=self._bucket)
        logger.info("S3Connector loaded: bucket=%s, prefix=%s", self._bucket, self._prefix)

    async def poll(self) -> list[dict[str, Any]]:
        """List objects in the configured S3 prefix and return entry dicts.

        Each entry includes source_path, content_hash (ETag-based),
        content_type, and size_bytes.
        """
        import aioboto3

        entries: list[dict[str, Any]] = []
        session = aioboto3.Session()
        async with session.client(**self._get_session_kwargs()) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    etag = obj.get("ETag", "").strip('"')
                    # Use ETag as content_hash; for multipart uploads this is
                    # not a pure MD5 but is still deterministic for change detection.
                    content_hash = hashlib.sha256(etag.encode()).hexdigest()
                    entries.append({
                        "source_path": key,
                        "content_hash": content_hash,
                        "size_bytes": obj.get("Size"),
                        "content_type": None,  # HEAD required for content type
                        "metadata": {"s3_etag": etag, "s3_bucket": self._bucket},
                    })
        logger.info("S3Connector polled %d objects from s3://%s/%s", len(entries), self._bucket, self._prefix)
        return entries

    async def get_content_url(self, vfs_ref: str) -> Optional[str]:
        """Generate a presigned GET URL for the source object.

        Note: This returns the upstream presigned URL directly, matching
        the "always upstream URLs" decision.
        """
        # In a full implementation, this would look up the source_path
        # from the VFS catalog and generate a presigned URL. For now,
        # return None (callers use blob_store URLs instead).
        return None

    async def get_metadata(self, vfs_ref: str) -> dict[str, Any]:
        """Return metadata for an S3 object."""
        return {"connector_type": CONNECTOR_TYPE}


ConnectorRegistry.register(CONNECTOR_TYPE, S3Connector)
