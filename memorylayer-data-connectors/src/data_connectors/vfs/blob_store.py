# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""S3-compatible blob storage backend.

Provides presigned URL generation (upload/download) and direct put/get/delete
operations via aioboto3.  Designed to work with any S3-compatible object store
(AWS S3, MinIO, DigitalOcean Spaces, etc.).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Default presigned URL TTL
_DEFAULT_UPLOAD_TTL_S = 3600  # 1 hour
_DEFAULT_DOWNLOAD_TTL_S = 3600  # 1 hour


class BlobStore:
    """S3-compatible blob storage with presigned URL support.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix for all objects (e.g. "data-connectors/").
        endpoint_url: S3-compatible endpoint URL (None for AWS).
        region: AWS region (default: us-east-1).
        access_key_id: AWS access key ID (None for instance role).
        secret_access_key: AWS secret access key (None for instance role).
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
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._endpoint_url = endpoint_url
        self._region = region
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key

    def _full_key(self, key: str) -> str:
        """Prepend the configured prefix to a storage key."""
        return f"{self._prefix}{key}"

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

    async def generate_upload_url(
        self,
        key: str,
        content_type: Optional[str] = None,
        ttl_seconds: int = _DEFAULT_UPLOAD_TTL_S,
        method: str = "POST",
    ) -> dict:
        """Generate a presigned upload URL.

        Args:
            key: Object key (without bucket prefix).
            content_type: Optional Content-Type. For POST it becomes a policy
                condition + form field; for PUT it becomes a signed header.
            ttl_seconds: URL validity window.
            method: "POST" for browser-form multipart upload
                (``generate_presigned_post``), "PUT" for raw-body upload
                (``generate_presigned_url("put_object")``). POST is the
                default because the existing browser client builds a
                FormData and POSTs.

        Returns:
            Dict with ``method``, ``url``, ``fields``, ``headers``,
            ``expires_at``. For POST: ``fields`` carries the multipart form
            entries (key, AWSAccessKeyId, policy, signature, etc.) and
            ``headers`` is empty. For PUT: ``headers`` carries the required
            signed headers (e.g. Content-Type) and ``fields`` is empty.
        """
        # Delayed import: aioboto3 pulls in botocore which has a slow import
        import aioboto3

        full_key = self._full_key(key)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

        method_upper = method.upper()
        session = aioboto3.Session()
        async with session.client(**self._get_session_kwargs()) as s3:
            if method_upper == "POST":
                # generate_presigned_post returns a dict whose `fields` already
                # contains everything the multipart form needs. Conditions must
                # match Fields exactly, or S3 rejects the upload at PUT time.
                kwargs: dict = {
                    "Bucket": self._bucket,
                    "Key": full_key,
                    "ExpiresIn": ttl_seconds,
                }
                if content_type:
                    kwargs["Fields"] = {"Content-Type": content_type}
                    kwargs["Conditions"] = [{"Content-Type": content_type}]
                post = await s3.generate_presigned_post(**kwargs)
                return {
                    "method": "POST",
                    "url": post["url"],
                    "fields": dict(post["fields"]),
                    "headers": {},
                    "expires_at": expires_at,
                }

            if method_upper == "PUT":
                params: dict = {"Bucket": self._bucket, "Key": full_key}
                if content_type:
                    params["ContentType"] = content_type
                url = await s3.generate_presigned_url(
                    "put_object",
                    Params=params,
                    ExpiresIn=ttl_seconds,
                )
                headers: dict[str, str] = {}
                if content_type:
                    headers["Content-Type"] = content_type
                return {
                    "method": "PUT",
                    "url": url,
                    "fields": {},
                    "headers": headers,
                    "expires_at": expires_at,
                }

            raise ValueError(
                f"Unsupported upload method {method!r}; expected 'POST' or 'PUT'"
            )

    async def generate_download_url(
        self,
        key: str,
        ttl_seconds: int = _DEFAULT_DOWNLOAD_TTL_S,
        subject: Optional[str] = None,
    ) -> tuple[str, dict[str, str], datetime]:
        """Generate a presigned GET URL for downloading.

        ``subject`` is accepted for signature parity with the blobgw backend
        (which binds an auth-bound token to the end user). S3 presigned URLs
        carry no asserted-identity subject, so it is ignored here.

        Returns:
            Tuple of (url, headers, expires_at).
        """
        import aioboto3

        full_key = self._full_key(key)
        params = {
            "Bucket": self._bucket,
            "Key": full_key,
        }

        session = aioboto3.Session()
        async with session.client(**self._get_session_kwargs()) as s3:
            url = await s3.generate_presigned_url(
                "get_object",
                Params=params,
                ExpiresIn=ttl_seconds,
            )

        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        return url, {}, expires_at

    async def generate_fetch_url(
        self,
        key: str,
        ttl_seconds: int = _DEFAULT_DOWNLOAD_TTL_S,
    ) -> tuple[str, dict[str, str], datetime]:
        """Internal server-side fetch URL.

        Method-for-method parity with the blobgw backend's ``generate_fetch_url``.
        S3 presigned GET URLs are directly reachable with no internal/external
        (public-vs-edge) or auth-bound distinction, so this mirrors
        :meth:`generate_download_url`.
        """
        return await self.generate_download_url(key, ttl_seconds=ttl_seconds)

    async def put_object(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> int:
        """Upload bytes directly to blob storage.

        Returns:
            Number of bytes stored.
        """
        import aioboto3

        full_key = self._full_key(key)
        session = aioboto3.Session()
        async with session.client(**self._get_session_kwargs()) as s3:
            await s3.put_object(
                Bucket=self._bucket,
                Key=full_key,
                Body=data,
                ContentType=content_type,
            )
        return len(data)

    async def get_object(self, key: str) -> bytes:
        """Download bytes directly from blob storage."""
        import aioboto3

        full_key = self._full_key(key)
        session = aioboto3.Session()
        async with session.client(**self._get_session_kwargs()) as s3:
            response = await s3.get_object(Bucket=self._bucket, Key=full_key)
            return await response["Body"].read()

    async def delete_object(self, key: str) -> None:
        """Delete an object from blob storage."""
        import aioboto3

        full_key = self._full_key(key)
        session = aioboto3.Session()
        async with session.client(**self._get_session_kwargs()) as s3:
            await s3.delete_object(Bucket=self._bucket, Key=full_key)

    async def head_object(self, key: str) -> Optional[dict]:
        """Get object metadata without downloading the body.

        Returns:
            Dict with ContentLength, ContentType, etc., or None if not found.
        """
        import aioboto3

        full_key = self._full_key(key)
        session = aioboto3.Session()
        async with session.client(**self._get_session_kwargs()) as s3:
            try:
                return await s3.head_object(Bucket=self._bucket, Key=full_key)
            except Exception:
                return None
