"""blobgw-backed blob storage backend.

A drop-in, duck-compatible alternative to the S3 :class:`BlobStore` that routes
VFS blob I/O through the **blobgw** object gateway plus its external **edge**.
It matches :class:`data_connectors.vfs.blob_store.BlobStore` method-for-method
(no ABC to inherit; the caller — :class:`UrlMinter` — is backend-agnostic and
only depends on the shared signatures):

    generate_upload_url(key, content_type, ttl_seconds, method) -> dict
    generate_download_url(key, ttl_seconds)                      -> (url, headers, expires_at)
    put_object(key, data, content_type)                         -> int
    get_object(key)                                             -> bytes
    delete_object(key)                                          -> None
    head_object(key)                                           -> Optional[dict]

Two distinct HTTP surfaces are involved:

* The **edge** (``DC_BLOBGW_EDGE_URL``) mints per-tenant *capabilities* from an
  asserted-identity header and presigns S3 for browser uploads/downloads. It is
  used by :meth:`generate_upload_url` / :meth:`generate_download_url` and by the
  server-side ``FINALIZE`` step (see ``finalize_blob`` and the app.py finalize
  path). Presigned uploads keep the browser UX unchanged: one PUT. dc resolves
  the two edge steps (mint STAGE cap -> POST /staged) server-side.

* The **internal** gateway (``DC_BLOBGW_URL``) serves direct server-side object
  I/O (``/v1/objects/{ref}``) for :meth:`put_object` / :meth:`get_object` /
  :meth:`delete_object` / :meth:`head_object`. When the optional
  ``blobgw_client`` package is importable it is reused (wrapped in
  ``asyncio.to_thread``); otherwise a minimal async ``httpx`` client speaks the
  same data-plane API.

Refs are addressed within a per-tenant *domain* (``DC_BLOBGW_DOMAIN``). The
exact domain string is parameterized (config-driven) so the platform can
finalize it later without a code change. The edge asserts the tenant from the
identity header; the internal gateway is addressed to the same domain via the
``X-Blobgw-Domain`` header.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Default TTLs (seconds). Mirror the S3 backend defaults so behavior matches.
_DEFAULT_UPLOAD_TTL_S = 3600
_DEFAULT_DOWNLOAD_TTL_S = 3600

# Identity headers the edge reads to mint per-tenant capabilities. The edge's
# HeaderIdentityProvider derives (tenant, subject) from them: tenant is the
# per-tenant domain (``X-Auth-Tenant-ID``), subject is the asserted user/service
# id (``X-Scitrera-User``). Kept as module constants so the header names live in
# one place. These are the canonical names the reconciled edge reads; older
# names (X-Tenant / X-Subject) are no longer recognized and mint 401.
_TENANT_HEADER = "X-Auth-Tenant-ID"
_SUBJECT_HEADER = "X-Scitrera-User"


class BlobgwNotFound(FileNotFoundError):
    """Raised when an object ref does not exist on the internal gateway.

    Subclasses :class:`FileNotFoundError` so callers that catch the S3 backend's
    "missing object" behavior (or a generic ``FileNotFoundError``) behave
    identically against this backend.
    """


class BlobgwBlobStore:
    """blobgw + edge backed blob storage, duck-compatible with ``BlobStore``.

    Args:
        internal_url: Internal blobgw base URL for direct server-side I/O
            (``DC_BLOBGW_URL``), e.g. ``http://blobgw.internal:8080``.
        edge_url: Internal edge base URL for capabilities + presigned URLs
            (``DC_BLOBGW_EDGE_URL``), e.g. ``https://edge.blobgw.example``. This
            is the mint/stage/finalize target: all edge round-trips that dc makes
            server-side (capability mint, ``POST /staged``, ``POST /finalize``)
            hit this internal edge, which is why it must NOT be strict-auth.
        public_url: Browser-facing download base (e.g. ``storage2``), distinct
            from ``edge_url``. Only the download URL returned by
            :meth:`generate_download_url` is built from this public base, since
            it is the one URL a browser GETs directly. Defaults to ``edge_url``
            when unset (SAFE/non-breaking), so an unconfigured deployment keeps
            serving downloads off the internal edge.
        download_require_auth: When True (default), download capabilities are
            minted auth-bound (``require_auth``) so only the intended user can
            redeem the returned URL. Minting still happens server-side against
            the internal edge; only the returned URL is auth-bound + public.
        domain: Per-tenant domain / tenant id asserted to the edge and used to
            address the internal gateway (``DC_BLOBGW_DOMAIN``). Parameterized:
            the platform finalizes the exact string later.
        service_id: dc service subject id asserted in the identity header.
        prefix: Optional key prefix applied to every ref (parity with the S3
            backend's ``prefix``). Empty by default.
        upload_ttl_s / download_ttl_s: Default capability/presign TTLs.
        timeout: Per-request HTTP timeout (seconds).
    """

    def __init__(
        self,
        internal_url: str,
        edge_url: str,
        domain: str,
        service_id: str = "data-connectors",
        prefix: str = "",
        upload_ttl_s: int = _DEFAULT_UPLOAD_TTL_S,
        download_ttl_s: int = _DEFAULT_DOWNLOAD_TTL_S,
        timeout: float = 60.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        public_url: Optional[str] = None,
        download_require_auth: bool = True,
    ) -> None:
        self._internal_url = internal_url.rstrip("/")
        self._edge_url = edge_url.rstrip("/")
        # Browser-facing download base; defaults to the internal edge when unset
        # (SAFE/non-breaking). Only generate_download_url uses it.
        self._public_url = (public_url or edge_url).rstrip("/")
        self._download_require_auth = download_require_auth
        self._domain = domain
        self._service_id = service_id
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._upload_ttl_s = upload_ttl_s
        self._download_ttl_s = download_ttl_s
        self._timeout = timeout
        # Optional injected transport (tests use httpx.MockTransport). None in
        # production, where httpx picks its default network transport.
        self._transport = transport

        # Optional reuse of the standard-library blobgw_client for direct I/O.
        # Imported lazily (and tolerantly) so this backend never hard-depends on
        # the package being installed in the data-connectors environment.
        self._client = None
        self._not_found: type[Exception] = BlobgwNotFound
        try:
            from blobgw_client import BlobGWClient, NotFound  # type: ignore

            self._client = BlobGWClient(self._internal_url, timeout=timeout)
            self._not_found = NotFound
            logger.info("blobgw backend: using blobgw_client for direct I/O")
        except Exception:
            logger.info(
                "blobgw backend: blobgw_client unavailable; using native async httpx for direct I/O"
            )

    # -- helpers ------------------------------------------------------------

    def _http(self) -> httpx.AsyncClient:
        """A per-call async HTTP client (honors an injected test transport)."""
        if self._transport is not None:
            return httpx.AsyncClient(timeout=self._timeout, transport=self._transport)
        return httpx.AsyncClient(timeout=self._timeout)

    def _full_key(self, key: str) -> str:
        """Apply the configured prefix and normalize to a gateway ref.

        Refs are not absolute, so a leading ``/`` is stripped (mirrors the
        enterprise blobgw service convention).
        """
        return f"{self._prefix}{key}".lstrip("/")

    def _identity_header(self, subject: Optional[str] = None) -> dict[str, str]:
        """Asserted-identity headers the edge mints capabilities from.

        Matches the edge's ``HeaderIdentityProvider`` contract, which reads the
        tenant from ``X-Auth-Tenant-ID`` and the subject from ``X-Scitrera-User``
        (the same proxy headers memorylayer/Aether inject). tenant = the
        per-tenant domain; subject = the asserted user. When ``subject`` is None
        (uploads/stage/finalize — bearer/no require_auth) it falls back to the dc
        service id; download passes the end user so an auth-bound token binds to
        that user. In production these are asserted by the auth proxy in front of
        the edge; dc sets them explicitly for the direct/dev path.
        """
        return {
            _TENANT_HEADER: self._domain,
            _SUBJECT_HEADER: subject or self._service_id,
        }

    def _domain_header(self) -> dict[str, str]:
        """Header addressing the internal gateway to this tenant's domain."""
        return {"X-Blobgw-Domain": self._domain}

    async def _mint_capability(
        self,
        op: str,
        ref: str,
        *,
        content_type: Optional[str] = None,
        max_size: Optional[int] = None,
        content_hash: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        require_auth: bool = False,
        subject: Optional[str] = None,
    ) -> dict:
        """POST {edge}/capabilities and return the capability descriptor.

        Returns the parsed JSON: ``{token, capability_url, op, ref, tenant,
        expires_at}``. ``capability_url`` is a path + ``?cap=`` that callers
        prefix with the edge (or public) base for an absolute URL.

        When ``require_auth`` is True, the edge's ``require_auth`` request field
        is set so the minted capability is auth-bound (redeemable only by the
        intended user). When False (default), the existing bearer behavior is
        kept and the field is omitted.

        ``subject`` overrides the asserted-identity subject header for this mint
        (see :meth:`_identity_header`). Download passes the end user so an
        auth-bound token's ``claims.Subject`` binds to that user; upload/stage/
        finalize pass nothing (→ the dc service id).
        """
        body: dict = {"op": op, "ref": ref}
        if content_type:
            body["content_type"] = content_type
        if max_size is not None:
            body["max_size"] = max_size
        if content_hash:
            body["content_hash"] = content_hash
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        if require_auth:
            # Auth-bind the capability. Do NOT send match_mode: the edge defaults
            # mm=exact when require_auth is true (only the intended user), which
            # is exactly what we want for the browser-facing download URL.
            body["require_auth"] = True

        async with self._http() as http:
            resp = await http.post(
                f"{self._edge_url}/capabilities",
                json=body,
                headers=self._identity_header(subject),
            )
            self._raise_for_status(resp, ref)
            return resp.json()

    def _absolute(self, capability_url: str) -> str:
        """Resolve an edge-relative ``capability_url`` to an absolute URL."""
        if capability_url.startswith("http://") or capability_url.startswith("https://"):
            return capability_url
        return f"{self._edge_url}{capability_url}"

    def _public_absolute(self, capability_url: str) -> str:
        """Resolve a ``capability_url`` against the browser-facing public base.

        Mirrors :meth:`_absolute` but prefixes the public download base
        (``public_url``) instead of the internal edge, for the one URL a browser
        GETs directly.
        """
        return f"{self._public_url}{capability_url}"

    @staticmethod
    def _raise_for_status(resp: httpx.Response, ref: str) -> None:
        """Map edge HTTP errors to backend-native exceptions.

        404 -> :class:`BlobgwNotFound` (so callers behave like the S3 backend's
        missing-object path). Other 4xx/5xx -> ``httpx.HTTPStatusError`` after a
        contextualized log, so raw upstream detail is not leaked verbatim to the
        caller beyond the status.
        """
        if resp.status_code == 404:
            raise BlobgwNotFound(ref)
        if resp.status_code >= 400:
            detail = ""
            try:
                payload = resp.json()
                if isinstance(payload, dict):
                    detail = str(payload.get("error") or payload.get("detail") or "")
            except Exception:
                detail = ""
            logger.warning(
                "blobgw edge error: status=%s ref=%s detail=%s",
                resp.status_code, ref, detail,
            )
            raise httpx.HTTPStatusError(
                f"blobgw edge returned {resp.status_code} for ref={ref}",
                request=resp.request,
                response=resp,
            )

    @staticmethod
    def _parse_expires_at(value: Optional[str], fallback_ttl_s: int) -> datetime:
        """Parse an RFC3339 ``expires_at`` string to an aware datetime.

        Falls back to ``now + fallback_ttl_s`` when the edge omits or returns an
        unparseable value, so callers always get a concrete expiry.
        """
        if value:
            v = value.strip()
            if v.endswith("Z"):
                v = v[:-1] + "+00:00"
            # fromisoformat rejects >6 fractional digits; truncate if present.
            if "." in v:
                head, frac = v.split(".", 1)
                digits, rest = "", ""
                for i, ch in enumerate(frac):
                    if ch.isdigit():
                        digits += ch
                    else:
                        rest = frac[i:]
                        break
                v = f"{head}.{digits[:6]}{rest}"
            try:
                dt = datetime.fromisoformat(v)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                pass
        return datetime.now(timezone.utc) + timedelta(seconds=fallback_ttl_s)

    # -- presigned URL surface (edge) ---------------------------------------

    async def generate_upload_url(
        self,
        key: str,
        content_type: Optional[str] = None,
        ttl_seconds: int = _DEFAULT_UPLOAD_TTL_S,
        method: str = "POST",
    ) -> dict:
        """Generate a presigned upload URL resolved server-side via the edge.

        Resolves the two edge steps server-side so the browser keeps its usual
        single-PUT UX:

          1. Mint a ``STAGE`` capability at ``POST {edge}/capabilities``.
          2. ``POST {edge}/staged/{ref}?cap={token}`` -> presigned S3
             ``upload_url`` the browser PUTs raw bytes to.

        ``method`` is accepted for interface parity with the S3 backend but the
        blobgw edge always returns a presigned PUT (raw-body upload); the return
        always advertises ``method="PUT"`` accordingly.
        """
        ref = self._full_key(key)

        cap = await self._mint_capability(
            "STAGE",
            ref,
            content_type=content_type,
            ttl_seconds=ttl_seconds,
        )
        staged_url = self._absolute(cap["capability_url"])

        async with self._http() as http:
            resp = await http.post(staged_url)
            self._raise_for_status(resp, ref)
            staged = resp.json()

        expires_at = self._parse_expires_at(
            staged.get("expires_at") or cap.get("expires_at"), ttl_seconds
        )
        headers: dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = content_type

        return {
            "method": "PUT",
            "url": staged["upload_url"],
            "fields": {},
            "headers": headers,
            "expires_at": expires_at,
        }

    async def generate_download_url(
        self,
        key: str,
        ttl_seconds: int = _DEFAULT_DOWNLOAD_TTL_S,
        subject: Optional[str] = None,
    ) -> tuple[str, dict[str, str], datetime]:
        """Generate a download URL via an edge ``GET`` capability.

        Returns ``(url, headers, expires_at)``. The capability is minted
        server-side against the internal edge, but the returned URL is built
        from the browser-facing public base (``public_url``) and is auth-bound
        (``require_auth``) by default, so only the intended user can redeem the
        external URL the browser GETs directly.

        When ``subject`` is supplied it becomes the asserted mint subject, so an
        auth-bound token's ``claims.Subject`` equals the end user — that same
        user's ``X-Scitrera-User`` then matches when they GET the URL. Absent a
        subject the mint falls back to the dc service id (back-compat).
        """
        ref = self._full_key(key)
        cap = await self._mint_capability(
            "GET", ref, ttl_seconds=ttl_seconds,
            require_auth=self._download_require_auth, subject=subject,
        )
        url = self._public_absolute(cap["capability_url"])
        expires_at = self._parse_expires_at(cap.get("expires_at"), ttl_seconds)
        return url, {}, expires_at

    async def generate_fetch_url(
        self,
        key: str,
        ttl_seconds: int = _DEFAULT_DOWNLOAD_TTL_S,
    ) -> tuple[str, dict[str, str], datetime]:
        """Generate an INTERNAL, server-side GET URL via an edge capability.

        Unlike :meth:`generate_download_url` (browser-facing: auth-bound and built
        from the PUBLIC ``storage2`` base), this is for IN-CLUSTER server-side
        fetchers — MemoryLayer workers and the sahara harness. The capability is
        minted **bearer** (``require_auth=False``) and the URL is built from the
        **internal** edge base (``_absolute`` → ``blobgw-edge.storage.svc``), which
        the caller reaches directly over the cluster network.

        Why not the public/auth-bound URL: the external ingress in front of
        ``storage2`` strips inbound identity headers and re-stamps from an
        authenticated session; a server-side fetcher has no session, so the edge
        would see no principal and 403. The internal edge is network-isolated (only
        in-cluster clients can reach it), so a short-lived bearer capability is the
        intended posture. (A future ACL will gate who may mint a fetch URL.)
        """
        ref = self._full_key(key)
        cap = await self._mint_capability("GET", ref, ttl_seconds=ttl_seconds, require_auth=False)
        url = self._absolute(cap["capability_url"])
        expires_at = self._parse_expires_at(cap.get("expires_at"), ttl_seconds)
        return url, {}, expires_at

    # -- finalize (edge) ----------------------------------------------------

    async def finalize_blob(
        self,
        key: str,
        *,
        content_type: Optional[str] = None,
    ) -> dict:
        """Finalize a staged upload via the edge and return its metadata.

        Mints a ``FINALIZE`` capability then POSTs ``/finalize/{ref}``. Returns
        the edge finalize payload, which includes ``size``, ``content_hash``,
        ``content_type``, ``domain`` and ``created_at``. This is what lets the
        app.py finalize path skip the ``head_object`` + ``get_object``
        round-trip the S3 backend needs.
        """
        ref = self._full_key(key)
        cap = await self._mint_capability(
            "FINALIZE", ref, content_type=content_type
        )
        final_url = self._absolute(cap["capability_url"])
        async with self._http() as http:
            resp = await http.post(final_url)
            self._raise_for_status(resp, ref)
            return resp.json()

    # -- direct server-side I/O (internal gateway) --------------------------

    def _object_url(self, ref: str) -> str:
        """Internal gateway data-plane object URL for a ref."""
        from urllib.parse import quote

        return f"{self._internal_url}/v1/objects/{quote(ref, safe='/')}"

    async def put_object(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> int:
        """Upload bytes directly to the internal gateway. Returns byte count."""
        ref = self._full_key(key)
        if self._client is not None:
            await asyncio.to_thread(self._client.put, ref, data, content_type)
            return len(data)

        headers = {"Content-Type": content_type, **self._domain_header()}
        async with self._http() as http:
            resp = await http.put(self._object_url(ref), content=data, headers=headers)
            self._raise_for_status(resp, ref)
        return len(data)

    async def get_object(self, key: str) -> bytes:
        """Download bytes directly from the internal gateway.

        Raises :class:`BlobgwNotFound` (a ``FileNotFoundError``) when the ref is
        absent, matching the S3 backend's missing-object behavior.
        """
        ref = self._full_key(key)
        if self._client is not None:
            try:
                return await asyncio.to_thread(self._client.get, ref)
            except self._not_found:
                raise BlobgwNotFound(ref) from None

        async with self._http() as http:
            resp = await http.get(self._object_url(ref), headers=self._domain_header())
            self._raise_for_status(resp, ref)
            return resp.content

    async def delete_object(self, key: str) -> None:
        """Delete an object from the internal gateway. Absent ref is a no-op."""
        ref = self._full_key(key)
        if self._client is not None:
            await asyncio.to_thread(self._client.delete, ref)
            return

        async with self._http() as http:
            resp = await http.delete(self._object_url(ref), headers=self._domain_header())
            # A missing object on delete is a no-op, matching S3 semantics.
            if resp.status_code == 404:
                return
            self._raise_for_status(resp, ref)

    async def head_object(self, key: str) -> Optional[dict]:
        """Object metadata without the body, in the S3-shaped dict callers expect.

        Returns a dict carrying at least ``ContentLength`` / ``ContentType``
        (plus ``ContentHash`` when available), or ``None`` when the object does
        not exist — mirroring the S3 backend's ``head_object`` contract.
        """
        ref = self._full_key(key)
        if self._client is not None:
            try:
                info = await asyncio.to_thread(self._client.head, ref)
            except self._not_found:
                return None
            except Exception:
                return None
            return {
                "ContentLength": info.size,
                "ContentType": info.content_type or None,
                "ContentHash": info.content_hash or None,
            }

        async with self._http() as http:
            try:
                resp = await http.head(self._object_url(ref), headers=self._domain_header())
            except httpx.HTTPError:
                return None
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            return None
        length = resp.headers.get("Content-Length")
        etag = (resp.headers.get("ETag") or "").strip('"')
        return {
            "ContentLength": int(length) if length is not None else None,
            "ContentType": resp.headers.get("Content-Type") or None,
            "ContentHash": etag or None,
        }
