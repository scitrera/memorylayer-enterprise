# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Per-tenant storage-usage rollup, cached, combining blobgw + mlfs metadata.

data-connectors is the single aggregation point for a tenant's storage usage. It
combines two sources into one rollup and caches the result per domain with a TTL:

  1. blobgw ``GET /admin/usage?domain=`` — the shared pack *substrate* numbers
     (physical, deduped-logical, compression, and the blobgw-object apparent
     bytes). blobgw owns the pack tables, so only it can produce these, and they
     already cover BOTH mlfs files and blobgw objects (one pack substrate).
  2. the tenant's mlfs meta DB (``mlfs_stats.get_mlfs_stats``) — the mlfs
     *reference* side blobgw can't see: slice-source (apparent) + file-logical
     bytes. Best-effort; when unavailable the rollup degrades to substrate-only
     numbers (no apparent/dedup_ratio) rather than reporting misleading figures.

This is deliberately a *shared* cache here rather than sampling each mlfs mount —
a tenant may have many mounts, but one cache entry per domain covers them all.
See ``.slop/tenant-storage-usage-spec.md``.

``force`` bypasses the TTL age check and re-fetches (e.g. right after a big
ingest/GC). Both sources are read fresh on a cache miss; the cache lives here.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, Optional

import httpx

logger = logging.getLogger(__name__)

# Default cache TTL; storage usage does not move second-to-second, so a coarse
# window keeps blobgw load bounded regardless of how often the UI is opened.
DEFAULT_TTL_S = 300

# Async callable (domain) -> mlfs rollup dict | None. Injectable for tests.
MlfsStatsFn = Callable[[str], Awaitable[Optional[dict[str, Any]]]]


def _combine(bg: dict[str, Any], mlfs: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Merge blobgw substrate numbers with mlfs reference stats.

    Substrate fields (physical / deduped / compression / object-apparent) always
    pass through from blobgw. apparent_bytes + dedup_ratio + file_logical_bytes
    are only added when mlfs metadata is present, because a true apparent figure
    requires the mlfs slice references blobgw cannot see; without them we'd report
    apparent < deduped (structurally impossible) and a nonsensical dedup ratio.
    """
    deduped = int(bg.get("logical_deduped_bytes", 0) or 0)
    object_apparent = int(bg.get("object_apparent_bytes", 0) or 0)
    out: dict[str, Any] = {
        "domain": bg.get("domain"),
        "physical_bytes": int(bg.get("physical_bytes", 0) or 0),
        "logical_deduped_bytes": deduped,
        "compression_ratio": float(bg.get("compression_ratio", 0.0) or 0.0),
        "object_apparent_bytes": object_apparent,
        "computed_at": bg.get("computed_at"),
        "mlfs": False,
    }
    if mlfs is not None:
        slice_source = int(mlfs.get("slice_source_bytes", 0) or 0)
        apparent = object_apparent + slice_source
        out["mlfs_apparent_bytes"] = slice_source
        out["file_logical_bytes"] = int(mlfs.get("file_logical_bytes", 0) or 0)
        out["apparent_bytes"] = apparent
        out["dedup_ratio"] = (apparent / deduped) if deduped > 0 else 0.0
        out["mlfs"] = True
    return out


class StorageUsageService:
    """Lazy, per-domain TTL cache over blobgw's ``/admin/usage`` endpoint."""

    def __init__(
        self,
        blobgw_url: Optional[str],
        ttl_s: int = DEFAULT_TTL_S,
        timeout: float = 30.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        mlfs_stats_fn: Optional[MlfsStatsFn] = None,
    ) -> None:
        self._blobgw_url = (blobgw_url or "").rstrip("/")
        self._ttl_s = ttl_s
        self._timeout = timeout
        # Optional injected transport (tests use httpx.MockTransport).
        self._transport = transport
        # Async mlfs stats source; defaults to the real meta-DB reader. Tests
        # inject a fake to exercise the combine without a live Postgres.
        self._mlfs_stats_fn = mlfs_stats_fn
        # domain -> (payload, monotonic_ts_at_fetch)
        self._cache: dict[str, tuple[dict[str, Any], float]] = {}

    def _http(self) -> httpx.AsyncClient:
        if self._transport is not None:
            return httpx.AsyncClient(timeout=self._timeout, transport=self._transport)
        return httpx.AsyncClient(timeout=self._timeout)

    async def get(self, domain: str, force: bool = False) -> dict[str, Any]:
        """Return the domain's usage rollup, served from cache when fresh.

        Raises ValueError for an empty domain; propagates httpx errors from the
        blobgw call so the route can map them to a gateway error.
        """
        if not domain:
            raise ValueError("domain is required")
        now = time.monotonic()
        if not force:
            cached = self._cache.get(domain)
            if cached is not None and (now - cached[1]) < self._ttl_s:
                # ``computed_at`` in the payload is blobgw's real compute time, so
                # the UI can still show "as of …" honestly even when cached.
                return {**cached[0], "cached": True}
        payload = await self._fetch(domain)
        self._cache[domain] = (payload, now)
        return {**payload, "cached": False}

    async def _fetch(self, domain: str) -> dict[str, Any]:
        if not self._blobgw_url:
            raise RuntimeError("blobgw url not configured (DC_BLOBGW_URL)")
        async with self._http() as http:
            resp = await http.get(
                f"{self._blobgw_url}/admin/usage", params={"domain": domain}
            )
            resp.raise_for_status()
            bg = resp.json()
        # Best-effort mlfs reference side; None => substrate-only rollup.
        mlfs = await self._mlfs(domain)
        return _combine(bg, mlfs)

    async def _mlfs(self, domain: str) -> Optional[dict[str, Any]]:
        """Read the mlfs meta rollup, swallowing any failure (best-effort)."""
        try:
            if self._mlfs_stats_fn is not None:
                return await self._mlfs_stats_fn(domain)
            # Delayed import: the mlfs reader pulls in asyncpg only when used.
            from data_connectors.vfs.mlfs_stats import get_mlfs_stats  # noqa: PLC0415

            return await get_mlfs_stats(domain)
        except Exception as e:  # noqa: BLE001 — never let mlfs break the rollup
            logger.info("mlfs stats fetch failed for domain=%s: %s", domain, e)
            return None
