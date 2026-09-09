# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Per-tenant mlfs metadata storage stats.

blobgw reports the shared pack substrate (physical / deduped-logical /
compression — which cover BOTH mlfs files and blobgw objects), but it cannot see
the mlfs *reference* side: the mlfs slice/node metadata lives in a per-tenant DB
``<domain>_meta`` on the SAME ml-postgres instance data-connectors already uses
(its own ``dataconnectors`` DB). We read that meta DB with DC's OWN credentials —
least-privilege: the ``dataconnectors`` role is granted SELECT on ``slice_ref`` +
``node`` in ``<domain>_meta`` (provisioned alongside the mlfs domain creds). This
is the mlfs half of the tenant storage rollup; StorageUsageService combines it
with blobgw's substrate numbers into the correct apparent + dedup figures.

Best-effort: any failure (DB unreachable, missing grant, absent meta DB) returns
None so the caller degrades to substrate-only numbers rather than misleading ones.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from data_connectors.db.engine import DC_POSTGRESQL_URL

logger = logging.getLogger(__name__)

# The mlfs meta DB name is "<domain><suffix>" (convention: "botwinick_meta").
DC_MLFS_META_DB_SUFFIX = "DC_MLFS_META_DB_SUFFIX"
_DEFAULT_META_SUFFIX = "_meta"
# Master switch; on by default when a DC Postgres URL is configured.
DC_MLFS_STATS_ENABLED = "DC_MLFS_STATS_ENABLED"


def _mlfs_enabled() -> bool:
    return os.environ.get(DC_MLFS_STATS_ENABLED, "true").lower() not in ("0", "false", "no")


def meta_dsn(domain: str) -> Optional[str]:
    """Derive the mlfs meta DB connection string from DC's own Postgres URL.

    Reuses DC's host + credentials (same ml-postgres instance) and only swaps the
    database name to ``<domain>_meta`` — so DC authenticates as its own
    least-privilege ``dataconnectors`` role, not the mlfs owner. Returns a plain
    ``postgresql://`` DSN (asyncpg-compatible), or None when unconfigured.
    """
    raw = os.environ.get(DC_POSTGRESQL_URL)
    if not raw or not domain:
        return None
    suffix = os.environ.get(DC_MLFS_META_DB_SUFFIX, _DEFAULT_META_SUFFIX)
    try:
        # Delayed import: keep SQLAlchemy's URL parser off the module import path.
        from sqlalchemy.engine import make_url  # noqa: PLC0415

        u = make_url(raw)
        # Force a plain scheme (asyncpg.connect wants postgresql://, not +asyncpg)
        # and point at the tenant's mlfs meta DB.
        u = u.set(drivername="postgresql", database=f"{domain}{suffix}")
        return u.render_as_string(hide_password=False)
    except Exception as e:  # noqa: BLE001 — never fail the caller on a parse issue
        logger.warning("mlfs stats: cannot derive meta DSN for domain=%s: %s", domain, e)
        return None


async def get_mlfs_stats(domain: str) -> Optional[dict[str, Any]]:
    """Read the tenant's mlfs meta rollup, or None if disabled/unavailable.

    Returns ``{slice_source_bytes, slice_live_bytes, file_logical_bytes}``:
      - slice_source = SUM(slice_ref.size)  — pre-dedup referenced bytes (apparent)
      - slice_live   = SUM(slice_ref.slen)  — live-referenced bytes
      - file_logical = SUM(node.length)     — apparent total of the files
    """
    if not _mlfs_enabled():
        return None
    dsn = meta_dsn(domain)
    if not dsn:
        return None
    try:
        import asyncpg  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        logger.info("mlfs stats: asyncpg unavailable; skipping mlfs metadata")
        return None

    conn = None
    try:
        conn = await asyncpg.connect(dsn=dsn, timeout=10.0)
        slice_row = await conn.fetchrow(
            "SELECT COALESCE(SUM(size), 0)::bigint AS src, "
            "COALESCE(SUM(slen), 0)::bigint AS live FROM slice_ref"
        )
        node_row = await conn.fetchrow(
            "SELECT COALESCE(SUM(length), 0)::bigint AS logical FROM node"
        )
        return {
            "slice_source_bytes": int(slice_row["src"]),
            "slice_live_bytes": int(slice_row["live"]),
            "file_logical_bytes": int(node_row["logical"]),
        }
    except Exception as e:  # noqa: BLE001 — degrade to substrate-only numbers
        logger.info("mlfs stats unavailable for domain=%s: %s", domain, e)
        return None
    finally:
        if conn is not None:
            try:
                await conn.close()
            except Exception:  # noqa: BLE001
                pass
