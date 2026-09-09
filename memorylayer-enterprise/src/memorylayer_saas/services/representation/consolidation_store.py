# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Persistence shape for the maintained (observer, subject) representation.

This module is the SINGLE source of truth for how a consolidated representation
is keyed, serialized, and freshness-stamped — shared by the WRITE side (the
``representation_consolidation`` task that derives + persists) and the READ side
(``EnterpriseRepresentationService.get_representation``, which serves the
maintained record when present + fresh).

Persistence-shape decision (FLAGGED for review)
-----------------------------------------------
The maintained representation is stored as a DEDICATED cache record via the
enterprise ``EnterpriseCacheService`` (the same surface that already backs the
consolidation lease), keyed by ``(workspace, observer, subject)`` — NOT as a
``MemorySubtype.PROFILE`` memory.

Rationale (mirrors existing patterns + avoids regression):
  * A PROFILE-subtype *memory* would have to be folded into the entity registry
    as a member to be discoverable by the OSS scoping path, which would (a)
    pollute the deterministic observation set, (b) risk appearing in recall, and
    (c) feed back into the LLM derivation prompt on the next run (a derived
    artifact contaminating its own inputs). That breaks the leakage-0 +
    deterministic-assembly contract the OSS slice guarantees.
  * The cache surface is already the enterprise persistence layer for derived /
    ephemeral artifacts and already carries the lease primitive used here. A
    cache record is read in ONE call (cheap serve), never touches recall, never
    perturbs decay, and is trivially fail-safe (a miss/parse-error -> fall back
    to on-demand derivation).

Freshness is governed by the workspace change watermark (the same dirty-watermark
the KB skip-generate gate uses): the record embeds the watermark at the time it
was derived; the serve path treats the record as FRESH iff the current workspace
watermark EQUALS the stored one (equality is the only safe "unchanged" predicate,
per ``StorageBackend.get_workspace_change_watermark``). A None/uncomputable
watermark is treated as STALE (fail-safe — never serve a possibly-stale record);
``consolidate()`` short-circuits on None and writes nothing (avoids burning an
LLM call to persist a permanently-dead record).

Known limitation — workspace-granularity watermark (MAJOR-2, follow-on):
    The watermark spans the ENTIRE workspace: any new memory for ANY subject
    advances it and invalidates ALL maintained profiles in that workspace. In an
    active workspace the serve-hit rate can trend toward zero while the
    consolidation layer still pays the LLM + cache write cost on every trigger.
    Operators should not expect a high cache-hit rate under sustained multi-subject
    write load with the current implementation.

    The correct fix is a per-subject change signal (a watermark over only the
    subject's member memories) rather than the coarse workspace watermark used
    here. The concurrent C1/C2 work introduced ``compute_entity_watermark`` over
    a subject's member memories — that is exactly the per-subject signal to reuse
    here in a follow-on slice. This layer is intentionally left at workspace
    granularity for now to ship the dark / fail-safe skeleton without taking a
    dependency on the C1/C2 entity-watermark API before it is stabilised.

The persisted blob is the full ``Representation`` (observations + profile +
derived_beliefs + provenance) so the serve path returns the identical shape the
on-demand path would, plus a ``derived=True`` profile marker so callers can tell
a maintained profile from a deterministic one.
"""
from __future__ import annotations

import hashlib

from memorylayer_server.models.representation import Representation

# Cache-key namespace for maintained representation records. Distinct from the
# ``lock:`` lease namespace the cache service prefixes internally.
_RECORD_KEY_PREFIX = "ml:representation:consolidated"
# Lease-key namespace passed to ``acquire_lock`` / ``release_lock``.
_LEASE_KEY_PREFIX = "ml:representation:consolidate"

# Persisted-blob schema version — bump if the on-disk shape changes so an old
# record is treated as a miss (re-derive) rather than mis-parsed.
_BLOB_VERSION = 1


def _identity_digest(observer: str, subject: str) -> str:
    """Stable, collision-resistant digest of the (observer, subject) pair.

    observer / subject are raw user-supplied names; hashing keeps the cache key
    bounded + free of delimiter-injection issues (a name containing the key
    separator could otherwise collide with another pair).
    """
    h = hashlib.sha256()
    h.update(observer.encode("utf-8"))
    h.update(b"\x00")
    h.update(subject.encode("utf-8"))
    return h.hexdigest()[:32]


def record_key(workspace_id: str, observer: str, subject: str) -> str:
    """Cache key for the maintained representation of (observer, subject)."""
    return f"{_RECORD_KEY_PREFIX}:{workspace_id}:{_identity_digest(observer, subject)}"


def lease_key(workspace_id: str, observer: str, subject: str) -> str:
    """Lease key collapsing concurrent consolidation triggers for one subject."""
    return f"{_LEASE_KEY_PREFIX}:{workspace_id}:{_identity_digest(observer, subject)}"


def serialize_record(representation: Representation, watermark, *, limit: int) -> dict:
    """Build the JSON-serializable persisted blob.

    ``watermark`` is the opaque workspace change-watermark token at derivation
    time (a ``tuple[str, str, int, int]`` or None). It is stored as a list so it
    round-trips through JSON; the serve path compares it for EQUALITY only.

    ``limit`` is the observation limit the record was derived at. The serve path
    uses it to detect a caller-limit mismatch (caller wants more observations
    than the record holds) and falls through to on-demand derivation in that
    case. Callers requesting *fewer or equal* observations are served from the
    record; callers requesting *more* get a fresh on-demand derivation.
    """
    return {
        "version": _BLOB_VERSION,
        # store the watermark as a list (JSON has no tuples); compared via
        # watermarks_equal which normalizes both sides to lists.
        "watermark": list(watermark) if watermark is not None else None,
        "limit": limit,
        "representation": representation.model_dump(mode="json"),
    }


def deserialize_record(blob) -> tuple[Representation, list | None, int] | None:
    """Parse a persisted blob back into (Representation, watermark-list, limit).

    Returns None on ANY unusable blob (missing/old version, bad shape, parse
    error) so the caller treats it as a miss and falls back to on-demand
    derivation. NEVER raises.
    """
    if not isinstance(blob, dict):
        return None
    if blob.get("version") != _BLOB_VERSION:
        return None
    rep_data = blob.get("representation")
    if not isinstance(rep_data, dict):
        return None
    try:
        representation = Representation.model_validate(rep_data)
    except Exception:  # noqa: BLE001 - any validation failure -> treat as miss
        return None
    watermark = blob.get("watermark")
    if watermark is not None and not isinstance(watermark, list):
        return None
    # limit defaults to 0 for records written before this field existed; 0
    # causes a mismatch against any real caller limit, so old records are
    # treated as a miss (re-derive) — safe rollout.
    limit = blob.get("limit", 0)
    if not isinstance(limit, int):
        return None
    return representation, watermark, limit


def watermarks_equal(stored, current) -> bool:
    """Equality predicate for the freshness check.

    A record is FRESH iff its stored watermark equals the current workspace
    watermark. Either side being None (uncomputable) means we CANNOT prove the
    subject is unchanged, so we report NOT equal (fail-safe -> re-derive / serve
    on-demand). Both sides are normalized to lists before comparison since the
    stored side round-tripped through JSON as a list.
    """
    if stored is None or current is None:
        return False
    return list(stored) == list(current)
