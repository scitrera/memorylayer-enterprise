# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Task handler for ``memorylayer-task.representation_consolidation`` pool tasks.

The P3 consolidation layer: a leased/coalesced background task that DERIVES +
PERSISTS a maintained per-(observer, subject) representation (profile + derived
beliefs) so ``EnterpriseRepresentationService.get_representation`` can serve a
maintained profile cheaply instead of deriving on every call.

Coalesce-join + lease (mirrors the pre-join ``kb_update`` hand-rolled pattern)
------------------------------------------------------------------------------
Belief derivation is expensive (an LLM call), and a burst of new observations
about one subject would otherwise spawn one derivation per trigger. We coalesce
via a short-TTL per-(workspace, observer, subject) *lease* in the enterprise
cache (``EnterpriseCacheService.acquire_lock`` / ``release_lock``):

  * The lease is acquired atomically (``acquire_lock`` is backed by Aether's
    ``kv_increment_if(ceiling=1)`` lease or Redis SETNX): the first caller wins;
    a concurrent caller sees ``acquired=False`` and COALESCES AWAY (returns) —
    the in-flight holder's single derivation already covers the burst's new work
    (the holder re-reads the latest observations + watermark when it derives).
  * Inside the lease, ``consolidate()`` is watermark-gated: an UNCHANGED subject
    (workspace watermark matches the persisted record) is a near-no-op — the
    LLM is NOT called.
  * The lease is released in ``try/finally`` on success AND failure — no stuck
    leases. A TTL bounds a crashed holder.

Cardinal constraints (mirror the on-demand derivation service):
  * DARK + fail-safe: gated behind ``MEMORYLAYER_REPRESENTATION_ENABLED`` (the
    surface) + ``MEMORYLAYER_REPRESENTATION_CONSOLIDATION_ENABLED`` (this layer,
    default OFF). When OFF the handler is a clean no-op.
  * ANY failure (LLM down, cache unavailable, lease contention, persist error)
    is logged and swallowed — the task never re-raises, and
    ``get_representation`` keeps deriving on-demand (the 0a79fb9 behavior).
  * Leakage-0 + perspective-only-scope are preserved: the persisted profile is
    derived ONLY from the observer's leakage-safe scoped observations (the
    derivation path is the same ``_derive_on_demand`` the service uses).

This handler is purely event/trigger driven (``get_schedule`` returns None — no
recurring sweep yet). A periodic workspace-level sweep is a possible follow-on
(see the task return notes).
"""
from typing import Optional

from scitrera_app_framework import Variables, ext_parse_bool, get_extension, get_logger

from memorylayer_server.config import (
    MEMORYLAYER_REPRESENTATION_ENABLED,
    DEFAULT_MEMORYLAYER_REPRESENTATION_ENABLED,
)
from memorylayer_server.services.tasks.handlers import TaskHandlerPlugin
from memorylayer_server.services.tasks.base import TaskSchedule
from memorylayer_server.services.representation import (
    EXT_REPRESENTATION_SERVICE,
)

from memorylayer_saas.config import (
    MEMORYLAYER_REPRESENTATION_CONSOLIDATION_ENABLED,
    DEFAULT_MEMORYLAYER_REPRESENTATION_CONSOLIDATION_ENABLED,
    MEMORYLAYER_REPRESENTATION_CONSOLIDATION_LEASE_TTL,
    DEFAULT_MEMORYLAYER_REPRESENTATION_CONSOLIDATION_LEASE_TTL,
)
from memorylayer_saas.services.cache import EXT_CACHE_SERVICE
from memorylayer_saas.services.cache.base import EnterpriseCacheService
from memorylayer_saas.services.representation.consolidation_store import lease_key

# Holder id for the lease. ``release_lock`` treats holder_id as advisory (the KV
# verbs can't atomically verify ownership), but we pass a stable id for symmetry
# with the EnterpriseCacheService contract.
_HOLDER_ID = "representation_consolidation"


class RepresentationConsolidationTaskHandler(TaskHandlerPlugin):
    """Task handler for ``representation_consolidation`` — leased + coalesced.

    Claims pool tasks of type ``memorylayer-task.representation_consolidation``
    and derives + persists the maintained representation for a single
    (observer, subject), coalescing concurrent triggers via a per-subject lease.
    """

    def get_task_type(self) -> str:
        return "representation_consolidation"

    async def handle(self, v: Variables, payload: dict) -> None:
        """Execute the consolidation handler.

        Args:
            v: Variables instance.
            payload: Dict with ``workspace_id``, ``observer``, ``subject`` (plus
                     runner-injected reserved keys ``_aether_task_id`` /
                     ``_task_metadata``). ``observer_type`` / ``subject_type`` /
                     ``limit`` are optional.
        """
        logger = get_logger(v, name="RepresentationConsolidationTaskHandler")

        # ---- DARK gate: surface flag + consolidation sub-flag (default OFF) ---
        surface_enabled = v.environ(
            MEMORYLAYER_REPRESENTATION_ENABLED,
            DEFAULT_MEMORYLAYER_REPRESENTATION_ENABLED,
            type_fn=ext_parse_bool,
        )
        consolidation_enabled = v.environ(
            MEMORYLAYER_REPRESENTATION_CONSOLIDATION_ENABLED,
            DEFAULT_MEMORYLAYER_REPRESENTATION_CONSOLIDATION_ENABLED,
            type_fn=ext_parse_bool,
        )
        if not (surface_enabled and consolidation_enabled):
            logger.debug(
                "representation_consolidation: disabled "
                "(surface=%s, consolidation=%s); no-op",
                surface_enabled, consolidation_enabled,
            )
            return  # terminal — dark

        workspace_id = payload.get("workspace_id", "")
        observer = payload.get("observer", "")
        subject = payload.get("subject", "")
        if not workspace_id or not observer or not subject:
            logger.error(
                "representation_consolidation payload missing workspace_id/observer/"
                "subject: %s", payload,
            )
            return  # terminal — malformed payload, don't requeue

        observer_type = payload.get("observer_type")
        subject_type = payload.get("subject_type")
        limit = payload.get("limit", 20)

        # ---- Resolve services (fail-safe) ---------------------------------
        rep_service = _resolve_representation_service(v, logger)
        if rep_service is None or not hasattr(rep_service, "consolidate"):
            logger.warning(
                "representation_consolidation: enterprise representation service "
                "unavailable; skipping (on-demand derivation unaffected)",
            )
            return

        cache = _resolve_cache_service(v, logger)
        lease_ttl = v.environ(
            MEMORYLAYER_REPRESENTATION_CONSOLIDATION_LEASE_TTL,
            DEFAULT_MEMORYLAYER_REPRESENTATION_CONSOLIDATION_LEASE_TTL,
            type_fn=int,
        )

        # ---- Coalesce: acquire the per-(workspace,observer,subject) lease --
        lkey = lease_key(workspace_id, observer, subject)
        acquired = await _try_acquire_lease(cache, lkey, lease_ttl, logger)
        if not acquired:
            # A consolidation is already running for this subject. The in-flight
            # holder re-reads the latest observations + watermark when it derives,
            # so our trigger's new work is already covered — coalesce away.
            logger.info(
                "representation_consolidation: lease held for %s/%s/%s; "
                "coalescing (in-flight run covers this trigger)",
                workspace_id, observer, subject,
            )
            return

        try:
            await rep_service.consolidate(
                workspace_id,
                observer,
                subject,
                observer_type=observer_type,
                subject_type=subject_type,
                limit=limit,
            )
        except Exception:  # noqa: BLE001 - consolidation must NEVER re-raise
            logger.exception(
                "representation_consolidation: derive/persist failed for %s/%s/%s; "
                "get_representation will derive on-demand",
                workspace_id, observer, subject,
            )
            # terminal — fall through to finally (release lease), don't re-raise.
        finally:
            await _release_lease(cache, lkey, logger)

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        # Event/trigger driven only — no recurring sweep (see module docstring).
        return None


# ---------------------------------------------------------------------------
# Internal helpers — service resolution + lease (all best-effort / fail-safe)
# ---------------------------------------------------------------------------

def _resolve_representation_service(v: Variables, logger):
    """Resolve the representation service, or None if unavailable."""
    try:
        return get_extension(EXT_REPRESENTATION_SERVICE, v)
    except Exception:  # noqa: BLE001 - defensive
        logger.debug(
            "representation_consolidation: could not resolve representation service",
            exc_info=True,
        )
        return None


def _resolve_cache_service(v: Variables, logger):
    """Resolve the enterprise cache service, or None (lease degrades open)."""
    try:
        cache = get_extension(EXT_CACHE_SERVICE, v)
    except Exception:  # noqa: BLE001 - defensive
        logger.debug(
            "representation_consolidation: could not resolve cache service",
            exc_info=True,
        )
        return None
    return cache


async def _try_acquire_lease(cache, lock_key: str, ttl: int, logger) -> bool:
    """Acquire the per-subject consolidation lease.

    Returns True if acquired (this caller runs the derivation), False if held by
    a concurrent run (this caller coalesces away). When the cache is unavailable
    or lacks the lock primitive, we fail-OPEN (acquire) so consolidation still
    runs — the duplicate-run cost is bounded by the watermark no-op gate inside
    ``consolidate()`` (an unchanged subject re-derives nothing).
    """
    if cache is None or not isinstance(cache, EnterpriseCacheService):
        return True
    try:
        return await cache.acquire_lock(lock_key, _HOLDER_ID, ttl=ttl)
    except Exception:  # noqa: BLE001 - lease errors fail-open (don't drop work)
        logger.warning(
            "representation_consolidation: lease acquire failed for %s, "
            "proceeding (best-effort)", lock_key, exc_info=True,
        )
        return True


async def _release_lease(cache, lock_key: str, logger) -> None:
    """Release the per-subject consolidation lease (best-effort)."""
    if cache is None or not isinstance(cache, EnterpriseCacheService):
        return
    try:
        await cache.release_lock(lock_key, _HOLDER_ID)
    except Exception:  # noqa: BLE001 - release is best-effort; TTL bounds a leak
        logger.warning(
            "representation_consolidation: lease release failed for %s "
            "(best-effort; TTL will reclaim)", lock_key, exc_info=True,
        )
