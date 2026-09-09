"""Backward-compatibility re-export — implementation moved to memorylayer_server.

Phase 1 (Aether convergence): the OSS module was renamed from
``aether_agent`` to ``aether_service`` and the class renamed from
``AetherAgentService`` to ``AetherServiceConnection``.  This shim
re-exports the new module's public surface (under both new and legacy
names) so existing enterprise imports keep working during the deprecation
window.

Imports through this module trigger a ``DeprecationWarning`` (forwarded
from the OSS shim) — callers should migrate to
``memorylayer_server.services.aether_service``.
"""
from memorylayer_server.services.aether_agent import *  # noqa: F401,F403
