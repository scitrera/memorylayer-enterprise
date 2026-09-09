# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Enterprise representation backend (LLM-derived beliefs layer).

Re-exports the enterprise service + plugin. The plugin is auto-discovered by the
enterprise ``register_package_plugins(services.__package__, ..., recursive=True)``
scan (see ``memorylayer_saas/dependencies.py``), so simply living in this package
registers it. Enable with ``MEMORYLAYER_REPRESENTATION_PROVIDER=enterprise``
(and the surface's master flag ``MEMORYLAYER_REPRESENTATION_ENABLED``).

The enterprise backend extends the OSS deterministic (observer, subject)
assembly with ONE additive, fail-safe layer — ``Representation.derived_beliefs``
populated via an LLM over the observer's leakage-safe scoped observations and
reconciled against known contradictions. OSS stays deterministic
(``derived_beliefs=[]``) as the parity reference; the leakage-0 property is
preserved (the LLM never sees other-observer content).
"""

from .default import (
    EnterpriseRepresentationService,
    EnterpriseRepresentationServicePlugin,
)

__all__ = (
    "EnterpriseRepresentationService",
    "EnterpriseRepresentationServicePlugin",
)
