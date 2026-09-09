# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Enterprise entity-registry backend (embedding-fuzzy resolution tier).

Re-exports the enterprise service + plugin. The plugin is auto-discovered by the
enterprise ``register_package_plugins(services.__package__, ..., recursive=True)``
scan (see ``memorylayer_saas/dependencies.py``), so simply living in this package
registers it. Enable with ``MEMORYLAYER_ENTITY_REGISTRY_PROVIDER=postgresql``
(and the registry's master flag ``MEMORYLAYER_ENTITY_REGISTRY_ENABLED``).

The enterprise tier extends the OSS deterministic exact+alias+create resolution
with a conservative semantic match (see ``postgresql.py``); OSS stays
exact+alias only as the parity reference.
"""

from .postgresql import (
    PostgreSQLEntityRegistryService,
    PostgreSQLEntityRegistryServicePlugin,
)

__all__ = (
    "PostgreSQLEntityRegistryService",
    "PostgreSQLEntityRegistryServicePlugin",
)
