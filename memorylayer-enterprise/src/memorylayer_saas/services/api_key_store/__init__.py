# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Enterprise API key store package.

Ships the Aether-backed :class:`AetherApiKeyStore`, selected when
``MEMORYLAYER_API_KEY_STORE=aether`` (the enterprise default). The plugin is
auto-discovered by ``register_package_plugins(services.__package__, recursive=True)``.
"""

from memorylayer_server.services.api_key_store import (
    EXT_API_KEY_STORE,
    ApiKeyStore,
    get_api_key_store,
)

__all__ = (
    "EXT_API_KEY_STORE",
    "ApiKeyStore",
    "get_api_key_store",
)
