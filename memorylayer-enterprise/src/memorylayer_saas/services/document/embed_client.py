# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Compatibility shim — embed-server REST client lives in OSS core now.

Phase 3 of the Aether convergence relocated ``EmbedServerClient`` and its
plugin to ``memorylayer_server.services.document.embed_client`` so OSS
deployments can use it without depending on the enterprise package. This
module re-exports the canonical OSS symbols so existing enterprise code
that imports from ``memorylayer_saas.services.document.embed_client``
keeps working without changes.
"""
from memorylayer_server.services.document.embed_client import (  # noqa: F401
    EmbedServerClient,
    EmbedServerClientPlugin,
)

__all__ = ["EmbedServerClient", "EmbedServerClientPlugin"]
