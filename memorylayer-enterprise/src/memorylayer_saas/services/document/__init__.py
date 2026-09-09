"""Enterprise document ingestion service package.

Phase 3 of the Aether convergence relocated the embed-server client plugin
base + extension key to the OSS core (``memorylayer_server.services.document``).
This module is now a compatibility shim: it re-exports the OSS symbols under
the legacy enterprise names so existing enterprise imports keep working,
plus declares enterprise-specific extension points (document ingestion,
blob storage) that remain enterprise-only.

Extension points:
- EXT_DOCUMENT_INGESTION_SERVICE: Core ingestion pipeline orchestrator (enterprise).
- EXT_EMBED_SERVER_CLIENT: HTTP client for the embed server (now OSS, re-exported).
- EXT_BLOB_STORAGE_SERVICE: Storage-agnostic blob service (enterprise).
"""
from logging import Logger

from memorylayer_server.config import (  # noqa: F401
    DEFAULT_MEMORYLAYER_EMBED_SERVER_SERVICE,
    MEMORYLAYER_EMBED_SERVER_SERVICE,
)

# Re-export OSS embed-server client surface so legacy enterprise imports
# (``from memorylayer_saas.services.document import EmbedServerClientPluginBase``)
# keep resolving without code changes elsewhere.
from memorylayer_server.services._constants import EXT_EMBED_SERVER_CLIENT  # noqa: F401
from memorylayer_server.services.document import (  # noqa: F401
    EmbedServerClientPluginBase,
    get_embed_server_client,
)
from scitrera_app_framework import Variables, get_extension
from scitrera_app_framework.api import Plugin, enabled_option_pattern

from ...config import (
    DEFAULT_MEMORYLAYER_DOCUMENT_INGESTION_SERVICE,
    MEMORYLAYER_DOCUMENT_INGESTION_SERVICE,
)

# Enterprise-only extension points
EXT_DOCUMENT_INGESTION_SERVICE = "memorylayer-enterprise-document-ingestion-service"
EXT_BLOB_STORAGE_SERVICE = "memorylayer-enterprise-blob-storage-service"

# Config keys for blob storage (enterprise-only)
MEMORYLAYER_BLOB_STORAGE_SERVICE = "MEMORYLAYER_BLOB_STORAGE_SERVICE"
DEFAULT_MEMORYLAYER_BLOB_STORAGE_SERVICE = "default"


# === Enterprise plugin base classes ===


# noinspection PyAbstractClass
class DocumentIngestionPluginBase(Plugin):
    """Base plugin for the enterprise document ingestion service."""

    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_DOCUMENT_INGESTION_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_DOCUMENT_INGESTION_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(
            self, v, MEMORYLAYER_DOCUMENT_INGESTION_SERVICE, self_attr="PROVIDER_NAME"
        )

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(
            MEMORYLAYER_DOCUMENT_INGESTION_SERVICE,
            DEFAULT_MEMORYLAYER_DOCUMENT_INGESTION_SERVICE,
        )

    def get_dependencies(self, v: Variables):
        return (EXT_EMBED_SERVER_CLIENT, EXT_BLOB_STORAGE_SERVICE)


# noinspection PyAbstractClass
class BlobStoragePluginBase(Plugin):
    """Base plugin for the enterprise blob storage service."""

    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_BLOB_STORAGE_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_BLOB_STORAGE_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(
            self, v, MEMORYLAYER_BLOB_STORAGE_SERVICE, self_attr="PROVIDER_NAME"
        )

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(
            MEMORYLAYER_BLOB_STORAGE_SERVICE,
            DEFAULT_MEMORYLAYER_BLOB_STORAGE_SERVICE,
        )


# === tokenary client plugin (enterprise-only) ===


class TokenaryClientPlugin(EmbedServerClientPluginBase):
    """Embed-server client plugin that targets tokenary instances.

    Selected via ``MEMORYLAYER_EMBED_SERVER_SERVICE=tokenary`` (matched against
    ``PROVIDER_NAME``). Because both the OSS embedding and reranker providers
    resolve their client via the ``EXT_EMBED_SERVER_CLIENT`` extension, swapping
    this one client reroutes single-vec, multi-vec, image, and MaxSim score to
    tokenary — no OSS provider plugins needed.

    Reads the proprietary per-concern URL config (each defaulting to
    ``MEMORYLAYER_EMBED_SERVER_URL``) plus the shared timeout/transport knobs,
    mirroring the default ``EmbedServerClientPlugin.initialize`` pattern.
    """

    PROVIDER_NAME = "tokenary"

    async def async_ready(self, v: Variables, logger: Logger, value: object | None) -> None:
        if value is None:
            return
        try:
            await value.connect()
        except Exception as e:  # noqa: BLE001 — connect failures shouldn't crash boot
            logger.warning("TokenaryClient.connect() failed at startup: %s", e)

    def initialize(self, v: Variables, logger: Logger):
        from memorylayer_server.config import (
            DEFAULT_EMBEDDING_DIMENSIONS_EMBED_SERVER,
            MEMORYLAYER_EMBEDDING_DIMENSIONS,
        )
        from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION
        from memorylayer_server.services.document.embed_client import (
            TRANSPORT_AETHER,
            TRANSPORT_HTTP,
        )

        from ...config import (
            DEFAULT_MEMORYLAYER_EMBED_AETHER_TARGET,
            DEFAULT_MEMORYLAYER_EMBED_SERVER_TIMEOUT,
            DEFAULT_MEMORYLAYER_EMBED_SERVER_URL,
            DEFAULT_MEMORYLAYER_EMBED_TRANSPORT,
            DEFAULT_MEMORYLAYER_TOKENARY_MULTIVEC_URL,
            DEFAULT_MEMORYLAYER_TOKENARY_SCORE_URL,
            DEFAULT_MEMORYLAYER_TOKENARY_TEXTVEC_URL,
            DEFAULT_MEMORYLAYER_TOKENARY_VISUALTOK_URL,
            MEMORYLAYER_EMBED_AETHER_TARGET,
            MEMORYLAYER_EMBED_SERVER_TIMEOUT,
            MEMORYLAYER_EMBED_SERVER_URL,
            MEMORYLAYER_EMBED_TRANSPORT,
            MEMORYLAYER_TOKENARY_MULTIVEC_URL,
            MEMORYLAYER_TOKENARY_SCORE_URL,
            MEMORYLAYER_TOKENARY_TEXTVEC_URL,
            MEMORYLAYER_TOKENARY_VISUALTOK_URL,
        )
        from .tokenary_client import TokenaryClient

        base_url = v.environ(MEMORYLAYER_EMBED_SERVER_URL, default=DEFAULT_MEMORYLAYER_EMBED_SERVER_URL)
        timeout = float(
            v.environ(MEMORYLAYER_EMBED_SERVER_TIMEOUT, default=str(DEFAULT_MEMORYLAYER_EMBED_SERVER_TIMEOUT))
        )
        transport = v.environ(MEMORYLAYER_EMBED_TRANSPORT, default=DEFAULT_MEMORYLAYER_EMBED_TRANSPORT).lower()

        aether_connection = None
        aether_target = DEFAULT_MEMORYLAYER_EMBED_AETHER_TARGET
        if transport == TRANSPORT_AETHER:
            aether_connection = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
            if aether_connection is None:
                raise RuntimeError(
                    "MEMORYLAYER_EMBED_TRANSPORT=aether requires the "
                    "AetherServiceConnection extension (EXT_AETHER_SERVICE_CONNECTION) "
                    "to be initialised first; no connection found."
                )
            aether_target = v.environ(
                MEMORYLAYER_EMBED_AETHER_TARGET, default=DEFAULT_MEMORYLAYER_EMBED_AETHER_TARGET
            )

        textvec_url = v.environ(MEMORYLAYER_TOKENARY_TEXTVEC_URL, default=DEFAULT_MEMORYLAYER_TOKENARY_TEXTVEC_URL)
        multivec_url = v.environ(MEMORYLAYER_TOKENARY_MULTIVEC_URL, default=DEFAULT_MEMORYLAYER_TOKENARY_MULTIVEC_URL)
        score_url = v.environ(MEMORYLAYER_TOKENARY_SCORE_URL, default=DEFAULT_MEMORYLAYER_TOKENARY_SCORE_URL)
        visualtok_url = v.environ(MEMORYLAYER_TOKENARY_VISUALTOK_URL, default=DEFAULT_MEMORYLAYER_TOKENARY_VISUALTOK_URL)

        # Pass tokenary the OpenAI `dimensions` so single-vec matches what the
        # embed-server returned (tokenary defaults to 2048).
        embedding_dimensions = v.environ(
            MEMORYLAYER_EMBEDDING_DIMENSIONS,
            default=DEFAULT_EMBEDDING_DIMENSIONS_EMBED_SERVER,
            type_fn=int,
        )

        logger.info(
            "Initializing tokenary client: transport=%s base=%s textvec=%s multivec=%s "
            "score=%s visualtok=%s dims=%s timeout=%.0fs",
            transport,
            base_url if transport == TRANSPORT_HTTP else "<aether>",
            textvec_url or "<base>",
            multivec_url or "<base>",
            score_url or "<base>",
            visualtok_url or "<base>",
            embedding_dimensions,
            timeout,
        )
        return TokenaryClient(
            base_url=base_url,
            timeout=timeout,
            logger=logger,
            transport=transport,
            aether_connection=aether_connection,
            aether_target=aether_target,
            textvec_url=textvec_url,
            multivec_url=multivec_url,
            score_url=score_url,
            visualtok_url=visualtok_url,
            embedding_dimensions=embedding_dimensions,
        )


# === Convenience getters ===


def get_document_ingestion_service(v: Variables = None):
    """Get the document ingestion service instance."""
    return get_extension(EXT_DOCUMENT_INGESTION_SERVICE, v)


def get_blob_storage_service(v: Variables = None):
    """Get the blob storage service instance."""
    return get_extension(EXT_BLOB_STORAGE_SERVICE, v)


__all__ = (
    # Enterprise extension points
    "EXT_DOCUMENT_INGESTION_SERVICE",
    "EXT_BLOB_STORAGE_SERVICE",
    # Re-exported OSS surface
    "EXT_EMBED_SERVER_CLIENT",
    "EmbedServerClientPluginBase",
    "get_embed_server_client",
    "MEMORYLAYER_EMBED_SERVER_SERVICE",
    "DEFAULT_MEMORYLAYER_EMBED_SERVER_SERVICE",
    # Enterprise plugin base classes
    "DocumentIngestionPluginBase",
    "BlobStoragePluginBase",
    # tokenary client plugin
    "TokenaryClientPlugin",
    # Enterprise config keys
    "MEMORYLAYER_BLOB_STORAGE_SERVICE",
    "DEFAULT_MEMORYLAYER_BLOB_STORAGE_SERVICE",
    # Enterprise convenience getters
    "get_document_ingestion_service",
    "get_blob_storage_service",
)
