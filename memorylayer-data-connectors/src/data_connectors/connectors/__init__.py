"""Connector implementations.

Each connector implements the following contract:
- ``load()`` — one-time initialization / credential validation.
- ``poll()`` — discover new or changed entries since the last checkpoint.
- ``get_content_url(vfs_ref)`` — return a fetch URL for a specific entry.
- ``get_metadata(vfs_ref)`` — return metadata for a specific entry.

Connectors self-register at module import time via
``ConnectorRegistry.register()``.  The sync engine and tests use
``ConnectorRegistry.get()`` to look up a connector class by name.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class Connector(Protocol):
    """Protocol that all connectors implement."""

    async def load(self) -> None:
        """Initialize the connector (validate credentials, etc.)."""
        ...

    async def poll(self) -> list[dict[str, Any]]:
        """Discover new or changed entries.

        Returns:
            List of dicts with keys: source_path, content_hash,
            content_type (optional), size_bytes (optional),
            blob_key (optional), metadata (optional).
        """
        ...

    async def get_content_url(self, vfs_ref: str) -> Optional[str]:
        """Return a fetch URL for a specific VFS entry, or None."""
        ...

    async def get_metadata(self, vfs_ref: str) -> dict[str, Any]:
        """Return metadata for a specific VFS entry."""
        ...


class ConnectorRegistry:
    """Global registry of available connector implementations.

    Connectors self-register at module import time via
    ``ConnectorRegistry.register(name, cls)``.  The sync engine uses
    ``ConnectorRegistry.get(name)`` to look up a connector class.
    """

    _connectors: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, connector_cls: type) -> None:
        """Register a connector class under *name*."""
        if name in cls._connectors:
            logger.warning("Overwriting existing connector registration: %s", name)
        # Publish the stable source kind on every instance.  The sync engine
        # stamps it into VFS metadata so downstream MemoryLayer ingestion can
        # use the OSS connector -> knowledge_work normalizer.
        connector_cls.connector_type = name
        cls._connectors[name] = connector_cls
        logger.debug("Registered connector: %s -> %s", name, connector_cls.__name__)

    @classmethod
    def get(cls, name: str) -> type | None:
        """Return the connector class registered under *name*, or ``None``."""
        return cls._connectors.get(name)

    @classmethod
    def list_connectors(cls) -> list[str]:
        """Return sorted list of registered connector names."""
        return sorted(cls._connectors.keys())


def import_all_connectors() -> list[str]:
    """Import every connector module so each self-registers in the registry.

    Connectors register at module import time, so a ``POST /v1/providers`` of any
    type followed by ``POST /v1/sync/trigger`` resolves only if the module was
    imported. Calling this once at startup populates the registry for all
    supported types.

    Heavy third-party SDKs (slack_sdk, google-api-python-client, dropbox, etc.)
    are delay-imported *inside* each connector's methods, so importing the
    modules here is cheap and does not require those packages to be installed
    until the connector actually runs.

    Returns:
        The sorted list of registered connector names (for logging/inspection).
    """
    # Delayed imports keep the connector SDKs off the package-load path; each
    # ``import`` triggers that module's ``ConnectorRegistry.register(...)``.
    from data_connectors.connectors import discord  # noqa: F401
    from data_connectors.connectors import dropbox  # noqa: F401
    from data_connectors.connectors import gdrive  # noqa: F401
    from data_connectors.connectors import github  # noqa: F401
    from data_connectors.connectors import local_fs  # noqa: F401
    from data_connectors.connectors import manual_upload  # noqa: F401
    from data_connectors.connectors import s3  # noqa: F401
    from data_connectors.connectors import slack  # noqa: F401
    from data_connectors.connectors import teams  # noqa: F401
    from data_connectors.connectors import web_scraper  # noqa: F401

    return ConnectorRegistry.list_connectors()
