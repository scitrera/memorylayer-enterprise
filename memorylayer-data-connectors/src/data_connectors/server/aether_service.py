"""Aether service registration for data-connectors.

Registers the FastAPI app at ``sv::data-connectors:{specifier}``, fronted by
Aether's ``ProxyHttpTerminator``. The specifier is the ``DC_SERVICE_SPECIFIER``
env var (if set), otherwise the container hostname, otherwise the literal
"default" (see :func:`_resolve_specifier`).

Callers should target the implementation-only address ``sv::data-connectors``
(no specifier) so the gateway routes to any healthy replica. Mirrors
MemoryLayer's existing service registration pattern at
``memorylayer_server/services/aether_service/__init__.py``.

Configuration (environment variables):
    AETHER_GATEWAY_ADDR: Aether gateway gRPC address (default: localhost:50051).
    AETHER_API_KEY: API key for Aether authentication.
    AETHER_AUTH: Set to "none" for local dev.
    DC_SERVICE_SPECIFIER: Service specifier (default: hostname or "default").
"""
from __future__ import annotations

import logging
import os
import socket

logger = logging.getLogger(__name__)

# Service identity
_SERVICE_IMPLEMENTATION = "data-connectors"
_DEFAULT_SPECIFIER = "default"

# Default allow paths for the proxy terminator
_TERMINATOR_ALLOW_PATHS: tuple[str, ...] = (
    "/v1/*",
    "/healthz",
)


def _resolve_specifier() -> str:
    """Resolve the service specifier from env or hostname."""
    specifier = os.environ.get("DC_SERVICE_SPECIFIER")
    if specifier:
        return specifier
    try:
        return socket.gethostname() or _DEFAULT_SPECIFIER
    except Exception:
        return _DEFAULT_SPECIFIER


def service_topic() -> str:
    """Return the exact Aether topic registered by this process."""
    return f"sv::{_SERVICE_IMPLEMENTATION}::{_resolve_specifier()}"


class AetherServiceRegistration:
    """Manages the Aether service connection for data-connectors.

    Connects as ``sv::data-connectors::{specifier}`` and registers
    an in-process ``ProxyHttpTerminator`` to serve the FastAPI app
    via Aether's ``proxy_http_async`` protocol.
    """

    def __init__(self) -> None:
        self._client = None
        self._terminator = None
        self._specifier = _resolve_specifier()

    @property
    def client(self):
        """Return the Aether service client, or None if not connected."""
        return self._client

    async def connect(self, app) -> None:
        """Connect to Aether and register the proxy terminator.

        Args:
            app: FastAPI application to serve via the terminator.
        """
        gateway_addr = os.environ.get("AETHER_GATEWAY_ADDR", "localhost:50051")
        auth_mode = os.environ.get("AETHER_AUTH", "")
        api_key = os.environ.get("AETHER_API_KEY")

        if not api_key:
            key_file = os.environ.get("AETHER_API_KEY_FILE")
            if key_file and os.path.isfile(key_file):
                with open(key_file) as f:
                    api_key = f.read().strip()

        if auth_mode.lower() == "none":
            credentials = None
            logger.warning("AETHER_AUTH=none: running without authentication (local dev only)")
        else:
            credentials = {"api_key": api_key} if api_key else None

        # TLS configuration
        tls_kwargs: dict = {}
        if os.environ.get("AETHER_TLS_ENABLED", "").lower() in ("true", "1", "yes"):
            tls_kwargs["tls_enabled"] = True
            ca_cert = os.environ.get("AETHER_TLS_CA_CERT")
            if ca_cert:
                tls_kwargs["tls_root_cert_path"] = ca_cert
            client_cert = os.environ.get("AETHER_TLS_CLIENT_CERT")
            if client_cert:
                tls_kwargs["tls_client_cert_path"] = client_cert
            client_key = os.environ.get("AETHER_TLS_CLIENT_KEY")
            if client_key:
                tls_kwargs["tls_client_key_path"] = client_key

        try:
            # Delayed import: pulling the SDK at module load forces gRPC init
            from scitrera_aether_client import AsyncServiceClient

            client = AsyncServiceClient(
                implementation=_SERVICE_IMPLEMENTATION,
                specifier=self._specifier,
                credentials=credentials,
                **tls_kwargs,
            )
            await client.connect(gateway_addr)
            self._client = client
            logger.info(
                "Connected to Aether gateway at %s as sv.%s.%s",
                gateway_addr, _SERVICE_IMPLEMENTATION, self._specifier,
            )
        except Exception:
            logger.error("Failed to connect to Aether gateway at %s", gateway_addr, exc_info=True)
            # Non-fatal: the service can still run on direct HTTP
            return

        # Register the in-process proxy terminator
        try:
            from scitrera_aether_client.proxy_terminator import ProxyHttpTerminator

            from data_connectors.server._asgi_bridge import asgi_dispatch

            async def handler(req):
                return await asgi_dispatch(app, req)

            # Per-deployment tenant stamped as X-Auth-Tenant-ID on every minted
            # request (data-connectors is one deployment per tenant; the platform
            # framework injects SCITRERA_TENANT). Without it the Python terminator
            # carries no tenant and a fail-closed downstream rejects every request.
            tenant_id = os.environ.get("SCITRERA_TENANT")
            if not tenant_id:
                logger.warning(
                    "SCITRERA_TENANT unset: ProxyHttpTerminator will not stamp "
                    "X-Auth-Tenant-ID; fail-closed REST auth will reject requests"
                )

            terminator = ProxyHttpTerminator(
                client=self._client,
                handler=handler,
                allow_paths=list(_TERMINATOR_ALLOW_PATHS),
                header_mode="strict",
                tenant_id=tenant_id,
            )
            await terminator.start()
            self._terminator = terminator
            logger.info(
                "ProxyHttpTerminator registered (allow_paths=%s)",
                list(_TERMINATOR_ALLOW_PATHS),
            )
        except Exception:
            logger.warning(
                "Failed to start ProxyHttpTerminator (REST-over-Aether unavailable)",
                exc_info=True,
            )

    async def disconnect(self) -> None:
        """Disconnect from Aether and stop the terminator."""
        if self._terminator:
            try:
                await self._terminator.stop()
            except Exception:
                logger.warning("Error stopping ProxyHttpTerminator", exc_info=True)
            self._terminator = None

        if self._client:
            try:
                await self._client.close()
                logger.info("Disconnected from Aether gateway")
            except Exception:
                logger.warning("Error closing Aether client", exc_info=True)
            self._client = None
