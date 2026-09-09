"""
Enterprise health dependencies endpoint.

Adds ``GET /v1/health/dependencies`` which reports the connectivity status of
external dependencies (Aether unified client, PostgreSQL storage).
"""
import logging

from fastapi import APIRouter, Depends, Response, status
from scitrera_app_framework import get_extension, Variables, Plugin

from memorylayer_server.api import EXT_MULTI_API_ROUTERS
from memorylayer_server.lifecycle.fastapi import get_variables_dep
from memorylayer_server.services._constants import EXT_STORAGE_BACKEND

router = APIRouter(tags=["health"])


@router.get("/livez")
async def livez(v: Variables = Depends(get_variables_dep)):
    """Connection-gated k8s liveness probe (readiness stays on /health/ready).

    Returns 200 while the shared Aether service connection is healthy — live
    now, or down-but-recently-up within the SDK reconnect grace (AETHER_MAX_
    RECONNECT_ATTEMPTS=0 retries forever, so a routine gateway roll heals on its
    own). Returns 503 only when the connection has been down past the grace (a
    stuck gateway), so kubelet restarts the pod. Before the Aether client
    connects (startup / not configured) the probe stays 200 — the SDK's
    pre-connect grace governs boot, mirroring connection_healthy().
    """
    try:
        from memorylayer_server.services.aether_service import (  # noqa: PLC0415
            AetherServiceConnection,
        )
        from memorylayer_server.services._constants import (  # noqa: PLC0415
            EXT_AETHER_SERVICE_CONNECTION,
        )
        conn = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
        client = conn.client if isinstance(conn, AetherServiceConnection) else None
    except Exception:
        # No Aether connection wired (e.g. non-aether config) — nothing to gate
        # liveness on, so stay alive and let readiness own the rest.
        client = None

    if client is None or client.connection_healthy():
        return {"status": "ok"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


@router.get("/v1/health/dependencies")
async def dependency_health(v: Variables = Depends(get_variables_dep)) -> dict:
    """Report health of external dependencies (Aether, PostgreSQL)."""
    dependencies: dict = {}

    # ------------------------------------------------------------------
    # Aether unified client (via AetherServiceConnection)
    # ------------------------------------------------------------------
    try:
        from memorylayer_server.services.aether_service import (  # noqa: PLC0415
            AetherServiceConnection,
        )
        from memorylayer_server.services._constants import (  # noqa: PLC0415
            EXT_AETHER_SERVICE_CONNECTION,
        )
        agent_service = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
        if isinstance(agent_service, AetherServiceConnection):
            connected = agent_service.is_connected
            aether_status = "connected" if connected else "disconnected"
            details: dict = {
                "workspace": agent_service.workspace,
                # Phase 1 (Aether convergence): the Aether identity is now
                # a Service principal — the workspace segment is gone.
                "identity": f"sv.memorylayer.{agent_service._specifier}",
            }
            dependencies["aether"] = {"status": aether_status, "details": details}
        else:
            dependencies["aether"] = {"status": "not_configured"}
    except Exception:
        logging.getLogger(__name__).warning("Failed to check Aether health", exc_info=True)
        dependencies["aether"] = {"status": "error"}

    # ------------------------------------------------------------------
    # PostgreSQL storage backend
    # ------------------------------------------------------------------
    try:
        storage = get_extension(EXT_STORAGE_BACKEND, v)
        session_factory = getattr(storage, "_session_factory", None)
        if session_factory is None:
            dependencies["postgresql"] = {"status": "not_configured"}
        else:
            try:
                async with session_factory() as session:
                    from sqlalchemy import text  # noqa: PLC0415
                    await session.execute(text("SELECT 1"))
                dependencies["postgresql"] = {"status": "connected"}
            except Exception:
                logging.getLogger(__name__).warning("PostgreSQL health check query failed", exc_info=True)
                dependencies["postgresql"] = {"status": "disconnected"}
    except Exception:
        logging.getLogger(__name__).warning("Failed to check PostgreSQL health", exc_info=True)
        dependencies["postgresql"] = {"status": "error"}

    # ------------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------------
    statuses = {dep["status"] for dep in dependencies.values()}
    if "disconnected" in statuses or "error" in statuses:
        if "postgresql" in dependencies and dependencies["postgresql"]["status"] in ("disconnected", "error"):
            overall = "unhealthy"
        else:
            overall = "degraded"
    else:
        overall = "healthy"

    return {"status": overall, "dependencies": dependencies}


class HealthDependenciesPlugin(Plugin):
    """Plugin that registers the ``/v1/health/dependencies`` route."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension; this is a multi-extension plugin

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        logger.info("Registering enterprise health dependencies route")
        return router
