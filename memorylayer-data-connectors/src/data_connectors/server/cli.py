"""CLI entry point for the data-connectors service."""
from __future__ import annotations

import os


def main() -> None:
    """Run the data-connectors FastAPI server via uvicorn."""
    # Delayed import: uvicorn and the app both have initialization cost
    import uvicorn

    host = os.environ.get("DC_HOST", "0.0.0.0")
    port = int(os.environ.get("DC_PORT", "8100"))

    # Collapse k8s probe access lines into a periodic rollup. kubelet polls
    # /healthz + /livez every few seconds forever, so the access log is almost
    # entirely probes whenever you go looking, pushing real requests out of the
    # retained window. Must go through log_config: uvicorn runs dictConfig at
    # startup and would discard a filter installed before this call.
    # 0 disables the summary and drops probe lines silently.
    from data_connectors.server.access_log import (
        DEFAULT_PROBE_ROLLUP_SECONDS,
        build_uvicorn_log_config,
    )

    try:
        probe_rollup_s = int(os.environ.get(
            "DC_ACCESS_LOG_PROBE_ROLLUP_S", DEFAULT_PROBE_ROLLUP_SECONDS,
        ))
    except ValueError:
        probe_rollup_s = DEFAULT_PROBE_ROLLUP_SECONDS

    uvicorn.run(
        "data_connectors.server.app:app",
        host=host,
        port=port,
        log_level="info",
        log_config=build_uvicorn_log_config(probe_rollup_s),
    )


if __name__ == "__main__":
    main()
