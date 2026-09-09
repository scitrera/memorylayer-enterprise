"""Rollup of uvicorn access-log lines for k8s probe endpoints.

WHY. Probes crowd out everything else. A 20-line tail of this service's log
held nothing but ``GET /livez`` and ``GET /healthz`` — kubelet polls both on a
few-second period forever, while real traffic arrives in bursts, so the log is
almost entirely probes at any moment you go looking. That is not just noise:
it pushes the requests you actually need out of the retained window.

WHAT. Individual probe lines are suppressed and replaced by ONE periodic
summary naming the paths and counts. Non-probe requests are untouched — real
traffic still logs line-per-request. An interval of 0 drops probe lines
silently without ever summarising.

HOW IT IS INSTALLED. uvicorn calls ``dictConfig`` during startup, which
REPLACES handlers and filters on its own loggers, so a filter attached before
``uvicorn.run`` is discarded. It has to arrive via ``log_config`` — see
:func:`build_uvicorn_log_config`.

Mirrors ``memorylayer_server.access_log``, deliberately as a copy rather than a
shared import: this service does not depend on memorylayer-core-python, and a
dependency in that direction (ingest service -> memory server) would be a worse
trade than ~100 duplicated lines. Keep the two in step when either changes.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

# Prefix-matched. Kept here rather than in a shared module because this service
# has exactly one consumer of the list; memorylayer split its copy out because
# its rate-limit middleware needed the same answer.
PROBE_PATH_PREFIXES: tuple[str, ...] = (
    "/healthz",
    "/livez",
    "/readyz",
    "/metrics",
)

# Emitted under its own logger so the summary cannot be swallowed by the very
# filter that produces it.
_ROLLUP_LOGGER = "data_connectors.access.probes"

DEFAULT_PROBE_ROLLUP_SECONDS = 300


def is_probe_path(path: str) -> bool:
    """Return True if ``path`` is an internal probe/metrics endpoint.

    Matches the path itself or a child segment, so ``/healthz`` also covers
    ``/healthz/ready`` but NOT ``/healthz-internal``, which is a different
    route that merely shares a prefix.
    """
    return any(path == p or path.startswith(p + "/") for p in PROBE_PATH_PREFIXES)


class ProbeAccessLogFilter(logging.Filter):
    """Suppress per-request probe access lines, emitting a periodic rollup.

    Attached to the ``uvicorn.access`` logger. uvicorn formats those records as
    ``'%s - "%s %s HTTP/%s" %d'`` with args
    ``(client_addr, method, full_path, http_version, status_code)``, so the
    path is ``record.args[2]``. Anything not matching that shape passes through
    unfiltered rather than being guessed at — a logging filter must never be
    the reason a line disappears.
    """

    def __init__(self, interval_seconds: int = DEFAULT_PROBE_ROLLUP_SECONDS) -> None:
        super().__init__()
        self._interval = max(0, int(interval_seconds))
        self._counts: Dict[str, int] = {}
        self._window_started = time.monotonic()
        self._logger = logging.getLogger(_ROLLUP_LOGGER)

    @staticmethod
    def _path_of(record: logging.LogRecord) -> Optional[str]:
        args: Any = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return None
        path = args[2]
        if not isinstance(path, str):
            return None
        # Strip any query string so ``/livez?x=1`` rolls up with ``/livez``.
        return path.split("?", 1)[0]

    def filter(self, record: logging.LogRecord) -> bool:
        path = self._path_of(record)
        if path is None or not is_probe_path(path):
            return True  # real traffic, or an unrecognised record shape

        self._counts[path] = self._counts.get(path, 0) + 1

        # interval 0 => suppress entirely, never summarise
        if self._interval:
            elapsed = time.monotonic() - self._window_started
            if elapsed >= self._interval:
                self._flush(elapsed)

        return False

    def _flush(self, elapsed: float) -> None:
        total = sum(self._counts.values())
        breakdown = ", ".join(
            f"{p}={n}" for p, n in sorted(self._counts.items(), key=lambda kv: -kv[1])
        )
        self._counts.clear()
        self._window_started = time.monotonic()
        self._logger.info(
            "health probes: %d requests in %.0fs (%s)", total, elapsed, breakdown
        )


def build_uvicorn_log_config(interval_seconds: int = DEFAULT_PROBE_ROLLUP_SECONDS) -> dict:
    """Return uvicorn's logging config with the probe rollup filter attached.

    Deep-copied from uvicorn's own ``LOGGING_CONFIG`` so formatters and handlers
    stay exactly as upstream defines them; only a filter and two loggers are
    added. Pass the result as ``uvicorn.run(log_config=...)``.
    """
    import copy

    from uvicorn.config import LOGGING_CONFIG

    config = copy.deepcopy(LOGGING_CONFIG)
    config.setdefault("filters", {})["probe_rollup"] = {
        "()": f"{__name__}.ProbeAccessLogFilter",
        "interval_seconds": interval_seconds,
    }
    access_logger = config.get("loggers", {}).get("uvicorn.access")
    if access_logger is not None:
        # Logger-level filter: uvicorn.access records all flow through this
        # logger, so this catches them regardless of handler wiring.
        access_logger.setdefault("filters", []).append("probe_rollup")

    # The rollup logger must reach a handler for the summary to surface at all.
    config.setdefault("loggers", {})[_ROLLUP_LOGGER] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }

    # This service's OWN loggers, which uvicorn's config does not mention and
    # which therefore reached no handler at all: every logger.info in the app
    # (connector registration, the abandoned-upload sweep's results) was being
    # written into the void. Quieting the probes is only half of making this
    # log useful; the other half is that what remains actually appears.
    config["loggers"]["data_connectors"] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }
    return config
