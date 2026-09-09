"""OpenTelemetry TRACE initialization for the data-connectors service.

Mirrors the MemoryLayer OTelInitPlugin
(``memorylayer_server/lifecycle/otel.py``) and shares the same
``MEMORYLAYER_OTEL_*`` environment conventions so a single set of envs configures
both services. Traces only (no MeterProvider), matching MemoryLayer.

Everything here is fail-soft and import-guarded: a missing OTel SDK or a
telemetry configuration error must never break the ingest service. ``init_otel``
returns early (no-op) when the SDK is unavailable or ``MEMORYLAYER_OTEL_ENABLED``
is not truthy, and every instrumentation step is wrapped in try/except.

data-connectors does not depend on ``scitrera_app_framework`` in its own source,
so env parsing is done directly via ``os.environ`` rather than through
``Variables``/``ext_parse_bool``.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Config constants — shared with MemoryLayer (same env names configure both).
MEMORYLAYER_OTEL_ENABLED = "MEMORYLAYER_OTEL_ENABLED"
MEMORYLAYER_OTEL_EXPORTER = "MEMORYLAYER_OTEL_EXPORTER"  # 'otlp', 'console', 'none'
MEMORYLAYER_OTEL_ENDPOINT = "MEMORYLAYER_OTEL_ENDPOINT"  # e.g., 'http://localhost:4317'
MEMORYLAYER_OTEL_SERVICE_NAME = "MEMORYLAYER_OTEL_SERVICE_NAME"

DEFAULT_OTEL_SERVICE_NAME = "data-connectors"
DEFAULT_OTEL_ENDPOINT = "http://localhost:4317"

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    HAS_OTEL_SDK = True
except ImportError:
    HAS_OTEL_SDK = False

# Optional OTLP exporter.
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    HAS_OTLP = True
except ImportError:
    HAS_OTLP = False


def _parse_bool(value: Optional[str]) -> bool:
    """Lenient truthy parse for env strings (mirrors ext_parse_bool semantics)."""
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def init_otel(app: FastAPI) -> Optional[Callable[[], None]]:
    """Initialize OpenTelemetry tracing and instrument the FastAPI app.

    No-op (returns None) when the OTel SDK is not importable or
    ``MEMORYLAYER_OTEL_ENABLED`` is not truthy. Otherwise sets up a
    ``TracerProvider`` with an exporter selected by ``MEMORYLAYER_OTEL_EXPORTER``
    ('otlp' | 'console' | 'none'), instruments the FastAPI app for per-route
    spans, and auto-instruments SQLAlchemy + httpx so DB queries and outbound
    HTTP calls get child spans.

    Returns a ``shutdown`` callable that flushes/closes the ``TracerProvider``
    (call it on app shutdown), or ``None`` when tracing was not initialized.

    Every step is fail-soft: a telemetry problem logs a warning and is swallowed
    so the ingest service keeps serving requests.
    """
    if not HAS_OTEL_SDK:
        logger.debug("OTel SDK not installed; tracing disabled")
        return None

    if not _parse_bool(os.environ.get(MEMORYLAYER_OTEL_ENABLED)):
        logger.debug("%s not truthy; tracing disabled", MEMORYLAYER_OTEL_ENABLED)
        return None

    try:
        service_name = os.environ.get(MEMORYLAYER_OTEL_SERVICE_NAME, DEFAULT_OTEL_SERVICE_NAME)
        exporter_type = os.environ.get(MEMORYLAYER_OTEL_EXPORTER, "none").lower()

        resource = Resource(attributes={SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)

        if exporter_type == "console":
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            logger.info("OTel initialized: exporter=%s", exporter_type)
        elif exporter_type == "otlp":
            if not HAS_OTLP:
                logger.warning(
                    "OTel exporter=otlp requested but opentelemetry-exporter-otlp-proto-grpc "
                    "is not installed; spans will not be exported"
                )
            else:
                endpoint = os.environ.get(MEMORYLAYER_OTEL_ENDPOINT, DEFAULT_OTEL_ENDPOINT)
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
                logger.info("OTel initialized: exporter=%s, endpoint=%s", exporter_type, endpoint)
        else:
            logger.info("OTel initialized: exporter=%s (spans stay in-process)", exporter_type)

        trace.set_tracer_provider(provider)
    except Exception:
        logger.warning("Failed to initialize OTel TracerProvider; tracing disabled", exc_info=True)
        return None

    # Instrument the FastAPI app for full per-route span coverage. Exclude the
    # health/liveness probes so kubelet polling does not flood the trace store.
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,livez")
        logger.debug("FastAPIInstrumentor registered for data-connectors app")
    except ImportError:
        logger.debug("opentelemetry-instrumentation-fastapi not installed; route spans disabled")
    except Exception:
        logger.warning("Failed to instrument FastAPI app", exc_info=True)

    # SQLAlchemy async (the PostgreSQL backend via asyncpg) — DB query child spans.
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument()
        logger.debug("SQLAlchemyInstrumentor registered")
    except ImportError:
        logger.debug("opentelemetry-instrumentation-sqlalchemy not installed; DB spans disabled")
    except Exception:
        logger.warning("Failed to instrument SQLAlchemy", exc_info=True)

    # httpx outbound client (connectors + URL fetches) — outbound HTTP child spans.
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
        logger.debug("HTTPXClientInstrumentor registered")
    except ImportError:
        logger.debug("opentelemetry-instrumentation-httpx not installed; httpx spans disabled")
    except Exception:
        logger.warning("Failed to instrument httpx", exc_info=True)

    def _shutdown() -> None:
        try:
            provider.shutdown()
            logger.info("OTel TracerProvider shut down")
        except Exception:
            logger.warning("Error shutting down OTel TracerProvider", exc_info=True)

    return _shutdown
