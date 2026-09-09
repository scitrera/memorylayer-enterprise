# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Worker runner - connects to Aether and dispatches tasks to handlers.

The WorkerRunner initializes the MemoryLayer plugin system (same services as
the server but without FastAPI routes), discovers all registered
TaskHandlerPlugin instances, and listens for task messages on Aether.

When a message arrives it deserializes the payload, looks up the matching
handler by ``task_type``, and executes it.
"""
from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import json
import logging
import os
import signal
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from scitrera_rt_data.serialization.msgpack import msgpack_deserialize

from scitrera_app_framework import Variables, get_extensions, get_logger

if TYPE_CHECKING:
    from scitrera_aether_client import AsyncAgentClient

logger = logging.getLogger(__name__)

# ``malloc_trim`` handle, resolved once. ``False`` means "looked and did not
# find it" so we do not retry the lookup on every task.
_MALLOC_TRIM: Callable[[int], int] | None | bool = None


def _malloc_trim() -> None:
    """Ask the allocator to return free heap arenas to the OS.

    Document tasks allocate in large, short-lived bursts (rasterized pages,
    base64 payloads, multivectors) from pool threads, and glibc gives each
    thread its own arena. Freeing those objects returns the memory to the
    allocator but not to the kernel, so a worker that peaked at 1 GB kept
    reporting ~1 GB RSS long after the document finished — which is what
    container memory limits and autoscalers actually measure.

    Best-effort by design: this is a glibc extension, so it is a no-op on musl
    or macOS, and a failure here is never worth failing a completed task over.
    """
    global _MALLOC_TRIM

    if _MALLOC_TRIM is None:
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            _MALLOC_TRIM = libc.malloc_trim
            _MALLOC_TRIM.argtypes = [ctypes.c_size_t]
            _MALLOC_TRIM.restype = ctypes.c_int
        except (OSError, AttributeError) as exc:
            logger.debug("malloc_trim unavailable on this platform: %s", exc)
            _MALLOC_TRIM = False

    if _MALLOC_TRIM is False:
        return

    try:
        _MALLOC_TRIM(0)
    except Exception as exc:  # noqa: BLE001 - memory hygiene must never fail a task
        logger.debug("malloc_trim call failed: %s", exc)


def _deserialize_task_payload(data: bytes) -> dict | None:
    """Deserialize a task payload, trying msgpack first then JSON.

    Tasks from both the ML server and the workflow engine use msgpack.
    JSON fallback is kept for backward compatibility with in-flight tasks.
    """
    try:
        return msgpack_deserialize(data)
    except Exception:
        pass
    try:
        import json as _json
        return _json.loads(data)
    except Exception:
        pass
    return None


def _ext_multi_task_handlers() -> str:
    """Return the EXT_MULTI_TASK_HANDLERS extension point constant.

    Deferred import to avoid requiring the full memorylayer_server package
    at module load time (e.g. during test collection before the venv has
    been fully synced).
    """
    from memorylayer_server.services.tasks.base import EXT_MULTI_TASK_HANDLERS
    return EXT_MULTI_TASK_HANDLERS


class WorkerRunner:
    """Distributed worker that receives tasks from Aether and dispatches to handlers.

    Lifecycle:
        1. ``initialize()`` -- boot plugin system, discover handlers
        2. ``run()`` -- connect to Aether and process messages until stopped
        3. ``shutdown()`` -- tear down services and close the connection

    Args:
        v: Initialized Variables instance (after preconfigure + initialize_services).
        aether_addr: Aether gateway ``host:port``.
        workspace: Aether workspace name.
        specifier: Unique worker identity specifier.
        task_type_filter: Optional set of task types to handle. ``None`` means all.
    """

    def __init__(
        self,
        v: Variables,
        aether_addr: str = "localhost:50051",
        workspace: str = "_system",
        specifier: str = "worker-1",
        task_type_filter: Optional[set[str]] = None,
    ) -> None:
        self._v = v
        self._aether_addr = aether_addr
        self._workspace = workspace
        self._specifier = specifier
        self._task_type_filter = task_type_filter

        self._handlers: dict[str, Callable[[Variables, dict], Awaitable[None]]] = {}
        self._client: Optional[AsyncAgentClient] = None
        self._shutdown_event = asyncio.Event()
        self._logger = get_logger(v, name="WorkerRunner")

    # ------------------------------------------------------------------
    # Handler discovery
    # ------------------------------------------------------------------

    def discover_handlers(self) -> dict[str, Callable[[Variables, dict], Awaitable[None]]]:
        """Discover TaskHandlerPlugin instances from the plugin system.

        Returns:
            Mapping of task_type to handler callable.
        """
        ext_key = _ext_multi_task_handlers()
        handlers: dict[str, Callable[[Variables, dict], Awaitable[None]]] = {}
        extensions = get_extensions(ext_key, self._v)

        for handler_plugin in extensions.values():
            task_type: str = handler_plugin.get_task_type()

            # Apply filter if provided
            if self._task_type_filter and task_type not in self._task_type_filter:
                self._logger.debug("Skipping handler for task type %s (not in filter)", task_type)
                continue

            handlers[task_type] = handler_plugin.handle
            self._logger.info("Registered handler for task type: %s", task_type)

        return handlers

    def initialize(self) -> None:
        """Discover handlers and prepare for running.

        Call this after the plugin system has been fully initialized
        (``initialize_services`` completed).
        """
        self._handlers = self.discover_handlers()
        self._logger.info(
            "Worker initialized with %d handler(s): %s",
            len(self._handlers),
            ", ".join(sorted(self._handlers.keys())) or "(none)",
        )

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_task_assignment(self, assignment) -> None:
        """Process a native Aether task assignment (``TaskAssignment`` protobuf).

        This callback is invoked when the worker receives a task via Aether's
        native task lifecycle (``create_task`` / POOL assignment mode).

        The task type is stored in ``assignment.task_type`` with the prefix
        ``memorylayer-task.`` which is stripped before handler lookup.  The
        original payload is deserialized from ``assignment.payload`` (msgpack)
        with a fallback to ``assignment.metadata["payload"]`` (JSON).

        After execution, reports task completion or failure back to Aether
        via ``complete_task`` / ``fail_task``.

        Args:
            assignment: ``aether_pb2.TaskAssignment`` with fields
                ``task_id``, ``task_type``, ``metadata``, ``payload``.
        """
        raw_task_type: str = assignment.task_type or ""
        prefix = "memorylayer-task."
        if raw_task_type.startswith(prefix):
            task_type = raw_task_type[len(prefix):]
        else:
            task_type = raw_task_type

        aether_task_id: str = assignment.task_id or ""
        task_id: str = assignment.metadata.get("task_id", aether_task_id or "<unknown>")

        # Apply filter if provided
        if self._task_type_filter and task_type not in self._task_type_filter:
            self._logger.debug(
                "Ignoring native task assignment for type %s (not in filter), task_id=%s",
                task_type,
                task_id,
            )
            return

        handler = self._handlers.get(task_type)
        if handler is None:
            self._logger.warning(
                "No handler registered for native task type %s (task_id=%s), failing task",
                task_type,
                task_id,
            )
            await self._report_task_failed(aether_task_id, f"no handler for task type: {task_type}")
            return

        # Deserialize payload: try msgpack first (from ML server or workflow engine),
        # then JSON (legacy/backward compat), then metadata fallback.
        if assignment.payload:
            task_payload = _deserialize_task_payload(assignment.payload)
            if task_payload is None:
                self._logger.error(
                    "Failed to deserialize payload for task %s (tried msgpack and JSON)",
                    task_id,
                )
                await self._report_task_failed(aether_task_id, "payload deserialization error")
                return
        else:
            raw_payload = assignment.metadata.get("payload", "{}")
            try:
                task_payload = json.loads(raw_payload)
            except json.JSONDecodeError as exc:
                self._logger.error(
                    "Invalid JSON payload in native task assignment %s: %s",
                    task_id,
                    exc,
                )
                await self._report_task_failed(aether_task_id, f"JSON decode error: {exc}")
                return

        # Thread the Aether task_id and task metadata into the payload under
        # reserved keys so handlers can correlate live progress with the
        # Background Tasks snapshot (which is keyed by the Aether task_id, NOT
        # the dctask id stored in metadata["task_id"]).  These keys are absent
        # on the legacy/scheduler paths, so progress-aware handlers degrade
        # gracefully (no Aether task_id -> no progress emission).
        if isinstance(task_payload, dict):
            task_payload.setdefault("_aether_task_id", aether_task_id)
            task_payload.setdefault("_task_metadata", dict(assignment.metadata))

        self._logger.info("Executing native task %s (type=%s)", task_id, task_type)
        try:
            await handler(self._v, task_payload)
            self._logger.info("Native task %s completed successfully", task_id)
            await self._report_task_completed(aether_task_id)
        except Exception as exc:
            self._logger.error("Native task %s failed", task_id, exc_info=True)
            await self._report_task_failed(aether_task_id, str(exc))
        finally:
            # Hand the task's peak allocation back to the OS before going idle,
            # so RSS reflects the worker's steady state rather than its
            # high-water mark. Also runs on the failure path, where a partially
            # processed document may have allocated just as much.
            _malloc_trim()

    async def _report_task_completed(self, aether_task_id: str) -> None:
        """Report task completion to Aether. Best-effort."""
        if not aether_task_id or self._client is None:
            return
        try:
            await self._client.complete_task(aether_task_id, timeout=5.0)
        except Exception:
            self._logger.warning(
                "Failed to report task %s as completed to Aether",
                aether_task_id,
                exc_info=True,
            )

    async def _report_task_failed(self, aether_task_id: str, reason: str = "") -> None:
        """Report task failure to Aether. Best-effort."""
        if not aether_task_id or self._client is None:
            return
        try:
            await self._client.fail_task(aether_task_id, reason=reason, timeout=5.0)
        except Exception:
            self._logger.warning(
                "Failed to report task %s as failed to Aether",
                aether_task_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Connect to Aether and process messages until shutdown.

        Installs SIGINT/SIGTERM handlers so the worker can be stopped
        gracefully with ``Ctrl-C`` or ``kill``.
        """
        from scitrera_aether_client import AsyncAgentClient

        loop = asyncio.get_running_loop()

        def _signal_handler() -> None:
            self._logger.info("Shutdown signal received")
            self._shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

        credentials = None
        api_key = os.environ.get("AETHER_API_KEY")
        if not api_key:
            key_file = os.environ.get("AETHER_API_KEY_FILE")
            if key_file and os.path.isfile(key_file):
                with open(key_file) as f:
                    api_key = f.read().strip()
        auth_mode = os.environ.get("AETHER_AUTH", "")
        if auth_mode.lower() != "none" and api_key:
            credentials = {"api_key": api_key}

        # TLS configuration
        tls_kwargs = {}
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

        self._client = AsyncAgentClient(
            workspace=self._workspace,
            implementation="memorylayer",
            specifier=self._specifier,
            credentials=credentials,
            auto_reconnect=True,
            max_retries=0,  # infinite reconnection
            initial_backoff=1.0,
            max_backoff=30.0,
            **tls_kwargs,
        )

        # Tasks are delivered exclusively via Aether's native task lifecycle
        # (``create_task`` / POOL assignment) -> ``on_task_assignment``.  The
        # legacy topic-publish ``on_message`` dispatch was superseded by the
        # native path (which reports completion/failure for retry/DLQ safety)
        # and has been removed.
        self._client.on_task_assignment = self._handle_task_assignment

        async def _on_connect() -> None:
            self._logger.info("Connected to Aether at %s", self._aether_addr)

        async def _on_disconnect(reason: str) -> None:
            self._logger.warning("Disconnected from Aether: %s", reason)

        async def _on_error(error) -> None:
            self._logger.error("Aether error: code=%s message=%s", error.code, error.message)

        self._client.on_connect = _on_connect
        self._client.on_disconnect = _on_disconnect
        self._client.on_error = _on_error

        self._logger.info(
            "Connecting to Aether at %s (workspace=%s, specifier=%s)",
            self._aether_addr,
            self._workspace,
            self._specifier,
        )

        async with self._client as client:
            await client.connect(self._aether_addr)

            shutdown_task = asyncio.create_task(self._shutdown_event.wait())
            disconnect_task = asyncio.create_task(client.wait_until_disconnected())

            done, pending = await asyncio.wait(
                [shutdown_task, disconnect_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._logger.info("Worker stopped")

    async def shutdown(self) -> None:
        """Request a graceful shutdown of the worker."""
        self._shutdown_event.set()
        if self._client is not None:
            await self._client.close()
