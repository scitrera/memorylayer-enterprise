"""ASGI bridge for dispatching Aether proxy requests into FastAPI.

Mirrors the pattern at ``memorylayer_server/services/aether_service/asgi_bridge.py``.
Converts an Aether ``ProxyHttpRequest`` into an ASGI scope/receive/send
cycle against the FastAPI app.
"""
from __future__ import annotations

import asyncio
import io
from typing import Any


async def asgi_dispatch(app: Any, request: Any) -> Any:
    """Dispatch a proxy HTTP request through the ASGI app.

    Args:
        app: FastAPI/Starlette ASGI application.
        request: Aether ``ProxyHttpRequest`` with ``.method``, ``.path``,
                 ``.headers`` (dict), ``.body`` (bytes).

    Returns:
        A ``(status, headers, body)`` tuple matching the
        ``ProxyHttpTerminator`` handler contract (``proxy_terminator.py::
        _coerce_response``). Returning a custom shim object with
        ``.status_code`` instead triggers
        ``TypeError: cannot unpack non-iterable _Response object``
        because _coerce_response tries to destructure the result as a tuple.
    """
    method = (request.method or "GET").upper()
    # ProxyHttpTerminator pre-splits ``path`` and ``query`` (see proxy_terminator.py:494
    # ``path, query = _split_path_query(req.path)``), so by the time we get here
    # ``request.path`` is just the path (no "?...") and the query string lives on
    # ``request.query``. Reading ``request.path`` and looking for "?" misses the
    # query entirely and causes every FastAPI route that uses ``Query(...)``
    # params to 422 with "Field required". The fallback split (in case some other
    # caller passes a path-with-query in ``request.path``) is harmless.
    raw_path = request.path or "/"
    raw_query = getattr(request, "query", "") or ""
    if "?" in raw_path and not raw_query:
        raw_path, raw_query = raw_path.split("?", 1)
    path = raw_path
    query_string = raw_query.encode("utf-8")

    # Build ASGI headers list
    headers_list: list[tuple[bytes, bytes]] = []
    req_headers = request.headers or {}
    for key, value in req_headers.items():
        headers_list.append((key.lower().encode("utf-8"), value.encode("utf-8")))

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "query_string": query_string,
        "root_path": "",
        "scheme": "http",
        "server": ("localhost", 80),
        "headers": headers_list,
        # Private transport metadata. Direct HTTP clients cannot populate ASGI
        # scope extensions; only the in-process Aether terminator can attach
        # the gateway-authored receipt here.
        "aether.access_receipt": getattr(request, "access_receipt", None),
    }

    body = request.body or b""
    body_sent = False

    async def receive() -> dict:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        # Should not be called again, but handle gracefully
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    status_code = 200
    response_headers: dict[str, str] = {}
    response_body = io.BytesIO()

    async def send(message: dict) -> None:
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
            for key, value in message.get("headers", []):
                response_headers[key.decode("utf-8")] = value.decode("utf-8")
        elif message["type"] == "http.response.body":
            response_body.write(message.get("body", b""))

    await app(scope, receive, send)

    return status_code, response_headers, response_body.getvalue()
