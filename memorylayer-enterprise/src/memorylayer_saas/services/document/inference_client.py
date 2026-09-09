"""Shared resolution of the prompt-embeds inference LLM client.

The grounded document-chat read path (``api/v1/document_chat.py``) and the
document-chat ingestion path (generating per-page memory text from a page's
precomputed ``image_embeds`` instead of OCR transcription) both need the SAME
inference client: a prompt-embeds-enabled vLLM, SEPARATE from the (L4)
preprocessing embed-server.

These helpers live here so both callers resolve the client identically without
the ingestion layer importing the API layer.

Resolution (see ``_get_inference_client``):
- If neither ``MEMORYLAYER_EMBED_LLM_SERVER_URL`` nor an aether target is set,
  reuse the default preprocessing client (colocated dev — e.g. the GB10 running
  both roles).
- Otherwise build a dedicated, persistent ``EmbedServerClient`` pointed at the
  inference node (http, or aether via the shared service connection).

The client is cached in ``Variables`` for the process lifetime so streaming
responses don't race a per-request close.
"""
from __future__ import annotations

import logging

from fastapi import HTTPException, status
from scitrera_app_framework import Variables, get_extension

from memorylayer_saas.config import (
    DEFAULT_MEMORYLAYER_EMBED_LLM_AETHER_TARGET,
    DEFAULT_MEMORYLAYER_EMBED_LLM_DEFAULT_MODEL,
    DEFAULT_MEMORYLAYER_EMBED_LLM_SERVER_URL,
    DEFAULT_MEMORYLAYER_EMBED_LLM_TIMEOUT,
    DEFAULT_MEMORYLAYER_EMBED_LLM_TRANSPORT,
    MEMORYLAYER_EMBED_LLM_AETHER_TARGET,
    MEMORYLAYER_EMBED_LLM_DEFAULT_MODEL,
    MEMORYLAYER_EMBED_LLM_SERVER_URL,
    MEMORYLAYER_EMBED_LLM_TIMEOUT,
    MEMORYLAYER_EMBED_LLM_TRANSPORT,
)
from memorylayer_saas.services.document import EXT_EMBED_SERVER_CLIENT

# Cached inference client (built lazily from config) lives under this key.
INFERENCE_CLIENT_KEY = "llm_inference_client"


def slugify_model(model: str) -> str:
    """Map a real model name to the slug under which its embeds are stored."""
    return model.replace("/", "--").replace(" ", "_").lower()


def default_inference_model(v: Variables) -> str:
    """The default model name when a caller omits ``model``."""
    return v.environ(
        MEMORYLAYER_EMBED_LLM_DEFAULT_MODEL,
        default=DEFAULT_MEMORYLAYER_EMBED_LLM_DEFAULT_MODEL,
    )


async def get_inference_client(v: Variables, logger: logging.Logger):
    """Return a connected client for LLM inference (prompt-embeds-enabled vLLM).

    The inference target is SEPARATE from the (L4) preprocessing embed-server.
    The client is cached in ``Variables`` for the process lifetime (like the
    default embed client), so streaming responses don't race a per-request
    close.
    """
    existing = v.get(INFERENCE_CLIENT_KEY, default=None)
    if existing is not None:
        await existing.connect()  # idempotent; reconnect if the shared client was closed
        return existing

    url = v.environ(MEMORYLAYER_EMBED_LLM_SERVER_URL, default=DEFAULT_MEMORYLAYER_EMBED_LLM_SERVER_URL)
    aether_target = v.environ(MEMORYLAYER_EMBED_LLM_AETHER_TARGET, default=DEFAULT_MEMORYLAYER_EMBED_LLM_AETHER_TARGET)

    if not url and not aether_target:
        # Colocated: the preprocessing embed-server also serves the LLM profile.
        # CAVEAT (pre-existing, colocated HTTP dev only): the returned client is
        # the shared EXT_EMBED_SERVER_CLIENT, which the ingestion embed phase
        # closes in its finally. A concurrent /v1/documents/chat stream sharing
        # this client could see it closed mid-stream; connect() is idempotent so
        # it self-heals on the next call. Configuring a dedicated inference target
        # (URL or aether) avoids the sharing entirely. Not worth ownership
        # plumbing until colocated concurrent chat+ingest is a real workload.
        client = get_extension(EXT_EMBED_SERVER_CLIENT, v)
        await client.connect()  # idempotent; the shared client may not be connected yet
        v.set(INFERENCE_CLIENT_KEY, client)
        return client

    from memorylayer_server.services.document.embed_client import (
        TRANSPORT_AETHER,
        EmbedServerClient,
    )

    transport = v.environ(MEMORYLAYER_EMBED_LLM_TRANSPORT, default=DEFAULT_MEMORYLAYER_EMBED_LLM_TRANSPORT).lower()
    timeout = float(v.environ(MEMORYLAYER_EMBED_LLM_TIMEOUT, default=DEFAULT_MEMORYLAYER_EMBED_LLM_TIMEOUT))

    aether_connection = None
    if transport == TRANSPORT_AETHER:
        from memorylayer_server.services.document._constants import EXT_AETHER_SERVICE_CONNECTION

        aether_connection = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
        if aether_connection is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM inference transport=aether but no AetherServiceConnection is configured.",
            )

    # When the embed-server service is "tokenary", build a TokenaryClient so the
    # chat content-block flatten (nested image_embeds -> tokenary's flat shape)
    # applies to this dedicated inference target too. Otherwise the default
    # EmbedServerClient (embed-server / vLLM) is used unchanged.
    from memorylayer_server.config import (
        DEFAULT_MEMORYLAYER_EMBED_SERVER_SERVICE,
        MEMORYLAYER_EMBED_SERVER_SERVICE,
    )

    embed_service = v.environ(
        MEMORYLAYER_EMBED_SERVER_SERVICE, default=DEFAULT_MEMORYLAYER_EMBED_SERVER_SERVICE
    )
    if embed_service == "tokenary":
        from .tokenary_client import TokenaryClient

        client_cls: type[EmbedServerClient] = TokenaryClient
    else:
        client_cls = EmbedServerClient

    client = client_cls(
        base_url=url or "http://unused",
        timeout=timeout,
        logger=logger,
        transport=transport,
        aether_connection=aether_connection,
        aether_target=aether_target or DEFAULT_MEMORYLAYER_EMBED_LLM_AETHER_TARGET,
    )
    await client.connect()
    v.set(INFERENCE_CLIENT_KEY, client)
    return client
