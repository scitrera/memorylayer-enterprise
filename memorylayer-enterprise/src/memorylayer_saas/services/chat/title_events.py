"""Thread-title change broadcast helper (enterprise).

Pushes a spec-compliant ``rename`` control message to the thread owner over
Aether so connected UX clients can update the displayed title live.
Persistence (the title column) covers cold reloads; this broadcast is a
best-effort live nudge and must never break the caller.

The wire payload is a Universal Messaging Spec ``ChatMessage`` (see
``scitrera-ecosystem-messaging-spec`` §3.11) serialized as UTF-8 JSON — a
``role: "system"`` message carrying a single ``control`` content part with
``kind == "rename"`` and ``payload.title``. This replaces the ad-hoc msgpack
``{type: "thread.title_changed", ...}`` dict so every consumer parses the same
canonical shape. Per the spec, an agent sending a ``rename`` control is the
sanctioned way to propagate an auto-title/rename to connected clients.

Called by BOTH:
- the ``chat_thread_title`` task handler (LLM-generated titles, runs in a worker), and
- the enterprise chat service ``update_thread`` override (manual renames, runs in the API server).

Both processes hold the shared Aether connection.
"""

import logging

from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION
from memorylayer_server.utils import generate_id
from scitrera_app_framework import Variables, get_extension, get_logger
from scitrera_messaging_spec import ChatMessage, ControlPart, MessageAddress

# Spec control kind for a thread rename (messaging spec §3.11). Primary transport
# is the Aether user-global broadcast channel (``uu::{user_id}``): the user is
# always subscribed to it regardless of which workspace/window they have open, so
# a title change reaches them even while viewing a different workspace's sidebar.
RENAME_CONTROL_KIND = "rename"


def _build_rename_message(thread) -> ChatMessage:
    """Build the spec ``rename`` control ChatMessage for a thread title change.

    ``role: "system"`` + a single ``control`` part (``kind == "rename"``,
    ``payload.title``). ``addr`` carries the thread/workspace/user routing ids;
    ``meta.scitrera.title_source`` echoes whether the rename was auto or manual
    (forward-compatible extension, spec §6).
    """
    metadata = getattr(thread, "metadata", None) or {}
    meta: dict = {}
    title_source = metadata.get("title_source")
    if title_source:
        meta["scitrera"] = {"title_source": title_source}

    return ChatMessage(
        id=generate_id("msg"),
        role="system",
        addr=MessageAddress(
            workspace_id=thread.workspace_id,
            thread_id=thread.id,
            user_id=thread.user_id,
        ),
        content=[ControlPart(kind=RENAME_CONTROL_KIND, payload={"title": thread.title})],
        meta=meta,
    )


async def _send_title_event_to_user(conn, user_id: str, payload: bytes) -> None:
    """Transport seam: deliver ``payload`` to a single user.

    Targets the Aether user-global broadcast channel via
    ``send_message_to_user_broadcast`` (topic ``uu::{user_id}``). Falls back to
    the user-workspace topic (``uw::{user_id}::{workspace}``) on older Aether
    clients that predate the broadcast helper, so this works across the SDK
    rollout window. The Aether client accepts an optional ``authorization``
    (AuthorizationContext); sends to user topics succeed without one today, so it
    is intentionally omitted here and can be threaded through this single function
    if the gateway later requires it.
    """
    # Imported lazily so importing this module does not hard-require the Aether
    # SDK (keeps unit tests and OSS-adjacent tooling importable without it).
    # CHAT is the SDK's public top-level message-type constant: the payload is a
    # canonical ChatMessage (the messaging spec's convention is "CHAT payload is
    # UTF-8 JSON of a ChatMessage"), so it routes as a chat message rather than a
    # generic event.
    from scitrera_aether_client import CHAT

    client = conn.client
    send_broadcast = getattr(client, "send_message_to_user_broadcast", None)
    if send_broadcast is not None:
        await send_broadcast(user_id, payload, message_type=CHAT)
        return

    # Fallback for SDKs older than the user-broadcast channel.
    await client.send_message_to_user_workspace(
        user_id,
        conn.workspace,
        payload,
        message_type=CHAT,
    )


async def broadcast_thread_title_changed(v: Variables, thread) -> None:
    """Best-effort broadcast of a thread title change to its owner.

    Applies to BOTH homes — sentinel (user-owned) and workspace-homed — since
    threads are per-user in each and therefore always have a user topic when they
    have an owner.

    No-ops (never raises) when the thread has no owner, no title, or the Aether
    connection/client is unavailable. All failures are warn-logged so a failed
    broadcast can never break message append or a manual rename.
    """
    logger: logging.Logger = get_logger(v, name="ChatTitleEvents")
    try:
        # Gate on having an OWNER to address, not on ownership. ``uu::{user_id}``
        # is the transport, so an unattributed thread (service/system write, no
        # OBO human) has no topic to target — that, and only that, is the reason
        # to skip.
        #
        # This used to additionally require ownership == "user", back when
        # workspace-homed threads stored user_id = NULL and genuinely had no user
        # topic. They are now per-user in both homes (see _scope_user_id in
        # memorylayer_server/api/v1/chat.py), so that condition silently dropped
        # every workspace-homed rename: the title persisted and appeared on
        # reload, but no live nudge ever reached the sidebar.
        if not getattr(thread, "user_id", None):
            return

        # Spec requires a non-empty rename title; skip defensively if absent.
        if not getattr(thread, "title", None):
            return

        conn = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
        if conn is None or conn.client is None:
            logger.debug(
                "Aether connection unavailable; skipping title broadcast for thread %s",
                getattr(thread, "id", "<unknown>"),
            )
            return

        # Universal Messaging Spec ChatMessage (rename control) as UTF-8 JSON.
        # exclude_none keeps the wire compact (matches the spec examples); unknown
        # fields still round-trip per spec §8.
        message = _build_rename_message(thread)
        payload = message.model_dump_json(exclude_none=True).encode("utf-8")
        await _send_title_event_to_user(conn, thread.user_id, payload)
        logger.info(
            "Broadcast title change for thread %s to user %s",
            thread.id,
            thread.user_id,
        )
    except Exception as e:  # noqa: BLE001 - broadcast is strictly best-effort
        logger.warning(
            "Failed to broadcast title change for thread %s: %s",
            getattr(thread, "id", "<unknown>"),
            e,
        )
