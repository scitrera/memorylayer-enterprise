"""Unit tests for enterprise chat-thread auto-titling.

All external dependencies (Aether client, LLM service, storage) are mocked — no
Postgres or live gateway required. Covers:

- checkpoint-crossing math (single + batched jumps, explicit + interval)
- flag gate (disabled => no scheduling)
- normalized meaningful-drift gate
- title_source=='user' skip / stale title_index skip / placeholder-first-write
- broadcast topic + payload, workspace-owned skip, conn.client is None no-op
- update_thread override marks title_source='user' + broadcasts on manual rename
- provider selection ('enterprise') + preconfigure default
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memorylayer_saas.config import (
    DEFAULT_MEMORYLAYER_CHAT_TITLE_MAX_MESSAGES,
    DEFAULT_MEMORYLAYER_CHAT_TITLE_MAX_TOKENS,
    DEFAULT_MEMORYLAYER_CHAT_TITLE_PROMPT,
    MEMORYLAYER_CHAT_TITLE_ENABLED,
    MEMORYLAYER_CHAT_TITLE_INTERVAL,
    MEMORYLAYER_CHAT_TITLE_MAX_MESSAGES,
    MEMORYLAYER_CHAT_TITLE_MAX_TOKENS,
    MEMORYLAYER_CHAT_TITLE_PROMPT,
)

CHAT_MOD = "memorylayer_saas.services.chat.service"
EVENTS_MOD = "memorylayer_saas.services.chat.title_events"
TASK_MOD = "memorylayer_saas.tasks.chat_thread_title"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_variables(config: dict | None = None):
    """MagicMock Variables whose ``environ`` returns configured values (already typed)."""
    cfg = config or {}

    def _environ(key, default, **kwargs):
        return cfg.get(key, default)

    v = MagicMock()
    v.environ = MagicMock(side_effect=_environ)
    return v


def _make_thread(**overrides):
    base = dict(
        id="thread-1",
        workspace_id="ws-1",
        user_id="user-1",
        ownership="user",
        title="thread-1",  # placeholder == id
        metadata={},
        message_count=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_message(role: str, content: str, index: int = 0):
    return SimpleNamespace(role=role, content=content, message_index=index)


def _make_chat_service(config: dict | None = None):
    from memorylayer_saas.services.chat.service import EnterpriseChatService

    v = _make_variables(config)
    storage = MagicMock()
    task_service = MagicMock()
    task_service.schedule_task = AsyncMock()
    svc = EnterpriseChatService(storage=storage, task_service=task_service, v=v)
    return svc


def _enabled_config(**extra):
    cfg = {MEMORYLAYER_CHAT_TITLE_ENABLED: True}
    cfg.update(extra)
    return cfg


# ===========================================================================
# _crosses_checkpoint / _maybe_schedule_titling — checkpoint math
# ===========================================================================


class TestCheckpointMath:
    @pytest.mark.parametrize(
        "old,new,expected",
        [
            (0, 1, True),    # first message hits checkpoint 1
            (1, 2, False),   # between 1 and 5
            (4, 5, True),    # crosses checkpoint 5
            (5, 9, False),   # between 5 and 10
            (9, 10, True),   # crosses checkpoint 10
            (10, 11, False), # just past 10, before next interval multiple (20)
            (19, 20, True),  # interval multiple 20
            (0, 12, True),   # batched jump spanning 1/5/10
            (11, 19, False), # batched jump entirely between 10 and 20
            (15, 25, True),  # batched jump crossing interval multiple 20
        ],
    )
    def test_crosses_checkpoint(self, old, new, expected):
        svc = _make_chat_service(_enabled_config())
        assert svc._crosses_checkpoint(old, new) is expected

    @pytest.mark.asyncio
    async def test_schedules_on_checkpoint(self):
        svc = _make_chat_service(_enabled_config())
        thread = _make_thread(message_count=4)
        await svc._maybe_schedule_titling("ws-1", "thread-1", thread, appended=1)  # 4 -> 5
        svc.task_service.schedule_task.assert_awaited_once()
        args = svc.task_service.schedule_task.call_args
        assert args.args[0] == "chat_thread_title"
        # Owner (user_id) rides on the payload so the title worker can re-resolve
        # the owner-scoped thread from a shared client id.
        assert args.args[1] == {
            "workspace_id": "ws-1",
            "thread_id": "thread-1",
            "user_id": "user-1",
            "at_index": 5,
        }

    @pytest.mark.asyncio
    async def test_no_schedule_between_checkpoints(self):
        svc = _make_chat_service(_enabled_config())
        thread = _make_thread(message_count=2)
        await svc._maybe_schedule_titling("ws-1", "thread-1", thread, appended=1)  # 2 -> 3
        svc.task_service.schedule_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_batched_jump_fires_once(self):
        svc = _make_chat_service(_enabled_config())
        thread = _make_thread(message_count=0)
        await svc._maybe_schedule_titling("ws-1", "thread-1", thread, appended=8)  # 0 -> 8
        svc.task_service.schedule_task.assert_awaited_once()
        assert svc.task_service.schedule_task.call_args.args[1]["at_index"] == 8

    @pytest.mark.asyncio
    async def test_disabled_flag_no_schedule(self):
        svc = _make_chat_service({MEMORYLAYER_CHAT_TITLE_ENABLED: False})
        thread = _make_thread(message_count=0)
        await svc._maybe_schedule_titling("ws-1", "thread-1", thread, appended=1)
        svc.task_service.schedule_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_none_thread_treated_as_zero(self):
        svc = _make_chat_service(_enabled_config())
        await svc._maybe_schedule_titling("ws-1", "thread-1", None, appended=1)  # 0 -> 1
        svc.task_service.schedule_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_interval_zero_only_explicit(self):
        svc = _make_chat_service(_enabled_config(**{MEMORYLAYER_CHAT_TITLE_INTERVAL: 0}))
        thread = _make_thread(message_count=19)
        await svc._maybe_schedule_titling("ws-1", "thread-1", thread, appended=1)  # 19 -> 20
        svc.task_service.schedule_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_schedule_failure_is_swallowed(self):
        svc = _make_chat_service(_enabled_config())
        svc.task_service.schedule_task = AsyncMock(side_effect=RuntimeError("aether down"))
        thread = _make_thread(message_count=0)
        # Must not raise.
        await svc._maybe_schedule_titling("ws-1", "thread-1", thread, appended=1)


# ===========================================================================
# append_messages override wiring
# ===========================================================================


class TestAppendMessagesOverride:
    @pytest.mark.asyncio
    async def test_append_calls_titling_after_super(self):
        from memorylayer_server.services.chat.default import DefaultChatService

        svc = _make_chat_service(_enabled_config())
        thread_before = _make_thread(message_count=4)
        svc.get_thread = AsyncMock(return_value=thread_before)
        result_msgs = [_make_message("user", "hi", 4)]

        with (
            patch.object(DefaultChatService, "append_messages", new=AsyncMock(return_value=result_msgs)),
            patch.object(svc, "_maybe_schedule_titling", new=AsyncMock()) as mock_titling,
        ):
            out = await svc.append_messages("ws-1", "thread-1", MagicMock(), tenant_id="t")

        assert out is result_msgs
        mock_titling.assert_awaited_once_with("ws-1", "thread-1", thread_before, 1)

    @pytest.mark.asyncio
    async def test_append_never_breaks_on_titling_error(self):
        from memorylayer_server.services.chat.default import DefaultChatService

        svc = _make_chat_service(_enabled_config())
        svc.get_thread = AsyncMock(return_value=_make_thread())
        result_msgs = [_make_message("user", "hi", 0)]

        with (
            patch.object(DefaultChatService, "append_messages", new=AsyncMock(return_value=result_msgs)),
            patch.object(svc, "_maybe_schedule_titling", new=AsyncMock(side_effect=RuntimeError("boom"))),
        ):
            out = await svc.append_messages("ws-1", "thread-1", MagicMock())

        assert out is result_msgs  # append result still returned


# ===========================================================================
# update_thread override — manual rename marking + broadcast
# ===========================================================================


class TestUpdateThreadOverride:
    @pytest.mark.asyncio
    async def test_manual_rename_marks_user_and_broadcasts(self):
        from memorylayer_server.services.chat.default import DefaultChatService

        svc = _make_chat_service(_enabled_config())
        existing = _make_thread(metadata={"keep": "me"})
        updated = _make_thread(title="New Name", metadata={"keep": "me", "title_source": "user"})
        svc.get_thread = AsyncMock(return_value=existing)

        captured = {}

        async def _fake_super(ws, tid, **updates):
            captured.update(updates)
            return updated

        with (
            patch.object(DefaultChatService, "update_thread", new=AsyncMock(side_effect=_fake_super)),
            patch(f"{CHAT_MOD}.broadcast_thread_title_changed", new=AsyncMock()) as mock_bcast,
        ):
            result = await svc.update_thread("ws-1", "thread-1", title="New Name")

        assert result is updated
        # merged metadata carries the pre-existing key AND the user marker
        assert captured["metadata"] == {"keep": "me", "title_source": "user"}
        mock_bcast.assert_awaited_once_with(svc._v, updated)

    @pytest.mark.asyncio
    async def test_non_title_update_does_not_broadcast(self):
        from memorylayer_server.services.chat.default import DefaultChatService

        svc = _make_chat_service(_enabled_config())
        svc.get_thread = AsyncMock(return_value=_make_thread())

        with (
            patch.object(DefaultChatService, "update_thread", new=AsyncMock(return_value=_make_thread())),
            patch(f"{CHAT_MOD}.broadcast_thread_title_changed", new=AsyncMock()) as mock_bcast,
        ):
            await svc.update_thread("ws-1", "thread-1", idle_action="hide")

        mock_bcast.assert_not_awaited()


# ===========================================================================
# Broadcast helper
# ===========================================================================


class TestBroadcastHelper:
    @pytest.mark.asyncio
    async def test_broadcast_sends_spec_rename_control_message(self):
        from scitrera_aether_client import CHAT
        from scitrera_messaging_spec import ChatMessage

        from memorylayer_saas.services.chat.title_events import broadcast_thread_title_changed

        client = AsyncMock()
        client.send_message_to_user_broadcast = AsyncMock()
        conn = SimpleNamespace(client=client, workspace="prod-ws")
        thread = _make_thread(title="Vacation Plans", metadata={"title_source": "auto"})

        with patch(f"{EVENTS_MOD}.get_extension", return_value=conn):
            await broadcast_thread_title_changed(_make_variables(), thread)

        # Primary path: user-global broadcast channel (uu::{user_id}) — no workspace arg.
        client.send_message_to_user_broadcast.assert_awaited_once()
        call = client.send_message_to_user_broadcast.call_args
        assert call.args[0] == "user-1"          # user_id
        assert isinstance(call.args[1], bytes)   # UTF-8 JSON payload
        # Routed as a CHAT message (payload is a canonical ChatMessage).
        assert call.kwargs["message_type"] == CHAT

        # Payload is a spec ChatMessage carrying a single `rename` control part.
        data = json.loads(call.args[1].decode("utf-8"))
        assert data["schema_version"] == "1.0"
        assert data["role"] == "system"
        assert data["addr"]["thread_id"] == "thread-1"
        assert data["addr"]["workspace_id"] == "ws-1"
        assert data["addr"]["user_id"] == "user-1"
        assert len(data["content"]) == 1
        part = data["content"][0]
        assert part["type"] == "control"
        assert part["kind"] == "rename"
        assert part["payload"] == {"title": "Vacation Plans"}
        assert data["meta"]["scitrera"]["title_source"] == "auto"

        # And it round-trips back through the spec model.
        msg = ChatMessage.model_validate(data)
        assert msg.role == "system"
        assert msg.content[0].kind == "rename"
        assert msg.content[0].payload["title"] == "Vacation Plans"

    @pytest.mark.asyncio
    async def test_broadcast_falls_back_to_workspace_on_older_sdk(self):
        """Clients predating send_message_to_user_broadcast use the uw:: topic."""
        from scitrera_aether_client import CHAT

        from memorylayer_saas.services.chat.title_events import broadcast_thread_title_changed

        # SimpleNamespace client lacks send_message_to_user_broadcast entirely,
        # so getattr(...) returns None and the seam falls back.
        client = SimpleNamespace(send_message_to_user_workspace=AsyncMock())
        conn = SimpleNamespace(client=client, workspace="prod-ws")

        with patch(f"{EVENTS_MOD}.get_extension", return_value=conn):
            await broadcast_thread_title_changed(_make_variables(), _make_thread(title="Legacy"))

        client.send_message_to_user_workspace.assert_awaited_once()
        call = client.send_message_to_user_workspace.call_args
        assert call.args[0] == "user-1"          # user_id
        assert call.args[1] == "prod-ws"         # workspace
        assert isinstance(call.args[2], bytes)   # UTF-8 JSON payload
        assert call.kwargs["message_type"] == CHAT
        # Same spec rename control shape on the fallback topic.
        data = json.loads(call.args[2].decode("utf-8"))
        assert data["content"][0]["kind"] == "rename"
        assert data["content"][0]["payload"] == {"title": "Legacy"}

    @pytest.mark.asyncio
    async def test_broadcast_covers_workspace_homed_thread(self):
        """A workspace-homed thread WITH an owner broadcasts like any other.

        Regression: the gate also required ownership == "user", written back when
        workspace-homed threads stored user_id = NULL. Once those became per-user,
        that condition silently dropped every workspace-homed rename — the title
        persisted (and showed up on reload) but the sidebar never updated live.
        """
        from memorylayer_saas.services.chat.title_events import broadcast_thread_title_changed

        client = AsyncMock()
        conn = SimpleNamespace(client=client, workspace="prod-ws")
        thread = _make_thread(ownership="workspace", user_id="alice@example.com")

        with patch(f"{EVENTS_MOD}.get_extension", return_value=conn):
            await broadcast_thread_title_changed(_make_variables(), thread)

        client.send_message_to_user_broadcast.assert_awaited_once()
        call = client.send_message_to_user_broadcast.call_args
        # Addressed to the thread's owner — the uu:: topic key.
        assert call.args[0] == "alice@example.com"
        data = json.loads(call.args[1].decode("utf-8"))
        assert data["content"][0]["kind"] == "rename"
        assert data["addr"]["thread_id"] == thread.id

    @pytest.mark.asyncio
    async def test_broadcast_skips_thread_without_owner(self):
        """No owner -> no ``uu::{user_id}`` topic to target, so skip.

        That is the ONLY reason to skip: the gate deliberately does not consult
        ``ownership`` (see test_broadcast_covers_workspace_homed_thread).
        """
        from memorylayer_saas.services.chat.title_events import broadcast_thread_title_changed

        conn = SimpleNamespace(client=AsyncMock(), workspace="prod-ws")
        thread = _make_thread(ownership="workspace", user_id=None)

        with patch(f"{EVENTS_MOD}.get_extension", return_value=conn) as mock_get:
            await broadcast_thread_title_changed(_make_variables(), thread)

        # returns before ever resolving the connection
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_noop_when_client_none(self):
        from memorylayer_saas.services.chat.title_events import broadcast_thread_title_changed

        conn = SimpleNamespace(client=None, workspace="prod-ws")
        thread = _make_thread()

        with patch(f"{EVENTS_MOD}.get_extension", return_value=conn):
            # Must not raise despite no client.
            await broadcast_thread_title_changed(_make_variables(), thread)

    @pytest.mark.asyncio
    async def test_broadcast_swallows_send_error(self):
        from memorylayer_saas.services.chat.title_events import broadcast_thread_title_changed

        client = AsyncMock()
        client.send_message_to_user_broadcast = AsyncMock(side_effect=RuntimeError("send failed"))
        conn = SimpleNamespace(client=client, workspace="prod-ws")

        with patch(f"{EVENTS_MOD}.get_extension", return_value=conn):
            # Must not raise.
            await broadcast_thread_title_changed(_make_variables(), _make_thread())


# ===========================================================================
# Task handler
# ===========================================================================


def _handler_config(**extra):
    cfg = {
        MEMORYLAYER_CHAT_TITLE_MAX_MESSAGES: DEFAULT_MEMORYLAYER_CHAT_TITLE_MAX_MESSAGES,
        MEMORYLAYER_CHAT_TITLE_MAX_TOKENS: DEFAULT_MEMORYLAYER_CHAT_TITLE_MAX_TOKENS,
        MEMORYLAYER_CHAT_TITLE_PROMPT: DEFAULT_MEMORYLAYER_CHAT_TITLE_PROMPT,
    }
    cfg.update(extra)
    return cfg


def _make_handler():
    from memorylayer_saas.tasks.chat_thread_title import ChatThreadTitleTaskHandler

    return ChatThreadTitleTaskHandler()


class TestTaskHandler:
    def test_task_type_and_schedule(self):
        h = _make_handler()
        assert h.get_task_type() == "chat_thread_title"
        assert h.get_schedule(_make_variables()) is None

    @pytest.mark.asyncio
    async def test_placeholder_first_write_persists_and_broadcasts(self):
        h = _make_handler()
        v = _make_variables(_handler_config())
        thread = _make_thread(title="thread-1", metadata={})  # placeholder title
        updated = _make_thread(title="Trip Planning", metadata={"title_source": "auto", "title_index": 1})

        storage = MagicMock()
        storage.get_thread = AsyncMock(return_value=thread)
        storage.get_messages = AsyncMock(return_value=[_make_message("user", "Help me plan a trip", 0)])
        storage.update_thread = AsyncMock(return_value=updated)

        llm = MagicMock()
        llm.synthesize = AsyncMock(return_value="Trip Planning")

        with (
            patch(f"{TASK_MOD}.get_extension", return_value=storage),
            patch(f"{TASK_MOD}.get_llm_service", return_value=llm),
            patch(f"{TASK_MOD}.broadcast_thread_title_changed", new=AsyncMock()) as mock_bcast,
        ):
            await h.handle(v, {"workspace_id": "ws-1", "thread_id": "thread-1", "at_index": 1})

        storage.update_thread.assert_awaited_once()
        kw = storage.update_thread.call_args
        assert kw.kwargs["title"] == "Trip Planning"
        assert kw.kwargs["metadata"] == {"title_source": "auto", "title_index": 1}
        mock_bcast.assert_awaited_once_with(v, updated)

    @pytest.mark.asyncio
    async def test_merges_existing_metadata(self):
        h = _make_handler()
        v = _make_variables(_handler_config())
        thread = _make_thread(title="thread-1", metadata={"foo": "bar"})

        storage = MagicMock()
        storage.get_thread = AsyncMock(return_value=thread)
        storage.get_messages = AsyncMock(return_value=[_make_message("user", "hi", 0)])
        storage.update_thread = AsyncMock(return_value=_make_thread(title="New"))

        llm = MagicMock()
        llm.synthesize = AsyncMock(return_value="New Title")

        with (
            patch(f"{TASK_MOD}.get_extension", return_value=storage),
            patch(f"{TASK_MOD}.get_llm_service", return_value=llm),
            patch(f"{TASK_MOD}.broadcast_thread_title_changed", new=AsyncMock()),
        ):
            await h.handle(v, {"workspace_id": "ws-1", "thread_id": "thread-1", "at_index": 5})

        meta = storage.update_thread.call_args.kwargs["metadata"]
        assert meta == {"foo": "bar", "title_source": "auto", "title_index": 5}

    @pytest.mark.asyncio
    async def test_skip_user_sourced_title(self):
        h = _make_handler()
        v = _make_variables(_handler_config())
        thread = _make_thread(title="Manual", metadata={"title_source": "user"})

        storage = MagicMock()
        storage.get_thread = AsyncMock(return_value=thread)
        storage.get_messages = AsyncMock()
        storage.update_thread = AsyncMock()
        llm = MagicMock()
        llm.synthesize = AsyncMock()

        with (
            patch(f"{TASK_MOD}.get_extension", return_value=storage),
            patch(f"{TASK_MOD}.get_llm_service", return_value=llm),
            patch(f"{TASK_MOD}.broadcast_thread_title_changed", new=AsyncMock()) as mock_bcast,
        ):
            await h.handle(v, {"workspace_id": "ws-1", "thread_id": "thread-1", "at_index": 5})

        llm.synthesize.assert_not_awaited()
        storage.update_thread.assert_not_awaited()
        mock_bcast.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skip_stale_title_index(self):
        h = _make_handler()
        v = _make_variables(_handler_config())
        thread = _make_thread(title="Recent", metadata={"title_source": "auto", "title_index": 10})

        storage = MagicMock()
        storage.get_thread = AsyncMock(return_value=thread)
        storage.get_messages = AsyncMock()
        storage.update_thread = AsyncMock()
        llm = MagicMock()
        llm.synthesize = AsyncMock()

        with (
            patch(f"{TASK_MOD}.get_extension", return_value=storage),
            patch(f"{TASK_MOD}.get_llm_service", return_value=llm),
            patch(f"{TASK_MOD}.broadcast_thread_title_changed", new=AsyncMock()),
        ):
            await h.handle(v, {"workspace_id": "ws-1", "thread_id": "thread-1", "at_index": 5})  # 5 < 10

        llm.synthesize.assert_not_awaited()
        storage.update_thread.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_normalized_drift_gate_skips_equivalent(self):
        h = _make_handler()
        v = _make_variables(_handler_config())
        # Current title is meaningfully equal to the synthesized one after
        # casefold + punctuation/whitespace collapse.
        thread = _make_thread(title="Trip Planning", metadata={"title_source": "auto", "title_index": 1})

        storage = MagicMock()
        storage.get_thread = AsyncMock(return_value=thread)
        storage.get_messages = AsyncMock(return_value=[_make_message("user", "hi", 0)])
        storage.update_thread = AsyncMock()

        llm = MagicMock()
        llm.synthesize = AsyncMock(return_value="trip   planning!!!")

        with (
            patch(f"{TASK_MOD}.get_extension", return_value=storage),
            patch(f"{TASK_MOD}.get_llm_service", return_value=llm),
            patch(f"{TASK_MOD}.broadcast_thread_title_changed", new=AsyncMock()) as mock_bcast,
        ):
            await h.handle(v, {"workspace_id": "ws-1", "thread_id": "thread-1", "at_index": 5})

        storage.update_thread.assert_not_awaited()
        mock_bcast.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_llm_failure_does_not_persist(self):
        h = _make_handler()
        v = _make_variables(_handler_config())
        thread = _make_thread(title="thread-1", metadata={})

        storage = MagicMock()
        storage.get_thread = AsyncMock(return_value=thread)
        storage.get_messages = AsyncMock(return_value=[_make_message("user", "hi", 0)])
        storage.update_thread = AsyncMock()

        llm = MagicMock()
        llm.synthesize = AsyncMock(side_effect=RuntimeError("llm down"))

        with (
            patch(f"{TASK_MOD}.get_extension", return_value=storage),
            patch(f"{TASK_MOD}.get_llm_service", return_value=llm),
            patch(f"{TASK_MOD}.broadcast_thread_title_changed", new=AsyncMock()) as mock_bcast,
        ):
            await h.handle(v, {"workspace_id": "ws-1", "thread_id": "thread-1", "at_index": 1})

        storage.update_thread.assert_not_awaited()
        mock_bcast.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_payload_fields_returns_early(self):
        h = _make_handler()
        v = _make_variables(_handler_config())
        storage = MagicMock()
        storage.get_thread = AsyncMock()

        with patch(f"{TASK_MOD}.get_extension", return_value=storage):
            await h.handle(v, {"thread_id": "thread-1"})  # no workspace_id

        storage.get_thread.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prompt_placeholder_is_honored(self):
        h = _make_handler()
        v = _make_variables(_handler_config(**{MEMORYLAYER_CHAT_TITLE_PROMPT: "PRE {convo} POST"}))
        thread = _make_thread(title="thread-1", metadata={})

        storage = MagicMock()
        storage.get_thread = AsyncMock(return_value=thread)
        storage.get_messages = AsyncMock(return_value=[_make_message("user", "hello world", 0)])
        storage.update_thread = AsyncMock(return_value=_make_thread(title="X"))

        llm = MagicMock()
        llm.synthesize = AsyncMock(return_value="X")

        with (
            patch(f"{TASK_MOD}.get_extension", return_value=storage),
            patch(f"{TASK_MOD}.get_llm_service", return_value=llm),
            patch(f"{TASK_MOD}.broadcast_thread_title_changed", new=AsyncMock()),
        ):
            await h.handle(v, {"workspace_id": "ws-1", "thread_id": "thread-1", "at_index": 1})

        sent_prompt = llm.synthesize.call_args.args[0]
        assert sent_prompt == "PRE user: hello world POST"


# ===========================================================================
# Provider selection
# ===========================================================================


class TestAsyncIOProviderEndToEnd:
    """Prove the full chain fires under ``MEMORYLAYER_TASK_PROVIDER=asyncio``.

    This is the exact configuration the .e2e deployment uses (asyncio provider,
    in-process worker). Isolated unit tests already cover the schedule gate and
    the handler; this test wires the REAL ``AsyncIOTaskService`` with the REAL
    ``ChatThreadTitleTaskHandler`` registered, then drives a checkpoint crossing
    and asserts the handler runs in-process and persists the title. It closes
    the "does it actually fire end-to-end?" gap.
    """

    @pytest.mark.asyncio
    async def test_checkpoint_crossing_persists_title_via_asyncio_provider(self):
        from memorylayer_server.services.tasks.asyncio_impl import AsyncIOTaskService

        from memorylayer_saas.services.chat.service import EnterpriseChatService
        from memorylayer_saas.tasks.chat_thread_title import ChatThreadTitleTaskHandler

        v = _make_variables({**_enabled_config(), **_handler_config()})

        # Real in-process asyncio provider with the real title handler registered
        # (mirrors handlers.py wiring: register_handler(get_task_type(), handle)).
        task_service = AsyncIOTaskService(v=v, tasks_enabled=True)
        handler = ChatThreadTitleTaskHandler()
        task_service.register_handler(handler.get_task_type(), handler.handle)

        svc = EnterpriseChatService(storage=MagicMock(), task_service=task_service, v=v)

        # Handler-side storage + LLM, patched at the task-module seam.
        thread = _make_thread(title="thread-1", metadata={})  # placeholder title == id
        updated = _make_thread(title="Trip Planning", metadata={"title_source": "auto", "title_index": 5})
        handler_storage = MagicMock()
        handler_storage.get_thread = AsyncMock(return_value=thread)
        handler_storage.get_messages = AsyncMock(return_value=[_make_message("user", "Help me plan a trip", 0)])
        handler_storage.update_thread = AsyncMock(return_value=updated)
        llm = MagicMock()
        llm.synthesize = AsyncMock(return_value="Trip Planning")

        with (
            patch(f"{TASK_MOD}.get_extension", return_value=handler_storage),
            patch(f"{TASK_MOD}.get_llm_service", return_value=llm),
            patch(f"{TASK_MOD}.broadcast_thread_title_changed", new=AsyncMock()) as mock_bcast,
        ):
            # Real scheduling gate: message_count 4 -> 5 crosses checkpoint 5.
            thread_before = _make_thread(message_count=4)
            await svc._maybe_schedule_titling("ws-1", "thread-1", thread_before, appended=1)

            # Exactly one background task was dispatched onto the asyncio provider;
            # await it deterministically instead of sleeping.
            assert len(task_service._tasks) == 1
            await asyncio.gather(*task_service._tasks.values())

        # The handler actually ran in-process and persisted the synthesized title.
        handler_storage.update_thread.assert_awaited_once()
        kw = handler_storage.update_thread.call_args.kwargs
        assert kw["title"] == "Trip Planning"
        assert kw["metadata"] == {"title_source": "auto", "title_index": 5}
        mock_bcast.assert_awaited_once_with(v, updated)

    @pytest.mark.asyncio
    async def test_no_dispatch_when_disabled(self):
        """With the flag off (the current .e2e default), nothing is dispatched."""
        from memorylayer_server.services.tasks.asyncio_impl import AsyncIOTaskService

        from memorylayer_saas.services.chat.service import EnterpriseChatService
        from memorylayer_saas.tasks.chat_thread_title import ChatThreadTitleTaskHandler

        v = _make_variables({MEMORYLAYER_CHAT_TITLE_ENABLED: False})
        task_service = AsyncIOTaskService(v=v, tasks_enabled=True)
        handler = ChatThreadTitleTaskHandler()
        task_service.register_handler(handler.get_task_type(), handler.handle)
        svc = EnterpriseChatService(storage=MagicMock(), task_service=task_service, v=v)

        await svc._maybe_schedule_titling("ws-1", "thread-1", _make_thread(message_count=4), appended=1)

        assert len(task_service._tasks) == 0


class TestProviderSelection:
    def test_plugin_provider_name(self):
        from memorylayer_saas.services.chat.service import EnterpriseChatServicePlugin

        assert EnterpriseChatServicePlugin.PROVIDER_NAME == "enterprise"

    def test_preconfigure_selects_enterprise_chat_service(self):
        from memorylayer_server.config import MEMORYLAYER_CHAT_SERVICE
        from scitrera_app_framework import Variables

        from memorylayer_saas.dependencies import _enterprise_preconfigure_hook

        v = Variables()
        with (
            patch("memorylayer_saas.dependencies.register_package_plugins"),
            patch("memorylayer_saas.dependencies._supersede_oss_routers"),
        ):
            _enterprise_preconfigure_hook(v)

        assert v.get(MEMORYLAYER_CHAT_SERVICE) == "enterprise"

    def test_enterprise_plugin_enabled_default_disabled(self):
        from memorylayer_server.config import MEMORYLAYER_CHAT_SERVICE
        from memorylayer_server.services.chat.default import DefaultChatServicePlugin
        from scitrera_app_framework import Variables

        from memorylayer_saas.services.chat.service import EnterpriseChatServicePlugin

        v = Variables()
        v.set_default_value(MEMORYLAYER_CHAT_SERVICE, "enterprise")

        assert EnterpriseChatServicePlugin().is_enabled(v) is True
        assert DefaultChatServicePlugin().is_enabled(v) is False
