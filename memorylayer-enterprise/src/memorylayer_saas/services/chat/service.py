# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Enterprise chat service — auto-titling extension over the OSS default.

Behaviour is identical to :class:`DefaultChatService` except for two dark
(flag-gated) additions:

- ``append_messages`` schedules a ``chat_thread_title`` background task when the
  running message count crosses a configured checkpoint.
- ``update_thread`` marks manual title renames (``title_source='user'``) and
  broadcasts a live title-change event to the thread owner.

Both additions are wrapped so a failure never breaks the underlying append/update.
The always-on selection of this provider is therefore safe: with
``MEMORYLAYER_CHAT_TITLE_ENABLED=False`` (the default) it behaves exactly like
the OSS default service.
"""

import logging

from memorylayer_server.models.chat import (
    AppendMessagesInput,
    ChatMessage,
    ChatThread,
)
from memorylayer_server.services._constants import EXT_STORAGE_BACKEND, EXT_TASK_SERVICE
from memorylayer_server.services.chat.base import ChatServicePluginBase
from memorylayer_server.services.chat.default import DefaultChatService
from memorylayer_server.services.storage import StorageBackend
from memorylayer_server.services.tasks import TaskService
from scitrera_app_framework import Variables, ext_parse_bool, get_extension

from ...config import (
    DEFAULT_MEMORYLAYER_CHAT_TITLE_CHECKPOINTS,
    DEFAULT_MEMORYLAYER_CHAT_TITLE_ENABLED,
    DEFAULT_MEMORYLAYER_CHAT_TITLE_INTERVAL,
    MEMORYLAYER_CHAT_TITLE_CHECKPOINTS,
    MEMORYLAYER_CHAT_TITLE_ENABLED,
    MEMORYLAYER_CHAT_TITLE_INTERVAL,
)
from .title_events import broadcast_thread_title_changed

CHAT_TITLE_TASK = "chat_thread_title"


class EnterpriseChatService(DefaultChatService):
    """DefaultChatService plus dark auto-titling + title-change broadcasts."""

    # ------------------------------------------------------------------
    # Config accessors
    # ------------------------------------------------------------------
    @property
    def _title_enabled(self) -> bool:
        return self._v.environ(
            MEMORYLAYER_CHAT_TITLE_ENABLED,
            DEFAULT_MEMORYLAYER_CHAT_TITLE_ENABLED,
            type_fn=ext_parse_bool,
        )

    @property
    def _title_checkpoints(self) -> set[int]:
        raw = self._v.environ(
            MEMORYLAYER_CHAT_TITLE_CHECKPOINTS,
            DEFAULT_MEMORYLAYER_CHAT_TITLE_CHECKPOINTS,
        )
        checkpoints: set[int] = set()
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                checkpoints.add(int(part))
            except ValueError:
                self.logger.warning("Ignoring invalid chat-title checkpoint %r", part)
        return checkpoints

    @property
    def _title_interval(self) -> int:
        return self._v.environ(
            MEMORYLAYER_CHAT_TITLE_INTERVAL,
            DEFAULT_MEMORYLAYER_CHAT_TITLE_INTERVAL,
            type_fn=int,
        )

    # ------------------------------------------------------------------
    # append_messages: schedule titling on checkpoint crossings
    # ------------------------------------------------------------------
    async def append_messages(
        self,
        workspace_id: str,
        thread_id: str,
        input: AppendMessagesInput,
        tenant_id: str = "",
        user_id: str | None = None,
    ) -> list[ChatMessage]:
        # Only pay for the pre-append snapshot when titling is enabled — this is
        # a hot path and the enterprise chat service is the default provider for
        # all deployments, so the disabled path must stay zero-cost. A missing
        # thread here means it will be auto-created by super() at count 0.
        thread_before = (
            await self.get_thread(workspace_id, thread_id, user_id=user_id)
            if self._title_enabled
            else None
        )
        result = await super().append_messages(
            workspace_id, thread_id, input, tenant_id=tenant_id, user_id=user_id
        )

        if self._title_enabled:
            try:
                await self._maybe_schedule_titling(workspace_id, thread_id, thread_before, len(result))
            except Exception as e:  # noqa: BLE001 - titling must never break append
                self.logger.warning("Failed to schedule titling for thread %s: %s", thread_id, e)

        return result

    async def _maybe_schedule_titling(
        self,
        workspace_id: str,
        thread_id: str,
        thread: ChatThread | None,
        appended: int,
    ) -> None:
        """Schedule a title (re)generation task if a checkpoint falls in the
        just-crossed count range ``(old_count, new_count]``.

        Handles batched appends that jump several counts at once: fires once if
        ANY explicit checkpoint or interval multiple lands inside the range.
        """
        # The reserved _default thread is never auto-titled — it's the implicit
        # per-workspace thread the UI shows without a title. The id is now stored
        # verbatim (owner scoping is a column, not part of the id), so compare it
        # directly.
        if thread_id == "_default":
            return
        if not self._title_enabled:
            return
        if appended <= 0:
            return

        old_count = thread.message_count if thread else 0
        new_count = old_count + appended

        if not self._crosses_checkpoint(old_count, new_count):
            return

        try:
            await self.task_service.schedule_task(
                CHAT_TITLE_TASK,
                {
                    "workspace_id": workspace_id,
                    "thread_id": thread_id,
                    "user_id": thread.user_id if thread else None,
                    "at_index": new_count,
                },
            )
            self.logger.info(
                "Scheduled chat-title task for thread %s at index %d",
                thread_id,
                new_count,
            )
        except Exception as e:  # noqa: BLE001 - scheduling must never break append
            self.logger.warning("Failed to schedule chat-title task for thread %s: %s", thread_id, e)

    def _crosses_checkpoint(self, old_count: int, new_count: int) -> bool:
        """True if any explicit checkpoint OR interval multiple is in (old, new]."""
        for checkpoint in self._title_checkpoints:
            if old_count < checkpoint <= new_count:
                return True

        interval = self._title_interval
        if interval > 0:
            # Smallest positive multiple of ``interval`` strictly greater than old.
            first_multiple = (old_count // interval + 1) * interval
            if first_multiple <= new_count:
                return True

        return False

    # ------------------------------------------------------------------
    # update_thread: mark manual renames + broadcast
    # ------------------------------------------------------------------
    async def update_thread(
        self,
        workspace_id: str,
        thread_id: str,
        user_id: str | None = None,
        **updates,
    ) -> ChatThread | None:
        # A ``title`` in updates on THIS path is a human/API rename — the auto
        # task writes titles via storage directly, bypassing the chat service.
        # Mark it as user-sourced (merged, not replacing existing metadata) so
        # review tasks respect the override.
        manual_title_rename = "title" in updates
        if manual_title_rename:
            existing = await self.get_thread(workspace_id, thread_id, user_id=user_id)
            merged_meta = dict(existing.metadata) if (existing and existing.metadata) else {}
            incoming_meta = updates.get("metadata")
            if incoming_meta:
                merged_meta.update(incoming_meta)
            merged_meta["title_source"] = "user"
            updates["metadata"] = merged_meta

        result = await super().update_thread(workspace_id, thread_id, user_id=user_id, **updates)

        if result is not None and manual_title_rename:
            try:
                await broadcast_thread_title_changed(self._v, result)
            except Exception as e:  # noqa: BLE001 - broadcast must never break rename
                self.logger.warning(
                    "Failed to broadcast manual title change for thread %s: %s",
                    thread_id,
                    e,
                )

        return result


class EnterpriseChatServicePlugin(ChatServicePluginBase):
    """Plugin registering the enterprise chat service under provider 'enterprise'."""

    PROVIDER_NAME = "enterprise"

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        task_service: TaskService = get_extension(EXT_TASK_SERVICE, v)
        return EnterpriseChatService(storage=storage, task_service=task_service, v=v)
