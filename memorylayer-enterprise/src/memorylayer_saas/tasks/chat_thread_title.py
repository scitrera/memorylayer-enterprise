"""Chat-thread auto-title task handler (enterprise).

Background task that (re)generates a short LLM display title for a chat thread
and broadcasts the change to the thread owner. Scheduled by the enterprise chat
service as the running message count crosses configured checkpoints.

Guards:
- Respects manual renames (``metadata.title_source == 'user'``): skips.
- Ignores stale tasks (``metadata.title_index > at_index``): a newer title already
  exists; last-write-wins under concurrent workers.
- Only persists + broadcasts when the new title MEANINGFULLY differs from the
  current one (normalized compare), so review churn does not spam updates.

Ships dark: nothing schedules this task unless MEMORYLAYER_CHAT_TITLE_ENABLED is set.
"""

import re
from logging import Logger

from memorylayer_server.models.generation import GenerationActivity
from memorylayer_server.services.llm import get_llm_service
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND, StorageBackend
from memorylayer_server.services.tasks import TaskHandlerPlugin, TaskSchedule
from scitrera_app_framework import Variables, get_extension, get_logger

from memorylayer_saas.config import (
    DEFAULT_MEMORYLAYER_CHAT_TITLE_MAX_MESSAGES,
    DEFAULT_MEMORYLAYER_CHAT_TITLE_MAX_TOKENS,
    DEFAULT_MEMORYLAYER_CHAT_TITLE_PROMPT,
    MEMORYLAYER_CHAT_TITLE_MAX_MESSAGES,
    MEMORYLAYER_CHAT_TITLE_MAX_TOKENS,
    MEMORYLAYER_CHAT_TITLE_PROMPT,
)
from memorylayer_saas.services.chat.title_events import broadcast_thread_title_changed

CHAT_TITLE_TASK = "chat_thread_title"

# Hard cap on persisted title length regardless of LLM verbosity.
_MAX_TITLE_LEN = 120


def _render_messages(messages: list) -> str:
    """Render messages as ``role: text`` lines for the titling prompt.

    Mirrors the chat-decomposition handler's handling of structured (non-str)
    message content so image/tool blocks degrade gracefully to text.
    """
    lines: list[str] = []
    for msg in messages:
        content = msg.content
        if not isinstance(content, str):
            parts: list[str] = []
            for block in content or []:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
                elif hasattr(block, "type"):
                    parts.append(f"[{block.type}]")
            content = " ".join(parts) if parts else "[structured content]"
        lines.append(f"{msg.role}: {content}")
    return "\n".join(lines)


def _sanitize_title(raw: str) -> str:
    """Strip surrounding quotes/markdown fences, collapse whitespace, clamp length."""
    if not raw:
        return ""
    text = raw.strip()
    # Strip a fenced code block wrapper if the model emitted one.
    if text.startswith("```"):
        text = text.strip("`").strip()
    # Drop a leading markdown heading marker (e.g. "# Title").
    text = re.sub(r"^#+\s*", "", text)
    # Strip a single layer of surrounding quotes.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("\"", "'"):
        text = text[1:-1]
    # Collapse all internal whitespace to single spaces.
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > _MAX_TITLE_LEN:
        text = text[:_MAX_TITLE_LEN].rstrip()
    return text


def _normalize(text: str) -> str:
    """Normalize for meaningful-drift comparison: casefold + collapse ws/punct."""
    if not text:
        return ""
    lowered = text.casefold().strip()
    # Punctuation -> space, then collapse runs of whitespace.
    collapsed = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", collapsed).strip()


class ChatThreadTitleTaskHandler(TaskHandlerPlugin):
    """On-demand handler that generates and persists a chat-thread title."""

    def get_task_type(self) -> str:
        return CHAT_TITLE_TASK

    def get_schedule(self, v: Variables) -> TaskSchedule | None:
        return None  # scheduled on demand by the chat service

    async def handle(self, v: Variables, payload: dict) -> None:
        logger: Logger = get_logger(v, name=self.get_task_type())
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)

        workspace_id = payload.get("workspace_id")
        thread_id = payload.get("thread_id")
        at_index = payload.get("at_index", 0)
        # Owner scope: threads are keyed by (workspace_id, user_id, id) — required
        # so a shared client id resolves to the correct owner's thread.
        user_id = payload.get("user_id")

        if not workspace_id or not thread_id:
            logger.warning(
                "Missing required payload fields: workspace_id=%s, thread_id=%s",
                workspace_id,
                thread_id,
            )
            return

        thread = await storage.get_thread(workspace_id, thread_id, user_id=user_id)
        if not thread:
            logger.debug("Thread %s not found in workspace %s; skipping titling", thread_id, workspace_id)
            return

        metadata = thread.metadata or {}

        # Respect manual renames.
        if metadata.get("title_source") == "user":
            logger.debug("Thread %s title is user-set; skipping auto-title", thread_id)
            return

        # Ignore stale tasks — a newer title already won.
        if metadata.get("title_index", -1) > at_index:
            logger.debug(
                "Stale title task for thread %s (title_index=%s > at_index=%s); skipping",
                thread_id,
                metadata.get("title_index"),
                at_index,
            )
            return

        max_messages = v.environ(
            MEMORYLAYER_CHAT_TITLE_MAX_MESSAGES,
            DEFAULT_MEMORYLAYER_CHAT_TITLE_MAX_MESSAGES,
            type_fn=int,
        )
        messages = await storage.get_messages(
            workspace_id=workspace_id,
            thread_id=thread_id,
            limit=max_messages,
            order="asc",
            user_id=user_id,
        )
        if not messages:
            logger.debug("No messages to title for thread %s", thread_id)
            return

        prompt_template = v.environ(
            MEMORYLAYER_CHAT_TITLE_PROMPT,
            DEFAULT_MEMORYLAYER_CHAT_TITLE_PROMPT,
        )
        max_tokens = v.environ(
            MEMORYLAYER_CHAT_TITLE_MAX_TOKENS,
            DEFAULT_MEMORYLAYER_CHAT_TITLE_MAX_TOKENS,
            type_fn=int,
        )

        convo = _render_messages(messages)
        # Honor an optional "{convo}" placeholder for operator overrides; otherwise
        # append the rendered conversation after the instruction (robust default).
        if "{convo}" in prompt_template:
            prompt = prompt_template.format(convo=convo)
        else:
            prompt = f"{prompt_template}\n\n{convo}"

        try:
            raw_title = await get_llm_service(v).synthesize(
                prompt,
                max_tokens=max_tokens,
                profile="titling",
                activity=GenerationActivity.SYNTHESIS,
            )
        except Exception as e:  # noqa: BLE001 - LLM failure is non-fatal for a title
            logger.warning("LLM title synthesis failed for thread %s: %s", thread_id, e)
            return

        new_title = _sanitize_title(raw_title or "")
        if not new_title:
            logger.debug("Empty title synthesized for thread %s; skipping", thread_id)
            return

        # Meaningful-drift gate. The first checkpoint always writes because the
        # placeholder title equals the thread_id.
        if _normalize(new_title) == _normalize(thread.title or ""):
            logger.debug("Title unchanged for thread %s; skipping persist/broadcast", thread_id)
            return

        # storage.update_thread(metadata=...) REPLACES the JSONB, so pass a fully
        # merged dict (preserve unrelated keys, stamp source + index).
        merged_metadata = {**metadata, "title_source": "auto", "title_index": at_index}
        updated = await storage.update_thread(
            workspace_id,
            thread_id,
            user_id=user_id,
            title=new_title,
            metadata=merged_metadata,
        )
        if updated is None:
            logger.warning("Failed to persist auto-title for thread %s", thread_id)
            return

        logger.info("Auto-titled thread %s -> %r (at_index=%s)", thread_id, new_title, at_index)
        await broadcast_thread_title_changed(v, updated)
