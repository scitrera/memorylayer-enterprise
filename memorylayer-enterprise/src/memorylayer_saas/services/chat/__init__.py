"""Enterprise chat service package.

Ships an :class:`EnterpriseChatService` (provider 'enterprise') that extends the
OSS default chat service with dark, flag-gated auto-titling and title-change
broadcasts. See ``service.py`` and ``title_events.py``.
"""

from .service import (
    CHAT_TITLE_TASK,
    EnterpriseChatService,
    EnterpriseChatServicePlugin,
)
from .title_events import (
    RENAME_CONTROL_KIND,
    broadcast_thread_title_changed,
)

__all__ = (
    "CHAT_TITLE_TASK",
    "EnterpriseChatService",
    "EnterpriseChatServicePlugin",
    "RENAME_CONTROL_KIND",
    "broadcast_thread_title_changed",
)
