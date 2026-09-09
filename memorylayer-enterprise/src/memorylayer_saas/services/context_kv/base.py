from abc import ABC, abstractmethod
from typing import Optional, List

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_CONTEXT_SERVICE, DEFAULT_MEMORYLAYER_CONTEXT_SERVICE
from memorylayer_server.models.workspace import Context
from memorylayer_server.services.storage.base import EXT_STORAGE_BACKEND

# Extension point constant
EXT_CONTEXT_SERVICE = 'memorylayer-context-service'

# Default context constants.
#
# "_default" is the sentinel for the (currently unused) context_id feature. It
# is still written NOT-NULL on sessions / chat_threads / documents / datasets
# (no FK there). For memories — which DO have an FK to contexts(id) — the write
# paths persist NULL instead of this sentinel, so no contexts row is required.
# context_id is RESERVED for future cross-concern filtering and is NOT used as a
# live retrieval filter today. See MemoryModel.context_id (storage/models.py)
# for the full rationale before extending or removing any of this.
DEFAULT_CONTEXT_ID = "_default"
DEFAULT_CONTEXT_NAME = "_default"


class ContextService(ABC):
    """Interface for context service."""

    @abstractmethod
    async def create_context(self, workspace_id: str, context: Context) -> Context:
        """Create a new context within a workspace."""
        pass

    @abstractmethod
    async def get_context(self, workspace_id: str, context_id: str) -> Optional[Context]:
        """Get context by ID within a workspace."""
        pass

    @abstractmethod
    async def get_context_by_name(self, workspace_id: str, name: str) -> Optional[Context]:
        """Get context by name within a workspace."""
        pass

    @abstractmethod
    async def list_contexts(self, workspace_id: str) -> List[Context]:
        """List all contexts in a workspace."""
        pass

    @abstractmethod
    async def ensure_default_context(self, workspace_id: str) -> Context:
        """Ensure the _default context exists for a workspace, creating if necessary."""
        pass

    @abstractmethod
    async def delete_context(self, workspace_id: str, context_id: str) -> bool:
        """Delete a context. Cannot delete _default context."""
        pass


# noinspection PyAbstractClass
class ContextServicePluginBase(Plugin):
    """Base plugin for context service."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_CONTEXT_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_CONTEXT_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_CONTEXT_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_CONTEXT_SERVICE, DEFAULT_MEMORYLAYER_CONTEXT_SERVICE)

    def get_dependencies(self, v: Variables):
        return (EXT_STORAGE_BACKEND,)
