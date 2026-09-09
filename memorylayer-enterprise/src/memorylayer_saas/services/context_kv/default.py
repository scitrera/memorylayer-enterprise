"""
Context Service - Manages logical groupings within workspaces.

Contexts (formerly MemorySpaces) provide logical boundaries for memories
within a workspace. Each workspace has a _default context.
"""
from typing import Optional, List
from logging import Logger

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from memorylayer_server.models.workspace import Context
from memorylayer_server.services.storage.base import StorageBackend, EXT_STORAGE_BACKEND
from .base import (
    ContextService,
    ContextServicePluginBase,
    DEFAULT_CONTEXT_ID,
    DEFAULT_CONTEXT_NAME,
)


class DefaultContextService(ContextService):
    """Default context service implementation using storage backend."""

    def __init__(self, storage: StorageBackend, v: Variables = None):
        self._storage = storage
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized DefaultContextService")

    async def create_context(self, workspace_id: str, context: Context) -> Context:
        """Create a new context within a workspace."""
        self.logger.info(
            "Creating context: %s in workspace: %s",
            context.name,
            workspace_id
        )

        # Ensure workspace_id matches
        if context.workspace_id != workspace_id:
            context = Context(
                id=context.id,
                workspace_id=workspace_id,
                name=context.name,
                description=context.description,
                settings=context.settings,
                created_at=context.created_at,
            )

        # Use storage backend
        created = await self._storage.create_context(workspace_id, context)
        self.logger.info("Created context: %s", created.id)
        return created

    async def get_context(self, workspace_id: str, context_id: str) -> Optional[Context]:
        """Get context by ID within a workspace."""
        self.logger.debug("Getting context: %s in workspace: %s", context_id, workspace_id)
        return await self._storage.get_context(workspace_id, context_id)

    async def get_context_by_name(self, workspace_id: str, name: str) -> Optional[Context]:
        """Get context by name within a workspace."""
        self.logger.debug("Getting context by name: %s in workspace: %s", name, workspace_id)
        # List all contexts and find by name
        contexts = await self._storage.list_contexts(workspace_id)
        for ctx in contexts:
            if ctx.name == name:
                return ctx
        return None

    async def list_contexts(self, workspace_id: str) -> List[Context]:
        """List all contexts in a workspace."""
        self.logger.debug("Listing contexts for workspace: %s", workspace_id)
        return await self._storage.list_contexts(workspace_id)

    async def ensure_default_context(self, workspace_id: str) -> Context:
        """Ensure the _default context exists for a workspace."""
        self.logger.debug("Ensuring _default context for workspace: %s", workspace_id)

        # Check if _default context already exists
        existing = await self.get_context_by_name(workspace_id, DEFAULT_CONTEXT_NAME)
        if existing:
            return existing

        # Create _default context
        default_context = Context(
            id=f"{workspace_id}:{DEFAULT_CONTEXT_ID}",
            workspace_id=workspace_id,
            name=DEFAULT_CONTEXT_NAME,
            description="Default context for the workspace",
            settings={}
        )

        created = await self.create_context(workspace_id, default_context)
        self.logger.info("Created _default context for workspace: %s", workspace_id)
        return created

    async def delete_context(self, workspace_id: str, context_id: str) -> bool:
        """Delete a context. Cannot delete _default context."""
        self.logger.info("Deleting context: %s in workspace: %s", context_id, workspace_id)

        # Get context to check if it's the default
        context = await self.get_context(workspace_id, context_id)
        if context and context.name == DEFAULT_CONTEXT_NAME:
            self.logger.warning("Cannot delete _default context")
            raise ValueError("Cannot delete the _default context")

        # Delegate to the storage backend. The enterprise PostgreSQL store
        # implements delete_context; backends without it (e.g. the OSS SQLite
        # store, which does not declare it on StorageBackend) must not silently
        # report success/failure — surface the gap instead of lying.
        delete_fn = getattr(self._storage, 'delete_context', None)
        if delete_fn is None:
            self.logger.error(
                "Storage backend %s does not implement delete_context",
                type(self._storage).__name__,
            )
            raise NotImplementedError(
                "Storage backend does not support delete_context"
            )

        return await delete_fn(workspace_id, context_id)


class DefaultContextServicePlugin(ContextServicePluginBase):
    """Default context service plugin."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> ContextService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        return DefaultContextService(storage=storage, v=v)
