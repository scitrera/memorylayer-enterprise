"""Context service package."""
from .base import (
    ContextService,
    ContextServicePluginBase,
    EXT_CONTEXT_SERVICE,
    DEFAULT_CONTEXT_ID,
    DEFAULT_CONTEXT_NAME,
)

from scitrera_app_framework import Variables, get_extension


def get_context_service(v: Variables = None) -> ContextService:
    """Get the context service instance."""
    return get_extension(EXT_CONTEXT_SERVICE, v)


__all__ = (
    'ContextService',
    'ContextServicePluginBase',
    'get_context_service',
    'EXT_CONTEXT_SERVICE',
    'DEFAULT_CONTEXT_ID',
    'DEFAULT_CONTEXT_NAME',
)
