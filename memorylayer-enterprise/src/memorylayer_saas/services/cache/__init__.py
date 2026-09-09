"""Cache service package."""
from scitrera_app_framework import Variables, get_extension

from .base import (
    EXT_CACHE_SERVICE,
    CacheServicePluginBase,
    CacheService,
    EnterpriseCacheService,
)


def get_cache_service(v: Variables = None):
    """Get the cache service instance."""
    return get_extension(EXT_CACHE_SERVICE, v)


__all__ = (
    'CacheServicePluginBase',
    'CacheService',
    'EnterpriseCacheService',
    'get_cache_service',
    'EXT_CACHE_SERVICE',
)
