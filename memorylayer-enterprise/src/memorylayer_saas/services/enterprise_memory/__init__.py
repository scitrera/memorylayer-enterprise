# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""
Memory Service SaaS - Extended memory service with cold tier support.

Extends the base MemoryService to seamlessly search both hot and cold tiers
during memory recall operations. When hot tier results are insufficient,
automatically searches cold tier to find additional relevant memories.

Key features:
- Seamless hot and cold tier recall
- Configurable cold tier search via TieringConfig
- Automatic result merging and deduplication
- Cold tier results marked with metadata
"""
from .base import (
    EXT_MEMORY_SERVICE,
    MemoryServicePluginBase,
    MEMORYLAYER_MEMORY_SERVICE,
    DEFAULT_MEMORYLAYER_MEMORY_SERVICE,
)
from .default import EnterpriseMemoryService, EnterpriseMemoryServicePlugin

from scitrera_app_framework import Variables, get_extension


def get_enterprise_memory_service(v: Variables = None):
    """Get the enterprise memory service instance."""
    return get_extension(EXT_MEMORY_SERVICE, v)


__all__ = (
    'EnterpriseMemoryService',
    'EnterpriseMemoryServicePlugin',
    'get_enterprise_memory_service',
    'EXT_MEMORY_SERVICE',
    'MemoryServicePluginBase',
    'MEMORYLAYER_MEMORY_SERVICE',
    'DEFAULT_MEMORYLAYER_MEMORY_SERVICE',
)
