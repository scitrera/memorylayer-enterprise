# TODO: this file should not exist; all plugins should be divided among their service/api packages
"""Enterprise plugin definitions for MemoryLayer SaaS.

This module defines plugins for:
- PostgreSQL storage backend with cold tier support
- Enhanced memory service with hot/cold tier recall
- Tiering service for automatic memory archival
- Tiering API endpoints
"""
# PostgreSQLStoragePlugin moved to storage/postgresql.py
# Import it here for backward compatibility
from .storage.postgresql import (
    PostgreSQLStoragePlugin,
    MEMORYLAYER_POSTGRESQL_URL,
    DEFAULT_MEMORYLAYER_POSTGRESQL_URL,
    MEMORYLAYER_POSTGRESQL_POOL_SIZE,
    DEFAULT_MEMORYLAYER_POSTGRESQL_POOL_SIZE,
)

# MemoryServiceSaaSPlugin moved to services/enterprise_memory/default.py
# Import it here for backward compatibility
from .services.enterprise_memory import EnterpriseMemoryServicePlugin as MemoryServiceSaaSPlugin

# TieringServicePlugin moved to services/tiering/default.py
# Import it here for backward compatibility
from .services.tiering import TieringServicePlugin, get_tiering_service, EXT_TIERING_SERVICE

# TieringAPIPlugin moved to api/v1/tiering.py
# Import it here for backward compatibility
from .api.v1.tiering import TieringAPIPlugin

__all__ = (
    'PostgreSQLStoragePlugin',
    'MemoryServiceSaaSPlugin',
    'TieringServicePlugin',
    'TieringAPIPlugin',
    'EXT_TIERING_SERVICE',
    'get_tiering_service',
    'MEMORYLAYER_POSTGRESQL_URL',
    'DEFAULT_MEMORYLAYER_POSTGRESQL_URL',
    'MEMORYLAYER_POSTGRESQL_POOL_SIZE',
    'DEFAULT_MEMORYLAYER_POSTGRESQL_POOL_SIZE',
)
