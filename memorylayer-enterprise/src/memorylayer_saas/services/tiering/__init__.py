# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

from scitrera_app_framework import Variables

from .base import TieringStats, ArchivalResult, RestoreResult, TieringServicePluginBase, EXT_TIERING_SERVICE
from .default import TieringService, TieringServicePlugin


# Service getter functions
def get_tiering_service(v: Variables = None):
    """Get the tiering service instance."""
    from scitrera_app_framework import get_extension
    return get_extension(EXT_TIERING_SERVICE, v)


__all__ = (
    'TieringService',
    'TieringServicePlugin',
    'TieringStats',
    'ArchivalResult',
    'RestoreResult',
    'EXT_TIERING_SERVICE',
    'TieringServicePluginBase',
    'get_tiering_service',
)
