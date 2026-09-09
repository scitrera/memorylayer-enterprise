# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Config service package."""
from .base import (
    ConfigServicePluginBase,
    ConfigServiceBase,
    EXT_CONFIG_SERVICE,
    DEFAULT_SERVER_SETTINGS,
)
from .default import DefaultConfigServicePlugin, ConfigService

from scitrera_app_framework import Variables, get_extension


def get_config_service(v: Variables = None) -> ConfigService:
    """Get the config service instance.

    Args:
        v: Variables instance (optional, uses global if not provided)

    Returns:
        ConfigService instance
    """
    return get_extension(EXT_CONFIG_SERVICE, v)


__all__ = (
    'ConfigService',
    'ConfigServiceBase',
    'ConfigServicePluginBase',
    'get_config_service',
    'EXT_CONFIG_SERVICE',
    'DefaultConfigServicePlugin',
    'DEFAULT_SERVER_SETTINGS',
)
