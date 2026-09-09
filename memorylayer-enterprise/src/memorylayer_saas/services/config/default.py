# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""
Config Service - Handles configuration cascade logic.

Resolves settings through the cascade:
Request → Context → Workspace → Tenant → Server defaults

This service centralizes configuration resolution so that all services
use consistent cascade logic.
"""
from logging import Logger
from typing import Optional, Any, Dict

from scitrera_app_framework import Variables

from .base import (
    ConfigServiceBase, ConfigServicePluginBase, DEFAULT_SERVER_SETTINGS,
    Tenant, Workspace, WorkspaceSettings, Context, ContextSettings
)


class ConfigService(ConfigServiceBase):
    """Service for resolving cascaded configuration.

    Configuration cascades from most specific to least specific:
    1. Request-level (explicit API call parameters)
    2. Context-level (ContextSettings)
    3. Workspace-level (WorkspaceSettings)
    4. Tenant-level (TenantSettings)
    5. Server defaults (DEFAULT_SERVER_SETTINGS)
    """

    def __init__(self, server_defaults: Optional[Dict[str, Any]] = None):
        """Initialize config service with optional custom server defaults."""
        self.server_defaults = {**DEFAULT_SERVER_SETTINGS, **(server_defaults or {})}

    def _get_setting_from_dict_or_model(
            self,
            settings: Any,
            key: str
    ) -> Optional[Any]:
        """Extract a setting from either a dict or a Pydantic model."""
        if settings is None:
            return None

        if isinstance(settings, dict):
            return settings.get(key)

        # Pydantic model - use getattr
        return getattr(settings, key, None)

    def get_session_auto_commit(
            self,
            request_value: Optional[bool] = None,
            context: Optional[Context] = None,
            workspace: Optional[Workspace] = None,
            tenant: Optional[Tenant] = None
    ) -> bool:
        """Resolve session_auto_commit through cascade.

        Args:
            request_value: Explicit request-level override
            context: Context for context-level settings
            workspace: Workspace for workspace-level settings
            tenant: Tenant for tenant-level settings

        Returns:
            Resolved session_auto_commit value
        """
        # Level 1: Request override
        if request_value is not None:
            return request_value

        # Level 2: Context settings
        if context:
            val = self._get_setting_from_dict_or_model(context.settings, 'session_auto_commit')
            if val is not None:
                return val

        # Level 3: Workspace settings
        if workspace:
            val = self._get_setting_from_dict_or_model(workspace.settings, 'session_auto_commit')
            if val is not None:
                return val

        # Level 4: Tenant settings
        if tenant and tenant.settings:
            if tenant.settings.session_auto_commit is not None:
                return tenant.settings.session_auto_commit

        # Level 5: Server defaults
        return self.server_defaults.get('session_auto_commit', True)

    def get_session_default_ttl(
            self,
            request_value: Optional[int] = None,
            context: Optional[Context] = None,
            workspace: Optional[Workspace] = None,
            tenant: Optional[Tenant] = None
    ) -> int:
        """Resolve session TTL through cascade.

        Args:
            request_value: Explicit request-level override
            context: Context for context-level settings
            workspace: Workspace for workspace-level settings
            tenant: Tenant for tenant-level settings (not used for TTL currently)

        Returns:
            Resolved session TTL in seconds
        """
        # Level 1: Request override
        if request_value is not None:
            return request_value

        # Level 2: Context settings
        if context:
            val = self._get_setting_from_dict_or_model(context.settings, 'session_default_ttl')
            if val is not None:
                return val

        # Level 3: Workspace settings
        if workspace:
            val = self._get_setting_from_dict_or_model(workspace.settings, 'session_default_ttl')
            if val is not None:
                return val

        # Level 5: Server defaults
        return self.server_defaults.get('session_default_ttl', 3600)

    def get_include_global(
            self,
            request_value: Optional[bool] = None,
            context: Optional[Context] = None,
            workspace: Optional[Workspace] = None
    ) -> bool:
        """Resolve include_global setting.

        Controls whether memories from _global workspace are included in recall.

        Args:
            request_value: Explicit request-level override
            context: Context for context-level settings
            workspace: Workspace for workspace-level settings

        Returns:
            Resolved include_global value
        """
        # Level 1: Request override
        if request_value is not None:
            return request_value

        # Level 2: Context settings
        if context:
            val = self._get_setting_from_dict_or_model(context.settings, 'include_global')
            if val is not None:
                return val

        # Level 3: Workspace settings
        if workspace:
            val = self._get_setting_from_dict_or_model(workspace.settings, 'include_global')
            if val is not None:
                return val

        # Level 5: Server defaults
        return self.server_defaults.get('include_global', True)

    def get_default_importance(
            self,
            request_value: Optional[float] = None,
            context: Optional[Context] = None,
            workspace: Optional[Workspace] = None
    ) -> float:
        """Resolve default_importance for memories.

        Args:
            request_value: Explicit request-level override
            context: Context for context-level settings
            workspace: Workspace for workspace-level settings

        Returns:
            Resolved default_importance value (0.0-1.0)
        """
        # Level 1: Request override
        if request_value is not None:
            return request_value

        # Level 2: Context settings
        if context:
            val = self._get_setting_from_dict_or_model(context.settings, 'default_importance')
            if val is not None:
                return val

        # Level 3: Workspace settings
        if workspace:
            val = self._get_setting_from_dict_or_model(workspace.settings, 'default_importance')
            if val is not None:
                return val

        # Level 5: Server defaults
        return self.server_defaults.get('default_importance', 0.5)

    def get_decay_enabled(
            self,
            request_value: Optional[bool] = None,
            context: Optional[Context] = None,
            workspace: Optional[Workspace] = None
    ) -> bool:
        """Resolve decay_enabled setting.

        Args:
            request_value: Explicit request-level override
            context: Context for context-level settings
            workspace: Workspace for workspace-level settings

        Returns:
            Resolved decay_enabled value
        """
        # Level 1: Request override
        if request_value is not None:
            return request_value

        # Level 2: Context settings
        if context:
            val = self._get_setting_from_dict_or_model(context.settings, 'decay_enabled')
            if val is not None:
                return val

        # Level 3: Workspace settings
        if workspace:
            val = self._get_setting_from_dict_or_model(workspace.settings, 'decay_enabled')
            if val is not None:
                return val

        # Level 5: Server defaults
        return self.server_defaults.get('decay_enabled', True)

    def get_decay_rate(
            self,
            request_value: Optional[float] = None,
            context: Optional[Context] = None,
            workspace: Optional[Workspace] = None
    ) -> float:
        """Resolve decay_rate setting.

        Args:
            request_value: Explicit request-level override
            context: Context for context-level settings
            workspace: Workspace for workspace-level settings

        Returns:
            Resolved decay_rate value (0.0-1.0 daily rate)
        """
        # Level 1: Request override
        if request_value is not None:
            return request_value

        # Level 2: Context settings
        if context:
            val = self._get_setting_from_dict_or_model(context.settings, 'decay_rate')
            if val is not None:
                return val

        # Level 3: Workspace settings
        if workspace:
            val = self._get_setting_from_dict_or_model(workspace.settings, 'decay_rate')
            if val is not None:
                return val

        # Level 5: Server defaults
        return self.server_defaults.get('decay_rate', 0.01)

    def resolve_workspace_settings(
            self,
            workspace: Workspace,
            tenant: Optional[Tenant] = None
    ) -> WorkspaceSettings:
        """Resolve full WorkspaceSettings with tenant defaults applied.

        Args:
            workspace: Workspace to resolve settings for
            tenant: Optional tenant for defaults

        Returns:
            Fully resolved WorkspaceSettings
        """
        # Start with server defaults
        resolved = WorkspaceSettings()

        # Apply tenant defaults if available
        if tenant and tenant.settings:
            resolved.session_auto_commit = tenant.settings.session_auto_commit

        # Apply workspace settings
        ws_settings = workspace.settings
        if isinstance(ws_settings, dict):
            for key, value in ws_settings.items():
                if hasattr(resolved, key) and value is not None:
                    setattr(resolved, key, value)
        elif isinstance(ws_settings, WorkspaceSettings):
            resolved = ws_settings

        return resolved

    def resolve_context_settings(
            self,
            context: Context,
            workspace: Optional[Workspace] = None,
            tenant: Optional[Tenant] = None
    ) -> ContextSettings:
        """Resolve full ContextSettings with inheritance.

        Args:
            context: Context to resolve settings for
            workspace: Optional workspace for inheritance
            tenant: Optional tenant for defaults

        Returns:
            Fully resolved ContextSettings
        """
        ctx_settings = context.settings
        if isinstance(ctx_settings, dict):
            resolved = ContextSettings(**{
                k: v for k, v in ctx_settings.items()
                if k in ContextSettings.model_fields
            })
        elif isinstance(ctx_settings, ContextSettings):
            resolved = ctx_settings
        else:
            resolved = ContextSettings()

        # If inherit_workspace_settings is True, apply workspace settings for unset values
        if resolved.inherit_workspace_settings and workspace:
            ws_settings = self.resolve_workspace_settings(workspace, tenant)

            if resolved.auto_remember_enabled is None:
                resolved.auto_remember_enabled = ws_settings.auto_remember_enabled
            if resolved.decay_enabled is None:
                resolved.decay_enabled = ws_settings.decay_enabled
            if resolved.default_importance is None:
                resolved.default_importance = ws_settings.default_importance
            if resolved.session_auto_commit is None:
                resolved.session_auto_commit = ws_settings.session_auto_commit

        return resolved


class DefaultConfigServicePlugin(ConfigServicePluginBase):
    """Default config service plugin."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> ConfigService:
        # Get server defaults from variables if available
        server_defaults = v.get('config_server_defaults', None)
        return ConfigService(server_defaults)
