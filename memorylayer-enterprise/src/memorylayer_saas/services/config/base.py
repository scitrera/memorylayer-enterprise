from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from memorylayer_saas.models.tenant import Tenant
from memorylayer_server.models.workspace import Workspace, WorkspaceSettings, Context, ContextSettings

from ...config import MEMORYLAYER_CONFIG_SERVICE, DEFAULT_MEMORYLAYER_CONFIG_SERVICE

# Extension point constant
EXT_CONFIG_SERVICE = 'memorylayer-config-service'

# Default server-level settings
DEFAULT_SERVER_SETTINGS = {
    'session_auto_commit': True,
    'session_default_ttl': 3600,
    'include_global': True,
    'default_importance': 0.5,
    'decay_enabled': True,
    'decay_rate': 0.01,
}


class ConfigServiceBase(ABC):
    """Abstract base class for configuration service.

    Configuration cascades from most specific to least specific:
    1. Request-level (explicit API call parameters)
    2. Context-level (ContextSettings)
    3. Workspace-level (WorkspaceSettings)
    4. Tenant-level (TenantSettings)
    5. Server defaults (DEFAULT_SERVER_SETTINGS)
    """

    @abstractmethod
    def get_session_auto_commit(
            self,
            request_value: Optional[bool] = None,
            context: Optional[Context] = None,
            workspace: Optional[Workspace] = None,
            tenant: Optional[Tenant] = None
    ) -> bool:
        """Resolve session_auto_commit through cascade."""
        pass

    @abstractmethod
    def get_session_default_ttl(
            self,
            request_value: Optional[int] = None,
            context: Optional[Context] = None,
            workspace: Optional[Workspace] = None,
            tenant: Optional[Tenant] = None
    ) -> int:
        """Resolve session TTL through cascade."""
        pass

    @abstractmethod
    def get_include_global(
            self,
            request_value: Optional[bool] = None,
            context: Optional[Context] = None,
            workspace: Optional[Workspace] = None
    ) -> bool:
        """Resolve include_global setting."""
        pass

    @abstractmethod
    def get_default_importance(
            self,
            request_value: Optional[float] = None,
            context: Optional[Context] = None,
            workspace: Optional[Workspace] = None
    ) -> float:
        """Resolve default_importance for memories."""
        pass

    @abstractmethod
    def get_decay_enabled(
            self,
            request_value: Optional[bool] = None,
            context: Optional[Context] = None,
            workspace: Optional[Workspace] = None
    ) -> bool:
        """Resolve decay_enabled setting."""
        pass

    @abstractmethod
    def get_decay_rate(
            self,
            request_value: Optional[float] = None,
            context: Optional[Context] = None,
            workspace: Optional[Workspace] = None
    ) -> float:
        """Resolve decay_rate setting."""
        pass

    @abstractmethod
    def resolve_workspace_settings(
            self,
            workspace: Workspace,
            tenant: Optional[Tenant] = None
    ) -> WorkspaceSettings:
        """Resolve full WorkspaceSettings with tenant defaults applied."""
        pass

    @abstractmethod
    def resolve_context_settings(
            self,
            context: Context,
            workspace: Optional[Workspace] = None,
            tenant: Optional[Tenant] = None
    ) -> ContextSettings:
        """Resolve full ContextSettings with inheritance."""
        pass


# noinspection PyAbstractClass
class ConfigServicePluginBase(Plugin):
    """Base plugin for config service - extensible for custom implementations."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_CONFIG_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_CONFIG_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_CONFIG_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_CONFIG_SERVICE, DEFAULT_MEMORYLAYER_CONFIG_SERVICE)

    def get_dependencies(self, v: Variables):
        return ()
