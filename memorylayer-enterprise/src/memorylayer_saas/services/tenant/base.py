"""
Tenant Service Base - Plugin interface and ABC.

This module defines the extension point and plugin base for tenant services.
Moved from OSS to enterprise as tenant service is only needed for multi-tenant deployments.
"""
from abc import ABC, abstractmethod
from typing import Optional

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from memorylayer_saas.models.tenant import Tenant

from ...config import MEMORYLAYER_TENANT_SERVICE, DEFAULT_MEMORYLAYER_TENANT_SERVICE

# Extension point constant
EXT_TENANT_SERVICE = 'memorylayer-tenant-service'


class TenantService(ABC):
    """Interface for tenant service."""

    @abstractmethod
    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        pass

    @abstractmethod
    async def get_default_tenant(self) -> Tenant:
        """Get the default tenant (OSS mode)."""
        pass

    @abstractmethod
    async def ensure_default_tenant(self) -> Tenant:
        """Ensure the default tenant exists, creating if necessary."""
        pass


# noinspection PyAbstractClass
class TenantServicePluginBase(Plugin):
    """Base plugin for tenant service."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_TENANT_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_TENANT_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_TENANT_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_TENANT_SERVICE, DEFAULT_MEMORYLAYER_TENANT_SERVICE)
