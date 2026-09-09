"""Tenant service package (enterprise)."""
from .base import (
    TenantService,
    TenantServicePluginBase,
    EXT_TENANT_SERVICE,
)

from scitrera_app_framework import Variables, get_extension


def get_tenant_service(v: Variables = None) -> TenantService:
    """Get the tenant service instance."""
    return get_extension(EXT_TENANT_SERVICE, v)


__all__ = (
    'TenantService',
    'TenantServicePluginBase',
    'get_tenant_service',
    'EXT_TENANT_SERVICE',
)
