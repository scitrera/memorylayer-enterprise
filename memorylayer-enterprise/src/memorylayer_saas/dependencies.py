# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Enterprise dependency injection and preconfiguration hooks.

This module provides the preconfiguration hooks that register enterprise plugins
on top of the OSS memorylayer-server plugins.

Usage:
    from memorylayer_saas.dependencies import preconfigure, initialize_services, shutdown_services

    # It gets used the same way as the OSS version (and actually IT IS the OSS version) EXCEPT
    # that when we import the enterprise version, we register enterprise plugins and defaults that
    # expands on and overrides the OSS defaults.
"""

from typing import Callable

from memorylayer_server import dependencies
from memorylayer_server.config import (
    MEMORYLAYER_CHAT_SERVICE,
    MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER,
    MEMORYLAYER_GRAPH_QUERY_PROVIDER,
    MEMORYLAYER_TASK_PROVIDER,
)

from scitrera_app_framework import register_package_plugins, Variables


def _add_preconfigure_hook(hook: Callable[[Variables], None]) -> None:
    """ Add a hook to be called during preconfiguration """
    # noinspection PyProtectedMember
    hooks = dependencies._preconfigure_hooks
    if hook not in hooks:
        hooks.append(hook)


def _supersede_oss_routers(v: Variables, plugin_names: list[str]) -> None:
    """Drop named OSS multi-extension API-router plugins from the registry.

    Multi-extension routers are keyed by ``Plugin.name()`` and ALL included by
    the FastAPI route setup; FastAPI then matches the first-registered router
    for a path. Removing a superseded OSS router here (after it registered, but
    before route setup) lets the enterprise router own the shared path. Best
    effort — uses the framework's extension-options registry (``=|EOR|``).
    """
    try:
        from memorylayer_server.api import EXT_MULTI_API_ROUTERS

        eor = v.get('=|EOR|', default=None)
        routers = eor.get(EXT_MULTI_API_ROUTERS, default=None) if eor is not None else None
        if not routers:
            return
        for name in plugin_names:
            routers.pop(name, None)
    except Exception:  # noqa: BLE001 - override is best-effort
        pass


def _enterprise_preconfigure_hook(v: Variables) -> None:
    """Register enterprise plugins and set enterprise defaults."""
    from . import api, services, storage, tasks

    # Register enterprise plugins (recursive discovery)
    register_package_plugins(services.__package__, v, recursive=True)
    register_package_plugins(storage.__package__, v, recursive=True)
    register_package_plugins(api.__package__, v, recursive=True)
    register_package_plugins(tasks.__package__, v, recursive=True)

    # Enterprise overrides OSS where the enterprise ships a full replacement.
    # Multi-extension API routers are ALL included (keyed by plugin name), and
    # FastAPI matches the first-registered router for a path — so the OSS-core
    # router (registered before this hook) would otherwise shadow the
    # enterprise one at the same prefix (e.g. /v1/documents, where only the
    # enterprise pipeline produces document_pages + visual-tokens). Drop the
    # superseded OSS router(s) from the multi-extension registry so the
    # enterprise versions own those paths.
    _supersede_oss_routers(v, [
        "memorylayer_server.api.v1.documents.DocumentsAPIPlugin",
    ])

    # Import configuration constants
    from .config import (
        MEMORYLAYER_TIERING_SERVICE, DEFAULT_MEMORYLAYER_TIERING_SERVICE,
        MEMORYLAYER_TRAJECTORY_SERVICE, DEFAULT_MEMORYLAYER_TRAJECTORY_SERVICE,
        MEMORYLAYER_MEMORY_SERVICE, DEFAULT_MEMORYLAYER_MEMORY_SERVICE,
        MEMORYLAYER_STORAGE_BACKEND, DEFAULT_MEMORYLAYER_STORAGE_BACKEND,
        MEMORYLAYER_COMPRESSION_SERVICE, DEFAULT_MEMORYLAYER_COMPRESSION_SERVICE,
        MEMORYLAYER_CACHE_SERVICE, DEFAULT_MEMORYLAYER_CACHE_SERVICE,
        MEMORYLAYER_EMBEDDING_SERVICE, DEFAULT_MEMORYLAYER_EMBEDDING_SERVICE,
        MEMORYLAYER_EMBEDDING_PROVIDER, DEFAULT_MEMORYLAYER_EMBEDDING_PROVIDER,
        MEMORYLAYER_REFLECT_SERVICE,
        MEMORYLAYER_LLM_SERVICE,
        MEMORYLAYER_LLM_REGISTRY,
        MEMORYLAYER_WORKSPACE_SERVICE,
        MEMORYLAYER_AUTHENTICATION_SERVICE,
        MEMORYLAYER_AUTHORIZATION_SERVICE,
        MEMORYLAYER_WORKSPACE_IMPLICIT_CREATE,
        MEMORYLAYER_REQUIRE_SECURE_AUTH, DEFAULT_MEMORYLAYER_REQUIRE_SECURE_AUTH,
        MEMORYLAYER_ASSOCIATION_SERVICE,
        MEMORYLAYER_SESSION_SERVICE,
        MEMORYLAYER_AUDIT_SERVICE, DEFAULT_MEMORYLAYER_AUDIT_SERVICE,
        MEMORYLAYER_RATE_LIMIT_SERVICE, DEFAULT_MEMORYLAYER_RATE_LIMIT_SERVICE,
        MEMORYLAYER_METRICS_SERVICE, DEFAULT_MEMORYLAYER_METRICS_SERVICE,
        MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS, DEFAULT_MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS,
        MEMORYLAYER_AUTO_KB_UPDATE_ON_INGEST, DEFAULT_MEMORYLAYER_AUTO_KB_UPDATE_ON_INGEST,
        MEMORYLAYER_KNOWLEDGEBASE_PROVIDER,
        MEMORYLAYER_EXTRACTION_SERVICE,
        MEMORYLAYER_ENTITY_REGISTRY_PROVIDER,
        DEFAULT_MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER,
        DEFAULT_MEMORYLAYER_GRAPH_QUERY_PROVIDER,
    )
    from memorylayer_server.services.api_key_store import MEMORYLAYER_API_KEY_STORE

    # Set enterprise defaults -- these override OSS defaults -- ENV configuration will still win
    v.set_default_value(MEMORYLAYER_TIERING_SERVICE, DEFAULT_MEMORYLAYER_TIERING_SERVICE)  # Tiering service - 'default'
    v.set_default_value(MEMORYLAYER_TRAJECTORY_SERVICE, DEFAULT_MEMORYLAYER_TRAJECTORY_SERVICE)  # Trajectory service - 'default'
    v.set_default_value(MEMORYLAYER_MEMORY_SERVICE, DEFAULT_MEMORYLAYER_MEMORY_SERVICE)  # Enterprise Memory Backend
    v.set_default_value(MEMORYLAYER_STORAGE_BACKEND, DEFAULT_MEMORYLAYER_STORAGE_BACKEND)  # Storage service - 'postgresql'
    v.set_default_value(MEMORYLAYER_COMPRESSION_SERVICE, DEFAULT_MEMORYLAYER_COMPRESSION_SERVICE)  # Compression service - 'default'
    v.set_default_value(MEMORYLAYER_CACHE_SERVICE, DEFAULT_MEMORYLAYER_CACHE_SERVICE)  # Cache service - 'redis'
    v.set_default_value(MEMORYLAYER_EMBEDDING_SERVICE, DEFAULT_MEMORYLAYER_EMBEDDING_SERVICE)  # Embedding Service - 'mv'
    v.set_default_value(MEMORYLAYER_EMBEDDING_PROVIDER, DEFAULT_MEMORYLAYER_EMBEDDING_PROVIDER)  # Embedding Provider - 'embed_server'
    v.set_default_value(MEMORYLAYER_AUDIT_SERVICE, DEFAULT_MEMORYLAYER_AUDIT_SERVICE)  # Audit service - 'postgresql'
    v.set_default_value(MEMORYLAYER_RATE_LIMIT_SERVICE, DEFAULT_MEMORYLAYER_RATE_LIMIT_SERVICE)  # Rate limiting - 'aether-kv'
    v.set_default_value(MEMORYLAYER_METRICS_SERVICE, DEFAULT_MEMORYLAYER_METRICS_SERVICE)  # Metrics - 'prometheus'
    v.set_default_value(MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER, DEFAULT_MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER)  # AGE graph analysis
    v.set_default_value(MEMORYLAYER_GRAPH_QUERY_PROVIDER, DEFAULT_MEMORYLAYER_GRAPH_QUERY_PROVIDER)  # AGE graph queries
    v.set_default_value(MEMORYLAYER_TASK_PROVIDER, 'aether')
    # Let first-party LLM traffic carry caller-asserted identity headers so the
    # internal AI gateway can attribute it. Scoped to the gateway's own hosts: OSS
    # defaults this to empty (stamp nothing), and anything not matched here — every
    # public provider — is never told which tenant this pod serves.
    v.set_default_value(
        MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS, DEFAULT_MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS
    )
    # Select the enterprise chat service (adds dark, flag-gated auto-titling on
    # top of the OSS default). Behavior is identical to 'default' until
    # MEMORYLAYER_CHAT_TITLE_ENABLED is set, so always-on selection is safe.
    v.set_default_value(MEMORYLAYER_CHAT_SERVICE, 'enterprise')  # Enterprise chat service (auto-titling)
    # Enterprise KB adds canonical-entity-registry articles on top of the OSS KB.
    # Additive + fail-safe (no registry -> OSS output), so always-on is safe.
    v.set_default_value(MEMORYLAYER_KNOWLEDGEBASE_PROVIDER, 'enterprise')  # Enterprise KB (entity-registry articles)
    # GLiNER2 typed NER extraction (person/org/project/location/event) instead of the
    # OSS regex extractor — the source of a proper typed entity graph. Fail-safe: on
    # any NER-service error it falls back to the regex path, so ingest is never
    # blocked (needs MEMORYLAYER_GLINER2_NER_URL pointing at the NER service).
    v.set_default_value(MEMORYLAYER_EXTRACTION_SERVICE, 'gliner2')  # Enterprise typed NER extraction
    # Enterprise entity registry (pgvector embedding-fuzzy resolution tier) instead of
    # the OSS relational default. Wires the embedding service (EXT_EMBEDDING_SERVICE)
    # so canonical entities carry a name embedding and duplicate surface forms
    # (e.g. "A. Jain"/"Jain") self-dedupe via the fuzzy tier. Fail-safe: no embedding
    # service -> exact+alias+create only (no regression vs the default registry).
    v.set_default_value(MEMORYLAYER_ENTITY_REGISTRY_PROVIDER, 'postgresql')  # Enterprise embedding-fuzzy registry
    v.set_default_value(MEMORYLAYER_AUTO_KB_UPDATE_ON_INGEST, DEFAULT_MEMORYLAYER_AUTO_KB_UPDATE_ON_INGEST)  # Auto KB update on ingest - True
    v.set_default_value(MEMORYLAYER_API_KEY_STORE, 'aether')  # Aether-first API key resolution (env fallback)

    # Auth/authz default to 'aether' (gateway-trusted identity) instead of the OSS
    # allow-all 'default' services. Enterprise is the platform data authority and
    # must fail closed; ENV configuration still wins for operators who deliberately
    # override (e.g. local dev with MEMORYLAYER_AUTHENTICATION_SERVICE=default).
    v.set_default_value(MEMORYLAYER_AUTHENTICATION_SERVICE, 'aether')  # Aether gateway auth
    v.set_default_value(MEMORYLAYER_AUTHORIZATION_SERVICE, 'aether')  # Aether access-level authz

    # Workspaces are never created as a side effect of resolving one. OSS defaults
    # this on for its "just works" story (MCP derives a workspace from the git repo
    # name and expects the first remember() to land), but in enterprise a workspace
    # is a billed, ACL'd, tenant-scoped container — it comes from an explicit create
    # (POST /v1/workspaces, or TenantInterface2.ensure_private_workspace for user
    # home workspaces), never from a request that merely names one.
    #
    # Leaving it on let the admin console's "Filter workspace" text box create a
    # workspace per debounced keystroke, because the cross-workspace list endpoints
    # take workspace_id as a FILTER and auth treated it as the request's context.
    v.set_default_value(MEMORYLAYER_WORKSPACE_IMPLICIT_CREATE, False)

    # Fail-closed startup guard: when MEMORYLAYER_REQUIRE_SECURE_AUTH is truthy,
    # refuse to boot if either resolved auth service is still the OSS allow-all
    # 'default'. Opt-in (default off) so dev/local still works, but prod can
    # guarantee fail-closed auth by exporting MEMORYLAYER_REQUIRE_SECURE_AUTH=1.
    from scitrera_app_framework import ext_parse_bool

    require_secure_auth = v.environ(
        MEMORYLAYER_REQUIRE_SECURE_AUTH,
        default=DEFAULT_MEMORYLAYER_REQUIRE_SECURE_AUTH,
        type_fn=ext_parse_bool,
    )
    if require_secure_auth:
        offenders = [
            key for key in (MEMORYLAYER_AUTHENTICATION_SERVICE, MEMORYLAYER_AUTHORIZATION_SERVICE)
            if v.get(key) == 'default'
        ]
        if offenders:
            raise RuntimeError(
                f"{MEMORYLAYER_REQUIRE_SECURE_AUTH} is set but the following are still the "
                f"OSS allow-all 'default' service: {', '.join(offenders)}. Refusing to boot "
                f"with open auth. Set these to 'aether' (or unset {MEMORYLAYER_REQUIRE_SECURE_AUTH})."
            )

    return


# bring in OSS configuration functions so that we stay entirely parallel
preconfigure = dependencies.preconfigure
initialize_services = dependencies.initialize_services
shutdown_services = dependencies.shutdown_services

_add_preconfigure_hook(_enterprise_preconfigure_hook)

__all__ = (
    'preconfigure',
    'initialize_services',
    'shutdown_services',
)
