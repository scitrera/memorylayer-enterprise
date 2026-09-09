"""Enterprise Apache-AGE graph-QUERY backend (P2 Track A).

Re-exports the AGE graph-query service + plugin. The plugin is auto-discovered
by the enterprise ``register_package_plugins(services.__package__, ...,
recursive=True)`` scan, so living in this package registers it. Enable with
``MEMORYLAYER_GRAPH_QUERY_PROVIDER=age``; enterprise preconfiguration selects
``"age"`` by default while OSS keeps the relational backend.
"""

from .age import AgeGraphQueryService, AgeGraphQueryServicePlugin

__all__ = (
    "AgeGraphQueryService",
    "AgeGraphQueryServicePlugin",
)
