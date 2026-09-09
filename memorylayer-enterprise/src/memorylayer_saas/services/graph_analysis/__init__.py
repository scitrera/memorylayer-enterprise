"""Enterprise Apache-AGE graph-analysis backend.

Re-exports the AGE service + plugin. The plugin is auto-discovered by the
enterprise ``register_package_plugins(services.__package__, ..., recursive=True)``
scan (see ``memorylayer_saas/dependencies.py``), so simply living in this package
registers it. Enterprise preconfiguration selects it with ``MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER=age``.
"""

from .age import AgeGraphAnalysisService, AgeGraphAnalysisServicePlugin

__all__ = (
    "AgeGraphAnalysisService",
    "AgeGraphAnalysisServicePlugin",
)
