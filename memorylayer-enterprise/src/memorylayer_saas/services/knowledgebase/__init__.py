"""Enterprise knowledgebase backend (canonical-entity articles).

Re-exports the enterprise service + plugin. The plugin is auto-discovered by the
enterprise ``register_package_plugins(services.__package__, ..., recursive=True)``
scan (see ``memorylayer_saas/dependencies.py``), so simply living in this package
registers it. Selected with ``MEMORYLAYER_KNOWLEDGEBASE_PROVIDER=enterprise`` (the
enterprise ``dependencies.py`` sets that as the default).

The enterprise backend extends the OSS ``DefaultKnowledgebaseService`` with ONE
additive, fail-safe layer — an article per canonical Entity Registry entity (the
"proper entity graph" view) — via the OSS ``_generate_extra_articles`` seam. OSS
stays the parity reference (no extra articles); any failure (no registry, no LLM,
registry error, feature disabled) degrades to the exact OSS KB.
"""

from .enterprise import (
    EnterpriseKnowledgebaseService,
    EnterpriseKnowledgebaseServicePlugin,
)

__all__ = (
    "EnterpriseKnowledgebaseService",
    "EnterpriseKnowledgebaseServicePlugin",
)
