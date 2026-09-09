"""Enterprise entity linker providers (Wikidata).

Auto-discovered by the enterprise ``register_package_plugins(services.__package__,
recursive=True)`` scan. Selected with ``MEMORYLAYER_ENTITY_LINKER_PROVIDER=wikidata``
(opt-in per tenant; OSS ``default`` no-op linker stays the default).
"""

from .wikidata import (
    WikidataEntityLinkerService,
    WikidataEntityLinkerServicePlugin,
)

__all__ = (
    "WikidataEntityLinkerService",
    "WikidataEntityLinkerServicePlugin",
)
