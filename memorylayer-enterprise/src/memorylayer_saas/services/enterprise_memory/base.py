"""Base definitions for enterprise memory service plugin."""
from scitrera_app_framework.api import Plugin, Variables

from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
from memorylayer_server.services.embedding import EXT_EMBEDDING_SERVICE
from memorylayer_server.services.cache import EXT_CACHE_SERVICE

# Re-export from OSS - enterprise memory extends OSS memory service plugin base
# The extension point is the same as OSS memory service
from memorylayer_server.services.memory import (
    EXT_MEMORY_SERVICE,
    MemoryServicePluginBase,
)

# Environment variable for selecting memory service provider
MEMORYLAYER_MEMORY_SERVICE = 'MEMORYLAYER_MEMORY_SERVICE'
DEFAULT_MEMORYLAYER_MEMORY_SERVICE = 'saas'  # Enterprise default is 'saas'

# Phase 1c: LLM update-vs-add consolidation on the write path. When enabled, the
# enterprise memory service overrides the deterministic ``_merge_memories`` merge
# with an LLM that decides whether a near-duplicate should UPDATE (reconcile into
# one memory) or ADD (stay distinct -> deterministic fallback), and records an
# update-history entry. Ships DARK (default OFF): byte-identical to the OSS
# deterministic merge until the flag is flipped, and every failure mode degrades
# to that same deterministic merge (fail-safe).
MEMORYLAYER_MERGE_LLM_ENABLED = 'MEMORYLAYER_MERGE_LLM_ENABLED'
DEFAULT_MEMORYLAYER_MERGE_LLM_ENABLED = False

# LLM profile used for the consolidation call. An unknown profile routes to the
# LLM service's "default" profile.
MEMORYLAYER_MERGE_LLM_PROFILE = 'MEMORYLAYER_MERGE_LLM_PROFILE'
DEFAULT_MEMORYLAYER_MERGE_LLM_PROFILE = 'merge'

# Phase 2: agentic EXPAND/RE_QUERY/STOP recall mode ("Memora-Control"). An
# iterative, LLM-controlled retrieval loop for multi-hop / "pointer" questions
# (the answer lives in a memory linked to, or a different entity than, the one
# the seed query surfaces). Ships DARK (default OFF): when disabled, a request
# for RecallMode.AGENTIC transparently falls back to RAG, so existing recall
# behaviour is unchanged. The whole loop is fail-safe — any LLM/graph error
# degrades to the seed RAG result, so AGENTIC is never worse than RAG.
MEMORYLAYER_AGENTIC_RECALL_ENABLED = 'MEMORYLAYER_AGENTIC_RECALL_ENABLED'
DEFAULT_MEMORYLAYER_AGENTIC_RECALL_ENABLED = False

# Maximum number of controller steps (EXPAND/RE_QUERY hops) before the loop is
# force-stopped and the accumulated working set is finalized. Bounds cost and
# guarantees termination.
MEMORYLAYER_AGENTIC_MAX_STEPS = 'MEMORYLAYER_AGENTIC_MAX_STEPS'
DEFAULT_MEMORYLAYER_AGENTIC_MAX_STEPS = 4

# Completion-token cap for the agentic-control LLM decision (the JSON control
# object that drives EXPAND/RE_QUERY/STOP). Env-tunable; the original 200 could
# truncate a reasoning model's control JSON to empty, breaking the loop.
MEMORYLAYER_AGENTIC_CONTROL_MAX_TOKENS = 'MEMORYLAYER_AGENTIC_CONTROL_MAX_TOKENS'
DEFAULT_MEMORYLAYER_AGENTIC_CONTROL_MAX_TOKENS = 1024

__all__ = (
    'EXT_MEMORY_SERVICE',
    'MemoryServicePluginBase',
    'MEMORYLAYER_MEMORY_SERVICE',
    'DEFAULT_MEMORYLAYER_MEMORY_SERVICE',
    'MEMORYLAYER_MERGE_LLM_ENABLED',
    'DEFAULT_MEMORYLAYER_MERGE_LLM_ENABLED',
    'MEMORYLAYER_MERGE_LLM_PROFILE',
    'DEFAULT_MEMORYLAYER_MERGE_LLM_PROFILE',
    'MEMORYLAYER_AGENTIC_RECALL_ENABLED',
    'DEFAULT_MEMORYLAYER_AGENTIC_RECALL_ENABLED',
    'MEMORYLAYER_AGENTIC_MAX_STEPS',
    'DEFAULT_MEMORYLAYER_AGENTIC_MAX_STEPS',
    'MEMORYLAYER_AGENTIC_CONTROL_MAX_TOKENS',
    'DEFAULT_MEMORYLAYER_AGENTIC_CONTROL_MAX_TOKENS',
)
