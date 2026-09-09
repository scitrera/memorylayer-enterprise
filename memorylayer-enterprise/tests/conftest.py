# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""
Pytest configuration and fixtures for MemoryLayer Enterprise tests.

This conftest sets up the plugin architecture and provides fixtures that
layer enterprise functionality on top of OSS.

Provides:
- Framework initialization with enterprise plugins
- Mock embedding service
- Test data factories for LEANN storage
"""
import os
import sys
import pytest
import pytest_asyncio
import numpy as np
from unittest.mock import MagicMock

from scitrera_app_framework import Variables

# Mock the Aether SDK only when it is genuinely NOT installed (tests mock all
# SDK interactions, so the stub is enough for them). Checking installation
# rather than sys.modules matters: a MagicMock has no __path__, so it cannot
# satisfy a submodule import — anything doing "import scitrera_aether_client.X"
# fails as "'scitrera_aether_client' is not a package" instead of getting a
# mock. That is what made every test importing the ASGI bridge (which needs
# .proto) error at setup even with the SDK present.
if 'scitrera_aether_client' not in sys.modules:
    from importlib.util import find_spec

    try:
        _aether_sdk_installed = find_spec('scitrera_aether_client') is not None
    except (ImportError, ValueError):
        _aether_sdk_installed = False

    if not _aether_sdk_installed:
        _mock_aether = MagicMock()
        _mock_aether.POOL = 1
        _mock_aether.CONTROL = 42
        _mock_aether.AsyncAgentClient = MagicMock()
        sys.modules['scitrera_aether_client'] = _mock_aether
        sys.modules['scitrera_aether_client.client_async'] = _mock_aether

from memorylayer_server.services.embedding import EmbeddingService, EmbeddingProvider


# Set test environment variables before any imports
os.environ.setdefault('MEMORYLAYER_STORAGE_BACKEND', 'sqlite')  # Use SQLite for tests by default
os.environ.setdefault('MEMORYLAYER_EMBEDDING_PROVIDER', 'mock')


def _preconfigure_all():
    """Run all preconfiguration hooks (OSS + Enterprise).

    Importing memorylayer_saas.dependencies auto-registers the enterprise
    preconfigure hook, so calling preconfigure() runs both OSS and enterprise.
    """
    import memorylayer_saas.dependencies  # noqa: F401 — registers enterprise hook on import
    from memorylayer_server.dependencies import preconfigure

    preconfigure()


# Mock embedding provider for tests
class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider that returns deterministic embeddings."""

    def __init__(self, dimensions: int = 1536):
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        # Generate deterministic embedding based on text hash
        import hashlib
        import random
        h = hashlib.sha256(text.encode()).digest()
        rng = random.Random(h)
        return [rng.uniform(-1, 1) for _ in range(self._dimensions)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]

    @property
    def dimensions(self) -> int:
        return self._dimensions


@pytest.fixture(scope='session')
def test_framework():
    """Initialize the scitrera-app-framework with enterprise plugins for testing.

    This fixture:
    1. Runs OSS preconfiguration (registers base plugins)
    2. Runs enterprise preconfiguration (registers enterprise plugins)
    3. Initializes the framework
    4. Returns Variables instance for dependency access
    """
    from scitrera_app_framework import init_framework_test_harness

    # Run preconfiguration before initialize
    _preconfigure_all()

    # Initialize framework
    v = init_framework_test_harness('memorylayer-enterprise-test')
    logger = v.get('logger', None)

    yield v, logger


@pytest.fixture(scope='session')
def v(test_framework) -> Variables:
    """Get Variables instance from test framework."""
    v, _ = test_framework
    return v


@pytest_asyncio.fixture
async def mock_embedding_service() -> EmbeddingService:
    """Create mock embedding service."""
    provider = MockEmbeddingProvider(dimensions=1536)
    return EmbeddingService(provider=provider, cache=None)


@pytest.fixture
def workspace_id() -> str:
    """Default test workspace ID."""
    return "test_workspace"


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return [np.random.randn(1536).tolist() for _ in range(10)]


@pytest.fixture
def sample_memory_ids() -> list[str]:
    """Create sample memory IDs for testing."""
    return [f"mem_{i:012d}" for i in range(10)]


@pytest.fixture
def small_sample_embeddings() -> list[list[float]]:
    """Create small set of embeddings for simple tests."""
    np.random.seed(123)
    return [np.random.randn(1536).tolist() for _ in range(5)]


@pytest.fixture
def small_sample_memory_ids() -> list[str]:
    """Create small set of memory IDs for simple tests."""
    return [f"mem_{i:012d}" for i in range(5)]


# Enterprise-specific fixtures

@pytest.fixture
def tiering_service(v):
    """Get the tiering service from the plugin system."""
    from memorylayer_saas.services.tiering import get_tiering_service
    return get_tiering_service(v)


@pytest.fixture
def storage_backend(v):
    """Get the storage backend from the plugin system."""
    from memorylayer_server.services.storage import get_storage_backend
    return get_storage_backend(v)
