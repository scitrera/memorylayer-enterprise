"""MemoryLayer SaaS - Enterprise extensions for MemoryLayer.ai.

This package provides enterprise features for MemoryLayer:
- PostgreSQL storage backend with pgvector support
- Cold tier storage with LEANN compression
- Automatic memory tiering based on importance and access patterns
- Enhanced memory service with hot/cold tier recall

Usage:
    # In your application initialization
    import memorylayer_saas.dependencies  # registers enterprise hooks on import
    from memorylayer_server.dependencies import preconfigure

    preconfigure()  # Runs both OSS and enterprise preconfiguration hooks

    # Or use the CLI
    memorylayer-enterprise serve
"""
from memorylayer_server import __version__

__all__ = ("__version__",)
