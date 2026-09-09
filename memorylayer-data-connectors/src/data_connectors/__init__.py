"""Data connectors and VFS catalog for the Scitrera AI platform.

Provides:
- Connector implementations (manual upload, S3, web scraper)
- VFS catalog (virtual filesystem entry registry)
- S3-compatible blob storage with presigned URL minting
- Sync engine that emits Aether POOL tasks for MemoryLayer ingestion
- FastAPI HTTP surface fronted by Aether's proxy_http_async
"""

__version__ = "0.0.1"
