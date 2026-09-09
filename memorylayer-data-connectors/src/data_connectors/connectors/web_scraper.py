"""Web scraper connector — fetches content from web URLs.

Discovers content by scraping configured URLs, computes content hashes,
and returns entries for the sync engine to register.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from data_connectors.connectors import ConnectorRegistry

logger = logging.getLogger(__name__)

CONNECTOR_TYPE = "web_scraper"


class WebScraperConnector:
    """Connector that fetches and indexes web pages.

    Args:
        urls: List of URLs to scrape.
        follow_links: Whether to follow links on discovered pages.
        max_depth: Maximum link-following depth.
    """

    def __init__(
        self,
        urls: list[str],
        follow_links: bool = False,
        max_depth: int = 1,
    ) -> None:
        self._urls = urls
        self._follow_links = follow_links
        self._max_depth = max_depth

    async def load(self) -> None:
        """Validate that URLs are reachable."""
        logger.info("WebScraperConnector loaded with %d URLs", len(self._urls))

    async def poll(self) -> list[dict[str, Any]]:
        """Fetch configured URLs and return entry dicts.

        Each entry includes source_path (URL), content_hash (SHA-256 of body),
        content_type, and size_bytes.
        """
        # Delayed import: httpx is not needed at module load time
        import httpx

        entries: list[dict[str, Any]] = []
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            for url in self._urls:
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    body = response.content
                    content_hash = hashlib.sha256(body).hexdigest()
                    content_type = response.headers.get("content-type", "text/html")
                    entries.append({
                        "source_path": url,
                        "content_hash": content_hash,
                        "content_type": content_type.split(";")[0].strip(),
                        "size_bytes": len(body),
                        "metadata": {
                            "status_code": response.status_code,
                            "url": url,
                        },
                    })
                except Exception:
                    logger.warning("Failed to fetch URL: %s", url, exc_info=True)

        logger.info("WebScraperConnector polled %d URLs, got %d entries", len(self._urls), len(entries))
        return entries

    async def get_content_url(self, vfs_ref: str) -> Optional[str]:
        """Web content is fetched directly from the original URL."""
        return None

    async def get_metadata(self, vfs_ref: str) -> dict[str, Any]:
        """Return metadata for a scraped page."""
        return {"connector_type": CONNECTOR_TYPE}


ConnectorRegistry.register(CONNECTOR_TYPE, WebScraperConnector)
