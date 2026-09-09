"""GitHub connector — syncs repository content, issues, and PRs via GitHub REST API.

Discovers content via ``httpx`` async client, computes content hashes,
and returns upstream raw URLs with PAT authorization for content fetch.

``get_content_url`` returns the upstream GitHub raw content URL with an
``Authorization`` header — no blob materialization needed.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Optional

from data_connectors.connectors import ConnectorRegistry
from data_connectors.vfs.catalog import VfsCatalog

logger = logging.getLogger(__name__)

CONNECTOR_TYPE = "github"

_GITHUB_API = "https://api.github.com"


def _parse_iso(s: str) -> float:
    """Parse an ISO-8601 timestamp to epoch seconds."""
    if not s:
        return 0
    try:
        from datetime import datetime
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0


class GitHubConnector:
    """Connector for GitHub repositories — README, issues, PRs.

    Args:
        credentials: Dict with optional ``token`` (PAT or OAuth token).
        settings: Dict with ``repos`` (list of ``owner/repo`` strings)
            and optional ``include_issues``, ``include_prs``, ``include_wiki``.
        catalog: VFS catalog for resolving vfs_ref entries.
    """

    def __init__(
        self,
        credentials: dict,
        settings: dict | None = None,
        catalog: VfsCatalog | None = None,
    ) -> None:
        self._credentials = credentials
        self._settings = settings or {}
        self._catalog = catalog

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for GitHub API requests."""
        headers = {"Accept": "application/vnd.github+json"}
        token = self._credentials.get("token")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def load(self) -> None:
        """Validate credentials by checking rate limit status."""
        # Delayed import: httpx is only needed when this connector runs
        import httpx

        async with httpx.AsyncClient(headers=self._build_headers(), timeout=30.0) as http:
            resp = await http.get(f"{_GITHUB_API}/rate_limit")
            resp.raise_for_status()
        repos = self._settings.get("repos", [])
        logger.info("GitHubConnector loaded (%d repos configured)", len(repos))

    async def poll(self) -> list[dict[str, Any]]:
        """Fetch README and issues/PRs from configured repos.

        Each entry includes source_path, content_hash, content_type,
        and size_bytes.
        """
        # Delayed import: httpx is only needed when this connector runs
        import httpx

        repos: list[str] = self._settings.get("repos", [])
        include_issues = self._settings.get("include_issues", True)
        include_prs = self._settings.get("include_prs", True)
        entries: list[dict[str, Any]] = []

        async with httpx.AsyncClient(headers=self._build_headers(), timeout=30.0) as http:
            for repo in repos:
                # README
                readme_entries = await self._fetch_readme(http, repo)
                entries.extend(readme_entries)

                # Issues and PRs
                if include_issues or include_prs:
                    issue_entries = await self._fetch_issues(
                        http, repo,
                        include_issues=include_issues,
                        include_prs=include_prs,
                    )
                    entries.extend(issue_entries)

        logger.info("GitHubConnector polled %d entries from %d repos", len(entries), len(repos))
        return entries

    async def get_content_url(self, vfs_ref: str) -> Optional[str]:
        """Return an upstream GitHub raw content URL with Authorization header.

        For README and file content, returns the raw GitHub URL.
        For issues/PRs, returns the API URL for the issue body.
        The caller should include the authorization header from
        ``get_metadata`` when fetching.

        Per the "always upstream URLs" decision, no blob materialization occurs.
        """
        if self._catalog is None:
            return None
        entry = await self._catalog.get(vfs_ref)
        if entry is None:
            return None

        # For README content, use the download_url
        download_url = entry.metadata.get("github_download_url")
        if download_url:
            return download_url

        # For issues/PRs, use the API URL
        html_url = entry.metadata.get("github_html_url")
        if html_url:
            return html_url

        return None

    async def get_metadata(self, vfs_ref: str) -> dict[str, Any]:
        """Return metadata for a GitHub entry.

        Includes ``authorization_header`` with the Bearer token so callers
        can authenticate when fetching the upstream URL.
        """
        if self._catalog is None:
            return {"connector_type": CONNECTOR_TYPE}
        entry = await self._catalog.get(vfs_ref)
        if entry is None:
            return {"connector_type": CONNECTOR_TYPE}

        token = self._credentials.get("token", "")
        meta: dict[str, Any] = {
            "connector_type": CONNECTOR_TYPE,
            "source_path": entry.source_path,
            "content_type": entry.content_type,
            "size_bytes": entry.size_bytes,
            **entry.metadata,
        }
        if token:
            meta["authorization_header"] = f"Bearer {token}"
        return meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_readme(self, http, repo: str) -> list[dict[str, Any]]:
        """Fetch the repository README and return as an entry dict."""
        url = f"{_GITHUB_API}/repos/{repo}/readme"
        try:
            resp = await http.get(url)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()

            # Content is base64-encoded
            import base64
            content_b64 = data.get("content", "")
            content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            return [{
                "source_path": f"github/{repo}/README",
                "content_hash": content_hash,
                "content_type": "text/markdown",
                "size_bytes": len(content.encode()),
                "metadata": {
                    "github_repo": repo,
                    "github_type": "readme",
                    "github_html_url": data.get("html_url", ""),
                    "github_download_url": data.get("download_url", ""),
                },
            }]
        except Exception:
            logger.error("Failed to fetch README for %s", repo, exc_info=True)
            return []

    async def _fetch_issues(
        self,
        http,
        repo: str,
        include_issues: bool = True,
        include_prs: bool = True,
    ) -> list[dict[str, Any]]:
        """Paginate through issues (and PRs) and return entry dicts."""
        entries: list[dict[str, Any]] = []
        page = 1

        while True:
            params: dict[str, str] = {
                "state": "all",
                "per_page": "100",
                "page": str(page),
                "sort": "updated",
                "direction": "desc",
            }

            url = f"{_GITHUB_API}/repos/{repo}/issues"
            try:
                resp = await http.get(url, params=params)
                resp.raise_for_status()
                items = resp.json()
            except Exception:
                logger.error("Error fetching issues for %s", repo, exc_info=True)
                break

            if not items:
                break

            for item in items:
                is_pr = "pull_request" in item
                if is_pr and not include_prs:
                    continue
                if not is_pr and not include_issues:
                    continue

                kind = "pr" if is_pr else "issue"
                number = item.get("number", 0)
                title = item.get("title", "")
                body = item.get("body", "") or ""
                full_content = f"# {title}\n\n{body}"
                content_hash = hashlib.sha256(full_content.encode()).hexdigest()

                entries.append({
                    "source_path": f"github/{repo}/{kind}/{number}",
                    "content_hash": content_hash,
                    "content_type": "text/markdown",
                    "size_bytes": len(full_content.encode()),
                    "metadata": {
                        "github_repo": repo,
                        "github_type": kind,
                        "github_number": number,
                        "github_title": title,
                        "github_state": item.get("state", ""),
                        "github_html_url": item.get("html_url", ""),
                        "github_labels": [label.get("name", "") for label in item.get("labels", [])],
                        "github_author": item.get("user") or {},
                        "github_assignees": item.get("assignees") or [],
                    },
                })

            # Stop when GitHub returns a partial page (last page) or after a
            # safety cap; protects against pagination loops if upstream
            # repeatedly returns full pages.
            if len(items) < 100 or page >= 100:
                break
            page += 1

        return entries


ConnectorRegistry.register(CONNECTOR_TYPE, GitHubConnector)
