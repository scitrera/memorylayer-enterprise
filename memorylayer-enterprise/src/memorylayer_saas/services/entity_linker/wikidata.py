# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Wikidata entity linker (enterprise) — canonical entity -> Wikidata QID.

Selected with ``MEMORYLAYER_ENTITY_LINKER_PROVIDER=wikidata`` (opt-in per tenant;
the OSS ``default`` no-op linker stays the default). Resolves a canonical entity
via the Wikidata ``wbsearchentities`` API and links ONLY on an exact (case-insensitive)
label match — a high-precision guard against wrong links. Combined with the
enrichment path's ``eligible_types`` filter (org/person/place, where Wikidata
coverage is good), this avoids linking niche/ambiguous concepts.

Fully fail-safe: any network / HTTP / parse error -> ``None`` (no link written).
The HTTP call runs off the event loop via ``asyncio.to_thread``.
"""

import asyncio
import json
import logging
import urllib.parse
import urllib.request

from scitrera_app_framework import Variables

from memorylayer_server.models.entity_registry import ExternalEntityLink
from memorylayer_server.services.entity_linker import (
    EntityLinkerService,
    EntityLinkerServicePluginBase,
)

# Env var NAMES (self-contained; read via v.environ with the defaults below).
MEMORYLAYER_WIKIDATA_API_URL = "MEMORYLAYER_WIKIDATA_API_URL"
MEMORYLAYER_WIKIDATA_TIMEOUT = "MEMORYLAYER_WIKIDATA_TIMEOUT"
MEMORYLAYER_WIKIDATA_USER_AGENT = "MEMORYLAYER_WIKIDATA_USER_AGENT"
MEMORYLAYER_WIKIDATA_LANGUAGE = "MEMORYLAYER_WIKIDATA_LANGUAGE"

DEFAULT_WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
DEFAULT_WIKIDATA_TIMEOUT = 10.0
# Wikidata's API policy requires a descriptive User-Agent.
DEFAULT_WIKIDATA_USER_AGENT = "MemoryLayer-EntityLinker/1.0 (https://scitrera.com)"
DEFAULT_WIKIDATA_LANGUAGE = "en"


class WikidataEntityLinkerService(EntityLinkerService):
    """Link canonical entities to Wikidata QIDs (exact-label, high precision)."""

    def __init__(self, *, api_url: str, timeout: float, user_agent: str, language: str, v: Variables = None):
        self._api_url = api_url
        self._timeout = timeout
        self._user_agent = user_agent
        self._language = language
        self.logger = logging.getLogger("memorylayer-server.WikidataEntityLinkerService")

    async def link(self, name: str, entity_type: str) -> ExternalEntityLink | None:
        name = (name or "").strip()
        if not name:
            return None
        try:
            candidates = await asyncio.to_thread(self._search, name)
        except Exception as e:  # noqa: BLE001 - linker must never raise
            self.logger.debug("Wikidata search failed for %r: %s", name, e)
            return None

        target = name.casefold()
        for cand in candidates:
            label = (cand.get("label") or "").strip()
            if label and label.casefold() == target:
                qid = cand.get("id")
                if not qid:
                    continue
                return ExternalEntityLink(
                    source="wikidata",
                    external_id=qid,
                    label=label,
                    description=cand.get("description") or None,
                    url=cand.get("concepturi") or f"https://www.wikidata.org/wiki/{qid}",
                    score=1.0,
                )
        return None

    def _search(self, name: str) -> list[dict]:
        """Sync call to wbsearchentities; returns the candidate list (may be empty)."""
        params = urllib.parse.urlencode(
            {
                "action": "wbsearchentities",
                "search": name,
                "language": self._language,
                "uselang": self._language,
                "type": "item",
                "format": "json",
                "limit": 5,
            }
        )
        req = urllib.request.Request(
            f"{self._api_url}?{params}",
            headers={"User-Agent": self._user_agent, "Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = json.loads(resp.read())
        return body.get("search", []) or []


class WikidataEntityLinkerServicePlugin(EntityLinkerServicePluginBase):
    """Plugin for the Wikidata entity linker. Selected via
    ``MEMORYLAYER_ENTITY_LINKER_PROVIDER=wikidata`` (opt-in per tenant)."""

    PROVIDER_NAME = "wikidata"

    def initialize(self, v: Variables, logger: logging.Logger) -> WikidataEntityLinkerService:
        return WikidataEntityLinkerService(
            api_url=v.environ(MEMORYLAYER_WIKIDATA_API_URL, default=DEFAULT_WIKIDATA_API_URL),
            timeout=float(v.environ(MEMORYLAYER_WIKIDATA_TIMEOUT, default=DEFAULT_WIKIDATA_TIMEOUT)),
            user_agent=v.environ(MEMORYLAYER_WIKIDATA_USER_AGENT, default=DEFAULT_WIKIDATA_USER_AGENT),
            language=v.environ(MEMORYLAYER_WIKIDATA_LANGUAGE, default=DEFAULT_WIKIDATA_LANGUAGE),
            v=v,
        )
