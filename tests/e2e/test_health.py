# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""E2E: Health and connectivity checks."""

import httpx


def test_health_endpoint(e2e_url):
    """Server health endpoint returns 200."""
    resp = httpx.get(f"{e2e_url}/health", timeout=10)
    assert resp.status_code == 200


def test_openapi_docs(e2e_url):
    """OpenAPI docs are accessible."""
    resp = httpx.get(f"{e2e_url}/docs", timeout=10)
    assert resp.status_code == 200
