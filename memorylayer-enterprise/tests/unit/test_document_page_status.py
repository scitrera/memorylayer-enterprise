# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only
"""Page counts must not make failed ingestion look like active preparation."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.requests import Request

from memorylayer_saas.api.v1 import documents
from memorylayer_saas.services.document import gap_analysis
from memorylayer_server.models.document import DocumentStatus


@pytest.mark.asyncio
@pytest.mark.parametrize('state', list(DocumentStatus))
async def test_page_status_exposes_lifecycle_and_preserves_gaps(monkeypatch, state):
    storage = SimpleNamespace(get_documents=AsyncMock(return_value=[SimpleNamespace(id='doc', status=state)]))
    monkeypatch.setattr(documents, 'get_extension', lambda *args: storage)
    gaps = [SimpleNamespace(document_id='doc', is_complete=False, first_missing_phase='transcribe',
        expected_page_count=3, page_count=3, pages_with_transcript=0, flags=None)]
    monkeypatch.setattr(gap_analysis, 'analyze_documents_gaps', AsyncMock(return_value=gaps))
    auth = SimpleNamespace(build_context=AsyncMock(return_value=SimpleNamespace(workspace_id='workspace')))
    authz = SimpleNamespace(require_authorization=AsyncMock())
    result = await documents.get_pages_status(Request({'type': 'http'}),
        documents.PagesStatusRequest(doc_ids=['doc', 'absent']), auth_service=auth,
        authz_service=authz, v=MagicMock(), logger=MagicMock())
    present, missing = result.documents
    assert present.ingestion_status == state.value
    assert present.first_missing_phase == 'transcribe'
    assert present.page_count == 3 and present.pages_with_transcript == 0
    assert missing.ingestion_status == 'missing' and not missing.is_complete
    assert not result.is_complete
    authz.require_authorization.assert_awaited_once_with(
        auth.build_context.return_value, 'documents', 'read', workspace_id='workspace')
