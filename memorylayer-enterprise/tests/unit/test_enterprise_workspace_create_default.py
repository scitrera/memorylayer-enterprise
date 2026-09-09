# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Enterprise must not create workspaces as a side effect of resolving one.

In enterprise a workspace is a billed, ACL'd, tenant-scoped container: it comes
from an explicit create (``POST /v1/workspaces``, or
``TenantInterface2.ensure_private_workspace`` for per-user home workspaces).
OSS defaults the opposite way for its "just works" story, so this is one of the
enterprise preconfigure overrides — and it is exactly the kind of default a
later refactor can silently drop, hence the test.

The leak this closes: the admin console's cross-workspace list endpoints take
``workspace_id`` as a FILTER, and auth treated it as the request's workspace
context. Typing into the "Filter workspace" box created a workspace per
debounced keystroke.
"""

import logging
from unittest.mock import MagicMock

from memorylayer_server.config import MEMORYLAYER_WORKSPACE_IMPLICIT_CREATE


def _captured_defaults() -> dict:
    """Run the enterprise preconfigure hook against a recording Variables."""
    from memorylayer_saas.dependencies import _enterprise_preconfigure_hook

    captured: dict = {}
    v = MagicMock()
    v.set_default_value = lambda key, value: captured.__setitem__(key, value)
    # environ() feeds the fail-closed auth guard; keep it off so the hook runs
    # to completion.
    v.environ = MagicMock(return_value=False)
    # The framework pulls its logger off Variables; hand it a real one so the
    # hook's own logging doesn't blow up on a stub.
    v.get = MagicMock(return_value=logging.getLogger("test-preconfigure"))

    _enterprise_preconfigure_hook(v)
    return captured


def test_enterprise_disables_implicit_workspace_creation():
    assert _captured_defaults()[MEMORYLAYER_WORKSPACE_IMPLICIT_CREATE] is False


def test_it_is_a_default_not_a_hard_setting():
    """Operators (and local dev) must still be able to override via env.

    ``set_default_value`` loses to an explicit environment variable, which is
    what every other enterprise override relies on too.
    """
    from memorylayer_saas.dependencies import _enterprise_preconfigure_hook

    calls = []
    v = MagicMock()
    v.set_default_value = lambda key, value: calls.append(key)
    v.set = MagicMock(side_effect=AssertionError("must not hard-set the workspace flag"))
    v.environ = MagicMock(return_value=False)
    # The framework pulls its logger off Variables; hand it a real one so the
    # hook's own logging doesn't blow up on a stub.
    v.get = MagicMock(return_value=logging.getLogger("test-preconfigure"))

    _enterprise_preconfigure_hook(v)

    assert MEMORYLAYER_WORKSPACE_IMPLICIT_CREATE in calls
