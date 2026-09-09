"""Unit tests for the WorkflowRegistrationPlugin (kb-coalesce join rules).

Covers:
- Flag off → no workflow_op call
- Flag on + both rules absent in LIST → CREATE_RULE for BOTH rules
- Both rules already present + identical → no CREATE/UPDATE
- The ingest rule reuses name ``kb-update-on-ingest`` and its destination is now
  the join (a drifted create_task destination is reconciled via UPDATE_RULE)
- The decompose rule is ``kb-coalesce-on-decompose`` on
  ``memorylayer.decompose_complete``
- Both join templates carry the identical ``name: kb-coalesce`` and
  ``correlation_key: "source.workspace"`` (drift guard)
- LIST_RULES timeout (None response) → still attempts CREATE_RULE for both
- workflow_op client attribute absent → no crash
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memorylayer_saas.services.workflow_registration import (
    WorkflowRegistrationPlugin,
    _RULE_NAME_INGEST,
    _RULE_NAME_DECOMPOSE,
    _RULE_INGEST,
    _RULE_DECOMPOSE,
    _JOIN_NAME,
    _JOIN_CORRELATION_KEY,
    _JOIN_DESTINATION_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_variables(enabled=True):
    """Build a mock Variables whose environ() returns the given enabled flag."""
    v = MagicMock()
    v.environ = MagicMock(return_value=enabled)
    return v


def _make_list_response(rule_names: list[str]):
    """Build a mock WorkflowResponse for LIST_RULES containing the given rule names."""
    resp = MagicMock()
    resp.success = True
    resp.data = json.dumps([{"RuleName": n} for n in rule_names]).encode("utf-8")
    return resp


def _make_list_response_full(rules: list[dict]):
    """Build a LIST_RULES response from full rule dicts (RuleName + fields)."""
    resp = MagicMock()
    resp.success = True
    resp.data = json.dumps(rules).encode("utf-8")
    return resp


def _make_op_response(success=True):
    resp = MagicMock()
    resp.success = success
    resp.error = "" if success else "server error"
    resp.message = ""
    return resp


def _make_mock_client(responses):
    """Build an async client mock whose workflow_op side-effects are staged."""
    client = MagicMock()
    client.workflow_op = AsyncMock(side_effect=list(responses))
    return client


def _make_agent_service(client):
    svc = MagicMock()
    svc.client = client
    return svc


def _ops_by_type(client):
    """Group workflow_op call op-arguments by their op enum value."""
    grouped: dict = {}
    for call in client.workflow_op.call_args_list:
        op_arg = call.args[0]
        grouped.setdefault(op_arg._op, []).append(op_arg)
    return grouped


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def plugin():
    return WorkflowRegistrationPlugin()


@pytest.fixture()
def mock_logger():
    return MagicMock()


# ---------------------------------------------------------------------------
# Static template invariants
# ---------------------------------------------------------------------------

class TestJoinTemplateInvariants:
    """Both member rules must render the SAME join name + correlation key."""

    def test_both_templates_are_identical(self):
        # Both members share the exact destination template (one shared builder).
        assert _RULE_INGEST["DestinationTemplate"] == _JOIN_DESTINATION_TEMPLATE
        assert _RULE_DECOMPOSE["DestinationTemplate"] == _JOIN_DESTINATION_TEMPLATE

    def test_templates_carry_join_coalesce_shape(self):
        for rule in (_RULE_INGEST, _RULE_DECOMPOSE):
            tmpl = rule["DestinationTemplate"]
            assert "type: join" in tmpl
            assert "mode: coalesce" in tmpl
            assert ("name: %s" % _JOIN_NAME) in tmpl
            assert ('correlation_key: "%s"' % _JOIN_CORRELATION_KEY) in tmpl
            assert "memorylayer-task.kb_update" in tmpl

    def test_join_name_and_correlation_key_values(self):
        assert _JOIN_NAME == "kb-coalesce"
        assert _JOIN_CORRELATION_KEY == "source.workspace"

    def test_rule_envelopes(self):
        assert _RULE_NAME_INGEST == "kb-update-on-ingest"
        assert _RULE_INGEST["SourceEvent"] == "memorylayer.ingest_complete"
        assert _RULE_NAME_DECOMPOSE == "kb-coalesce-on-decompose"
        assert _RULE_DECOMPOSE["SourceEvent"] == "memorylayer.decompose_complete"


# ---------------------------------------------------------------------------
# Plugin behavior
# ---------------------------------------------------------------------------

class TestWorkflowRegistrationPlugin:

    def test_extension_point_name(self, plugin):
        v = MagicMock()
        assert plugin.extension_point_name(v) == "memorylayer-workflow-registration"

    def test_get_dependencies_includes_aether(self, plugin):
        from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION
        v = MagicMock()
        deps = plugin.get_dependencies(v)
        assert EXT_AETHER_SERVICE_CONNECTION in deps

    def test_initialize_returns_sentinel(self, plugin, mock_logger):
        v = MagicMock()
        result = plugin.initialize(v, mock_logger)
        assert result is not None  # trivial sentinel

    @pytest.mark.asyncio
    async def test_flag_off_no_workflow_op(self, plugin, mock_logger):
        """When the config flag is false, workflow_op must never be called."""
        v = _make_variables(enabled=False)
        client = MagicMock()
        client.workflow_op = AsyncMock()
        agent_svc = _make_agent_service(client)

        with patch(
            "memorylayer_saas.services.workflow_registration.get_extension",
            return_value=agent_svc,
        ):
            await plugin.async_ready(v, mock_logger, object())

        client.workflow_op.assert_not_called()

    @pytest.mark.asyncio
    async def test_both_rules_absent_creates_both(self, plugin, mock_logger):
        """Flag on + neither rule in LIST → CREATE_RULE for BOTH rules."""
        v = _make_variables(enabled=True)

        # One LIST (no rules) then two CREATEs.
        responses = [
            _make_list_response([]),
            _make_op_response(True),
            _make_op_response(True),
        ]
        client = _make_mock_client(responses)
        agent_svc = _make_agent_service(client)
        fake_pb2 = _make_fake_pb2()

        with patch(
            "memorylayer_saas.services.workflow_registration.get_extension",
            return_value=agent_svc,
        ), patch.dict(
            "sys.modules",
            {"scitrera_aether_client.proto": MagicMock(aether_pb2=fake_pb2)},
        ):
            await plugin.async_ready(v, mock_logger, object())

        grouped = _ops_by_type(client)
        assert len(grouped.get("LIST_RULES", [])) == 1
        creates = grouped.get("CREATE_RULE", [])
        assert len(creates) == 2

        created = {}
        for op in creates:
            rule_data = json.loads(op._data.decode("utf-8"))
            created[rule_data["RuleName"]] = rule_data

        # Ingest rule
        ingest = created[_RULE_NAME_INGEST]
        assert ingest["SourceEvent"] == "memorylayer.ingest_complete"
        assert "type: join" in ingest["DestinationTemplate"]
        assert "create_task" not in ingest["DestinationTemplate"].split("\n")[0]
        assert ("name: %s" % _JOIN_NAME) in ingest["DestinationTemplate"]

        # Decompose rule
        decompose = created[_RULE_NAME_DECOMPOSE]
        assert decompose["SourceEvent"] == "memorylayer.decompose_complete"
        assert "type: join" in decompose["DestinationTemplate"]

        # Both share identical join destination (one instance).
        assert ingest["DestinationTemplate"] == decompose["DestinationTemplate"]

    @pytest.mark.asyncio
    async def test_both_rules_present_identical_no_write(self, plugin, mock_logger):
        """Both rules already in LIST with identical templates → no CREATE/UPDATE."""
        v = _make_variables(enabled=True)

        list_resp = _make_list_response_full([
            {
                "ID": 7,
                "RuleName": _RULE_NAME_INGEST,
                "DestinationTemplate": _JOIN_DESTINATION_TEMPLATE,
            },
            {
                "ID": 8,
                "RuleName": _RULE_NAME_DECOMPOSE,
                "DestinationTemplate": _JOIN_DESTINATION_TEMPLATE,
            },
        ])
        client = _make_mock_client([list_resp])
        agent_svc = _make_agent_service(client)
        fake_pb2 = _make_fake_pb2()

        with patch(
            "memorylayer_saas.services.workflow_registration.get_extension",
            return_value=agent_svc,
        ), patch.dict(
            "sys.modules",
            {"scitrera_aether_client.proto": MagicMock(aether_pb2=fake_pb2)},
        ):
            await plugin.async_ready(v, mock_logger, object())

        # Only the single LIST call should have happened.
        assert client.workflow_op.call_count == 1
        assert client.workflow_op.call_args_list[0].args[0]._op == "LIST_RULES"

    @pytest.mark.asyncio
    async def test_ingest_rule_drifted_create_task_updates_to_join(self, plugin, mock_logger):
        """Existing kb-update-on-ingest with the old create_task dest → UPDATE to join."""
        v = _make_variables(enabled=True)

        # Old-style create_task destination on the reused ingest rule name.
        list_resp = _make_list_response_full([
            {
                "ID": 42,
                "RuleName": _RULE_NAME_INGEST,
                "DestinationTemplate": (
                    "type: create_task\n"
                    "task_type: memorylayer-task.kb_update\n"
                    "target_implementation: memorylayer\n"
                ),
            },
        ])
        # decompose rule is absent → it will be CREATEd.
        responses = [
            list_resp,
            _make_op_response(True),  # UPDATE ingest
            _make_op_response(True),  # CREATE decompose
        ]
        client = _make_mock_client(responses)
        agent_svc = _make_agent_service(client)
        fake_pb2 = _make_fake_pb2()

        with patch(
            "memorylayer_saas.services.workflow_registration.get_extension",
            return_value=agent_svc,
        ), patch.dict(
            "sys.modules",
            {"scitrera_aether_client.proto": MagicMock(aether_pb2=fake_pb2)},
        ):
            await plugin.async_ready(v, mock_logger, object())

        grouped = _ops_by_type(client)
        updates = grouped.get("UPDATE_RULE", [])
        creates = grouped.get("CREATE_RULE", [])
        assert len(updates) == 1
        assert len(creates) == 1

        # The ingest rule was reconciled in place to the join destination.
        update_op = updates[0]
        assert update_op._id == "42"
        rule_data = json.loads(update_op._data.decode("utf-8"))
        assert rule_data["RuleName"] == _RULE_NAME_INGEST
        assert rule_data["DestinationTemplate"] == _JOIN_DESTINATION_TEMPLATE
        assert "type: join" in rule_data["DestinationTemplate"]
        # The UPDATE_RULE payload must carry Active=True so reconciling a
        # previously-disabled rule re-activates it (reactivation-on-update is a
        # server-behavior assumption; this guards our side of the contract).
        assert rule_data["Active"] is True

        # The decompose rule was created.
        create_data = json.loads(creates[0]._data.decode("utf-8"))
        assert create_data["RuleName"] == _RULE_NAME_DECOMPOSE
        assert create_data["SourceEvent"] == "memorylayer.decompose_complete"

    @pytest.mark.asyncio
    async def test_list_timeout_still_creates_both(self, plugin, mock_logger):
        """LIST_RULES returning None (timeout) → still attempts CREATE_RULE for both."""
        v = _make_variables(enabled=True)

        client = MagicMock()
        client.workflow_op = AsyncMock(side_effect=[
            None,                     # LIST timeout
            _make_op_response(True),  # CREATE ingest
            _make_op_response(True),  # CREATE decompose
        ])
        agent_svc = _make_agent_service(client)
        fake_pb2 = _make_fake_pb2()

        with patch(
            "memorylayer_saas.services.workflow_registration.get_extension",
            return_value=agent_svc,
        ), patch.dict(
            "sys.modules",
            {"scitrera_aether_client.proto": MagicMock(aether_pb2=fake_pb2)},
        ):
            await plugin.async_ready(v, mock_logger, object())

        grouped = _ops_by_type(client)
        assert len(grouped.get("CREATE_RULE", [])) == 2

    @pytest.mark.asyncio
    async def test_no_workflow_op_attr_no_crash(self, plugin, mock_logger):
        """Client without workflow_op attribute → logs warning, no crash."""
        v = _make_variables(enabled=True)

        client = MagicMock(spec=[])  # no attributes
        agent_svc = _make_agent_service(client)

        with patch(
            "memorylayer_saas.services.workflow_registration.get_extension",
            return_value=agent_svc,
        ):
            # Must not raise even without workflow_op
            await plugin.async_ready(v, mock_logger, object())


# ---------------------------------------------------------------------------
# Fake protobuf helper
# ---------------------------------------------------------------------------

class _FakeWorkflowOp:
    """Minimal stand-in for aether_pb2.WorkflowOperation."""

    def __init__(self, op=None, workspace="", id="", data=b"", request_id=""):
        self._op = op
        self._workspace = workspace
        self._id = id
        self._data = data
        self._request_id = request_id
        self.request_id = request_id  # mutable so workflow_op() can set it

    LIST_RULES = "LIST_RULES"
    CREATE_RULE = "CREATE_RULE"
    UPDATE_RULE = "UPDATE_RULE"


class _FakePb2:
    """Minimal proto stub with just the WorkflowOperation we need."""

    class WorkflowOperation(_FakeWorkflowOp):
        LIST_RULES = "LIST_RULES"
        CREATE_RULE = "CREATE_RULE"
        UPDATE_RULE = "UPDATE_RULE"

    class UpstreamMessage:
        def __init__(self, workflow_op=None):
            self.workflow_op = workflow_op


def _make_fake_pb2():
    return _FakePb2()
