# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Workflow rule registration for the ingest/decompose → KB coalesce-join pipeline.

Registers two Aether workflow rules that feed a single per-workspace
**coalesce join** named ``kb-coalesce``.  Arrivals of either
``memorylayer.ingest_complete`` or ``memorylayer.decompose_complete`` land in
the same per-workspace join instance (keyed on ``source.workspace``); the
server-side coalesce gate debounces the burst and fires a single
``memorylayer-task.kb_update`` pool task on completion (or on timeout, with a
``degraded`` flag).

The coalesce join replaces the hand-rolled per-workspace KV lease that
``kb_update.py`` used to carry — the workflow engine now owns the debounce.

The two rules:

* ``kb-update-on-ingest`` — member event ``memorylayer.ingest_complete``.  This
  reuses the name of the previous (``create_task``) rule so the idempotent
  reconcile updates it in place — replacing its old ``create_task`` destination
  with the ``join`` destination — without needing a ``DELETE_RULE``.
* ``kb-coalesce-on-decompose`` — member event ``memorylayer.decompose_complete``.

Both render an **identical** join destination (same ``name: kb-coalesce`` and
``correlation_key: "source.workspace"``) so the two events share ONE
per-workspace coalesce instance.  A drift between the two would silently create
two separate join instances — the two templates are kept identical via a shared
builder and guarded by tests.

Each rule is reconciled idempotently (LIST_RULES → compare ``DestinationTemplate``
→ CREATE_RULE if absent / UPDATE_RULE if drifted).

Registration is best-effort: any failure is logged and swallowed so it never
blocks startup or breaks the Aether connection lifecycle.
"""
from __future__ import annotations

import json
from uuid import uuid4

from scitrera_app_framework import Variables, ext_parse_bool, get_extension, get_logger
from scitrera_app_framework.api import Plugin

from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION

from memorylayer_saas.config import (
    MEMORYLAYER_AUTO_KB_UPDATE_ON_INGEST,
    DEFAULT_MEMORYLAYER_AUTO_KB_UPDATE_ON_INGEST,
)

# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

# Shared coalesce-join name + correlation key.  BOTH member rules MUST render
# these identical values so arrivals from either event share ONE per-workspace
# join instance (a drift would silently fork into two separate instances).
_JOIN_NAME = "kb-coalesce"
_JOIN_CORRELATION_KEY = "source.workspace"

# YAML destination template interpreted by the Aether workflow engine.  Uses Go
# template syntax ({{ .source.workspace }}) for field interpolation and renders
# a ``type: join`` destination (coalesce mode).  Both member rules share this
# exact template so they coalesce into the same per-workspace instance.
_JOIN_DESTINATION_TEMPLATE = (
    "type: join\n"
    "join:\n"
    "  name: %s\n"
    "  correlation_key: \"%s\"\n"
    "  mode: coalesce\n"
    "  window: \"5s\"\n"
    "  timeout: \"10m\"\n"
    "  linger: \"1m\"\n"
    "  on_complete:\n"
    "    type: create_task\n"
    "    task_type: memorylayer-task.kb_update\n"
    "    target_implementation: memorylayer\n"
    "    workspace: \"{{ .source.workspace }}\"\n"
    "    payload:\n"
    "      workspace_id: \"{{ .source.workspace }}\"\n"
    "    metadata:\n"
    "      bg_kind: kb\n"
    "      visibility: workspace\n"
    "      title: \"Updating knowledge base\"\n"
    "  on_timeout:\n"
    "    type: create_task\n"
    "    task_type: memorylayer-task.kb_update\n"
    "    target_implementation: memorylayer\n"
    "    workspace: \"{{ .source.workspace }}\"\n"
    "    payload:\n"
    "      workspace_id: \"{{ .source.workspace }}\"\n"
    "      degraded: true\n"
) % (_JOIN_NAME, _JOIN_CORRELATION_KEY)


def _build_rule(rule_name: str, source_event: str) -> dict:
    """Build a coalesce-join member rule for ``source_event``.

    Both members share the identical ``_JOIN_DESTINATION_TEMPLATE`` (same join
    name + correlation key); only the rule envelope's ``SourceEvent`` differs.

    Rule struct keys are PascalCase — no json tags on the Go side, so
    encoding/json uses the struct field name directly (PascalCase).
    """
    return {
        "RuleName": rule_name,
        "SourceAgent": "memorylayer",
        "SourceEvent": source_event,
        "TriggerCondition": "",
        "DestinationTemplate": _JOIN_DESTINATION_TEMPLATE,
        "Workspace": "*",
        "Priority": 0,
        # CREATE_RULE forces Active=true server-side, but UPDATE_RULE persists
        # whatever is in the JSON, so stamp it here to keep the rule active when
        # reconciling via UPDATE_RULE.
        "Active": True,
    }


# Ingest member reuses the previous rule name so the idempotent reconcile
# updates it in place (create_task → join destination) with no DELETE_RULE.
_RULE_NAME_INGEST = "kb-update-on-ingest"
_RULE_NAME_DECOMPOSE = "kb-coalesce-on-decompose"

_RULE_INGEST = _build_rule(_RULE_NAME_INGEST, "memorylayer.ingest_complete")
_RULE_DECOMPOSE = _build_rule(_RULE_NAME_DECOMPOSE, "memorylayer.decompose_complete")

# (rule_name, rule_dict) pairs reconciled at startup.
_RULES = (
    (_RULE_NAME_INGEST, _RULE_INGEST),
    (_RULE_NAME_DECOMPOSE, _RULE_DECOMPOSE),
)

# Extension point key used to register this plugin with the framework.
EXT_WORKFLOW_REGISTRATION = "memorylayer-workflow-registration"


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class WorkflowRegistrationPlugin(Plugin):
    """Plugin that registers the kb-coalesce workflow rules in ``async_ready``.

    Subclasses ``Plugin`` directly (not a provider-selection plugin) because
    this is a one-of-a-kind startup side-effect, not a swappable service.

    Depends on ``EXT_AETHER_SERVICE_CONNECTION`` so the framework guarantees
    the Aether client is connected before ``async_ready`` fires.

    Lifecycle:
        ``initialize`` — registers a trivial sentinel under
            ``EXT_WORKFLOW_REGISTRATION`` (no I/O).
        ``async_ready`` — idempotently registers the workflow rules.
    """

    def extension_point_name(self, v: Variables) -> str:
        return EXT_WORKFLOW_REGISTRATION

    def get_dependencies(self, v: Variables):
        """Depend on EXT_AETHER_SERVICE_CONNECTION so async_ready fires after connect."""
        return (EXT_AETHER_SERVICE_CONNECTION,)

    def initialize(self, v: Variables, logger) -> object:  # type: ignore[override]
        """Return a trivial sentinel; actual registration happens in async_ready."""
        return object()

    async def async_ready(self, v: Variables, logger, value: object) -> None:  # type: ignore[override]
        """Register the kb-coalesce workflow rules (best-effort)."""
        logger = get_logger(v, name="WorkflowRegistrationPlugin")

        enabled = v.environ(
            MEMORYLAYER_AUTO_KB_UPDATE_ON_INGEST,
            DEFAULT_MEMORYLAYER_AUTO_KB_UPDATE_ON_INGEST,
            type_fn=ext_parse_bool,
        )
        if not enabled:
            logger.info(
                "WorkflowRegistration: auto KB update on ingest disabled "
                "(%s=false), skipping rule registration",
                MEMORYLAYER_AUTO_KB_UPDATE_ON_INGEST,
            )
            return

        try:
            agent_service = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
            client = agent_service.client
        except Exception:
            logger.warning(
                "WorkflowRegistration: could not resolve Aether client, "
                "skipping rule registration (best-effort)",
                exc_info=True,
            )
            return

        if not hasattr(client, "workflow_op"):
            logger.warning(
                "WorkflowRegistration: Aether client does not expose workflow_op, "
                "skipping rule registration",
            )
            return

        # Delayed import — aether_pb2 has a non-trivial load time.
        try:
            from scitrera_aether_client.proto import aether_pb2
        except ImportError:
            logger.warning(
                "WorkflowRegistration: scitrera_aether_client not available, "
                "skipping rule registration",
            )
            return

        await _register_rules_idempotent(client, aether_pb2, logger)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rule_field(rule: dict, *names: str):
    """Return the first present value among ``names`` (PascalCase or snake_case)."""
    for name in names:
        if name in rule:
            return rule[name]
    return None


async def _register_rules_idempotent(client, aether_pb2, logger) -> None:
    """Reconcile every kb-coalesce member rule (create if absent, update if drifted).

    Lists the existing rules once, then reconciles each ``(rule_name, rule_dict)``
    in ``_RULES`` against that snapshot: if a rule with the same name exists and
    its ``DestinationTemplate`` matches, skip; if it drifted, UPDATE_RULE (by id);
    if absent, CREATE_RULE.

    All I/O is wrapped in best-effort try/except so a workflow-engine hiccup
    never breaks the MemoryLayer startup sequence.
    """
    existing_by_name = await _list_existing_rules(client, aether_pb2, logger)

    for rule_name, rule_dict in _RULES:
        await _reconcile_rule(
            client, aether_pb2, logger,
            rule_name=rule_name,
            rule_dict=rule_dict,
            existing_rule=existing_by_name.get(rule_name),
        )


async def _list_existing_rules(client, aether_pb2, logger) -> dict:
    """LIST_RULES → return a ``{rule_name: rule_dict}`` map (best-effort, may be empty).

    A failure (or unparseable response) yields an empty map; callers then fall
    through to CREATE_RULE, where a duplicate-name conflict is harmless.
    """
    existing_by_name: dict = {}
    try:
        list_op = aether_pb2.WorkflowOperation(
            op=aether_pb2.WorkflowOperation.LIST_RULES,
            workspace="*",
        )
        resp = await client.workflow_op(list_op, timeout=10.0)
        if resp is not None and resp.data:
            try:
                rules = json.loads(resp.data)
                if isinstance(rules, list):
                    for rule in rules:
                        name = _rule_field(rule, "RuleName", "rule_name") or ""
                        if name:
                            existing_by_name[name] = rule
            except Exception:
                logger.warning(
                    "WorkflowRegistration: could not parse LIST_RULES response "
                    "(best-effort), will attempt CREATE_RULE anyway",
                    exc_info=True,
                )
    except Exception:
        logger.warning(
            "WorkflowRegistration: LIST_RULES failed (best-effort), "
            "will attempt CREATE_RULE anyway",
            exc_info=True,
        )
    return existing_by_name


async def _reconcile_rule(
    client, aether_pb2, logger, *, rule_name: str, rule_dict: dict,
    existing_rule: dict | None,
) -> None:
    """Reconcile a single rule: skip if up to date, UPDATE_RULE if drifted, else CREATE_RULE."""
    desired_template = rule_dict["DestinationTemplate"]

    # ---- Reconcile an existing rule ------------------------------------
    if existing_rule is not None:
        current_template = _rule_field(
            existing_rule, "DestinationTemplate", "destination_template"
        )
        if current_template == desired_template:
            logger.info(
                "WorkflowRegistration: rule %r already up to date, skipping",
                rule_name,
            )
            return

        rule_id = _rule_field(existing_rule, "ID", "Id", "id")
        if rule_id is None:
            logger.warning(
                "WorkflowRegistration: rule %r drifted but LIST response has no id; "
                "cannot UPDATE_RULE (best-effort), leaving as-is",
                rule_name,
            )
            return

        logger.info(
            "WorkflowRegistration: rule %r destination drifted, reconciling via UPDATE_RULE (id=%s)",
            rule_name,
            rule_id,
        )
        try:
            update_op = aether_pb2.WorkflowOperation(
                op=aether_pb2.WorkflowOperation.UPDATE_RULE,
                workspace="*",
                id=str(rule_id),
                data=json.dumps(rule_dict).encode("utf-8"),
                request_id=str(uuid4()),
            )
            resp = await client.workflow_op(update_op, timeout=10.0)
            if resp is not None and resp.success:
                logger.info(
                    "WorkflowRegistration: reconciled workflow rule %r (%s → kb_update join)",
                    rule_name,
                    _rule_field(rule_dict, "SourceEvent"),
                )
            elif resp is not None:
                logger.warning(
                    "WorkflowRegistration: UPDATE_RULE returned success=false for %r: %s",
                    rule_name,
                    resp.error or resp.message or "(no detail)",
                )
            else:
                logger.warning(
                    "WorkflowRegistration: UPDATE_RULE timed out for %r (best-effort)",
                    rule_name,
                )
        except Exception:
            logger.warning(
                "WorkflowRegistration: UPDATE_RULE failed for %r (best-effort)",
                rule_name,
                exc_info=True,
            )
        return

    # ---- CREATE the rule -----------------------------------------------
    # If the LIST succeeded and the rule was simply absent, we CREATE. If the
    # LIST failed/was unparseable we also attempt CREATE; a duplicate-name
    # conflict there is harmless and logged below.
    try:
        create_op = aether_pb2.WorkflowOperation(
            op=aether_pb2.WorkflowOperation.CREATE_RULE,
            workspace="*",
            data=json.dumps(rule_dict).encode("utf-8"),
            request_id=str(uuid4()),
        )
        resp = await client.workflow_op(create_op, timeout=10.0)
        if resp is not None and resp.success:
            logger.info(
                "WorkflowRegistration: registered workflow rule %r (%s → kb_update join)",
                rule_name,
                _rule_field(rule_dict, "SourceEvent"),
            )
        elif resp is not None:
            logger.warning(
                "WorkflowRegistration: CREATE_RULE returned success=false for %r: %s",
                rule_name,
                resp.error or resp.message or "(no detail)",
            )
        else:
            logger.warning(
                "WorkflowRegistration: CREATE_RULE timed out for %r (best-effort)",
                rule_name,
            )
    except Exception:
        logger.warning(
            "WorkflowRegistration: CREATE_RULE failed for %r (best-effort)",
            rule_name,
            exc_info=True,
        )
