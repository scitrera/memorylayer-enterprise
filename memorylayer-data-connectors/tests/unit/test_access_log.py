"""Probe access lines are rolled up instead of logged one-per-request.

kubelet polls /healthz + /livez every few seconds forever while real traffic
arrives in bursts, so the access log is almost entirely probes at any moment
you go looking — which pushes the requests you actually need out of the
retained window.

The danger in a filter that drops log lines is dropping the WRONG ones, so
most of what follows pins down what still gets through.
"""
from __future__ import annotations

import logging

import pytest

from data_connectors.server.access_log import (
    ProbeAccessLogFilter,
    build_uvicorn_log_config,
    is_probe_path,
)


def _access_record(path: str, method: str = "GET", status: int = 200):
    """A record shaped exactly as uvicorn's access logger emits."""
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:1234", method, path, "1.1", status),
        exc_info=None,
    )


# =========================================================================
# what gets suppressed
# =========================================================================

@pytest.mark.parametrize("path", ["/healthz", "/livez", "/readyz", "/metrics"])
def test_probe_lines_are_suppressed(path):
    assert ProbeAccessLogFilter().filter(_access_record(path)) is False


def test_a_query_string_does_not_defeat_the_match():
    assert ProbeAccessLogFilter().filter(_access_record("/livez?verbose=1")) is False


def test_a_child_path_is_still_a_probe():
    assert is_probe_path("/healthz/ready") is True


# =========================================================================
# what must still be logged
# =========================================================================

@pytest.mark.parametrize("path", [
    "/v1/vfs/entries",
    "/v1/urls/upload",
    "/v1/providers",
    "/v1/admin/vfs/gc-abandoned-uploads",
])
def test_real_traffic_is_untouched(path):
    assert ProbeAccessLogFilter().filter(_access_record(path)) is True


def test_a_route_that_merely_shares_a_prefix_is_not_a_probe():
    """/healthz-internal is a different route, not a child of /healthz."""
    assert is_probe_path("/healthz-internal") is False
    assert ProbeAccessLogFilter().filter(_access_record("/healthz-internal")) is True


def test_an_unrecognised_record_shape_passes_through():
    """A filter must never be the reason a line vanishes.

    If uvicorn changes its access format, the path is no longer args[2] --
    fail open and keep logging rather than guess and silently drop.
    """
    rec = logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg="something else entirely", args=None, exc_info=None,
    )
    assert ProbeAccessLogFilter().filter(rec) is True


def test_a_short_arg_tuple_passes_through():
    rec = logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg="%s %s", args=("a", "b"), exc_info=None,
    )
    assert ProbeAccessLogFilter().filter(rec) is True


def test_a_non_string_path_passes_through():
    rec = logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg="%s %s %s", args=("a", "b", 42), exc_info=None,
    )
    assert ProbeAccessLogFilter().filter(rec) is True


# =========================================================================
# the rollup itself
# =========================================================================

def test_a_summary_is_emitted_once_the_window_elapses(caplog):
    """Probes accumulate silently, then one line reports the whole window.

    The flush rides on the next probe AFTER the interval passes, so the
    summary covers everything counted up to and including it.
    """
    f = ProbeAccessLogFilter(interval_seconds=60)

    with caplog.at_level(logging.INFO, logger="data_connectors.access.probes"):
        for _ in range(3):
            f.filter(_access_record("/livez"))
        assert caplog.text == "", "must not report before the window elapses"

        f._window_started -= 120  # the window has now passed
        f.filter(_access_record("/healthz"))

    assert "health probes: 4 requests" in caplog.text
    assert "/livez=3" in caplog.text
    assert "/healthz=1" in caplog.text


def test_counts_reset_after_a_flush(caplog):
    f = ProbeAccessLogFilter(interval_seconds=1)
    f._window_started -= 5
    with caplog.at_level(logging.INFO, logger="data_connectors.access.probes"):
        f.filter(_access_record("/livez"))
    assert f._counts == {}


def test_an_interval_of_zero_suppresses_without_summarising(caplog):
    f = ProbeAccessLogFilter(interval_seconds=0)
    with caplog.at_level(logging.INFO, logger="data_connectors.access.probes"):
        for _ in range(10):
            assert f.filter(_access_record("/livez")) is False
    assert caplog.text == ""


def test_a_negative_interval_is_clamped_not_treated_as_elapsed():
    """Otherwise it would flush on EVERY probe -- noisier than no filter."""
    assert ProbeAccessLogFilter(interval_seconds=-5)._interval == 0


# =========================================================================
# installation
# =========================================================================

def test_the_filter_is_attached_to_the_access_logger():
    config = build_uvicorn_log_config(300)

    assert "probe_rollup" in config["filters"]
    assert "probe_rollup" in config["loggers"]["uvicorn.access"]["filters"]


def test_the_rollup_logger_reaches_a_handler():
    """Without a handler the summary replaces the lines with nothing at all."""
    logger = build_uvicorn_log_config(300)["loggers"]["data_connectors.access.probes"]
    assert logger["handlers"] == ["default"]


def test_the_services_own_loggers_reach_a_handler():
    """uvicorn's config names only its own loggers, so everything this app
    logged -- including the abandoned-upload sweep's results -- went nowhere."""
    logger = build_uvicorn_log_config(300)["loggers"]["data_connectors"]
    assert logger["handlers"] == ["default"]
    assert logger["level"] == "INFO"


def test_upstream_formatters_and_handlers_are_preserved():
    from uvicorn.config import LOGGING_CONFIG

    config = build_uvicorn_log_config(300)
    assert config["formatters"] == LOGGING_CONFIG["formatters"]
    assert config["handlers"] == LOGGING_CONFIG["handlers"]


def test_building_the_config_does_not_mutate_uvicorns_own():
    """It is a module-level dict -- mutating it would leak across the process."""
    from uvicorn.config import LOGGING_CONFIG

    build_uvicorn_log_config(300)
    assert "probe_rollup" not in LOGGING_CONFIG.get("filters", {})
    assert "filters" not in LOGGING_CONFIG["loggers"]["uvicorn.access"]
