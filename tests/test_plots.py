"""
test_plots.py
-------------
Tests for drawdown statistics and chart generation.

The drawdown maths carries the headline claim of the project, so it is tested
against hand-computed answers rather than against itself. The chart functions
get smoke tests -- a chart cannot be asserted correct, but it can be asserted
to exist and not to raise on the awkward inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbeast_crash.plots import (
    drawdown_series,
    drawdown_stats,
    plot_drawdown_scatter,
    plot_portfolio,
    plot_symbol,
)


def _equity(values, start="2021-01-04"):
    return pd.Series(np.asarray(values, dtype=float),
                     index=pd.bdate_range(start, periods=len(values)))


# =====================================================================
# Drawdown maths
# =====================================================================
def test_drawdown_is_zero_on_a_monotonic_rise():
    assert drawdown_series(_equity([1, 2, 3, 4, 5])).max() == pytest.approx(0.0)
    assert drawdown_series(_equity([1, 2, 3, 4, 5])).min() == pytest.approx(0.0)


def test_drawdown_measures_from_the_running_peak_not_the_start():
    """A fall from 200 to 100 is -50%, even though it started at 100."""
    dd = drawdown_series(_equity([100, 200, 100]))
    assert dd.iloc[-1] == pytest.approx(-0.5)


def test_max_drawdown_and_its_dates():
    eq = _equity([100, 120, 90, 110, 130])
    s = drawdown_stats(eq)
    assert s.max_drawdown == pytest.approx(-0.25)          # 120 -> 90
    assert s.max_dd_start == eq.index[1]
    assert s.max_dd_trough == eq.index[2]
    assert s.max_dd_recovered == eq.index[4]               # first back above 120
    assert s.time_to_recover == 2


def test_unrecovered_drawdown_reports_none_rather_than_guessing():
    """
    Never regaining the peak is the answer, not a missing value to paper over.
    """
    s = drawdown_stats(_equity([100, 150, 80, 90, 95]))
    assert s.max_dd_recovered is None
    assert s.time_to_recover is None


def test_time_under_water_counts_sessions_below_a_prior_peak():
    eq = _equity([100, 90, 95, 105, 100])
    s = drawdown_stats(eq)
    assert s.days_under_water == 3          # the 90, 95 and final 100 bars
    assert s.longest_underwater == 2        # 90, 95 before the new high


def test_longest_spell_differs_from_total_time_under_water():
    """Two short dips are a different experience from one long one."""
    eq = _equity([100, 90, 105, 100, 95, 92, 90, 110])
    s = drawdown_stats(eq)
    assert s.days_under_water > s.longest_underwater


def test_empty_equity_does_not_raise():
    s = drawdown_stats(pd.Series(dtype=float))
    assert np.isnan(s.max_drawdown)
    assert s.days_under_water == 0


# =====================================================================
# Charts
# =====================================================================
@pytest.fixture
def symbol_inputs():
    idx = pd.bdate_range("2021-01-04", periods=400)
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, 400))), index=idx)
    sig = pd.DataFrame({
        "in_position": True,
        "action": "",
        "exit_signal": False,
        "reentry_ok": True,
    }, index=idx)
    sig.iloc[150:170, sig.columns.get_loc("in_position")] = False
    sig.iloc[150, sig.columns.get_loc("action")] = "EXIT"
    sig.iloc[170, sig.columns.get_loc("action")] = "ENTER"
    return close, sig


def test_symbol_chart_is_written(tmp_path, symbol_inputs):
    close, sig = symbol_inputs
    path = plot_symbol("TEST", close, sig, out_dir=tmp_path)
    assert path is not None and path.exists() and path.stat().st_size > 5_000


def test_symbol_chart_handles_intensity_and_onsets(tmp_path, symbol_inputs):
    close, sig = symbol_inputs
    intensity = pd.Series(np.random.default_rng(1).uniform(0, 1, len(close)),
                          index=close.index)
    onsets = pd.DatetimeIndex([close.index[100], close.index[200]])
    assert plot_symbol("TEST", close, sig, intensity, onsets, tmp_path).exists()


def test_symbol_chart_returns_none_for_unusable_input(tmp_path):
    idx = pd.bdate_range("2021-01-04", periods=5)
    close = pd.Series([np.nan] * 5, index=idx)
    sig = pd.DataFrame({"in_position": [np.nan] * 5, "action": [""] * 5}, index=idx)
    assert plot_symbol("EMPTY", close, sig, out_dir=tmp_path) is None


def test_portfolio_chart_is_written(tmp_path):
    idx = pd.bdate_range("2021-01-04", periods=300)
    rng = np.random.default_rng(2)
    a = pd.Series(1e6 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 300))), index=idx)
    b = pd.Series(1e6 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 300))), index=idx)
    assert plot_portfolio(a, b, tmp_path).exists()


def test_scatter_splits_three_ways_not_two(tmp_path):
    """
    Unchanged symbols must not be counted as failures. The strategy does
    nothing for most names, and a two-way split would present that inaction as
    a deeper drawdown.
    """
    table = pd.DataFrame({
        "bh_max_drawdown": [-0.30, -0.40, -0.25, -0.50],
        "strategy_max_drawdown": [-0.30, -0.35, -0.28, -0.50],
    }, index=["FLAT1", "BETTER", "WORSE", "FLAT2"])
    assert plot_drawdown_scatter(table, tmp_path).exists()
