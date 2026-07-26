"""
test_labels.py
--------------
Tests for crash/rally labelling and lead-time measurement.

These guard the EVALUATION, which matters as much as guarding the model: a bug
here would not crash anything, it would just quietly report a better number
than the truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbeast_crash.labels import (
    CrashDefinition,
    forward_drawdown,
    forward_runup,
    label_events,
    lead_time_report,
)


def _series(values, start="2021-01-04"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=idx)


# =====================================================================
# Forward drawdown
# =====================================================================
def test_forward_drawdown_uses_the_worst_point_not_the_endpoint():
    """
    A stock that falls 20% and recovers has still caused a 20% drawdown.

    A close-to-close measure would score that week as flat, which is exactly
    the mistake this definition exists to avoid.
    """
    close = _series([100, 90, 80, 95, 100, 100])
    dd = forward_drawdown(close, horizon=4)
    assert dd.iloc[0] == pytest.approx(-0.20)


def test_forward_drawdown_respects_the_horizon():
    close = _series([100, 99, 98, 50, 50])
    assert forward_drawdown(close, horizon=2).iloc[0] == pytest.approx(-0.02)
    assert forward_drawdown(close, horizon=3).iloc[0] == pytest.approx(-0.50)


def test_forward_drawdown_is_nan_at_the_end():
    close = _series([100] * 5)
    assert np.isnan(forward_drawdown(close, horizon=3).iloc[-1])


def test_forward_runup_mirrors_drawdown():
    close = _series([100, 120, 100, 100])
    assert forward_runup(close, horizon=3).iloc[0] == pytest.approx(0.20)


# =====================================================================
# Event detection
# =====================================================================
def test_a_crash_is_labelled():
    close = _series([100] * 10 + [85] + [85] * 10)
    ev = label_events(close, "TEST", CrashDefinition(crash_threshold=-0.10))
    assert ev.crash_days.any()
    assert len(ev.crash_onsets) == 1


def test_overlapping_days_collapse_into_one_event():
    """
    Without merging, one crash produces a run of daily 'events' and every
    recall figure is inflated by counting the same episode many times.
    """
    close = _series(list(100 * np.exp(np.cumsum([0.0] * 5 + [-0.05] * 8 + [0.0] * 12))))
    ev = label_events(close, "TEST", CrashDefinition(crash_threshold=-0.10, min_gap=10))
    assert ev.crash_days.sum() > 3          # many days flagged
    assert len(ev.crash_onsets) == 1        # but one event


def test_distant_crashes_stay_separate():
    calm = [100.0] * 15
    close = _series(calm + [80.0] * 15 + [80.0] * 15 + [64.0] * 10)
    ev = label_events(close, "TEST", CrashDefinition(crash_threshold=-0.10, min_gap=10))
    assert len(ev.crash_onsets) >= 2


def test_no_crash_in_a_calm_series():
    rng = np.random.default_rng(0)
    close = _series(100 * np.exp(np.cumsum(rng.normal(0, 0.002, 200))))
    ev = label_events(close, "TEST", CrashDefinition(crash_threshold=-0.10))
    assert len(ev.crash_onsets) == 0


def test_stock_and_index_thresholds_differ():
    """
    Individual stocks are ~2x as volatile as the index. Applying the
    index-calibrated -5% to stocks produced 7.6 'crashes' per stock per year --
    ordinary pullbacks, not crashes.
    """
    d = CrashDefinition()
    assert d.crash_threshold == -0.10
    assert d.index_crash_threshold == -0.05
    assert d.crash_threshold < d.index_crash_threshold


# =====================================================================
# Lead time
# =====================================================================
def _cal(n=100):
    return pd.bdate_range("2021-01-04", periods=n)


def test_lead_time_counts_trading_sessions_not_calendar_days():
    """A weekend is not two days of warning."""
    cal = _cal()
    onset = cal[50]
    signals = pd.Series(False, index=cal)
    signals.iloc[47] = True                       # 3 sessions earlier

    rep = lead_time_report(signals, pd.DatetimeIndex([onset]), cal)
    assert rep["leads"] == [3]
    assert rep["median_lead"] == 3


def test_earliest_alert_wins():
    cal = _cal()
    signals = pd.Series(False, index=cal)
    signals.iloc[44] = True
    signals.iloc[49] = True
    rep = lead_time_report(signals, pd.DatetimeIndex([cal[50]]), cal)
    assert rep["leads"] == [6]


def test_a_missed_event_is_recorded_not_dropped():
    """
    Misses must survive into the report.

    Reporting a mean lead over only the caught events would look excellent
    while hiding every event the detector never saw.
    """
    cal = _cal()
    signals = pd.Series(False, index=cal)
    rep = lead_time_report(signals, pd.DatetimeIndex([cal[50]]), cal)
    assert rep["leads"] == [None]
    assert rep["distribution"]["missed"] == 1
    assert rep["n_caught"] == 0


def test_alerts_beyond_the_lookback_do_not_count():
    cal = _cal()
    signals = pd.Series(False, index=cal)
    signals.iloc[20] = True                       # 30 sessions before onset
    rep = lead_time_report(signals, pd.DatetimeIndex([cal[50]]), cal, max_lookback=15)
    assert rep["n_caught"] == 0


def test_lead_zero_still_counts_as_a_warning():
    """
    The label is forward-looking, so day t is the last day BEFORE the fall.
    A lead of 0 is a one-day warning in trading terms -- enough to exit at the
    next open.
    """
    cal = _cal()
    signals = pd.Series(False, index=cal)
    signals.iloc[50] = True
    rep = lead_time_report(signals, pd.DatetimeIndex([cal[50]]), cal)
    assert rep["leads"] == [0]
    assert rep["distribution"]["0-1 days"] == 1


def test_false_alarms_are_counted():
    cal = _cal()
    signals = pd.Series(False, index=cal)
    signals.iloc[10] = True                       # nowhere near the onset
    signals.iloc[48] = True                       # genuine warning
    rep = lead_time_report(signals, pd.DatetimeIndex([cal[50]]), cal, max_lookback=15)
    assert rep["false_alarms"] == 1
    assert rep["n_caught"] == 1


def test_hit_rate_is_monotone_in_horizon():
    cal = _cal()
    signals = pd.Series(False, index=cal)
    signals.iloc[47] = True
    rep = lead_time_report(signals, pd.DatetimeIndex([cal[50]]), cal)
    rates = [rep["hit_rate"][h] for h in (0, 1, 2, 3, 5, 10)]
    assert rates == sorted(rates)


def test_report_handles_no_events():
    cal = _cal()
    signals = pd.Series(True, index=cal)
    rep = lead_time_report(signals, pd.DatetimeIndex([]), cal)
    assert rep["n_events"] == 0
    assert np.isnan(rep["median_lead"])
