"""
test_signals.py
---------------
Tests for the hold/cash state machine and the market overlay.

The causality test is the important one. A state machine is easy to write in a
way that acts on the same bar's close, which is a time machine, and no
backtest would complain -- it would just report a better number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbeast_crash.signals import (
    ReentryRule,
    SignalConfig,
    generate_signals,
    market_signal,
    signal_strength,
)


def _frame(n=200, intensity=0.5, phase="Flat"):
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(
        {"intensity": np.full(n, intensity),
         "phase": np.array([phase] * n, dtype=object),
         "slope_z": np.zeros(n)},
        index=idx,
    )


# =====================================================================
# Causality -- decisions apply from the NEXT bar
# =====================================================================
def test_exit_takes_effect_the_day_after_the_signal():
    """
    The signal is computed at the close, so acting on that same close would
    require a time machine. `in_position` must reflect the state actually
    achievable.
    """
    f = _frame()
    f.iloc[50, f.columns.get_loc("intensity")] = 0.999
    f.iloc[50, f.columns.get_loc("phase")] = "AcceleratingDecline"

    sig = generate_signals(f)
    assert sig["action"].iloc[50] == "EXIT"
    assert sig["in_position"].iloc[50]           # still held ON the signal day
    assert not sig["in_position"].iloc[51]       # out from the next bar


def test_future_does_not_change_the_past():
    f = _frame(n=300)
    rng = np.random.default_rng(0)
    f["intensity"] = rng.uniform(0, 1, 300)
    f["phase"] = np.where(rng.random(300) > 0.7, "AcceleratingDecline", "Flat")

    base = generate_signals(f)
    tampered = f.copy()
    tampered.iloc[200:, tampered.columns.get_loc("intensity")] = 1.0
    after = generate_signals(tampered)

    pd.testing.assert_series_equal(
        base["in_position"].iloc[:200], after["in_position"].iloc[:200],
        check_names=False,
    )


# =====================================================================
# The state machine
# =====================================================================
def test_long_is_the_default_state():
    """Being invested is the default; the model only decides when to leave."""
    assert generate_signals(_frame())["in_position"].all()


def test_exit_needs_both_intensity_and_direction():
    """
    Isolation Forest is direction-blind: a violent rally scores as high as a
    crash. Intensity alone must never trigger an exit.
    """
    f = _frame(intensity=0.999, phase="AcceleratingAdvance")
    assert generate_signals(f)["in_position"].all()
    assert not (generate_signals(f)["action"] == "EXIT").any()


def test_cooldown_blocks_immediate_reentry():
    f = _frame()
    f.iloc[50, f.columns.get_loc("intensity")] = 0.999
    f.iloc[50, f.columns.get_loc("phase")] = "AcceleratingDecline"

    cfg = SignalConfig(cooldown=10)
    sig = generate_signals(f, cfg, reentry=ReentryRule.NOT_DECLINING)
    entries = np.flatnonzero((sig["action"] == "ENTER").to_numpy())
    assert entries.size == 1
    assert entries[0] - 50 > cfg.cooldown


def test_min_hold_blocks_immediate_re_exit():
    f = _frame()
    for i in (50, 51, 52, 53):
        f.iloc[i, f.columns.get_loc("intensity")] = 0.999
        f.iloc[i, f.columns.get_loc("phase")] = "AcceleratingDecline"
    sig = generate_signals(f, SignalConfig(min_hold=20, cooldown=1))
    assert int((sig["action"] == "EXIT").sum()) == 1


def test_forced_reentry_caps_time_in_cash():
    """
    An indefinite cash position is a bet the model was never asked to make.
    Long is the default state, so cash must be time-limited.
    """
    f = _frame(intensity=0.999, phase="AcceleratingDecline")
    cfg = SignalConfig(max_cash_days=10)
    sig = generate_signals(f, cfg, reentry=ReentryRule.RALLY_SIGNAL)
    # Re-entry can never fire on its own here, so only the cap can bring us back.
    entries = int((sig["action"] == "ENTER").sum())
    assert entries > 5, "the cap should force repeated re-entries"
    # Cycle is min_hold in, max_cash_days out, so roughly 3 in 13.
    assert 0.15 < sig["in_position"].mean() < 0.45


def test_persistence_requires_consecutive_days():
    f = _frame()
    f.iloc[50, f.columns.get_loc("intensity")] = 0.999
    f.iloc[50, f.columns.get_loc("phase")] = "AcceleratingDecline"
    assert not (generate_signals(f, SignalConfig(exit_persistence=3))["action"] == "EXIT").any()

    for i in (50, 51, 52):
        f.iloc[i, f.columns.get_loc("intensity")] = 0.999
        f.iloc[i, f.columns.get_loc("phase")] = "AcceleratingDecline"
    assert (generate_signals(f, SignalConfig(exit_persistence=3))["action"] == "EXIT").any()


@pytest.mark.parametrize("rule", ReentryRule.ALL)
def test_every_reentry_rule_runs_and_returns(rule):
    f = _frame(n=300)
    rng = np.random.default_rng(1)
    f["intensity"] = rng.uniform(0, 1, 300)
    f["phase"] = rng.choice(
        ["AcceleratingDecline", "DeceleratingDecline", "AcceleratingAdvance", "Flat"], 300
    )
    sig = generate_signals(f, reentry=rule)
    assert sig["in_position"].dtype == bool
    assert len(sig) == 300


def test_unknown_reentry_rule_raises():
    with pytest.raises(ValueError, match="unknown re-entry rule"):
        generate_signals(_frame(), reentry="nonsense")


# =====================================================================
# Market overlay
# =====================================================================
def test_market_signal_needs_both_conditions():
    """
    Breadth alone cannot separate a systemic crash from a broad pullback --
    ~90% of stocks were declining on both COVID and mild pullbacks. Only the
    median slope distinguishes them.
    """
    m = pd.DataFrame({
        "breadth_decline": [90.0, 90.0, 40.0, 80.0],
        "median_slope_z": [-2.5, -0.5, -2.5, -1.6],
    })
    assert list(market_signal(m)) == [True, False, False, True]


def test_market_signal_handles_missing_data():
    m = pd.DataFrame({"breadth_decline": [np.nan, 90.0], "median_slope_z": [-2.0, np.nan]})
    assert not market_signal(m).any()


# =====================================================================
# Ranking
# =====================================================================
def test_signal_strength_ranks_by_intensity_and_move_size():
    f = pd.DataFrame({"intensity": [0.99, 0.99, 0.50], "slope_z": [-3.0, -0.5, -3.0]})
    s = signal_strength(f)
    assert s.iloc[0] > s.iloc[1] and s.iloc[0] > s.iloc[2]


def test_signal_strength_is_never_nan():
    f = pd.DataFrame({"intensity": [np.nan, 0.99], "slope_z": [-1.0, np.nan]})
    assert signal_strength(f).notna().all()
