"""
test_signals.py
---------------
Tests for the hold/cash state machine and the market overlay.

The causality test is the important one. A state machine is easy to write in a
way that acts on the same bar's close, which is a time machine, and no
backtest would complain -- it would just report a better number.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from qbeast_crash.config import DEFAULT_CONFIG
from qbeast_crash.signals import (
    ReentryRule,
    SignalConfig,
    equal_weight_equity,
    generate_signals,
    market_signal,
    performance,
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


# =====================================================================
# Portfolio aggregation
# =====================================================================
def test_equal_weight_equity_is_a_portfolio_not_a_geometric_mean():
    """
    Averaging per-stock LOG equity and exponentiating gives the geometric mean
    of individual outcomes, not a portfolio -- it discards the diversification
    benefit. Measured on the real universe, that understated CAGR by 4.75pp.

    Two anti-correlated assets are the clean demonstration: a rebalanced
    portfolio of them grows, while the geometric mean of the pair does not.
    """
    idx = pd.bdate_range("2021-01-04", periods=200)
    rng = np.random.default_rng(0)
    a = rng.normal(0.0005, 0.02, 200)
    ret = pd.DataFrame({"A": a, "B": -a}, index=idx)

    portfolio = equal_weight_equity(ret)
    geometric = np.exp(np.log1p(ret).cumsum().mean(axis=1))

    assert portfolio.iloc[-1] > geometric.iloc[-1]


def test_equal_weight_equity_matches_a_single_asset():
    idx = pd.bdate_range("2021-01-04", periods=100)
    ret = pd.DataFrame({"A": np.full(100, 0.01)}, index=idx)
    assert equal_weight_equity(ret).iloc[-1] == pytest.approx(1.01 ** 100)


def test_positions_gate_the_returns():
    """A stock held out of the portfolio must contribute nothing."""
    idx = pd.bdate_range("2021-01-04", periods=50)
    ret = pd.DataFrame({"A": np.full(50, 0.02)}, index=idx)
    pos = pd.DataFrame({"A": [True] * 25 + [False] * 25}, index=idx)

    held = equal_weight_equity(ret, pos)
    assert held.iloc[-1] == pytest.approx(1.02 ** 25)
    assert held.iloc[-1] < equal_weight_equity(ret).iloc[-1]


def test_performance_computes_cagr_and_drawdown():
    idx = pd.bdate_range("2021-01-04", periods=253)
    eq = pd.Series(np.linspace(1.0, 2.0, 253), index=idx)
    stats = performance(eq, years=1.0)
    assert stats["cagr"] == pytest.approx(1.0, abs=1e-9)
    assert stats["max_drawdown"] == pytest.approx(0.0, abs=1e-12)

    dipped = pd.Series([1.0, 1.5, 0.9, 1.2], index=pd.bdate_range("2021-01-04", periods=4))
    assert performance(dipped, years=1.0)["max_drawdown"] == pytest.approx(-0.4)


def test_performance_handles_empty_input():
    stats = performance(pd.Series(dtype=float), years=1.0)
    assert np.isnan(stats["cagr"]) and np.isnan(stats["max_drawdown"])


# =====================================================================
# The two-tier logic: severe acts immediately, mild opens a watch
# =====================================================================
def _watch_frame(n=200, intensity=0.5, phase="Flat", slope=0.0, ret=0.0, regime="Sideways"):
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({
        "intensity": np.full(n, intensity),
        "phase": np.array([phase] * n, dtype=object),
        "slope_z": np.full(n, slope),
        "ret_1d": np.full(n, ret),
        "vol": np.ones(n),
        "regime": np.array([regime] * n, dtype=object),
    }, index=idx)


def test_severe_anomaly_acts_the_next_day_without_waiting():
    """
    The edge is concentrated at a one-day horizon and decays fast, so a severe
    anomaly must not wait for confirmation.
    """
    f = _watch_frame()
    f.iloc[50, f.columns.get_loc("intensity")] = 0.999
    f.iloc[50, f.columns.get_loc("phase")] = "AcceleratingDecline"

    sig = generate_signals(f)
    assert sig["action"].iloc[50] == "EXIT"
    assert sig["in_position"].iloc[50]          # still held ON the signal day
    assert not sig["in_position"].iloc[51]      # out from the next day


def test_mild_anomaly_opens_a_watch_instead_of_selling():
    f = _watch_frame(slope=-0.5, regime="Sideways")
    f.iloc[50, f.columns.get_loc("intensity")] = 0.96      # mild
    f.iloc[50, f.columns.get_loc("phase")] = "DeceleratingDecline"

    sig = generate_signals(f)
    assert sig["watching"].iloc[51:55].any(), "a watch should be open"
    assert (sig["action"] == "EXIT").sum() == 0, "a mild anomaly must not sell outright"


def test_watch_sells_when_the_fall_accelerates_and_regime_confirms():
    """
    Inside the watch: today's move worse than the 5-day slope means the decline
    is steepening. With the regime confirming, sell without waiting out the
    remaining days.
    """
    f = _watch_frame(slope=-0.5, ret=0.0, regime="Crashing")
    f.iloc[50, f.columns.get_loc("intensity")] = 0.96
    f.iloc[50, f.columns.get_loc("phase")] = "DeceleratingDecline"
    # day 52: a sharp down day, worse than the prevailing 5-day slope
    f.iloc[52, f.columns.get_loc("ret_1d")] = -3.0

    sig = generate_signals(f)
    exits = np.flatnonzero((sig["action"] == "EXIT_WATCH").to_numpy())
    assert exits.size >= 1
    assert 50 <= exits[0] <= 55, "should sell inside the watch window"


def test_watch_stands_down_if_the_stock_turns_up():
    f = _watch_frame(slope=-0.5, regime="Crashing")
    f.iloc[50, f.columns.get_loc("intensity")] = 0.96
    f.iloc[50, f.columns.get_loc("phase")] = "DeceleratingDecline"
    f.iloc[51:, f.columns.get_loc("slope_z")] = +0.8        # recovers

    sig = generate_signals(f)
    assert (sig["action"] == "EXIT_WATCH").sum() == 0
    assert sig["in_position"].all()


def test_watch_expires_after_watch_days():
    """A watch that resolves into nothing must close, not linger."""
    f = _watch_frame(slope=-0.2, regime="Sideways")
    f.iloc[50, f.columns.get_loc("intensity")] = 0.96
    f.iloc[50, f.columns.get_loc("phase")] = "DeceleratingDecline"

    cfg = dataclasses.replace(DEFAULT_CONFIG.signals, watch_days=5)
    sig = generate_signals(f, cfg)
    assert not sig["watching"].iloc[60:].any(), "watch should have expired"


def test_regime_can_veto_a_watch_exit():
    """The backdrop check is a real gate, not decoration."""
    f = _watch_frame(slope=-0.5, regime="Rally")       # regime disagrees
    f.iloc[50, f.columns.get_loc("intensity")] = 0.96
    f.iloc[50, f.columns.get_loc("phase")] = "DeceleratingDecline"
    f.iloc[52, f.columns.get_loc("ret_1d")] = -3.0

    assert (generate_signals(f)["action"] == "EXIT_WATCH").sum() == 0


def test_mild_path_still_works_without_a_regime_column():
    """Older frames have no regime; the check must pass rather than crash."""
    f = _watch_frame(slope=-0.5).drop(columns=["regime"])
    f.iloc[50, f.columns.get_loc("intensity")] = 0.96
    f.iloc[50, f.columns.get_loc("phase")] = "DeceleratingDecline"
    f.iloc[52, f.columns.get_loc("ret_1d")] = -3.0
    sig = generate_signals(f)
    assert (sig["action"] == "EXIT_WATCH").sum() >= 1
