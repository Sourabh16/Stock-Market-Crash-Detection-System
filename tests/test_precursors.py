"""
test_precursors.py
------------------
Causality and correctness tests for the per-stock precursor features and the
cross-sectional market block.

The causality tests carry the same contract as every other feature module:
rewrite the future, assert the past is bit-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbeast_crash.data.loader import load_symbol
from qbeast_crash.features.market import (
    MARKET_FEATURE_COLUMNS,
    _true_average_pairwise_correlation,
    average_pairwise_correlation,
    compute_market_features,
)
from qbeast_crash.features.precursors import FEATURE_COLUMNS, compute_precursors


@pytest.fixture(scope="module")
def reliance():
    frame, _ = load_symbol("RELIANCE")
    return frame


# =====================================================================
# Causality
# =====================================================================
def test_precursors_are_causal(reliance):
    """Rewriting the tail must not move a single earlier feature value."""
    cut = 3000
    base = compute_precursors(reliance)

    tampered = reliance.copy()
    price_cols = ["open", "high", "low", "close"]
    tampered.iloc[cut:, tampered.columns.get_indexer(price_cols)] *= 0.4
    tampered.iloc[cut:, tampered.columns.get_indexer(["volume"])] *= 7.0
    after = compute_precursors(tampered)

    for col in FEATURE_COLUMNS:
        pd.testing.assert_series_equal(
            base[col].iloc[:cut], after[col].iloc[:cut], check_names=False
        )


def test_precursors_streaming_equals_batch(reliance):
    """The live-trading contract: day-by-day must reproduce the batch result."""
    frame = reliance.iloc[-800:]
    batch = compute_precursors(frame)
    for t in (600, 700, 799):
        streamed = compute_precursors(frame.iloc[: t + 1])
        for col in FEATURE_COLUMNS:
            assert streamed[col].iloc[-1] == pytest.approx(
                batch[col].iloc[t], nan_ok=True, rel=1e-9
            ), f"{col} drifted at t={t}"


def test_market_features_are_causal():
    """Cross-sectional features must not leak the future either."""
    dates = pd.bdate_range("2018-01-01", periods=600)
    rng = np.random.default_rng(3)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.012, (600, 30)), axis=0)),
        index=dates, columns=[f"S{i}" for i in range(30)],
    )
    phase = pd.DataFrame("AcceleratingAdvance", index=dates, columns=close.columns)
    slope = pd.DataFrame(rng.normal(0, 1, (600, 30)), index=dates, columns=close.columns)

    cut = 400
    base = compute_market_features(close, phase, slope)
    tampered = close.copy()
    tampered.iloc[cut:] *= 0.3
    after = compute_market_features(tampered, phase, slope)

    for col in ("dispersion", "avg_corr", "pct_below_ma50"):
        pd.testing.assert_series_equal(
            base[col].iloc[:cut], after[col].iloc[:cut], check_names=False
        )


# =====================================================================
# The leading-feature contract
# =====================================================================
def test_same_day_return_never_reaches_the_model():
    """
    ret_1d is diagnostic only.

    Including it would make the detector coincident by construction -- it would
    flag a -6% day on the day it happened, which is not a prediction. This test
    is the guard against a convenience column quietly leaking into the model.
    """
    assert "ret_1d" not in FEATURE_COLUMNS
    assert "vol_short" not in FEATURE_COLUMNS
    assert "vol_long" not in FEATURE_COLUMNS


def test_all_declared_features_are_produced(reliance):
    out = compute_precursors(reliance)
    for col in FEATURE_COLUMNS:
        assert col in out.columns


def test_no_infinities_reach_the_model(reliance):
    """A single inf makes every finite value look identical by comparison."""
    out = compute_precursors(reliance)
    assert not np.isinf(out[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).any()


def test_features_are_populated_in_the_backtest_window(reliance):
    out = compute_precursors(reliance).loc["2021-01-01":]
    for col in FEATURE_COLUMNS:
        assert out[col].notna().mean() > 0.95, f"{col} is sparsely populated"


def test_warmup_is_nan_not_a_default(reliance):
    """
    Fabricated warmup values would form a dense, very 'normal' cluster --
    exactly the wrong lesson for an unsupervised detector.
    """
    out = compute_precursors(reliance)
    assert out["vol_ratio"].iloc[:59].isna().all()
    assert out["illiquidity"].iloc[:19].isna().all()


# =====================================================================
# Feature semantics
# =====================================================================
def test_vol_ratio_spikes_at_the_volatility_transition():
    dates = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(1)
    quiet = rng.normal(0, 0.004, 150)
    loud = rng.normal(0, 0.030, 50)
    close = pd.Series(100 * np.exp(np.cumsum(np.r_[quiet, loud])), index=dates)
    frame = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1e6, "is_flat": False,
    }, index=dates)

    out = compute_precursors(frame)
    calm = out["vol_ratio"].iloc[100:145].mean()
    transition = out["vol_ratio"].iloc[152:168].mean()

    # Measured at the transition, not 30 bars later. vol_ratio is 5d/60d, so
    # once the 60-day denominator fills with loud days the ratio decays back
    # toward 1 -- the same self-normalising effect that made a crash mute its
    # own slope_z. That decay is correct behaviour: this feature marks the
    # CHANGE of regime, not its persistence.
    assert transition > 2.0 * calm


def test_semidev_asym_detects_downside_asymmetry():
    dates = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(2)
    ret = rng.normal(0, 0.01, 120)
    # Mostly-down, but with occasional up days -- a ratio formulation would
    # return NaN on an all-down window, discarding the strongest signal.
    tail = -np.abs(rng.normal(0.02, 0.005, 25))
    tail[::7] = np.abs(rng.normal(0.004, 0.002, len(tail[::7])))
    ret[-25:] = tail
    close = pd.Series(100 * np.exp(np.cumsum(ret)), index=dates)
    frame = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1e6, "is_flat": False,
    }, index=dates)
    out = compute_precursors(frame)
    assert out["semidev_asym"].iloc[-1] > 0.4          # strongly downside-skewed
    assert -1.0 <= out["semidev_asym"].dropna().min() <= 1.0


def test_semidev_asym_is_defined_when_there_are_no_up_days():
    """The ratio form divides by zero exactly when asymmetry is maximal."""
    dates = pd.bdate_range("2020-01-01", periods=80)
    ret = np.r_[np.random.default_rng(6).normal(0, 0.01, 55), -np.full(25, 0.01)]
    close = pd.Series(100 * np.exp(np.cumsum(ret)), index=dates)
    frame = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1e6, "is_flat": False,
    }, index=dates)
    out = compute_precursors(frame)
    assert out["semidev_asym"].iloc[-1] == pytest.approx(1.0)


def test_flat_bars_are_masked_from_range_features():
    """BAJFINANCE has 1,092 flat bars; treating them as zero-range would drag
    the baseline down and make every ordinary day look like an expansion."""
    frame, _ = load_symbol("BAJFINANCE")
    out = compute_precursors(frame)
    flat = frame["is_flat"].astype(bool).reindex(out.index, fill_value=False)
    assert out.loc[flat, "range_expansion"].isna().all()


def test_dd_from_high_is_never_positive(reliance):
    out = compute_precursors(reliance)
    assert (out["dd_from_high"].dropna() <= 1e-9).all()


# =====================================================================
# Average pairwise correlation -- the O(N) shortcut
# =====================================================================
def test_fast_correlation_matches_direct_computation():
    """
    The variance-identity shortcut must track the true O(N^2) average.

    Measured on the real panel: r = 0.994, max deviation 0.046, 28x faster.
    """
    dates = pd.bdate_range("2019-01-01", periods=400)
    rng = np.random.default_rng(11)
    common = rng.normal(0, 0.01, 400)
    rets = pd.DataFrame(
        {f"S{i}": 0.6 * common + 0.8 * rng.normal(0, 0.01, 400) for i in range(25)},
        index=dates,
    )
    fast = average_pairwise_correlation(rets, 60)
    slow = _true_average_pairwise_correlation(rets, 60)
    both = pd.DataFrame({"f": fast, "s": slow}).dropna()

    assert len(both) > 250
    assert both["f"].corr(both["s"]) > 0.95
    assert (both["f"] - both["s"]).abs().mean() < 0.06


def test_correlation_is_bounded():
    dates = pd.bdate_range("2019-01-01", periods=300)
    rng = np.random.default_rng(5)
    rets = pd.DataFrame(rng.normal(0, 0.01, (300, 25)),
                        index=dates, columns=[f"S{i}" for i in range(25)])
    rho = average_pairwise_correlation(rets, 60).dropna()
    assert (rho >= -1).all() and (rho <= 1).all()


def test_identical_series_give_correlation_near_one():
    dates = pd.bdate_range("2019-01-01", periods=300)
    rng = np.random.default_rng(9)
    base = rng.normal(0, 0.01, 300)
    rets = pd.DataFrame({f"S{i}": base for i in range(25)}, index=dates)
    assert average_pairwise_correlation(rets, 60).dropna().mean() > 0.95


# =====================================================================
# Market block
# =====================================================================
def test_listing_mask_governs_the_denominator():
    """
    Breadth must be a fraction of companies that EXISTED.

    Without the mask, an unlisted symbol counts as 'not declining' and silently
    dilutes the measure.
    """
    dates = pd.bdate_range("2020-01-01", periods=200)
    cols = ["A", "B", "C", "D"]
    close = pd.DataFrame(100.0, index=dates, columns=cols)
    phase = pd.DataFrame("AcceleratingDecline", index=dates, columns=cols)
    slope = pd.DataFrame(-1.0, index=dates, columns=cols)

    mask = pd.DataFrame(True, index=dates, columns=cols)
    mask.loc[:, "D"] = False                       # D never listed

    out = compute_market_features(close, phase, slope, mask, min_symbols=1)
    assert (out["n_live"] == 3).all()
    assert out["breadth_decline"].iloc[-1] == pytest.approx(100.0)


def test_market_block_produces_all_columns():
    dates = pd.bdate_range("2018-01-01", periods=400)
    rng = np.random.default_rng(4)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (400, 30)), axis=0)),
        index=dates, columns=[f"S{i}" for i in range(30)],
    )
    phase = pd.DataFrame("Flat", index=dates, columns=close.columns)
    slope = pd.DataFrame(0.0, index=dates, columns=close.columns)
    out = compute_market_features(close, phase, slope)
    for col in MARKET_FEATURE_COLUMNS:
        assert col in out.columns
