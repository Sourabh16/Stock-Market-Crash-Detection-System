"""
test_features.py
----------------
Correctness and causality tests for the Phase 2 feature block.

The causality tests are the important ones. They perturb the FUTURE and assert
the PAST does not move -- any feature that secretly fits on the full sample
fails immediately. That is the check the original regime module would fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbeast_crash.data import load_symbol
from qbeast_crash.features import (
    FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    _true_average_pairwise_correlation,
    average_pairwise_correlation,
    classify_phase,
    compute_market_features,
    compute_precursors,
    compute_slope_accel,
    realized_vol,
    rolling_accel,
    rolling_slope,
)

WARMUP = 60


@pytest.fixture(scope="module")
def reliance():
    frame, _ = load_symbol("RELIANCE")
    return frame



# =====================================================================
# Slope and acceleration
# =====================================================================

def _series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))


# =====================================================================
# Causality -- the tests that matter
# =====================================================================
@pytest.mark.parametrize("fn", [rolling_slope, rolling_accel, realized_vol])
def test_future_perturbation_does_not_change_past(fn):
    """Rewriting the tail must leave every earlier bar bit-identical."""
    price = _series()
    cut = 250

    base = fn(price)
    tampered = price.copy()
    tampered[cut:] *= 3.0                       # violent, obvious future change
    after = fn(tampered)

    np.testing.assert_array_equal(base[:cut], after[:cut])


def test_full_frame_is_causal():
    """Same check across every column of the assembled feature frame."""
    price = _series()
    cut = 250

    base = compute_slope_accel(price)
    tampered = price.copy()
    tampered[cut:] *= 0.4                       # simulate a crash that never was
    after = compute_slope_accel(tampered)

    for col in base.columns:
        pd.testing.assert_series_equal(
            base[col].iloc[:cut], after[col].iloc[:cut], check_names=False
        )


def test_streaming_equals_batch():
    """
    Growing the series one bar at a time must reproduce the batch result.

    This is the live-trading contract: what the model saw on day t in the
    backtest is exactly what it would have seen on day t in production.
    """
    price = _series(n=200)
    batch = compute_slope_accel(price)

    for t in (120, 160, 199):
        streamed = compute_slope_accel(price[: t + 1])
        for col in ("slope", "accel", "vol", "slope_z", "accel_z"):
            assert streamed[col].iloc[-1] == pytest.approx(
                batch[col].iloc[t], nan_ok=True, rel=1e-12
            ), f"{col} drifted at t={t}"


# =====================================================================
# Correctness against known analytic answers
# =====================================================================
def test_slope_recovers_constant_growth():
    """A fixed 1%/day compounding series must give slope == 1%/day."""
    price = 100.0 * np.exp(np.arange(100) * 0.01)
    slope = rolling_slope(price, window=5)
    assert slope[10:] == pytest.approx(1.0, abs=1e-9)


def test_accel_is_zero_on_pure_exponential():
    """Constant growth has no curvature in log space."""
    price = 100.0 * np.exp(np.arange(100) * 0.01)
    accel = rolling_accel(price, window=7)
    assert accel[10:] == pytest.approx(0.0, abs=1e-9)


def test_accel_sign_matches_curvature():
    """Quadratic log price: accel must equal 2c and hold its sign."""
    t = np.arange(120, dtype=float)
    for c2 in (0.0004, -0.0004):
        price = np.exp(4.0 + 0.001 * t + c2 * t ** 2)
        accel = rolling_accel(price, window=7)
        assert accel[20:] == pytest.approx(2.0 * c2 * 100.0, abs=1e-7)


def test_slope_is_scale_invariant():
    """
    A 10:1 split must not change the trend reading. Log space guarantees it.

    Compared on an absolute tolerance: slope legitimately passes through zero,
    where a relative tolerance is meaningless.
    """
    price = _series()
    np.testing.assert_allclose(
        rolling_slope(price), rolling_slope(price / 10.0), rtol=1e-9, atol=1e-12
    )


def test_robust_slope_resists_a_bad_print():
    """One corrupted bar should move Theil-Sen far less than OLS."""
    price = 100.0 * np.exp(np.arange(80) * 0.005)
    spiked = price.copy()
    spiked[50] *= 1.5                            # single bad tick

    at = 52
    ols_shift = abs(rolling_slope(spiked, 7)[at] - rolling_slope(price, 7)[at])
    ts_shift = abs(
        rolling_slope(spiked, 7, robust=True)[at]
        - rolling_slope(price, 7, robust=True)[at]
    )
    assert ts_shift < ols_shift


# =====================================================================
# Warmup and bad-data handling
# =====================================================================
def test_slope_warmup_is_nan_not_a_default():
    """
    Bars without a full window must be NaN.

    The regime module returned "Sideways" during warmup, which is
    indistinguishable downstream from a genuine sideways reading.
    """
    price = _series(n=100)
    assert np.all(np.isnan(rolling_slope(price, 5)[:4]))
    assert np.all(np.isnan(rolling_accel(price, 7)[:6]))
    assert np.all(np.isnan(realized_vol(price, 60)[:59]))
    assert np.isfinite(rolling_slope(price, 5)[4])


def test_nonpositive_prices_become_nan_not_inf():
    price = _series(n=80)
    price[40] = 0.0
    slope = rolling_slope(price, 5)
    assert np.isnan(slope[40:45]).all()
    assert not np.isinf(slope).any()
    assert np.isfinite(slope[46:]).all()


def test_flat_bars_give_zero_slope_not_nan():
    """Repeated identical closes are common in the raw data (BAJFINANCE: 19%)."""
    price = np.concatenate([_series(n=60), np.full(20, 150.0)])
    slope = rolling_slope(price, 5)
    assert slope[-1] == pytest.approx(0.0, abs=1e-9)


def test_index_length_mismatch_raises():
    with pytest.raises(ValueError, match="index length"):
        compute_slope_accel(_series(n=50), index=pd.RangeIndex(49))


# =====================================================================
# Phase classification
# =====================================================================
def test_phase_quadrants():
    slope_z = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    accel_z = np.array([-1.0, 1.0, 1.0, -1.0, 0.0])
    assert list(classify_phase(slope_z, accel_z)) == [
        "AcceleratingDecline",
        "DeceleratingDecline",
        "AcceleratingAdvance",
        "DeceleratingAdvance",
        "Flat",
    ]


def test_deadband_suppresses_noise_phases():
    """Sub-threshold readings collapse to Flat -- phase churn becomes trade churn."""
    tiny = np.array([0.01, -0.01, 0.02])
    assert set(classify_phase(tiny, tiny)) == {"Flat"}


def test_phase_is_none_where_inputs_are_nan():
    out = classify_phase(np.array([np.nan, -1.0]), np.array([np.nan, -1.0]))
    assert out[0] is None and out[1] == "AcceleratingDecline"


# =====================================================================
# Behaviour on real data
# =====================================================================
def _reliance():
    raw = pd.read_csv("data/raw/RELIANCE.csv", usecols=["date", "close"])
    raw["date"] = pd.to_datetime(raw["date"], format="%d-%m-%Y")
    return raw.set_index("date").sort_index()


def test_detects_covid_crash_on_real_reliance_data():
    """
    End-to-end check on the actual CSV: March 2020 must register as an
    accelerating decline that is extreme against the stock's OWN history.

    Ranking against the full history beats comparing to a hand-picked "calm"
    window -- the obvious choice of mid-2019 is not calm at all, since the
    August 2019 Aramco announcement produced a rally the detector correctly
    flags as a large positive slope.
    """
    raw = _reliance()
    feats = compute_slope_accel(raw["close"].to_numpy(), index=raw.index)
    crash = feats.loc["2020-03-15":"2020-03-31"]

    assert (crash["phase"] == "AcceleratingDecline").any()

    floor = feats["slope_z"].quantile(0.01)
    assert crash["slope_z"].min() < floor, (
        "COVID should sit in the worst 1% of this stock's 26-year slope_z history"
    )


def test_vol_lag_stops_a_crash_muting_its_own_signal():
    """
    Regression test for the lag=0 flaw: a contemporaneous volatility
    denominator fills with crash days and deflates the very move it measures.
    """
    raw = _reliance()
    close, index = raw["close"].to_numpy(), raw.index

    naive = compute_slope_accel(close, index=index, vol_lag=0)
    lagged = compute_slope_accel(close, index=index, vol_lag=20)

    window = slice("2020-03-15", "2020-03-31")
    assert lagged["slope_z"].loc[window].min() < naive["slope_z"].loc[window].min() - 1.0


def test_vol_lag_is_still_causal():
    price = _series()
    cut = 250
    base = compute_slope_accel(price, vol_lag=20)
    tampered = price.copy()
    tampered[cut:] *= 0.4
    after = compute_slope_accel(tampered, vol_lag=20)
    pd.testing.assert_series_equal(
        base["slope_z"].iloc[:cut], after["slope_z"].iloc[:cut], check_names=False
    )


# =====================================================================
# Precursors and market state
# =====================================================================



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


def test_precursor_warmup_is_nan_not_a_default(reliance):
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
