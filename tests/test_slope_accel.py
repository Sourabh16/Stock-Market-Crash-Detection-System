"""
test_slope_accel.py
-------------------
Correctness and causality tests for qbeast_crash.features.slope_accel.

The causality tests are the important ones. They perturb the FUTURE and assert
the PAST does not move. Any feature that secretly fits on the full sample fails
this immediately -- it is the check that would have caught the full-sample
percentile bug in qbeast_regime_detection._classify_market_series_percentile.

Run:  python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbeast_crash.features.slope_accel import (
    classify_phase,
    compute_slope_accel,
    realized_vol,
    rolling_accel,
    rolling_slope,
)

WARMUP = 60  # longest window used in the default config


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
def test_warmup_is_nan_not_a_default():
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
