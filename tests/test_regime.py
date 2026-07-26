"""
test_regime.py
--------------
Causality and behaviour tests for the regime detector.

test_future_perturbation_does_not_change_past is the one that matters, and it
is the test the ORIGINAL qbeast_regime_detection.py fails. Point it at the HMM
detector before wiring that in -- fitting an HMM on the full sample and decoding
is the same look-ahead bug in a smarter costume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbeast_crash.data import load_symbol
from qbeast_crash.regime import (
    MARKET_REGIMES,
    VOL_REGIMES,
    detect_regimes,
    realized_hv,
)


def _prices(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.013, n)))


def test_future_perturbation_does_not_change_past():
    """Rewriting the tail must not move a single earlier label."""
    price = _prices()
    cut = 900

    mkt_a, vol_a = detect_regimes(price, realized_hv(price))
    tampered = price.copy()
    tampered[cut:] *= 0.35
    mkt_b, vol_b = detect_regimes(tampered, realized_hv(tampered))

    np.testing.assert_array_equal(mkt_a[:cut], mkt_b[:cut])
    np.testing.assert_array_equal(vol_a[:cut], vol_b[:cut])


def test_original_full_sample_approach_would_fail_this():
    """
    Demonstrates the bug being guarded against.

    Full-sample percentiles are recomputed here exactly as the original did.
    Perturbing the future changes past labels, which is the definition of
    look-ahead bias.
    """
    price = _prices()
    cut = 900

    def full_sample_labels(p):
        logp = np.log(p)
        cum = pd.Series(logp).diff().rolling(20).sum().to_numpy()
        ppd = cum / 20.0 * 100.0
        valid = ppd[np.isfinite(ppd)]
        c, ra = np.percentile(valid, 5), np.percentile(valid, 95)
        out = np.full(len(ppd), "Sideways", dtype=object)
        out[np.isfinite(ppd) & (ppd <= c)] = "Crashing"
        out[np.isfinite(ppd) & (ppd >= ra)] = "Rally"
        return out

    tampered = price.copy()
    tampered[cut:] *= 0.35
    assert not np.array_equal(
        full_sample_labels(price)[:cut], full_sample_labels(tampered)[:cut]
    ), "the original approach should leak the future -- if this passes, the demo is broken"


def test_streaming_equals_batch():
    """Bar-by-bar growth must reproduce the batch labels."""
    price = _prices(n=1200)
    mkt_batch, vol_batch = detect_regimes(price, realized_hv(price))
    for t in (800, 1000, 1199):
        mkt_s, vol_s = detect_regimes(price[: t + 1], realized_hv(price[: t + 1]))
        assert mkt_s[-1] == mkt_batch[t]
        assert vol_s[-1] == vol_batch[t]


def test_labels_are_from_the_declared_vocabulary():
    price = _prices()
    mkt, vol = detect_regimes(price, realized_hv(price))
    assert set(mkt) <= set(MARKET_REGIMES)
    assert set(vol) <= set(VOL_REGIMES)


def test_warmup_returns_neutral_defaults_not_null():
    """The original contract: downstream signal code never sees a null."""
    price = _prices(n=400)
    mkt, vol = detect_regimes(price, realized_hv(price), mkt_warmup=250)
    assert (mkt[:200] == "Sideways").all()
    assert None not in set(mkt) and None not in set(vol)


def test_length_mismatch_raises():
    price = _prices(n=300)
    with pytest.raises(ValueError, match="same length"):
        detect_regimes(price, realized_hv(price)[:200])


def test_covid_registers_as_crashing_on_real_data():
    frame, _ = load_symbol("RELIANCE")
    close = frame["close"].to_numpy()
    mkt, vol = detect_regimes(close, realized_hv(close))
    series = pd.Series(mkt, index=frame.index)
    volser = pd.Series(vol, index=frame.index)

    covid = series.loc["2020-03-15":"2020-04-15"]
    assert (covid == "Crashing").any()
    assert (volser.loc["2020-03-15":"2020-04-15"] == "HighVol").any()


def test_sideways_is_the_plurality_regime():
    """The 5/25/75/95 split should leave roughly half the days Sideways."""
    frame, _ = load_symbol("RELIANCE")
    mkt, _ = detect_regimes(frame["close"].to_numpy(), realized_hv(frame["close"].to_numpy()))
    share = pd.Series(mkt).value_counts(normalize=True)
    assert share["Sideways"] > 0.35
