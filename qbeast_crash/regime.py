"""
regime.py
---------
Regime detection, adapted from the team's qbeast_regime_detection.py.

What it does:   Labels each bar with a market regime (Crashing / Falling /
                Sideways / Rising / Rally) and a volatility regime (High / Med /
                LowVol).
Why we do it:   Anomaly intensity says "today is unusual" and slope/accel say
                "in which direction". Regime supplies the backdrop: the same
                anomaly means different things in a calm uptrend and in an
                unwinding market.
How (method):   Trailing cumulative log-return of a smoothed price series,
                classified against ROLLING percentiles of the symbol's own
                history. Volatility gets the same treatment on realised HV.
Where:          regime.py -> detect_regimes()

ROLE IN THE PIPELINE
--------------------
Regime is a CONTEXT FILTER, never a trigger. Its 20-bar window cannot say
"Crashing" until a crash is largely over, which is fine -- Isolation Forest
fires fast and regime confirms slowly. Do not try to speed it up; a fast regime
detector is just a noisy one.

WHAT CHANGED FROM THE ORIGINAL, AND WHY
---------------------------------------
1. LOOK-AHEAD BUG FIXED (the important one). The original
   `_classify_market_series_percentile` computed its percentile thresholds from
   the ENTIRE series at once:

       valid = pct_per_day[np.isfinite(pct_per_day)]   # the whole array
       c = np.percentile(valid, pct_crash)

   and applied them to every bar, so a 2016 label used the return distribution
   through 2026. The module docstring claimed "Causal (no forward bias)", but
   that held only for the volatility regime, which correctly used `hv[:i+1]`.

   Measured on RELIANCE 2016-2026: 9.9% of labels change once made causal, and
   the biased version produced MORE Rally and MORE Crashing calls -- knowing the
   full distribution, it placed bars confidently into the tails. Those are
   exactly the labels entries and exits key off.

2. Rolling, not expanding. A pure expanding window ranks a 2026 bar against the
   2003 distribution, and Indian equity volatility has shifted structurally
   since. A trailing 3-year window keeps thresholds adaptive.

3. Warmup added for the market regime. The original had none, so early bars got
   thresholds fitted on a handful of points.

4. Vectorised. The original volatility loop recomputed percentiles over a
   growing slice per bar (O(N^2)); pandas rolling quantile is equivalent and
   far faster across a 100-symbol sweep.

5. `locus` is now optional. Passing raw close is fine and preferred: the
   trailing cumulative return is already a smoother, and a P-spline fitted over
   the full sample would reintroduce look-ahead through the back door.

The public signature is unchanged, so the HMM detector the other team is
building can drop in behind `RegimeDetector` without touching callers.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

__all__ = [
    "MARKET_REGIMES",
    "VOL_REGIMES",
    "RegimeDetector",
    "detect_regimes",
    "realized_hv",
]

MARKET_REGIMES = ("Crashing", "Falling", "Sideways", "Rising", "Rally")
VOL_REGIMES = ("HighVol", "MedVol", "LowVol")

#: Trailing window for percentile thresholds. ~3 years of sessions.
DEFAULT_LOOKBACK = 750

#: Percentile cut points. Bottom 5% Crashing, 5-25% Falling, 25-75% Sideways,
#: 75-95% Rising, top 5% Rally -- roughly half the days Sideways, which matches
#: Indian equities and leaves room for signals to fire.
_PCTS = {"crash": 5, "fall": 25, "rise": 75, "rally": 95}


class RegimeDetector(Protocol):
    """
    Interface every regime detector must satisfy.

    The HMM replacement should implement this and nothing else. Two rules bind
    any implementation:

    1. CAUSAL. The label at bar t may use data up to and including bar t, never
       beyond. For an HMM this means refitting as you go, or at minimum using
       FILTERED state probabilities P(state_t | data up to t) rather than
       SMOOTHED ones P(state_t | all data). Fitting an HMM on the full sample
       and then decoding is the same look-ahead bug in a more sophisticated
       costume, and it is the usual way this goes wrong.

    2. Warmup bars return the neutral defaults ("Sideways", "MedVol") so
       downstream signal code never has to handle nulls.

    tests/test_regime.py applies a future-perturbation check to any object
    satisfying this Protocol -- run it against the HMM before wiring it in.
    """

    def __call__(self, prices, hist_vol, index=None, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        ...


def realized_hv(close, window: int = 252, *, annualise: bool = True) -> np.ndarray:
    """
    Trailing realised volatility from log returns.

    Annualised by sqrt(252) -- the convention that makes 0.20 read as "20% a
    year", which is how volatility is normally quoted.
    """
    close = np.asarray(close, dtype=float).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(close > 0, np.log(np.where(close > 0, close, 1.0)), np.nan)
    ret = pd.Series(np.diff(logp, prepend=np.nan))
    hv = ret.rolling(window, min_periods=max(window // 4, 20)).std(ddof=1)
    return (hv * np.sqrt(252.0) if annualise else hv).to_numpy()


def _rolling_percentile_labels(
    values: np.ndarray,
    lookback: int,
    warmup: int,
) -> np.ndarray:
    """
    Label each bar against ROLLING percentiles of its own trailing history.

    Every threshold at bar t is computed from values[t-lookback+1 .. t]. This is
    the fix for the original's full-sample percentiles.
    """
    series = pd.Series(values)
    roll = series.rolling(lookback, min_periods=warmup)
    q = {k: roll.quantile(p / 100.0).to_numpy() for k, p in _PCTS.items()}

    out = np.full(len(values), "Sideways", dtype=object)
    ready = np.isfinite(values) & np.isfinite(q["crash"]) & np.isfinite(q["rally"])

    out[ready & (values <= q["crash"])] = "Crashing"
    out[ready & (values > q["crash"]) & (values <= q["fall"])] = "Falling"
    out[ready & (values >= q["rally"])] = "Rally"
    out[ready & (values < q["rally"]) & (values >= q["rise"])] = "Rising"
    return out


def _rolling_vol_labels(hv: np.ndarray, lookback: int, warmup: int) -> np.ndarray:
    """Volatility terciles on a rolling window. Same causality contract."""
    series = pd.Series(hv)
    roll = series.rolling(lookback, min_periods=warmup)
    p33 = roll.quantile(0.33).to_numpy()
    p66 = roll.quantile(0.66).to_numpy()

    out = np.full(len(hv), "MedVol", dtype=object)
    ready = np.isfinite(hv) & np.isfinite(p33) & np.isfinite(p66)
    out[ready & (hv >= p66)] = "HighVol"
    out[ready & (hv <= p33)] = "LowVol"
    return out


def detect_regimes(
    locus,
    hist_vol,
    index=None,
    *,
    reg_win: int = 20,
    vol_warmup: int = 60,
    lookback: int = DEFAULT_LOOKBACK,
    mkt_warmup: int = 250,
    mode: str = "per_symbol_percentile",
    thresholds: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect market and volatility regimes. Strictly causal.

    Parameters
    ----------
    locus      : price series. Raw close is fine and preferred -- see note 5 in
                 the module docstring.
    hist_vol   : annualised historical volatility, same length. Use realized_hv().
    reg_win    : trailing bars for the cumulative return that sets direction.
                 20 sessions ~ one month.
    lookback   : trailing bars used to compute percentile thresholds.
    mkt_warmup : minimum observations before the market regime leaves its
                 neutral default.
    mode       : "per_symbol_percentile" (default) or "absolute".

    Returns
    -------
    (market_regime, vol_regime) : object arrays of length N.

    Warmup bars return "Sideways"/"MedVol" rather than NaN, preserving the
    original contract that downstream code never sees a null.
    """
    loc = np.asarray(locus, dtype=float).ravel()
    n = len(loc)
    if len(np.asarray(hist_vol).ravel()) != n:
        raise ValueError("locus and hist_vol must be the same length")

    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(loc > 0, np.log(np.where(loc > 0, loc, 1.0)), np.nan)

    # Trailing cumulative log-return, expressed as % per day.
    cum = pd.Series(logp).diff().rolling(reg_win).sum().to_numpy()
    pct_per_day = (cum / float(reg_win)) * 100.0

    if mode == "per_symbol_percentile":
        mkt = _rolling_percentile_labels(pct_per_day, lookback, mkt_warmup)
    elif mode == "absolute":
        thr = thresholds or {"crash": -0.20, "falling": -0.05, "rising": 0.05, "rally": 0.20}
        mkt = np.full(n, "Sideways", dtype=object)
        ok = np.isfinite(pct_per_day)
        mkt[ok & (pct_per_day <= thr["crash"])] = "Crashing"
        mkt[ok & (pct_per_day > thr["crash"]) & (pct_per_day <= thr["falling"])] = "Falling"
        mkt[ok & (pct_per_day >= thr["rally"])] = "Rally"
        mkt[ok & (pct_per_day < thr["rally"]) & (pct_per_day >= thr["rising"])] = "Rising"
    else:
        raise ValueError(f"unknown mode {mode!r}")

    vol = _rolling_vol_labels(
        np.asarray(hist_vol, dtype=float).ravel(), lookback, vol_warmup
    )
    return mkt, vol
