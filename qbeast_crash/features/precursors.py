"""
precursors.py
-------------
Per-stock leading features for the crash/rally detector.

What it does:   Builds the feature vector Isolation Forest scores, one row per
                symbol per day.
Why we do it:   The requirement is to detect a crash 2-10 days BEFORE it
                happens. That is a property of the FEATURES, not of the model.
                Isolation Forest finds whatever is unusual in what you give it;
                if you give it today's return, it will faithfully tell you that
                today's -6% day was unusual, on the day it happened, which is
                not a prediction.
How (method):   Every feature is a slow-moving measure of market STRESS rather
                than of price movement. Stress builds before price breaks.
Where:          precursors.py -> compute_precursors()

WHY THESE FEATURES LEAD
-----------------------
Crashes are not bolts from a clear sky. Before a large decline, several things
usually shift while price is still roughly flat:

  * short-horizon volatility rises against its own longer-term baseline
    (volatility clusters -- turbulence precedes and follows turbulence)
  * downside moves start to outweigh upside moves in size
  * volume picks up as positioning changes
  * daily ranges widen; overnight gaps become more frequent
  * liquidity thins, so a given rupee of volume moves price further

Each is measurable, each is slow, and none of them requires today to have been
a bad day. That is what makes them precursors instead of a description of a
crash already in progress.

THE ONE DELIBERATE OMISSION
---------------------------
`ret_1d` is computed and returned, because it is useful for labelling,
plotting and debugging -- but it is NOT in FEATURE_COLUMNS and never reaches
the model. Including it would make the detector coincident by construction,
and no amount of downstream cleverness would recover the lead time.

CAUSALITY
---------
Every window is trailing. Bar t uses bars <= t and nothing later.
tests/test_precursors.py enforces this by perturbing the future and asserting
the past is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qbeast_crash.config import DEFAULT_CONFIG, Config
from qbeast_crash.features.slope_accel import compute_slope_accel

__all__ = ["FEATURE_COLUMNS", "compute_precursors"]

#: Exactly the columns that reach Isolation Forest. Anything not listed here is
#: diagnostic only. Keeping this explicit is what stops a convenience column
#: from silently leaking into the model months later.
FEATURE_COLUMNS = (
    "vol_ratio",
    "vol_of_vol",
    "semidev_asym",
    "volume_z",
    "range_expansion",
    "gap_freq",
    "illiquidity",
    "dd_from_high",
    "slope_z",
    "accel_z",
)

_EPS = 1e-12


def _log_returns(close: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(close > 0, np.log(close.where(close > 0, 1.0)), np.nan)
    return pd.Series(np.diff(logp, prepend=np.nan), index=close.index)


def compute_precursors(
    frame: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
    *,
    short_vol: int = 5,
    long_vol: int = 60,
    med_window: int = 20,
    gap_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Build the precursor feature block for one symbol.

    Parameters
    ----------
    frame : cleaned per-symbol frame from data.loader -- needs open, high, low,
            close, volume and the is_flat / is_zero_vol quality flags.

    Returns
    -------
    DataFrame indexed by date with FEATURE_COLUMNS plus diagnostics
    (ret_1d, vol_short, vol_long).

    Warmup bars are NaN, never a neutral default. A fabricated "average" value
    during warmup is indistinguishable downstream from a real reading, and
    Isolation Forest would treat a cluster of them as a dense, very normal
    region -- exactly the wrong lesson.
    """
    fcfg = config.features
    close = frame["close"].astype(float)
    ret = _log_returns(close)

    out = pd.DataFrame(index=frame.index)
    out["ret_1d"] = ret * 100.0

    # --- 1. volatility term structure --------------------------------------
    # Short-horizon volatility measured against its own longer baseline. A
    # ratio, not a level, so it is already comparable across stocks: 2.0 means
    # "twice as turbulent as this stock's own normal", whether that normal is
    # 15% or 45% annualised.
    vol_s = ret.rolling(short_vol, min_periods=short_vol).std(ddof=1)
    vol_l = ret.rolling(long_vol, min_periods=long_vol).std(ddof=1)
    out["vol_short"] = vol_s * 100.0
    out["vol_long"] = vol_l * 100.0
    out["vol_ratio"] = vol_s / vol_l.where(vol_l > _EPS)

    # --- 2. volatility of volatility ---------------------------------------
    # Is the turbulence itself steady or erratic? A market that is reliably
    # volatile is a different regime from one whose volatility is lurching, and
    # the second tends to precede dislocations.
    vol_med = ret.rolling(med_window, min_periods=med_window).std(ddof=1)
    out["vol_of_vol"] = (
        vol_med.rolling(long_vol, min_periods=long_vol).std(ddof=1)
        / vol_med.rolling(long_vol, min_periods=long_vol).mean().where(lambda s: s > _EPS)
    )

    # --- 3. downside / upside asymmetry ------------------------------------
    # Semi-deviation splits volatility by direction. Ordinary volatility treats
    # a +3% day and a -3% day as identical; this does not.
    #
    # Expressed as a bounded asymmetry index rather than the obvious ratio
    # semi_d / semi_u, for two reasons:
    #
    #   * The ratio is undefined exactly when the signal is strongest. A 20-day
    #     window with no up days at all -- maximum bearish asymmetry -- divides
    #     by zero and yields NaN, silently discarding the most extreme reading.
    #   * The ratio is unbounded and right-skewed. Isolation Forest splits at
    #     random points within a feature's observed range, so one huge outlier
    #     stretches that range and wastes most candidate splits on empty space.
    #
    #   -1 = all upside     0 = symmetric     +1 = all downside
    down = ret.where(ret < 0, 0.0)
    up = ret.where(ret > 0, 0.0)
    semi_d = down.pow(2).rolling(med_window, min_periods=med_window).mean().pow(0.5)
    semi_u = up.pow(2).rolling(med_window, min_periods=med_window).mean().pow(0.5)
    total = semi_d + semi_u
    out["semidev_asym"] = (semi_d - semi_u) / total.where(total > _EPS)

    # --- 4. volume surge ----------------------------------------------------
    # Log volume, because raw volume is heavily right-skewed and a z-score of a
    # skewed variable is dominated by its tail. Zero-volume bars are excluded
    # rather than treated as genuinely quiet days.
    volume = frame["volume"].astype(float)
    log_vol = np.log(volume.where(volume > 0))
    v_mean = log_vol.rolling(long_vol, min_periods=long_vol // 2).mean()
    v_std = log_vol.rolling(long_vol, min_periods=long_vol // 2).std(ddof=1)
    out["volume_z"] = (log_vol - v_mean) / v_std.where(v_std > _EPS)

    # --- 5. range expansion -------------------------------------------------
    # Intraday range against its own baseline. Flat bars (open=high=low=close)
    # are masked, not zeroed: BAJFINANCE has 1,092 of them, and treating each as
    # a genuine zero-range session would drag the baseline down and make every
    # ordinary day afterwards look like an expansion.
    rng = (frame["high"].astype(float) - frame["low"].astype(float)) / close
    rng = rng.where(~frame["is_flat"].astype(bool))
    rng_base = rng.rolling(long_vol, min_periods=long_vol // 2).mean()
    out["range_expansion"] = rng / rng_base.where(rng_base > _EPS)

    # --- 6. overnight gap frequency ----------------------------------------
    # How often price jumps between the close and the next open. Gaps are
    # information arriving while the market is shut, so a rising gap rate means
    # the story is increasingly being written outside trading hours.
    gap = (frame["open"].astype(float) / close.shift(1)) - 1.0
    out["gap_freq"] = (
        (gap.abs() > gap_threshold).astype(float)
        .rolling(med_window, min_periods=med_window).mean()
    )

    # --- 7. illiquidity (Amihud) -------------------------------------------
    # Absolute return per rupee of turnover: how far a given amount of trading
    # pushes price. Liquidity thinning is one of the more reliable precursors,
    # because market makers widen before they withdraw. Logged, since the raw
    # measure spans orders of magnitude across a 100-stock universe.
    turnover = (close * volume).where(volume > 0)
    amihud = (ret.abs() / turnover).rolling(med_window, min_periods=med_window).mean()
    out["illiquidity"] = np.log(amihud.where(amihud > 0))

    # --- 8. drawdown from rolling high -------------------------------------
    # Where price sits inside its recent range. Not a stress measure but a
    # context one: the same volatility spike means something different at a
    # high than 15% below it.
    roll_high = close.rolling(long_vol, min_periods=long_vol // 2).max()
    out["dd_from_high"] = (close / roll_high.where(roll_high > _EPS)) - 1.0

    # --- 9. trend shape -----------------------------------------------------
    sa = compute_slope_accel(
        close.to_numpy(), index=frame.index,
        slope_window=fcfg.slope_window, accel_window=fcfg.accel_window,
        vol_window=fcfg.vol_window, vol_lag=fcfg.vol_lag,
        slope_deadband=fcfg.slope_deadband, accel_deadband=fcfg.accel_deadband,
    )
    out["slope_z"] = sa["slope_z"]
    out["accel_z"] = sa["accel_z"]
    out["phase"] = sa["phase"]

    # Infinities can arise from near-zero denominators surviving the guards.
    # They must not reach the model: a single inf makes every finite value look
    # identical by comparison.
    return out.replace([np.inf, -np.inf], np.nan)
