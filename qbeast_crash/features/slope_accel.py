"""
slope_accel.py
--------------
Causal slope and acceleration features for the QBEAST crash/rally detector.

What it does:   Fits rolling least-squares trends to LOG price and returns the
                first derivative (slope, %/day) and second derivative
                (acceleration, %/day^2), plus volatility-normalised versions
                and a trend-phase label.
Why we do it:   Anomaly intensity alone says "today is unusual". It does not
                say which direction, or whether the move is building or
                exhausting. Slope answers direction; acceleration answers
                building-vs-exhausting. Together they turn a raw anomaly flag
                into a crash-exit or rally-entry decision.
How (method):   For an equally spaced trailing window the OLS coefficients
                collapse to a FIXED weight vector, so both derivatives are a
                single dot product against the trailing window. No loops, and
                causality is structural rather than something we remember to
                enforce.
Where:          slope_accel.py -> compute_slope_accel()

CAUSALITY CONTRACT
------------------
Every value at bar t is a function of bars [t-window+1 .. t] ONLY. Nothing in
this module fits on the full sample. tests/test_slope_accel.py enforces this by
perturbing the future and asserting the past does not move -- the check that
would have caught the full-sample percentile bug in the regime module.

WHY LOG PRICE
-------------
d(log P)/dt is fractional return per day, so a slope of -0.9 means "-0.9% per
day" whether the stock trades at 200 or 20,000. Slopes on raw price are not
comparable across stocks and are not comparable across time for one stock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "rolling_slope",
    "rolling_accel",
    "realized_vol",
    "classify_phase",
    "compute_slope_accel",
    "PHASES",
]

# Trend phases: the sign pair (slope, acceleration) is the whole signal logic.
#
#   slope < 0, accel < 0  ->  falling and steepening   -> crash developing
#   slope < 0, accel > 0  ->  falling but flattening   -> selloff exhausting
#   slope > 0, accel > 0  ->  rising and steepening    -> rally developing
#   slope > 0, accel < 0  ->  rising but flattening    -> advance topping out
#
PHASES = (
    "AcceleratingDecline",
    "DeceleratingDecline",
    "AcceleratingAdvance",
    "DeceleratingAdvance",
    "Flat",
)

_MIN_WINDOW = 3          # need >=3 points for a quadratic fit
_EPS = 1e-12


# =====================================================================
# Fixed OLS weight vectors (computed once per window size)
# =====================================================================
def _linear_slope_weights(window: int) -> np.ndarray:
    """
    Weights w such that  w . y == OLS slope of y on t, for t = 0..window-1.

    slope = sum((t - tbar) * y) / sum((t - tbar)^2)

    The denominator sum((t-tbar)^2) = window*(window^2 - 1)/12 is constant for
    an equally spaced window, which is what makes this a fixed linear filter.
    """
    t = np.arange(window, dtype=float)
    tc = t - t.mean()
    return tc / np.dot(tc, tc)


def _quadratic_accel_weights(window: int) -> np.ndarray:
    """
    Weights w such that  w . y == 2*c2, where y ~ c0 + c1*t + c2*t^2.

    2*c2 is the second derivative of the fitted parabola, i.e. acceleration.

    We take it from a single quadratic fit rather than by differencing two
    slope series. Differencing stacks two smoothers and roughly doubles the
    lag, which is exactly what we cannot afford when the whole point is to
    detect a turn 2-10 days early.
    """
    t = np.arange(window, dtype=float)
    design = np.vstack([np.ones(window), t, t ** 2]).T      # (window, 3)
    pinv = np.linalg.pinv(design)                           # (3, window)
    return 2.0 * pinv[2]                                    # row for c2


def _rolling_dot(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Apply a fixed weight vector to every trailing window.

    Output index t uses values[t-window+1 .. t]. The first window-1 entries are
    NaN because they have no complete trailing window -- they are never filled
    with a neutral default, since a fabricated value here is indistinguishable
    from a real reading downstream.
    """
    n = len(values)
    window = len(weights)
    out = np.full(n, np.nan, dtype=float)
    if n < window:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    out[window - 1:] = windows @ weights                    # NaN propagates
    return out


def _theil_sen_slope(values: np.ndarray, window: int) -> np.ndarray:
    """
    Median of pairwise slopes over the trailing window (Theil-Sen).

    Breakdown point ~29% vs 0% for OLS, so one bad print or one gap-filled bar
    cannot drag the trend estimate. Useful given the flat-bar and stitched-data
    problems found in the raw CSVs. O(window^2) per bar, which is fine for the
    small windows used here.
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if n < window:
        return out
    i_idx, j_idx = np.triu_indices(window, k=1)
    dt = (j_idx - i_idx).astype(float)
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    pair_slopes = (windows[:, j_idx] - windows[:, i_idx]) / dt
    with np.errstate(invalid="ignore"):
        out[window - 1:] = np.nanmedian(pair_slopes, axis=1)
    # A window containing any NaN must stay NaN; nanmedian would hide it.
    out[window - 1:][np.isnan(windows).any(axis=1)] = np.nan
    return out


# =====================================================================
# Public feature functions
# =====================================================================
def rolling_slope(
    close,
    window: int = 5,
    *,
    robust: bool = False,
    as_pct: bool = True,
) -> np.ndarray:
    """
    First derivative of log price -- the trend, in % per day.

    Parameters
    ----------
    close  : array-like of positive prices.
    window : trailing bars in the fit. 5 = one trading week.
    robust : use Theil-Sen instead of OLS (see _theil_sen_slope).
    as_pct : return %/day rather than fraction/day.

    Notes
    -----
    A literal "1-day slope" is just the daily return; use the ret_1d column
    from compute_slope_accel() for that. Any window below 3 bars gives an
    estimate dominated by single-bar noise, so window is clamped at 3.
    """
    window = max(int(window), _MIN_WINDOW)
    logp = _safe_log(close)
    if robust:
        slope = _theil_sen_slope(logp, window)
    else:
        slope = _rolling_dot(logp, _linear_slope_weights(window))
    return slope * 100.0 if as_pct else slope


def rolling_accel(
    close,
    window: int = 7,
    *,
    as_pct: bool = True,
) -> np.ndarray:
    """
    Second derivative of log price -- is the trend building or exhausting,
    in % per day per day.

    Sign is what matters, not magnitude:
      accel < 0  the move is bending downward (decline steepening, or an
                 advance rolling over)
      accel > 0  the move is bending upward (decline flattening, or an
                 advance steepening)

    window should exceed the slope window; a parabola needs more room than a
    line to be identified. Default 7 vs slope 5.
    """
    window = max(int(window), _MIN_WINDOW)
    logp = _safe_log(close)
    accel = _rolling_dot(logp, _quadratic_accel_weights(window))
    return accel * 100.0 if as_pct else accel


def realized_vol(
    close,
    window: int = 60,
    *,
    lag: int = 0,
    as_pct: bool = True,
) -> np.ndarray:
    """
    Causal daily realised volatility from log returns, in %/day.

    Used to normalise slope and acceleration. Without it, a -1%/day slope reads
    as identical stress for HINDUNILVR and ADANIENT, when for one it is a
    three-sigma event and for the other it is a Tuesday.

    lag : shift the estimate back this many bars before use.

        This is not cosmetic. A crash inflates its own denominator: once the
        trailing window fills with crash days, the normaliser grows and the
        signal shrinks precisely when it should be loudest. Measured on
        RELIANCE across March 2020, the COVID slope_z reads -1.67 at lag=0 but
        -3.01 at lag=20.

        Lagging means the denominator describes the regime the move is
        departing FROM, not the one it is creating. Still fully causal -- bar t
        uses returns up to bar t-lag.
    """
    window = max(int(window), 2)
    logp = _safe_log(close)
    ret = np.diff(logp, prepend=np.nan)
    vol = pd.Series(ret).rolling(window, min_periods=window).std(ddof=1)
    if lag:
        vol = vol.shift(int(lag))
    vol = vol.to_numpy()
    return vol * 100.0 if as_pct else vol


def classify_phase(
    slope_z: np.ndarray,
    accel_z: np.ndarray,
    *,
    slope_deadband: float = 0.10,
    accel_deadband: float = 0.05,
) -> np.ndarray:
    """
    Map the (slope, acceleration) sign pair to a trend phase.

    Deadbands keep near-zero readings out of the four directional buckets. With
    no deadband, sign flips on numerical noise produce phase churn, and phase
    churn becomes trade churn -- which this strategy explicitly cannot afford.

    Inputs are the volatility-normalised z-scores, so the deadbands are in
    units of daily sigma and mean the same thing across every stock.
    """
    slope_z = np.asarray(slope_z, dtype=float)
    accel_z = np.asarray(accel_z, dtype=float)

    out = np.full(len(slope_z), "Flat", dtype=object)
    valid = np.isfinite(slope_z) & np.isfinite(accel_z)
    out[~valid] = None

    strong = valid & (np.abs(slope_z) >= slope_deadband) & (np.abs(accel_z) >= accel_deadband)
    down, up = strong & (slope_z < 0), strong & (slope_z > 0)

    out[down & (accel_z < 0)] = "AcceleratingDecline"
    out[down & (accel_z > 0)] = "DeceleratingDecline"
    out[up & (accel_z > 0)] = "AcceleratingAdvance"
    out[up & (accel_z < 0)] = "DeceleratingAdvance"
    return out


def compute_slope_accel(
    close,
    index=None,
    *,
    slope_window: int = 5,
    accel_window: int = 7,
    vol_window: int = 60,
    vol_lag: int = 20,
    robust_slope: bool = False,
    slope_deadband: float = 0.10,
    accel_deadband: float = 0.05,
) -> pd.DataFrame:
    """
    Full causal slope/acceleration feature block for one symbol.

    Returns a DataFrame with:
        ret_1d      daily log return, %                (the literal 1-day slope)
        slope       trend, %/day
        accel       curvature, %/day^2
        vol         realised daily volatility, %/day, lagged by vol_lag
        slope_z     slope / vol       -> daily sigmas of drift per day
        accel_z     accel / vol       -> change in that drift per day
        phase       one of PHASES

    slope_z is the field signal logic should threshold on. Raw slope is not
    comparable across stocks; slope_z is, which is what lets one global
    threshold serve all 100 symbols instead of 100 hand-tuned ones.

    vol_lag defaults to 20 bars so a crash cannot mute its own signal -- see
    realized_vol(). Set vol_lag=0 to reproduce the naive normalisation.
    """
    close = np.asarray(close, dtype=float).ravel()
    if index is None:
        index = pd.RangeIndex(len(close))
    index = pd.Index(index)
    if len(index) != len(close):
        raise ValueError(f"index length {len(index)} != close length {len(close)}")

    logp = _safe_log(close)
    ret_1d = np.diff(logp, prepend=np.nan) * 100.0

    slope = rolling_slope(close, slope_window, robust=robust_slope)
    accel = rolling_accel(close, accel_window)
    vol = realized_vol(close, vol_window, lag=vol_lag)

    with np.errstate(divide="ignore", invalid="ignore"):
        safe_vol = np.where(np.isfinite(vol) & (vol > _EPS), vol, np.nan)
        slope_z = slope / safe_vol
        accel_z = accel / safe_vol

    return pd.DataFrame(
        {
            "ret_1d": ret_1d,
            "slope": slope,
            "accel": accel,
            "vol": vol,
            "slope_z": slope_z,
            "accel_z": accel_z,
            "phase": classify_phase(
                slope_z, accel_z,
                slope_deadband=slope_deadband,
                accel_deadband=accel_deadband,
            ),
        },
        index=index,
    )


# =====================================================================
# Internal
# =====================================================================
def _safe_log(close) -> np.ndarray:
    """
    Log price, with non-positive values sent to NaN rather than -inf.

    The raw CSVs contain gap-filled and stitched bars; a zero or negative close
    is a data error, and -inf would silently poison every downstream window it
    touches instead of failing visibly as NaN.
    """
    arr = np.asarray(close, dtype=float).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(arr > 0, np.log(np.where(arr > 0, arr, 1.0)), np.nan)
