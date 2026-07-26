"""
features.py
-----------
Phase 2: the feature block Isolation Forest scores.

Three layers, all strictly causal:

  slope / acceleration   trend shape from a rolling regression on log price
  per-stock precursors   stress measures that move before price breaks
  cross-sectional        what the universe is doing as a whole

THE LEAD-TIME REQUIREMENT IS A PROPERTY OF THE FEATURES
-------------------------------------------------------
Isolation Forest finds whatever is unusual in what you give it. Feed it today's
return and it will faithfully report that today's -6% day was unusual, on the
day it happened, which is not a prediction. So `ret_1d` is computed for
labelling and plotting but is NOT in FEATURE_COLUMNS and never reaches the
model. A test guards against it leaking in later.

CAUSALITY CONTRACT
------------------
Every value at bar t is a function of bars <= t. Nothing here fits on the full
sample. tests/test_features.py enforces this by perturbing the future and
asserting the past does not move -- the check that catches the class of bug
found in the original regime module.

WHY LOG PRICE
-------------
d(log P)/dt is fractional return per day, so a slope of -0.9 means "-0.9% per
day" whether the stock trades at 200 or 20,000. Slopes on raw price are
comparable neither across stocks nor across time for one stock, and log space
also makes the measure immune to splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qbeast_crash.config import DEFAULT_CONFIG, Config

__all__ = [
    # trend shape
    "rolling_slope", "rolling_accel", "realized_vol", "classify_phase",
    "compute_slope_accel", "PHASES",
    # per-stock precursors
    "FEATURE_COLUMNS", "compute_precursors",
    # cross-sectional
    "MARKET_FEATURE_COLUMNS", "compute_market_features",
    "average_pairwise_correlation",
]



# =====================================================================
# LAYER 1  Slope and acceleration
# =====================================================================

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

# =====================================================================
# LAYER 2  Per-stock precursors
# =====================================================================

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

# =====================================================================
# LAYER 3  Cross-sectional market state
# =====================================================================

MARKET_FEATURE_COLUMNS = (
    "breadth_decline",
    "breadth_advance",
    "median_slope_z",
    "dispersion",
    "avg_corr",
    "pct_below_ma50",
    "n_live",
)

_EPS = 1e-12


def average_pairwise_correlation(
    returns: pd.DataFrame,
    window: int = 60,
    min_symbols: int = 20,
) -> pd.Series:
    """
    Rolling average pairwise correlation across the universe.

    The direct route is to build a correlation matrix per day and average its
    off-diagonal, which is O(N^2) per day -- roughly 4,500 pairs x 5,820 days,
    repeated four times over in the Phase 7 retraining comparison.

    There is an exact shortcut. For an equal-weighted portfolio of N assets:

        Var(portfolio) * N^2  =  sum(var_i)  +  sum_{i != j} cov_ij

    If every pairwise correlation is taken to be the same value rho, then
    cov_ij = rho * sigma_i * sigma_j, and the cross terms collapse:

        sum_{i != j} cov_ij  =  rho * ( (sum sigma_i)^2 - sum sigma_i^2 )

    Rearranging gives

        rho  =  ( N^2 * Var_p - sum sigma_i^2 ) / ( (sum sigma_i)^2 - sum sigma_i^2 )

    Every term is a rolling mean, variance or sum, so the whole series is O(N*T)
    and fully vectorised. rho is the average pairwise correlation implied by how
    much diversification the universe is actually delivering -- which is the
    quantity we care about, and arguably a better one than the raw mean of a
    correlation matrix.

    Two honest caveats. The equal-correlation assumption makes this exact only
    when correlations are homogeneous; in practice it tracks the true average
    closely (see tests). And the portfolio variance is computed from the
    cross-sectional mean return, which assumes membership is roughly stable
    across the window -- true here except around a new listing.
    """
    live = returns.notna()
    n_live = live.sum(axis=1)

    # Equal-weighted portfolio return, over whatever existed that day.
    port = returns.mean(axis=1, skipna=True)
    var_p = port.rolling(window, min_periods=window).var(ddof=1)

    sig = returns.rolling(window, min_periods=window).std(ddof=1)
    sum_sig = sig.sum(axis=1, skipna=True)
    sum_sig_sq = (sig ** 2).sum(axis=1, skipna=True)

    n = sig.notna().sum(axis=1)

    numer = (n ** 2) * var_p - sum_sig_sq
    denom = sum_sig ** 2 - sum_sig_sq

    rho = numer / denom.where(denom.abs() > _EPS)
    rho[n < min_symbols] = np.nan
    return rho.clip(-1.0, 1.0)


def _true_average_pairwise_correlation(
    returns: pd.DataFrame,
    window: int = 60,
    min_symbols: int = 20,
) -> pd.Series:
    """
    Direct O(N^2) computation. Reference implementation used only to validate
    average_pairwise_correlation() in tests -- far too slow for production.
    """
    out = pd.Series(np.nan, index=returns.index)
    values = returns.to_numpy()
    for t in range(window - 1, len(returns)):
        block = values[t - window + 1: t + 1]
        cols = ~np.isnan(block).any(axis=0)
        if cols.sum() < min_symbols:
            continue
        corr = np.corrcoef(block[:, cols], rowvar=False)
        iu = np.triu_indices_from(corr, k=1)
        out.iloc[t] = np.nanmean(corr[iu])
    return out


def compute_market_features(
    close_panel: pd.DataFrame,
    phase_panel: pd.DataFrame,
    slope_z_panel: pd.DataFrame,
    listing_mask: pd.DataFrame | None = None,
    *,
    corr_window: int = 60,
    ma_window: int = 50,
    min_symbols: int = 20,
) -> pd.DataFrame:
    """
    Build the daily market-state block.

    Parameters
    ----------
    close_panel   : dates x symbols closes (from Phase 1).
    phase_panel   : dates x symbols trend phase labels (from precursors).
    slope_z_panel : dates x symbols volatility-normalised slope.
    listing_mask  : dates x symbols, True where the symbol was tradeable.

    Returns
    -------
    DataFrame indexed by date with MARKET_FEATURE_COLUMNS.
    """
    if listing_mask is not None:
        close_panel = close_panel.where(listing_mask)
        phase_panel = phase_panel.where(listing_mask)
        slope_z_panel = slope_z_panel.where(listing_mask)

    out = pd.DataFrame(index=close_panel.index)

    # Denominator: symbols with an actual observation today.
    live = phase_panel.notna() & (phase_panel != "")
    n_live = live.sum(axis=1)
    out["n_live"] = n_live

    # --- breadth ------------------------------------------------------------
    # The measured calibration from Phase 2: on 2020-03-12, 92.1% of the
    # universe was in AcceleratingDecline. But ~90% breadth also occurred on
    # mild pullbacks, so breadth ALONE cannot separate a systemic crash from a
    # broad wobble. It needs median_slope_z alongside it.
    safe_n = n_live.replace(0, np.nan)
    out["breadth_decline"] = (phase_panel == "AcceleratingDecline").sum(axis=1) / safe_n * 100.0
    out["breadth_advance"] = (phase_panel == "AcceleratingAdvance").sum(axis=1) / safe_n * 100.0

    # --- depth --------------------------------------------------------------
    # The dimension breadth is missing. COVID showed -2.11 median sigma/day;
    # the mild pullbacks with similar breadth showed roughly -0.8.
    out["median_slope_z"] = slope_z_panel.median(axis=1, skipna=True)

    # --- dispersion ---------------------------------------------------------
    # Cross-sectional spread of daily returns. Collapses as correlation rises:
    # when everything moves together there is nothing to tell stocks apart.
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.log(close_panel / close_panel.shift(1))
    out["dispersion"] = rets.std(axis=1, skipna=True) * 100.0

    # --- correlation --------------------------------------------------------
    out["avg_corr"] = average_pairwise_correlation(rets, corr_window, min_symbols)

    # --- participation ------------------------------------------------------
    ma = close_panel.rolling(ma_window, min_periods=ma_window).mean()
    below = (close_panel < ma) & close_panel.notna() & ma.notna()
    denom = (close_panel.notna() & ma.notna()).sum(axis=1).replace(0, np.nan)
    out["pct_below_ma50"] = below.sum(axis=1) / denom * 100.0

    out.loc[n_live < min_symbols, [c for c in MARKET_FEATURE_COLUMNS if c != "n_live"]] = np.nan
    return out