"""
market.py
---------
Cross-sectional features: what the universe is doing as a whole.

What it does:   Collapses the 96-symbol panel into a handful of daily
                market-state series.
Why we do it:   A single stock falling on its own news is an exit for that
                stock. The whole market falling together is a different event
                that needs a different response. Per-stock features cannot tell
                the two apart -- only the cross-section can.
How (method):   Breadth, median trend, dispersion and average pairwise
                correlation, all computed on the LIVE universe each day.
Where:          market.py -> compute_market_features()

WHY CORRELATION IS THE INTERESTING ONE
--------------------------------------
In calm markets stocks move on their own news, so average pairwise correlation
is low. As stress builds, everything starts moving together -- investors sell
what they can rather than what they want to, and idiosyncratic stories stop
mattering. Correlation therefore RISES BEFORE the index breaks, which makes it
one of the few genuinely leading cross-sectional measures.

Dispersion is the same signal read from the other side: as correlation climbs,
the spread of daily returns across the universe collapses.

THE LIVE-UNIVERSE RULE
----------------------
Every measure is computed only over symbols that actually existed and traded on
that day, using the listing mask from Phase 1. Otherwise "40% of stocks are
falling" silently becomes "40% of stocks are falling, out of a denominator that
includes companies which had not yet listed" -- which is a different and
meaningless quantity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["MARKET_FEATURE_COLUMNS", "compute_market_features", "average_pairwise_correlation"]

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
