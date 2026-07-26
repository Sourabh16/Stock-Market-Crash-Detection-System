"""
retrain.py
----------
Phase 7: walk-forward retraining, and the four-way comparison.

What it does:   Refits the detector repeatedly through the backtest window
                under four different rules for choosing training data, then
                scores each out of sample.
Why we do it:   Phase 3 fitted once on 2016-2020 and scored five years forward.
                That is not how the system would run. A live model is refitted
                as data arrives, and the question is which rule for choosing
                that data produces the shallowest drawdown.
Where:          retrain.py -> walk_forward(), compare_schemes()

WHY THIS COMPARISON IS VALID AT ALL
-----------------------------------
Because intensity is a percentile of each fit's OWN training distribution
rather than a raw anomaly score.

The raw score's scale shifts with the training window and with whichever
random trees happened to be built. A fixed cut on it would select a different
fraction of days under each scheme, so the comparison would be measuring the
scoring scale rather than the schemes. With a percentile, 0.99 means "more
unusual than 99% of what this model was trained on" under every scheme.

That design decision was taken in Phase 3 specifically to make this phase
possible.

THE EXPERIMENT VARIES ONE THING
-------------------------------
All four schemes refit on the SAME schedule -- the first session of each month.
Only the training SET differs. Varying the refit frequency at the same time
would confound the two effects and neither could be attributed.

WALK-FORWARD IS THE ONLY HONEST WAY TO DO THIS
----------------------------------------------
At each refit the model sees data strictly before that date, and scores only
the days until the next refit. No day is ever scored by a model that has seen
it. tests/test_retrain.py enforces this by perturbing the future and asserting
earlier intensities are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qbeast_crash.config import DEFAULT_CONFIG, Config
from qbeast_crash.model import AnomalyDetector, purge_crisis_dates

__all__ = ["SCHEMES", "training_slice", "walk_forward", "refit_dates"]

#: The four schemes. Names are used as keys throughout the reporting.
SCHEMES = ("rolling", "incremental", "ewma", "vol_purged")


def refit_dates(calendar: pd.DatetimeIndex, start, end) -> pd.DatetimeIndex:
    """
    First trading session of each month in [start, end].

    Monthly is what the project brief specifies for the incremental scheme, and
    applying it to all four keeps the schedule constant so only the training
    set varies.
    """
    window = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if window.empty:
        return pd.DatetimeIndex([])
    firsts = pd.Series(window).groupby(window.to_period("M")).min()
    return pd.DatetimeIndex(firsts.to_numpy())


def training_slice(
    features: pd.DataFrame,
    as_of: pd.Timestamp,
    scheme: str,
    config: Config = DEFAULT_CONFIG,
    market: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex | None]:
    """
    Choose the training rows visible at `as_of` under a given scheme.

    Returns (rows, dates_to_exclude). Every scheme sees data strictly BEFORE
    as_of -- never the day itself, which the model is about to score.
    """
    cfg = config.retrain
    dates = features.index.get_level_values("date")
    visible = features[dates < as_of]
    if visible.empty:
        return visible, None

    if scheme == "incremental":
        # Expanding window: everything since the beginning. Longest memory, so
        # it adapts slowest -- and it accumulates every past crisis, which is
        # what makes crises progressively less anomalous over time.
        return visible, None

    if scheme == "rolling":
        # Fixed 3-year lookback. Forgets old regimes entirely, which is the
        # point -- Indian market volatility in 2016 is not evidence about 2025.
        cutoff = as_of - pd.DateOffset(years=cfg.rolling_window_years)
        return visible[visible.index.get_level_values("date") >= cutoff], None

    if scheme == "vol_purged":
        # Rolling, then drop the most turbulent dates. Isolation Forest learns
        # "normal" from what it is shown, so training on crises makes crises
        # unremarkable. Measured in Phase 3 this HURT on the pooled per-stock
        # cross-section, but it was measured under a single static fit -- under
        # walk-forward the training window is shorter and a crisis occupies a
        # larger share of it, so the effect may differ. That is the hypothesis
        # this phase tests.
        cutoff = as_of - pd.DateOffset(years=cfg.rolling_window_years)
        rows = visible[visible.index.get_level_values("date") >= cutoff]
        if market is None or rows.empty:
            return rows, None
        return rows, purge_crisis_dates(rows, market, cfg.vol_purge_quantile)

    if scheme == "ewma":
        # Exponentially weighted memory: recent days matter more, old days fade
        # rather than being cut off.
        #
        # Isolation Forest has no sample_weight parameter, so weighting is done
        # by resampling. The obvious way -- draw with replacement in proportion
        # to weight -- turns out to be actively harmful here:
        #
        #     drawing 100,000 rows with replacement gave only 38,043 unique
        #     rows, a duplication factor of 2.63x
        #
        # Duplicates are not neutral for this model. Identical points cannot be
        # separated from each other, so a tree stops splitting and their path
        # length is inflated -- they score as LESS anomalous. That would
        # systematically under-flag exactly the recent rows EWMA exists to
        # emphasise, which is the opposite of the intent.
        #
        # Sampling WITHOUT replacement removes the problem. Done with the
        # Gumbel top-k trick rather than numpy's replace=False, which is
        # prohibitively slow on hundreds of thousands of rows: adding Gumbel
        # noise to log-weights and taking the top k is mathematically identical
        # to sequential weighted sampling without replacement, in O(n log n).
        rows = visible
        rng = rng or np.random.default_rng(0)
        row_dates = rows.index.get_level_values("date")
        age_days = np.asarray((as_of - row_dates).days, dtype=float)
        # Calendar age -> approximate trading-day age.
        age_sessions = age_days * (cfg.trading_days_per_year / 365.25)

        log_w = age_sessions * np.log(cfg.ewma_decay)      # log(decay ** age)
        if not np.isfinite(log_w).any():
            return rows, None

        gumbel = -np.log(-np.log(rng.random(len(rows)) + 1e-300) + 1e-300)
        keys = log_w + gumbel

        # Draw enough rows to fit on, but no more than the weights meaningfully
        # support. The effective sample size of a geometric decay is
        # 1/(1-decay) sessions; beyond a few multiples of that, extra rows carry
        # almost no weight and only add stale data.
        eff_sessions = 1.0 / (1.0 - cfg.ewma_decay)
        n_symbols = max(rows.index.get_level_values("symbol").nunique(), 1)
        n_draw = int(min(len(rows), max(4.0 * eff_sessions * n_symbols, 5_000)))

        picked = np.argpartition(-keys, n_draw - 1)[:n_draw]
        return rows.iloc[picked], None

    raise ValueError(f"unknown scheme {scheme!r}. Known: {SCHEMES}")


def walk_forward(
    features: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    scheme: str,
    config: Config = DEFAULT_CONFIG,
    market: pd.DataFrame | None = None,
    min_train_rows: int = 2_000,
    verbose: bool = False,
) -> pd.Series:
    """
    Refit monthly and score forward, producing one intensity series.

    Each block of days is scored by the model fitted at the start of that
    block, using only data from before it. No day is ever scored by a model
    that has seen it.
    """
    w = config.windows
    dates = refit_dates(calendar, w.backtest_start, w.backtest_end)
    rng = np.random.default_rng(config.model.random_state)

    out = pd.Series(np.nan, index=features.index, dtype=float)
    feature_dates = features.index.get_level_values("date")
    n_fits = 0

    for i, as_of in enumerate(dates):
        block_end = dates[i + 1] if i + 1 < len(dates) else pd.Timestamp(w.backtest_end)
        block = (feature_dates >= as_of) & (feature_dates < block_end) \
            if i + 1 < len(dates) else \
            (feature_dates >= as_of) & (feature_dates <= block_end)
        if not block.any():
            continue

        train, exclude = training_slice(features, as_of, scheme, config, market, rng)
        if len(train) < min_train_rows:
            continue

        try:
            detector = AnomalyDetector(
                n_estimators=config.model.n_estimators,
                max_samples=config.model.max_samples,
                random_state=config.model.random_state,
            ).fit(train, exclude_dates=exclude)
        except ValueError:
            continue                       # no complete rows this month

        out.loc[block] = detector.intensity(features[block]).to_numpy()
        n_fits += 1

    if verbose:
        print(f"    {scheme:12s} {n_fits} refits, "
              f"{out.notna().sum():,} scored symbol-days")
    return out


@dataclass
class SchemeResult:
    """One scheme's outcome, for ranking."""

    scheme: str
    cagr: float
    max_drawdown: float
    bh_max_drawdown: float
    trades_per_symbol_year: int | float
    signals: int

    @property
    def drawdown_saved(self) -> float:
        """Percentage points of drawdown avoided versus buy-and-hold."""
        return (self.max_drawdown - self.bh_max_drawdown) * 100

    @property
    def efficiency(self) -> float:
        """
        Drawdown saved per unit of turnover.

        The ranking metric, and deliberately not raw return. A scheme that
        trades constantly can buy a better drawdown figure at a cost that only
        surfaces later in brokerage and tax -- efficiency prices that in.
        """
        if self.trades_per_symbol_year <= 0:
            return 0.0
        return self.drawdown_saved / self.trades_per_symbol_year
