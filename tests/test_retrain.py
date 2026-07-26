"""
test_retrain.py
---------------
Tests for walk-forward retraining and the four schemes.

The causality tests matter most here. Walk-forward is easy to write in a way
that lets a model score a day it was trained on, and nothing would complain --
the comparison would just quietly favour whichever scheme leaked most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbeast_crash.config import DEFAULT_CONFIG
from qbeast_crash.features import FEATURE_COLUMNS
from qbeast_crash.retrain import SCHEMES, refit_dates, training_slice, walk_forward


def _features(start="2015-01-01", periods=1600, n_sym=4, seed=0):
    dates = pd.bdate_range(start, periods=periods)
    syms = [f"S{i}" for i in range(n_sym)]
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 1, (len(idx), len(FEATURE_COLUMNS))),
        index=idx, columns=list(FEATURE_COLUMNS),
    )


def _calendar(frame):
    return pd.DatetimeIndex(frame.index.get_level_values("date").unique())


# =====================================================================
# Causality
# =====================================================================
@pytest.mark.parametrize("scheme", SCHEMES)
def test_training_data_never_includes_the_scoring_day(scheme):
    """
    Every scheme must see data strictly BEFORE the refit date.

    Including the day itself would let a model score a row it was trained on,
    and the scheme that leaked most would appear to win.
    """
    f = _features()
    as_of = pd.Timestamp("2019-01-02")
    rows, _ = training_slice(f, as_of, scheme, DEFAULT_CONFIG)
    if len(rows):
        assert rows.index.get_level_values("date").max() < as_of


def test_rolling_window_forgets_old_data():
    f = _features()
    as_of = pd.Timestamp("2020-01-02")
    rows, _ = training_slice(f, as_of, "rolling", DEFAULT_CONFIG)
    span_years = (rows.index.get_level_values("date").max()
                  - rows.index.get_level_values("date").min()).days / 365.25
    assert span_years <= DEFAULT_CONFIG.retrain.rolling_window_years + 0.05


def test_incremental_window_keeps_everything():
    f = _features()
    as_of = pd.Timestamp("2020-01-02")
    roll, _ = training_slice(f, as_of, "rolling", DEFAULT_CONFIG)
    incr, _ = training_slice(f, as_of, "incremental", DEFAULT_CONFIG)
    assert len(incr) > len(roll)
    assert incr.index.get_level_values("date").min() == f.index.get_level_values("date").min()


def test_ewma_weights_recent_data_more_heavily():
    """
    Isolation Forest has no sample_weight, so weighting is done by resampling.
    The drawn sample must skew recent.
    """
    f = _features()
    as_of = pd.Timestamp("2020-01-02")
    rows, _ = training_slice(f, as_of, "ewma", DEFAULT_CONFIG,
                             rng=np.random.default_rng(0))
    drawn = rows.index.get_level_values("date")
    available = f[f.index.get_level_values("date") < as_of].index.get_level_values("date")
    assert drawn.to_series().median() > available.to_series().median()


def test_ewma_sampling_draws_no_duplicates():
    """
    Regression guard for a real defect.

    Drawing WITH replacement produced 2.63x duplication -- 100,000 rows for
    only 38,043 unique ones. Duplicates are not neutral here: identical points
    cannot be separated from each other, so a tree stops splitting and their
    path length inflates, scoring them as LESS anomalous. That would
    systematically under-flag exactly the recent rows EWMA exists to emphasise.
    """
    f = _features(periods=2000, n_sym=8)
    rows, _ = training_slice(f, pd.Timestamp("2021-06-01"), "ewma",
                             DEFAULT_CONFIG, rng=np.random.default_rng(0))
    assert len(rows) == len(rows.index.unique()), "EWMA resampling drew duplicates"


def test_ewma_memory_is_shorter_than_the_rolling_window():
    """
    Worth asserting because it is easy to assume otherwise: decay 0.994 implies
    a half-life of 115 sessions and an effective sample of 167 -- about 0.66
    years, against the rolling scheme's 3. The schemes differ in LOOKBACK as
    well as in weighting.
    """
    f = _features(periods=2000, n_sym=8)
    as_of = pd.Timestamp("2021-06-01")
    ewma, _ = training_slice(f, as_of, "ewma", DEFAULT_CONFIG,
                             rng=np.random.default_rng(0))
    roll, _ = training_slice(f, as_of, "rolling", DEFAULT_CONFIG)

    ewma_age = pd.Series((as_of - ewma.index.get_level_values("date")).days).median()
    roll_age = pd.Series((as_of - roll.index.get_level_values("date")).days).median()
    assert ewma_age < roll_age


def test_vol_purge_excludes_turbulent_dates():
    f = _features()
    dates = _calendar(f)
    market = pd.DataFrame({"dispersion": np.linspace(1.0, 5.0, len(dates))}, index=dates)
    as_of = pd.Timestamp("2020-01-02")
    _, excluded = training_slice(f, as_of, "vol_purged", DEFAULT_CONFIG, market=market)
    assert excluded is not None and len(excluded) > 0


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unknown scheme"):
        training_slice(_features(), pd.Timestamp("2020-01-02"), "nonsense")


# =====================================================================
# Refit schedule
# =====================================================================
def test_refits_are_monthly_and_on_trading_days():
    """
    All four schemes share this schedule so that only the training SET varies.
    Varying the frequency too would confound the two effects.
    """
    cal = pd.bdate_range("2021-01-01", "2021-12-31")
    dates = refit_dates(cal, "2021-01-01", "2021-12-31")
    assert len(dates) == 12
    assert dates.is_monotonic_increasing
    assert set(dates) <= set(cal)
    assert len(set(d.month for d in dates)) == 12


def test_refit_dates_handles_an_empty_window():
    cal = pd.bdate_range("2021-01-01", periods=10)
    assert len(refit_dates(cal, "2030-01-01", "2030-12-31")) == 0


# =====================================================================
# Walk-forward
# =====================================================================
def test_walk_forward_scores_the_backtest_window():
    f = _features(periods=1800)
    out = walk_forward(f, _calendar(f), "rolling", DEFAULT_CONFIG, min_train_rows=500)
    scored = out.dropna()
    assert len(scored) > 0
    assert scored.index.get_level_values("date").min() >= pd.Timestamp(
        DEFAULT_CONFIG.windows.backtest_start)


def test_walk_forward_intensity_is_bounded():
    f = _features(periods=1800)
    out = walk_forward(f, _calendar(f), "rolling", DEFAULT_CONFIG, min_train_rows=500).dropna()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_future_data_cannot_change_earlier_intensities():
    """
    The defining property of walk-forward. Rewrite the tail, and every earlier
    score must be untouched -- otherwise a scheme is scoring days it has seen.
    """
    f = _features(periods=1800)
    cal = _calendar(f)
    cut = pd.Timestamp("2021-07-01")

    base = walk_forward(f, cal, "rolling", DEFAULT_CONFIG, min_train_rows=500)
    tampered = f.copy()
    mask = tampered.index.get_level_values("date") >= cut
    tampered.loc[mask] = tampered.loc[mask] * 50.0
    after = walk_forward(tampered, cal, "rolling", DEFAULT_CONFIG, min_train_rows=500)

    before_cut = base.index.get_level_values("date") < cut
    pd.testing.assert_series_equal(base[before_cut], after[before_cut], check_names=False)


def test_insufficient_training_data_is_skipped_not_faked():
    """Better an unscored day than one scored by a model fitted on nothing."""
    f = _features(periods=1800)
    out = walk_forward(f, _calendar(f), "rolling", DEFAULT_CONFIG,
                       min_train_rows=10_000_000)
    assert out.isna().all()
