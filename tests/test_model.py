"""
test_model.py
-------------
Tests for the Isolation Forest detector and the intensity mapping.

The load-bearing property is that intensity means the same thing under every
fit. Without it the Phase 7 retraining comparison is meaningless, because a
fixed threshold would select a different fraction of days under each scheme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbeast_crash.features import FEATURE_COLUMNS
from qbeast_crash.model import AnomalyDetector, IntensityBands, purge_crisis_dates


def _frame(n=4000, seed=0, scale=1.0):
    """Synthetic feature frame indexed by (date, symbol)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n // 4)
    idx = pd.MultiIndex.from_product(
        [dates, ["A", "B", "C", "D"]], names=["date", "symbol"]
    )
    data = rng.normal(0, scale, (len(idx), len(FEATURE_COLUMNS)))
    return pd.DataFrame(data, index=idx, columns=list(FEATURE_COLUMNS))


# =====================================================================
# The intensity contract
# =====================================================================
def test_intensity_is_a_percentile_of_the_training_distribution():
    """On its own training data, intensity must be ~uniform on [0, 1]."""
    X = _frame()
    det = AnomalyDetector().fit(X)
    inten = det.intensity(X).dropna()

    assert inten.min() >= 0.0 and inten.max() <= 1.0
    for q in (0.10, 0.50, 0.90, 0.99):
        assert abs((inten <= q).mean() - q) < 0.03, f"not uniform at q={q}"


def test_threshold_means_the_same_thing_under_different_fits():
    """
    The property the whole Phase 7 comparison rests on.

    Two detectors with different settings must both select ~1% of their own
    training data at intensity >= 0.99. A raw-score threshold would not.
    """
    X = _frame()
    for kwargs in ({"max_samples": 64}, {"max_samples": 512}, {"n_estimators": 100}):
        det = AnomalyDetector(**kwargs).fit(X)
        share = (det.intensity(X) >= 0.99).mean()
        assert 0.005 < share < 0.02, f"{kwargs} selected {share:.3%}"


def test_raw_scores_are_not_comparable_but_intensity_is():
    """Motivates the mapping: raw score scales shift between fits."""
    X = _frame()
    a = AnomalyDetector(max_samples=64).fit(X)
    b = AnomalyDetector(max_samples=1024).fit(X)

    raw_gap = abs(a.raw_score(X).mean() - b.raw_score(X).mean())
    int_gap = abs(a.intensity(X).mean() - b.intensity(X).mean())
    assert int_gap < raw_gap or int_gap < 0.02


def test_outliers_score_higher_than_the_bulk():
    X = _frame(seed=1)
    det = AnomalyDetector().fit(X)

    outliers = X.iloc[:20].copy() + 12.0
    assert det.intensity(outliers).mean() > det.intensity(X).mean() + 0.3


# =====================================================================
# Missing data
# =====================================================================
def test_incomplete_rows_score_nan_not_imputed():
    """
    Imputing would invent a dense cluster of identical rows that the model
    learns as extremely normal -- the opposite of what is wanted.
    """
    X = _frame(seed=2)
    det = AnomalyDetector().fit(X)

    holed = X.copy()
    holed.iloc[:50, 0] = np.nan
    inten = det.intensity(holed)

    assert inten.iloc[:50].isna().all()
    assert inten.iloc[50:].notna().all()
    assert len(inten) == len(holed)          # index preserved, rows not dropped


def test_fit_ignores_incomplete_rows():
    X = _frame(seed=3)
    X.iloc[:100, 2] = np.nan
    det = AnomalyDetector().fit(X)
    assert det.n_train_ == len(X) - 100


def test_all_nan_training_data_raises():
    X = _frame(n=400)
    X.iloc[:, :] = np.nan
    with pytest.raises(ValueError, match="no complete feature rows"):
        AnomalyDetector().fit(X)


def test_unfitted_detector_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        AnomalyDetector().intensity(_frame(n=100))


# =====================================================================
# Bands
# =====================================================================
def test_band_labels():
    b = IntensityBands()
    assert b.label(0.995) == "High"
    assert b.label(0.97) == "Moderate"
    assert b.label(0.92) == "Low"
    assert b.label(0.5) == "None"
    assert b.label(np.nan) == "None"


def test_band_series_maps_every_row():
    X = _frame(seed=4)
    det = AnomalyDetector().fit(X)
    bands = det.band(det.intensity(X))
    assert set(bands.unique()) <= {"High", "Moderate", "Low", "None"}
    assert len(bands) == len(X)


# =====================================================================
# Purging
# =====================================================================
def test_purge_excludes_the_requested_dates():
    X = _frame(seed=5)
    dates = X.index.get_level_values("date").unique()
    excluded = dates[:100]

    det = AnomalyDetector().fit(X, exclude_dates=excluded)
    kept = X[~X.index.get_level_values("date").isin(excluded)]
    assert det.n_train_ == len(kept)
    assert det.train_dates_[0] > excluded[-1]


def test_purge_selects_the_most_turbulent_dates():
    """Purging is by DATE, not by row: a crisis is a property of a day."""
    X = _frame(seed=6)
    dates = X.index.get_level_values("date").unique()
    market = pd.DataFrame({"dispersion": np.linspace(1.0, 5.0, len(dates))}, index=dates)

    purged = purge_crisis_dates(X, market, quantile=0.90)
    assert 0 < len(purged) <= len(dates) * 0.12
    assert purged.min() > dates[int(len(dates) * 0.8)]      # only the tail


def test_purge_handles_missing_market_data():
    X = _frame(seed=7)
    empty = pd.DataFrame({"dispersion": []}, index=pd.DatetimeIndex([]))
    assert len(purge_crisis_dates(X, empty)) == 0


# =====================================================================
# Persistence
# =====================================================================
def test_save_load_roundtrip_preserves_intensity(tmp_path):
    """
    The training score distribution must persist with the forest.

    The forest alone cannot produce intensity -- the scale lives in the
    training distribution -- and storing both together is also what lets a
    later robustness experiment be re-run without repeating the backtest.
    """
    X = _frame(seed=8)
    det = AnomalyDetector().fit(X)
    path = tmp_path / "d.pkl"
    det.save(path)

    reloaded = AnomalyDetector.load(path)
    assert reloaded.n_train_ == det.n_train_
    pd.testing.assert_series_equal(reloaded.intensity(X), det.intensity(X))


# =====================================================================
# Determinism
# =====================================================================
def test_same_seed_gives_identical_scores():
    X = _frame(seed=9)
    a = AnomalyDetector(random_state=42).fit(X).intensity(X)
    b = AnomalyDetector(random_state=42).fit(X).intensity(X)
    pd.testing.assert_series_equal(a, b)
