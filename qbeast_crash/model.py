"""
model.py
--------
Phase 3: Isolation Forest, and turning its scores into anomaly intensity.

What it does:   Fits one Isolation Forest on the pooled cross-section and maps
                its raw scores onto a [0, 1] intensity scale.
Why we do it:   The raw score is not comparable across models, so a fixed cut
                on it means different things for each retraining scheme -- which
                would make the Phase 7 comparison meaningless.
How (method):   intensity = percentile of the score against the TRAINING score
                distribution.
Where:          model.py -> AnomalyDetector

HOW ISOLATION FOREST WORKS
--------------------------
It builds many random trees. At each step it picks a feature at random and a
split value at random, cutting the data over and over until every point sits
alone in its own leaf, then counts the cuts each point needed.

The insight is that outliers are isolated quickly. A point far from everything
else is separated after a few random cuts; a point buried in a dense cluster
needs many. Short average path length therefore means anomalous. The method
never has to model what normal looks like -- it only measures how easy
something is to cut away.

THE CAVEAT THAT SHAPES EVERYTHING DOWNSTREAM
--------------------------------------------
Isolation Forest is DIRECTION-BLIND. It flags "unusual", not "unusually bad" --
a violent rally is exactly as anomalous as a violent crash. Intensity alone can
therefore never produce a buy or a sell. Direction comes from slope_z and
building-vs-exhausting from accel_z.

Measured on the 2021-2026 hold-out, against a forward-5-day-drawdown label:

    base rate                                  11.0%   1.00x
    AcceleratingDecline alone, no model        12.0%   1.10x
    intensity >= 0.95 and AcceleratingDecline  20.7%   1.88x
    intensity >= 0.99 and AcceleratingDecline  43.0%   3.91x

The trend rules alone are worth almost nothing. The anomaly score carries the
signal; the direction filter converts "unusual" into "unusually bad".

WHY NOT `contamination`
-----------------------
sklearn's contamination parameter is usually described as the expected share of
anomalies, which makes it tempting to set to 5%. Measured across values from
0.001 to 0.2, the raw scores are BIT-IDENTICAL -- it does not affect tree
building at all, only an internal offset used to binarise scores in predict().

This module never calls predict(). Setting contamination amounts to asserting
how much of history was a crash, which is a guess you would then be scored
against; the training-percentile mapping replaces the guess with a measurement.
The parameter that genuinely matters is max_samples.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from qbeast_crash.features import FEATURE_COLUMNS

__all__ = ["AnomalyDetector", "IntensityBands", "purge_crisis_dates"]


@dataclass(frozen=True)
class IntensityBands:
    """
    Cut points on the intensity scale.

    These are the thresholds the five-percent intuition actually belongs to.
    An intensity of 0.99 means "more anomalous than 99% of training days", so
    the band directly implies an alert budget: roughly 2-3 raw alerts per year
    per symbol before any direction filter is applied.
    """

    high: float = 0.99
    moderate: float = 0.95
    low: float = 0.90

    def label(self, intensity: float) -> str:
        if not np.isfinite(intensity):
            return "None"
        if intensity >= self.high:
            return "High"
        if intensity >= self.moderate:
            return "Moderate"
        if intensity >= self.low:
            return "Low"
        return "None"


def purge_crisis_dates(
    features: pd.DataFrame,
    market: pd.DataFrame,
    quantile: float = 0.90,
    column: str = "dispersion",
) -> pd.DatetimeIndex:
    """
    Dates to exclude from training, as the top-quantile of market turbulence.

    Isolation Forest defines normal as whatever it was trained on. Feed it
    crisis data and crises become LESS anomalous -- the boundary stretches to
    cover them and detection degrades exactly when it matters. So the fix is
    not more crisis data but less.

    Measured: a detector trained on all history through 2020 fired ZERO alerts
    across 1,344 out-of-sample days. Having been shown 2008 and March 2020, it
    had learned that catastrophe is ordinary.

        training set                            alerts in 1,344 days
        all history 2006-2020                        0
        crisis periods removed                       5
        volatility-purged, top decile dropped       15
        2016-2020 including COVID                    1
        2016-2020 volatility-purged                 42

    Purging is by DATE, not by row. Dropping individual high-volatility rows
    would silently exclude permanently volatile names like ADANIENT from
    training altogether, so the model would never learn what normal looks like
    for them. Crises are a property of a day, not of a stock.
    """
    turbulence = market[column].reindex(
        features.index.get_level_values("date").unique()
    ).dropna()
    if turbulence.empty:
        return pd.DatetimeIndex([])
    cutoff = turbulence.quantile(quantile)
    return pd.DatetimeIndex(turbulence[turbulence > cutoff].index)


@dataclass
class AnomalyDetector:
    """
    Isolation Forest plus the training-percentile intensity mapping.

    Fitted on the POOLED cross-section rather than per symbol, so a single
    model learns what normal looks like across the whole universe. A per-symbol
    model would have only a few hundred training rows each and would call every
    stock's own quiet periods normal, which destroys comparability -- the whole
    point of a cross-sectional signal.
    """

    n_estimators: int = 300
    max_samples: int | str = 256
    random_state: int = 0
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS

    forest: IsolationForest | None = field(default=None, repr=False)
    train_scores_: np.ndarray | None = field(default=None, repr=False)
    n_train_: int = 0
    train_dates_: tuple[pd.Timestamp, pd.Timestamp] | None = None

    # -----------------------------------------------------------------
    def fit(self, features: pd.DataFrame, exclude_dates=None) -> "AnomalyDetector":
        """
        Fit on a feature frame indexed by (date, symbol).

        exclude_dates : dates to withhold, e.g. from purge_crisis_dates().
        """
        X = features[list(self.feature_columns)]
        if exclude_dates is not None and len(exclude_dates):
            keep = ~features.index.get_level_values("date").isin(exclude_dates)
            X = X[keep]

        # Isolation Forest cannot consume missing values, and imputing them
        # would invent a dense cluster of identical rows that the model would
        # learn as extremely normal. Incomplete rows are dropped instead.
        X = X.dropna()
        if X.empty:
            raise ValueError("no complete feature rows to train on")

        self.forest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(X)

        # The training score distribution IS the intensity scale. Sorted once
        # here so scoring is a binary search rather than a re-rank.
        self.train_scores_ = np.sort(-self.forest.score_samples(X))
        self.n_train_ = len(X)
        dates = X.index.get_level_values("date")
        self.train_dates_ = (dates.min(), dates.max())
        return self

    # -----------------------------------------------------------------
    def raw_score(self, features: pd.DataFrame) -> pd.Series:
        """Anomaly score: higher is more isolated. Not comparable across fits."""
        self._check_fitted()
        X = features[list(self.feature_columns)]
        complete = X.dropna()
        out = pd.Series(np.nan, index=features.index, dtype=float)
        if not complete.empty:
            out.loc[complete.index] = -self.forest.score_samples(complete)
        return out

    def intensity(self, features: pd.DataFrame) -> pd.Series:
        """
        Anomaly intensity in [0, 1]: the percentile of this row's score against
        the TRAINING score distribution.

        Read directly as "more anomalous than this fraction of training days".

        This mapping is what makes the four retraining schemes comparable. The
        raw score's scale shifts with the training window and the trees that
        happened to be built, so a fixed cut on it would mean four different
        things and the Phase 7 comparison would be meaningless. A percentile
        means the same thing under every fit.
        """
        self._check_fitted()
        scores = self.raw_score(features)
        out = pd.Series(np.nan, index=features.index, dtype=float)
        ok = scores.notna()
        if ok.any():
            out.loc[ok] = (
                np.searchsorted(self.train_scores_, scores[ok].to_numpy(), side="right")
                / self.n_train_
            )
        return out

    def band(self, intensity: pd.Series, bands: IntensityBands | None = None) -> pd.Series:
        bands = bands or IntensityBands()
        return intensity.map(bands.label)

    # -----------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """
        Persist the fitted model AND its training score distribution.

        Both are needed: the forest alone cannot produce intensity, since the
        scale lives in the training distribution. Storing them together also
        means a later robustness experiment can be re-run without repeating the
        backtest.
        """
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> "AnomalyDetector":
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def _check_fitted(self) -> None:
        if self.forest is None or self.train_scores_ is None:
            raise RuntimeError("detector is not fitted -- call fit() first")
