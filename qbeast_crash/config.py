"""
config.py
---------
Single source of truth for paths, date windows, and cleaning thresholds.

Why a config module at all: every threshold in this project shows up in a
result somewhere. If a number lives in three files it will eventually disagree
with itself, and a backtest that disagrees with itself is worse than no
backtest. Everything tunable lives here and nowhere else.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
MODELS = PROJECT_ROOT / "models"

#: The index file doubles as our master trading calendar. It is the only
#: series guaranteed to have a bar on every day the exchange was open.
INDEX_SYMBOL = "NIFTY100"

#: Raw CSVs use Indian day-first format. Parsing this as month-first silently
#: succeeds for the first twelve days of each month and then corrupts the rest,
#: so the format is pinned explicitly and never inferred.
RAW_DATE_FORMAT = "%d-%m-%Y"


@dataclass(frozen=True)
class DataConfig:
    """Rules for turning raw CSVs into a trustworthy panel."""

    #: A gap longer than this many CALENDAR days is never a holiday. NSE has
    #: never closed for ten straight days. Such a gap means the vendor stitched
    #: sparse or monthly data in front of the real listing history, so
    #: everything before the LAST such gap is discarded.
    max_gap_days: int = 10

    #: Symbols with fewer usable bars than this cannot support a 60-day
    #: volatility warmup plus any meaningful backtest.
    min_bars: int = 300

    #: A one-day LOG return beyond this is treated as an unadjusted corporate
    #: action (demerger, spin-off, unadjusted split), not a market move.
    #:
    #: 0.50 in log terms is roughly -39% / +65% simple. The margin is chosen so
    #: real crashes survive and artefacts do not:
    #:
    #:    CGPOWER  2016-03-15   log -1.074  (-65.8%)  demerger    -> caught
    #:    TRENT    2026-06-02   log -0.384  (-31.9%)  market      -> kept
    #:    ADANIENT 2023-02-03   log -0.302  (-26.1%)  Hindenburg  -> kept
    #:
    #: Indian large caps do genuinely fall 20-30% in a day. They do not fall
    #: 66% and stay there, which is the signature of a capital restructuring.
    max_abs_log_return: float = 0.50

    #: Ragged vendor end-dates. 98 of 100 symbols have data through this date;
    #: trimming here keeps the cross-section square, which the market-wide
    #: breadth signal depends on.
    hard_end: dt.date = dt.date(2026, 6, 5)

    #: Bars where open == high == low == close. Real for illiquid names, a
    #: vendor artefact otherwise. Either way they carry zero intraday range and
    #: would corrupt any range-based feature, so they are FLAGGED not dropped:
    #: dropping a row silently shifts every trailing window that spans it.
    flag_flat_bars: bool = True

    #: Columns kept from the raw file. The underscore-prefixed metadata is
    #: retained deliberately -- _source records which vendor supplied the bar
    #: (kite vs upstox), which is what makes the two-independent-sensors
    #: framing possible later. Dropping it is irreversible.
    keep_columns: tuple[str, ...] = (
        "open", "high", "low", "close", "volume",
        "_source", "_dq_score", "_gap_filled",
    )


@dataclass(frozen=True)
class WindowConfig:
    """
    Train/backtest split.

    The split is strict and never overlaps. Any feature, threshold, or scaler
    fitted on backtest-window data would leak the future into the past and
    inflate every number downstream.
    """

    train_start: dt.date = dt.date(2016, 1, 1)
    train_end: dt.date = dt.date(2020, 12, 31)
    backtest_start: dt.date = dt.date(2021, 1, 1)
    backtest_end: dt.date = dt.date(2026, 6, 5)

    def __post_init__(self) -> None:
        if self.train_end >= self.backtest_start:
            raise ValueError("train window must end before the backtest begins")


@dataclass(frozen=True)
class FeatureConfig:
    """Defaults for the slope/acceleration block. See features/slope_accel.py."""

    slope_window: int = 5          # one trading week
    accel_window: int = 7          # a parabola needs more room than a line
    vol_window: int = 60           # ~3 months of trading
    vol_lag: int = 20              # stops a crash muting its own signal
    slope_deadband: float = 0.10   # in daily sigmas
    accel_deadband: float = 0.05


@dataclass(frozen=True)
class ModelConfig:
    """Isolation Forest settings.

    Note what is absent: `contamination`. Measured across values from 0.001 to
    0.2, the raw anomaly scores are bit-identical -- it does not affect tree
    building, only an internal offset used to binarise scores in predict(),
    which this pipeline never calls. See model.py.
    """

    n_estimators: int = 300

    #: The subsample each tree isolates from. Measured across a 64x range on
    #: this data it barely matters -- lift moved only 2.18x to 2.29x from 64 to
    #: 4096 -- so 256 (the sklearn default) is kept. Recorded because an earlier
    #: note in this file claimed it was the parameter that mattered most; on
    #: this dataset that is not borne out.
    max_samples: int = 256

    random_state: int = 0

    #: Withhold the top-quantile of market turbulence from training.
    #: None disables purging, which is the BASELINE.
    #:
    #: The idea is sound in the situation that motivated it: Isolation Forest
    #: learns normal from what it is shown, so training on crises makes crises
    #: unremarkable. A detector trained on MARKET-level features across all
    #: history through 2020 -- containing both 2008 and COVID -- fired ZERO
    #: alerts across 1,344 out-of-sample days.
    #:
    #: But on the POOLED PER-STOCK cross-section it is counterproductive.
    #: Measured on 2021-2026, signal = intensity 0.99+ and AcceleratingDecline,
    #: against a base crash rate of 11.0%:
    #:
    #:     purge   signals   P(crash)   lift
    #:     none        125      38.4%   3.48x
    #:     0.99        205      29.3%   2.65x
    #:     0.95        420      24.8%   2.24x
    #:     0.90        444      24.5%   2.22x
    #:     0.75        645      20.8%   1.88x
    #:
    #: Monotone: purging buys coverage at the cost of precision. Pooling 96
    #: symbols over 2016-2020 gives a training distribution rich enough that no
    #: single crisis dominates it, so the silence problem never arises and the
    #: cure is worse than the disease.
    #:
    #: Purging therefore remains a Phase 7 retraining VARIANT, not the default.
    purge_quantile: float | None = None


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


DEFAULT_CONFIG = Config()


def ensure_dirs() -> None:
    """Create output directories. Raw data is read-only and never created."""
    for path in (DATA_INTERIM, DATA_PROCESSED, REPORTS, REPORTS / "figures", MODELS):
        path.mkdir(parents=True, exist_ok=True)
