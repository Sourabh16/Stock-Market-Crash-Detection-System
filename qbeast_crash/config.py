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
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)


DEFAULT_CONFIG = Config()


def ensure_dirs() -> None:
    """Create output directories. Raw data is read-only and never created."""
    for path in (DATA_INTERIM, DATA_PROCESSED, REPORTS, REPORTS / "figures"):
        path.mkdir(parents=True, exist_ok=True)
