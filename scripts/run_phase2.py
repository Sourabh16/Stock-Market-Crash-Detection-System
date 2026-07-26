"""
run_phase2.py
-------------
Phase 2 entry point: build per-stock precursor features and the cross-sectional
market-state block from the Phase 1 panel.

    python scripts/run_phase2.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from qbeast_crash.config import DATA_PROCESSED, DEFAULT_CONFIG, REPORTS, ensure_dirs
from qbeast_crash.data.calendar import listing_mask, trading_calendar
from qbeast_crash.data.loader import load_universe
from qbeast_crash.features.market import MARKET_FEATURE_COLUMNS, compute_market_features
from qbeast_crash.features.precursors import FEATURE_COLUMNS, compute_precursors


def main() -> int:
    ensure_dirs()
    cfg = DEFAULT_CONFIG

    print("loading Phase 1 universe ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames, _ = load_universe(config=cfg)
    calendar = trading_calendar(cfg)
    mask = listing_mask(frames, calendar)
    print(f"  {len(frames)} symbols, {len(calendar)} trading days\n")

    print("computing per-stock precursors ...")
    feats = {sym: compute_precursors(f, cfg) for sym, f in frames.items()}

    # Long format: one row per (date, symbol). This is what Isolation Forest
    # consumes -- it is fitted on the pooled cross-section, not per symbol, so
    # one model learns what "normal" looks like across the whole universe.
    long = (
        pd.concat(feats, names=["symbol", "date"])
        .reorder_levels(["date", "symbol"])
        .sort_index()
    )
    print(f"  {long.shape[0]:,} symbol-days x {len(FEATURE_COLUMNS)} model features")

    print("\ncomputing cross-sectional market features ...")
    panel = lambda col: pd.DataFrame(
        {s: f[col].reindex(calendar) for s, f in feats.items()}, index=calendar
    )
    market = compute_market_features(
        pd.read_parquet(DATA_PROCESSED / "close_panel.parquet"),
        panel("phase"), panel("slope_z"), mask,
    )

    long.to_parquet(DATA_PROCESSED / "features.parquet")
    market.to_parquet(DATA_PROCESSED / "market_features.parquet")

    print("\nfeature coverage in the backtest window (2021-01-01 onward):")
    usable = long.loc[str(cfg.windows.backtest_start):]
    for col in FEATURE_COLUMNS:
        pct = usable[col].notna().mean() * 100
        flag = "" if pct > 95 else "   <-- low"
        print(f"  {col:18s} {pct:5.1f}% non-null{flag}")

    print("\nmarket features:")
    m = market.loc["2016-01-01":]
    for col in MARKET_FEATURE_COLUMNS:
        s = m[col].dropna()
        if s.empty:
            print(f"  {col:18s} (empty)")
            continue
        print(f"  {col:18s} median {s.median():8.2f}   p1 {s.quantile(.01):8.2f}   p99 {s.quantile(.99):8.2f}")

    print(f"\nwrote features.parquet {long.shape} and market_features.parquet {market.shape}")
    print(f"  -> {DATA_PROCESSED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
