"""
run_phase1.py
-------------
Phase 1 entry point: load every raw CSV, clean it, validate it, and persist a
usable panel plus an audit trail.

    python scripts/run_phase1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from qbeast_crash.config import DATA_INTERIM, DATA_PROCESSED, DEFAULT_CONFIG, REPORTS, ensure_dirs
from qbeast_crash.data.calendar import listing_mask, trading_calendar
from qbeast_crash.data.loader import build_close_panel, load_universe
from qbeast_crash.data.quality import run_quality_gate


def main() -> int:
    ensure_dirs()
    cfg = DEFAULT_CONFIG

    print("loading universe ...")
    frames, reports = load_universe(config=cfg)
    calendar = trading_calendar(cfg)

    print(f"  {len(frames)} usable symbols, {len(calendar)} trading days\n")

    print("quality gate")
    report = run_quality_gate(frames, reports, calendar, cfg, raise_on_error=False)
    print(report.render())

    if not report.ok:
        print("\nGATE FAILED - not writing outputs")
        return 1

    panel = build_close_panel(frames, calendar)
    mask = listing_mask(frames, calendar)

    panel.to_parquet(DATA_PROCESSED / "close_panel.parquet")
    mask.to_parquet(DATA_PROCESSED / "listing_mask.parquet")
    reports.to_csv(REPORTS / "phase1_data_audit.csv")
    for sym, frame in frames.items():
        frame.to_parquet(DATA_INTERIM / f"{sym}.parquet")

    live = mask.sum(axis=1)
    print(f"\nwrote close_panel {panel.shape} -> {DATA_PROCESSED}")
    print(f"live symbols: {live.iloc[0]} at {calendar[0].date()} "
          f"-> {live.iloc[-1]} at {calendar[-1].date()}")
    print(f"audit trail   -> {REPORTS / 'phase1_data_audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
