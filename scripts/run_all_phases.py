"""
run_all_phases.py
-----------------
Single entry point for the whole pipeline.

    python scripts/run_all_phases.py              # every phase, in order
    python scripts/run_all_phases.py --phase 2    # one phase
    python scripts/run_all_phases.py --from 2     # phase 2 onward
    python scripts/run_all_phases.py --list       # what exists

Each phase is a function that reads what earlier phases wrote and writes its own
artefacts to data/ and reports/. Phases are pure with respect to data/raw/,
which is never modified -- deleting data/interim/, data/processed/ and reports/
and re-running reproduces every artefact.

Adding a phase: write phase_N(ctx) returning a dict of results, and register it
in PHASES. Anything a later phase needs goes in the returned dict.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from qbeast_crash.config import (
    DATA_INTERIM,
    DATA_PROCESSED,
    DEFAULT_CONFIG,
    REPORTS,
    ensure_dirs,
)
from qbeast_crash.data import (
    build_close_panel,
    listing_mask,
    load_universe,
    run_quality_gate,
    trading_calendar,
)
from qbeast_crash.features import (
    FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    compute_market_features,
    compute_precursors,
)

CFG = DEFAULT_CONFIG


def _rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# =====================================================================
# Phase 1 -- data layer
# =====================================================================
def phase_1(ctx: dict) -> dict:
    """Raw CSVs -> cleaned frames, aligned panel, listing mask, audit trail."""
    _rule("PHASE 1  Data layer")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames, reports = load_universe(config=CFG)
    calendar = trading_calendar(CFG)
    print(f"loaded {len(frames)} usable symbols over {len(calendar)} trading days\n")

    gate = run_quality_gate(frames, reports, calendar, CFG, raise_on_error=False)
    print(gate.render())
    if not gate.ok:
        raise RuntimeError("data quality gate failed -- refusing to continue")

    panel = build_close_panel(frames, calendar)
    mask = listing_mask(frames, calendar)

    panel.to_parquet(DATA_PROCESSED / "close_panel.parquet")
    mask.to_parquet(DATA_PROCESSED / "listing_mask.parquet")
    reports.to_csv(REPORTS / "phase1_data_audit.csv")
    for sym, frame in frames.items():
        frame.to_parquet(DATA_INTERIM / f"{sym}.parquet")

    live = mask.sum(axis=1)
    print(f"\nclose_panel {panel.shape}   live symbols "
          f"{live.iloc[0]} at {calendar[0].date()} -> {live.iloc[-1]} at {calendar[-1].date()}")
    return {"frames": frames, "calendar": calendar, "panel": panel, "mask": mask}


# =====================================================================
# Phase 2 -- features
# =====================================================================
def phase_2(ctx: dict) -> dict:
    """Per-stock precursors plus the cross-sectional market-state block."""
    _rule("PHASE 2  Features")

    frames, calendar = ctx["frames"], ctx["calendar"]
    feats = {sym: compute_precursors(f, CFG) for sym, f in frames.items()}

    # Long format, one row per (date, symbol). Isolation Forest is fitted on
    # the pooled cross-section rather than per symbol, so a single model learns
    # what normal looks like across the whole universe.
    long = (
        pd.concat(feats, names=["symbol", "date"])
        .reorder_levels(["date", "symbol"])
        .sort_index()
    )
    print(f"per-stock: {long.shape[0]:,} symbol-days x {len(FEATURE_COLUMNS)} model features")

    def panel_of(col: str) -> pd.DataFrame:
        return pd.DataFrame({s: f[col].reindex(calendar) for s, f in feats.items()},
                            index=calendar)

    market = compute_market_features(
        ctx["panel"], panel_of("phase"), panel_of("slope_z"), ctx["mask"]
    )

    long.to_parquet(DATA_PROCESSED / "features.parquet")
    market.to_parquet(DATA_PROCESSED / "market_features.parquet")

    print("\nfeature coverage in the backtest window:")
    usable = long.loc[str(CFG.windows.backtest_start):]
    for col in FEATURE_COLUMNS:
        pct = usable[col].notna().mean() * 100
        print(f"  {col:18s} {pct:5.1f}% non-null{'' if pct > 95 else '   <-- low'}")

    print("\ncross-sectional market state (2016 onward):")
    m = market.loc["2016-01-01":]
    for col in MARKET_FEATURE_COLUMNS:
        s = m[col].dropna()
        if not s.empty:
            print(f"  {col:18s} median {s.median():8.2f}   "
                  f"p1 {s.quantile(.01):8.2f}   p99 {s.quantile(.99):8.2f}")

    print(f"\nwrote features {long.shape} and market_features {market.shape}")
    return {"features": long, "market": market}


PHASES = {
    1: ("Data layer", phase_1),
    2: ("Features", phase_2),
    # 3: ("Isolation Forest + intensity", phase_3),
    # 4: ("Labels + lead time", phase_4),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="QBEAST crash detection pipeline")
    ap.add_argument("--phase", type=int, help="run one phase only")
    ap.add_argument("--from", dest="start", type=int, help="run from this phase onward")
    ap.add_argument("--list", action="store_true", help="list phases and exit")
    args = ap.parse_args()

    if args.list:
        for n, (name, _) in sorted(PHASES.items()):
            print(f"  {n}  {name}")
        return 0

    if args.phase is not None:
        wanted = [args.phase]
    elif args.start is not None:
        wanted = [n for n in sorted(PHASES) if n >= args.start]
    else:
        wanted = sorted(PHASES)

    unknown = [n for n in wanted if n not in PHASES]
    if unknown:
        print(f"unknown phase(s): {unknown}. Known: {sorted(PHASES)}")
        return 2

    # Running a later phase alone still needs its predecessors' in-memory
    # results, so we replay the earlier phases rather than silently using stale
    # files. Cheap here, and it keeps a partial run honest.
    to_run = [n for n in sorted(PHASES) if n <= max(wanted)]

    ensure_dirs()
    ctx: dict = {}
    started = time.time()
    for n in to_run:
        name, fn = PHASES[n]
        t0 = time.time()
        ctx.update(fn(ctx))
        if n in wanted:
            print(f"\nphase {n} ({name}) completed in {time.time() - t0:.1f}s")

    _rule(f"DONE  {len(to_run)} phase(s) in {time.time() - started:.1f}s")
    print(f"artefacts -> {DATA_PROCESSED}")
    print(f"reports   -> {REPORTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
