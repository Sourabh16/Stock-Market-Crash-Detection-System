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

import numpy as np
import pandas as pd

from qbeast_crash.config import (
    DATA_INTERIM,
    DATA_PROCESSED,
    DEFAULT_CONFIG,
    MODELS,
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
from qbeast_crash.labels import LEAD_BUCKETS, CrashDefinition, label_events, lead_time_report
from qbeast_crash.model import AnomalyDetector, purge_crisis_dates
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


# =====================================================================
# Phase 3 -- Isolation Forest + intensity
# =====================================================================
def phase_3(ctx: dict) -> dict:
    """Fit the detector on the training window and score the full history."""
    _rule("PHASE 3  Isolation Forest + intensity")

    features, market = ctx["features"], ctx["market"]
    w = CFG.windows
    dates = features.index.get_level_values("date")
    train = features[(dates >= pd.Timestamp(w.train_start)) & (dates <= pd.Timestamp(w.train_end))]

    purge = (
        purge_crisis_dates(train, market, quantile=CFG.model.purge_quantile)
        if CFG.model.purge_quantile is not None else None
    )
    print(f"training window {w.train_start} to {w.train_end}: {len(train):,} symbol-days")
    print("volatility purge: disabled (baseline)" if purge is None
          else f"volatility purge at q={CFG.model.purge_quantile}: withholding {len(purge)} dates")
    print()

    detector = AnomalyDetector(
        n_estimators=CFG.model.n_estimators,
        max_samples=CFG.model.max_samples,
        random_state=CFG.model.random_state,
    ).fit(train, exclude_dates=purge)
    print(f"fitted on {detector.n_train_:,} complete rows "
          f"({detector.train_dates_[0].date()} to {detector.train_dates_[1].date()})")

    intensity = detector.intensity(features)
    scored = features.assign(
        intensity=intensity,
        band=detector.band(intensity),
    )

    detector.save(MODELS / "detector_baseline.pkl")
    scored[["intensity", "band", "slope_z", "accel_z", "phase"]].to_parquet(
        DATA_PROCESSED / "intensity.parquet"
    )

    oos = scored[scored.index.get_level_values("date") >= pd.Timestamp(w.backtest_start)]
    print(f"\nout-of-sample ({w.backtest_start} onward): {len(oos):,} symbol-days")
    counts = oos["band"].value_counts()
    for lab in ("High", "Moderate", "Low"):
        n = int(counts.get(lab, 0))
        print(f"  {lab:9s} {n:7,d}  ({n / len(oos) * 100:5.2f}%)")

    n_sym = oos.index.get_level_values("symbol").nunique()
    n_yrs = (oos.index.get_level_values("date").max()
             - oos.index.get_level_values("date").min()).days / 365.25
    high = int(counts.get("High", 0))
    print(f"\n  High-band alerts: {high / n_sym / n_yrs:.1f} per symbol per year")

    directional = oos[(oos["band"] == "High") & (oos["phase"] == "AcceleratingDecline")]
    print(f"  ...of which AcceleratingDecline: {len(directional):,} "
          f"({len(directional) / max(high, 1) * 100:.0f}%)")

    print(f"\nwrote intensity.parquet {scored.shape} and detector_baseline.pkl")
    return {"detector": detector, "scored": scored}


# =====================================================================
# Phase 4 -- labels and lead time  (the go/no-go gate)
# =====================================================================
def phase_4(ctx: dict) -> dict:
    """
    Measure how many trading days BEFORE each crash the signal fired.

    Everything before this measures whether the signal fires when crashes are
    near. This measures how early -- which is the actual requirement.
    """
    _rule("PHASE 4  Labels and lead time")

    scored, frames, calendar = ctx["scored"], ctx["frames"], ctx["calendar"]
    w = CFG.windows
    defn = CrashDefinition()
    oos = calendar[(calendar >= pd.Timestamp(w.backtest_start))
                   & (calendar <= pd.Timestamp(w.backtest_end))]

    print(f"crash definition: forward {defn.horizon}-day drawdown <= "
          f"{defn.crash_threshold:.0%}, events merged within {defn.min_gap} days")
    print(f"out-of-sample window: {oos[0].date()} to {oos[-1].date()} "
          f"({len(oos)} sessions)\n")

    rules = {
        "intensity >= 0.99 + AccelDecline": lambda f: (f["intensity"] >= 0.99)
            & (f["phase"] == "AcceleratingDecline"),
        "intensity >= 0.95 + AccelDecline": lambda f: (f["intensity"] >= 0.95)
            & (f["phase"] == "AcceleratingDecline"),
        "intensity >= 0.99 (no direction)": lambda f: f["intensity"] >= 0.99,
    }

    events = {}
    for sym, frame in frames.items():
        ev = label_events(frame["close"].astype(float), sym, defn)
        onsets = ev.crash_onsets[(ev.crash_onsets >= oos[0]) & (ev.crash_onsets <= oos[-1])]
        if len(onsets):
            events[sym] = onsets
    total = sum(len(v) for v in events.values())
    print(f"crash onsets across {len(events)} symbols: {total} "
          f"({total / len(frames) / 5.4:.1f} per stock per year)\n")

    # A RANDOM signal with the same firing rate is the only honest baseline.
    # Crash events are frequent enough that a rule firing often will land near
    # one by chance, so raw recall on its own says nothing about skill.
    rng = np.random.default_rng(0)

    def measure(rule, randomise=False):
        caught = n_ev = n_sig = fa = 0
        leads = []
        for sym, onsets in events.items():
            sub = scored.xs(sym, level="symbol")
            sub = sub[(sub.index >= oos[0]) & (sub.index <= oos[-1])]
            sig = rule(sub)
            if randomise:
                k = int(sig.sum())
                arr = np.zeros(len(sub), dtype=bool)
                if k:
                    arr[rng.choice(len(sub), k, replace=False)] = True
                sig = pd.Series(arr, index=sub.index)
            rep = lead_time_report(sig, onsets, oos)
            caught += rep["n_caught"]; n_ev += rep["n_events"]
            n_sig += rep["n_signals"]; fa += rep["false_alarms"]
            leads += [x for x in rep["leads"] if x is not None]
        return {
            "signals": n_sig, "events": n_ev, "caught": caught,
            "recall": caught / max(n_ev, 1),
            "median_lead": float(np.median(leads)) if leads else float("nan"),
            "fa_per_sym_yr": fa / len(frames) / 5.4,
        }

    print("EARLY WARNING vs A RANDOM SIGNAL OF THE SAME FREQUENCY")
    print(f"{'rule':34s}{'signals':>8s}{'recall':>9s}{'random':>9s}{'skill':>8s}{'FA/sym/yr':>11s}")
    rows = []
    for name, rule in rules.items():
        real = measure(rule)
        rand = measure(rule, randomise=True)
        skill = real["recall"] / max(rand["recall"], 1e-9)
        print(f"{name:34s}{real['signals']:8d}{real['recall']:9.1%}"
              f"{rand['recall']:9.1%}{skill:7.2f}x{real['fa_per_sym_yr']:11.2f}")
        rows.append({"rule": name, **real, "random_recall": rand["recall"], "skill": skill})

    # Horizon decay -- where the information actually is.
    print("\nP(drawdown <= -10% within H days), signal = intensity 0.99 + AccelDecline")
    print(f"{'H':>3s}{'base rate':>11s}{'given signal':>14s}{'lift':>9s}")
    closes = {s: f["close"].astype(float) for s, f in frames.items()}
    blocks = []
    for sym, close in closes.items():
        sub = scored.xs(sym, level="symbol").reindex(close.index)
        v = close.to_numpy()
        cols = {f"dd{H}": np.array(
            [(v[i + 1:i + 1 + H].min() / v[i] - 1) if i + 1 + H <= len(v) else np.nan
             for i in range(len(v))]) for H in (1, 2, 3, 5, 10)}
        blk = pd.DataFrame(cols, index=close.index)
        blk["intensity"] = sub["intensity"]; blk["phase"] = sub["phase"]
        blocks.append(blk.loc[str(w.backtest_start):str(w.backtest_end)])
    allc = pd.concat(blocks)
    sig = (allc["intensity"] >= 0.99) & (allc["phase"] == "AcceleratingDecline")
    decay = []
    for H in (1, 2, 3, 5, 10):
        col = f"dd{H}"
        base = (allc[col] <= -0.10).mean()
        given = (allc.loc[sig, col] <= -0.10).mean()
        print(f"{H:3d}{base:10.2%}{given:14.2%}{given / base:8.1f}x")
        decay.append({"horizon": H, "base": base, "given_signal": given, "lift": given / base})

    print(f"\nmedian same-day return on signal days: {allc.loc[sig, 'dd1'].median():+.2%}"
          f"   (all days {allc['dd1'].median():+.2%})")

    table = pd.DataFrame(rows).set_index("rule")
    pd.DataFrame(decay).to_csv(REPORTS / "phase4_horizon_decay.csv", index=False)
    table.to_csv(REPORTS / "phase4_lead_time.csv")
    print(f"\nwrote {REPORTS / 'phase4_lead_time.csv'}")
    return {"lead_time": table, "events": events}


PHASES = {
    1: ("Data layer", phase_1),
    2: ("Features", phase_2),
    3: ("Isolation Forest + intensity", phase_3),
    4: ("Labels + lead time", phase_4),
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
