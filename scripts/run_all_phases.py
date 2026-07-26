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
from qbeast_crash.backtest import RateCard, TaxRates, run_backtest
from qbeast_crash.labels import LEAD_BUCKETS, CrashDefinition, label_events, lead_time_report
from qbeast_crash.model import AnomalyDetector, purge_crisis_dates
from qbeast_crash.plots import (
    drawdown_stats,
    plot_drawdown_scatter,
    plot_portfolio,
    plot_portfolio_schemes,
    plot_scheme_comparison,
    plot_symbol,
)
from qbeast_crash.retrain import SCHEMES, SchemeResult, walk_forward
from qbeast_crash.signals import (
    ReentryRule,
    SignalConfig,
    equal_weight_equity,
    generate_signals,
    market_signal,
    performance,
    signal_strength,
)
from qbeast_crash.features import (
    FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    compute_market_features,
    compute_precursors,
)

CFG = DEFAULT_CONFIG

#: None means the full universe. Set by --dev or --symbols.
UNIVERSE: list[str] | None = None


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
        frames, reports = load_universe(UNIVERSE, config=CFG)
    calendar = trading_calendar(CFG)
    print(f"loaded {len(frames)} usable symbols over {len(calendar)} trading days\n")

    # The gate's universe-size and coverage checks assume the full 96 symbols.
    # On a restricted run they would fail for a reason that is not a data
    # problem, so they are reported but not enforced.
    gate = run_quality_gate(frames, reports, calendar, CFG, raise_on_error=False)
    if UNIVERSE is not None:
        gate.checks = [c for c in gate.checks
                       if c.name not in ("universe_size", "train_window_coverage")]
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

    # A universe smaller than the configured floor would blank every
    # cross-sectional feature. Scale the floor instead, and rely on the warning
    # printed at startup rather than silently emitting NaNs.
    floor = min(CFG.features.min_symbols, max(3, len(frames) // 2))
    market = compute_market_features(
        ctx["panel"], panel_of("phase"), panel_of("slope_z"), ctx["mask"],
        corr_window=CFG.features.corr_window,
        ma_window=CFG.features.ma_window,
        min_symbols=floor,
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


# =====================================================================
# Phase 5 -- signal generation
# =====================================================================
def phase_5(ctx: dict) -> dict:
    """
    Turn intensity into a hold/cash position, and pick the re-entry rule by
    measurement rather than by opinion.

    Exiting is the easy half -- the exit rule has a 110x edge at one day. The
    hard half is coming back: a strategy that exits well but re-enters late
    loses more to the missed rebound than it saved on the decline.
    """
    _rule("PHASE 5  Signal generation")

    scored, frames, calendar = ctx["scored"], ctx["frames"], ctx["calendar"]
    w = CFG.windows
    cfg = SignalConfig()
    oos = calendar[(calendar >= pd.Timestamp(w.backtest_start))
                   & (calendar <= pd.Timestamp(w.backtest_end))]
    years = (oos[-1] - oos[0]).days / 365.25

    print(f"exit rule: intensity >= {cfg.exit_intensity} AND AcceleratingDecline")
    print(f"guards: min_hold {cfg.min_hold}, cooldown {cfg.cooldown}, "
          f"max_cash {cfg.max_cash_days} sessions\n")

    # ---- compare re-entry rules -------------------------------------
    # Gross of costs. Costs come in Phase 6, but a rule that cannot beat
    # buy-and-hold before costs will not beat it after them.
    print("RE-ENTRY RULE COMPARISON (gross of costs)")
    print(f"{'rule':16s}{'trades/sym/yr':>15s}{'days in cash':>14s}"
          f"{'strat CAGR':>12s}{'B&H CAGR':>10s}{'maxDD':>9s}{'B&H maxDD':>11s}")

    # SIMPLE returns, aligned as a panel -- equal_weight_equity compounds the
    # cross-sectional mean each day, which is a real portfolio. Averaging
    # per-stock log equity would discard the diversification benefit and
    # understate every figure by ~4.75pp CAGR.
    ret_panel = pd.DataFrame(
        {s: f["close"].astype(float).pct_change() for s, f in frames.items()}
    ).reindex(oos)

    results = {}
    for rule in ReentryRule.ALL:
        pos, n_trades = {}, 0
        for sym, frame in frames.items():
            sub = scored.xs(sym, level="symbol")
            sub = sub[(sub.index >= oos[0]) & (sub.index <= oos[-1])]
            if len(sub) < 250:
                continue
            sig = generate_signals(sub, cfg, reentry=rule)
            pos[sym] = sig["in_position"].reindex(oos, fill_value=True)
            n_trades += int((sig["action"] == "EXIT").sum())

        pos_panel = pd.DataFrame(pos).reindex(columns=ret_panel.columns, fill_value=True)
        cols = ret_panel.columns.intersection(pos_panel.columns)

        strat = performance(equal_weight_equity(ret_panel[cols], pos_panel[cols]), years)
        bh = performance(equal_weight_equity(ret_panel[cols]), years)

        tpy = n_trades / len(frames) / years
        cash = 1.0 - pos_panel.to_numpy().mean()
        print(f"{rule:16s}{tpy:15.2f}{cash:13.1%}"
              f"{strat['cagr']:12.2%}{bh['cagr']:10.2%}"
              f"{strat['max_drawdown']:9.1%}{bh['max_drawdown']:11.1%}")
        results[rule] = {"cagr": strat["cagr"], "maxdd": strat["max_drawdown"],
                         "trades_per_sym_yr": tpy, "cash_share": float(cash),
                         "bh_cagr": bh["cagr"], "bh_maxdd": bh["max_drawdown"]}

    best = max(results, key=lambda r: results[r]["maxdd"])     # least negative
    print(f"\nshallowest drawdown: {best}")
    delta_pp = (results[best]["maxdd"] - results[best]["bh_maxdd"]) * 100
    print(f"buy-and-hold maxDD {results[best]['bh_maxdd']:.1%} -> "
          f"strategy {results[best]['maxdd']:.1%}  ({delta_pp:+.1f} pp)")

    # ---- stress test: does it work when there IS a sharp crash? ------
    # The 2021-2026 window contains no violent decline. Its worst drawdown ran
    # 154 days at 16.5% annualised volatility with a single 3%+ day -- a slow
    # bleed with nothing for an anomaly detector to fire on. A crash-reaction
    # system correctly does almost nothing there, so the window cannot tell us
    # whether the system works.
    #
    # 2020 can. Training on 2016-2019 keeps COVID genuinely out of sample.
    print("\nSTRESS TEST -- 2020, model trained 2016-2019 (COVID never seen)")
    tr_dates = ctx["features"].index.get_level_values("date")
    stress_det = AnomalyDetector(
        n_estimators=CFG.model.n_estimators, max_samples=CFG.model.max_samples,
        random_state=CFG.model.random_state,
    ).fit(ctx["features"][(tr_dates >= pd.Timestamp("2016-01-01"))
                          & (tr_dates <= pd.Timestamp("2019-12-31"))])
    stressed = ctx["features"].assign(intensity=stress_det.intensity(ctx["features"]))

    print(f"{'window':26s}{'strat ret':>11s}{'B&H ret':>10s}"
          f"{'strat DD':>10s}{'B&H DD':>9s}{'DD saved':>10s}{'trades':>9s}")
    for label, a, b in [("2020 Feb-Apr (crash)", "2020-02-01", "2020-04-30"),
                        ("2020 full year", "2020-01-01", "2020-12-31"),
                        ("2021-2026 (no crash)", "2021-01-01", "2026-06-05")]:
        win = calendar[(calendar >= pd.Timestamp(a)) & (calendar <= pd.Timestamp(b))]
        yrs = max((win[-1] - win[0]).days / 365.25, 1 / 12)
        pos, n_tr = {}, 0
        for sym, frame in frames.items():
            sub = stressed.xs(sym, level="symbol")
            sub = sub[(sub.index >= win[0]) & (sub.index <= win[-1])]
            if len(sub) < 40:
                continue
            sg = generate_signals(sub, cfg, reentry=best)
            pos[sym] = sg["in_position"].reindex(win, fill_value=True)
            n_tr += int((sg["action"] == "EXIT").sum())

        rp = pd.DataFrame(
            {s: f["close"].astype(float).pct_change() for s, f in frames.items()}
        ).reindex(win)
        pp = pd.DataFrame(pos)
        cols = rp.columns.intersection(pp.columns)

        S2 = equal_weight_equity(rp[cols], pp[cols])
        B2 = equal_weight_equity(rp[cols])
        ds = performance(S2, yrs)["max_drawdown"] * 100
        db = performance(B2, yrs)["max_drawdown"] * 100
        print(f"{label:26s}{(S2.iloc[-1] - 1) * 100:10.1f}%{(B2.iloc[-1] - 1) * 100:9.1f}%"
              f"{ds:9.1f}%{db:8.1f}%{ds - db:+9.1f}pp{n_tr / len(frames) / yrs:9.2f}")

    # ---- market-wide overlay ----------------------------------------
    mkt = market_signal(ctx["market"]).reindex(oos).fillna(False)
    print(f"\nmarket-wide de-risk flag: {int(mkt.sum())} sessions of {len(oos)} "
          f"({mkt.mean():.2%}, {mkt.sum() / years:.1f}/yr)")
    if mkt.any():
        runs = (mkt != mkt.shift()).cumsum()[mkt]
        print(f"  distinct episodes: {runs.nunique()}   "
              f"dates: {', '.join(str(d.date()) for d in mkt[mkt].index[:5])}")

    # ---- persist the chosen configuration ---------------------------
    signals = {}
    for sym, frame in frames.items():
        sub = scored.xs(sym, level="symbol")
        sub = sub[(sub.index >= oos[0]) & (sub.index <= oos[-1])]
        if len(sub) < 250:
            continue
        sig = generate_signals(sub, cfg, reentry=best)
        sig["strength"] = signal_strength(sub)
        signals[sym] = sig

    panel = pd.concat(signals, names=["symbol", "date"]).reorder_levels(
        ["date", "symbol"]).sort_index()
    panel.to_parquet(DATA_PROCESSED / "signals.parquet")
    pd.DataFrame(results).T.to_csv(REPORTS / "phase5_reentry_comparison.csv")

    print(f"\nwrote signals.parquet {panel.shape} (re-entry rule: {best})")
    return {"signals": panel, "reentry_rule": best, "market_flag": mkt}


# =====================================================================
# Phase 6 -- backtest with real costs and tax
# =====================================================================
def phase_6(ctx: dict) -> dict:
    """
    Apply the NSE delivery cost stack and Indian capital gains tax.

    At 0.4 trades per symbol per year, brokerage cannot meaningfully erode
    this strategy. The interesting question is TAX: every crash exit converts a
    long-term holding into a short-term one, moving the rate from 12.5% to 20%.
    """
    _rule("PHASE 6  Backtest with costs and tax")

    signals, frames, calendar = ctx["signals"], ctx["frames"], ctx["calendar"]
    w = CFG.windows
    oos = calendar[(calendar >= pd.Timestamp(w.backtest_start))
                   & (calendar <= pd.Timestamp(w.backtest_end))]
    years = (oos[-1] - oos[0]).days / 365.25
    capital = 1_000_000.0

    prices = pd.DataFrame(
        {s: f["close"].astype(float) for s, f in frames.items()}
    ).reindex(oos)

    pos = signals["in_position"].unstack("symbol").reindex(
        index=oos, columns=prices.columns).fillna(True)

    # Volatility ratio drives slippage: this strategy trades only on unusual
    # days, when spreads are genuinely wider.
    vol = ctx["features"]["vol_ratio"].unstack("symbol").reindex(
        index=oos, columns=prices.columns)

    always = pd.DataFrame(True, index=oos, columns=prices.columns)

    print(f"capital Rs {capital:,.0f}  |  {len(prices.columns)} symbols  "
          f"|  {len(oos)} sessions\n")

    runs = {}
    for label, p in (("strategy", pos), ("buy & hold", always)):
        runs[label] = run_backtest(prices, p, capital, vol_ratio=vol,
                                   rates=RateCard(), tax=TaxRates())

    # Equity is already net of transaction costs -- they are deducted inside the
    # engine at each fill. The tax column is the additional deduction.
    print(f"{'':14s}{'CAGR net of':>13s}{'CAGR after':>12s}{'':>3s}"
          f"{'maxDD':>9s}{'trades':>8s}{'costs Rs':>11s}{'tax Rs':>11s}")
    print(f"{'':14s}{'costs':>13s}{'tax':>12s}")
    summary = {}
    for label, res in runs.items():
        eq, eq_t = res["equity"], res["equity_after_tax"]
        net_costs = (eq.iloc[-1] / capital) ** (1 / years) - 1
        net_all = (eq_t.iloc[-1] / capital) ** (1 / years) - 1
        dd = (eq_t / eq_t.cummax() - 1).min()
        print(f"{label:14s}{net_costs:12.2%}{net_all:12.2%}{'':>3s}{dd:9.1%}"
              f"{len(res['trades']):8d}{res['total_costs']:11,.0f}"
              f"{res['total_tax']:11,.0f}")
        summary[label] = {"cagr_net_costs": net_costs, "cagr_net_all": net_all,
                          "maxdd": dd, "trades": len(res["trades"]),
                          "costs": res["total_costs"], "tax": res["total_tax"],
                          "tax_liquidated": res["total_tax_liquidated"],
                          "deferred_tax": res["deferred_tax"]}

    s_, b_ = summary["strategy"], summary["buy & hold"]
    print(f"\ndrawdown: buy & hold {b_['maxdd']:.1%} -> strategy {s_['maxdd']:.1%}"
          f"  ({(s_['maxdd'] - b_['maxdd']) * 100:+.1f} pp)")
    print(f"cost drag: Rs {s_['costs'] - b_['costs']:,.0f} extra over {years:.1f} years "
          f"({(s_['costs'] - b_['costs']) / capital / years * 100:.3f}% of capital per year)")

    # ---- the tax question -------------------------------------------
    print("\nTAX: does exiting convert long-term gains into short-term?")
    for label, res in runs.items():
        rl = res["realised"]
        if rl.empty:
            print(f"  {label:12s} no realised gains")
            continue
        short = (rl["holding_days"] < 365)
        print(f"  {label:12s} {len(rl):4d} sales   "
              f"{short.mean():5.1%} short-term   "
              f"Rs {res['total_tax']:>10,.0f} tax   "
              f"({res['total_tax'] / capital * 100:.2f}% of capital)")

    print("\n  Buy & hold never sells, so on a realised basis it pays no tax at all.")
    print("  That is deferral, not saving -- it ends the period holding an")
    print("  unrealised liability. Liquidating both books at the final close:")
    print(f"\n{'':14s}{'tax paid':>12s}{'deferred':>12s}{'total if':>12s}")
    print(f"{'':14s}{'as we go':>12s}{'liability':>12s}{'liquidated':>12s}")
    for label, res in runs.items():
        print(f"  {label:12s}{res['total_tax']:12,.0f}{res['deferred_tax']:12,.0f}"
              f"{res['total_tax_liquidated']:12,.0f}")

    extra = s_["tax_liquidated"] - b_["tax_liquidated"]
    print(f"\n  like-for-like tax difference: Rs {extra:,.0f} "
          f"({extra / capital * 100:+.2f}% of capital)")

    # The only comparison that settles it: what you walk away with.
    print(f"\n{'':14s}{'terminal':>13s}{'tax if':>13s}{'AFTER-TAX':>13s}{'CAGR':>9s}")
    print(f"{'':14s}{'equity':>13s}{'liquidated':>13s}{'WEALTH':>13s}")
    for label, res in runs.items():
        term = res["equity"].iloc[-1]
        after = term - res["total_tax_liquidated"]
        print(f"  {label:12s}{term:13,.0f}{res['total_tax_liquidated']:13,.0f}"
              f"{after:13,.0f}{(after / capital) ** (1 / years) - 1:9.2%}")

    s_after = runs["strategy"]["equity"].iloc[-1] - s_["tax_liquidated"]
    b_after = runs["buy & hold"]["equity"].iloc[-1] - b_["tax_liquidated"]
    print(f"\n  after-tax difference: Rs {s_after - b_after:,.0f} "
          f"({(s_after - b_after) / capital * 100:+.2f}% of capital)")
    print(f"  short-term share of strategy sales: "
          f"{(runs['strategy']['realised']['holding_days'] < 365).mean():.1%}")

    for label, res in runs.items():
        key = label.replace(" & ", "_").replace(" ", "_")
        res["equity_after_tax"].to_frame("equity").to_parquet(
            DATA_PROCESSED / f"equity_{key}.parquet")
        if len(res["trades"]):
            res["trades"].to_csv(REPORTS / f"phase6_trades_{key}.csv", index=False)
    pd.DataFrame(summary).T.to_csv(REPORTS / "phase6_summary.csv")

    if UNIVERSE is not None and len(UNIVERSE) < 30:
        # A small universe concentrates position weight, so one stock's crash
        # can carry the entire result. Measured on the 10-symbol dev subset,
        # ADANIENT alone contributed 93% of the edge while six of ten symbols
        # never traded. The same subset reports +33% where the full universe
        # reports -4.4%.
        contrib = []
        for sym in prices.columns:
            sub = signals.xs(sym, level="symbol")
            sub = sub[(sub.index >= oos[0]) & (sub.index <= oos[-1])]
            ret = prices[sym].pct_change().fillna(0.0)
            bh = (1 + ret).cumprod().iloc[-1]
            st = (1 + ret * sub["in_position"].astype(float)).cumprod().iloc[-1]
            contrib.append((sym, (st - bh) * 100, int((sub["action"] == "EXIT").sum())))
        contrib.sort(key=lambda x: -abs(x[1]))
        total = sum(abs(c[1]) for c in contrib) or 1.0
        untraded = sum(1 for c in contrib if c[2] == 0)

        print("\n" + "!" * 68)
        print("SMALL-UNIVERSE WARNING -- do not read portfolio results from this run")
        print("!" * 68)
        print(f"  top contributor {contrib[0][0]} accounts for "
              f"{abs(contrib[0][1]) / total:.0%} of the total edge")
        print(f"  {untraded} of {len(contrib)} symbols never traded at all")
        print("  Position weight is concentrated, so one stock's crash can carry")
        print("  the whole result. Confirm on the full universe before concluding")
        print("  anything about whether the strategy works.")

    print(f"\nwrote equity curves and trade logs -> {REPORTS}")
    return {"backtest": runs, "summary": summary}


# =====================================================================
# Phase 7 -- retraining scheme comparison
# =====================================================================
def phase_7(ctx: dict) -> dict:
    """
    Refit monthly under four rules for choosing training data, and rank them.

    Phase 3 fitted once and scored five years forward, which is not how the
    system would run live. This is walk-forward: at each refit the model sees
    only data from before that date, and scores only until the next refit.
    """
    _rule("PHASE 7  Retraining scheme comparison")

    features, market, frames = ctx["features"], ctx["market"], ctx["frames"]
    calendar, w = ctx["calendar"], CFG.windows
    oos = calendar[(calendar >= pd.Timestamp(w.backtest_start))
                   & (calendar <= pd.Timestamp(w.backtest_end))]
    years = w.backtest_years

    print(f"walk-forward, monthly refits, {oos[0].date()} to {oos[-1].date()}")
    print("only the training SET varies -- the schedule is identical across "
          "schemes, so\nthe effect is attributable\n")

    ret_panel = pd.DataFrame(
        {s: f["close"].astype(float).pct_change() for s, f in frames.items()}
    ).reindex(oos)
    bh_equity = equal_weight_equity(ret_panel)
    bh = performance(bh_equity, years)

    results, intensities = [], {}
    scheme_signals, scheme_equity = {}, {}
    for scheme in SCHEMES:
        t0 = time.time()
        intensity = walk_forward(features, calendar, scheme, CFG, market, verbose=True)
        intensities[scheme] = intensity

        scored = features.assign(intensity=intensity)
        pos, n_exits, sig_by_symbol = {}, 0, {}
        for sym in frames:
            try:
                sub = scored.xs(sym, level="symbol")
            except KeyError:
                continue
            sub = sub[(sub.index >= oos[0]) & (sub.index <= oos[-1])]
            if len(sub) < 250 or sub["intensity"].notna().sum() < 100:
                continue
            sig = generate_signals(sub, CFG.signals, reentry=ctx.get("reentry_rule", "time"))
            pos[sym] = sig["in_position"].reindex(oos, fill_value=True)
            sig_by_symbol[sym] = sig
            n_exits += int((sig["action"] == "EXIT").sum())

        pos_panel = pd.DataFrame(pos).reindex(columns=ret_panel.columns, fill_value=True)
        cols = ret_panel.columns.intersection(pos_panel.columns)
        eq = equal_weight_equity(ret_panel[cols], pos_panel[cols])
        strat = performance(eq, years)
        scheme_equity[scheme] = eq
        scheme_signals[scheme] = sig_by_symbol

        results.append(SchemeResult(
            scheme=scheme,
            cagr=strat["cagr"],
            max_drawdown=strat["max_drawdown"],
            bh_max_drawdown=bh["max_drawdown"],
            trades_per_symbol_year=n_exits / max(len(frames), 1) / years,
            signals=int((scored["intensity"] >= CFG.model.intensity_high).sum()),
        ))
        print(f"    {'':12s} fitted in {time.time() - t0:.0f}s")

    print(f"\n{'scheme':14s}{'CAGR':>9s}{'maxDD':>9s}{'vs B&H':>10s}"
          f"{'trades/sym/yr':>15s}{'efficiency':>12s}")
    print(f"{'buy & hold':14s}{bh['cagr']:8.2%}{bh['max_drawdown']:9.1%}"
          f"{'--':>10s}{0.0:15.2f}{'--':>12s}")
    for r in results:
        print(f"{r.scheme:14s}{r.cagr:8.2%}{r.max_drawdown:9.1%}"
              f"{r.drawdown_saved:+9.1f}pp{r.trades_per_symbol_year:15.2f}"
              f"{r.efficiency:12.1f}")

    ranked = sorted(results, key=lambda r: -r.drawdown_saved)
    print(f"\nshallowest drawdown : {ranked[0].scheme} "
          f"({ranked[0].drawdown_saved:+.1f}pp)")
    by_eff = sorted([r for r in results if r.trades_per_symbol_year > 0],
                    key=lambda r: -r.efficiency)
    if by_eff:
        print(f"best per unit turnover: {by_eff[0].scheme} "
              f"({by_eff[0].efficiency:.1f}pp per trade/sym/yr)")

    spread = max(r.drawdown_saved for r in results) - min(r.drawdown_saved for r in results)
    print(f"\nspread across schemes : {spread:.1f}pp of drawdown")
    if spread < 2.0:
        print("  The schemes are within noise of each other. On this evidence the")
        print("  choice of retraining rule is not a meaningful lever -- which is")
        print("  worth knowing, and is a result rather than a failure.")

    table = pd.DataFrame([{
        "scheme": r.scheme, "cagr": r.cagr, "max_drawdown": r.max_drawdown,
        "bh_max_drawdown": r.bh_max_drawdown, "drawdown_saved_pp": r.drawdown_saved,
        "trades_per_sym_yr": r.trades_per_symbol_year,
        "efficiency": r.efficiency, "high_band_days": r.signals,
    } for r in results]).set_index("scheme")
    table.to_csv(REPORTS / "phase7_scheme_comparison.csv")
    pd.DataFrame(intensities).to_parquet(DATA_PROCESSED / "intensity_by_scheme.parquet")

    # ---- coverage: how many stocks does each scheme actually act on? ----
    # This matters more than it looks. A static fit on 2016-2020 reaches only
    # 3 of 10 dev symbols, because that window contains COVID and sets a bar
    # calm stocks can never clear. Walk-forward windows drop COVID as they
    # roll, so coverage roughly triples.
    print(f"\nCOVERAGE -- symbols the scheme actually traded")
    print(f"{'scheme':14s}{'traded':>10s}{'of':>4s}{'exits':>9s}{'per sym/yr':>13s}")
    for scheme, sigs in scheme_signals.items():
        traded = sum(1 for s_ in sigs.values() if (s_["action"] == "EXIT").any())
        tot = sum(int((s_["action"] == "EXIT").sum()) for s_ in sigs.values())
        print(f"{scheme:14s}{traded:10d}{len(frames):4d}{tot:9d}"
              f"{tot / max(len(frames), 1) / years:13.2f}")

    # ---- charts ---------------------------------------------------------
    figures = REPORTS / "figures"
    main = [s_ for s_ in ("rolling", "incremental", "ewma") if s_ in scheme_equity]
    plot_portfolio_schemes({k: scheme_equity[k] for k in main},
                           equal_weight_equity(ret_panel), figures)

    drawn = 0
    for sym, frame in frames.items():
        close = frame["close"].astype(float).reindex(oos).dropna()
        if len(close) < 250:
            continue
        per_scheme = {k: scheme_signals[k][sym] for k in main if sym in scheme_signals[k]}
        if not per_scheme:
            continue
        if plot_scheme_comparison(sym, close, per_scheme, figures / "schemes") is not None:
            drawn += 1

    # ---- per-stock drawdown, per scheme ---------------------------------
    rows = []
    for sym, frame in frames.items():
        close = frame["close"].astype(float).reindex(oos).dropna()
        if len(close) < 250:
            continue
        r_ = close.pct_change().fillna(0.0)
        row = {"symbol": sym,
               "bh_max_dd": drawdown_stats((1 + r_).cumprod()).max_drawdown}
        for k in main:
            sig = scheme_signals[k].get(sym)
            if sig is None:
                continue
            held = sig["in_position"].reindex(close.index).fillna(True).astype(bool)
            eq = (1 + r_ * held.astype(float)).cumprod()
            row[f"{k}_max_dd"] = drawdown_stats(eq).max_drawdown
            row[f"{k}_return"] = eq.iloc[-1] - 1
            row[f"{k}_exits"] = int((sig["action"] == "EXIT").sum())
        rows.append(row)
    per_stock = pd.DataFrame(rows).set_index("symbol")
    per_stock.to_csv(REPORTS / "phase7_drawdown_by_symbol_by_scheme.csv")

    print(f"\nwrote {REPORTS / 'phase7_scheme_comparison.csv'}")
    print(f"      {REPORTS / 'phase7_drawdown_by_symbol_by_scheme.csv'}")
    print(f"      {figures / 'portfolio_schemes.png'}")
    print(f"      {figures / 'schemes'}/  ({drawn} per-stock charts)")
    return {"schemes": table, "scheme_intensity": intensities,
            "scheme_signals": scheme_signals, "per_stock_schemes": per_stock}


# =====================================================================
# Phase 8 -- drawdown analysis and per-stock charts
# =====================================================================
def phase_8(ctx: dict) -> dict:
    """
    Per-symbol drawdown statistics and the charts that let them be inspected.

    A single averaged drawdown number can hide almost anything. The per-symbol
    table and the scatter chart show whether a headline is broad-based or
    carried by a handful of names -- which, on this data, is the question that
    matters.
    """
    _rule("PHASE 8  Drawdown analysis and charts")

    signals, frames, calendar = ctx["signals"], ctx["frames"], ctx["calendar"]
    scored, w = ctx["scored"], CFG.windows
    oos = calendar[(calendar >= pd.Timestamp(w.backtest_start))
                   & (calendar <= pd.Timestamp(w.backtest_end))]
    figures = REPORTS / "figures"
    defn = CFG.labels

    rows, drawn = [], 0
    for sym, frame in frames.items():
        try:
            sig = signals.xs(sym, level="symbol")
        except KeyError:
            continue
        sig = sig.reindex(oos)
        close = frame["close"].astype(float).reindex(oos).dropna()
        if len(close) < 250:
            continue

        held = sig["in_position"].reindex(close.index).fillna(True).astype(bool)
        ret = close.pct_change().fillna(0.0)
        strat = (1 + ret * held.astype(float)).cumprod()
        bench = (1 + ret).cumprod()

        s_dd, b_dd = drawdown_stats(strat), drawdown_stats(bench)
        rows.append({
            "symbol": sym,
            "strategy_max_drawdown": s_dd.max_drawdown,
            "bh_max_drawdown": b_dd.max_drawdown,
            "drawdown_saved_pp": (s_dd.max_drawdown - b_dd.max_drawdown) * 100,
            "strategy_return": strat.iloc[-1] - 1,
            "bh_return": bench.iloc[-1] - 1,
            "days_under_water": s_dd.days_under_water,
            "bh_days_under_water": b_dd.days_under_water,
            "longest_underwater": s_dd.longest_underwater,
            "bh_longest_underwater": b_dd.longest_underwater,
            "time_to_recover": s_dd.time_to_recover,
            "bh_time_to_recover": b_dd.time_to_recover,
            "exits": int((sig["action"] == "EXIT").sum()),
            "pct_days_in_cash": 100 * (1 - held.mean()),
        })

        ev = label_events(frame["close"].astype(float), sym, defn)
        onsets = ev.crash_onsets[(ev.crash_onsets >= oos[0]) & (ev.crash_onsets <= oos[-1])]
        if plot_symbol(sym, close, sig, scored.xs(sym, level="symbol")["intensity"],
                       onsets, figures / "symbols") is not None:
            drawn += 1

    table = pd.DataFrame(rows).set_index("symbol").sort_values("drawdown_saved_pp")
    table.to_csv(REPORTS / "phase8_drawdown_by_symbol.csv")

    helped = (table["drawdown_saved_pp"] > 0.1).sum()
    hurt = (table["drawdown_saved_pp"] < -0.1).sum()
    untouched = (table["exits"] == 0).sum()

    print(f"symbols analysed : {len(table)}   charts drawn: {drawn}")
    print(f"  shallower drawdown than buy & hold : {helped}")
    print(f"  deeper                             : {hurt}")
    print(f"  never traded at all                : {untouched}")
    print(f"\n  mean drawdown saved   : {table['drawdown_saved_pp'].mean():+.2f}pp")
    print(f"  median drawdown saved : {table['drawdown_saved_pp'].median():+.2f}pp")

    traded = table[table["exits"] > 0]
    if len(traded):
        print(f"\n  among the {len(traded)} symbols it actually traded:")
        print(f"    mean drawdown saved : {traded['drawdown_saved_pp'].mean():+.2f}pp")
        print(f"    best  {traded.index[-1]:12s} {traded['drawdown_saved_pp'].iloc[-1]:+.1f}pp")
        print(f"    worst {traded.index[0]:12s} {traded['drawdown_saved_pp'].iloc[0]:+.1f}pp")

    print(f"\n  time under water (sessions, median):")
    print(f"    strategy   {table['days_under_water'].median():.0f}   "
          f"longest spell {table['longest_underwater'].median():.0f}")
    print(f"    buy & hold {table['bh_days_under_water'].median():.0f}   "
          f"longest spell {table['bh_longest_underwater'].median():.0f}")

    scatter = plot_drawdown_scatter(table, figures)

    eq_s = DATA_PROCESSED / "equity_strategy.parquet"
    eq_b = DATA_PROCESSED / "equity_buy_hold.parquet"
    if eq_s.exists() and eq_b.exists():
        plot_portfolio(pd.read_parquet(eq_s)["equity"],
                       pd.read_parquet(eq_b)["equity"], figures)
        print(f"\n  portfolio chart -> {figures / 'portfolio.png'}")
    print(f"  scatter         -> {scatter}")
    print(f"  per-symbol      -> {figures / 'symbols'}/  ({drawn} charts)")
    print(f"  table           -> {REPORTS / 'phase8_drawdown_by_symbol.csv'}")
    return {"drawdown_table": table}


PHASES = {
    1: ("Data layer", phase_1),
    2: ("Features", phase_2),
    3: ("Isolation Forest + intensity", phase_3),
    4: ("Labels + lead time", phase_4),
    5: ("Signal generation", phase_5),
    6: ("Backtest with costs", phase_6),
    7: ("Retraining comparison", phase_7),
    8: ("Drawdown analysis + charts", phase_8),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="QBEAST crash detection pipeline")
    ap.add_argument("--phase", type=int, help="run one phase only")
    ap.add_argument("--from", dest="start", type=int, help="run from this phase onward")
    ap.add_argument("--list", action="store_true", help="list phases and exit")
    ap.add_argument("--dev", action="store_true",
                    help="run on the 10-symbol dev universe (seconds, not minutes)")
    ap.add_argument("--symbols", type=str,
                    help="comma-separated symbols, e.g. RELIANCE,TCS,ITC")
    args = ap.parse_args()

    global UNIVERSE
    if args.symbols:
        UNIVERSE = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.dev:
        UNIVERSE = list(CFG.data.dev_universe)
    if UNIVERSE is not None:
        print(f"universe restricted to {len(UNIVERSE)} symbols: {', '.join(UNIVERSE)}")
        if len(UNIVERSE) < 20:
            print("NOTE: cross-sectional features (breadth, correlation) are noisy on a\n"
                  "      small universe. Confirm market-wide results on the full set.")

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
