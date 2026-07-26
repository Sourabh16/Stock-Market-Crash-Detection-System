<h1 align="center">QBEAST — Stock Market Crash Detection System</h1>

<p align="center">
  <em>Isolation-Forest anomaly detection for crash and rally prediction on Indian equities</em><br>
  <sub>NIFTY 100 · end-of-day bars · 2016–2026</sub>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white">
  <img alt="tests" src="https://img.shields.io/badge/tests-48%20passing-1E7B34">
  <img alt="universe" src="https://img.shields.io/badge/universe-96%20symbols-1F4E79">
  <img alt="bars" src="https://img.shields.io/badge/trading%20days-5820-1F4E79">
  <img alt="status" src="https://img.shields.io/badge/status-phase%202%20of%2010-orange">
</p>

---

## The idea

Most trading systems try to earn returns by trading often. This one does the opposite.

It stays **fully invested by default**, and only acts when it thinks something is about to
break — selling into cash before a crash, buying back before a rally, and otherwise doing
nothing. Every trade costs brokerage, STT, stamp duty, GST and slippage, so a trade has to
earn its place.

The system is judged on two numbers:

| Metric | Question it answers |
|:--|:--|
| **Drawdown reduction** | How much of buy-and-hold's worst peak-to-trough loss did we avoid? |
| **Lead time** | How many trading days *before* the crash did we raise the alarm? |

Lead time is reported as a **distribution across every crash event**, never as an average —
an average silently hides the events that were missed entirely.

---

## How it works

Isolation Forest is an unsupervised anomaly detector. It is never shown a labelled crash.
It builds many random trees, splitting on random features at random values until every
data point sits alone, then counts the cuts each point needed. Outliers get isolated in a
few cuts; ordinary points buried in dense regions need many. Short path length = anomalous.

**The catch that shapes the entire architecture:** Isolation Forest is *direction-blind*. It
flags **unusual**, not **unusually bad** — a violent rally is exactly as anomalous as a
violent crash. So the anomaly score alone can never produce a buy or a sell.

That is why the detector is split into three parts:

```
   Isolation Forest  ──▶  something unusual is happening today
        + slope      ──▶  ...and it is moving DOWN
        + acceleration ─▶  ...and it is getting WORSE, not exhausting
                            ═══════════════════════════════════
                                  → confirmed sell signal
```

Slope and acceleration are the first and second derivatives of **log** price. The sign pair
is the whole signal logic:

| Slope | Acceleration | Phase | Reading |
|:--|:--|:--|:--|
| ▼ | ▼ | `AcceleratingDecline` | falling and worsening → **exit** |
| ▼ | ▲ | `DeceleratingDecline` | selloff exhausting → watch |
| ▲ | ▲ | `AcceleratingAdvance` | rally building → **enter** |
| ▲ | ▼ | `DeceleratingAdvance` | topping out → caution |

---

## Findings so far

**The market-wide trigger, calibrated from data rather than guessed.** With no tuning, the
fraction of the universe in `AcceleratingDecline` isolates real market events:

| Date | Universe declining | Median slope (σ/day) | Event |
|:--|--:|--:|:--|
| 2020-03-12 | 92.1% | **−2.11** | COVID crash |
| 2026-03-31 | 90.5% | −0.79 | — |
| 2016-02-11 | 90.1% | −0.94 | Feb-2016 global selloff |
| 2022-09-26 | 87.2% | −0.67 | Fed / GBP crisis |
| 2021-12-20 | 86.2% | −0.82 | Omicron |

Breadth **alone is not enough** — ~90% of the universe was falling during both COVID and
mild pullbacks. Only median slope separates a systemic crash from a wobble (−2.11 vs ≈−0.8).
Hence the two-condition trigger: `breadth ≥ 75% AND median slope ≤ −1.5σ`.

**A crash was muting its own signal.** Normalising slope by *contemporaneous* 60-day
volatility scored the COVID crash at only −1.67σ. By mid-March 2020 the volatility window
was itself full of crash days, so the denominator had grown and the crash was quietly
dividing away its own severity. Lagging the denominator by 20 days — so it describes the
calm the move is *departing from* — scores the same event at **−3.01σ**.

**A look-ahead bug in the inherited regime detector.** It computed percentile thresholds
from the *entire* price history and applied them to every bar, while documenting itself as
causal. On RELIANCE 2016–2026, **9.9% of regime labels change** once made properly causal —
and the biased version produced *more* `Rally` and *more* `Crashing` calls, because knowing
the full distribution let it place days into the tails with unearned confidence. Those are
precisely the labels that drive entries and exits.

---

## Data integrity

Before any model code was written, the 100 raw CSVs were audited. Five defects were found,
each of which would have silently corrupted results:

| # | Defect | Evidence | Handling |
|:--|:--|:--|:--|
| 1 | `adj_close` is not an adjusted series | Differs from `close` on ~30 of ~5,800 bars | Discarded; `close` is already back-adjusted |
| 2 | Fabricated pre-IPO history | MAZDOCK carries monthly bars from 2017 but listed Oct 2020 | Truncated by calendar-gap rule |
| 3 | Zero-range bars | BAJFINANCE: 1,092 of 5,803 bars have `O=H=L=C` | Flagged, never dropped |
| 4 | Ragged end dates | Files end 2026-06-03 → 2026-06-22 | Trimmed to common 2026-06-05 |
| 5 | Survivorship bias | Universe is *today's* NIFTY 100 | Documented as a limitation |

**The gap rule recovers real NSE listing dates exactly.** Any calendar gap over 10 days is a
vendor artefact — the NSE has never closed that long — so the first genuine bar is the one
after the last such gap:

| Symbol | Rule detected | Actual NSE listing | |
|:--|:--|:--|:--|
| VBL | 2016-11-08 | 2016-11-08 | ✅ |
| DMART | 2017-03-21 | 2017-03-21 | ✅ |
| SBILIFE | 2017-10-03 | 2017-10-03 | ✅ |
| MAZDOCK | 2020-10-12 | 2020-10-12 | ✅ |

Validated against dates sourced *independently of this codebase*, not against its own
output — which is what makes it a real check rather than a tautology.

**Gaps are never forward-filled.** A carried-forward price manufactures a return of exactly
zero, and a run of zeros reads to the model as a stretch of unnatural calm, biasing every
volatility estimate downward. It is the kind of quiet lie that survives most sanity checks.
Missing stays missing.

---

## The causality rule

Every feature must be **causal**: the value at day *t* may use data up to day *t* and never
beyond. Look-ahead bias is the easiest way to build a backtest that looks superb and loses
money live — the model is being graded on an exam whose answers it has already seen.

This is **enforced, not assumed**:

- **Future-perturbation test** — compute a feature, violently rewrite the future portion of
  the data, recompute, and assert every earlier value is bit-for-bit identical.
- **Streaming-equals-batch test** — feeding data one day at a time must reproduce the
  full-history result exactly. That is the live-trading contract.

`tests/test_regime.py` includes a test demonstrating that the *original* full-sample
approach fails this check, so the guard is proven to bite.

> **Adding a feature? Add its causality test.**

---

## Getting started

```bash
pip install numpy pandas pyarrow scikit-learn pytest
```

Price data is **not tracked in git**. Place per-symbol CSVs as:

```
data/
├── raw/                    ← your CSVs go here
│   ├── RELIANCE.csv
│   ├── TCS.csv
│   └── NIFTY100.csv        ← required: supplies the master trading calendar
├── interim/                ← generated: cleaned per-symbol parquet
└── processed/              ← generated: aligned panel + listing mask
```

Expected columns: `date` (`DD-MM-YYYY`), `open`, `high`, `low`, `close`, `volume`.
Optional metadata (`_source`, `_dq_score`, `_gap_filled`) is preserved if present.

```bash
python scripts/run_phase1.py     # load, repair, validate, persist
python -m pytest tests/ -q       # 48 tests
```

Expected output:

```
loading universe ...
  96 usable symbols, 5820 trading days

[PASS] universe_size            96 usable symbols loaded (need >= 80)
[PASS] no_residual_gaps         no calendar gaps remain after truncation
[PASS] ohlc_consistent          high/low bracket open/close everywhere
[PASS] train_window_coverage    82 symbols have history at 2016-01-01
[WARN] flat_bar_burden          BAJFINANCE 1092 zero-range bars
[WARN] symbols_dropped          ENRIN, TATACAP, TMCV (too little history)
```

---

## Layout

```
qbeast_crash/
├── config.py              single source of truth — paths, windows, thresholds
├── data/
│   ├── loader.py          raw CSV → clean frame; one tested rule per defect
│   ├── calendar.py        master calendar + listing mask
│   └── quality.py         the audit, enforced as a blocking gate
├── features/
│   └── slope_accel.py     causal slope, acceleration, trend phase
└── signals/
    └── regime.py          regime detection; HMM drops in behind RegimeDetector

scripts/
├── run_phase1.py          data pipeline entry point
└── build_docx.js          regenerates the documentation
tests/                     48 tests, causality enforced
docs/                      implementation plan + full project documentation
```

Both regressions in `slope_accel.py` collapse to **fixed weight vectors** over an evenly
spaced window, so each feature is a single weighted sum. That is not just fast — it makes
causality *structural*. There is no code path that could look forward, so it cannot be
broken by a later edit.

---

## Roadmap

| | Phase | Status |
|:--|:--|:--|
| 0 | Data audit — five defects identified | ✅ done |
| 1 | Data layer — loader, calendar, quality gate | ✅ done |
| 2 | Features — slope/accel done; precursors + cross-sectional pending | 🔶 part |
| 3 | Isolation Forest + anomaly intensity | ⬜ |
| 4 | Crash labels + lead-time measurement | ⬜ **go/no-go gate** |
| 5 | Signals — per-stock and market-wide | ⬜ |
| 6 | Backtest engine + Indian cost/tax model | ⬜ |
| 7 | Retraining method comparison | ⬜ |
| 8 | Drawdown analysis + per-stock charts | ⬜ |
| 9 | Robustness experiment | ⬜ |
| 10 | HTML dashboard | ⬜ |

**Phase 4 decides whether the project is viable.** It measures how many days of warning the
model actually gives. If the lead time is not there, then signals, backtesting and
dashboards are all scaffolding around a detector that does not work — far better to learn
that from a lead-time histogram in Phase 4 than from an equity curve in Phase 8.

### Retraining methods to be compared

1. Rolling 3-year window
2. Incremental retraining, 1-month steps
3. EWMA, decay factor 0.994
4. **Volatility-purged rolling window** — proposed

The fourth rests on a counterintuitive point. Isolation Forest defines *normal* as whatever
it was trained on, so feeding it crisis data makes crises **less** anomalous — the model
stretches its notion of normal to cover them and detection degrades exactly when it matters.
The fix is therefore not *more* crisis data but *less*: fit the rolling window after
discarding days whose volatility sits in the top decile, so the model learns a clean normal
and anything crisis-like falls far outside it.

Ranking is by **drawdown reduction per unit of turnover** — not raw return, since a scheme
that trades constantly can buy a better drawdown figure at a cost that only surfaces in the
tax bill.

---

## Cost model

The Indian per-trade stack is modelled in full: STT on both legs for delivery, stamp duty on
buys only, DP charges on sells only, exchange transaction charges, SEBI turnover fees, and
GST on the appropriate base. Three refinements are planned:

- **Capital gains tax** at portfolio level on a financial-year basis, date-aware (rates
  changed 23 July 2024).
- **Volatility-scaled slippage** — a flat percentage is the wrong shape here, because this
  strategy trades *only* on crash and rally days, precisely when spreads widen several-fold.
- **Date-aware exchange charges** (changed 1 October 2024).

> An effect most published drawdown-reduction results ignore: every crash-exit converts a
> long-term holding into a short-term one, moving the Indian tax rate from 12.5% to 20%.
> Drawdown reduction quietly pays a **tax penalty** on top of transaction costs.

---

## Notes for contributors

- **Regime detection** — `signals/regime.py` defines a `RegimeDetector` interface. An HMM
  replacement must be causal: refit as you go, or at minimum use *filtered* state
  probabilities `P(state_t | data ≤ t)` rather than *smoothed* ones `P(state_t | all data)`.
  Fitting an HMM on the full sample and then decoding is the same look-ahead bug in a
  smarter costume. Point `tests/test_regime.py` at it before wiring it in.
- **Data quality** — `data/quality.py` blocks the pipeline on ERROR-level failures. If it
  fires, fix the data rather than loosening the check.
- **Documentation** — `docs/*.docx` is generated. Edit `scripts/build_docx.js` and re-run
  `node scripts/build_docx.js`; never hand-edit the docx, or it will drift from the code.
