# QBEAST — Isolation Forest Crash/Rally Detection: Implementation Plan

Strategy in one line: **stay fully invested by default; step out before crashes, step
back in before rallies, trade rarely.** The headline result is drawdown reduction
versus buy-and-hold, with lead time reported in trading days.

---

## 0. Settled decisions

| Question | Decision | Reason |
|---|---|---|
| Crash threshold | Defined upfront, **evaluation only**, never touches training | Deriving it from the model makes the evaluation circular |
| Action on signal | Long-only, exit to cash | Realistic for EOD cash equities; makes drawdown the KPI |
| Lead time | Leading features + windowed events + causal scoring | An IF fed same-day returns is coincident, not predictive |
| Universe | Current Nifty 100, bias documented | Inflates the benchmark as much as the strategy |
| Regime | Fixed causality bug now, HMM later behind one interface | Detector is swappable; nothing downstream should care |
| Backtester | Vectorised first, Nautilus wrapper after | Nautilus scaffolding would dominate the retrain-comparison loop |
| Price series | `close` (already back-adjusted); `adj_close` discarded | It differs on ~30 of ~5,800 bars — not an adjustment factor |

---

## 1. Anomaly intensity — the formula, and why

`sklearn`'s `score_samples()` returns `-2^(-E[h(x)]/c(n))`, where `E[h(x)]` is mean path
length across trees and `c(n)` normalises for sample size. Define the raw score:

```
s(x) = -score_samples(x)          s -> 1 = highly isolated, s ~ 0.5 = ordinary
```

**Do not threshold on `s` directly.** Its scale shifts with the `contamination`
parameter, tree structure, and training window — so a fixed cut means different
things across the four retraining schemes, making them incomparable, which defeats
the entire comparison.

Instead map through the **training-set empirical CDF**:

```
intensity(x) = F_train(s(x)) = (1/N) * #{ s_i in train : s_i <= s(x) }
```

`intensity ∈ [0,1]`, read directly as *"more anomalous than this fraction of training
days."* Three properties we need:

1. **Retrain-invariant** — every scheme sees the same threshold semantics.
2. **Cross-sectionally comparable** — one global threshold for all 100 symbols.
3. **Directly interpretable** — 0.99 means a 1-in-100-day event, and therefore an
   expected alert budget of ~2.5 days/year before confirmation.

Bands: **High ≥ 0.99**, **Moderate 0.95–0.99**, **Low 0.90–0.95**.

**The critical caveat: Isolation Forest is direction-blind.** It flags *unusual*, not
*unusually bad* — a violent rally scores as high as a crash. Intensity alone can never
produce a buy or sell. Direction comes entirely from `slope_z`, and
building-vs-exhausting from `accel_z`. That is the whole reason the slope/acceleration
block exists.

---

## 2. Features (all strictly causal)

**Per-stock precursors** — deliberately excludes same-day return, which is what makes
the detector leading rather than coincident:

- vol ratio 5d/60d, vol-of-vol
- downside/upside semi-deviation asymmetry
- volume z-score, volume/price divergence
- high-low range expansion, overnight gap frequency
- Amihud illiquidity
- `slope_z`, `accel_z` from `features/slope_accel.py` *(built)*

**Cross-sectional** (drives the market-wide signal):

- breadth: % of universe in `AcceleratingDecline`
- median `slope_z` across the universe
- average pairwise correlation (rises before systemic events)
- return dispersion (collapses as correlation spikes)
- % below 20/50 DMA

---

## 3. Signal logic

### Per stock

```
HIGH intensity (>= 0.99)
  AND slope_z <= -1.0
  AND accel_z <  0            (phase == AcceleratingDecline)
  -> STRONG SELL, exit at next open

MODERATE/LOW intensity (0.90 - 0.99)
  -> WATCH for 3-5 days. Resolve by which phase persists:
       AcceleratingDecline  -> exit
       DeceleratingDecline  -> hold; candidate re-entry
       reverts to Flat      -> discard, no trade

RE-ENTRY (rally)
  intensity >= 0.95
  AND slope_z >= +1.0
  AND accel_z >  0            (phase == AcceleratingAdvance)
  -> BUY, ranked by signal strength, top-N filled
```

### Market-wide

Calibrated from the measured universe distribution (breadth p90 = 51.6%, p99 = 78.5%):

```
breadth >= 75%  AND  median slope_z <= -1.5   -> SYSTEMIC SELL, de-risk whole book
```

Both conditions are required. Breadth alone fires on COVID *and* on mild pullbacks —
only median `slope_z` separates them (−2.11 vs ≈−0.8). Expected firing rate ≈ 1% of
days.

### Whipsaw suppression — a hard constraint, not a nicety

- signal persistence: 2 consecutive confirming days before acting
- minimum holding period after entry
- re-entry cooldown after exit
- **Budget: < 10 round trips per stock over 5.5 years.** More than that is a failure
  mode regardless of gross return.

---

## 4. Crash/rally labels (evaluation only, in `labels/`, never imported by `model/`)

Events are **windows**, not days — a crash is a process, and a detector that fires on
the first wobble should be credited, not penalised.

- **Stock crash**: forward 5-day max drawdown ≤ −10%, or single-day ≤ −6%
- **Stock rally**: forward 5-day return ≥ +10%
- **Market crash**: NIFTY100 5-day drawdown ≤ −6%, or single-day ≤ −3%
- **Onset** = first day of the window. Overlapping windows merge into one event.

Thresholds get a sensitivity sweep (±2%) so no headline number rests on one arbitrary cut.

---

## 5. Lead-time measurement

`lead = onset − first confirmed alert`, in trading days, reported as a **distribution**:

| bucket | reading |
|---|---|
| ≥ 10 d | very early — risks sitting out too long |
| 5–9 d | early |
| **2–4 d** | **target band** |
| 0–1 d | coincident; still usable for T+1 exit |
| missed | no alert before onset |

Paired with a **hit-rate vs. false-alarms-per-year** curve. False alarms per year is
the number that decides tradeability — an alert on every wobble is worthless at any
lead time.

---

## 6. Retraining comparison

Train Jan 2016 – Dec 2020, backtest Jan 2021 – Jun 2026.

1. Rolling 3-year window
2. Incremental, 1-month steps
3. EWMA, decay 0.994
4. **Volatility-purged rolling window** *(proposed)*

**Why #4.** Isolation Forest defines "normal" as whatever it was trained on. Feed it
crisis bars and crises become *less* anomalous — the boundary stretches to cover them
and detection degrades exactly when it matters. So the fix is not more crisis data but
**less**: fit the rolling window after dropping bars whose trailing volatility sits in
the top decile. The model learns a clean normal manifold, and anything crisis-like
falls far outside it. This should give the deepest drawdown reduction of the four, and
it is a defensible paper contribution.

**Benchmark protocol** — identical features, labels, costs, and signal thresholds
across all four; only the fit schedule varies. Ranked on:

- max drawdown, and drawdown vs. buy-and-hold *(primary)*
- lead-time distribution, hit rate, false alarms/year
- CAGR, Sharpe, Sortino, Calmar
- round trips and total cost drag
- stability of results across a walk-forward split

Selection is by **drawdown reduction per unit of turnover**, not raw return.

---

## 7. Capital, sizing, costs

- Capital ₹10,00,000, long-only, no leverage
- Max 15–20 concurrent positions; equal-weight, or inverse-vol as a variant
- Top-N by signal strength on re-entry, never first-come-first-served
- Whole shares only; residual stays in cash

**Costs.** `costs.py` is integrated as-is — the per-leg stack (STT both legs on
delivery, stamp duty buy-only, DP sell-only, GST base) is correct. Three additions:

1. **Capital gains tax** at portfolio level, financial-year basis. Rates change
   mid-backtest on 23 Jul 2024 (STCG 15%→20%, LTCG 10%→12.5%, exemption ₹1L→₹1.25L),
   so the model must be date-aware. **Verify current rates before hard-coding.**
2. **Volatility-scaled slippage.** Flat slippage is the wrong shape here: this strategy
   trades *only* on crash and rally days, precisely when spreads widen 3–5×. Scale by
   that day's ATR.
3. **Date-aware exchange charges** (NSE changed 1 Oct 2024).

**A result worth its own paper section:** every crash-exit converts a long-term holding
into a short-term one — 12.5% LTCG becomes 20% STCG. Drawdown reduction therefore
carries a *tax* penalty on top of transaction costs. Published results almost
universally ignore this; quantifying it is a genuine contribution.

---

## 8. Drawdown

```
equity_t   = mark-to-market portfolio value, net of all costs
peak_t     = max(equity_0..t)                    running maximum
dd_t       = equity_t / peak_t - 1               <= 0
max_dd     = min(dd_t)
```

Reported alongside **time under water** (bars below prior peak) and **MTTR** (bars to
recover). Those two are reliability metrics, which is what lets the CPS framing be
literal rather than decorative.

Per stock and at portfolio level, always as **strategy vs. buy-and-hold on the same
symbol over the same window** — buy-and-hold is the benchmark that matters, since the
strategy *is* buy-and-hold plus an exit rule.

---

## 9. Visualisation

**Per-stock panel, 2021–2026** — price with sell markers (▼ at signal, ▲ at re-entry),
shaded held-vs-cash bands, labelled crash windows with the measured lead time annotated
per event; `intensity` and `slope_z` in subplots; strategy vs. buy-and-hold drawdown
underneath.

**Dashboard, 4–5 tabs, built last:** Overview · Per-stock explorer · Retraining
comparison · Lead-time analysis · Costs & tax drag.

---

## 10. Phases

| # | Phase | Output | State |
|---|---|---|---|
| 0 | Data audit | 5 defects found | **done** |
| 1 | Data layer | loader, calendar, quality gate | next |
| 2 | Features | slope/accel **done**; precursors + cross-sectional | partial |
| 3 | IF + intensity | intensity series per stock | |
| 4 | Labels + lead time | **first real result** | |
| 5 | Signals | per-stock + market-wide | |
| 6 | Backtest + costs | equity curves | |
| 7 | Retrain comparison | the 4-way benchmark | |
| 8 | Drawdown + plots | per-stock panels | |
| 9 | Cyber experiment | poisoning robustness | |
| 10 | Dashboard | HTML tabs | last |

Phase 4 is the go/no-go gate. If lead time is not there, signals and backtest are wasted
effort — better to learn it from a lead-time histogram than from an equity curve.

---

## 11. Phase 1 scope (next)

1. Parse `DD-MM-YYYY`; use `close`; document dropping `adj_close`
2. **First-valid-bar truncation** — auto-detect the last monthly-spaced gap per symbol
   (kills fabricated pre-IPO history in MAZDOCK, DMART, SBILIFE, VBL)
3. Flat-bar (O=H=L=C) masking — flag, exclude from range features, keep the row
4. Common calendar from NIFTY100; ragged end-dates cut at 2026-06-05
5. Carry `_source`, `_dq_score`, `_gap_filled` through — needed for the cyber angle
6. Quality gate re-running the Phase 0 audit, failing the build on regression
7. Extend the causality test suite to every new feature

---

## Open items

- `qbeast_slope_accel.py` — superseded by `features/slope_accel.py`; confirm nothing
  else imports it
- `config.py` — still needed to verify cost rates
- Confirm `close` is fully back-adjusted with whoever built the data pipeline
- Cyber angle: deferred, no rework cost as long as Phase 6 persists per-retrain models
  and raw feature matrices
