# QBEAST Crash Detection — Project Context

Everything needed to resume this project without re-deriving it. Written to be
read by a person picking the work up cold, or by an assistant starting a fresh
session.

**Status:** phases 1–8 and 10 complete · 190 tests passing · phase 9 (robustness)
not started.

> **Which universe produced which number.** Results in this document come from
> two different runs: the 96-symbol full universe (phases 6–8, severe-only
> logic) and a 10-symbol run (after the two-tier change). Every table says
> which. The artefacts currently in `reports/` are from the **10-symbol** run —
> re-run `--all` before quoting full-universe figures.

---

## 1. What this is

An Isolation Forest anomaly detector on NIFTY 100 end-of-day bars, used to
reduce **drawdown** rather than to increase return.

The strategy holds every stock all the time and only sells when it sees an
unusual day *and* the price is falling with the fall accelerating. It buys back
afterwards. The goal is to fall less in a crash, not to earn more.

| | |
|---|---|
| Universe | Up to 96 usable NIFTY 100 symbols (99 CSVs; 3 too short) |
| Training | Jan 2016 – Dec 2020 |
| Backtest | Jan 2021 – Jun 2026 (1,344 sessions) |
| Capital | ₹10,00,000 |
| Direction | Long only; sell to cash, buy back |

**Author context:** MTech Cyber-Physical Systems, IIT Jodhpur. Intended as a
startup component, a research paper, and a portfolio project.

---

## 2. The single most important finding

**The model has no early-warning ability.** Measured against a random signal
firing at exactly the same rate:

| rule | recall | random | skill |
|---|---|---|---|
| intensity ≥ 0.99 + falling | 2.2% | 2.5% | **0.88×** |
| intensity ≥ 0.95 + falling | 8.0% | 14.5% | 0.55× |

All at or below 1.00×. **The requirement to predict crashes 2–3 days ahead, in
the sense of forecasting from a calm market, is not met.**

But the same signal is enormously informative about the *immediate* next days:

| horizon | base rate | given a signal | lift |
|---|---|---|---|
| 1 day | 0.06% | 6.92% | **110×** |
| 3 days | 0.57% | 20.00% | 35× |
| 10 days | 4.82% | 26.15% | 5× |

Lift decays sharply with horizon — the signature of a **coincident** detector.
Median same-day return on signal days is −1.14% versus +0.02% overall: it fires
*as a decline begins*, not before.

**Honest framing: fast reaction, not prediction.** "Unsupervised anomaly
detection reduces drawdown through rapid exit" is defensible. "Predicts crashes
three days ahead" is not.

### Why it can't be early

Every feature — ours and the earlier project's — is built from **the stock's own
recent history**. By the time a stock's own data looks unusual, its own move has
largely happened. Measured: whenever the model fires, **~70% of the decline is
already done**, and that fraction is stable across every threshold from 0.99 to
0.80. It is a property of the features, not the cutoff.

The one promising lead: **peer stress is 3.4× more predictive** than the stock's
own return (correlation −0.1225 vs −0.0358 with a forward 5-day drop), and beats
it for 73 of 93 stocks. Adding three peer features raised capture from 37% → 47%
on 10 symbols, but on only 14 signals. **Not yet validated on the full universe —
this is the top open item.**

---

## 3. What actually works

The **two-tier signal logic** (added late, and it mattered):

- **Severe** anomaly (intensity ≥ 0.99) → check slope and acceleration → sell
  next trading day. No waiting; the edge decays within a day.
- **Mild** anomaly (0.95–0.99) → open a **5-day watch**. Each day compare the
  1-day move against the 5-day slope and check the regime. If the fall is
  accelerating and the regime confirms, sell immediately. If the stock turns up,
  stand down.

Measured on 10 symbols, trained 2016–2019 so COVID is genuinely unseen:

| | severe-only | **two-tier** |
|---|---|---|
| 2020 Feb–Apr crash | +7.9pp saved | **+18.0pp** (−2.8% vs −15.3%) |
| 2020 full year | +7.7pp | **+17.7pp** (+41.1% vs +31.0%) |
| 2021–26 (no crash) | +0.3pp | +0.9pp |

**30 of 36 exits come from the mild path.** It is also what lets all 10 symbols
trade where severe-only reached 3.

---

## 4. What does not work, stated plainly

1. **No early warning** (§2).
2. **Duration is unchanged.** Median sessions under water are identical to
   buy-and-hold. The strategy reduces how *far* you fall, not how *long* you
   stay down.
3. **Per-stock savings do not aggregate.** ADANIENT saved 17.5pp and ITC 12.0pp,
   yet the portfolio improved 1.0pp — stocks trough at different times and
   diversification already absorbs most of it. **It is a single-stock risk tool
   that aggregates weakly, not a portfolio hedge.**
4. **The backtest window contains no sharp crash.** The worst drawdown of
   2021–26 was a 154-day slow bleed at 16.5% annualised volatility with one day
   beyond 3%. Nothing there for an anomaly detector to fire on.
5. **The stress test is n=1.** 2020 is one event. Nothing generalises from it.

---

## 5. The recurring pattern — eight quiet defects

**None of these raised an error, a warning, or an implausible number.** Each
produced output that looked entirely reasonable and was wrong. This is the most
transferable lesson in the project and the spine of the documentation.

| # | Defect | Caught by |
|---|---|---|
| 1 | Six data defects, incl. a demerger passing every structural check | Auditing raw data before writing model code |
| 2 | Look-ahead in the inherited regime detector — 9.9% of labels changed | Future-perturbation test |
| 3 | A crash muting its own signal via a contemporaneous volatility denominator (−1.67σ vs −3.01σ) | Test on real COVID data |
| 4 | Double lag in the state machine — every trade delayed a second day | Test asserting T+1 execution |
| 5 | Geometric mean masquerading as a portfolio — understated CAGR by 4.75pp | Code review |
| 6 | EWMA resampling with replacement — 2.63× duplication made recent rows look *less* anomalous | Recheck before documenting |
| 7 | Scatter-chart axis label inverted — said the opposite of the truth | Looking at the picture |
| 8 | `EXIT_WATCH` dropped from eight counts testing `action == "EXIT"` | Two numbers in one run disagreeing |

Plus: **two-thirds of symbols could not fire a signal at all** — found by
measuring *coverage* separately from *performance*.

---

## 6. The six data defects

Found in Phase 0, before any model code.

| # | Defect | Evidence | Handling |
|---|---|---|---|
| 1 | `adj_close` is not adjusted | Differs from `close` on ~30 of ~5,800 bars | Discarded; `close` is already back-adjusted |
| 2 | Fabricated pre-IPO history | MAZDOCK has monthly bars from 2017, listed Oct 2020 | Truncate at calendar gap > 10 days |
| 3 | Zero-range bars | BAJFINANCE: 1,092 of 5,803 `O=H=L=C` | Flag, never drop |
| 4 | Ragged end dates | 2026-06-03 → 2026-06-22 | Trim to 2026-06-05 |
| 5 | Survivorship bias | Universe is *today's* NIFTY 100 | Documented, not fixed |
| 6 | Unadjusted corporate actions | CGPOWER 155.30 → 53.05 (demerger) | Truncate; real crashes preserved |

**The gap rule recovers real NSE listing dates exactly** — VBL 2016-11-08, DMART
2017-03-21, SBILIFE 2017-10-03, MAZDOCK 2020-10-12. Validated against dates
sourced independently of the code, which is what makes it a real check.

**Gaps are never forward-filled.** A carried-forward price manufactures a return
of exactly zero, which reads as unnatural calm and biases every volatility
estimate downward.

---

## 7. Architecture

```
qbeast_crash/
├── config.py      single source of truth — every threshold, with the measurement
├── data.py        Phase 1: load, clean, calendar, quality gate
├── features.py    Phase 2: slope/accel, precursors, cross-sectional
├── model.py       Phase 3: Isolation Forest + intensity
├── labels.py      Phase 4: crash ground truth, lead-time measurement
├── signals.py     Phase 5: two-tier state machine
├── backtest.py    Phase 6: NSE costs + capital gains tax
├── retrain.py     Phase 7: walk-forward, four schemes
├── plots.py       Phase 8: drawdown analysis and charts
└── regime.py      regime detection (HMM drops in behind RegimeDetector)

scripts/
├── run_all_phases.py    --symbols A,B,C  (required)  |  --all
└── build_dashboard.py   → reports/dashboard.html
tests/                   190 tests, causality enforced
docs/phases/             one docx per phase
```

### Running

```bash
python3 scripts/run_all_phases.py --symbols RELIANCE,TCS,ITC
python3 scripts/run_all_phases.py --all
python3 scripts/build_dashboard.py
python3 -m pytest tests/ -q
```

**There is no default universe.** Running bare lists the available symbols and
exits — picking one silently would mean every result depended on a choice the
user never made.

---

## 8. Key design decisions

**Anomaly intensity = percentile of the score against the *training*
distribution.** The raw sklearn score shifts scale with training window and tree
structure, so a fixed cut would mean four different things under four retraining
schemes. The percentile makes the Phase 7 comparison valid at all.

**`contamination` does nothing.** Measured across 0.001–0.2, raw scores are
bit-identical. It only sets `offset_` for `predict()`, which this pipeline never
calls. `max_samples` also barely matters (2.18× → 2.29× across a 64× range).

**Isolation Forest is direction-blind.** A violent rally scores as high as a
crash. Intensity alone can never produce a buy or sell — direction comes from
slope, building-vs-exhausting from acceleration.

**Volatility denominator is lagged 20 days.** A crash inflates its own
denominator; lagging means it describes the calm the move is departing *from*.

**Pooled model, not per-symbol.** One model across all symbols keeps scores
comparable. Per-symbol models would learn each stock's own quiet periods as
normal. (But see the coverage problem — pooled intensity means a calm stock can
never reach the top of a distribution set by volatile ones;
`intensity_per_symbol()` exists as the alternative.)

**Sleeves, not daily rebalancing.** Daily rebalancing would trade every symbol
every day and cost more than the strategy could save.

**Every feature must be causal**, enforced by future-perturbation tests: rewrite
the future, assert every earlier value is bit-identical.

---

## 9. Measured results

### Retraining schemes

> ⚠️ **These numbers predate the two-tier signal logic (§3).** They were measured
> on all 96 symbols with the severe-only rule, which fired on ~3× fewer days.
> **The comparison needs re-running** — a ~13-minute job — before it is quoted
> anywhere. The *conclusion* (schemes are indistinguishable) is expected to
> hold, since it rests on the spread being smaller than run-to-run noise, but
> the individual figures will move.

*96 symbols, walk-forward, monthly refits, severe-only logic:*

| scheme | CAGR | maxDD | vs B&H | trades/sym/yr |
|---|---|---|---|---|
| buy & hold | 25.64% | −20.0% | — | 0.00 |
| rolling | 25.56% | −19.6% | +0.4pp | 0.35 |
| incremental | 25.82% | −19.8% | +0.2pp | 0.28 |
| ewma | 25.78% | −19.6% | +0.4pp | 0.32 |
| vol_purged | 25.40% | −19.6% | +0.3pp | 0.64 |

**Spread is 0.2pp — indistinguishable.** The scheme choice is not a lever.
`vol_purged` is retired: twice the turnover for no benefit.

**But retraining matters enormously for coverage**: static fit trades 44/96
symbols, walk-forward 79/96. A static fit on 2016–2020 includes COVID, setting a
bar calm years never reach.

**EWMA decay 0.994 is vindicated** — 3.1 standard deviations better than 0.997
across four seeds, and the lowest variance. Its effective memory is 0.66 years,
not 3.

### Costs and tax

*96 symbols, severe-only logic — same caveat as above.*

Cost drag **0.088% of capital per year** — a non-issue at this turnover.

**The tax hypothesis was wrong.** Only 23.3% of sales are short-term; the
strategy holds for years between trades. And comparing realised tax against
buy-and-hold is meaningless because buy-and-hold never sells. Liquidating both:
strategy ₹273,473 vs buy-and-hold ₹317,958 — the strategy pays *less*, but mainly
because it earned less.

---

## 10. Open questions

1. **Do peer features hold on the full universe?** +11pp capture on 14 signals is
   the only promising lead. Top priority.
2. **Confirm `close` is fully back-adjusted** with the data pipeline team. The
   RELIANCE 2017-bonus check says yes; if wrong, every return is wrong.
3. **Verify the cost and tax rates** against a live contract note. Built from
   documented Indian rates, never confirmed; they change each budget.
4. **Extend training back to 2008** for a second crash to stress-test against.
   Everything currently rests on 2020 alone.
5. **Phase 9 (robustness) not started.** The proposed angle is adversarial
   poisoning: inject crafted bars and measure how small an attack budget blinds
   the detector. ~200 lines, and it's the cyber-physical contribution.
6. **HMM regime detection** is being built by another team. `RegimeDetector` in
   `regime.py` is the interface. **Warn them:** fitting an HMM on the full sample
   and decoding is the same look-ahead bug — they need *filtered* state
   probabilities, not *smoothed*.

---

## 11. The earlier project (`QBEAST_AI copy/Models/isolation_forest`)

Reviewed on request. **It was not performing better.** Its own log reports
`Anomaly precision: 0.4%, recall: 0.0%`, `calibration quality: poor`,
`Symbols processed: 2`, `Portfolio trades: 4`. The headline 33.81% return is 4
trades on 2 stocks measured against a 15% *target* rather than buy-and-hold —
and Indian large caps returned far more than 15% over that window.

Issues found in it:

- All 9 features are coincident or trailing (`daily_return`, `rsi_14`,
  `down_days_5d`, `vol_pressure`…) — same root cause as ours
- `_penalised_spline_locus` fits a spline over the **entire series** —
  look-ahead, and it feeds regime detection
- `config.py`: `dp_charge_flat = 15.0` double-charges GST (should be 13.00 for
  their formula); `stt_pct` comment says "sell side" when delivery charges both
  legs; `anomaly_intensity_threshold = -0.60` is a raw-score cut that invalidates
  their own 3-scheme comparison; `anomaly_actual_threshold = 0.02` labels **60.5%
  of all days** as anomalies
- A corrected `config_fixed.py` was written alongside it (not overwriting)

---

## 12. Glossary

| term | meaning |
|---|---|
| **pp** | Percentage points. −20% → −18% is 2pp better, not 2% |
| **Anomaly score** | Raw Isolation Forest output; higher = more unusual |
| **Anomaly intensity** | That score as a 0–1 rank against training days |
| **Slope** | Price velocity in the stock's own daily sigmas |
| **Acceleration** | Whether the move is speeding up or slowing |
| **Drawdown** | How far below the peak the portfolio sits |
| **Precision** | Of alarms raised, how many were crashes (**high here**) |
| **Recall** | Of crashes, how many were caught (**low here**) |
| **Look-ahead bias** | Using information unavailable at the time |
| **Causal** | Value at day *t* uses only data up to day *t* |

---

## 13. Working principles that earned their place

1. **Bad data doesn't announce itself.** All six defects produced normal-looking
   output. None raised an error.
2. **Unsupervised models have no immune system.** A supervised model pushes back
   against a wrong label; Isolation Forest absorbs it silently.
3. **Self-normalisation hides what you're measuring.** Bit us three times — a
   crash inflating its own volatility denominator, and Bollinger Bands where the
   band *is* the normaliser.
4. **Validate against reality, not against yourself.** The gap rule was checked
   against real NSE listing dates. A test comparing code to code only proves the
   code is consistent.
5. **Always ask what a dumb baseline scores.** 2.2% recall meant nothing until
   random scored 2.5%.
6. **Measure, then claim.** Nearly every confident guess here was wrong —
   `contamination` did nothing, `max_samples` barely mattered, volatility purging
   helped then hurt, Bollinger added +0.0000, the tax hypothesis reversed sign.

---

*Generated 27 July 2026. Regenerate the dashboard with
`python3 scripts/build_dashboard.py`; regenerate phase docs with
`node scripts/build_docs.js`.*
