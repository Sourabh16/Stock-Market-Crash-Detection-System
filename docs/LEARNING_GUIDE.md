# Learning Guide

A hands-on guide to running, testing, and understanding this project — written for
someone learning data science by building something real.

---

## 1. Setup (once)

Open the **Terminal** tab at the bottom of PyCharm. Everything below runs there.

```bash
cd ~/crashDetection
```

Check Python and install what's needed:

```bash
python3 --version
```

```bash
python3 -m pip install numpy pandas pyarrow scikit-learn pytest
```

**PyCharm setup that will save you pain:**

- **Settings → Project → Python Interpreter** — make sure it points at the same Python
  your terminal uses (`which python3` tells you the path).
- **Mark `qbeast_crash` as a source root** — right-click the folder → *Mark Directory as* →
  *Sources Root*. Without this, PyCharm underlines every `from qbeast_crash...` import in
  red even though the code runs fine.
- **Enable pytest** — Settings → Tools → Python Integrated Tools → *Default test runner:
  pytest*. Now you get a green ▶ next to every test function.

---

## 2. Running the pipeline

One command runs everything:

```bash
python3 scripts/run_all_phases.py
```

That takes about 15 seconds and rebuilds every artifact from `data/raw/`. Run one phase
at a time while you're learning:

```bash
python3 scripts/run_all_phases.py --phase 1
```

```bash
python3 scripts/run_all_phases.py --phase 4
```

```bash
python3 scripts/run_all_phases.py --list
```

**What each phase does:**

| Phase | Reads | Writes | Question it answers |
|:--|:--|:--|:--|
| 1 | `data/raw/*.csv` | `close_panel.parquet` | Is the data trustworthy? |
| 2 | cleaned frames | `features.parquet` | What does stress look like? |
| 3 | features | `intensity.parquet` | Which days are unusual? |
| 4 | intensity + prices | `reports/phase4_*.csv` | How early do we know? |

Phases pass results in memory, so running `--phase 4` replays 1–3 first. That's
deliberate — it stops a partial run from quietly using stale files.

---

## 3. Running the tests

```bash
python3 -m pytest tests/ -q
```

You should see `103 passed`. To see each test name:

```bash
python3 -m pytest tests/ -v
```

Run one file, or one test:

```bash
python3 -m pytest tests/test_labels.py -v
```

```bash
python3 -m pytest tests/test_labels.py::test_lead_zero_still_counts_as_a_warning -v
```

**In PyCharm:** click the green ▶ in the gutter next to any test. Right-click a test file →
*Run pytest in test_labels.py*. Failures show a diff you can click through.

### The habit worth building

When a test fails, **read the assertion before changing anything**. Twice in this project a
failing test was right and the code was wrong:

- A volatility denominator was letting a crash mute its own signal (−1.67σ vs −3.01σ)
- A ratio was returning `NaN` exactly when the signal was strongest

Both looked like test bugs at first glance. Neither was.

---

## 4. Exploring the data yourself

This is where the learning actually happens. Start Python in the project directory:

```bash
python3
```

Then:

```python
import pandas as pd
from qbeast_crash.data import load_symbol

frame, report = load_symbol("RELIANCE")
frame.tail(10)
report                       # what the loader had to repair
```

Look at the feature block for one stock:

```python
from qbeast_crash.features import compute_precursors
feats = compute_precursors(frame)
feats.loc["2020-03-01":"2020-03-31"]      # the COVID crash
```

Load what the pipeline produced:

```python
panel = pd.read_parquet("data/processed/close_panel.parquet")
market = pd.read_parquet("data/processed/market_features.parquet")
intensity = pd.read_parquet("data/processed/intensity.parquet")

market.loc["2020-03-01":"2020-03-31"]
```

**PyCharm tip:** use a **Jupyter notebook** (`File → New → Jupyter Notebook`) for this kind
of poking around. You get tables you can scroll and plots inline. Keep exploration in
notebooks; keep anything that must be *correct* in `qbeast_crash/` with a test.

### Questions worth answering yourself

Try these — each one teaches something:

1. Which stock had the most anomalous day in 2023? *(sort `intensity` and look)*
2. What did `avg_corr` do during COVID versus 2019?
3. Pick a stock, plot its price with High-band days marked. Do they look sensible?
4. How many `AcceleratingDecline` days does RELIANCE have per year?

---

## 5. Concepts, in the order they'll make sense

### Foundations (do these first)

**pandas** — the whole project is pandas. Specifically: `Series` vs `DataFrame`,
`.rolling()`, `.shift()`, `MultiIndex`, `.groupby()`. If you understand `.rolling(20).mean()`
and *why* `.shift(1)` matters for avoiding look-ahead, you understand most of this codebase.

**Log returns** — `log(P_t / P_{t-1})`. Used everywhere here because they're comparable
across price levels and immune to stock splits. Understand why we don't use raw price
differences.

**Look-ahead bias** — the single most important idea in this project. Using information you
wouldn't have had at the time. It produces backtests that look brilliant and lose money.
Read `tests/test_features.py::test_future_perturbation_does_not_change_past` — it's the
whole idea in ten lines.

### Modelling

**Supervised vs unsupervised** — supervised learns from labelled examples; unsupervised
finds structure without labels. Isolation Forest is unsupervised, which is why it has no
"immune system" against bad data (see §7).

**Isolation Forest** — read `qbeast_crash/model.py`, the module docstring. The mechanism is
genuinely elegant and takes five minutes to understand.

**Precision and recall** — the pair that made Phase 4 interpretable:
- *Precision*: of the times we raised the alarm, how often was there a crash? (**high here**)
- *Recall*: of all the crashes, how many did we catch? (**low here**)

Both are true simultaneously. A rare, accurate alarm has high precision and low recall.
Confusing them is the most common mistake in applied ML.

**Baselines** — Phase 4's central lesson. A recall of 2.2% sounds bad, but is it? Only a
comparison tells you: a random signal firing at the same rate achieved 2.5%. **Always ask
what a stupid baseline would score.** Without it, no number means anything.

### Finance

**Drawdown** — how far below your peak you are. The number that decides whether a strategy
is livable. **Volatility clustering** — turbulent days cluster; this is what makes
volatility features predictive at all. **Survivorship bias** — studying only today's
winners.

---

## 6. Reading the codebase

Read in this order:

```
qbeast_crash/config.py      start here — every threshold, with the reasoning
qbeast_crash/data.py        Phase 1 — the six data defects
qbeast_crash/features.py    Phase 2 — three layers of features
qbeast_crash/model.py       Phase 3 — Isolation Forest + intensity
qbeast_crash/labels.py      Phase 4 — ground truth and lead time
```

Every module opens with a docstring saying **what it does, why, and how**. The comments
explain *why* a choice was made, not what the line does — because "why" is the part you
can't recover by reading code.

The measured numbers live in the comments deliberately. When `config.py` says purging
scored 3.48× against 2.22×, that's a real experiment, and it's there so nobody (including
us) has to re-litigate the decision from memory.

---

## 7. The five ideas this project taught that generalise

These are worth more than the code:

**1. Bad data doesn't announce itself.** All six data defects produced output that looked
completely normal — valid dates, positive prices, sensible returns. Not one would have
raised an error. A demerger bar has `high > close > low`; it passes every structural check
and simply describes a different company.

**2. Unsupervised models have no immune system.** Show a supervised model a wrong label and
its error rises — training pushes back. Isolation Forest just absorbs it. One −66% day in
training would drag the anomaly boundary out and make real crashes look ordinary forever
after.

**3. Self-normalisation hides the thing you're measuring.** This bit us three times. A crash
inflates its own volatility denominator, so dividing by recent volatility makes the worst
events look mild. Bollinger Bands have the same flaw and it can't be fixed — the band *is*
the normaliser.

**4. Validate against reality, not against yourself.** The gap rule was checked against real
NSE listing dates from an external source, not against the code's own output. A test
comparing code to code only proves the code is consistent — which it always is.

**5. Measure, then claim.** Almost every confident guess in this project was wrong:
`contamination` did nothing, `max_samples` barely mattered, volatility purging helped in one
setting and hurt in another, Bollinger Bands added +0.0000. Each was corrected only because
it was measured.

---

## 8. Useful commands

```bash
python3 -m pytest tests/ -q                    # all tests, quiet
```

```bash
python3 -m pytest tests/ -v -k "causal"        # only tests matching "causal"
```

```bash
python3 -m pytest tests/ -x                    # stop at first failure
```

```bash
python3 scripts/run_all_phases.py --from 3     # phase 3 onward
```

```bash
node scripts/build_docs.js                     # regenerate the docx files
```

```bash
git log --oneline                              # what changed, and why
```

```bash
git show HEAD                                  # the full reasoning of the last change
```

That last one is worth the habit. The commit messages in this repo record *why* each
decision was made and what was measured — including the ones that turned out wrong.

---

## 9. Where to go next as a data scientist

**Strengthen first:** pandas fluency (rolling windows, MultiIndex, groupby), then
matplotlib, then scikit-learn's API pattern (`fit` / `predict` / `score`) — every model in
sklearn works the same way, so learning one teaches you all of them.

**Then:** cross-validation and why time-series needs *walk-forward* rather than random
splits (random splits leak the future — the same bug as look-ahead bias, wearing a
different hat).

**The habit that matters most:** for every result, ask *"what would a dumb baseline
score?"* Phase 4 is the whole lesson in one number — 2.2% recall meant nothing until random
scored 2.5%.
