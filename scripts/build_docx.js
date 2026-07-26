/*
 * build_docx.js
 * -------------
 * Generates docs/QBEAST_Crash_Detection_Documentation.docx
 *
 * This is the living project document. It is REGENERATED, not hand-edited, so
 * that it can never drift out of step with the code. Add a new section per
 * phase and re-run:
 *
 *     node scripts/build_docx.js
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  TableOfContents, PageBreak, LevelFormat, ExternalHyperlink,
} = require("docx");

const CONTENT_W = 9000;
const ACCENT = "1F4E79";
const MUTED = "595959";
const CODE_BG = "F2F2F2";
const OK = "1E7B34";
const BAD = "B22222";

// ---------------------------------------------------------------- helpers
const H1 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 } });
const H2 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120 } });
const H3 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_3, spacing: { before: 220, after: 100 } });

const P = (text, opts = {}) =>
  new Paragraph({
    spacing: { after: opts.after ?? 120, line: 300 },
    alignment: opts.align,
    children: [new TextRun({ text, italics: opts.italics, bold: opts.bold, color: opts.color, size: opts.size })],
  });

/** Paragraph from an array of {text, bold, italics, code} fragments. */
const RichP = (frags, opts = {}) =>
  new Paragraph({
    spacing: { after: opts.after ?? 120, line: 300 },
    children: frags.map((f) =>
      new TextRun({
        text: f.text,
        bold: f.bold,
        italics: f.italics,
        color: f.color,
        font: f.code ? "Consolas" : undefined,
        shading: f.code ? { type: ShadingType.CLEAR, fill: CODE_BG } : undefined,
      })
    ),
  });

const Bullet = (text, level = 0) =>
  new Paragraph({ text, numbering: { reference: "bullets", level }, spacing: { after: 80, line: 290 } });

const Num = (text, level = 0) =>
  new Paragraph({ text, numbering: { reference: "numbers", level }, spacing: { after: 80, line: 290 } });

const Code = (lines) =>
  lines.map((l, i) =>
    new Paragraph({
      spacing: { after: i === lines.length - 1 ? 160 : 0, before: i === 0 ? 60 : 0 },
      shading: { type: ShadingType.CLEAR, fill: CODE_BG },
      indent: { left: 240 },
      children: [new TextRun({ text: l || " ", font: "Consolas", size: 18 })],
    })
  );

const Callout = (title, body) =>
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: ACCENT },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: ACCENT },
      left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT },
      right: { style: BorderStyle.SINGLE, size: 2, color: ACCENT },
      insideHorizontal: { style: BorderStyle.NONE },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            width: { size: CONTENT_W, type: WidthType.DXA },
            shading: { type: ShadingType.CLEAR, fill: "F7FAFD" },
            margins: { top: 140, bottom: 140, left: 200, right: 200 },
            children: [
              new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: title, bold: true, color: ACCENT })] }),
              ...body.map((b) => P(b, { after: 40 })),
            ],
          }),
        ],
      }),
    ],
  });

/** Table with a header row. widths must sum to CONTENT_W. */
const Tbl = (headers, rows, widths) =>
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) =>
          new TableCell({
            width: { size: widths[i], type: WidthType.DXA },
            shading: { type: ShadingType.CLEAR, fill: ACCENT },
            margins: { top: 80, bottom: 80, left: 120, right: 120 },
            children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, color: "FFFFFF", size: 19 })] })],
          })
        ),
      }),
      ...rows.map((r, ri) =>
        new TableRow({
          children: r.map((c, i) =>
            new TableCell({
              width: { size: widths[i], type: WidthType.DXA },
              shading: { type: ShadingType.CLEAR, fill: ri % 2 ? "FFFFFF" : "F5F8FB" },
              margins: { top: 70, bottom: 70, left: 120, right: 120 },
              children: [new Paragraph({ children: [new TextRun({ text: String(c), size: 19 })] })],
            })
          ),
        })
      ),
    ],
  });

// ---------------------------------------------------------------- content
const children = [];

// --- cover ---
children.push(
  new Paragraph({ spacing: { before: 2600, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "QBEAST", bold: true, size: 72, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Stock Market Crash Detection System", bold: true, size: 40 })] }),
  new Paragraph({ spacing: { after: 600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Isolation-Forest anomaly detection for crash and rally prediction on Indian equities", italics: true, size: 24, color: MUTED })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "Project documentation — living document", size: 22, color: MUTED })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "Universe: NIFTY 100  •  Data: EOD daily bars  •  Capital: ₹10,00,000", size: 22, color: MUTED })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: `Generated ${new Date().toISOString().slice(0, 10)}`, size: 20, color: MUTED })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// --- TOC ---
children.push(
  H1("Contents"),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-3" }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ================================================================ 1
children.push(
  H1("1. What we are building"),
  P("A system that watches roughly 100 large Indian stocks every day and answers two questions: is a crash coming, and is a rally coming. When it thinks a crash is coming it sells and moves to cash. When it thinks a rally is coming it buys back in. The rest of the time it simply holds."),
  P("That last sentence is the important one. This is not a strategy that trades constantly looking for small edges. It is buy-and-hold plus an exit rule. Every trade costs money in brokerage, taxes and slippage, so a trade has to earn its place."),
  H2("1.1 What success looks like"),
  RichP([
    { text: "The headline result is " },
    { text: "drawdown reduction", bold: true },
    { text: ". If you had simply bought these stocks and held them from 2021 to 2026, you would have suffered some worst-case loss from peak to trough. Our claim is that the model sidesteps a meaningful part of that loss while giving up as little upside as possible." },
  ]),
  RichP([
    { text: "The second result is " },
    { text: "lead time", bold: true },
    { text: ": how many trading days before a crash did we raise the alarm? This is reported as a distribution across all crash events, not a single average, because an average hides the events we missed entirely." },
  ]),
  H2("1.2 Design constraints agreed up front"),
  Tbl(
    ["Constraint", "Decision", "Why"],
    [
      ["Direction", "Long-only, exit to cash", "Realistic for EOD cash equities; no borrowing or shorting complications"],
      ["Trade frequency", "Under ~10 round trips per stock over 5.5 years", "Costs and taxes make frequent trading self-defeating"],
      ["Data", "End-of-day daily bars", "No intraday data required"],
      ["Capital", "₹10,00,000, no leverage", "Fixed budget; position sizes must be whole shares"],
      ["Look-ahead", "Strictly forbidden, enforced by tests", "See section 3.3 — this is the single easiest way to fool yourself"],
    ],
    [1900, 2700, 4400]
  ),
  new Paragraph({ children: [new PageBreak()] }),
);

// ================================================================ 2
children.push(
  H1("2. Concepts you need, explained plainly"),
  P("This section assumes basic familiarity with markets and machine learning, and fills in the specific ideas this project leans on."),

  H2("2.1 Isolation Forest, and what it actually does"),
  P("Isolation Forest is an unsupervised anomaly detector. Unsupervised means nobody hands it labelled examples of crashes; it never sees a crash and is told “this is a crash”. It only learns what ordinary looks like, and then flags whatever fails to resemble that."),
  P("The mechanism is unusual and worth understanding, because it explains the score. The algorithm builds many random decision trees. At each step it picks a feature at random and a split value at random, cutting the data in half over and over until every point sits alone in its own leaf. It then counts how many cuts each point needed."),
  RichP([
    { text: "The insight is that " },
    { text: "outliers get isolated quickly", bold: true },
    { text: ". A point sitting far from everything else gets separated after only a few random cuts. A point buried in a dense cluster needs many cuts before it is alone. So the average number of cuts — the " },
    { text: "path length", italics: true },
    { text: " — is a direct measure of how anomalous a point is. Short path means anomalous. This is why the method is fast: it never has to model what normal looks like, it just measures how easy something is to cut away." },
  ]),
  Callout("The single most important caveat", [
    "Isolation Forest is direction-blind. It flags unusual, not unusually bad. A violent rally is exactly as anomalous as a violent crash — both are far from ordinary.",
    "This means anomaly score alone can never produce a buy or a sell signal. It tells us something is happening. Slope tells us which direction. Acceleration tells us whether it is building or exhausting.",
    "That is the entire reason the slope/acceleration module exists.",
  ]),

  H2("2.2 Anomaly intensity: turning the score into something usable"),
  RichP([
    { text: "The raw score from scikit-learn is not stable across models. Its scale shifts with the " },
    { text: "contamination", code: true },
    { text: " setting, with the trees that happened to be built, and with the training window. Since we plan to compare four different retraining schemes, a fixed cutoff on the raw score would mean four different things and the comparison would be meaningless." },
  ]),
  P("So we convert the score into a percentile rank against the training data:"),
  ...Code([
    "intensity(x) = fraction of TRAINING days that were less anomalous than x",
    "",
    "  intensity = 0.99  ->  more unusual than 99% of training days",
    "  intensity = 0.50  ->  a perfectly ordinary day",
  ]),
  P("This gives three properties we need. It means the same thing across all four retraining schemes. It means the same thing for HDFCBANK and for ADANIENT. And it is directly interpretable as an alert budget: a 0.99 threshold implies roughly two or three alerts per year before any confirmation is applied."),
  Tbl(
    ["Band", "Intensity", "What we do"],
    [
      ["High", "≥ 0.99", "Act, if slope and acceleration confirm"],
      ["Moderate", "0.95 – 0.99", "Do not act. Watch for 3–5 days and see which way it resolves"],
      ["Low", "0.90 – 0.95", "Record only"],
    ],
    [1500, 2200, 5300]
  ),

  H2("2.3 Slope and acceleration"),
  P("If you plot a stock price, slope is how steeply it is moving and acceleration is whether that steepness is increasing or easing off. In calculus terms they are the first and second derivative."),
  RichP([
    { text: "We measure both on the " },
    { text: "logarithm", bold: true },
    { text: " of price rather than on price itself. The reason is comparability: a ₹10 move means something very different for a ₹200 stock than for a ₹20,000 stock. The slope of log price is percentage change per day, which means the same thing everywhere. It also makes the measure immune to stock splits." },
  ]),
  P("The sign of the pair is the whole signal logic:"),
  Tbl(
    ["Slope", "Acceleration", "Phase", "Reading"],
    [
      ["Negative", "Negative", "AcceleratingDecline", "Falling and getting worse — crash developing, EXIT"],
      ["Negative", "Positive", "DeceleratingDecline", "Still falling but easing — selloff exhausting, watch"],
      ["Positive", "Positive", "AcceleratingAdvance", "Rising and speeding up — rally developing, ENTER"],
      ["Positive", "Negative", "DeceleratingAdvance", "Still rising but tiring — topping out, caution"],
    ],
    [1300, 1600, 2500, 3600]
  ),

  H2("2.4 Normalising by volatility, and a trap inside it"),
  P("A slope of minus one percent per day is a routine Tuesday for a volatile stock and a genuine emergency for a stable one. To compare across the universe we divide slope by the stock's own recent volatility, giving a reading in units of “daily standard deviations”. That lets one global threshold serve all 100 stocks instead of 100 hand-tuned ones."),
  Callout("A bug this project actually hit", [
    "The first version divided by volatility measured over the most recent 60 days. On real RELIANCE data the COVID crash then scored only -1.67 sigma, when it plainly deserved worse.",
    "The reason: by mid-March 2020 the 60-day volatility window was itself full of crash days. The denominator had grown, so the crash was quietly dividing away its own signal.",
    "The fix is to lag the volatility estimate by 20 days, so it describes the calm the move is departing FROM rather than the chaos it is creating. The same crash then scores -3.01 sigma. There is a regression test pinning this.",
  ]),

  H2("2.5 Drawdown"),
  P("Drawdown is how far you are below your best-ever value. If your portfolio peaked at ₹12 lakh and now sits at ₹9 lakh, you are in a 25% drawdown. Maximum drawdown is the worst such figure over the whole period, and it is the number that decides whether a strategy is actually livable."),
  ...Code([
    "equity_t = portfolio value on day t, after all costs",
    "peak_t   = highest equity seen up to day t",
    "dd_t     = equity_t / peak_t - 1        (always <= 0)",
    "max_dd   = the most negative dd_t",
  ]),
  P("We also report time under water (how many days spent below the previous peak) and time to recovery. Those two are reliability metrics, which is what lets the cyber-physical-systems framing in the research paper be literal rather than decorative."),
  new Paragraph({ children: [new PageBreak()] }),
);

// ================================================================ 3
children.push(
  H1("3. Phase 0 — Data audit"),
  P("Before writing any model code we audited the 100 supplied CSV files. Five defects were found, each of which would have silently corrupted results."),

  H2("3.1 What was found"),
  Tbl(
    ["#", "Defect", "Evidence", "Consequence if ignored"],
    [
      ["1", "adj_close is not an adjusted series", "Differs from close on ~30 of ~5,800 bars; a real adjustment factor differs on every bar before the last corporate action", "Wrong returns around splits and bonuses"],
      ["2", "Fabricated pre-IPO history", "MAZDOCK carries monthly-spaced bars from 2017 though it listed Oct 2020. Same in DMART, SBILIFE, VBL", "Model trains on prices that never existed"],
      ["3", "Zero-range bars", "BAJFINANCE has 1,092 of 5,803 bars with open=high=low=close", "Any range or volatility feature is corrupted"],
      ["4", "Ragged end dates", "Files end anywhere from 2026-06-03 to 2026-06-22", "Cross-sectional breadth becomes unreliable at the edge"],
      ["5", "Survivorship bias", "The universe is today's NIFTY 100, not the historical membership", "Results flatter both strategy and benchmark"],
    ],
    [500, 2300, 3400, 2800]
  ),
  H2("3.2 Decisions taken"),
  Bullet("Use close, discard adj_close. Verification on RELIANCE across the Sept-2017 bonus shows close is already fully back-adjusted."),
  Bullet("Truncate fabricated history using a calendar-gap rule (section 4.2)."),
  Bullet("Flag zero-range bars rather than deleting them — deleting a row silently shifts every rolling window that spans it."),
  Bullet("Trim all series to a common end date of 2026-06-05, which 98 of 100 symbols reach."),
  Bullet("Accept survivorship bias and document it. Because our claim is drawdown reduction rather than stock-picking skill, the bias inflates the buy-and-hold benchmark at least as much as the strategy, which makes the result harder to achieve rather than easier."),

  H2("3.3 Look-ahead bias: the mistake that matters most"),
  P("Look-ahead bias means using information that would not have been available at the time. It is the single easiest way to produce a backtest that looks superb and loses money in production, because the model is effectively being graded on an exam whose answers it has already seen."),
  P("A real example from this project. The team's existing regime detection module classified each day by comparing it against percentile thresholds — but it computed those thresholds from the entire price history at once, including years that had not happened yet, and then applied them to every day including the earliest. Its own documentation stated it was causal. It was not."),
  P("Measured on RELIANCE from 2016 to 2026, 9.9% of regime labels change once the calculation is made properly causal. More tellingly, the biased version produced MORE “Rally” and MORE “Crashing” calls, because knowing the full distribution let it place days into the tails with unearned confidence. Those are precisely the labels that drive entries and exits."),
  Callout("How we prevent this from recurring", [
    "Every feature has a future-perturbation test: take the data, compute the feature, then violently rewrite the future portion and recompute. Every value before the change must be bit-for-bit identical.",
    "If a calculation is peeking ahead, this test fails immediately and unambiguously. It requires no judgement to interpret.",
    "There is also a streaming-equals-batch test: feeding data one day at a time must reproduce exactly what the full-history computation produced. That is the live-trading contract.",
  ]),
  new Paragraph({ children: [new PageBreak()] }),
);

// ================================================================ 4
children.push(
  H1("4. Phase 1 — The data layer"),
  P("Phase 1 turns 100 raw CSV files into a panel that can be trusted, and installs an automated gate so that trust does not decay over time."),

  H2("4.1 Modules built"),
  Tbl(
    ["Module", "Responsibility"],
    [
      ["config.py", "Every path, date window and threshold in one place, so no number can disagree with itself"],
      ["data/loader.py", "Raw CSV to clean per-symbol frame; repairs the five audited defects"],
      ["data/calendar.py", "The master trading calendar, plus a listing mask marking when each symbol actually existed"],
      ["data/quality.py", "The Phase 0 audit encoded as pass/fail checks that block the pipeline"],
      ["scripts/run_phase1.py", "Entry point: load, validate, persist"],
    ],
    [2400, 6600]
  ),

  H2("4.2 The gap rule, and why we trust it"),
  P("To find where a symbol's genuine history begins, we look for gaps longer than 10 calendar days. The NSE has never closed for ten consecutive days, so such a gap is always a data artefact — it means the vendor stitched sparse or monthly data in front of the real listing. The first genuine bar is the one immediately following the last such gap."),
  P("This rule was validated against externally known listing dates, and it recovers all four exactly:"),
  Tbl(
    ["Symbol", "Rule detected", "Actual NSE listing", "Match"],
    [
      ["VBL", "2016-11-08", "2016-11-08", "Exact"],
      ["DMART", "2017-03-21", "2017-03-21", "Exact"],
      ["SBILIFE", "2017-10-03", "2017-10-03", "Exact"],
      ["MAZDOCK", "2020-10-12", "2020-10-12", "Exact"],
    ],
    [2000, 2600, 2800, 1600]
  ),
  P("Validating against dates sourced independently of the code, rather than against the code's own output, is what makes this a real check.", { italics: true }),

  H2("4.3 Why gaps are never filled in"),
  P("Where a stock has no bar, the value stays empty. It is tempting to carry the last price forward, but a forward-filled price manufactures a return of exactly zero — and a run of zero returns reads to the model as a stretch of unnatural calm. That is a quiet lie which survives most sanity checks and biases every volatility estimate downward. An empty value propagates visibly instead and forces an honest decision."),

  H2("4.4 Results"),
  ...Code([
    "loading universe ...",
    "  96 usable symbols, 5820 trading days",
    "",
    "[PASS] universe_size            96 usable symbols loaded (need >= 80)",
    "[PASS] calendar_nonempty        5820 trading days in master calendar",
    "[PASS] no_residual_gaps         no calendar gaps remain after truncation",
    "[PASS] dates_sorted             all symbol indices sorted",
    "[PASS] dates_unique             no duplicate dates",
    "[PASS] prices_positive          all closes > 0",
    "[PASS] ohlc_consistent          high/low bracket open/close everywhere",
    "[PASS] end_dates_aligned        last-bar dates span 2 days",
    "[PASS] train_window_coverage    82 symbols have history at 2016-01-01",
    "[WARN] flat_bar_burden          BAJFINANCE 1092 zero-range bars",
    "[WARN] prelisting_truncation    10 symbols truncated",
    "[WARN] symbols_dropped          ENRIN, TATACAP, TMCV (too little history)",
  ]),
  P("Three symbols are excluded for having fewer than 300 bars, which cannot support a 60-day volatility warmup plus a meaningful backtest. Errors stop the pipeline; warnings are recorded and allowed through."),

  H2("4.5 The quality gate"),
  P("The audit is not a document, it is a test that runs every time. Data gets refreshed, a vendor changes a format, and six weeks later a backtest is quietly wrong. Encoding the audit as a gate means a regression fails loudly instead of producing a slightly different Sharpe ratio that nobody questions."),
  new Paragraph({ children: [new PageBreak()] }),
);

// ================================================================ 5
children.push(
  H1("5. Phase 2 (part) — Slope and acceleration"),
  P("Built ahead of schedule because the signal logic depends on it and because it removed a blocking dependency on the P-spline smoother."),

  H2("5.1 How it is computed"),
  P("Slope is a least-squares straight-line fit through the last few days of log price. Acceleration is the curvature term from a single quadratic fit — deliberately not the difference between two slope readings, because stacking two smoothers roughly doubles the lag, and lag is exactly what we cannot afford when the goal is seeing a turn several days early."),
  P("Because the trailing window is evenly spaced in time, both regressions collapse mathematically to a fixed set of weights. Each feature is therefore a single weighted sum over the trailing window. This is not merely fast: it makes causality structural. There is no code path that could look forward, so it cannot be broken by a later edit."),

  H2("5.2 Validation on the real universe"),
  P("With no tuning at all, the fraction of the universe in AcceleratingDecline picks out genuine market events:"),
  Tbl(
    ["Date", "% of universe declining", "Median slope (sigma/day)", "Event"],
    [
      ["2020-03-12", "92.1%", "-2.11", "COVID crash"],
      ["2026-03-31", "90.5%", "-0.79", "—"],
      ["2016-02-11", "90.1%", "-0.94", "Feb-2016 global selloff"],
      ["2022-09-26", "87.2%", "-0.67", "Fed / GBP crisis"],
      ["2021-12-20", "86.2%", "-0.82", "Omicron"],
      ["2017-09-25", "84.5%", "-0.72", "—"],
    ],
    [1700, 2400, 2400, 2500]
  ),
  Callout("What this told us about the market-wide signal", [
    "Breadth alone is not enough. Around 90% of the universe was falling on both the COVID crash and on mild pullbacks.",
    "Only the median slope separates them: -2.11 sigma for COVID versus roughly -0.8 for the others.",
    "So the market-wide trigger needs BOTH conditions: breadth at or above 75% AND median slope at or below -1.5 sigma. This threshold was calibrated from measured data rather than guessed.",
  ]),
  new Paragraph({ children: [new PageBreak()] }),
);

// ================================================================ 6
children.push(
  H1("6. Regime detection"),
  P("The team's existing regime module has been brought into the codebase with the look-ahead bug of section 3.3 fixed, and with its public interface left unchanged so that the HMM version now being built elsewhere can drop straight in."),
  H2("6.1 Changes made"),
  Num("Look-ahead fixed. Percentile thresholds are now computed from a trailing window instead of from the whole series."),
  Num("Rolling rather than expanding. A three-year trailing window keeps thresholds adaptive; ranking a 2026 day against the 2003 distribution ignores how much Indian market volatility has structurally shifted."),
  Num("Warmup added for the market regime, which previously had none and so fitted early thresholds on a handful of points."),
  Num("Vectorised. The original recomputed percentiles over a growing slice on every bar."),
  Num("The P-spline input is now optional. Raw close works and is preferred, because a smoother fitted over the full sample would reintroduce look-ahead through the back door."),

  H2("6.2 Regime is context, not a trigger"),
  P("Regime uses a 20-day window, so it cannot say “Crashing” until a crash is largely over. That is acceptable and by design: Isolation Forest fires fast, and regime confirms slowly. Attempting to speed regime up would only make it noisy."),

  H2("6.3 Note for the HMM team"),
  RichP([
    { text: "A " },
    { text: "RegimeDetector", code: true },
    { text: " interface is defined in the module. Any replacement must satisfy two rules. First, it must be causal: for an HMM this means refitting as you go, or at minimum using " },
    { text: "filtered", bold: true },
    { text: " state probabilities — the probability of today's state given data up to today — rather than " },
    { text: "smoothed", bold: true },
    { text: " ones, which use the entire series. Fitting an HMM on all the data and then decoding it is the same look-ahead bug in a more sophisticated costume, and it is the usual way this goes wrong. Second, warmup bars must return the neutral defaults so downstream code never sees a null." },
  ]),
  P("The future-perturbation test in tests/test_regime.py can be pointed at any detector satisfying the interface. Run it against the HMM before wiring it in."),
  new Paragraph({ children: [new PageBreak()] }),
);

// ================================================================ 7
children.push(
  H1("7. Roadmap"),
  Tbl(
    ["Phase", "Deliverable", "Status"],
    [
      ["0", "Data audit — five defects identified", "Done"],
      ["1", "Data layer — loader, calendar, quality gate", "Done"],
      ["2", "Features — slope/accel done; precursors and cross-sectional pending", "Part done"],
      ["3", "Isolation Forest and anomaly intensity", "Pending"],
      ["4", "Crash labels and lead-time measurement", "Pending — go/no-go gate"],
      ["5", "Signal generation, per-stock and market-wide", "Pending"],
      ["6", "Backtest engine and cost model", "Pending"],
      ["7", "Retraining method comparison", "Pending"],
      ["8", "Drawdown analysis and per-stock charts", "Pending"],
      ["9", "Cyber robustness experiment", "Pending"],
      ["10", "HTML dashboard", "Pending"],
    ],
    [900, 6300, 1800]
  ),
  Callout("Phase 4 is the decision point", [
    "Phase 4 measures how many days of warning the model actually gives. If the lead time is not there, then signals, backtesting and dashboards are all wasted effort built on a detector that does not work.",
    "It is far better to learn this from a lead-time histogram in Phase 4 than from an equity curve in Phase 8.",
  ]),

  H2("7.1 Retraining methods to be compared"),
  Num("Rolling three-year window"),
  Num("Incremental retraining in one-month steps"),
  Num("Exponentially weighted with decay factor 0.994"),
  Num("Volatility-purged rolling window — proposed addition"),
  P("The fourth is our own proposal and rests on a counterintuitive point. Isolation Forest defines normal as whatever it was trained on. Feed it crisis data and crises become LESS anomalous, because the model stretches its notion of normal to cover them, and detection degrades exactly when it matters. The fix is therefore not more crisis data but less: fit the rolling window after discarding days whose volatility sits in the top decile. The model then learns a clean picture of normal, and anything crisis-like falls far outside it."),

  H2("7.2 How the comparison will be judged"),
  P("All four schemes share identical features, labels, costs and thresholds. Only the fitting schedule varies. Ranking is by drawdown reduction per unit of turnover — not by raw return, since a scheme that trades constantly can buy a better drawdown figure at a cost that only shows up in the tax bill."),
  new Paragraph({ children: [new PageBreak()] }),
);

// ================================================================ 8
children.push(
  H1("8. Costs and taxes"),
  P("The existing cost model computes the Indian per-trade stack correctly: securities transaction tax on both legs for delivery trades, stamp duty on purchases only, depository charges on sales only, and GST on the appropriate base. Three additions are required before backtesting."),
  Num("Capital gains tax, computed at portfolio level on a financial-year basis. This cannot live in the per-trade function. Rates also changed partway through our backtest window on 23 July 2024, so the model must be date-aware. Current rates should be confirmed before being hard-coded."),
  Num("Volatility-scaled slippage. A flat slippage percentage is the wrong shape for this strategy specifically, because it trades only on crash and rally days — precisely when bid-ask spreads widen several-fold. A flat figure understates cost exactly where all the trades are."),
  Num("Date-aware exchange charges, which changed on 1 October 2024."),
  Callout("A finding worth its own section in the paper", [
    "Every crash-exit converts a long-term holding into a short-term one. In India that moves the tax rate from 12.5% to 20%.",
    "So drawdown reduction carries a tax penalty on top of transaction costs — the strategy is quietly paying for its own safety.",
    "Published drawdown-reduction results almost universally ignore this. Quantifying it is a genuine contribution.",
  ]),
  new Paragraph({ children: [new PageBreak()] }),
);

// ================================================================ 9
children.push(
  H1("9. Glossary"),
  Tbl(
    ["Term", "Meaning"],
    [
      ["Anomaly intensity", "Percentile rank of a day's anomaly score against the training distribution. 0.99 means more unusual than 99% of training days"],
      ["Backtest", "Simulating the strategy over historical data to estimate how it would have performed"],
      ["Breadth", "The fraction of the universe doing something at once, e.g. percent of stocks in accelerating decline"],
      ["Causal", "A calculation that uses only information available at the time. The opposite of look-ahead bias"],
      ["Contamination", "The Isolation Forest setting for the expected proportion of anomalies in training data"],
      ["Drawdown", "How far below the previous peak the portfolio currently sits, as a percentage"],
      ["EOD", "End of day. One bar per trading day, no intraday detail"],
      ["Lead time", "Trading days between raising an alert and the crash actually beginning"],
      ["Log return", "The natural logarithm of the price ratio. Comparable across price levels and immune to splits"],
      ["Look-ahead bias", "Accidentally using future information in a historical simulation. Produces excellent backtests and real losses"],
      ["LTCG / STCG", "Long-term and short-term capital gains, taxed differently depending on holding period"],
      ["OHLCV", "Open, high, low, close, volume — the five fields of a price bar"],
      ["Path length", "How many random cuts an Isolation Forest needed to isolate a point. Short means anomalous"],
      ["Regime", "The prevailing market backdrop, e.g. Crashing, Sideways, Rally"],
      ["Slippage", "The gap between the price you expected and the price you actually got"],
      ["Survivorship bias", "Studying only the companies that are still in the index today, so the failures are invisible"],
      ["Unsupervised", "Learning without labelled examples. The model is never shown a labelled crash"],
      ["Whipsaw", "Being stopped out and re-entered repeatedly, paying costs each time for no gain"],
    ],
    [2200, 6800]
  ),
);

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "QBEAST",
  title: "QBEAST Stock Market Crash Detection System",
  description: "Project documentation",
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 460, hanging: 260 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 860, hanging: 260 } } } },
      ] },
      { reference: "numbers", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 460, hanging: 260 } } } },
      ] },
    ],
  },
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 21 } },
      heading1: { run: { font: "Calibri", size: 34, bold: true, color: ACCENT } },
      heading2: { run: { font: "Calibri", size: 26, bold: true, color: "2E5F8A" } },
      heading3: { run: { font: "Calibri", size: 23, bold: true, color: MUTED } },
    },
  },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children,
  }],
});

const out = path.join(__dirname, "..", "docs", "QBEAST_Crash_Detection_Documentation.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(out, buf);
  console.log("wrote", out, `(${(buf.length / 1024).toFixed(0)} KB)`);
});
