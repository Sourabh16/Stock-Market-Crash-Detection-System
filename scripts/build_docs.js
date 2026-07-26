/*
 * build_docs.js
 * -------------
 * Generates per-phase documentation into docs/phases/.
 *
 *     node scripts/build_docs.js            # all phases
 *     node scripts/build_docs.js phase1     # one phase
 *
 * Documents are GENERATED, never hand-edited, so they cannot drift from the
 * code. To add a phase, append an entry to PHASES near the bottom.
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  TableOfContents, PageBreak, LevelFormat, Header, Footer, PageNumber,
} = require("docx");

const CONTENT_W = 9000;
const ACCENT = "1F4E79";
const ACCENT2 = "2E5F8A";
const MUTED = "595959";
const CODE_BG = "F2F2F2";
const OK = "1E7B34";
const BAD = "B22222";

const H1 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 } });
const H2 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2, spacing: { before: 300, after: 120 } });
const H3 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_3, spacing: { before: 240, after: 100 } });

const P = (text, opts = {}) =>
  new Paragraph({
    spacing: { after: opts.after ?? 130, line: 300 },
    alignment: opts.align,
    indent: opts.indent,
    children: [new TextRun({ text, italics: opts.italics, bold: opts.bold, color: opts.color, size: opts.size })],
  });

/** Paragraph built from fragments: {text, bold, italics, code, color}. */
const RichP = (frags, opts = {}) =>
  new Paragraph({
    spacing: { after: opts.after ?? 130, line: 300 },
    children: frags.map((f) =>
      new TextRun({
        text: f.text,
        bold: f.bold,
        italics: f.italics,
        color: f.color,
        font: f.code ? "Consolas" : undefined,
        size: f.code ? 19 : undefined,
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
      spacing: { after: i === lines.length - 1 ? 170 : 0, before: i === 0 ? 70 : 0 },
      shading: { type: ShadingType.CLEAR, fill: CODE_BG },
      indent: { left: 240 },
      children: [new TextRun({ text: l || " ", font: "Consolas", size: 18 })],
    })
  );

/** Left-bar callout box. tone: "info" | "warn" | "good". */
const Callout = (title, body, tone = "info") => {
  const bar = tone === "warn" ? BAD : tone === "good" ? OK : ACCENT;
  const bg = tone === "warn" ? "FDF6F6" : tone === "good" ? "F5FBF6" : "F7FAFD";
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: bar },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: bar },
      left: { style: BorderStyle.SINGLE, size: 20, color: bar },
      right: { style: BorderStyle.SINGLE, size: 2, color: bar },
      insideHorizontal: { style: BorderStyle.NONE },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [new TableRow({ children: [new TableCell({
      width: { size: CONTENT_W, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: bg },
      margins: { top: 150, bottom: 150, left: 220, right: 200 },
      children: [
        new Paragraph({ spacing: { after: 70 }, children: [new TextRun({ text: title, bold: true, color: bar })] }),
        ...body.map((b, i) => P(b, { after: i === body.length - 1 ? 0 : 70 })),
      ],
    })] })],
  });
};

/** Table with header row. widths must sum to CONTENT_W. */
const Tbl = (headers, rows, widths, opts = {}) =>
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) => new TableCell({
          width: { size: widths[i], type: WidthType.DXA },
          shading: { type: ShadingType.CLEAR, fill: ACCENT },
          margins: { top: 90, bottom: 90, left: 130, right: 130 },
          children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, color: "FFFFFF", size: 19 })] })],
        })),
      }),
      ...rows.map((r, ri) => new TableRow({
        children: r.map((c, i) => new TableCell({
          width: { size: widths[i], type: WidthType.DXA },
          shading: { type: ShadingType.CLEAR, fill: ri % 2 ? "FFFFFF" : "F5F8FB" },
          margins: { top: 80, bottom: 80, left: 130, right: 130 },
          children: [new Paragraph({ children: [new TextRun({
            text: String(c),
            size: 19,
            font: opts.mono ? "Consolas" : undefined,
            bold: opts.boldFirstCol && i === 0,
          })] })],
        })),
      })),
    ],
  });

const Rule = () =>
  new Paragraph({ spacing: { before: 120, after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "D0D7DE" } }, children: [] });

const Break = () => new Paragraph({ children: [new PageBreak()] });

const helpers = { H1, H2, H3, P, RichP, Bullet, Num, Code, Callout, Tbl, Rule, Break,
                  CONTENT_W, ACCENT, ACCENT2, MUTED, OK, BAD };

/** Cover page for a phase document. */
function cover({ phase, title, subtitle, status, meta }) {
  return [
    new Paragraph({ spacing: { before: 2200, after: 40 }, alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "QBEAST", bold: true, size: 56, color: ACCENT })] }),
    new Paragraph({ spacing: { after: 500 }, alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Stock Market Crash Detection System", size: 24, color: MUTED })] }),
    new Paragraph({ spacing: { after: 60 }, alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: phase, bold: true, size: 30, color: ACCENT2 })] }),
    new Paragraph({ spacing: { after: 100 }, alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: title, bold: true, size: 44 })] }),
    new Paragraph({ spacing: { after: 500 }, alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: subtitle, italics: true, size: 23, color: MUTED })] }),
    new Paragraph({ spacing: { after: 60 }, alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: status, bold: true, size: 22, color: OK })] }),
    ...meta.map((m) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 50 },
      children: [new TextRun({ text: m, size: 20, color: MUTED })] })),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200 },
      children: [new TextRun({ text: `Generated ${new Date().toISOString().slice(0, 10)}`, size: 19, color: MUTED })] }),
    Break(),
  ];
}

function buildDocument({ title, docTitle, children }) {
  return new Document({
    creator: "QBEAST",
    title: docTitle,
    description: "QBEAST crash detection system — phase documentation",
    numbering: { config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 460, hanging: 260 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 880, hanging: 260 } } } },
      ] },
      { reference: "numbers", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 460, hanging: 260 } } } },
      ] },
    ] },
    styles: { default: {
      document: { run: { font: "Calibri", size: 21 }, paragraph: { spacing: { line: 300 } } },
      heading1: { run: { font: "Calibri", size: 32, bold: true, color: ACCENT } },
      heading2: { run: { font: "Calibri", size: 25, bold: true, color: ACCENT2 } },
      heading3: { run: { font: "Calibri", size: 22, bold: true, color: MUTED } },
    } },
    sections: [{
      properties: { page: { margin: { top: 1300, bottom: 1300, left: 1400, right: 1400 } } },
      headers: { default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "D0D7DE" } },
        children: [new TextRun({ text: title, size: 17, color: MUTED })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 17, color: MUTED })] })] }) },
      children,
    }],
  });
}


// =====================================================================
// Phase content
// =====================================================================
const HELPERS = { H1, H2, H3, P, RichP, Bullet, Num, Code, Callout, Tbl, Rule, Break,
                  CONTENT_W, ACCENT, ACCENT2, MUTED, OK, BAD, cover, TableOfContents };

const PHASES = {
  phase1: {
    slug: "Phase1_Data_Layer",
    docTitle: "QBEAST Phase 1 — The Data Layer",
    runningHead: "QBEAST · Phase 1 — Data Layer",
    build(h) {
      const { H1, H2, H3, P, RichP, Bullet, Num, Code, Callout, Tbl, Rule, Break,
              cover, TableOfContents } = h;
  const c = [];

  c.push(...cover({
    phase: "PHASE 1",
    title: "The Data Layer",
    subtitle: "Turning 100 raw CSV files into a panel that can be trusted",
    status: "STATUS: COMPLETE — 53 tests passing",
    meta: [
      "96 usable symbols · 5,820 trading days · 2003-01-02 to 2026-06-05",
      "Six data defects identified and handled",
      "All 10 ERROR-level quality checks passing",
    ],
  }));

  c.push(H1("Contents"),
    new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
    Break());

  // ============================================================ 1
  c.push(
    H1("1. Purpose"),
    P("Phase 1 has one job: turn the raw price files into data that can be believed, and then make that trust permanent."),
    P("This sounds like plumbing, and it is tempting to rush. It is worth resisting that temptation, because of an asymmetry specific to this project. Isolation Forest is unsupervised — nobody hands it labelled crashes, so it has no way to check its own understanding against ground truth. It simply learns that whatever it was shown is what normal looks like. Feed it one impossible day and it will faithfully expand its notion of normal to accommodate it."),
    P("A supervised model would fight back. Shown a mislabelled example, its error signal rises and training pushes against the mistake. An unsupervised model has no such immune system. It absorbs bad data silently and produces confident, plausible, wrong scores forever afterwards."),
    Callout("The rule this phase is built on", [
      "Data defects in this pipeline do not announce themselves. Every one found in Phase 0 and Phase 1 produced output that looked completely normal — correctly formatted dates, positive prices, sensible-looking returns.",
      "Not one would have caused a crash, an exception, or a warning. They would simply have produced a slightly different, quietly wrong answer.",
      "That is why this phase ends in an automated gate rather than a document.",
    ]),

    H2("1.1 What Phase 1 delivers"),
    Tbl(["Deliverable", "Description"], [
      ["Cleaned per-symbol frames", "One frame per stock with OHLCV plus quality flags, written to data/interim/"],
      ["Aligned close panel", "Wide table, 5,820 dates × 96 symbols, written to data/processed/"],
      ["Listing mask", "Boolean panel marking when each symbol actually existed"],
      ["Audit trail", "Per-symbol CSV recording every repair made, in reports/"],
      ["Quality gate", "10 blocking checks plus 3 advisory warnings"],
      ["Test suite", "53 tests, including causality and regression guards"],
    ], [2900, 6100]),
    Break(),
  );

  // ============================================================ 2
  c.push(
    H1("2. The data contract"),
    H2("2.1 Input"),
    P("One CSV per symbol in data/raw/, named by ticker. The index file is required — it supplies the master trading calendar."),
    ...Code([
      "data/raw/",
      "├── RELIANCE.csv",
      "├── TCS.csv",
      "├── ...",
      "└── NIFTY100.csv        <- required",
    ]),
    Tbl(["Column", "Required", "Notes"], [
      ["date", "yes", "DD-MM-YYYY. Parsed with an explicit format, never inferred"],
      ["open, high, low, close", "yes", "close is used; adj_close is discarded — see defect 1"],
      ["volume", "yes", "Zero throughout for the index, which is expected"],
      ["_source", "no", "Vendor that supplied the bar (kite / upstox). Preserved"],
      ["_dq_score", "no", "Vendor quality score. Preserved"],
      ["_gap_filled", "no", "Vendor interpolation flag. Preserved"],
    ], [2200, 1200, 5600]),
    RichP([
      { text: "Why the date format is pinned rather than inferred: parsing " },
      { text: "DD-MM-YYYY", code: true },
      { text: " as month-first succeeds silently for the first twelve days of every month and corrupts the rest. It produces no error — just a series with a scrambled calendar." },
    ]),
    P("The underscore-prefixed metadata is retained deliberately. The natural instinct is to strip metadata columns during cleaning, but _source records which vendor supplied each bar. Two vendors observing one market is, in cyber-physical terms, two independent sensors observing one plant — and vendor disagreement is a sensor-fault signature. That option is only available if the column survives cleaning. Dropping it is irreversible."),

    H2("2.2 Output"),
    Tbl(["Artefact", "Shape", "Contents"], [
      ["data/interim/<SYM>.parquet", "per symbol", "OHLCV plus is_flat, is_zero_vol, gap_filled"],
      ["data/processed/close_panel.parquet", "5820 × 96", "Aligned closes; gaps left as missing"],
      ["data/processed/listing_mask.parquet", "5820 × 96", "True where the symbol was listed and tradeable"],
      ["reports/phase1_data_audit.csv", "99 rows", "Every repair, per symbol"],
    ], [3400, 1500, 4100]),
    Break(),
  );

  // ============================================================ 3
  c.push(
    H1("3. The six defects"),
    P("Five were found in the Phase 0 audit. The sixth was found during Phase 1 testing, by checking the cleaned output against reality rather than against the unit tests — which is a distinction worth holding on to."),

    H3("Defect 1 — adj_close is not an adjusted series"),
    P("Evidence: adj_close differs from close on roughly 30 of 5,800 bars per symbol. A genuine adjustment factor differs on EVERY bar before the last corporate action, because the whole series is rescaled. Differing on thirty scattered days means the column is a near-copy of close with a handful of ad-hoc edits."),
    P("Cross-check: RELIANCE across its September 2017 1:1 bonus shows close moving 392.10 to 389.90 — no gap. An unadjusted series would halve. So close is already back-adjusted."),
    RichP([{ text: "Handling: use " }, { text: "close", code: true }, { text: "; discard " }, { text: "adj_close", code: true },
            { text: ". Confirming this with whoever built the data pipeline remains an open item, because if it is wrong then every return in the project is wrong." }]),

    H3("Defect 2 — fabricated pre-IPO history"),
    P("Evidence: MAZDOCK carries monthly-spaced bars from December 2017, but the company listed in October 2020. The same pattern appears in DMART, SBILIFE and VBL. Someone stitched sparse data in front of the real listing history."),
    P("Rule: a calendar gap longer than 10 days is never a holiday — the NSE has never closed that long. The first genuine bar is the one immediately after the LAST such gap."),
    P("Validation, against listing dates sourced independently of this codebase:"),
    Tbl(["Symbol", "Rule detected", "Actual NSE listing", "Match"], [
      ["VBL", "2016-11-08", "2016-11-08", "exact"],
      ["DMART", "2017-03-21", "2017-03-21", "exact"],
      ["SBILIFE", "2017-10-03", "2017-10-03", "exact"],
      ["MAZDOCK", "2020-10-12", "2020-10-12", "exact"],
    ], [1900, 2500, 2900, 1700]),
    P("Four for four, to the day. Validating against external facts rather than against the code's own output is what makes this a real check rather than the code grading its own homework.", { italics: true }),

    H3("Defect 3 — zero-range bars"),
    P("Evidence: BAJFINANCE has 1,092 bars out of 5,803 where open, high, low and close are all identical. ADANIPOWER has 267, MOTHERSON 277, NESTLEIND 234."),
    P("A bar with no intraday range carries no information about volatility or trading pressure, and would corrupt any feature built on the high-low span."),
    Callout("Flagged, never dropped", [
      "The obvious move is to delete these rows. That would be a mistake.",
      "Deleting a row silently shifts every rolling window that spans it: a five-day slope would quietly cover six calendar days, and every affected window would be measuring something slightly different from what its name claims.",
      "Instead each bar carries an is_flat flag. The row stays, the calendar stays intact, and range-based features mask it explicitly.",
    ]),

    H3("Defect 4 — ragged end dates"),
    P("Evidence: files end anywhere between 2026-06-03 and 2026-06-22. One symbol runs nearly three weeks past the rest."),
    P("This matters because the market-wide signal asks what fraction of the universe is falling today. If symbols drop out of the sample at different dates, that fraction drifts for reasons that have nothing to do with markets. All series are trimmed to 2026-06-05, which 98 of 100 symbols reach."),

    H3("Defect 5 — survivorship bias"),
    P("The universe is today's NIFTY 100, not the historical membership. Companies that fell out of the index are invisible, so the sample is quietly biased towards firms that did well."),
    P("Documented rather than fixed. Because the claim of this project is drawdown reduction rather than stock-picking skill, the bias inflates the buy-and-hold benchmark at least as much as it inflates the strategy — which makes the result harder to achieve, not easier. Sourcing historical index membership would be the rigorous fix and remains available if a reviewer presses."),

    H3("Defect 6 — unadjusted corporate actions (found in Phase 1 testing)"),
    P("The price series is back-adjusted for splits and bonuses but NOT for demergers."),
    ...Code([
      "CGPOWER   2016-03-15    155.30 -> 53.05   (-65.8%)   and it stays there",
      "ADANIENT  2015-06-04                       (-78.4%)",
    ]),
    P("CGPOWER is the Crompton Greaves Consumer demerger; ADANIENT is the Adani Ports / Transmission / Power spin-off. No shareholder lost 66% — they received shares in the demerged entity. These are artefacts of the price series, not events in the market."),
    Callout("Why every existing check missed it", [
      "A demerger bar is internally consistent. High is above close, close is above low, the price is positive, the date is valid, there is no gap. It passes every structural test in the gate.",
      "It simply describes a different company from the day before. Only the RETURN betrays it.",
      "CGPOWER passed all nine checks that existed at the time.",
    ], "warn"),
    P("Why one bar matters so much: Isolation Forest is unsupervised, so it learns normal from whatever it is shown. A minus 66% day inside the training window becomes the most extreme point in the entire sample, and the anomaly boundary stretches out towards it. Genuine crashes then look ordinary by comparison. One bad bar degrades every score that follows."),
    P("The difficult part was not filtering the artefact out — it was not filtering too much:"),
    Tbl(["Event", "Move", "Nature", "Decision"], [
      ["CGPOWER 2016-03-15", "-65.8%", "demerger", "remove"],
      ["TRENT 2026-06-02", "-31.9%", "market move", "KEEP"],
      ["ADANIENT 2023-02-03", "-26.1%", "Hindenburg report", "KEEP"],
    ], [2700, 1500, 2700, 1900]),
    P("A blanket outlier filter would have deleted the Hindenburg crash — one of the exact events this project exists to detect. The threshold sits at 0.50 in log terms, roughly minus 39% to plus 65%, and a test asserts that the Hindenburg move survives cleaning."),
    P("Handling mirrors the pre-listing rule: history before the break belongs to a different security and is discarded. Prior prices are deliberately NOT re-adjusted, because the correct factor depends on the demerger ratio, which is not in this data — and inferring it from the price jump would assume exactly what we are trying to detect."),

    H2("3.1 A related finding, documented not fixed"),
    P("Five unrelated symbols move more than 30% on the same day, 2005-01-03, and three more on 2007-01-02. Unrelated companies do not do that on a January 2nd. These are vendor adjustment-basis splices in the older history."),
    P("Both dates fall well before the 2016 training start, so they lie outside every window this project uses. Recorded here because they indicate the older history is stitched from sources with differing adjustment conventions, which is worth knowing if the training window is ever extended backwards."),
    Break(),
  );

  // ============================================================ 4
  c.push(
    H1("4. Modules"),
    Tbl(["Module", "Responsibility"], [
      ["config.py", "Every path, date window and threshold, in one place"],
      ["data/loader.py", "Raw CSV to clean frame; one explicit rule per defect"],
      ["data/calendar.py", "Master trading calendar and listing mask"],
      ["data/quality.py", "The audit, encoded as a blocking gate"],
      ["scripts/run_phase1.py", "Entry point: load, validate, persist"],
    ], [2500, 6500]),

    H2("4.1 config.py"),
    P("Every threshold in this project ends up in a result somewhere. If a number lives in three files it will eventually disagree with itself, and a backtest that disagrees with itself is worse than no backtest. Everything tunable lives here and nowhere else."),

    H2("4.2 data/loader.py"),
    P("One explicit, tested rule per defect. Nothing is repaired by inference or by filling."),
    P("Where a symbol has both stitched pre-listing bars and a later restructuring, whichever break is later bounds the usable history."),

    H2("4.3 data/calendar.py"),
    P("The calendar is taken from the index rather than from a union of all stock dates. A union would inherit every vendor artefact in every file and invent trading sessions that never happened. The index has a bar on every session the exchange was open, which no individual stock guarantees — a stock can be suspended or halted."),
    P("The listing mask records when each symbol actually existed. Without it, breadth would count HYUNDAI as not falling in 2016, when in truth it had not yet listed. The denominator has to be companies that existed."),

    H2("4.4 data/quality.py"),
    P("The audit is not a document, it is a test that runs every time. Data gets refreshed, a vendor changes a format, and six weeks later a backtest is quietly wrong. Encoding the audit as a gate means a regression fails loudly instead of shifting a Sharpe ratio that nobody questions."),
    Break(),
  );

  // ============================================================ 5
  c.push(
    H1("5. Design decisions"),

    H2("5.1 Gaps are never forward-filled"),
    P("Where a stock has no bar, the value stays missing. Carrying the last price forward is the conventional move and it is wrong here."),
    P("A forward-filled price manufactures a return of exactly zero. A run of zero returns reads to the model as a stretch of unnatural calm, and biases every volatility estimate downward — which matters enormously, since volatility is the denominator that normalises slope and acceleration."),
    P("Measured on the current panel: 0.53% of daily returns are exactly zero. Forward-filling would inflate that substantially, and nothing in the output would look wrong."),

    H2("5.2 Flag, do not drop"),
    P("Bad bars are marked, not deleted, for the reason given under defect 3: deleting a row silently changes the meaning of every rolling window that spans it. A flagged row keeps the calendar honest and forces downstream code to make an explicit decision."),

    H2("5.3 Errors block, warnings inform"),
    P("The gate distinguishes conditions that make the data unusable from conditions that are known and tolerated. Ten checks are ERROR level and stop the pipeline. Three are WARN level: they are recorded in the audit trail and allowed through."),

    H2("5.4 Validate against reality, not against yourself"),
    P("The listing-date and corporate-action rules are both checked against externally sourced facts — NSE listing dates, known demerger dates. A test that compares code output to code output proves only that the code is consistent, which it always is."),
    P("This is also how defect 6 was found: unit tests all passed, and the defect only appeared when the cleaned output was compared against what the market actually did."),
    Break(),
  );

  // ============================================================ 6
  c.push(
    H1("6. Quality gate reference"),
    H2("6.1 Blocking checks"),
    Tbl(["Check", "Asserts"], [
      ["universe_size", "At least 80 usable symbols loaded"],
      ["calendar_nonempty", "More than 1,000 trading days in the master calendar"],
      ["no_residual_gaps", "No calendar gaps remain after truncation"],
      ["dates_sorted", "Every symbol index is monotonically increasing"],
      ["dates_unique", "No duplicate dates"],
      ["prices_positive", "All closes greater than zero"],
      ["ohlc_consistent", "High and low bracket open and close on every bar"],
      ["no_unadjusted_corporate_actions", "No daily move beyond the log-return limit"],
      ["end_dates_aligned", "Last-bar dates span no more than 7 days"],
      ["train_window_coverage", "At least 80 symbols have history at the training start"],
    ], [3600, 5400]),
    H2("6.2 Advisory checks"),
    Tbl(["Check", "Reports"], [
      ["flat_bar_burden", "Symbols exceeding 10% zero-range bars"],
      ["prelisting_truncation", "Symbols whose history was truncated, and by how much"],
      ["symbols_dropped", "Symbols excluded for insufficient history"],
    ], [3600, 5400]),
    Break(),
  );

  // ============================================================ 7
  c.push(
    H1("7. Test coverage"),
    P("53 tests. What matters is not the count but what each one proves."),
    Tbl(["Group", "Proves"], [
      ["Listing-date recovery", "The gap rule reproduces four real NSE listing dates exactly"],
      ["Corporate-action truncation", "CGPOWER and ADANIENT truncate at their true demerger dates"],
      ["Real crashes survive", "Hindenburg and TRENT are not mistaken for artefacts"],
      ["No extreme returns remain", "No symbol retains an implausible daily move after cleaning"],
      ["Flag not drop", "Flat bars are marked and retained; row counts stay consistent"],
      ["No forward fill", "Gaps remain missing; HYUNDAI is empty before its 2024 listing"],
      ["Listing mask", "A 2024 listing cannot dilute 2016 breadth"],
      ["Gate catches injected gap", "Fabricated history sneaking back in fails the build"],
      ["Gate catches injected action", "A synthetic demerger fails the build"],
      ["Gate raises by default", "A degraded universe stops the pipeline rather than warning"],
    ], [3200, 5800]),
    Callout("The regression guards are the point", [
      "Two tests deliberately poison good data — one injects a stitched pre-listing bar, the other a synthetic demerger — and assert the gate fails.",
      "Without these, the gate is only documentation. With them, it is proven to bite.",
    ], "good"),
    Break(),
  );

  // ============================================================ 8
  c.push(
    H1("8. Results"),
    ...Code([
      "loading universe ...",
      "  96 usable symbols, 5820 trading days",
      "",
      "quality gate",
      "[PASS] universe_size                    96 usable symbols (need >= 80)",
      "[PASS] calendar_nonempty                5820 trading days",
      "[PASS] no_residual_gaps                 no gaps remain after truncation",
      "[PASS] dates_sorted                     all symbol indices sorted",
      "[PASS] dates_unique                     no duplicate dates",
      "[PASS] prices_positive                  all closes > 0",
      "[PASS] ohlc_consistent                  high/low bracket open/close",
      "[PASS] no_unadjusted_corporate_actions  no move exceeds 0.50 log return",
      "[PASS] end_dates_aligned                last-bar dates span 2 days",
      "[PASS] train_window_coverage            81 symbols at 2016-01-01",
      "[WARN] flat_bar_burden                  BAJFINANCE 1092 zero-range bars",
      "[WARN] prelisting_truncation            20 symbols truncated",
      "[WARN] symbols_dropped                  ENRIN, TATACAP, TMCV",
      "",
      "wrote close_panel (5820, 96)",
      "live symbols: 31 at 2003-01-02 -> 95 at 2026-06-05",
    ]),
    Tbl(["Measure", "Value"], [
      ["Usable symbols", "96 of 100"],
      ["Trading days", "5,820 (2003-01-02 to 2026-06-05)"],
      ["Symbols at training start (2016-01-01)", "81"],
      ["Symbols live at end", "95"],
      ["Symbols dropped for short history", "3 — ENRIN, TATACAP, TMCV"],
      ["Median annualised volatility, 2016-2026", "30.6%"],
      ["Exact-zero daily returns", "0.53% — consistent with no forward filling"],
    ], [4700, 4300]),
    P("Three symbols are excluded for having fewer than 300 bars, which cannot support a 60-day volatility warmup plus a meaningful backtest."),
    P("One number moved during Phase 1 testing: symbols covering the training start fell from 82 to 81, because CGPOWER now begins at its March 2016 demerger. That is a correct loss — the earlier bars belonged to a different company."),
    Break(),
  );

  // ============================================================ 9
  c.push(
    H1("9. Known limitations"),
    Num("Survivorship bias remains. The universe is today's NIFTY 100. Documented in defect 5; the fix is historical index membership, which we have chosen not to source."),
    Num("close is assumed fully back-adjusted. Verified on RELIANCE across its 2017 bonus, but not confirmed with the data provider. If wrong, every return in the project is wrong. This is the single highest-value open question and it is a five-minute conversation."),
    Num("Corporate actions are truncated, not adjusted. A demerger discards prior history rather than rescaling it, because the demerger ratio is not in the data. CGPOWER therefore loses 2003 to 2016."),
    Num("Pre-2016 vendor splices are documented but untreated. They sit outside all working windows; extending the training window backwards would require addressing them."),
    Num("The corporate-action threshold is a judgement. 0.50 log return cleanly separates every case in the current data, but a genuine 45% one-day crash would be misclassified as an artefact. No such event exists in this universe."),

    H1("10. Reproducing"),
    ...Code([
      "pip install numpy pandas pyarrow scikit-learn pytest",
      "",
      "python scripts/run_phase1.py      # load, repair, validate, persist",
      "python -m pytest tests/ -q        # 53 tests",
    ]),
    P("The pipeline is deterministic and rebuilds from data/raw/ alone. Deleting data/interim/, data/processed/ and reports/ and re-running reproduces every artefact byte for byte."),

    H1("11. Handoff to Phase 2"),
    P("Phase 2 builds the features Isolation Forest will consume. It inherits from Phase 1:"),
    Bullet("A clean per-symbol frame with OHLCV and quality flags"),
    Bullet("An aligned close panel on a shared calendar, so cross-sectional features are well defined"),
    Bullet("A listing mask, so breadth counts only companies that existed"),
    Bullet("Preserved vendor metadata, keeping the sensor-disagreement angle available"),
    P("Two obligations carry forward. Range-based features must mask flat bars explicitly rather than assuming every bar has a usable high-low span. And every new feature must carry a causality test — the future-perturbation check that Phase 2's slope and acceleration module already uses."),
    Callout("Open question to close before Phase 3", [
      "Confirm with the data pipeline team that close is fully back-adjusted.",
      "The RELIANCE cross-check says yes. If it is wrong, every return in the project is wrong, and no amount of downstream rigour would recover it.",
    ], "warn"),
  );

  return c;
    },
  },
};

// =====================================================================
// Runner
// =====================================================================
const outDir = path.join(__dirname, "..", "docs", "phases");
fs.mkdirSync(outDir, { recursive: true });
const wanted = process.argv[2] ? [process.argv[2]] : Object.keys(PHASES);

(async () => {
  for (const key of wanted) {
    const mod = PHASES[key];
    if (!mod) {
      console.error(`unknown phase '${key}' - known: ${Object.keys(PHASES).join(", ")}`);
      process.exitCode = 1;
      continue;
    }
    const doc = buildDocument({
      title: mod.runningHead,
      docTitle: mod.docTitle,
      children: mod.build(HELPERS),
    });
    const buf = await Packer.toBuffer(doc);
    const out = path.join(outDir, `${mod.slug}.docx`);
    fs.writeFileSync(out, buf);
    console.log(`wrote ${path.relative(process.cwd(), out)}  (${(buf.length / 1024).toFixed(0)} KB)`);
  }
})();
