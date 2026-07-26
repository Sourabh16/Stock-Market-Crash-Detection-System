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

  phase2: {
    slug: "Phase2_Features",
    docTitle: "QBEAST Phase 2 — Features",
    runningHead: "QBEAST · Phase 2 — Features",
    build(h) {
      const { H1, H2, H3, P, RichP, Bullet, Num, Code, Callout, Tbl, Rule, Break,
              cover, TableOfContents } = h;
      const c = [];

      c.push(...cover({
        phase: "PHASE 2",
        title: "Features",
        subtitle: "Building the inputs that decide whether early detection is possible at all",
        status: "STATUS: COMPLETE — 71 tests passing",
        meta: [
          "468,266 symbol-days × 10 model features · 5,820 days × 7 market features",
          "All features >98.6% populated in the backtest window",
          "Strongest measured signal: 3.91× lift over base rate",
        ],
      }));

      c.push(H1("Contents"),
        new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
        Break());

      // ---------------------------------------------------------- 1
      c.push(
        H1("1. The central idea"),
        P("The project requires detecting a crash two to ten days before it happens. It is tempting to treat that as a modelling problem — pick a better algorithm, tune it harder. It is not. It is a feature problem, and no amount of downstream cleverness can recover from getting it wrong."),
        P("Isolation Forest finds whatever is unusual in what you give it. Give it today's return and it will faithfully report that today's minus six percent day was unusual — on the day it happened. That is a description, not a prediction. The model has done nothing wrong; it answered exactly the question it was asked."),
        Callout("The design rule for this phase", [
          "Every feature must measure market STRESS rather than price MOVEMENT.",
          "Stress is slow. It builds while price is still roughly flat, which is what makes it available before the break rather than during it.",
          "ret_1d is computed and returned for labelling, plotting and debugging, but it is NOT in FEATURE_COLUMNS and never reaches the model. A test guards against it leaking in later.",
        ]),

        H2("1.1 Three layers"),
        Tbl(["Layer", "Scope", "Answers"], [
          ["Slope and acceleration", "per stock", "Which direction, and is the move building or exhausting?"],
          ["Precursors", "per stock", "Is stress accumulating in this name?"],
          ["Cross-sectional", "whole universe", "Is this one stock, or is it everything at once?"],
        ], [2600, 1700, 4700]),
        P("The third layer exists because a single stock falling on its own news is an exit for that stock, whereas the whole market falling together is a different event needing a different response. Per-stock features cannot tell those apart. Only the cross-section can."),
        Break(),
      );

      // ---------------------------------------------------------- 2
      c.push(
        H1("2. Layer 1 — slope and acceleration"),
        P("Slope is how steeply price is moving; acceleration is whether that steepness is increasing or easing. In calculus terms, the first and second derivative."),
        RichP([
          { text: "Both are measured on the " }, { text: "logarithm", bold: true },
          { text: " of price. A ten-rupee move means something very different for a 200-rupee stock than for a 20,000-rupee one; the slope of log price is percentage change per day, which means the same thing everywhere. It also makes the measure immune to stock splits." },
        ]),

        H2("2.1 How they are computed"),
        P("Slope is a least-squares straight line fitted through the last few days of log price. Acceleration is the curvature term from a single quadratic fit — deliberately not the difference between two slope readings, because stacking two smoothers roughly doubles the lag, and lag is precisely what cannot be afforded when the goal is seeing a turn several days early."),
        P("Because the trailing window is evenly spaced in time, both regressions collapse mathematically to a fixed set of weights. Each feature is therefore a single weighted sum over the trailing window. This is not merely fast: it makes causality structural. There is no code path that could look forward, so it cannot be broken by a later edit."),

        H2("2.2 The sign pair is the signal logic"),
        Tbl(["Slope", "Acceleration", "Phase", "Reading"], [
          ["negative", "negative", "AcceleratingDecline", "falling and worsening — crash developing, EXIT"],
          ["negative", "positive", "DeceleratingDecline", "selloff exhausting — watch for re-entry"],
          ["positive", "positive", "AcceleratingAdvance", "rally building — ENTER"],
          ["positive", "negative", "DeceleratingAdvance", "topping out — caution"],
        ], [1300, 1600, 2500, 3600]),

        H2("2.3 A bug worth understanding"),
        P("To compare across stocks, slope is divided by the stock's own recent volatility, giving a reading in daily standard deviations. The first implementation used volatility measured over the most recent sixty days. On real RELIANCE data the COVID crash then scored only minus 1.67 sigma, which was plainly too mild."),
        Callout("A crash was muting its own signal", [
          "By mid-March 2020 the sixty-day volatility window was itself full of crash days. The denominator had grown, so the crash was quietly dividing away its own severity.",
          "Lagging the volatility estimate by twenty days — so it describes the calm the move is departing FROM rather than the chaos it is creating — scores the same event at minus 3.01 sigma.",
          "This self-normalisation trap recurs throughout the project. It is the same reason Bollinger Bands were rejected as a crash label in section 6.",
        ], "warn"),
        Break(),
      );

      // ---------------------------------------------------------- 3
      c.push(
        H1("3. Layer 2 — per-stock precursors"),
        P("Crashes are not bolts from a clear sky. Before a large decline, several things usually shift while price is still roughly flat. Each is measurable, each is slow, and none requires today to have been a bad day."),
        Tbl(["Feature", "Measures", "Why it should lead"], [
          ["vol_ratio", "5-day volatility over 60-day", "Volatility clusters — turbulence precedes turbulence. A ratio, so 2.0 means twice this stock's own normal, whether that normal is 15% or 45%"],
          ["vol_of_vol", "Instability of volatility itself", "A reliably volatile market is a different regime from one whose volatility is lurching; the second tends to precede dislocations"],
          ["semidev_asym", "Downside versus upside deviation", "Ordinary volatility treats +3% and -3% as identical. This does not. Down-moves outgrowing up-moves is the shape that precedes a break"],
          ["volume_z", "Log volume against 60-day baseline", "Volume picks up as positioning changes, often before price resolves"],
          ["range_expansion", "Intraday range against baseline", "Widening daily ranges signal disagreement about price"],
          ["gap_freq", "Frequency of overnight gaps", "Gaps are information arriving while the market is shut — a rising rate means the story is being written outside trading hours"],
          ["illiquidity", "Amihud: return per rupee traded", "Liquidity thins before it disappears; market makers widen before they withdraw"],
          ["dd_from_high", "Drawdown from 60-day high", "Context, not stress: the same volatility spike means something different at a high than 15% below it"],
          ["slope_z, accel_z", "Trend shape", "See section 2"],
        ], [1700, 2300, 5000]),

        H2("3.1 A design choice a test forced"),
        P("Downside asymmetry was originally written as the obvious ratio: downside deviation divided by upside deviation. A test hit a twenty-day window containing no up days at all — maximum bearish asymmetry — and the feature returned nothing, because it had divided by zero."),
        P("The formulation was undefined exactly where the signal was strongest. It was replaced by a bounded index:"),
        ...Code([
          "semidev_asym = (downside - upside) / (downside + upside)",
          "",
          "   -1 = all upside      0 = symmetric      +1 = all downside",
        ]),
        P("Two benefits. It is always defined. And it is bounded, which matters for this model specifically: Isolation Forest splits at random points within a feature's observed range, so a single extreme outlier stretches that range and wastes most candidate splits on empty space."),

        H2("3.2 Warmup is missing, not averaged"),
        P("Bars without a complete window return nothing rather than a neutral default. A fabricated average during warmup is indistinguishable downstream from a genuine reading, and Isolation Forest would see a cluster of identical fabricated values as a dense, extremely normal region — precisely the wrong lesson."),
        Break(),
      );

      // ---------------------------------------------------------- 4
      c.push(
        H1("4. Layer 3 — cross-sectional"),
        Tbl(["Feature", "Measures"], [
          ["breadth_decline", "Percent of the live universe in AcceleratingDecline"],
          ["breadth_advance", "Percent in AcceleratingAdvance"],
          ["median_slope_z", "How hard the typical stock is moving"],
          ["dispersion", "Cross-sectional spread of daily returns"],
          ["avg_corr", "Average pairwise correlation across the universe"],
          ["pct_below_ma50", "Participation — percent below the 50-day average"],
          ["n_live", "Universe size that day"],
        ], [2500, 6500]),

        H2("4.1 Why correlation is the interesting one"),
        P("In calm markets, stocks move on their own news, so average pairwise correlation is low. As stress builds, everything begins moving together — investors sell what they can rather than what they want to, and individual stories stop mattering. Correlation therefore rises before the index breaks, which makes it one of the few genuinely leading cross-sectional measures. Dispersion is the same signal read from the other side: as correlation climbs, the spread of returns across the universe collapses."),

        H2("4.2 An O(N) shortcut for an O(N-squared) problem"),
        P("The direct route builds a correlation matrix each day and averages its off-diagonal: roughly 4,500 pairs across 5,820 days, repeated four times over in the Phase 7 retraining comparison."),
        P("There is an exact shortcut. For an equal-weighted portfolio, the variance of the whole is the sum of individual variances plus all the cross-covariances. If every pairwise correlation is taken to be the same value, the cross terms collapse into a closed form that rearranges to give that correlation directly. Every term is a rolling mean, variance or sum."),
        ...Code([
          "rho = ( N^2 * Var_portfolio  -  sum(sigma_i^2) )",
          "      -----------------------------------------------",
          "      (  (sum sigma_i)^2     -  sum(sigma_i^2) )",
        ]),
        Callout("Validated, not assumed", [
          "Against the direct O(N-squared) computation on the real panel: correlation of 0.9937, mean absolute deviation 0.011, maximum deviation 0.046, and 28 times faster.",
          "The slow reference implementation ships alongside the fast one and is used only in tests, so the claim can be re-checked at any time rather than trusted.",
        ], "good"),

        H2("4.3 The live-universe rule"),
        P("Every cross-sectional measure is computed only over symbols that actually existed and traded that day, using the listing mask from Phase 1. Otherwise “40% of stocks are falling” silently becomes “40% of stocks are falling, out of a denominator including companies that had not yet listed”, which is a different and meaningless quantity."),
        Break(),
      );

      // ---------------------------------------------------------- 5
      c.push(
        H1("5. Findings"),
        H2("5.1 The market-wide trigger, calibrated rather than guessed"),
        P("With no tuning whatsoever, the fraction of the universe in AcceleratingDecline isolates genuine market events:"),
        Tbl(["Date", "Universe declining", "Median slope", "Event"], [
          ["2020-03-12", "92.1%", "-2.11", "COVID crash"],
          ["2026-03-31", "90.5%", "-0.79", "—"],
          ["2016-02-11", "90.1%", "-0.94", "Feb-2016 global selloff"],
          ["2022-09-26", "87.2%", "-0.67", "Fed / GBP crisis"],
          ["2021-12-20", "86.2%", "-0.82", "Omicron"],
          ["2017-09-25", "84.5%", "-0.72", "—"],
        ], [1800, 2400, 2200, 2600]),
        P("Breadth alone is not enough. Around ninety percent of the universe was falling during both the COVID crash and ordinary pullbacks. Only the median slope separates them: minus 2.11 sigma against roughly minus 0.8. Hence a two-condition trigger, breadth at or above 75 percent AND median slope at or below minus 1.5 sigma — a threshold derived from measurement rather than chosen."),

        H2("5.2 Signal strength, measured out of sample"),
        P("Fitted on 2016-2020, tested on 2021-2026, against a label of forward five-day drawdown at or below minus five percent:"),
        Tbl(["Rule", "Days", "P(crash)", "Lift"], [
          ["base rate", "—", "11.0%", "1.00x"],
          ["AcceleratingDecline alone, no model", "38,612", "12.0%", "1.10x"],
          ["slope_z below -1 alone, no model", "3,451", "13.2%", "1.20x"],
          ["intensity 0.95+ and AcceleratingDecline", "875", "20.7%", "1.88x"],
          ["intensity 0.99+ and AcceleratingDecline", "107", "43.0%", "3.91x"],
        ], [3600, 1600, 1800, 2000]),
        Callout("Isolation Forest is doing the work", [
          "The trend rules alone are worth almost nothing — 1.10x and 1.20x against a base rate of 1.00x.",
          "The anomaly score carries the signal; slope and acceleration convert \"unusual\" into \"unusually bad\".",
          "The rally side tells the same story. AcceleratingAdvance alone scores 0.98x — literally no information. Combined with intensity above 0.99 it reaches 2.44x.",
        ], "good"),

        H2("5.3 A warning that must not be buried"),
        P("Ranked across all days, the detector achieves an area under the curve of 0.53 against the crash label — barely above a coin flip. Some of that is expected and unfair to the model: Isolation Forest is direction-blind, so it ranks violent rallies exactly as highly as violent crashes, and a crash-only label penalises it for doing its job. Applying the direction filter is what lifts precision to 3.91 times base rate."),
        P("But a separate probe on the market-level features gave a harder result. Trained on all history through 2020, the detector fired zero alerts across 1,344 out-of-sample days — never once in five years. Volatility purging restored firing to roughly three per year, yet it still caught only one of eight crash onsets."),
        Callout("Read this honestly", [
          "This is a warning sign, not a verdict. The probe used only the six market features rather than the 468,266 rows of per-stock features, and eight out-of-sample events cannot support a confident conclusion.",
          "But it is real enough that Phase 4 — measuring actual lead time — must come before any backtest, dashboard, or retraining comparison is built on top of it.",
          "Far better to learn this from a lead-time histogram in Phase 4 than from an equity curve in Phase 8.",
        ], "warn"),

        H2("5.4 A retraining hypothesis, confirmed early"),
        P("The zero-alert result is not a bug. Isolation Forest defines normal as whatever it was trained on, and the training set contained 2008 and March 2020. Having been shown catastrophe, the model learned that catastrophe is normal, and no subsequent day was extreme enough to clear the threshold."),
        Tbl(["Training set", "Alerts in 1,344 out-of-sample days"], [
          ["All history 2006-2020", "0"],
          ["Crisis periods removed", "5"],
          ["Volatility-purged, top decile dropped", "15"],
          ["2016-2020 including COVID", "1"],
          ["2016-2020 volatility-purged", "42"],
        ], [5000, 4000]),
        P("This was proposed in the implementation plan as a fourth challenger to the three retraining methods. It is no longer a theoretical argument — it is measured, and it should now be treated as the default rather than the alternative."),
        Break(),
      );

      // ---------------------------------------------------------- 6
      c.push(
        H1("6. Considered and rejected: Bollinger Bands"),
        P("Bollinger Bands were evaluated both as a crash label and as a feature. Recording the reasoning because the conclusion is not obvious and the investigation produced a genuinely useful result."),

        H2("6.1 As a crash label — rejected"),
        P("Band position is volatility-relative, so a sustained crash widens its own band. The reading becomes LESS extreme as the drawdown deepens:"),
        Tbl(["Date", "Drawdown from peak", "Band position"], [
          ["2020-02-28", "-9%", "-2.85 SD"],
          ["2020-03-12", "-22%", "-2.70 SD"],
          ["2020-03-23", "-38%", "-2.01 SD"],
        ], [2600, 3200, 3200]),
        P("The band half-width grew from 3.1 percent of price to 32.1 percent. During the worst crash in the dataset, a 2.5-sigma band flagged three days out of thirty, and at the minus 38 percent bottom price sat comfortably inside it. The relationship runs backwards for exactly the events that matter most, and unlike the slope_z case there is no fix available — the band IS the normaliser."),
        P("To be fair to the method, it is not noise. Measured contemporaneously against sharp drops it reaches 33 percent precision at 2.5 sigma, and 70 percent recall at 1.5 sigma. It is a legitimate coincident detector of sudden drops. It fails specifically on SUSTAINED crashes, which is the failure mode a drawdown-focused strategy can least afford."),

        H2("6.2 As a feature — rejected"),
        P("Measured incremental value on the hold-out period:"),
        Tbl(["Feature set", "AUC", "Precision at top 5%"], [
          ["baseline, 10 features", "0.5300", "20.1%"],
          ["plus band position", "0.5300", "19.2%"],
          ["plus bandwidth", "0.5324", "21.4%"],
          ["plus both", "0.5348", "21.5%"],
        ], [3600, 2400, 3000]),
        P("Band position contributes exactly nothing on its own and slightly lowers precision. It correlates 0.69 with the downside asymmetry measure and 0.68 with slope, so it is largely a restatement of information already present. Bandwidth adds a marginal amount. Neither was judged to earn its place, and both were left out."),

        H2("6.3 What the investigation did settle"),
        P("The crash label stays defined on ABSOLUTE magnitude, never on band position. Drawdown is what hurts a portfolio, and it is absolute: losing thirty percent hurts identically whether it happened in a calm year or a wild one. Measured base rates on the index:"),
        Tbl(["Forward 5-day drawdown worse than", "Share of days"], [
          ["-3%", "14.49%"],
          ["-5%", "5.44%"],
          ["-6%", "3.27%"],
          ["-10%", "0.85%"],
        ], [5000, 4000]),
        P("A minus five percent threshold occurs on 5.44 percent of days, which happens to align almost exactly with the five percent anomaly rate conventionally quoted for this kind of model. That agreement is a useful cross-check on both numbers."),
        Break(),
      );

      // ---------------------------------------------------------- 7
      c.push(
        H1("7. A note on contamination"),
        P("Isolation Forest exposes a contamination parameter, usually described as the expected proportion of anomalies. It is natural to reach for it and set it to five percent."),
        P("Measured across values from 0.001 to 0.2, the raw anomaly scores are bit-for-bit identical. Contamination does not affect tree building at all. It sets only an internal offset used to convert scores into a binary in-or-out label."),
        RichP([
          { text: "This pipeline never uses that binary label. Intensity is the raw score mapped through the " },
          { text: "training-set empirical distribution", bold: true },
          { text: ", so contamination never enters the calculation. That is deliberate: setting contamination amounts to asserting how much of history was a crash, which is a guess you would then be scored against. A measured percentile replaces the guess." },
        ]),
        P("The five percent intuition is still useful — it just belongs to the intensity THRESHOLD rather than to the model. An intensity cut at 0.95 means acting on the most unusual five percent of days; a cut at 0.99, the most unusual one percent."),

        H1("8. Tests"),
        P("71 passing. What matters is what they prove."),
        Tbl(["Group", "Proves"], [
          ["Future perturbation", "Rewriting the future leaves every earlier feature bit-identical"],
          ["Streaming equals batch", "Day-by-day computation reproduces the batch result — the live-trading contract"],
          ["No same-day return", "ret_1d cannot leak into the model"],
          ["Correlation shortcut", "The O(N) form tracks the direct O(N-squared) computation"],
          ["Flat-bar masking", "Zero-range bars are excluded from range features"],
          ["Asymmetry defined", "The bounded form survives a window with no up days"],
          ["Volatility transition", "vol_ratio spikes at a regime change"],
          ["Listing mask", "Breadth counts only companies that existed"],
          ["No infinities", "A single infinity would make every finite value look identical"],
        ], [2900, 6100]),
        Break(),
      );

      // ---------------------------------------------------------- 8
      c.push(
        H1("9. Results"),
        ...Code([
          "per-stock: 468,266 symbol-days x 10 model features",
          "",
          "  vol_ratio           99.8% non-null",
          "  vol_of_vol          99.7% non-null",
          "  semidev_asym        99.9% non-null",
          "  volume_z            99.9% non-null",
          "  range_expansion     98.6% non-null",
          "  gap_freq            99.9% non-null",
          "  illiquidity         99.9% non-null",
          "  dd_from_high        99.9% non-null",
          "  slope_z             99.7% non-null",
          "  accel_z             99.7% non-null",
          "",
          "cross-sectional (2016 onward), median / p1 / p99:",
          "  breadth_decline    19.10    1.04   77.95",
          "  median_slope_z      0.04   -0.78    0.64",
          "  dispersion          1.64    0.94    3.39",
          "  avg_corr            0.24    0.11    0.48",
          "  pct_below_ma50     39.29    3.16   92.13",
        ]),

        H1("10. Known limitations"),
        Num("Predictive power is modest and unproven. The strongest rule reaches 3.91 times base rate, on 107 observations that are heavily overlapping — the same market event appears across many correlated stocks on the same day, so the effective independent sample is far smaller than the count suggests."),
        Num("The leading-feature thesis is not yet demonstrated. Features were chosen because there are good reasons they should lead. Whether they actually do is what Phase 4 measures, and the market-level probe gives grounds for caution."),
        Num("Pre-crash markets look calmer and stronger than average. Median slope is positive ten days before an onset. Crashes here begin from strength, not from visible weakness, which complicates the premise that stress accumulates visibly beforehand."),
        Num("The correlation shortcut assumes homogeneous correlations and roughly stable membership within each window. It tracks the true average closely, but it is an approximation."),
        Num("Feature parameters — 5, 20 and 60-day windows, the 20-day volatility lag — are conventional choices, not optimised. Optimising them against the hold-out would be a form of look-ahead."),

        H1("11. Handoff to Phase 3"),
        P("Phase 3 fits Isolation Forest and converts its scores into intensity. It inherits a long-format table of 468,266 symbol-days by 10 features, a daily market-state block, and a settled crash definition based on absolute magnitude."),
        P("Three decisions carry forward. The model is fitted on the POOLED cross-section rather than per symbol, so one model learns what normal looks like across the whole universe. Intensity is the percentile of the score against the TRAINING distribution, which is what makes the four retraining schemes comparable. And max_samples, not contamination, is the parameter that actually needs tuning."),
        Callout("Phase 4 is the decision point", [
          "Phase 3 produces intensity. Phase 4 measures whether it arrives early enough to be worth anything.",
          "No backtest, no retraining comparison, and no dashboard should be built until that lead-time histogram exists.",
        ], "warn"),
      );

      return c;
    },
  },

  phase3: {
    slug: "Phase3_Isolation_Forest",
    docTitle: "QBEAST Phase 3 — Isolation Forest and Anomaly Intensity",
    runningHead: "QBEAST · Phase 3 — Isolation Forest",
    build(h) {
      const { H1, H2, H3, P, RichP, Bullet, Num, Code, Callout, Tbl, Rule, Break,
              cover, TableOfContents } = h;
      const c = [];

      c.push(...cover({
        phase: "PHASE 3",
        title: "Isolation Forest & Anomaly Intensity",
        subtitle: "Fitting the detector, and turning its scores into a scale that means something",
        status: "STATUS: COMPLETE — 86 tests passing",
        meta: [
          "One model fitted on 106,157 pooled symbol-days",
          "328 High-band alerts across 128,737 out-of-sample symbol-days",
          "Strongest signal: 38.4% crash rate against an 11.0% base rate (3.48×)",
        ],
      }));

      c.push(H1("Contents"),
        new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
        Break());

      // ------------------------------------------------------- 1
      c.push(
        H1("1. How Isolation Forest works"),
        P("Most anomaly detectors work by building a model of what is normal and then measuring how far each point sits from it. Isolation Forest does something stranger and much cheaper: it never models normality at all."),
        P("It builds many random trees. At each step it picks a feature at random, picks a split value at random within that feature's range, and cuts the data in two. It repeats until every point sits alone in its own leaf, and then counts how many cuts each point needed."),
        Callout("The insight", [
          "Outliers are isolated quickly. A point sitting far from everything else gets separated after a handful of random cuts, because almost any cut you make happens to fall between it and the crowd.",
          "A point buried inside a dense cluster needs many cuts before it is alone, because most random cuts land elsewhere.",
          "So the average number of cuts — the path length — is itself the anomaly measure. Short path means anomalous. Nothing else has to be computed.",
        ]),
        P("This is why the method is fast, why it needs no assumption about the shape of the data, and why it scales comfortably to 468,266 rows."),

        H2("1.1 Unsupervised, and what that costs"),
        P("The model is never shown a labelled crash. It has no notion of good or bad, only of usual and unusual. That is a genuine advantage: crashes are rare, and a supervised model trained on the handful of crashes in Indian market history would have almost nothing to learn from."),
        P("But it has a cost that shapes every downstream decision, and it is worth stating bluntly."),
        Callout("Isolation Forest is direction-blind", [
          "It flags UNUSUAL, not UNUSUALLY BAD. A violent rally is exactly as anomalous as a violent crash, because both are equally far from an ordinary Tuesday.",
          "So the anomaly score alone can never produce a buy or a sell signal. It can only say that something is happening.",
          "Direction comes from slope. Building-versus-exhausting comes from acceleration. That is the entire reason those features exist, and why the architecture has three parts rather than one.",
        ]),
        Break(),
      );

      // ------------------------------------------------------- 2
      c.push(
        H1("2. Anomaly intensity"),
        H2("2.1 The problem with the raw score"),
        P("The score that comes out of the model is not comparable between one fitted model and another. Its scale shifts with the size of the training window, with how many trees were built, and with which random cuts those trees happened to make."),
        P("That would not matter if we only ever fitted one model. But Phase 7 compares four retraining schemes against each other, and a fixed cutoff on the raw score would mean four different things under four schemes. The comparison would be measuring the scoring scale rather than the schemes."),

        H2("2.2 The fix: percentile against the training distribution"),
        ...Code([
          "intensity(x) = fraction of TRAINING rows that scored lower than x",
          "",
          "  intensity = 0.99   more unusual than 99% of training days",
          "  intensity = 0.50   a perfectly ordinary day",
        ]),
        P("Three properties follow, and all three are needed:"),
        Num("It means the same thing under every fit, which is what makes the retraining comparison valid."),
        Num("It means the same thing for every stock, so one global threshold serves all 96 symbols rather than 96 hand-tuned ones."),
        Num("It is directly interpretable as an alert budget. A 0.99 cut implies roughly one alert per hundred days, before any direction filter is applied."),
        P("A test asserts this property directly: two detectors with different settings must each select about one percent of their own training data at intensity 0.99 or above. A raw-score threshold would not."),

        H2("2.3 Bands"),
        Tbl(["Band", "Intensity", "Intended use"], [
          ["High", "0.99 and above", "Act, if slope and acceleration confirm"],
          ["Moderate", "0.95 to 0.99", "Do not act — watch for 3 to 5 days and see how it resolves"],
          ["Low", "0.90 to 0.95", "Record only"],
        ], [1500, 2300, 5200]),
        Break(),
      );

      // ------------------------------------------------------- 3
      c.push(
        H1("3. Design decisions"),

        H2("3.1 One pooled model, not one per stock"),
        P("The detector is fitted on all 96 symbols together rather than separately per symbol."),
        P("A per-symbol model would see only a few hundred training rows each. Worse, it would learn each stock's own quiet periods as that stock's normal — so a permanently turbulent name would have a high bar for anomaly and a placid one a low bar. The scores would no longer be comparable across the universe, which destroys the entire point of a cross-sectional signal: choosing which stocks to act on requires that a 0.99 for one stock means what a 0.99 means for another."),

        H2("3.2 Missing rows are dropped, never imputed"),
        P("Isolation Forest cannot consume missing values. The usual response is to fill them with a column mean, and here that would be actively harmful."),
        P("Every imputed row would carry the same fabricated values. The model would see a large, perfectly dense cluster of identical points and learn that this configuration is extremely normal — the single most normal thing in the dataset. Incomplete rows are therefore excluded from fitting and score as missing, with the row itself preserved so nothing downstream silently shifts."),

        H2("3.3 What is persisted, and why both halves matter"),
        P("Saving the fitted forest alone would be useless, because the forest cannot produce intensity by itself — the scale lives in the training score distribution. Both are stored together."),
        P("There is a second reason. Storing the training distribution alongside the model means a later robustness experiment can be re-run without repeating the backtest, which turns a day of computation into a minute of it."),
        Break(),
      );

      // ------------------------------------------------------- 4
      c.push(
        H1("4. A parameter that does not do what its name suggests"),
        RichP([
          { text: "Isolation Forest exposes a " }, { text: "contamination", code: true },
          { text: " parameter, usually documented as the expected proportion of anomalies in the data. It is entirely natural to reach for it and set it to five percent, and much of the literature suggests exactly that." },
        ]),
        P("Measured across every value from 0.001 to 0.2, the raw anomaly scores are bit-for-bit identical."),
        Tbl(["contamination", "scores identical?", "internal offset", "predict() flags"], [
          ["0.001", "yes", "-0.6012", "0.1%"],
          ["0.01", "yes", "-0.5644", "1.0%"],
          ["0.05", "yes", "-0.5242", "5.0%"],
          ["0.20", "yes", "-0.4800", "20.0%"],
        ], [2400, 2400, 2200, 2000]),
        P("Contamination does not affect tree building at all. It sets only an internal offset used to convert scores into a binary in-or-out label. It is a labelling knob, not a learning one."),
        RichP([
          { text: "This pipeline never uses that binary label, so contamination never enters the calculation. That is deliberate. Setting contamination amounts to " },
          { text: "asserting", italics: true },
          { text: " what fraction of history was a crash — and then being scored against your own assertion. The training-percentile mapping replaces the assumption with a measurement." },
        ]),
        P("The five percent intuition is still sound. It simply belongs to the intensity THRESHOLD rather than to the model. Cutting at 0.95 means acting on the most unusual five percent of days."),
        Callout("A pleasing coincidence", [
          "Measured on the index, a forward five-day drawdown of five percent or worse occurs on 5.44% of days.",
          "So the conventional five percent figure and an absolute five percent crash definition agree almost exactly — arrived at from completely different directions. That agreement is a useful cross-check on both numbers.",
        ], "good"),
        Break(),
      );

      // ------------------------------------------------------- 5
      c.push(
        H1("5. Two corrections to earlier claims"),
        P("Both were stated confidently in earlier phases and both turned out to be wrong when measured. They are recorded here rather than quietly fixed, because the reasoning matters more than the conclusion."),

        H2("5.1 max_samples is not the parameter that matters"),
        P("Earlier documentation described max_samples — the subsample each tree isolates from — as the parameter that genuinely needed tuning, in contrast to contamination. Measured across a 64-fold range:"),
        Tbl(["max_samples", "signals", "P(crash)", "lift"], [
          ["64", "445", "24.0%", "2.18x"],
          ["128", "411", "24.6%", "2.23x"],
          ["256", "444", "24.5%", "2.22x"],
          ["512", "429", "24.9%", "2.26x"],
          ["1024", "420", "24.5%", "2.22x"],
          ["4096", "399", "25.3%", "2.29x"],
        ], [2200, 2000, 2400, 2400]),
        P("Lift moves from 2.18 to 2.29 across the whole range. It barely matters on this data. The default of 256 is kept, now on evidence rather than on assertion."),

        H2("5.2 Volatility purging is not the right baseline"),
        P("Phase 2 found something striking: a detector trained on all market history through 2020 fired ZERO alerts across 1,344 out-of-sample days. Having been shown 2008 and March 2020, it had learned that catastrophe is ordinary. Volatility purging — withholding the most turbulent days from training — restored firing, and the Phase 2 documentation concluded that purging should become the default."),
        P("On the per-stock model that conclusion is wrong. Measured on 2021 to 2026, signal defined as intensity 0.99 or above combined with AcceleratingDecline, against a base crash rate of 11.0%:"),
        Tbl(["Purge quantile", "Training rows", "Signals", "P(crash)", "Lift"], [
          ["none", "106,157", "125", "38.4%", "3.48x"],
          ["0.99", "105,006", "205", "29.3%", "2.65x"],
          ["0.95", "100,726", "420", "24.8%", "2.24x"],
          ["0.90", "95,329", "444", "24.5%", "2.22x"],
          ["0.75", "79,336", "645", "20.8%", "1.88x"],
        ], [1900, 2100, 1500, 1800, 1700]),
        Callout("Why the earlier finding did not transfer", [
          "The zero-alert result came from MARKET-level features — six series — trained across 2006 to 2020, a window containing both the global financial crisis and COVID. In that setting a couple of crises really can dominate the training distribution.",
          "The per-stock model pools 96 symbols over 2016 to 2020: 106,157 rows, most of which are ordinary days for ordinary stocks even during a crisis. No single episode dominates, so the silence problem never arises.",
          "Purging then only removes useful information. The relationship is monotone — it buys coverage at the cost of precision.",
        ], "warn"),
        P("Purging reverts to what it originally was: one of four retraining variants to be compared in Phase 7, not the baseline. The lesson generalises — a finding measured on one representation of the data does not automatically transfer to another."),
        Break(),
      );

      // ------------------------------------------------------- 6
      c.push(
        H1("6. Results"),
        ...Code([
          "training window 2016-01-01 to 2020-12-31: 108,008 symbol-days",
          "volatility purge: disabled (baseline)",
          "",
          "fitted on 106,157 complete rows",
          "",
          "out-of-sample (2021-01-01 onward): 128,737 symbol-days",
          "  High          328  ( 0.25%)",
          "  Moderate    2,432  ( 1.89%)",
          "  Low         4,760  ( 3.70%)",
          "",
          "  High-band alerts: 0.6 per symbol per year",
          "  ...of which AcceleratingDecline: 130 (40%)",
        ]),
        P("Roughly forty percent of High-band alerts are accompanied by an accelerating decline. The remainder are the direction-blindness in action: violent rallies and other unusual configurations that are genuinely anomalous but not bearish."),

        H2("6.1 Signal strength"),
        Tbl(["Rule", "Days", "P(crash)", "Lift"], [
          ["base rate", "—", "11.0%", "1.00x"],
          ["AcceleratingDecline alone, no model", "38,612", "12.0%", "1.10x"],
          ["slope_z below -1 alone, no model", "3,451", "13.2%", "1.20x"],
          ["intensity 0.95+ with AcceleratingDecline", "875", "20.7%", "1.88x"],
          ["intensity 0.99+ with AcceleratingDecline", "125", "38.4%", "3.48x"],
        ], [3800, 1500, 1800, 1900]),
        P("The trend rules alone are worth almost nothing — 1.10 and 1.20 against a baseline of 1.00. The anomaly score carries the signal; slope and acceleration convert unusual into unusually bad. Neither half works without the other."),
        Break(),
      );

      // ------------------------------------------------------- 7
      c.push(
        H1("7. Tests"),
        P("86 passing. The ones specific to this phase:"),
        Tbl(["Test", "Proves"], [
          ["Intensity is a training percentile", "On its own training data, intensity is uniform on [0,1]"],
          ["Threshold means the same under every fit", "Different settings each select ~1% at 0.99 — the property Phase 7 depends on"],
          ["Outliers score higher", "Shifted rows score well above the bulk"],
          ["Incomplete rows score missing", "Never imputed; the index is preserved"],
          ["Save/load round trip", "Reloaded detector reproduces intensity exactly"],
          ["Determinism", "Same seed gives identical scores"],
          ["Purge selects by date", "Crises are a property of a day, not of a stock"],
        ], [3400, 5600]),
        RichP([
          { text: "The purge-by-date test deserves a note. Purging individual high-volatility ROWS would silently exclude permanently volatile names such as ADANIENT from training altogether, so the model would never learn what normal looks like for them. A crisis is a property of a " },
          { text: "day", italics: true },
          { text: ", not of a stock, and the implementation reflects that." },
        ]),

        H1("8. Known limitations"),
        Num("The sample is smaller than it looks. 125 signals sounds substantial, but one market event hits many correlated stocks on the same day, so the effective number of independent episodes is far lower."),
        Num("Lift is measured against a coincident label. P(crash) here means the probability a crash follows within five days. It is genuinely forward-looking, but it is not the same as lead time, which Phase 4 measures."),
        Num("Ranked across all days the detector reaches an area under the curve of only 0.53. Much of that is unfair to a direction-blind model being scored on a crash-only label, but it is not nothing."),
        Num("Thresholds of 0.99, 0.95 and 0.90 are conventional round numbers. They have not been optimised, and optimising them against the hold-out would be a form of look-ahead."),
        Num("A single training window is used. Whether the detector degrades as the market drifts away from 2016-2020 is exactly what Phase 7 exists to measure."),

        H1("9. Handoff to Phase 4"),
        P("Phase 4 inherits an intensity series for every symbol-day, a settled crash definition based on absolute magnitude, and a signal rule combining intensity with trend phase."),
        P("Its job is the question this project actually turns on: not whether the signal fires on crash days, but how many days BEFORE the crash it fires."),
        Callout("Why Phase 4 comes before the backtest", [
          "Everything measured so far says the signal fires when crashes are near. None of it says how early.",
          "If the answer is zero days, the strategy is a fast-reaction system rather than a predictive one — still useful, but a different claim, a different paper, and a different set of expectations.",
          "Learning that from a lead-time histogram now costs an afternoon. Learning it from an equity curve in Phase 8 would cost weeks of work built on a premise that does not hold.",
        ], "warn"),
      );

      return c;
    },
  },

  phase4: {
    slug: "Phase4_Lead_Time",
    docTitle: "QBEAST Phase 4 - Labels and Lead Time",
    runningHead: "QBEAST \u00b7 Phase 4 - Lead Time",
    build(h) {
      const { H1, H2, H3, P, RichP, Bullet, Num, Code, Callout, Tbl, Rule, Break,
              cover, TableOfContents } = h;
      const c = [];

      c.push(...cover({
        phase: "PHASE 4",
        title: "Labels & Lead Time",
        subtitle: "The go/no-go gate: does the signal arrive early enough to be worth anything?",
        status: "STATUS: COMPLETE - the answer changes what this project can claim",
        meta: [
          "705 crash onsets out of sample, measured against a random baseline",
          "Early-warning skill: 0.88x - no better than chance",
          "Short-horizon lift: 110x at one day",
        ],
      }));

      c.push(H1("Contents"),
        new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
        Break());

      c.push(
        H1("1. Why this phase came before the backtest"),
        P("Every measurement so far showed that the signal fires when crashes are near. None showed how EARLY. Those are different questions, and only the second one matches the requirement."),
        P("The phase was deliberately placed ahead of the backtest, the retraining comparison and the dashboard. If the answer had come out badly after those were built, weeks of work would have been resting on a premise that does not hold. Learning it from a histogram costs an afternoon."),
        Callout("The label never touches training", [
          "Nothing in this module is imported by the model, and no threshold defined here influences fitting.",
          "Isolation Forest is unsupervised. If the crash definition were derived from the model, the model would be defining the event it is then scored on detecting -- the evaluation would always look good and would mean nothing.",
        ]),
        Break(),
      );

      c.push(
        H1("2. Two calibration errors, found before the result"),
        H2("2.1 The threshold was calibrated on the wrong thing"),
        P("A forward five-day drawdown of five percent is genuinely rare for the index -- 5.44% of days. Applied to individual stocks, which are roughly twice as volatile, it is an ordinary pullback."),
        Tbl(["Threshold", "Onsets 2021-2026", "Per stock per year"], [
          ["-5%", "3,950", "7.6   (ordinary pullbacks)"],
          ["-8%", "1,479", "2.9"],
          ["-10%", "705", "1.4   (adopted)"],
          ["-12%", "350", "0.7"],
          ["-20%", "36", "0.1"],
        ], [2200, 3000, 3800]),
        P("At -5% the event count is inflated sevenfold and every recall figure computed against it is meaningless. Stocks now use -10%; the index keeps -5%."),

        H2("2.2 Raw recall is not evidence of skill"),
        P("Crash events are frequent enough that any rule firing often will land near one by chance. A recall figure on its own therefore says nothing at all. Every rule is now measured against a RANDOM signal firing at exactly the same rate, on the same calendar."),
        P("This is the single most transferable habit in the project: before believing any number, ask what a deliberately stupid baseline would score."),
        Break(),
      );

      c.push(
        H1("3. The result"),
        H2("3.1 Early warning: none"),
        Tbl(["Rule", "Signals", "Recall", "Random", "Skill"], [
          ["intensity 0.99+ and AccelDecline", "130", "2.2%", "2.5%", "0.88x"],
          ["intensity 0.95+ and AccelDecline", "958", "8.0%", "14.5%", "0.55x"],
          ["intensity 0.99+ (no direction)", "326", "4.8%", "5.9%", "0.80x"],
        ], [3200, 1400, 1400, 1400, 1600]),
        P("Every rule sits at or below 1.00x. At a fifteen-day horizon the detector provides no early warning beyond chance. The requirement to predict crashes two to three days ahead, in the sense of forecasting from a calm market, is not met."),

        H2("3.2 Short-horizon information: a great deal"),
        Tbl(["Horizon", "Base rate", "Given a signal", "Lift"], [
          ["1 day", "0.06%", "6.92%", "110.0x"],
          ["2 days", "0.23%", "13.08%", "57.9x"],
          ["3 days", "0.57%", "20.00%", "35.1x"],
          ["5 days", "1.48%", "23.08%", "15.6x"],
          ["10 days", "4.82%", "26.15%", "5.4x"],
        ], [2000, 2200, 2400, 2400]),
        P("Probability of a ten percent drawdown within H days. The lift decays sharply with horizon, which is the signature of a coincident detector rather than a predictive one."),
        P("Confirmed directly: the median same-day return on signal days is -1.14%, against +0.02% across all days. The signal fires as a decline BEGINS, not before it."),

        H2("3.3 Reconciling with Phase 3"),
        RichP([
          { text: "Phase 3 reported 3.48x lift and this phase reports no skill. Both are correct, because they measure different quantities. Phase 3 measured P(crash given a signal) -- " },
          { text: "precision", bold: true },
          { text: ", which is high. This phase measures P(signal given a crash) -- " },
          { text: "recall", bold: true },
          { text: ", which is low. The signal is precise but rare, firing on 130 of 128,737 symbol-days, so it can only ever cover a fraction of 705 events." },
        ]),
        P("Confusing precision with recall is the most common mistake in applied machine learning, and this is a clean example of why it matters: the same detector looks excellent under one and useless under the other."),
        Break(),
      );

      c.push(
        H1("4. What it means"),
        P("The honest framing is fast reaction rather than prediction: recognise within a day that a decline has started, and leave before it deepens. A 110-fold edge at one day is exactly what an exit rule needs."),
        P("This is a weaker claim than the original requirement, and it is the one the evidence supports. It is also enough. Drawdown reduction was always the primary objective, and reducing drawdown does not require forecasting from a calm market -- it requires reacting faster than the decline completes."),
        Callout("For the research paper", [
          "\u201cUnsupervised anomaly detection reduces drawdown through rapid exit\u201d is honest, measurable, and defensible.",
          "\u201cPredicts crashes three days ahead\u201d would not survive review, and now there is a measurement showing exactly why.",
          "Reporting the negative result alongside the positive one is a strength, not a weakness. It is evidence the evaluation was designed to be capable of failing.",
        ], "good"),

        H1("5. Known limitations"),
        Num("The random baseline assumes signals could have fallen anywhere. In reality they cluster in volatile periods, which are also when crashes occur, so the true baseline is arguably even higher and the measured skill even weaker."),
        Num("A fifteen-day lookback was chosen as the search window. A longer one would find more coincidental hits, not more genuine warnings."),
        Num("Events overlap across correlated stocks: one market episode produces many symbol-level events on the same day, so the effective independent sample is far smaller than 705."),
        Num("The -10% threshold is a judgement, though the sensitivity table shows the conclusion is not delicate."),
      );

      return c;
    },
  },

  phase5: {
    slug: "Phase5_Signals",
    docTitle: "QBEAST Phase 5 - Signal Generation",
    runningHead: "QBEAST \u00b7 Phase 5 - Signals",
    build(h) {
      const { H1, H2, H3, P, RichP, Bullet, Num, Code, Callout, Tbl, Rule, Break,
              cover, TableOfContents } = h;
      const c = [];

      c.push(...cover({
        phase: "PHASE 5",
        title: "Signal Generation",
        subtitle: "From anomaly score to position, under the fast-reaction framing",
        status: "STATUS: COMPLETE - 125 tests passing",
        meta: [
          "Hold/cash state machine with whipsaw guards",
          "2020 stress test: 7.7 points of drawdown saved, with higher return",
          "2021-2026: 0.4 trades per symbol per year - correctly quiet",
        ],
      }));

      c.push(H1("Contents"),
        new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
        Break());

      c.push(
        H1("1. What this phase decides"),
        P("Intensity says today is unusual and phase says the move is downward. Neither is a position. Something has to decide when to leave, when to come back, and how to avoid doing either too often."),
        P("Long is the default state. The model is not asked to pick stocks or time the market continuously -- only to step out when a decline begins and step back in afterwards."),

        H2("1.1 Re-entry is the hard half"),
        P("Exiting is comparatively easy: the exit rule carries a hundredfold edge at a one-day horizon. Coming back is the difficulty. Crashes are followed by recoveries, and a strategy that exits well but re-enters late loses more to the missed rebound than it ever saved on the decline. That failure mode is why most crash-avoidance strategies underperform buy-and-hold despite being right about the crash."),
        P("Four re-entry rules were therefore implemented and compared rather than one being assumed: on a confirmed rally, when the decline stops accelerating, after a fixed number of sessions, and as soon as the phase is no longer an accelerating decline."),
        Callout("The comparison was less decisive than expected", [
          "All four land within 0.7 percentage points of CAGR and a fraction of a point of drawdown.",
          "That is itself worth knowing: the re-entry rule matters far less than the effort spent choosing it would suggest, because the whipsaw guards -- cooldown, minimum hold, and a hard cap on time in cash -- dominate the outcome.",
        ]),

        H2("1.2 Whipsaw guards are not polish"),
        P("Every round trip pays brokerage, securities transaction tax, stamp duty, GST and slippage, and converts a long-term holding into a short-term one for tax. A rule that is right slightly more often than it is wrong can still lose money if it trades enough. The guards are load-bearing."),
        Break(),
      );

      c.push(
        H1("2. The headline result"),
        H2("2.1 On the 2021-2026 window, almost nothing happens"),
        Tbl(["Re-entry rule", "Trades/sym/yr", "Days in cash", "CAGR", "B&H CAGR", "maxDD", "B&H maxDD"], [
          ["rally_signal", "0.13", "0.7%", "26.12%", "25.64%", "-19.9%", "-20.0%"],
          ["decel", "0.14", "0.5%", "25.85%", "25.64%", "-19.9%", "-20.0%"],
          ["time", "0.14", "0.2%", "25.94%", "25.64%", "-19.9%", "-20.0%"],
          ["not_declining", "0.14", "0.2%", "26.00%", "25.64%", "-19.9%", "-20.0%"],
        ], [1900, 1500, 1300, 1200, 1300, 900, 900]),
        P("Roughly one trade per symbol every seven years, half a percent of days in cash, and drawdown within a tenth of a point of simply holding. Lowering the exit threshold from 0.99 to 0.70 does not change this -- it only trades more and earns less, with CAGR falling from 20.9% to 17.7% in an earlier sweep."),

        H2("2.2 The reason is the window, not the model"),
        Tbl(["", "2020-01 to 2020-03", "2024-09 to 2025-02"], [
          ["Duration", "67 days", "154 days"],
          ["Worst single day", "-8.3%", "-3.2%"],
          ["Days beyond 3%", "8", "1"],
          ["Annualised volatility", "38.5%", "16.5%"],
          ["Drawdown", "-37.8%", "-21.2%"],
        ], [3000, 3000, 3000]),
        P("The worst drawdown of the backtest window was a slow bleed at near-normal volatility with a single day beyond three percent. There is nothing in it for an anomaly detector to fire on. The system targets sharp declines, and this window contains none -- so it correctly does almost nothing."),
        P("A backtest window that lacks the risk a system is designed for cannot tell you whether the system works. It can only tell you the system does not misbehave in its absence, which is worth knowing but is a different question."),
        Break(),
      );

      c.push(
        H1("3. Stress test"),
        P("Training on 2016-2019 keeps COVID genuinely out of sample -- the model has never seen a crash of that magnitude when it is asked to react to one."),
        Tbl(["Window", "Strategy", "Buy & hold", "Strat DD", "B&H DD", "DD saved", "Trades"], [
          ["2020 Feb-Apr (the crash)", "-16.4%", "-21.1%", "-28.7%", "-36.6%", "+7.9pp", "10.30"],
          ["2020 full year", "+32.2%", "+24.6%", "-29.2%", "-36.8%", "+7.7pp", "3.17"],
          ["2021-2026 (no crash)", "+245.2%", "+244.9%", "-19.7%", "-20.0%", "+0.3pp", "0.40"],
        ], [2400, 1300, 1300, 1300, 1200, 1200, 1300]),
        Callout("What this shows", [
          "Nearly eight points of drawdown saved in the crash year, with HIGHER return, on a model that never saw COVID.",
          "In the quiet window it stays out of the way at 0.40 trades per symbol per year. The guards hold.",
          "Read as insurance: it costs very little in quiet years and pays back meaningfully when a crash arrives.",
        ], "good"),
        P("The caveat that must accompany every statement of this result: 2020 is ONE event. Nothing generalises from a sample of one. Extending the training window back to include 2008 would give a second crash to test against, and that is the obvious next step for anyone wanting confidence in the number."),
        Break(),
      );

      c.push(
        H1("4. Two bugs found by review"),
        H2("4.1 A double lag, caught by a test"),
        P("The state machine assigned the position for each bar BEFORE processing that bar's decision, which already implements next-day execution correctly. A further shift had been applied on top, delaying every trade by a second day."),
        P("No backtest would have complained. It would simply have reported worse numbers forever, and they would have looked plausible."),

        H2("4.2 A geometric mean masquerading as a portfolio"),
        P("Per-stock log equity curves were being averaged and exponentiated. That gives the geometric mean of individual stock outcomes, which is not a portfolio -- it silently discards the diversification benefit, since the mean of logarithms sits below the logarithm of the mean."),
        Tbl(["Method", "CAGR", "maxDD"], [
          ["Geometric mean of stocks (wrong)", "20.69%", "-21.2%"],
          ["Equal-weight, daily rebalanced (right)", "25.44%", "-20.0%"],
        ], [4400, 2300, 2300]),
        P("A 4.75 point understatement of CAGR. Correcting it also moved the headline stress-test figure from a claimed ten points of drawdown saved to a true 7.7 -- in other words, the error had been flattering the result in one place and penalising it in another."),
        Callout("Why both were worth finding", [
          "Neither produced an error, a warning, or an implausible number. Both simply produced slightly wrong answers that looked entirely reasonable.",
          "This is the recurring lesson of the project: in a numerical pipeline, the dangerous failures are the ones that stay quiet.",
        ], "warn"),
        Break(),
      );

      c.push(
        H1("5. The market-wide overlay"),
        P("A systemic de-risking flag fires when breadth reaches 75 percent and the median slope falls to -1.5 sigma. Both conditions are required, because breadth alone cannot separate a systemic crash from a broad pullback -- around ninety percent of stocks were declining on the COVID crash and on several ordinary pullbacks alike, and only the median slope distinguishes them."),
        P("Out of sample it fired zero times in 1,344 sessions. The threshold was calibrated on history including COVID, and 2021-2026 never came close."),
        P("That is arguably correct behaviour rather than a fault. A systemic-crash trigger SHOULD be rare, and a version tuned to fire in a period containing no systemic crash would be fitted to noise."),

        H1("6. Known limitations"),
        Num("The stress test rests on a single crash. Everything about the magnitude of the benefit is uncertain from n=1."),
        Num("All figures are gross of costs. Phase 6 adds the Indian cost and tax stack, though at 0.4 trades per symbol per year the impact should be small - itself a finding worth stating."),
        Num("The strategy is evaluated as an equal-weighted portfolio with no position sizing. Top-N selection by signal strength is implemented but not yet exercised."),
        Num("Re-entry rules differ by less than the noise in a single backtest window, so the choice between them is not well supported by evidence."),
        Num("The market overlay is untested, having never fired out of sample."),

        H1("7. Handoff to Phase 6"),
        P("Phase 6 adds the realistic cost and tax model and produces the definitive equity curves. It inherits the position panel, the equal-weight portfolio helper, and a settled framing."),
        P("The interesting question for Phase 6 is not whether costs erode the strategy -- at this trade frequency they cannot -- but how much of the drawdown benefit survives the tax treatment, since every crash exit converts a long-term holding into a short-term one and moves the rate from 12.5% to 20%."),
      );

      return c;
    },
  },

  phase6: {
    slug: "Phase6_Costs_and_Tax",
    docTitle: "QBEAST Phase 6 - Backtest with Costs and Tax",
    runningHead: "QBEAST \u00b7 Phase 6 - Costs and Tax",
    build(h) {
      const { H1, H2, H3, P, RichP, Bullet, Num, Code, Callout, Tbl, Rule, Break,
              cover, TableOfContents } = h;
      const c = [];

      c.push(...cover({
        phase: "PHASE 6",
        title: "Costs & Tax",
        subtitle: "What survives the NSE cost stack and Indian capital gains treatment",
        status: "STATUS: COMPLETE - 151 tests passing",
        meta: [
          "Cost drag: 0.088% of capital per year",
          "The tax hypothesis did not survive measurement",
          "Two latent bugs found and fixed on recheck",
        ],
      }));

      c.push(H1("Contents"),
        new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
        Break());

      // ------------------------------------------------------- 1
      c.push(
        H1("1. What this phase measures"),
        P("A backtest without costs describes a market, not a strategy. But for this strategy specifically, the interesting cost was never expected to be brokerage. At 0.4 trades per symbol per year there is simply not enough turnover for friction to matter."),
        P("The question worth asking was about tax, and the answer turned out to be different from the one anticipated."),

        H2("1.1 The cost stack"),
        Tbl(["Component", "Rate", "Applies to"], [
          ["Brokerage", "0%", "Delivery at a discount broker"],
          ["STT", "0.1%", "BOTH legs on delivery - the largest component"],
          ["Exchange transaction", "0.00297%", "Both legs; revised 1 Oct 2024"],
          ["SEBI turnover fee", "Rs 10 per crore", "Both legs"],
          ["Stamp duty", "0.015%", "BUY side only"],
          ["GST", "18%", "Brokerage + exchange + SEBI, not on taxes"],
          ["DP charge", "Rs 15.34", "SELL side, per scrip per day"],
          ["Slippage", "5bp, volatility-scaled", "Both legs"],
        ], [2300, 1900, 4800]),
        P("Measured round trip on a Rs 1,00,000 position: about 0.238%."),
        Callout("The component most often modelled wrongly", [
          "STT is charged on BOTH legs for delivery trades. Sell-side-only is INTRADAY.",
          "Halving it by mistake makes every backtest look better, and nothing else in the output flags the error. It is the single largest line in the stack, so the mistake is worth roughly 0.1% per round trip.",
        ], "warn"),

        H2("1.2 Two things a static rate card gets wrong"),
        P("Two rates moved INSIDE the backtest window, so a single fixed value would be wrong for roughly half the period:"),
        ...Code([
          "2024-07-23   STCG 15% -> 20%",
          "             LTCG 10% -> 12.5%",
          "             LTCG exemption Rs 1,00,000 -> Rs 1,25,000",
          "",
          "2024-10-01   NSE transaction charge 0.00325% -> 0.00297%",
        ]),
        P("Every rate is therefore a function of the trade date rather than a constant."),

        H2("1.3 Why slippage is not flat"),
        P("A constant slippage percentage is the wrong SHAPE for this strategy, not merely the wrong size. The strategy trades only on crash and rally days, which is precisely when spreads widen -- often three to five times. A flat figure is simultaneously too high on the quiet days it never trades and too low on the volatile days it always does, so slippage scales with the day's realised volatility."),
        Break(),
      );

      // ------------------------------------------------------- 2
      c.push(
        H1("2. Portfolio construction"),
        P("Capital is split equally across symbols at the start, and each symbol then keeps its own sleeve of cash and shares. On an exit the sleeve sells to cash; on a re-entry it buys back with whatever that sleeve holds."),
        Callout("Why sleeves rather than daily rebalancing", [
          "A daily-rebalanced equal-weight portfolio trades every symbol every day. It would cost more in brokerage than this strategy could ever save in drawdown.",
          "Sleeves trade only when the signal changes, which is the entire point of a low-turnover design.",
          "The cost is some weight drift as sleeves grow apart. Accepted deliberately, and stated rather than hidden.",
        ]),
        P("One consequence worth recording: five symbols listed after the backtest began, so their sleeves sit in cash until the stock exists. HYUNDAI's sleeve is idle for 3.8 years and JIOFIN's for 2.6. That is roughly two sleeves out of ninety-six earning nothing, which depresses the reported CAGR. It affects the strategy and the benchmark identically, so the drawdown comparison is unaffected, but the absolute return figures are lower than a cleaner universe would give."),
        Break(),
      );

      // ------------------------------------------------------- 3
      c.push(
        H1("3. Costs: a non-issue, as predicted"),
        Tbl(["", "Trades", "Costs", "Per year"], [
          ["Strategy", "239", "Rs 5,868", "0.109% of capital"],
          ["Buy & hold", "94", "Rs 1,102", "0.020% of capital"],
          ["Difference", "145", "Rs 4,766", "0.088% of capital"],
        ], [2400, 1800, 2400, 2400]),
        P("Under nine hundredths of one percent of capital per year. There was never room for friction to matter at this turnover, and it is now measured rather than assumed. The whipsaw guards from Phase 5 are what make this true -- without them the same signal could easily trade ten times as often."),
        Break(),
      );

      // ------------------------------------------------------- 4
      c.push(
        H1("4. The tax hypothesis, and why it failed"),
        P("The expectation going into this phase was clear and, in hindsight, insufficiently examined. Every crash exit converts a long-term holding into a short-term one, moving the Indian rate from 12.5% to 20%. Drawdown reduction would therefore carry a hidden tax penalty that published results ignore, and quantifying it would be a genuine contribution."),
        P("Two separate problems with that argument emerged on measurement."),

        H2("4.1 The conversion barely happens"),
        P("Only 23.3% of the strategy's sales are short-term. The strategy holds for YEARS between trades, so by the time an exit fires most positions are long past the twelve-month boundary. The effect is real but small."),

        H2("4.2 The comparison itself was flawed"),
        P("Buy-and-hold never sells, so on a realised basis it appears to pay no tax whatsoever. That is not a saving -- it is a deferral. It ends the period holding a large unrealised liability that crystallises the moment anyone actually wants the money."),
        P("Comparing a strategy that realises gains against one that defers them, without pricing the deferral, overstates the strategy's penalty by the entire buy-and-hold liability. Both books are therefore liquidated at the final close:"),
        Tbl(["", "Tax paid as we go", "Deferred liability", "Total if liquidated"], [
          ["Strategy", "Rs 27,295", "Rs 246,178", "Rs 273,473"],
          ["Buy & hold", "Rs 0", "Rs 317,958", "Rs 317,958"],
        ], [2200, 2300, 2300, 2200]),
        Callout("The strategy pays LESS tax, and that is not good news", [
          "Rs 44,484 less, in fact - the opposite sign to the hypothesis.",
          "But it pays less mainly because it EARNED less, and a smaller gain carries a smaller liability. Lower tax on a lower return is not a benefit.",
          "This is why after-tax terminal wealth, not tax paid, is the only figure that settles the question.",
        ], "warn"),

        H2("4.3 What you actually walk away with"),
        Tbl(["", "Terminal equity", "Tax if liquidated", "After-tax wealth", "CAGR"], [
          ["Strategy", "Rs 35,78,159", "Rs 273,473", "Rs 33,04,685", "24.66%"],
          ["Buy & hold", "Rs 36,67,081", "Rs 317,958", "Rs 33,49,123", "24.96%"],
        ], [1800, 2000, 1900, 1900, 1400]),
        P("A difference of -4.44% of capital over 5.4 years. In a window containing no crash, the protection costs roughly 0.8% a year and delivers nothing -- which is exactly what Phase 5 predicted, and is the correct behaviour for insurance in a year without a fire."),
        P("The honest conclusion is that the tax effect is real but small, and swamped by the return difference. Reporting it as a headline contribution would have been overclaiming. What survives is narrower and still worth having: the deferred-liability comparison IS the part most backtests get wrong, and correcting it reverses the sign of the apparent result."),
        Break(),
      );

      // ------------------------------------------------------- 5
      c.push(
        H1("5. Two latent bugs found on recheck"),
        P("Neither was affecting current results. Both would have been severe on different data, and both would have produced output that looked like a finding rather than an error."),

        H2("5.1 A missing price was valuing a holding at zero"),
        P("When a price was absent, the equity calculation treated the position as worth nothing. That prints a fake crash in the equity curve on the day of the gap and a fake recovery the day after."),
        P("A drawdown study that invents drawdowns is worse than useless. The current window contains only two such cells, so the measured impact was nil -- but the failure mode is exactly the kind that survives review, because the output is a plausible-looking dip rather than an error."),
        P("Open positions are now valued at the last KNOWN price. Stale prices are used for valuation only; trading is still skipped entirely when the price is missing, because you cannot fill at a price the market never printed. A regression test pins both halves."),

        H2("5.2 A tax bill that fell due after the backtest ended"),
        P("Indian financial years end on 31 March, and the backtest ends on 5 June 2026. The final partial year's tax therefore falls due after the window closes, and was never deducted from the equity curve -- while still being counted in the reported total."),
        P("The summary would have reported a liability the curve never paid. It happens to be zero in the current run, so nothing was wrong in practice, but the two figures could silently disagree. Any bill falling due beyond the window is now charged at the final bar."),
        Callout("The pattern, again", [
          "Neither bug raised an error, a warning, or an implausible number.",
          "This is now the fifth time in the project that a defect has been found which produced quietly wrong output rather than a visible failure - alongside the six data defects, the regime look-ahead, the double lag in the state machine, and the geometric mean masquerading as a portfolio.",
          "In a numerical pipeline the dangerous failures are the silent ones, and the only reliable defence is checking outputs against reality rather than against the code's own expectations.",
        ], "warn"),
        Break(),
      );

      // ------------------------------------------------------- 6
      c.push(
        H1("6. Known limitations"),
        Num("The rate values are unverified. They were built from documented Indian rates but not confirmed against a live broker contract note, and statutory rates change with each budget. They are isolated in CostConfig and TaxConfig so checking takes five minutes, and it should be done before publication."),
        Num("Carry-forward of capital losses across financial years is not modelled. That understates the strategy's after-tax result, so the figures here err on the conservative side."),
        Num("Tax is deducted on 31 March rather than as advance tax through the year. The timing affects the shape of the equity curve slightly, not the total."),
        Num("The DP charge is applied per sell leg, not per scrip per day. Identical here since exits are never split across multiple orders in a day, but it would over-count if they were."),
        Num("Slippage is a spread proxy and does not model the price impact of a large order. Irrelevant at Rs 10 lakh across ninety-six names; it would matter at institutional size."),
        Num("Five sleeves sit in cash for part of the window because those stocks had not listed. This depresses both CAGRs equally and leaves the drawdown comparison intact, but absolute returns are understated."),

        H1("7. Handoff to Phase 7"),
        P("Phase 7 compares the four retraining schemes: rolling three-year, incremental monthly, EWMA with decay 0.994, and the volatility-purged variant."),
        P("That comparison is only valid because intensity is a percentile of the TRAINING distribution rather than a raw score. A fixed cut on the raw score would mean four different things under four schemes, and the benchmark would be measuring the scoring scale instead of the schemes it claims to compare. The design decision that makes Phase 7 possible was taken in Phase 3, for exactly this reason."),
        P("Ranking should be by drawdown reduction per unit of turnover rather than by raw return, since a scheme that trades constantly can buy a better drawdown figure at a cost that only surfaces in the tax bill."),
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
