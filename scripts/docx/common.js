/*
 * common.js
 * ---------
 * Shared layout, styling and helpers for the per-phase documentation.
 *
 * Every phase document is GENERATED, never hand-edited, so it cannot drift out
 * of step with the code. Add a phase by creating scripts/docx/phaseN.js that
 * exports { slug, title, subtitle, sections(h) } and registering it in
 * scripts/build_docs.js.
 */

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

module.exports = { ...helpers, Packer, TableOfContents, cover, buildDocument };
