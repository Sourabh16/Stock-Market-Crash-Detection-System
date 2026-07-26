/*
 * build_docs.js
 * -------------
 * Regenerates every per-phase documentation file.
 *
 *     node scripts/build_docs.js            # all phases
 *     node scripts/build_docs.js phase1     # one phase
 *
 * To add a phase: create scripts/docx/phaseN.js exporting
 * { slug, docTitle, runningHead, build(h) } and register it below.
 */

const fs = require("fs");
const path = require("path");
const common = require("./docx/common");

const PHASES = {
  phase1: require("./docx/phase1"),
};

const outDir = path.join(__dirname, "..", "docs", "phases");
fs.mkdirSync(outDir, { recursive: true });

const wanted = process.argv[2] ? [process.argv[2]] : Object.keys(PHASES);

(async () => {
  for (const key of wanted) {
    const mod = PHASES[key];
    if (!mod) {
      console.error(`unknown phase '${key}' — known: ${Object.keys(PHASES).join(", ")}`);
      process.exitCode = 1;
      continue;
    }
    const doc = common.buildDocument({
      title: mod.runningHead,
      docTitle: mod.docTitle,
      children: mod.build(common),
    });
    const out = path.join(outDir, `${mod.slug}.docx`);
    const buf = await common.Packer.toBuffer(doc);
    fs.writeFileSync(out, buf);
    console.log(`wrote ${path.relative(process.cwd(), out)}  (${(buf.length / 1024).toFixed(0)} KB)`);
  }
})();
