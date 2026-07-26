"""
build_dashboard.py
------------------
Phase 10: the HTML dashboard.

    python scripts/build_dashboard.py

Reads everything the pipeline wrote and emits a single self-contained file at
reports/dashboard.html. Plotly comes from a CDN; all data is inlined, so the
file works offline once loaded and can be emailed or committed.

A DESIGN CONSTRAINT THAT SHAPED THIS FILE
-----------------------------------------
Most of the honest results in this project are narrow or negative:

  * the strategy never trades two thirds of the universe
  * it reduces drawdown DEPTH slightly and DURATION not at all
  * per-stock savings do not aggregate into portfolio savings
  * it has no early-warning skill beyond chance

A dashboard makes it trivially easy to show only the flattering slice -- lead
with the best symbol, quote the mean rather than the median, omit the
untouched-symbol count. Every one of those would be defensible individually and
misleading together.

So the untouched count, the median alongside the mean, and a dedicated
Limitations tab are all first-class here rather than buried. The Overview tab
states the negative results before the positive ones.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from qbeast_crash.config import DATA_PROCESSED, DEFAULT_CONFIG, REPORTS
from qbeast_crash.data import load_universe, trading_calendar

CFG = DEFAULT_CONFIG
OUT = REPORTS / "dashboard.html"


# =====================================================================
# Data collection
# =====================================================================
def _csv(name: str, **kw) -> pd.DataFrame:
    path = REPORTS / name
    return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame()


def collect() -> dict:
    """Gather every artefact the pipeline produced."""
    frames, audit = load_universe()
    cal = trading_calendar()
    w = CFG.windows
    oos = cal[(cal >= pd.Timestamp(w.backtest_start)) & (cal <= pd.Timestamp(w.backtest_end))]

    signals = pd.read_parquet(DATA_PROCESSED / "signals.parquet")
    intensity = pd.read_parquet(DATA_PROCESSED / "intensity.parquet")

    # Per-symbol series, rounded hard. Full precision would triple the file
    # for digits nobody can see on a chart.
    symbols = {}
    for sym, frame in frames.items():
        close = frame["close"].astype(float).reindex(oos).dropna()
        if len(close) < 250:
            continue
        try:
            sig = signals.xs(sym, level="symbol").reindex(close.index)
            inten = intensity.xs(sym, level="symbol")["intensity"].reindex(close.index)
        except KeyError:
            continue

        held = sig["in_position"].fillna(True).astype(bool)
        ret = close.pct_change().fillna(0.0)
        strat = (1 + ret * held.astype(float)).cumprod()
        bench = (1 + ret).cumprod()

        symbols[sym] = {
            "dates": [d.strftime("%Y-%m-%d") for d in close.index],
            "close": [round(float(v), 2) for v in close],
            "held": [int(v) for v in held],
            "intensity": [None if not np.isfinite(v) else round(float(v), 4) for v in inten],
            "dd_strategy": [round(float(v), 4) for v in (strat / strat.cummax() - 1)],
            "dd_bench": [round(float(v), 4) for v in (bench / bench.cummax() - 1)],
            "sells": [d.strftime("%Y-%m-%d") for d in sig.index[sig["action"] == "EXIT"]],
            "buys": [d.strftime("%Y-%m-%d") for d in sig.index[sig["action"] == "ENTER"]],
        }

    portfolio = {}
    for label, fname in (("strategy", "equity_strategy.parquet"),
                         ("buy_hold", "equity_buy_hold.parquet")):
        path = DATA_PROCESSED / fname
        if path.exists():
            eq = pd.read_parquet(path)["equity"]
            portfolio[label] = {
                "dates": [d.strftime("%Y-%m-%d") for d in eq.index],
                "equity": [round(float(v), 0) for v in eq],
                "drawdown": [round(float(v), 4) for v in (eq / eq.cummax() - 1)],
            }

    return {
        "generated": dt.datetime.now().strftime("%d %b %Y, %H:%M"),
        "window": f"{oos[0]:%d %b %Y} to {oos[-1]:%d %b %Y}",
        "train_window": f"{w.train_start:%b %Y} to {w.train_end:%b %Y}",
        "n_symbols": len(symbols),
        "n_sessions": len(oos),
        "symbols": symbols,
        "portfolio": portfolio,
        "drawdown_table": _csv("phase8_drawdown_by_symbol.csv", index_col=0),
        "schemes": _csv("phase7_scheme_comparison.csv", index_col=0),
        "scheme_by_symbol": _csv("phase7_drawdown_by_symbol_by_scheme.csv", index_col=0),
        "decay": _csv("phase7_decay_sweep.csv"),
        "horizon": _csv("phase4_horizon_decay.csv"),
        "leadtime": _csv("phase4_lead_time.csv", index_col=0),
        "cost_summary": _csv("phase6_summary.csv", index_col=0),
        "audit": audit,
    }


def png_b64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


# =====================================================================
# HTML
# =====================================================================
CSS = """
:root{
  --bg:#0A0E17; --panel:#111827; --panel-2:#0F1521; --line:#1E293B;
  --ink:#F1F5F9; --muted:#94A3B8; --dim:#64748B;
  --accent:#00E5FF; --good:#34D399; --warn:#FBBF24; --bad:#F87171;
  --violet:#A78BFA; --orange:#FB923C; --blue:#60A5FA;
  --r:14px;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:'Inter',system-ui,-apple-system,sans-serif;font-size:14px;line-height:1.6;
  -webkit-font-smoothing:antialiased}
code,.mono{font-family:'JetBrains Mono','SF Mono',Menlo,monospace}
a{color:var(--accent)}

.wrap{max-width:1440px;margin:0 auto;padding:0 22px 70px}

header{border-bottom:1px solid var(--line);
  background:linear-gradient(180deg,#0D1420 0%,var(--bg) 100%);padding:26px 0 0}
.brand{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.brand h1{margin:0;font-size:25px;font-weight:700;letter-spacing:-.4px}
.brand .accent{color:var(--accent)}
.brand .tagline{color:var(--muted);font-size:13px}
.meta{margin-top:9px;color:var(--dim);font-size:12.5px;display:flex;gap:9px;flex-wrap:wrap}
.meta span:not(:last-child)::after{content:'·';margin-left:9px;color:var(--line)}

.tabs{display:flex;gap:3px;margin-top:22px;overflow-x:auto;scrollbar-width:none}
.tabs::-webkit-scrollbar{display:none}
.tab{padding:11px 17px;border:0;background:transparent;color:var(--muted);
  font:inherit;font-size:13.5px;font-weight:500;cursor:pointer;white-space:nowrap;
  border-bottom:2px solid transparent;transition:.15s}
.tab:hover{color:var(--ink)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}

.panel{display:none;padding-top:28px}
.panel.active{display:block}

h2{font-size:17px;margin:34px 0 4px;font-weight:650;letter-spacing:-.2px}
h2:first-child{margin-top:6px}
.sub{color:var(--muted);font-size:13px;margin:0 0 16px;max-width:82ch}

.grid{display:grid;gap:14px}
.g2{grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
.g3{grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}
.g4{grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}

.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:18px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:16px 18px}
.kpi .lbl{color:var(--muted);font-size:11.5px;text-transform:uppercase;
  letter-spacing:.7px;font-weight:600}
.kpi .val{font-size:26px;font-weight:700;margin-top:6px;letter-spacing:-.6px}
.kpi .note{color:var(--dim);font-size:12px;margin-top:3px}
.pos{color:var(--good)} .neg{color:var(--bad)} .cy{color:var(--accent)}
.wn{color:var(--warn)} .vi{color:var(--violet)}

.chart{background:var(--panel);border:1px solid var(--line);
  border-radius:var(--r);padding:14px 12px 6px;margin:14px 0}
.chart h3{margin:2px 0 10px 6px;font-size:14px;font-weight:600}

table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--muted);font-weight:600;font-size:11.5px;
  text-transform:uppercase;letter-spacing:.6px;padding:9px 11px;
  border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel)}
td{padding:8px 11px;border-bottom:1px solid rgba(30,41,59,.55)}
tbody tr:hover{background:rgba(0,229,255,.04)}
.num{text-align:right;font-family:'JetBrains Mono',monospace;font-size:12.5px}
.scroll{max-height:560px;overflow:auto;border-radius:10px}
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:var(--accent)}

.callout{border-left:3px solid var(--accent);background:rgba(0,229,255,.05);
  padding:14px 18px;border-radius:0 10px 10px 0;margin:16px 0}
.callout.warn{border-left-color:var(--warn);background:rgba(251,191,36,.06)}
.callout.bad{border-left-color:var(--bad);background:rgba(248,113,113,.06)}
.callout.good{border-left-color:var(--good);background:rgba(52,211,153,.06)}
.callout>b{display:block;margin-bottom:6px;font-size:13.5px;color:var(--ink)}
.callout p b{display:inline;color:var(--ink)}
.callout p{margin:5px 0;color:var(--muted);font-size:13px}

.pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;
  font-weight:600;letter-spacing:.3px}
.pill.ok{background:rgba(52,211,153,.14);color:var(--good)}
.pill.no{background:rgba(248,113,113,.14);color:var(--bad)}
.pill.mid{background:rgba(251,191,36,.14);color:var(--warn)}
.pill.dim{background:rgba(148,163,184,.12);color:var(--muted)}

.picker{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
select,input[type=search]{background:var(--panel-2);border:1px solid var(--line);
  color:var(--ink);border-radius:9px;padding:9px 12px;font:inherit;font-size:13px}
select:focus,input:focus{outline:1px solid var(--accent)}

.chips{display:flex;flex-wrap:wrap;gap:6px;max-height:132px;overflow:auto;padding:2px}
.chip{padding:5px 11px;border-radius:8px;border:1px solid var(--line);
  background:var(--panel-2);color:var(--muted);font-size:12px;cursor:pointer;
  font-family:'JetBrains Mono',monospace;transition:.12s}
.chip:hover{border-color:var(--accent);color:var(--ink)}
.chip.active{background:var(--accent);color:#06121A;border-color:var(--accent);font-weight:600}

.imgwrap{background:var(--panel);border:1px solid var(--line);
  border-radius:var(--r);padding:12px;margin:14px 0}
.imgwrap img{width:100%;display:block;border-radius:8px}

footer{color:var(--dim);font-size:12px;border-top:1px solid var(--line);
  margin-top:50px;padding-top:18px}
"""

JS_HELPERS = """
const F={
  pct:(v,d=1)=>v==null||isNaN(v)?'—':(v*100).toFixed(d)+'%',
  pp:(v,d=1)=>v==null||isNaN(v)?'—':(v>0?'+':'')+v.toFixed(d)+'pp',
  num:(v,d=2)=>v==null||isNaN(v)?'—':v.toFixed(d),
  rs:v=>v==null||isNaN(v)?'—':'₹'+Math.round(v).toLocaleString('en-IN'),
  cls:v=>v>0?'pos':(v<0?'neg':'')
};
const LAYOUT={
  paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{family:'Inter,system-ui,sans-serif',color:'#94A3B8',size:11.5},
  margin:{l:56,r:22,t:12,b:40},
  xaxis:{gridcolor:'rgba(30,41,59,.65)',zerolinecolor:'#1E293B',linecolor:'#1E293B'},
  yaxis:{gridcolor:'rgba(30,41,59,.65)',zerolinecolor:'#1E293B',linecolor:'#1E293B'},
  legend:{orientation:'h',y:-0.18,font:{size:11}},
  hovermode:'x unified',
  hoverlabel:{bgcolor:'#111827',bordercolor:'#1E293B',font:{color:'#F1F5F9',size:12}}
};
const CFGP={displayModeBar:false,responsive:true};
const SCHEME_COLOUR={rolling:'#60A5FA',incremental:'#FB923C',ewma:'#34D399',vol_purged:'#A78BFA'};

function tab(id,btn){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  window.dispatchEvent(new Event('resize'));
  if(id==='explorer'&&!window._expDrawn){drawExplorer(EXP_SYMBOL);window._expDrawn=1;}
}

function sortTable(th,idx,numeric){
  const tb=th.closest('table').querySelector('tbody');
  const rows=[...tb.rows];
  const dir=th.dataset.dir==='asc'?-1:1;
  th.closest('tr').querySelectorAll('th').forEach(x=>delete x.dataset.dir);
  th.dataset.dir=dir===1?'asc':'desc';
  rows.sort((a,b)=>{
    let x=a.cells[idx].dataset.v??a.cells[idx].textContent;
    let y=b.cells[idx].dataset.v??b.cells[idx].textContent;
    if(numeric){x=parseFloat(x)||0;y=parseFloat(y)||0;return (x-y)*dir;}
    return x.localeCompare(y)*dir;
  });
  rows.forEach(r=>tb.appendChild(r));
}
"""


def kpi(label, value, note="", cls="") -> str:
    return (f'<div class="kpi"><div class="lbl">{label}</div>'
            f'<div class="val {cls}">{value}</div>'
            f'<div class="note">{note}</div></div>')


def callout(title, body, tone="") -> str:
    paras = "".join(f"<p>{b}</p>" for b in body)
    return f'<div class="callout {tone}"><b>{title}</b>{paras}</div>'


# =====================================================================
# Tabs
# =====================================================================
def tab_overview(d: dict) -> str:
    t = d["drawdown_table"]
    cs = d["cost_summary"]
    sch = d["schemes"]

    traded = int((t["exits"] > 0).sum()) if len(t) else 0
    untouched = len(t) - traded
    helped = int((t["drawdown_saved_pp"] > 0.1).sum()) if len(t) else 0
    hurt = int((t["drawdown_saved_pp"] < -0.1).sum()) if len(t) else 0
    mean_saved = float(t["drawdown_saved_pp"].mean()) if len(t) else 0.0
    med_saved = float(t["drawdown_saved_pp"].median()) if len(t) else 0.0

    s_row = cs.loc["strategy"] if "strategy" in cs.index else None
    b_row = cs.loc["buy & hold"] if "buy & hold" in cs.index else None

    kpis = [
        kpi("Universe", f'{d["n_symbols"]}', f'{d["n_sessions"]} sessions'),
        kpi("Symbols traded", f"{traded}", f"{untouched} never touched", "cy"),
        kpi("Mean DD saved", f"{mean_saved:+.2f}pp", "across all symbols",
            "pos" if mean_saved > 0 else "neg"),
        kpi("Median DD saved", f"{med_saved:+.2f}pp", "most symbols untouched", "wn"),
    ]
    if s_row is not None and b_row is not None:
        kpis += [
            kpi("Strategy CAGR", F_pct(s_row.get("cagr_net_all")),
                "net of costs and tax"),
            kpi("Buy & hold CAGR", F_pct(b_row.get("cagr_net_all")), "same basis"),
            kpi("Strategy max DD", F_pct(s_row.get("maxdd")), "portfolio level"),
            kpi("Buy & hold max DD", F_pct(b_row.get("maxdd")), "portfolio level"),
        ]

    return f"""
    <h2>What this system does</h2>
    <p class="sub">Stays fully invested by default and steps out when an Isolation Forest
    flags an unusual day that slope and acceleration confirm is a decline. Trained
    {d["train_window"]}, tested {d["window"]}.</p>

    {callout("Read these three limits before the numbers", [
      "It has <b>no early-warning skill</b>. Against a random signal of identical firing rate, recall was 0.88x — no better than chance at a 15-day horizon. This is a fast-reaction system, not a predictive one.",
      "It reduces drawdown <b>depth</b> slightly and <b>duration</b> not at all. Median sessions under water are identical to buy-and-hold.",
      f"It never trades <b>{untouched} of {len(t)}</b> symbols, so the median drawdown saved is {med_saved:+.2f}pp while the mean is {mean_saved:+.2f}pp. The mean alone would imply a broad effect where there is a narrow one."
    ], "warn")}

    <div class="grid g4">{"".join(kpis)}</div>

    <h2>Portfolio</h2>
    <p class="sub">Equity and drawdown, net of the full NSE cost stack and capital
    gains tax.</p>
    <div class="chart"><h3>Equity</h3><div id="pf-eq" style="height:330px"></div></div>
    <div class="chart"><h3>Drawdown</h3><div id="pf-dd" style="height:230px"></div></div>

    <h2>Where it helped, and where it did not</h2>
    <div class="grid g3">
      {kpi("Shallower drawdown", f"{helped}", "symbols", "pos")}
      {kpi("Deeper drawdown", f"{hurt}", "symbols", "neg")}
      {kpi("Unchanged", f"{untouched}", "never traded", "dim")}
    </div>
    <div class="chart"><h3>Per-symbol drawdown — above the diagonal means the model helped</h3>
      <div id="ov-scatter" style="height:460px"></div></div>

    {callout("The result that reframes who this is for", [
      "Per-stock savings do not aggregate. On the ten-stock portfolio ADANIENT alone saved 17.5pp and ITC 12.0pp, yet the portfolio improved only 1.0pp.",
      "Stocks trough at different times. ADANIENT's worst moment was February 2023; the portfolio's was March 2026, when ADANIENT was not the problem. Diversification had already absorbed most of what the model avoided.",
      "It is a <b>single-stock risk tool that aggregates weakly</b>, not a portfolio hedge."
    ], "bad")}
    """


def tab_data(d: dict) -> str:
    a = d["audit"]
    trunc = a[a["truncated_rows"] > 0] if "truncated_rows" in a else pd.DataFrame()
    dropped = a[~a["usable"]] if "usable" in a else pd.DataFrame()

    defects = [
        ("1", "adj_close is not an adjusted series",
         "Differs from close on ~30 of ~5,800 bars. A real adjustment factor differs on every bar before the last corporate action.",
         "Discarded; close is already back-adjusted"),
        ("2", "Fabricated pre-IPO history",
         "MAZDOCK carries monthly-spaced bars from 2017 but listed October 2020. Same in DMART, SBILIFE, VBL.",
         "Truncated by a calendar-gap rule"),
        ("3", "Zero-range bars",
         "BAJFINANCE has 1,092 of 5,803 bars where open = high = low = close.",
         "Flagged, never dropped — deleting a row shifts every window spanning it"),
        ("4", "Ragged end dates",
         "Files end anywhere between 2026-06-03 and 2026-06-22.",
         "Trimmed to a common 2026-06-05"),
        ("5", "Survivorship bias",
         "The universe is today's NIFTY 100, so companies that fell out are invisible.",
         "Documented as a limitation, not fixed"),
        ("6", "Unadjusted corporate actions",
         "CGPOWER falls 155.30 to 53.05 on 2016-03-15 — the Crompton Greaves demerger. Passes every structural check; only the return betrays it.",
         "Truncated; real crashes such as Hindenburg preserved"),
    ]
    rows = "".join(
        f"<tr><td class='mono cy'>{n}</td><td><b>{title}</b></td>"
        f"<td style='color:var(--muted)'>{ev}</td><td>{fix}</td></tr>"
        for n, title, ev, fix in defects)

    listings = [("VBL", "2016-11-08"), ("DMART", "2017-03-21"),
                ("SBILIFE", "2017-10-03"), ("MAZDOCK", "2020-10-12")]
    lrows = "".join(
        f"<tr><td class='mono'>{s}</td><td class='mono num'>{v}</td>"
        f"<td class='mono num'>{v}</td><td><span class='pill ok'>exact</span></td></tr>"
        for s, v in listings)

    return f"""
    <h2>Six defects, none of which would have raised an error</h2>
    <p class="sub">Every one produced output that looked completely normal — valid dates,
    positive prices, plausible returns. This matters more here than in most pipelines
    because Isolation Forest is unsupervised: a supervised model shown a bad example
    pushes back through its error signal, whereas this one silently absorbs it and
    learns that whatever it was shown is normal.</p>

    <div class="card"><div class="scroll"><table>
      <thead><tr><th>#</th><th>Defect</th><th>Evidence</th><th>Handling</th></tr></thead>
      <tbody>{rows}</tbody></table></div></div>

    <h2>The gap rule, validated against reality</h2>
    <p class="sub">A calendar gap over 10 days is always a vendor artefact — the NSE has
    never closed that long. Validated against listing dates sourced independently of this
    codebase, not against the code's own output.</p>
    <div class="card"><table>
      <thead><tr><th>Symbol</th><th class="num">Rule detected</th>
      <th class="num">Actual NSE listing</th><th>Match</th></tr></thead>
      <tbody>{lrows}</tbody></table></div>

    <div class="grid g3" style="margin-top:16px">
      {kpi("Usable symbols", f'{len(a[a["usable"]]) if "usable" in a else 0}', "of 100 supplied")}
      {kpi("Symbols truncated", f"{len(trunc)}", "fabricated history removed", "wn")}
      {kpi("Symbols dropped", f"{len(dropped)}", "insufficient history", "neg")}
    </div>

    {callout("Gaps are never forward-filled", [
      "A carried-forward price manufactures a return of exactly zero, and a run of zeros reads to the model as a stretch of unnatural calm — biasing every volatility estimate downward.",
      "Measured: 0.53% of daily returns are exactly zero. Forward-filling would inflate that substantially, and nothing in the output would look wrong."
    ])}
    """


def tab_detection(d: dict) -> str:
    h = d["horizon"]
    lt = d["leadtime"]

    hrows = "".join(
        f"<tr><td class='num mono'>{int(r.horizon)}</td>"
        f"<td class='num mono'>{r.base*100:.2f}%</td>"
        f"<td class='num mono cy'>{r.given_signal*100:.2f}%</td>"
        f"<td class='num mono pos'>{r.lift:.1f}x</td></tr>"
        for _, r in h.iterrows()) if len(h) else ""

    lrows = ""
    if len(lt):
        for name, r in lt.iterrows():
            lrows += (f"<tr><td>{name}</td><td class='num mono'>{int(r.get('signals',0))}</td>"
                      f"<td class='num mono'>{r.get('recall',0)*100:.1f}%</td>"
                      f"<td class='num mono'>{r.get('random_recall',0)*100:.1f}%</td>"
                      f"<td class='num mono {'pos' if r.get('skill',0)>1 else 'neg'}'>"
                      f"{r.get('skill',0):.2f}x</td></tr>")

    return f"""
    <h2>The go/no-go result</h2>
    <p class="sub">This phase existed to answer one question: how many days before a crash
    does the signal fire? The answer changed what the project can claim.</p>

    {callout("No early warning beyond chance", [
      "Crash events are frequent enough that any rule firing often will land near one by chance, so raw recall proves nothing. Every rule was measured against a <b>random signal of identical firing rate</b>.",
      "All rules scored at or below 1.00x skill. At a 15-day horizon the detector provides no early warning. The requirement to predict crashes 2–3 days ahead, in the sense of forecasting from a calm market, is <b>not met</b>."
    ], "bad")}

    <div class="card"><table>
      <thead><tr><th>Rule</th><th class="num">Signals</th><th class="num">Recall</th>
      <th class="num">Random</th><th class="num">Skill</th></tr></thead>
      <tbody>{lrows}</tbody></table></div>

    <h2>But the signal is enormously informative at short horizon</h2>
    <p class="sub">Probability of a 10% drawdown within H days, given a signal.
    The lift decays sharply with horizon — the signature of a coincident detector
    rather than a predictive one.</p>
    <div class="grid g2">
      <div class="card"><table>
        <thead><tr><th class="num">H</th><th class="num">Base rate</th>
        <th class="num">Given signal</th><th class="num">Lift</th></tr></thead>
        <tbody>{hrows}</tbody></table></div>
      <div class="chart"><h3>Lift decay</h3><div id="det-lift" style="height:280px"></div></div>
    </div>

    {callout("Precision and recall are both true at once", [
      "Phase 3 reported 3.48x lift; this phase reports no skill. Both are correct because they measure different things.",
      "<b>Precision</b> — P(crash | signal) — is high. <b>Recall</b> — P(signal | crash) — is low. The signal is precise but rare, firing on 130 of 128,737 symbol-days, so it can only ever cover a fraction of events.",
      "The median same-day return on signal days is −1.14% against +0.02% overall: it fires as a decline <i>begins</i>, not before it."
    ])}
    """


def tab_schemes(d: dict) -> str:
    sc = d["schemes"]
    dec = d["decay"]

    rows = "".join(
        f"<tr><td><span class='pill dim'>{name}</span></td>"
        f"<td class='num mono'>{r.cagr*100:.2f}%</td>"
        f"<td class='num mono'>{r.max_drawdown*100:.1f}%</td>"
        f"<td class='num mono {F_cls(r.drawdown_saved_pp)}'>{r.drawdown_saved_pp:+.1f}pp</td>"
        f"<td class='num mono'>{r.trades_per_sym_yr:.2f}</td>"
        f"<td class='num mono'>{r.efficiency:.1f}</td></tr>"
        for name, r in sc.iterrows()) if len(sc) else ""

    drows = "".join(
        f"<tr><td class='mono'>{r.decay:.4f}</td>"
        f"<td class='num mono'>{r.eff_years:.2f}y</td>"
        f"<td class='num mono {F_cls(r.dd_saved_pp)}'>{r.dd_saved_pp:+.1f}pp</td>"
        f"<td class='num mono'>{r.trades_per_sym_yr:.2f}</td>"
        f"<td class='num mono'>{r.efficiency:.1f}</td></tr>"
        for _, r in dec.iterrows()) if len(dec) else ""

    return f"""
    <h2>Four ways to choose training data</h2>
    <p class="sub">Walk-forward, monthly refits. All four share the same schedule so only
    the training <i>set</i> varies — changing the frequency too would confound the effects.
    The comparison is only valid because intensity is a percentile of each fit's own
    training distribution rather than a raw score.</p>

    <div class="card"><table>
      <thead><tr><th>Scheme</th><th class="num">CAGR</th><th class="num">Max DD</th>
      <th class="num">vs B&amp;H</th><th class="num">Trades/sym/yr</th>
      <th class="num">Efficiency</th></tr></thead>
      <tbody>{rows}</tbody></table></div>

    {callout("The scheme choice does not matter", [
      "The spread across all four is 0.2pp of drawdown. They are within noise of each other, which tells you where <i>not</i> to spend further effort.",
      "If one must be chosen: <b>rolling</b> — best efficiency and simplest to reason about. <code>vol_purged</code> trades nearly twice as often for no benefit and can be retired."
    ], "warn")}

    <h2>But retraining matters enormously — for coverage, not accuracy</h2>
    <p class="sub">Measuring coverage separately from performance surfaced the largest
    effect in the project.</p>
    <div class="grid g4">
      {kpi("Static fit", "44 / 96", "symbols the model could trade", "neg")}
      {kpi("Walk-forward", "79 / 96", "rolling scheme", "pos")}
      {kpi("Dev universe, static", "3 / 10", "before", "neg")}
      {kpi("Dev universe, walk-forward", "8 / 10", "after", "pos")}
    </div>
    {callout("Why most stocks could never fire", [
      "Intensity was a percentile of the <b>pooled</b> training distribution, dominated by the most volatile names. RELIANCE's highest intensity in 5.4 years was 0.9897 — a 0.99 threshold was unreachable by construction.",
      "And the 2016–2020 training window contains COVID, so each stock's own top percentile is set by March 2020, which calm 2021–2026 never approaches. A rolling three-year window in 2024 contains no COVID, so the bar drops to something reachable.",
      "<b>Retraining is what makes the model act at all</b> — a far larger effect than any accuracy difference between schemes."
    ], "good")}

    <h2>EWMA decay — the one parameter that does matter</h2>
    <p class="sub">A single sweep proves nothing, so the noise floor was measured first
    (sd 0.05pp), then the candidates were run on four seeds each.</p>
    <div class="grid g2">
      <div class="card"><table>
        <thead><tr><th>Decay</th><th class="num">Memory</th><th class="num">DD saved</th>
        <th class="num">Trades</th><th class="num">Efficiency</th></tr></thead>
        <tbody>{drows}</tbody></table></div>
      <div class="chart"><h3>Drawdown saved by decay</h3>
        <div id="sc-decay" style="height:290px"></div></div>
    </div>
    {callout("0.994 is vindicated on measurement", [
      "0.994 and 0.997 sit 0.27pp apart with a pooled sd of 0.09 — a separation of <b>3.1 standard deviations</b>. Real signal, not luck.",
      "0.994 also has the lowest variance across seeds, so it is the most stable as well as the best. Note the contrast: the choice of <i>scheme</i> does not matter, but within EWMA the <i>decay</i> does."
    ], "good")}

    <div class="chart"><h3>Portfolio by scheme</h3>
      <div class="imgwrap"><img src="data:image/png;base64,{d['img_schemes']}" alt="portfolio by scheme"></div></div>
    """


def tab_portfolio(d: dict) -> str:
    cs = d["cost_summary"]
    if not len(cs):
        return "<p class='sub'>Run phase 6 first.</p>"
    s = cs.loc["strategy"]; b = cs.loc["buy & hold"]

    return f"""
    <h2>Costs and tax</h2>
    <p class="sub">The full NSE delivery stack — STT on both legs, stamp duty on buys,
    DP charges on sells, GST, exchange and SEBI fees — plus Indian capital gains,
    with rates that change inside the backtest window.</p>

    <div class="grid g4">
      {kpi("Strategy trades", f'{int(s.get("trades",0))}', "over the window")}
      {kpi("Buy &amp; hold trades", f'{int(b.get("trades",0))}', "initial purchases only")}
      {kpi("Extra cost", F_rs(s.get("costs",0)-b.get("costs",0)),
           f'{(s.get("costs",0)-b.get("costs",0))/1e6*100/5.42:.3f}% of capital a year')}
      {kpi("Tax paid", F_rs(s.get("tax",0)), "realised, as we go")}
    </div>

    {callout("Costs are a non-issue; the tax hypothesis was wrong", [
      "Extra cost is about 0.088% of capital a year. At 0.4 trades per symbol per year there was never room for friction to matter.",
      "The expectation was that crash exits convert long-term holdings into short-term ones, so drawdown reduction would carry a hidden tax penalty. Only <b>23.3%</b> of sales are short-term — the strategy holds for years between trades.",
      "And comparing realised tax against buy-and-hold is meaningless, because buy-and-hold never sells and appears to pay nothing. That is deferral, not saving."
    ], "warn")}

    <h2>Like-for-like, both books liquidated at the final close</h2>
    <div class="card"><table>
      <thead><tr><th></th><th class="num">Tax paid as we go</th>
      <th class="num">Deferred liability</th><th class="num">Total if liquidated</th></tr></thead>
      <tbody>
        <tr><td>Strategy</td><td class="num mono">{F_rs(s.get("tax",0))}</td>
          <td class="num mono">{F_rs(s.get("tax_liquidated",0)-s.get("tax",0))}</td>
          <td class="num mono cy">{F_rs(s.get("tax_liquidated",0))}</td></tr>
        <tr><td>Buy &amp; hold</td><td class="num mono">{F_rs(b.get("tax",0))}</td>
          <td class="num mono">{F_rs(b.get("tax_liquidated",0)-b.get("tax",0))}</td>
          <td class="num mono cy">{F_rs(b.get("tax_liquidated",0))}</td></tr>
      </tbody></table></div>
    <p class="sub" style="margin-top:12px">The strategy pays less tax — but mainly because
    it earned less, and a smaller gain carries a smaller liability. Lower tax on a lower
    return is not a benefit, which is why after-tax terminal wealth is the only figure
    that settles it.</p>
    """


def tab_symbols(d: dict) -> str:
    t = d["drawdown_table"]
    if not len(t):
        return "<p class='sub'>Run phase 8 first.</p>"

    rows = ""
    for sym, r in t.sort_values("drawdown_saved_pp", ascending=False).iterrows():
        saved = r["drawdown_saved_pp"]
        pill = ("<span class='pill ok'>helped</span>" if saved > 0.1 else
                "<span class='pill no'>hurt</span>" if saved < -0.1 else
                "<span class='pill dim'>untouched</span>")
        rows += (
            f"<tr><td class='mono'><b>{sym}</b></td>"
            f"<td class='num mono' data-v='{r['strategy_max_drawdown']}'>{r['strategy_max_drawdown']*100:.1f}%</td>"
            f"<td class='num mono' data-v='{r['bh_max_drawdown']}'>{r['bh_max_drawdown']*100:.1f}%</td>"
            f"<td class='num mono {F_cls(saved)}' data-v='{saved}'>{saved:+.1f}pp</td>"
            f"<td class='num mono' data-v='{r['strategy_return']}'>{r['strategy_return']*100:.0f}%</td>"
            f"<td class='num mono' data-v='{r['bh_return']}'>{r['bh_return']*100:.0f}%</td>"
            f"<td class='num mono' data-v='{r['exits']}'>{int(r['exits'])}</td>"
            f"<td class='num mono' data-v='{r['pct_days_in_cash']}'>{r['pct_days_in_cash']:.1f}%</td>"
            f"<td>{pill}</td></tr>")

    heads = ["Symbol", "Strat DD", "B&amp;H DD", "Saved", "Strat ret", "B&amp;H ret",
             "Exits", "In cash", "Verdict"]
    th = "".join(
        f'<th class="sortable {"num" if i else ""}" onclick="sortTable(this,{i},{str(i>0).lower()})">{h}</th>'
        for i, h in enumerate(heads))

    return f"""
    <h2>All {len(t)} symbols</h2>
    <p class="sub">Click any column to sort. Two-thirds are never traded — that is the
    dominant fact about this strategy, and sorting by <i>Saved</i> makes it visible
    immediately.</p>
    <div class="picker">
      <input type="search" id="symfilter" placeholder="filter symbols…"
             oninput="filterSyms(this.value)" style="min-width:220px">
      <span class="pill dim" id="symcount">{len(t)} shown</span>
    </div>
    <div class="card"><div class="scroll"><table id="symtable">
      <thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div></div>
    """


def tab_explorer(d: dict) -> str:
    syms = sorted(d["symbols"])
    chips = "".join(f'<div class="chip" onclick="drawExplorer(\'{s}\')" '
                    f'id="chip-{s}">{s}</div>' for s in syms)
    opts = "".join(f'<option value="{s}">{s}</option>' for s in syms)
    return f"""
    <h2>Symbol explorer</h2>
    <p class="sub">Price with buy and sell markers, anomaly intensity against the action
    threshold, and drawdown against buy-and-hold. The shaded bands show periods held in
    cash — more informative than the markers, because they show what the strategy was
    holding <i>through</i>.</p>
    <div class="picker">
      <select id="symsel" onchange="drawExplorer(this.value)">{opts}</select>
      <span class="pill dim" id="exp-summary"></span>
    </div>
    <div class="chips">{chips}</div>
    <div class="chart"><h3 id="exp-title">—</h3><div id="exp-price" style="height:340px"></div></div>
    <div class="chart"><h3>Anomaly intensity</h3><div id="exp-int" style="height:190px"></div></div>
    <div class="chart"><h3>Drawdown</h3><div id="exp-dd" style="height:230px"></div></div>
    """


def tab_limits(d: dict) -> str:
    items = [
        ("No early-warning skill",
         "Against a random signal of identical firing rate, recall was 0.88x. The 2–3 day prediction requirement is not met. The honest claim is fast reaction, not prediction."),
        ("Duration is unchanged",
         "Median sessions under water are identical to buy-and-hold (1,256 either way). The strategy reduces how far you fall, not how long you stay down."),
        ("Per-stock savings do not aggregate",
         "ADANIENT saved 17.5pp and ITC 12.0pp, yet the portfolio improved 1.0pp. Stocks trough at different times, and diversification already absorbs most of what the model avoids."),
        ("The backtest window contains no sharp crash",
         "The worst drawdown of 2021–2026 was a 154-day slow bleed at 16.5% annualised volatility with one day beyond 3%. There is nothing in it for an anomaly detector to fire on."),
        ("The stress test rests on a single event",
         "2020 showed +7.7pp of drawdown saved on a model that never saw COVID. That is one event. Nothing generalises from n=1."),
        ("Survivorship bias",
         "The universe is today's NIFTY 100. Companies that fell out of the index are invisible. It inflates the benchmark as much as the strategy, but it is present."),
        ("Cost and tax rates are unverified",
         "Built from documented Indian rates but not confirmed against a live contract note, and statutory rates change with each budget."),
        ("Thresholds are conventional, not optimised",
         "0.99, 0.95 and 0.90 are round numbers. Optimising them against the hold-out would be a form of look-ahead."),
    ]
    cards = "".join(
        f'<div class="card"><b style="color:var(--warn)">{t}</b>'
        f'<p style="color:var(--muted);margin:8px 0 0;font-size:13px">{b}</p></div>'
        for t, b in items)

    return f"""
    <h2>Known limitations</h2>
    <p class="sub">Stated in full rather than distributed through footnotes. Most results
    in this project are narrow or negative, and a dashboard makes it easy to show only
    the flattering slice.</p>
    <div class="grid g2">{cards}</div>

    <h2>Defects found during development</h2>
    <p class="sub">Recorded because the pattern is the most transferable thing here:
    <b>none of these raised an error, a warning, or an implausible number.</b> Each
    produced output that looked entirely reasonable and was quietly wrong.</p>
    <div class="card"><table>
      <thead><tr><th>#</th><th>Defect</th><th>How it was caught</th></tr></thead>
      <tbody>
        <tr><td class="mono cy">1–6</td><td>Six data defects, including a demerger that passes every structural check</td><td>Auditing raw data before writing model code</td></tr>
        <tr><td class="mono cy">7</td><td>Look-ahead in the inherited regime detector — 9.9% of labels changed once fixed</td><td>Future-perturbation test</td></tr>
        <tr><td class="mono cy">8</td><td>A crash muting its own signal via a contemporaneous volatility denominator</td><td>Test on real COVID data</td></tr>
        <tr><td class="mono cy">9</td><td>Double lag in the state machine — every trade delayed a second day</td><td>Test asserting T+1 execution</td></tr>
        <tr><td class="mono cy">10</td><td>Geometric mean masquerading as a portfolio — understated CAGR by 4.75pp</td><td>Code review</td></tr>
        <tr><td class="mono cy">11</td><td>EWMA resampling with replacement — 2.63x duplication made recent rows look less anomalous</td><td>Recheck before documenting</td></tr>
        <tr><td class="mono cy">12</td><td>Scatter chart axis label inverted — said the opposite of the truth</td><td>Looking at the picture</td></tr>
        <tr><td class="mono cy">13</td><td>Two-thirds of symbols could not fire a signal at all</td><td>Measuring coverage separately from performance</td></tr>
      </tbody></table></div>
    """


# small server-side formatters used inside f-strings
def F_pct(v, d=2):
    try:
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v*100:.{d}f}%"
    except Exception:
        return "—"


def F_rs(v):
    try:
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"₹{round(float(v)):,}"
    except Exception:
        return "—"


def F_cls(v):
    try:
        return "pos" if v > 0 else ("neg" if v < 0 else "")
    except Exception:
        return ""


# =====================================================================
# Assembly
# =====================================================================
def build(d: dict) -> str:
    payload = {
        "symbols": d["symbols"],
        "portfolio": d["portfolio"],
        "scatter": [
            {"s": s, "x": float(r["bh_max_drawdown"]), "y": float(r["strategy_max_drawdown"]),
             "e": int(r["exits"])}
            for s, r in d["drawdown_table"].iterrows()
        ] if len(d["drawdown_table"]) else [],
        "horizon": d["horizon"].to_dict("records") if len(d["horizon"]) else [],
        "decay": d["decay"].to_dict("records") if len(d["decay"]) else [],
    }

    tabs = [
        ("overview", "Overview", tab_overview(d)),
        ("data", "Data Quality", tab_data(d)),
        ("detection", "Detection", tab_detection(d)),
        ("schemes", "Retraining", tab_schemes(d)),
        ("portfolio", "Costs &amp; Tax", tab_portfolio(d)),
        ("symbols", "All Symbols", tab_symbols(d)),
        ("explorer", "Explorer", tab_explorer(d)),
        ("limits", "Limitations", tab_limits(d)),
    ]
    buttons = "".join(
        f'<button class="tab{" active" if i == 0 else ""}" onclick="tab(\'{k}\',this)">{lab}</button>'
        for i, (k, lab, _) in enumerate(tabs))
    panels = "".join(
        f'<div class="panel{" active" if i == 0 else ""}" id="{k}">{body}</div>'
        for i, (k, _, body) in enumerate(tabs))

    first = sorted(d["symbols"])[0] if d["symbols"] else ""

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QBEAST — Crash Detection Report</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>{CSS}</style></head>
<body>
<header><div class="wrap">
  <div class="brand">
    <h1>QBEAST <span class="accent">·</span> Crash Detection</h1>
    <span class="tagline">Isolation Forest anomaly detection on Indian equities</span>
  </div>
  <div class="meta">
    <span>{d['n_symbols']} symbols</span><span>{d['n_sessions']} sessions</span>
    <span>trained {d['train_window']}</span><span>tested {d['window']}</span>
    <span>generated {d['generated']}</span>
  </div>
  <div class="tabs">{buttons}</div>
</div></header>
<div class="wrap">{panels}
<footer>QBEAST · Isolation Forest crash detection · generated {d['generated']} ·
all figures out-of-sample unless stated</footer>
</div>
<script>
const DATA={json.dumps(payload, separators=(",", ":"))};
const EXP_SYMBOL={json.dumps(first)};
{JS_HELPERS}
{JS_CHARTS}
</script></body></html>"""


JS_CHARTS = """
// ---------- portfolio ----------
(function(){
  const p=DATA.portfolio; if(!p.strategy) return;
  Plotly.newPlot('pf-eq',[
    {x:p.buy_hold.dates,y:p.buy_hold.equity,name:'buy & hold',type:'scatter',mode:'lines',
     line:{color:'#64748B',width:1.6}},
    {x:p.strategy.dates,y:p.strategy.equity,name:'strategy',type:'scatter',mode:'lines',
     line:{color:'#00E5FF',width:1.8}}
  ],{...LAYOUT,yaxis:{...LAYOUT.yaxis,tickprefix:'₹'}},CFGP);

  Plotly.newPlot('pf-dd',[
    {x:p.buy_hold.dates,y:p.buy_hold.drawdown,name:'buy & hold',type:'scatter',
     fill:'tozeroy',mode:'lines',line:{color:'#64748B',width:1},
     fillcolor:'rgba(100,116,139,.28)'},
    {x:p.strategy.dates,y:p.strategy.drawdown,name:'strategy',type:'scatter',mode:'lines',
     line:{color:'#00E5FF',width:1.6}}
  ],{...LAYOUT,yaxis:{...LAYOUT.yaxis,tickformat:'.0%'}},CFGP);
})();

// ---------- scatter ----------
(function(){
  const s=DATA.scatter; if(!s.length) return;
  const g=(f)=>s.filter(f);
  const helped=g(d=>d.y-d.x>0.001), hurt=g(d=>d.y-d.x<-0.001),
        flat=g(d=>Math.abs(d.y-d.x)<=0.001);
  const lo=Math.min(...s.map(d=>Math.min(d.x,d.y)))*1.05;
  const mk=(arr,name,col)=>({x:arr.map(d=>d.x),y:arr.map(d=>d.y),text:arr.map(d=>d.s),
    name:name+' ('+arr.length+')',mode:'markers',type:'scatter',
    marker:{size:8,color:col,opacity:.85},
    hovertemplate:'%{text}<br>B&H %{x:.1%}<br>strategy %{y:.1%}<extra></extra>'});
  Plotly.newPlot('ov-scatter',[
    {x:[lo,0],y:[lo,0],mode:'lines',line:{color:'#475569',dash:'dash',width:1},
     name:'no change',hoverinfo:'skip'},
    mk(flat,'unchanged','#64748B'),mk(helped,'shallower','#34D399'),mk(hurt,'deeper','#F87171')
  ],{...LAYOUT,hovermode:'closest',
     xaxis:{...LAYOUT.xaxis,title:'buy & hold max drawdown',tickformat:'.0%'},
     yaxis:{...LAYOUT.yaxis,title:'strategy max drawdown',tickformat:'.0%'}},CFGP);
})();

// ---------- lift decay ----------
(function(){
  const h=DATA.horizon; if(!h.length) return;
  Plotly.newPlot('det-lift',[
    {x:h.map(r=>r.horizon),y:h.map(r=>r.lift),type:'scatter',mode:'lines+markers',
     name:'lift',line:{color:'#00E5FF',width:2.4},marker:{size:8},
     hovertemplate:'H=%{x}d<br>%{y:.1f}x<extra></extra>'}
  ],{...LAYOUT,hovermode:'closest',
     xaxis:{...LAYOUT.xaxis,title:'horizon (trading days)'},
     yaxis:{...LAYOUT.yaxis,title:'lift over base rate',type:'log'}},CFGP);
})();

// ---------- decay sweep ----------
(function(){
  const d=DATA.decay; if(!d.length) return;
  Plotly.newPlot('sc-decay',[
    {x:d.map(r=>r.eff_years),y:d.map(r=>r.dd_saved_pp),type:'scatter',
     mode:'lines+markers',line:{color:'#A78BFA',width:2.2},marker:{size:9},
     text:d.map(r=>'decay '+r.decay.toFixed(4)),
     hovertemplate:'%{text}<br>memory %{x:.2f}y<br>saved %{y:+.2f}pp<extra></extra>'}
  ],{...LAYOUT,hovermode:'closest',
     xaxis:{...LAYOUT.xaxis,title:'effective memory (years)'},
     yaxis:{...LAYOUT.yaxis,title:'drawdown saved (pp)'}},CFGP);
})();

// ---------- explorer ----------
function drawExplorer(sym){
  const s=DATA.symbols[sym]; if(!s) return;
  document.getElementById('symsel').value=sym;
  document.querySelectorAll('.chip').forEach(c=>c.classList.remove('active'));
  const chip=document.getElementById('chip-'+sym); if(chip)chip.classList.add('active');

  const cash=1-s.held.reduce((a,b)=>a+b,0)/s.held.length;
  document.getElementById('exp-summary').textContent=
    s.sells.length+' sells · '+s.buys.length+' buys · '+(cash*100).toFixed(1)+'% of days in cash';
  document.getElementById('exp-title').textContent=
    sym+' — strategy '+F.pct(s.dd_strategy.reduce((m,v)=>Math.min(m,v),0))+
    ' max DD vs buy & hold '+F.pct(s.dd_bench.reduce((m,v)=>Math.min(m,v),0));

  // shade every stretch held in cash
  const shapes=[]; let start=null;
  for(let i=0;i<s.held.length;i++){
    if(!s.held[i]&&start===null)start=s.dates[i];
    if((s.held[i]||i===s.held.length-1)&&start!==null){
      shapes.push({type:'rect',xref:'x',yref:'paper',x0:start,x1:s.dates[i],y0:0,y1:1,
        fillcolor:'rgba(248,113,113,.13)',line:{width:0},layer:'below'});
      start=null;
    }
  }
  const at=(ds)=>ds.map(d=>s.close[s.dates.indexOf(d)]);
  Plotly.newPlot('exp-price',[
    {x:s.dates,y:s.close,type:'scatter',mode:'lines',name:'close',
     line:{color:'#E2E8F0',width:1.3}},
    {x:s.sells,y:at(s.sells),mode:'markers',name:'sell',
     marker:{symbol:'triangle-down',size:11,color:'#F87171'}},
    {x:s.buys,y:at(s.buys),mode:'markers',name:'buy',
     marker:{symbol:'triangle-up',size:11,color:'#34D399'}}
  ],{...LAYOUT,shapes},CFGP);

  Plotly.newPlot('exp-int',[
    {x:s.dates,y:s.intensity,type:'scatter',mode:'lines',name:'intensity',
     line:{color:'#A78BFA',width:1.1}}
  ],{...LAYOUT,yaxis:{...LAYOUT.yaxis,range:[0,1.02]},
     shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:0.99,y1:0.99,
       line:{color:'#F87171',dash:'dash',width:1.2}}]},CFGP);

  Plotly.newPlot('exp-dd',[
    {x:s.dates,y:s.dd_bench,type:'scatter',fill:'tozeroy',mode:'lines',name:'buy & hold',
     line:{color:'#64748B',width:1},fillcolor:'rgba(100,116,139,.28)'},
    {x:s.dates,y:s.dd_strategy,type:'scatter',mode:'lines',name:'strategy',
     line:{color:'#00E5FF',width:1.6}}
  ],{...LAYOUT,yaxis:{...LAYOUT.yaxis,tickformat:'.0%'}},CFGP);
}

function filterSyms(q){
  q=q.trim().toUpperCase();
  const rows=document.querySelectorAll('#symtable tbody tr');
  let n=0;
  rows.forEach(r=>{
    const hit=r.cells[0].textContent.toUpperCase().includes(q);
    r.style.display=hit?'':'none'; if(hit)n++;
  });
  document.getElementById('symcount').textContent=n+' shown';
}
"""


def main() -> int:
    print("collecting artefacts ...")
    d = collect()
    d["img_schemes"] = png_b64(REPORTS / "figures" / "portfolio_schemes.png")
    print(f"  {d['n_symbols']} symbols, {d['n_sessions']} sessions")

    # Artefacts from a --dev run silently mix with a full run and produce a
    # dashboard saying "96 symbols" beside "3 traded". Refuse rather than
    # publish a report whose numbers disagree with each other.
    n_table = len(d["drawdown_table"])
    if n_table and abs(n_table - d["n_symbols"]) > 2:
        print(f"\nMISMATCH: signals cover {d['n_symbols']} symbols but "
              f"phase8_drawdown_by_symbol.csv has {n_table} rows.")
        print("Some reports/ artefacts are from a --dev run. Re-run the full "
              "pipeline first:\n    python scripts/run_all_phases.py")
        return 1

    html = build(d)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({len(html)/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
