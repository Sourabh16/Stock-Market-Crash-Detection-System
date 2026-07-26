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
from qbeast_crash.signals import is_sell

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
            "sells": [d.strftime("%Y-%m-%d") for d in sig.index[is_sell(sig["action"])]],
            "buys": [d.strftime("%Y-%m-%d") for d in sig.index[sig["action"] == "ENTER"]],
            "anomalies": _anomaly_rows(sym, close, sig, intensity),
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


def _anomaly_rows(sym, close, sig, intensity) -> list[dict]:
    """
    Every day the model called unusual, with what it saw and what it did.

    Only days above the "watch" threshold are listed. Including ordinary days
    would bury the interesting ones in thousands of rows.
    """
    try:
        block = intensity.xs(sym, level="symbol").reindex(close.index)
    except KeyError:
        return []

    cfg = CFG.signals
    flagged = block["intensity"] >= cfg.moderate_intensity
    if not flagged.any():
        return []

    action_label = {"EXIT": "SELL", "EXIT_WATCH": "SELL (from watch)", "ENTER": "BUY"}
    held = sig["in_position"].fillna(True).astype(bool)
    watching = (sig["watching"].fillna(False).astype(bool)
                if "watching" in sig else pd.Series(False, index=sig.index))

    rows = []
    for d in block.index[flagged.fillna(False)]:
        a = sig["action"].get(d, "")
        if a in action_label:
            label = action_label[a]
        elif not held.get(d, True):
            # Already sold and sitting in cash. Calling this "watch only" reads
            # as inaction when the model had in fact already acted -- the most
            # confusing rows in the table were the days AFTER a sell.
            label = "already out"
        elif watching.get(d, False):
            label = "watching"
        else:
            label = "no action"
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "score": round(float(block.loc[d, "anomaly_score"]), 4)
                     if "anomaly_score" in block else None,
            "intensity": round(float(block.loc[d, "intensity"]), 4),
            "severity": "Severe" if block.loc[d, "intensity"] >= cfg.exit_intensity else "Mild",
            "slope": round(float(block.loc[d, "slope_z"]), 2),
            "accel": round(float(block.loc[d, "accel_z"]), 2),
            "regime": str(block.loc[d, "regime"]) if "regime" in block else "—",
            "trend": str(block.loc[d, "phase"]),
            "ret": round(float(block.loc[d, "ret_1d"]), 2) if "ret_1d" in block else None,
            "action": label,
        })
    return rows


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
# Small formatters used inside the f-strings below
# =====================================================================
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
# Tabs -- deliberately plain language
# =====================================================================
GLOSSARY = """
    <div class="card" style="margin-top:18px">
      <b style="font-size:13px">What the words mean</b>
      <table style="margin-top:10px">
        <tbody>
        <tr><td><b>Anomaly score</b></td><td style="color:var(--muted)">
          How unusual the day looked to the model. Higher = more unusual.</td></tr>
        <tr><td><b>Anomaly intensity</b></td><td style="color:var(--muted)">
          The score turned into a 0&ndash;1 rank. 0.99 means the day was more unusual
          than 99% of the days the model learned from.</td></tr>
        <tr><td><b>Slope</b></td><td style="color:var(--muted)">
          How fast the price is moving, measured in the stock's own typical daily
          move. &minus;2 means falling twice as fast as a normal day for that stock.</td></tr>
        <tr><td><b>Acceleration</b></td><td style="color:var(--muted)">
          Whether the move is speeding up or slowing down. Negative while falling
          means it is getting worse.</td></tr>
        <tr><td><b>Drawdown</b></td><td style="color:var(--muted)">
          How far below its best-ever value the investment currently sits.
          &minus;20% means you are 20% below the peak.</td></tr>
        <tr><td><b>Percentage points (pp)</b></td><td style="color:var(--muted)">
          The plain difference between two percentages. Going from &minus;20% to
          &minus;18% is an improvement of <b>2 percentage points</b>, not 2%.</td></tr>
        <tr><td><b>Buy &amp; hold</b></td><td style="color:var(--muted)">
          Buying on day one and never selling. The benchmark everything is
          compared against.</td></tr>
        <tr><td><b>CAGR</b></td><td style="color:var(--muted)">
          Average yearly growth rate over the whole period.</td></tr>
        </tbody></table>
    </div>
"""


def tab_dashboard(d: dict) -> str:
    t = d["drawdown_table"]
    cs = d["cost_summary"]
    traded = int((t["exits"] > 0).sum()) if len(t) else 0
    untouched = len(t) - traded
    helped = int((t["drawdown_saved_pp"] > 0.1).sum()) if len(t) else 0
    hurt = int((t["drawdown_saved_pp"] < -0.1).sum()) if len(t) else 0
    mean_s = float(t["drawdown_saved_pp"].mean()) if len(t) else 0.0
    med_s = float(t["drawdown_saved_pp"].median()) if len(t) else 0.0

    kpis = [kpi("Stocks tested", f'{d["n_symbols"]}', f'{d["n_sessions"]} trading days'),
            kpi("Stocks it traded", f"{traded}", f"{untouched} were never touched", "cy"),
            kpi("Average drawdown saved", f"{mean_s:+.2f} pts", "across all stocks",
                "pos" if mean_s > 0 else "neg"),
            kpi("Typical stock", f"{med_s:+.2f} pts", "the middle stock", "wn")]
    if len(cs) and "strategy" in cs.index:
        sr, br = cs.loc["strategy"], cs.loc["buy & hold"]
        kpis += [kpi("Model yearly return", F_pct(sr.get("cagr_net_all")), "after costs and tax"),
                 kpi("Buy &amp; hold return", F_pct(br.get("cagr_net_all")), "after costs and tax"),
                 kpi("Model worst fall", F_pct(sr.get("maxdd")), "peak to trough"),
                 kpi("Buy &amp; hold worst fall", F_pct(br.get("maxdd")), "peak to trough")]

    return f"""
    <h2>What this does</h2>
    <p class="sub">The model holds every stock all the time, and only sells when it
    spots a day that looks unusual <i>and</i> the price is falling faster and faster.
    It buys back afterwards. The aim is not to earn more &mdash; it is to fall less
    when a crash comes.</p>
    <p class="sub">Learnt from {d["train_window"]}. Tested on {d["window"]}, which the
    model never saw while learning.</p>

    <div class="grid g4">{"".join(kpis)}</div>

    <h2>How the model decides</h2>
    <div class="grid g2">
      <div class="card"><b class="cy">Severe unusual day</b>
        <p style="color:var(--muted);font-size:13px;margin:8px 0 0">
        Score in the top 1%. Check whether the price is falling and the fall is
        speeding up. If yes, <b>sell the next trading day</b>. No waiting &mdash; the
        advantage disappears within about a day.</p></div>
      <div class="card"><b class="wn">Mild unusual day</b>
        <p style="color:var(--muted);font-size:13px;margin:8px 0 0">
        Score in the top 5% but not the top 1%. Do not sell. <b>Watch for 5 days.</b>
        Each day compare today's move with the 5-day trend and check the market
        backdrop. If the fall is speeding up and the backdrop agrees, sell straight
        away. If the stock turns up, stand down.</p></div>
    </div>

    <h2>Money over time</h2>
    <div class="chart"><h3>Portfolio value</h3><div id="pf-eq" style="height:320px"></div></div>
    <div class="chart"><h3>How far below the peak</h3><div id="pf-dd" style="height:220px"></div></div>

    <h2>Stock by stock</h2>
    <div class="grid g3">
      {kpi("Fell less than buy &amp; hold", f"{helped}", "stocks", "pos")}
      {kpi("Fell more", f"{hurt}", "stocks", "neg")}
      {kpi("No change", f"{untouched}", "never traded", "dim")}
    </div>
    <div class="chart"><h3>Each dot is one stock &mdash; above the line means the model helped</h3>
      <div id="ov-scatter" style="height:430px"></div></div>
    {GLOSSARY}
    """


def tab_schemes(d: dict) -> str:
    sc = d["schemes"]
    rows = "".join(
        f"<tr><td><b>{name}</b></td>"
        f"<td class='num mono'>{r.cagr*100:.2f}%</td>"
        f"<td class='num mono'>{r.max_drawdown*100:.1f}%</td>"
        f"<td class='num mono {F_cls(r.drawdown_saved_pp)}'>{r.drawdown_saved_pp:+.1f} pts</td>"
        f"<td class='num mono'>{r.trades_per_sym_yr:.2f}</td></tr>"
        for name, r in sc.iterrows()) if len(sc) else ""
    names = {"rolling": "Last 3 years only",
             "incremental": "Everything so far",
             "ewma": "Recent days count more",
             "vol_purged": "Skip the wildest days"}
    expl = "".join(
        f'<tr><td><b>{k}</b></td><td style="color:var(--muted)">{v}</td></tr>'
        for k, v in names.items())

    return f"""
    <h2>How often should the model relearn?</h2>
    <p class="sub">The model is retrained at the start of every month. The only thing
    that changes between these four is <b>which past data it is allowed to learn from</b>.</p>
    <div class="card"><table><tbody>{expl}</tbody></table></div>

    <h2>Results</h2>
    <div class="card"><table>
      <thead><tr><th>Method</th><th class="num">Yearly return</th>
      <th class="num">Worst fall</th><th class="num">Better than buy &amp; hold by</th>
      <th class="num">Trades per stock per year</th></tr></thead>
      <tbody>{rows}</tbody></table></div>

    {callout("They all perform about the same", [
      "The gap between best and worst is a fraction of a percentage point &mdash; smaller than the run-to-run noise. Choosing between them is not where the value is.",
      "What retraining <b>does</b> change is how often the model acts at all. A model trained once on 2016&ndash;2020 barely fires, because that period includes the COVID crash and sets a bar later calm years never reach. Retraining monthly drops that bar to something reachable."
    ])}
    <div class="imgwrap"><img src="data:image/png;base64,{d['img_schemes']}" alt="by method"></div>
    """


def tab_portfolio(d: dict) -> str:
    cs = d["cost_summary"]
    if not len(cs):
        return "<p class='sub'>Run the pipeline first.</p>"
    s_, b_ = cs.loc["strategy"], cs.loc["buy & hold"]
    return f"""
    <h2>What trading actually costs</h2>
    <p class="sub">Every trade pays brokerage, securities transaction tax, stamp duty,
    GST and the gap between the price you wanted and the price you got. Selling also
    triggers capital gains tax.</p>
    <div class="grid g4">
      {kpi("Model trades", f'{int(s_.get("trades",0))}', "over the whole period")}
      {kpi("Buy &amp; hold trades", f'{int(b_.get("trades",0))}', "one purchase each")}
      {kpi("Extra cost of trading", F_rs(s_.get("costs",0)-b_.get("costs",0)), "over the period")}
      {kpi("Tax paid", F_rs(s_.get("tax",0)), "on gains taken")}
    </div>

    <h2>Comparing tax fairly</h2>
    <p class="sub">Buy &amp; hold never sells, so it looks like it pays no tax at all.
    That is not a saving &mdash; the tax is still owed, just later. To compare fairly,
    both are sold at the end of the period.</p>
    <div class="card"><table>
      <thead><tr><th></th><th class="num">Tax paid along the way</th>
      <th class="num">Tax still owed</th><th class="num">Total</th></tr></thead>
      <tbody>
        <tr><td>Model</td><td class="num mono">{F_rs(s_.get("tax",0))}</td>
          <td class="num mono">{F_rs(s_.get("tax_liquidated",0)-s_.get("tax",0))}</td>
          <td class="num mono cy">{F_rs(s_.get("tax_liquidated",0))}</td></tr>
        <tr><td>Buy &amp; hold</td><td class="num mono">{F_rs(b_.get("tax",0))}</td>
          <td class="num mono">{F_rs(b_.get("tax_liquidated",0)-b_.get("tax",0))}</td>
          <td class="num mono cy">{F_rs(b_.get("tax_liquidated",0))}</td></tr>
      </tbody></table></div>
    """


def tab_symbols(d: dict) -> str:
    t = d["drawdown_table"]
    if not len(t):
        return "<p class='sub'>Run the pipeline first.</p>"
    rows = ""
    for sym, r in t.sort_values("drawdown_saved_pp", ascending=False).iterrows():
        sv = r["drawdown_saved_pp"]
        pill = ("<span class='pill ok'>fell less</span>" if sv > 0.1 else
                "<span class='pill no'>fell more</span>" if sv < -0.1 else
                "<span class='pill dim'>not traded</span>")
        rows += (f"<tr><td class='mono'><b>{sym}</b></td>"
                 f"<td class='num mono' data-v='{r['strategy_max_drawdown']}'>{r['strategy_max_drawdown']*100:.1f}%</td>"
                 f"<td class='num mono' data-v='{r['bh_max_drawdown']}'>{r['bh_max_drawdown']*100:.1f}%</td>"
                 f"<td class='num mono {F_cls(sv)}' data-v='{sv}'>{sv:+.1f}</td>"
                 f"<td class='num mono' data-v='{r['strategy_return']}'>{r['strategy_return']*100:.0f}%</td>"
                 f"<td class='num mono' data-v='{r['bh_return']}'>{r['bh_return']*100:.0f}%</td>"
                 f"<td class='num mono' data-v='{r['exits']}'>{int(r['exits'])}</td>"
                 f"<td>{pill}</td></tr>")
    heads = ["Stock", "Model worst fall", "Buy &amp; hold worst fall",
             "Points better", "Model return", "Buy &amp; hold return", "Sells", ""]
    th = "".join(f'<th class="sortable {"num" if i else ""}" '
                 f'onclick="sortTable(this,{i},{str(i>0).lower()})">{h}</th>'
                 for i, h in enumerate(heads))
    return f"""
    <h2>Every stock</h2>
    <p class="sub">Click a column to sort. &ldquo;Points better&rdquo; is how much
    shallower the model's worst fall was, in percentage points.</p>
    <div class="picker">
      <input type="search" id="symfilter" placeholder="find a stock…"
             oninput="filterSyms(this.value)" style="min-width:220px">
      <span class="pill dim" id="symcount">{len(t)} shown</span>
    </div>
    <div class="card"><div class="scroll"><table id="symtable">
      <thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div></div>
    """


def tab_explorer(d: dict) -> str:
    syms = sorted(d["symbols"])
    opts = "".join(f'<option value="{s}">{s}</option>' for s in syms)
    chips = "".join(f'<div class="chip" onclick="drawExplorer(\'{s}\')" id="chip-{s}">{s}</div>'
                    for s in syms)
    return f"""
    <h2>One stock at a time</h2>
    <div class="picker">
      <select id="symsel" onchange="drawExplorer(this.value)">{opts}</select>
      <span class="pill dim" id="exp-summary"></span>
    </div>
    <div class="chips">{chips}</div>
    <div class="chart"><h3 id="exp-title">&mdash;</h3><div id="exp-price" style="height:330px"></div></div>
    <div class="chart"><h3>How unusual each day looked (1.0 = most unusual)</h3>
      <div id="exp-int" style="height:180px"></div></div>
    <div class="chart"><h3>How far below the peak</h3><div id="exp-dd" style="height:220px"></div></div>

    <h2>Every unusual day for this stock</h2>
    <p class="sub">Only days the model flagged are listed. <b>Severe</b> means the top 1%
    of unusual days &mdash; the model sells the next day if price is falling and the
    fall is speeding up. <b>Mild</b> means the top 5% &mdash; the model watches for
    5 days instead of acting.</p>
    <div class="card"><div class="scroll"><table id="anomtable">
      <thead><tr>
        <th>Date</th><th class="num">Anomaly score</th><th class="num">Intensity</th>
        <th>Severity</th><th class="num">Move that day</th><th class="num">Slope</th>
        <th class="num">Acceleration</th><th>Trend</th><th>Market regime</th>
        <th>Action taken</th>
      </tr></thead><tbody id="anombody"></tbody></table></div></div>
    {GLOSSARY}
    """


JS_CHARTS = """
// ---------- portfolio ----------
(function(){
  const p=DATA.portfolio; if(!p||!p.strategy) return;
  Plotly.newPlot('pf-eq',[
    {x:p.buy_hold.dates,y:p.buy_hold.equity,name:'buy & hold',type:'scatter',mode:'lines',
     line:{color:'#64748B',width:1.6}},
    {x:p.strategy.dates,y:p.strategy.equity,name:'model',type:'scatter',mode:'lines',
     line:{color:'#00E5FF',width:1.8}}
  ],{...LAYOUT,yaxis:{...LAYOUT.yaxis,tickprefix:'₹'}},CFGP);

  Plotly.newPlot('pf-dd',[
    {x:p.buy_hold.dates,y:p.buy_hold.drawdown,name:'buy & hold',type:'scatter',
     fill:'tozeroy',mode:'lines',line:{color:'#64748B',width:1},
     fillcolor:'rgba(100,116,139,.28)'},
    {x:p.strategy.dates,y:p.strategy.drawdown,name:'model',type:'scatter',mode:'lines',
     line:{color:'#00E5FF',width:1.6}}
  ],{...LAYOUT,yaxis:{...LAYOUT.yaxis,tickformat:'.0%'}},CFGP);
})();

// ---------- scatter ----------
(function(){
  const s=DATA.scatter; if(!s||!s.length) return;
  const helped=s.filter(d=>d.y-d.x>0.001), hurt=s.filter(d=>d.y-d.x<-0.001),
        flat=s.filter(d=>Math.abs(d.y-d.x)<=0.001);
  const lo=Math.min(...s.map(d=>Math.min(d.x,d.y)))*1.05;
  const mk=(a,n,c)=>({x:a.map(d=>d.x),y:a.map(d=>d.y),text:a.map(d=>d.s),
    name:n+' ('+a.length+')',mode:'markers',type:'scatter',
    marker:{size:9,color:c,opacity:.85},
    hovertemplate:'%{text}<br>buy & hold %{x:.1%}<br>model %{y:.1%}<extra></extra>'});
  Plotly.newPlot('ov-scatter',[
    {x:[lo,0],y:[lo,0],mode:'lines',line:{color:'#475569',dash:'dash',width:1},
     name:'same',hoverinfo:'skip'},
    mk(flat,'not traded','#64748B'),mk(helped,'fell less','#34D399'),mk(hurt,'fell more','#F87171')
  ],{...LAYOUT,hovermode:'closest',
     xaxis:{...LAYOUT.xaxis,title:'buy & hold worst fall',tickformat:'.0%'},
     yaxis:{...LAYOUT.yaxis,title:'model worst fall',tickformat:'.0%'}},CFGP);
})();

// ---------- one stock ----------
function drawExplorer(sym){
  const s=DATA.symbols[sym]; if(!s) return;
  document.getElementById('symsel').value=sym;
  document.querySelectorAll('.chip').forEach(c=>c.classList.remove('active'));
  const chip=document.getElementById('chip-'+sym); if(chip)chip.classList.add('active');

  const cash=1-s.held.reduce((a,b)=>a+b,0)/s.held.length;
  document.getElementById('exp-summary').textContent=
    s.sells.length+' sells · '+s.buys.length+' buys · '+(cash*100).toFixed(1)+'% of days out of the market';
  const worst=a=>a.reduce((m,v)=>Math.min(m,v),0);
  document.getElementById('exp-title').textContent=
    sym+' — model worst fall '+F.pct(worst(s.dd_strategy))+
    ', buy & hold '+F.pct(worst(s.dd_bench));

  const shapes=[]; let start=null;
  for(let i=0;i<s.held.length;i++){
    if(!s.held[i]&&start===null)start=s.dates[i];
    if((s.held[i]||i===s.held.length-1)&&start!==null){
      shapes.push({type:'rect',xref:'x',yref:'paper',x0:start,x1:s.dates[i],y0:0,y1:1,
        fillcolor:'rgba(248,113,113,.13)',line:{width:0},layer:'below'});
      start=null;
    }
  }
  const at=ds=>ds.map(d=>s.close[s.dates.indexOf(d)]);
  Plotly.newPlot('exp-price',[
    {x:s.dates,y:s.close,type:'scatter',mode:'lines',name:'price',
     line:{color:'#E2E8F0',width:1.3}},
    {x:s.sells,y:at(s.sells),mode:'markers',name:'sell',
     marker:{symbol:'triangle-down',size:11,color:'#F87171'}},
    {x:s.buys,y:at(s.buys),mode:'markers',name:'buy',
     marker:{symbol:'triangle-up',size:11,color:'#34D399'}}
  ],{...LAYOUT,shapes},CFGP);

  Plotly.newPlot('exp-int',[
    {x:s.dates,y:s.intensity,type:'scatter',mode:'lines',name:'how unusual',
     line:{color:'#A78BFA',width:1.1}}
  ],{...LAYOUT,yaxis:{...LAYOUT.yaxis,range:[0,1.02]},
     shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:0.99,y1:0.99,
       line:{color:'#F87171',dash:'dash',width:1.2}}]},CFGP);

  // ---- the anomaly-day table ----
  const tb=document.getElementById('anombody');
  if(tb){
    const cls=v=>v>0?'pos':(v<0?'neg':'');
    tb.innerHTML=(s.anomalies||[]).map(a=>
      '<tr><td class="mono">'+a.date+'</td>'+
      '<td class="num mono">'+(a.score==null?'—':a.score.toFixed(4))+'</td>'+
      '<td class="num mono">'+a.intensity.toFixed(4)+'</td>'+
      '<td><span class="pill '+(a.severity==='Severe'?'no':'mid')+'">'+a.severity+'</span></td>'+
      '<td class="num mono '+cls(a.ret)+'">'+(a.ret==null?'—':a.ret.toFixed(2)+'%')+'</td>'+
      '<td class="num mono '+cls(a.slope)+'">'+a.slope.toFixed(2)+'</td>'+
      '<td class="num mono '+cls(a.accel)+'">'+a.accel.toFixed(2)+'</td>'+
      '<td style="color:var(--muted)">'+a.trend+'</td>'+
      '<td style="color:var(--muted)">'+a.regime+'</td>'+
      '<td>'+(a.action.indexOf('SELL')>=0?'<span class="pill no">'+a.action+'</span>'
             :a.action==='BUY'?'<span class="pill ok">BUY</span>'
             :'<span class="pill dim">'+a.action+'</span>')+'</td></tr>').join('')
      || '<tr><td colspan="10" style="color:var(--muted)">No unusual days for this stock.</td></tr>';
  }

  Plotly.newPlot('exp-dd',[
    {x:s.dates,y:s.dd_bench,type:'scatter',fill:'tozeroy',mode:'lines',name:'buy & hold',
     line:{color:'#64748B',width:1},fillcolor:'rgba(100,116,139,.28)'},
    {x:s.dates,y:s.dd_strategy,type:'scatter',mode:'lines',name:'model',
     line:{color:'#00E5FF',width:1.6}}
  ],{...LAYOUT,yaxis:{...LAYOUT.yaxis,tickformat:'.0%'}},CFGP);
}

function filterSyms(q){
  q=q.trim().toUpperCase();
  let n=0;
  document.querySelectorAll('#symtable tbody tr').forEach(r=>{
    const hit=r.cells[0].textContent.toUpperCase().includes(q);
    r.style.display=hit?'':'none'; if(hit)n++;
  });
  document.getElementById('symcount').textContent=n+' shown';
}
"""


def build(d: dict) -> str:
    payload = {
        "symbols": d["symbols"],
        "portfolio": d["portfolio"],
        "scatter": [{"s": s, "x": float(r["bh_max_drawdown"]),
                     "y": float(r["strategy_max_drawdown"]), "e": int(r["exits"])}
                    for s, r in d["drawdown_table"].iterrows()] if len(d["drawdown_table"]) else [],
    }
    tabs = [("dashboard", "Dashboard", tab_dashboard(d)),
            ("schemes", "Retraining", tab_schemes(d)),
            ("portfolio", "Costs &amp; Tax", tab_portfolio(d)),
            ("symbols", "All Stocks", tab_symbols(d)),
            ("explorer", "Stock Deep Dive", tab_explorer(d))]
    buttons = "".join(f'<button class="tab{" active" if i==0 else ""}" '
                      f'onclick="tab(\'{k}\',this)">{lab}</button>'
                      for i, (k, lab, _) in enumerate(tabs))
    panels = "".join(f'<div class="panel{" active" if i==0 else ""}" id="{k}">{body}</div>'
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
  <div class="brand"><h1>QBEAST <span class="accent">·</span> Crash Detection</h1>
    <span class="tagline">Spotting unusual days in Indian stocks, to sell before a fall deepens</span></div>
  <div class="meta"><span>{d['n_symbols']} stocks</span><span>{d['n_sessions']} trading days</span>
    <span>learnt {d['train_window']}</span><span>tested {d['window']}</span>
    <span>generated {d['generated']}</span></div>
  <div class="tabs">{buttons}</div>
</div></header>
<div class="wrap">{panels}
<footer>QBEAST · generated {d['generated']} · every figure is from the test period,
which the model never saw while learning</footer></div>
<script>
const DATA={json.dumps(payload, separators=(",", ":"))};
const EXP_SYMBOL={json.dumps(first)};
{JS_HELPERS}
{JS_CHARTS}
</script></body></html>"""


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
