"""
plots.py
--------
Phase 8: drawdown analysis and the per-stock visualisations.

What it does:   Computes drawdown statistics, and draws one chart per stock
                showing where the model sold and what it avoided.
Why we do it:   Drawdown reduction is the headline claim, so it needs to be
                measurable AND inspectable. A single summary number can hide
                almost anything; a chart per stock lets you see whether the
                exits landed where they should have.
Where:          plots.py -> drawdown_stats(), plot_symbol(), plot_portfolio()

WHY TIME UNDER WATER MATTERS AS MUCH AS DEPTH
---------------------------------------------
Maximum drawdown says how far you fell. It says nothing about how long you
stayed down, and the second is what actually decides whether a strategy is
livable -- a 20% fall recovered in a month is a different experience from the
same 20% taking three years.

Both are reported, plus recovery time, which is the reliability metric that
makes the cyber-physical framing literal rather than decorative: depth is the
magnitude of degradation, and recovery time is mean time to recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                       # no display on a headless machine
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

__all__ = ["DrawdownStats", "drawdown_series", "drawdown_stats",
           "plot_symbol", "plot_portfolio", "plot_drawdown_scatter"]

#: One palette, used everywhere, so a colour means the same thing on every
#: chart: the strategy is always blue, buy-and-hold always grey, exits red.
STRATEGY = "#1F4E79"
BENCHMARK = "#8C8C8C"
EXIT = "#B22222"
ENTRY = "#1E7B34"
GRID = "#E4E8EC"


@dataclass
class DrawdownStats:
    max_drawdown: float           # deepest fall from a peak, negative
    max_dd_start: pd.Timestamp | None
    max_dd_trough: pd.Timestamp | None
    max_dd_recovered: pd.Timestamp | None
    days_under_water: int         # sessions spent below a prior peak
    longest_underwater: int       # longest single spell
    time_to_recover: int | None   # sessions from trough back to the peak

    def as_row(self) -> dict:
        return {
            "max_drawdown": self.max_drawdown,
            "max_dd_start": self.max_dd_start,
            "max_dd_trough": self.max_dd_trough,
            "max_dd_recovered": self.max_dd_recovered,
            "days_under_water": self.days_under_water,
            "longest_underwater": self.longest_underwater,
            "time_to_recover": self.time_to_recover,
        }


def drawdown_series(equity: pd.Series) -> pd.Series:
    """
    Fractional distance below the running peak. Always <= 0.

    The running maximum is taken over the equity curve itself rather than over
    prices, so it reflects what the portfolio actually experienced.
    """
    equity = equity.dropna()
    if equity.empty:
        return equity
    return equity / equity.cummax() - 1.0


def drawdown_stats(equity: pd.Series) -> DrawdownStats:
    """Depth, duration, and recovery for one equity curve."""
    dd = drawdown_series(equity)
    if dd.empty:
        return DrawdownStats(np.nan, None, None, None, 0, 0, None)

    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()

    # First date after the trough at which the old peak was regained. None
    # means it never was, which is itself the answer worth reporting.
    after = equity.loc[trough:]
    regained = after[after >= equity.loc[peak]]
    recovered = regained.index[0] if len(regained) else None

    under = dd < -1e-9
    spells, run = [], 0
    for flag in under.to_numpy():
        if flag:
            run += 1
        elif run:
            spells.append(run)
            run = 0
    if run:
        spells.append(run)

    return DrawdownStats(
        max_drawdown=float(dd.min()),
        max_dd_start=peak,
        max_dd_trough=trough,
        max_dd_recovered=recovered,
        days_under_water=int(under.sum()),
        longest_underwater=max(spells) if spells else 0,
        time_to_recover=(
            int(equity.loc[trough:recovered].shape[0] - 1) if recovered is not None else None
        ),
    )


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _pct(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))


def plot_symbol(
    symbol: str,
    close: pd.Series,
    signals: pd.DataFrame,
    intensity: pd.Series | None = None,
    crash_onsets: pd.DatetimeIndex | None = None,
    out_dir: Path | None = None,
) -> Path | None:
    """
    Three stacked panels for one stock.

        price with exit and re-entry markers, cash periods shaded
        anomaly intensity, with the action threshold drawn
        drawdown, strategy against buy-and-hold

    The shading matters more than the markers: a red triangle tells you a sell
    fired, but the shaded band shows what the strategy was holding through, and
    that is where you can see whether an exit helped or simply missed a rebound.
    """
    close = close.dropna()
    signals = signals.reindex(close.index)
    if close.empty or signals["in_position"].isna().all():
        return None

    held = signals["in_position"].fillna(True).astype(bool)
    ret = close.pct_change().fillna(0.0)
    strat = (1 + ret * held.astype(float)).cumprod()
    bench = (1 + ret).cumprod()

    n_panels = 3 if intensity is not None else 2
    heights = [3, 1, 1.5] if intensity is not None else [3, 1.5]
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(13, 8.5), sharex=True,
        gridspec_kw={"height_ratios": heights, "hspace": 0.12},
    )

    # ---- price ------------------------------------------------------
    ax = axes[0]
    ax.plot(close.index, close.to_numpy(), color=STRATEGY, linewidth=1.1, zorder=3)

    # Shade every stretch spent in cash.
    in_cash = ~held
    if in_cash.any():
        starts = in_cash & ~in_cash.shift(1, fill_value=False)
        ends = in_cash & ~in_cash.shift(-1, fill_value=False)
        for a, b in zip(close.index[starts], close.index[ends]):
            ax.axvspan(a, b, color=EXIT, alpha=0.10, zorder=1)

    exits = signals.index[signals["action"] == "EXIT"]
    entries = signals.index[signals["action"] == "ENTER"]
    if len(exits):
        ax.scatter(exits, close.reindex(exits), marker="v", s=70, color=EXIT,
                   zorder=5, label=f"exit ({len(exits)})")
    if len(entries):
        ax.scatter(entries, close.reindex(entries), marker="^", s=70, color=ENTRY,
                   zorder=5, label=f"re-entry ({len(entries)})")

    if crash_onsets is not None:
        inside = crash_onsets[(crash_onsets >= close.index[0]) & (crash_onsets <= close.index[-1])]
        for d in inside:
            ax.axvline(d, color="black", linewidth=0.7, alpha=0.30, linestyle="--", zorder=2)
        if len(inside):
            ax.plot([], [], color="black", linewidth=0.7, alpha=0.4,
                    linestyle="--", label=f"crash onset ({len(inside)})")

    ax.set_ylabel("price")
    ax.set_title(
        f"{symbol}   strategy {(strat.iloc[-1] - 1) * 100:+.1f}%   "
        f"buy & hold {(bench.iloc[-1] - 1) * 100:+.1f}%   "
        f"in cash {100 * (1 - held.mean()):.1f}% of days",
        loc="left", fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _style(ax)

    # ---- intensity --------------------------------------------------
    if intensity is not None:
        ax = axes[1]
        vals = intensity.reindex(close.index)
        ax.plot(close.index, vals.to_numpy(), color="#2E5F8A", linewidth=0.8)
        ax.axhline(0.99, color=EXIT, linewidth=0.9, linestyle="--")
        ax.text(close.index[0], 0.99, " act threshold 0.99", va="bottom",
                fontsize=8, color=EXIT)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("intensity")
        _style(ax)

    # ---- drawdown ---------------------------------------------------
    ax = axes[-1]
    dd_s, dd_b = drawdown_series(strat), drawdown_series(bench)
    ax.fill_between(dd_b.index, dd_b.to_numpy(), 0, color=BENCHMARK, alpha=0.45,
                    label=f"buy & hold  {dd_b.min():.1%}")
    ax.plot(dd_s.index, dd_s.to_numpy(), color=STRATEGY, linewidth=1.2,
            label=f"strategy  {dd_s.min():.1%}")
    ax.set_ylabel("drawdown")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    _pct(ax)
    _style(ax)

    fig.align_ylabels(axes)
    if out_dir is None:
        plt.close(fig)
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_portfolio(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    out_dir: Path,
    name: str = "portfolio",
) -> Path:
    """Portfolio equity and drawdown, strategy against buy-and-hold."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})

    ax = axes[0]
    ax.plot(benchmark_equity.index, benchmark_equity.to_numpy(), color=BENCHMARK,
            linewidth=1.4, label="buy & hold")
    ax.plot(strategy_equity.index, strategy_equity.to_numpy(), color=STRATEGY,
            linewidth=1.4, label="strategy")
    ax.set_ylabel("equity (Rs)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e5:.1f}L"))
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Portfolio equity, net of costs and tax", loc="left",
                 fontsize=12, fontweight="bold")
    _style(ax)

    ax = axes[1]
    dd_s, dd_b = drawdown_series(strategy_equity), drawdown_series(benchmark_equity)
    ax.fill_between(dd_b.index, dd_b.to_numpy(), 0, color=BENCHMARK, alpha=0.45,
                    label=f"buy & hold  {dd_b.min():.1%}")
    ax.plot(dd_s.index, dd_s.to_numpy(), color=STRATEGY, linewidth=1.3,
            label=f"strategy  {dd_s.min():.1%}")
    ax.set_ylabel("drawdown")
    ax.legend(loc="lower left", frameon=False)
    _pct(ax)
    _style(ax)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_drawdown_scatter(table: pd.DataFrame, out_dir: Path) -> Path:
    """
    Every symbol as one point: buy-and-hold drawdown against the strategy's.

    Both axes are negative, so a SHALLOWER strategy drawdown (-36% against
    -52%) plots ABOVE the diagonal. Getting that direction wrong is easy and
    the chart would still look plausible, so the axis is labelled explicitly.

    Points are split three ways rather than two. Lumping "unchanged" in with
    "deeper" would hide the most important fact about this strategy: it does
    nothing at all for most symbols, and a two-way split would present that
    inaction as failure.
    """
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    x = table["bh_max_drawdown"] * 100
    y = table["strategy_max_drawdown"] * 100

    lo = min(x.min(), y.min()) * 1.05
    ax.plot([lo, 0], [lo, 0], color=BENCHMARK, linewidth=1.0, linestyle="--")

    diff = y - x                        # positive = shallower = helped
    shallower = diff > 0.1
    deeper = diff < -0.1
    unchanged = ~(shallower | deeper)

    ax.scatter(x[unchanged], y[unchanged], s=26, color=BENCHMARK, alpha=0.55,
               label=f"unchanged - never traded ({int(unchanged.sum())})")
    ax.scatter(x[shallower], y[shallower], s=30, color=ENTRY, alpha=0.85,
               label=f"shallower ({int(shallower.sum())})")
    ax.scatter(x[deeper], y[deeper], s=30, color=EXIT, alpha=0.85,
               label=f"deeper ({int(deeper.sum())})")

    ax.set_xlabel("buy & hold max drawdown (%)")
    ax.set_ylabel("strategy max drawdown (%)")
    ax.set_title("Per-symbol drawdown\nabove the diagonal = shallower = the model helped",
                 loc="left", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _style(ax)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "drawdown_scatter.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
