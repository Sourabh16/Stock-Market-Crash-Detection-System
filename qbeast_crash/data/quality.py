"""
quality.py
----------
The data quality gate.

What it does:   Re-runs the Phase 0 audit as a set of pass/fail checks against
                the CLEANED data, and refuses to let bad data through.
Why we do it:   Phase 0 found five defects by hand. Hand-run audits rot -- the
                data gets refreshed, a vendor changes a format, and six weeks
                later a backtest is quietly wrong. Encoding the audit as a gate
                means a regression fails loudly instead of producing a slightly
                different Sharpe ratio nobody questions.
How (method):   Each check returns a QualityCheck with a measured value. Checks
                are either ERROR (stop) or WARN (record and continue).
Where:          quality.py -> run_quality_gate()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qbeast_crash.config import DEFAULT_CONFIG, Config

__all__ = ["QualityCheck", "QualityReport", "run_quality_gate"]

ERROR, WARN = "ERROR", "WARN"


@dataclass
class QualityCheck:
    name: str
    passed: bool
    severity: str
    detail: str

    def __str__(self) -> str:
        mark = "PASS" if self.passed else ("FAIL" if self.severity == ERROR else "WARN")
        return f"[{mark:4s}] {self.name}: {self.detail}"


@dataclass
class QualityReport:
    checks: list[QualityCheck]

    @property
    def failed_errors(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed and c.severity == ERROR]

    @property
    def warnings(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed and c.severity == WARN]

    @property
    def ok(self) -> bool:
        return not self.failed_errors

    def render(self) -> str:
        return "\n".join(str(c) for c in self.checks)


def run_quality_gate(
    frames: dict[str, pd.DataFrame],
    reports: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: Config = DEFAULT_CONFIG,
    *,
    raise_on_error: bool = True,
) -> QualityReport:
    """
    Validate the cleaned universe. Raises RuntimeError on any ERROR-level
    failure unless raise_on_error is False.
    """
    checks: list[QualityCheck] = []
    add = checks.append
    dcfg = config.data

    # --- structural -------------------------------------------------------
    add(QualityCheck(
        "universe_size", len(frames) >= 80, ERROR,
        f"{len(frames)} usable symbols loaded (need >= 80)",
    ))
    add(QualityCheck(
        "calendar_nonempty", len(calendar) > 1000, ERROR,
        f"{len(calendar)} trading days in master calendar",
    ))

    # --- no residual oversized gaps after truncation -----------------------
    # If this fails, _find_first_valid_bar missed a stitched region and
    # fabricated bars are still in the panel.
    offenders = {}
    for sym, frame in frames.items():
        if len(frame) < 2:
            continue
        gaps = frame.index.to_series().diff().dt.days
        n = int((gaps > dcfg.max_gap_days).sum())
        if n:
            offenders[sym] = n
    add(QualityCheck(
        "no_residual_gaps", not offenders, ERROR,
        "no calendar gaps remain after truncation" if not offenders
        else f"{len(offenders)} symbols still contain gaps: {dict(list(offenders.items())[:5])}",
    ))

    # --- dates monotonic and unique ---------------------------------------
    bad_order = [s for s, f in frames.items() if not f.index.is_monotonic_increasing]
    add(QualityCheck(
        "dates_sorted", not bad_order, ERROR,
        "all symbol indices sorted" if not bad_order else f"unsorted: {bad_order[:5]}",
    ))
    dupes = [s for s, f in frames.items() if f.index.has_duplicates]
    add(QualityCheck(
        "dates_unique", not dupes, ERROR,
        "no duplicate dates" if not dupes else f"duplicates in: {dupes[:5]}",
    ))

    # --- prices sane ------------------------------------------------------
    bad_price = [s for s, f in frames.items() if not f.empty and (f["close"] <= 0).any()]
    add(QualityCheck(
        "prices_positive", not bad_price, ERROR,
        "all closes > 0" if not bad_price else f"non-positive closes in: {bad_price[:5]}",
    ))

    # high >= max(open, close) and low <= min(open, close) must hold by
    # definition. A violation means the OHLC fields are scrambled.
    ohlc_bad = []
    for sym, frame in frames.items():
        if frame.empty:
            continue
        hi_ok = frame["high"] >= frame[["open", "close", "low"]].max(axis=1) - 1e-9
        lo_ok = frame["low"] <= frame[["open", "close", "high"]].min(axis=1) + 1e-9
        if not (hi_ok.all() and lo_ok.all()):
            ohlc_bad.append((sym, int((~hi_ok).sum() + (~lo_ok).sum())))
    add(QualityCheck(
        "ohlc_consistent", not ohlc_bad, ERROR,
        "high/low bracket open/close everywhere" if not ohlc_bad
        else f"{len(ohlc_bad)} symbols violate OHLC ordering: {ohlc_bad[:5]}",
    ))

    # --- end-date alignment -----------------------------------------------
    ends = pd.Series({s: f.index[-1] for s, f in frames.items() if not f.empty})
    span_days = (ends.max() - ends.min()).days if len(ends) else 0
    add(QualityCheck(
        "end_dates_aligned", span_days <= 7, ERROR,
        f"last-bar dates span {span_days} days ({ends.min().date()} to {ends.max().date()})",
    ))

    # --- coverage over the train window -----------------------------------
    train_start = pd.Timestamp(config.windows.train_start)
    n_train = sum(1 for f in frames.values() if not f.empty and f.index[0] <= train_start)
    add(QualityCheck(
        "train_window_coverage", n_train >= 80, ERROR,
        f"{n_train} symbols have history at {train_start.date()}",
    ))

    # --- WARN level: known, tolerated defects -----------------------------
    flat_heavy = {
        s: int(f["is_flat"].sum()) for s, f in frames.items()
        if not f.empty and f["is_flat"].mean() > 0.10
    }
    add(QualityCheck(
        "flat_bar_burden", not flat_heavy, WARN,
        "no symbol exceeds 10% zero-range bars" if not flat_heavy
        else f"{len(flat_heavy)} symbols >10% flat bars (masked from range features): "
             f"{dict(sorted(flat_heavy.items(), key=lambda kv: -kv[1])[:5])}",
    ))

    truncated = reports[reports["truncated_rows"] > 0] if "truncated_rows" in reports else pd.DataFrame()
    add(QualityCheck(
        "prelisting_truncation", truncated.empty, WARN,
        "no fabricated pre-listing history found" if truncated.empty
        else f"{len(truncated)} symbols truncated: "
             f"{dict(truncated['truncated_rows'].nlargest(5))}",
    ))

    dropped = reports[reports["usable"] == False] if "usable" in reports else pd.DataFrame()
    add(QualityCheck(
        "symbols_dropped", dropped.empty, WARN,
        "every symbol usable" if dropped.empty
        else f"{len(dropped)} dropped for insufficient history: {list(dropped.index)}",
    ))

    report = QualityReport(checks)
    if raise_on_error and not report.ok:
        raise RuntimeError(
            "data quality gate FAILED:\n"
            + "\n".join(str(c) for c in report.failed_errors)
        )
    return report
