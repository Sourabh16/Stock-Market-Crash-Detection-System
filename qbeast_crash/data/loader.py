"""
loader.py
---------
Raw CSV -> trustworthy per-symbol price frame.

What it does:   Loads one symbol, repairs the six known data defects, and
                returns a frame whose every row can be believed.
Why we do it:   The raw files contain fabricated pre-IPO history, a fake
                adjusted-close column, zero-range bars, and ragged end dates.
                None of these announce themselves -- they silently produce
                plausible-looking returns that never happened.
How (method):   Each defect gets one explicit, tested rule. Nothing is repaired
                by inference or by filling.
Where:          loader.py -> load_symbol(), load_universe()

THE SIX DEFECTS (measured)
--------------------------
1. `adj_close` is not an adjusted series. It differs from `close` on ~30 of
   ~5,800 bars per symbol; a real adjustment factor differs on EVERY bar before
   the last corporate action. `close` is already back-adjusted, so we use it
   and discard `adj_close`.

2. Fabricated pre-IPO history. Post-2016 listings carry monthly-spaced bars in
   front of their real listing date (MAZDOCK, DMART, SBILIFE, VBL). Detected as
   a calendar gap > max_gap_days and truncated. This rule recovers actual NSE
   listing dates to the day -- see tests.

3. Zero-range bars (open == high == low == close). BAJFINANCE has 1,092 of
   5,803. Flagged, never dropped.

4. Ragged end dates spanning 2026-06-03 to 2026-06-22. Trimmed to a common
   hard_end so the cross-section stays square.

5. Non-positive and duplicate-date rows. Removed with a warning.

6. Unadjusted corporate actions. `close` is back-adjusted for splits and
   bonuses but NOT for demergers. CGPOWER falls 155.30 -> 53.05 on 2016-03-15
   (Crompton Greaves Consumer demerger) and ADANIENT falls 78% on 2015-06-04
   (Adani Ports/Transmission/Power spin-off). Both are internally consistent
   bars that pass every structural check -- only the RETURN betrays them.
   Found in Phase 1 testing, after the original audit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qbeast_crash.config import (
    DATA_RAW,
    DEFAULT_CONFIG,
    INDEX_SYMBOL,
    RAW_DATE_FORMAT,
    Config,
)

__all__ = ["SymbolReport", "list_symbols", "load_symbol", "load_universe", "load_index"]


@dataclass
class SymbolReport:
    """What the loader had to repair. Surfaced so nothing is fixed in silence."""

    symbol: str
    raw_rows: int
    kept_rows: int
    first_valid: pd.Timestamp | None
    truncated_rows: int          # fabricated pre-IPO bars removed
    n_gaps: int                  # calendar gaps that triggered truncation
    n_corp_actions: int          # unadjusted corporate-action breaks found
    flat_bars: int
    zero_volume: int
    dropped_bad_price: int
    dropped_duplicates: int
    usable: bool
    reason: str = ""

    def as_row(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def list_symbols(include_index: bool = False) -> list[str]:
    """Every symbol with a CSV in data/raw, sorted."""
    names = sorted(p.stem for p in DATA_RAW.glob("*.csv"))
    if not include_index:
        names = [n for n in names if n != INDEX_SYMBOL]
    return names


def _find_corporate_action_break(
    close: pd.Series,
    max_abs_log_return: float,
) -> tuple[pd.Timestamp | None, int]:
    """
    Locate unadjusted corporate actions (demergers, spin-offs, missed splits).

    The `close` series is back-adjusted for splits and bonuses but NOT for
    capital restructurings. CGPOWER falls from 155.30 to 53.05 on 2016-03-15
    and stays there -- the Crompton Greaves Consumer demerger. No shareholder
    lost 66%; they received shares in the demerged entity. The return is an
    artefact of the price series, not an event in the market.

    This matters more than its rarity suggests. Isolation Forest is
    unsupervised: it learns "normal" from whatever it is shown. A -66% day
    inside the training window becomes the most extreme point in the sample and
    drags the anomaly boundary out towards it, making genuine crashes look
    comparatively ordinary. One bad bar degrades every score that follows.

    Handled the same way as fabricated pre-listing history: everything before
    the break is a different security and is discarded. We deliberately do not
    try to re-adjust prior prices -- the correct factor depends on the demerger
    ratio, which is not in this data, and inferring it from the price jump
    would assume exactly what we are trying to detect.

    Returns (first_valid_date, n_breaks). None if the series is clean.
    """
    if len(close) < 2:
        return None, 0

    log_ret = np.log(close / close.shift(1))
    breaks = np.flatnonzero(log_ret.abs().to_numpy() > max_abs_log_return)
    if breaks.size == 0:
        return None, 0
    # The bar AT the last break is the first of the restructured security.
    return close.index[breaks[-1]], int(breaks.size)


def _find_first_valid_bar(dates: pd.DatetimeIndex, max_gap_days: int) -> tuple[pd.Timestamp, int]:
    """
    Locate the true start of a symbol's daily history.

    A gap longer than max_gap_days calendar days cannot be a holiday -- NSE has
    never closed that long. It means the vendor stitched sparse data in front of
    the real listing. Everything up to and including the LAST such gap is
    fabricated, so the first genuine bar is the one immediately after it.

    Returns (first_valid_date, number_of_gaps_found).
    """
    if len(dates) < 2:
        return (dates[0] if len(dates) else pd.NaT), 0

    deltas = dates.to_series().diff().dt.days
    gap_positions = np.flatnonzero(deltas.to_numpy() > max_gap_days)
    if gap_positions.size == 0:
        return dates[0], 0
    # The bar AFTER the last oversized gap is the first trustworthy one.
    return dates[gap_positions[-1]], int(gap_positions.size)


def load_symbol(
    symbol: str,
    config: Config = DEFAULT_CONFIG,
    *,
    strict: bool = False,
) -> tuple[pd.DataFrame, SymbolReport]:
    """
    Load and clean one symbol.

    Returns (frame, report). The frame is indexed by date and carries the OHLCV
    columns plus boolean quality flags:

        is_flat      open == high == low == close (no intraday range)
        is_zero_vol  volume == 0
        gap_filled   vendor marked this bar as interpolated

    Downstream code must treat these as first-class: a flat bar is a real row
    with an unusable range, not a normal bar.
    """
    dcfg = config.data
    path = DATA_RAW / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"no CSV for symbol {symbol!r} at {path}")

    raw = pd.read_csv(path)
    raw_rows = len(raw)

    # The index file uses `adjusted_close`; stocks use `adj_close`. Neither is
    # used -- normalised only so one code path handles both schemas.
    raw = raw.rename(columns={"adjusted_close": "adj_close"})

    raw["date"] = pd.to_datetime(raw["date"], format=RAW_DATE_FORMAT)
    raw = raw.sort_values("date")

    before = len(raw)
    raw = raw.drop_duplicates(subset="date", keep="last")
    dropped_duplicates = before - len(raw)

    # A non-positive or missing price is a data error, not a market event.
    price_cols = ["open", "high", "low", "close"]
    valid_price = (raw[price_cols] > 0).all(axis=1) & raw[price_cols].notna().all(axis=1)
    dropped_bad_price = int((~valid_price).sum())
    raw = raw[valid_price]

    if raw.empty:
        return _empty_frame(), SymbolReport(
            symbol, raw_rows, 0, None, 0, 0, 0, 0, 0,
            dropped_bad_price, dropped_duplicates, False, "no valid price rows",
        )

    raw = raw.set_index("date")

    # --- defect 2: fabricated pre-IPO history -------------------------------
    first_valid, n_gaps = _find_first_valid_bar(raw.index, dcfg.max_gap_days)

    # --- defect 6: unadjusted corporate actions -----------------------------
    # Whichever break is later bounds the usable history: a symbol can have
    # both stitched pre-listing bars and a later restructuring.
    ca_date, n_breaks = _find_corporate_action_break(
        raw["close"], dcfg.max_abs_log_return
    )
    if ca_date is not None and ca_date > first_valid:
        first_valid = ca_date

    truncated_rows = int((raw.index < first_valid).sum())
    raw = raw.loc[raw.index >= first_valid]

    # --- defect 4: ragged end dates -----------------------------------------
    raw = raw.loc[raw.index <= pd.Timestamp(dcfg.hard_end)]

    keep = [c for c in dcfg.keep_columns if c in raw.columns]
    out = raw[keep].copy()

    # --- defect 3: zero-range bars ------------------------------------------
    out["is_flat"] = (
        (out["open"] == out["high"])
        & (out["high"] == out["low"])
        & (out["low"] == out["close"])
    )
    out["is_zero_vol"] = out["volume"].fillna(0) == 0
    out["gap_filled"] = (
        out["_gap_filled"].astype(bool) if "_gap_filled" in out else False
    )

    usable = len(out) >= dcfg.min_bars
    reason = "" if usable else f"only {len(out)} bars (need {dcfg.min_bars})"

    report = SymbolReport(
        symbol=symbol,
        raw_rows=raw_rows,
        kept_rows=len(out),
        first_valid=first_valid,
        truncated_rows=truncated_rows,
        n_gaps=n_gaps,
        n_corp_actions=n_breaks,
        flat_bars=int(out["is_flat"].sum()),
        zero_volume=int(out["is_zero_vol"].sum()),
        dropped_bad_price=dropped_bad_price,
        dropped_duplicates=dropped_duplicates,
        usable=usable,
        reason=reason,
    )

    if strict and not usable:
        raise ValueError(f"{symbol}: {reason}")
    if truncated_rows:
        cause = (
            f"unadjusted corporate action on {ca_date.date()}"
            if ca_date is not None and ca_date == first_valid
            else f"{n_gaps} calendar gaps"
        )
        warnings.warn(
            f"{symbol}: dropped {truncated_rows} bars before "
            f"{first_valid.date()} ({cause})",
            stacklevel=2,
        )
    return out, report


def load_index(config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """The NIFTY100 index frame. Volume is zero throughout, as expected."""
    frame, _ = load_symbol(INDEX_SYMBOL, config)
    return frame


def load_universe(
    symbols: list[str] | None = None,
    config: Config = DEFAULT_CONFIG,
    *,
    usable_only: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load every symbol.

    Returns (frames_by_symbol, report_table). The report table is the audit
    trail -- it is written to disk each run so a data regression shows up as a
    diff rather than as a mysteriously different backtest.
    """
    symbols = symbols or list_symbols()
    frames: dict[str, pd.DataFrame] = {}
    reports: list[dict] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sym in symbols:
            frame, report = load_symbol(sym, config)
            reports.append(report.as_row())
            if report.usable or not usable_only:
                frames[sym] = frame

    return frames, pd.DataFrame(reports).set_index("symbol")


def build_close_panel(
    frames: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align every symbol's close onto one master calendar -- a wide panel with
    dates as rows and symbols as columns.

    Missing bars stay NaN. They are deliberately NOT forward-filled: a
    forward-filled price produces a fabricated zero return, which reads to the
    model as a day of unnatural calm and is exactly the kind of quiet lie that
    survives every sanity check. NaN propagates visibly instead.

    Values outside a symbol's own listed life stay NaN too, so a stock that
    listed in 2021 contributes nothing to 2016 breadth.
    """
    panel = pd.DataFrame(index=calendar)
    for sym, frame in frames.items():
        panel[sym] = frame["close"].reindex(calendar)
    return panel


def _empty_frame() -> pd.DataFrame:
    cols = ["open", "high", "low", "close", "volume", "is_flat", "is_zero_vol", "gap_filled"]
    return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"))
