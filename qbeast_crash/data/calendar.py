"""
calendar.py
-----------
The master trading calendar.

What it does:   Produces the canonical list of trading days every symbol is
                aligned onto.
Why we do it:   The market-wide breadth signal is "what fraction of the universe
                is falling today". That question only means something if every
                symbol is measured on the same days. Without a shared calendar,
                a symbol missing a bar quietly drops out of the denominator and
                breadth drifts for reasons that have nothing to do with markets.
How (method):   NIFTY100 supplies the calendar. The index has a bar on every
                session the exchange was open, which no individual stock
                guarantees -- a stock can be suspended, halted, or simply
                untraded.
Where:          calendar.py -> trading_calendar()
"""

from __future__ import annotations

import pandas as pd

from qbeast_crash.config import DEFAULT_CONFIG, Config
from qbeast_crash.data.loader import load_index

__all__ = ["trading_calendar", "listing_mask"]


def trading_calendar(
    config: Config = DEFAULT_CONFIG,
    *,
    start=None,
    end=None,
) -> pd.DatetimeIndex:
    """
    Canonical trading days, taken from the index series.

    Deriving the calendar from the index rather than from a union of all stock
    dates matters: a union would inherit every vendor artefact in every file,
    inventing sessions that never happened.
    """
    index_frame = load_index(config)
    dates = index_frame.index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    return pd.DatetimeIndex(dates, name="date")


def listing_mask(
    frames: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Boolean panel: was this symbol tradeable on this day?

    True only between a symbol's own first and last observed bar. This is what
    keeps a 2021 listing out of the 2016 cross-section, so breadth is always
    "fraction of stocks that EXISTED and are falling" rather than a figure
    quietly diluted by companies that had not listed yet.
    """
    mask = pd.DataFrame(False, index=calendar, columns=list(frames))
    for sym, frame in frames.items():
        if frame.empty:
            continue
        live = (calendar >= frame.index[0]) & (calendar <= frame.index[-1])
        mask[sym] = live
    return mask
