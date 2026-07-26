"""
labels.py
---------
Phase 4: crash/rally ground truth, and lead-time measurement.

What it does:   Defines what counts as a crash, finds when each one started,
                and measures how many trading days before that the detector
                raised the alarm.
Why we do it:   Everything so far shows the signal fires when crashes are NEAR.
                None of it shows how EARLY. That is the requirement, and it is
                the thing the project turns on.
Where:          labels.py -> label_events(), lead_time_report()

THE LABEL IS EVALUATION-ONLY
----------------------------
Nothing in this module is imported by model.py, and nothing here ever touches
training. That separation is structural, not a convention.

The reason is that Isolation Forest is unsupervised. If the crash threshold
were derived from the model, the model would be defining the event it is then
scored on detecting, and the evaluation would be circular -- it would always
look good, and the number would mean nothing.

WHY ABSOLUTE MAGNITUDE, NOT VOLATILITY-RELATIVE
-----------------------------------------------
A crash is defined as a forward drawdown of a fixed percentage, not as a move
of N standard deviations. Drawdown is what actually hurts a portfolio, and it
is absolute: losing 30% hurts identically whether it happened in a calm year or
a wild one.

The alternative was tested and rejected. Bollinger band position is
volatility-relative, and during a sustained crash the band widens faster than
price falls, so the reading becomes LESS extreme as the drawdown deepens.
Measured on NIFTY100 in 2020: -9% from peak scored -2.85 SD, but -38% from peak
scored only -2.01 SD. A label built on that would systematically under-count
exactly the events that matter most.

WHY EVENTS ARE WINDOWS, NOT DAYS
--------------------------------
A crash is a process. Most begin with a wobble -- a day or two of unease -- and
only then break. A detector that fires on the wobble should be credited with an
early warning, not penalised for being wrong about the specific day. So events
are intervals, and lead time is measured to the interval's ONSET.
"""

from __future__ import annotations

from dataclasses import dataclass

from qbeast_crash.config import DEFAULT_CONFIG, LabelConfig

import numpy as np
import pandas as pd

__all__ = [
    "LabelConfig",
    "EventTable",
    "forward_drawdown",
    "forward_runup",
    "label_events",
    "lead_time_report",
    "LEAD_BUCKETS",
]

#: Reporting buckets for the lead-time distribution. Reported as a histogram
#: rather than a mean, because a mean silently hides the events that were
#: missed entirely -- and those are the ones that cost money.
LEAD_BUCKETS = (
    ("10+ days", 10, 10_000),
    ("5-9 days", 5, 9),
    ("2-4 days", 2, 4),      # the target band
    ("0-1 days", 0, 1),      # coincident; still usable for a T+1 exit
)


@dataclass
class EventTable:
    """Crash/rally events for one symbol, plus the per-day label series."""

    symbol: str
    crash_days: pd.Series          # bool, indexed by date
    rally_days: pd.Series
    crash_onsets: pd.DatetimeIndex
    rally_onsets: pd.DatetimeIndex


def forward_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    """
    Worst return over the NEXT `horizon` trading days.

    Deliberately the minimum over the window rather than the close-to-close
    return. A stock that falls 8% and recovers to -1% by day five has still put
    the portfolio through an 8% drawdown, and a close-to-close measure would
    score that as a quiet week.
    """
    values = close.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(n - 1):
        window = values[i + 1: i + 1 + horizon]
        if len(window):
            out[i] = window.min() / values[i] - 1.0
    return pd.Series(out, index=close.index)


def forward_runup(close: pd.Series, horizon: int) -> pd.Series:
    """Best return over the next `horizon` trading days. Mirror of the above."""
    values = close.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(n - 1):
        window = values[i + 1: i + 1 + horizon]
        if len(window):
            out[i] = window.max() / values[i] - 1.0
    return pd.Series(out, index=close.index)


def _onsets(flags: pd.Series, min_gap: int) -> pd.DatetimeIndex:
    """
    Collapse a boolean series into distinct event start dates.

    An onset is the first day of a run of flagged days. Runs separated by fewer
    than `min_gap` trading days are treated as one continuing episode rather
    than two, so a choppy decline is not counted as five separate crashes.
    """
    flagged = np.flatnonzero(flags.fillna(False).to_numpy())
    if flagged.size == 0:
        return pd.DatetimeIndex([])

    # Walk the flagged positions and keep one start per episode: a flagged day
    # begins a NEW event only if it is at least min_gap sessions after the last
    # one we kept. Everything closer is the same episode continuing.
    keep: list[int] = []
    last = -10_000
    for position in flagged:
        if position - last >= min_gap:
            keep.append(int(position))
            last = position
    return pd.DatetimeIndex(flags.index[keep])


def label_events(
    close: pd.Series,
    symbol: str = "",
    definition: LabelConfig | None = None,
) -> EventTable:
    """Label crash and rally events for one symbol. Never used in training."""
    d = definition or DEFAULT_CONFIG.labels

    dd = forward_drawdown(close, d.horizon)
    ru = forward_runup(close, d.horizon)

    crash_days = dd <= d.crash_threshold
    rally_days = ru >= d.rally_threshold

    return EventTable(
        symbol=symbol,
        crash_days=crash_days,
        rally_days=rally_days,
        crash_onsets=_onsets(crash_days, d.min_gap),
        rally_onsets=_onsets(rally_days, d.min_gap),
    )


def lead_time_report(
    signals: pd.Series,
    onsets: pd.DatetimeIndex,
    calendar: pd.DatetimeIndex,
    max_lookback: int | None = None,
) -> dict:
    """
    How many trading days before each onset did the first signal fire?

    Parameters
    ----------
    signals      : boolean series indexed by date -- True where the rule fired.
    onsets       : event start dates.
    calendar     : the trading calendar, so lead is counted in SESSIONS rather
                   than calendar days. A weekend is not two days of warning.
    max_lookback : how far back to search. Beyond this an alert is more likely
                   coincidence than warning, given the base alert rate.

    Returns
    -------
    dict with the lead distribution, hit rates by horizon, and false-alarm count.

    A lead of 0 still means the alert arrived BEFORE the decline: the label is
    forward-looking, so day t is the last day before the fall. In trading terms
    that is a one-day warning -- enough to exit at the next open.
    """
    if max_lookback is None:
        max_lookback = DEFAULT_CONFIG.labels.max_lookback
    positions = {d: i for i, d in enumerate(calendar)}
    fired = set(signals[signals.fillna(False)].index)

    leads: list[int | None] = []
    for onset in onsets:
        if onset not in positions:
            continue
        i = positions[onset]
        lead = None
        for back in range(0, max_lookback + 1):
            j = i - back
            if j < 0:
                break
            if calendar[j] in fired:
                lead = back          # keep searching for an EARLIER alert
        leads.append(lead)

    caught = [x for x in leads if x is not None]
    n_events = len(leads)

    distribution = {}
    for name, lo, hi in LEAD_BUCKETS:
        distribution[name] = sum(1 for x in caught if lo <= x <= hi)
    distribution["missed"] = n_events - len(caught)

    hit_rate = {}
    for horizon in (0, 1, 2, 3, 5, 10):
        hit_rate[horizon] = (
            sum(1 for x in caught if x <= horizon) / n_events if n_events else np.nan
        )

    # A signal is a false alarm if no onset follows within max_lookback days.
    onset_positions = {positions[o] for o in onsets if o in positions}
    false_alarms = 0
    for date in fired:
        if date not in positions:
            continue
        i = positions[date]
        if not any(i <= p <= i + max_lookback for p in onset_positions):
            false_alarms += 1

    return {
        "n_events": n_events,
        "n_caught": len(caught),
        "n_signals": len(fired),
        "false_alarms": false_alarms,
        "distribution": distribution,
        "hit_rate": hit_rate,
        "median_lead": float(np.median(caught)) if caught else np.nan,
        "leads": leads,
    }


#: Backwards-compatible alias -- the type now lives in config.py, which is
#: the single source of truth for every threshold in the pipeline.
CrashDefinition = LabelConfig
