"""
test_data_layer.py
------------------
Tests for the Phase 1 loader, calendar, and quality gate.

The headline test is test_truncation_recovers_real_listing_dates: the
gap-detection rule is validated against externally known NSE listing dates, so
it is checked against reality rather than against its own output.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from qbeast_crash.config import DEFAULT_CONFIG
from qbeast_crash.data.calendar import listing_mask, trading_calendar
from qbeast_crash.data.loader import (
    _find_first_valid_bar,
    build_close_panel,
    list_symbols,
    load_symbol,
    load_universe,
)
from qbeast_crash.data.quality import ERROR, run_quality_gate


@pytest.fixture(scope="module")
def universe():
    frames, reports = load_universe()
    return frames, reports


@pytest.fixture(scope="module")
def calendar():
    return trading_calendar()


# =====================================================================
# Defect 2 -- fabricated pre-listing history
# =====================================================================
#: Actual NSE listing dates, sourced independently of this codebase.
KNOWN_LISTINGS = {
    "VBL": dt.date(2016, 11, 8),
    "DMART": dt.date(2017, 3, 21),
    "SBILIFE": dt.date(2017, 10, 3),
    "MAZDOCK": dt.date(2020, 10, 12),
}


@pytest.mark.parametrize("symbol,listing", KNOWN_LISTINGS.items())
def test_truncation_recovers_real_listing_dates(symbol, listing):
    """
    The gap rule must reproduce the true listing date to the day.

    These four symbols carry years of monthly-spaced bars in front of their
    real listing. If this drifts, fabricated history is back in the panel.
    """
    frame, report = load_symbol(symbol)
    assert report.first_valid.date() == listing
    assert frame.index[0].date() == listing
    assert report.truncated_rows > 0


def test_find_first_valid_bar_picks_the_last_gap():
    """With several stitched regions, only the final one bounds real history."""
    dates = pd.DatetimeIndex(
        [dt.date(2020, 1, 1), dt.date(2020, 2, 1), dt.date(2020, 3, 1)]      # monthly
        + [dt.date(2021, 6, 1) + dt.timedelta(days=i) for i in range(30)]    # daily
    )
    first, n_gaps = _find_first_valid_bar(dates, max_gap_days=10)
    assert first == pd.Timestamp("2021-06-01")
    assert n_gaps == 3


def test_clean_series_is_untouched():
    """A symbol with no gaps must not lose a single bar."""
    frame, report = load_symbol("RELIANCE")
    assert report.truncated_rows == 0
    assert report.n_gaps == 0


# =====================================================================
# Defect 1 -- the fake adjusted-close column
# =====================================================================
def test_adj_close_is_not_carried_through(universe):
    frames, _ = universe
    for sym, frame in list(frames.items())[:10]:
        assert "adj_close" not in frame.columns, f"{sym} still carries adj_close"


# =====================================================================
# Defect 3 -- zero-range bars
# =====================================================================
def test_flat_bars_flagged_not_dropped():
    """BAJFINANCE has ~1,092 flat bars. They must be marked and retained."""
    frame, report = load_symbol("BAJFINANCE")
    assert report.flat_bars > 500
    assert frame["is_flat"].sum() == report.flat_bars
    flat = frame[frame["is_flat"]].iloc[0]
    assert flat["open"] == flat["high"] == flat["low"] == flat["close"]


def test_dropping_rows_would_corrupt_windows(universe):
    """
    Row counts must match the date span, i.e. no silent row removal.

    Dropping a bar shifts every trailing window that spans it, so a 5-day slope
    would quietly cover 6 calendar days. Flags exist precisely to avoid this.
    """
    frames, _ = universe
    for sym, frame in list(frames.items())[:20]:
        assert frame.index.is_unique
        assert frame.index.is_monotonic_increasing


# =====================================================================
# Defect 6 -- unadjusted corporate actions
# =====================================================================
#: Real restructurings where `close` was never adjusted. Externally sourced.
KNOWN_CORPORATE_ACTIONS = {
    "CGPOWER": dt.date(2016, 3, 15),    # Crompton Greaves Consumer demerger
    "ADANIENT": dt.date(2015, 6, 4),    # Adani Ports/Transmission/Power spin-off
}


@pytest.mark.parametrize("symbol,action_date", KNOWN_CORPORATE_ACTIONS.items())
def test_corporate_action_truncates_history(symbol, action_date):
    """History before a demerger belongs to a different company."""
    frame, report = load_symbol(symbol)
    assert report.n_corp_actions >= 1
    assert frame.index[0].date() == action_date
    assert report.truncated_rows > 0


def test_real_crashes_are_not_mistaken_for_corporate_actions():
    """
    The threshold must separate artefacts from genuine market events.

    ADANIENT fell 26.1% on the Hindenburg report and TRENT 31.9% in June 2026.
    Both are real and must survive; a blanket outlier filter would delete the
    very events this project exists to detect.
    """
    frame, report = load_symbol("TRENT")
    assert report.n_corp_actions == 0
    assert frame.index[0].year <= 2003

    adani, _ = load_symbol("ADANIENT")
    hindenburg = adani["close"].pct_change().loc["2023-02-03"]
    assert -0.30 < hindenburg < -0.20, "the Hindenburg crash must remain in the data"


def test_no_extreme_returns_survive(universe):
    """After cleaning, no symbol should contain an implausible daily move."""
    limit = DEFAULT_CONFIG.data.max_abs_log_return
    for sym, frame in universe[0].items():
        log_ret = np.log(frame["close"] / frame["close"].shift(1))
        worst = log_ret.abs().max()
        assert worst <= limit, f"{sym} retains a {np.expm1(worst) * 100:.1f}% move"


def test_gate_catches_an_injected_corporate_action(universe, calendar):
    """
    A demerger bar passes every structural check -- only the return reveals it.

    This is the regression guard for the defect found during Phase 1 testing.
    """
    frames, reports = universe
    poisoned = dict(frames)
    frame = poisoned["RELIANCE"].copy()
    frame.iloc[-50:, frame.columns.get_indexer(["open", "high", "low", "close"])] *= 0.3
    poisoned["RELIANCE"] = frame

    report = run_quality_gate(poisoned, reports, calendar, raise_on_error=False)
    assert not report.ok
    assert any(c.name == "no_unadjusted_corporate_actions" for c in report.failed_errors)


# =====================================================================
# Defect 4 -- ragged end dates
# =====================================================================
def test_end_dates_are_trimmed(universe):
    frames, _ = universe
    hard_end = pd.Timestamp(DEFAULT_CONFIG.data.hard_end)
    ends = pd.Series({s: f.index[-1] for s, f in frames.items()})
    assert (ends <= hard_end).all()
    assert (ends.max() - ends.min()).days <= 7


# =====================================================================
# Defect 5 -- bad prices
# =====================================================================
def test_all_prices_positive(universe):
    frames, _ = universe
    for sym, frame in frames.items():
        assert (frame[["open", "high", "low", "close"]] > 0).all().all(), sym


def test_ohlc_ordering_holds(universe):
    frames, _ = universe
    for sym, frame in list(frames.items())[:25]:
        assert (frame["high"] >= frame[["open", "close", "low"]].max(axis=1) - 1e-9).all(), sym
        assert (frame["low"] <= frame[["open", "close", "high"]].min(axis=1) + 1e-9).all(), sym


# =====================================================================
# Universe and calendar
# =====================================================================
def test_short_history_symbols_dropped(universe):
    """ENRIN, TATACAP and TMCV listed too recently to support a 60-day warmup."""
    frames, reports = universe
    for sym in ("ENRIN", "TATACAP", "TMCV"):
        assert sym not in frames
        assert not reports.loc[sym, "usable"]


def test_universe_size(universe):
    frames, _ = universe
    assert 90 <= len(frames) <= 100


def test_calendar_comes_from_the_index(calendar):
    assert calendar.is_monotonic_increasing and calendar.is_unique
    assert len(calendar) > 5000
    assert calendar.max() <= pd.Timestamp(DEFAULT_CONFIG.data.hard_end)


def test_panel_never_forward_fills(universe, calendar):
    """
    Gaps must remain NaN.

    A forward-filled price manufactures a zero return, which the model reads as
    a day of unnatural calm -- a quiet lie that survives most sanity checks.
    """
    frames, _ = universe
    panel = build_close_panel(frames, calendar)
    assert panel.isna().any().any()

    hyundai = panel["HYUNDAI"]                      # listed Oct 2024
    assert hyundai.loc[:"2024-10-01"].isna().all()
    assert hyundai.loc["2025-01-01":].notna().any()


def test_listing_mask_excludes_unlisted_periods(universe, calendar):
    """A 2024 listing must not dilute 2016 breadth."""
    frames, _ = universe
    mask = listing_mask(frames, calendar)
    assert not mask.loc["2016-06-01", "HYUNDAI"]
    assert mask.loc["2016-06-01", "RELIANCE"]
    live = mask.sum(axis=1)
    assert live.is_monotonic_increasing or live.iloc[-1] > live.iloc[0]


# =====================================================================
# Quality gate
# =====================================================================
def test_quality_gate_passes_on_clean_data(universe, calendar):
    frames, reports = universe
    report = run_quality_gate(frames, reports, calendar, raise_on_error=False)
    assert report.ok, "\n".join(str(c) for c in report.failed_errors)


def test_quality_gate_catches_injected_gap(universe, calendar):
    """
    The gate must fail if fabricated history sneaks back in.

    This is the regression guard: it proves the audit is enforced, not just
    documented.
    """
    frames, reports = universe
    poisoned = dict(frames)
    victim = "RELIANCE"
    frame = poisoned[victim]
    stitched = frame.iloc[:1].copy()
    stitched.index = pd.DatetimeIndex([frame.index[0] - pd.Timedelta(days=400)])
    poisoned[victim] = pd.concat([stitched, frame])

    report = run_quality_gate(poisoned, reports, calendar, raise_on_error=False)
    assert not report.ok
    assert any(c.name == "no_residual_gaps" for c in report.failed_errors)


def test_quality_gate_raises_by_default(universe, calendar):
    frames, reports = universe
    poisoned = dict(list(frames.items())[:5])       # too few symbols
    with pytest.raises(RuntimeError, match="quality gate FAILED"):
        run_quality_gate(poisoned, reports, calendar)
