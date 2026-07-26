"""
test_backtest.py
----------------
Tests for the Indian cost stack, capital gains treatment, and the portfolio
engine.

Cost bugs are the quietest kind: they never raise, they just shift every
return by a fraction of a percent in a direction nobody notices.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from qbeast_crash.backtest import (
    Lot,
    RateCard,
    TaxRates,
    apply_slippage,
    capital_gains,
    financial_year,
    leg_cost,
    run_backtest,
)


# =====================================================================
# The cost stack
# =====================================================================
def test_stt_applies_to_both_legs_on_delivery():
    """
    The most commonly mis-modelled component. Intraday charges STT on the sell
    side only; DELIVERY charges both legs at 0.1%, and this is a delivery
    strategy.
    """
    buy = leg_cost("buy", 100.0, 1000, dt.date(2023, 1, 2))
    sell = leg_cost("sell", 100.0, 1000, dt.date(2023, 1, 2))
    assert buy["stt"] == pytest.approx(100.0)          # 0.1% of 100,000
    assert sell["stt"] == pytest.approx(100.0)


def test_stamp_duty_is_buy_side_only():
    on = dt.date(2023, 1, 2)
    assert leg_cost("buy", 100.0, 1000, on)["stamp_duty"] == pytest.approx(15.0)
    assert leg_cost("sell", 100.0, 1000, on)["stamp_duty"] == 0.0


def test_dp_charge_is_sell_side_only():
    on = dt.date(2023, 1, 2)
    assert leg_cost("sell", 100.0, 1000, on)["dp_charge"] > 0
    assert leg_cost("buy", 100.0, 1000, on)["dp_charge"] == 0.0


def test_gst_base_excludes_stt_and_stamp_duty():
    """GST applies to brokerage, exchange and SEBI charges -- not to taxes."""
    c = leg_cost("buy", 100.0, 1000, dt.date(2023, 1, 2))
    expected = 0.18 * (c["brokerage"] + c["exchange"] + c["sebi"])
    assert c["gst"] == pytest.approx(expected)


def test_exchange_charge_changed_in_october_2024():
    before = leg_cost("buy", 100.0, 10_000, dt.date(2024, 9, 30))["exchange"]
    after = leg_cost("buy", 100.0, 10_000, dt.date(2024, 10, 1))["exchange"]
    assert after < before


def test_total_is_the_sum_of_its_parts():
    c = leg_cost("sell", 250.0, 400, dt.date(2023, 6, 1))
    parts = c["brokerage"] + c["exchange"] + c["sebi"] + c["gst"] \
        + c["stamp_duty"] + c["stt"] + c["dp_charge"]
    assert c["total_cost"] == pytest.approx(parts)


def test_invalid_side_raises():
    with pytest.raises(ValueError, match="'buy' or 'sell'"):
        leg_cost("hold", 100.0, 10, dt.date(2023, 1, 2))


# =====================================================================
# Slippage
# =====================================================================
def test_slippage_moves_against_you_on_both_sides():
    assert apply_slippage(100.0, "buy") > 100.0
    assert apply_slippage(100.0, "sell") < 100.0


def test_slippage_scales_with_volatility():
    """
    Flat slippage is the wrong shape here: this strategy trades ONLY on unusual
    days, which is exactly when spreads widen.
    """
    calm = apply_slippage(100.0, "buy", vol_ratio=1.0)
    wild = apply_slippage(100.0, "buy", vol_ratio=3.0)
    assert wild > calm


def test_slippage_is_capped():
    r = RateCard()
    extreme = apply_slippage(100.0, "buy", vol_ratio=1000.0, rates=r)
    assert extreme <= 100.0 * (1 + r.max_slippage_pct) + 1e-9


def test_slippage_handles_nan_volatility():
    assert apply_slippage(100.0, "buy", vol_ratio=float("nan")) > 100.0


# =====================================================================
# Capital gains
# =====================================================================
def test_financial_year_runs_april_to_march():
    assert financial_year("2024-04-01") == 2024
    assert financial_year("2025-03-31") == 2024
    assert financial_year("2025-04-01") == 2025


def test_short_and_long_term_split_at_365_days():
    sold = pd.Timestamp("2023-06-01")
    df = pd.DataFrame([
        {"sold_on": sold, "gain": 100_000.0, "holding_days": 364},
        {"sold_on": sold, "gain": 100_000.0, "holding_days": 365},
    ])
    out = capital_gains(df)
    assert out["stcg_gain"].iloc[0] == pytest.approx(100_000.0)
    assert out["ltcg_gain"].iloc[0] == pytest.approx(100_000.0)


def test_rates_changed_in_july_2024():
    """Budget 2024: STCG 15% -> 20%, LTCG 10% -> 12.5%."""
    gain = pd.DataFrame([{"sold_on": pd.Timestamp("2024-07-22"),
                          "gain": 1_000_000.0, "holding_days": 100}])
    before = capital_gains(gain)["tax"].iloc[0]
    gain["sold_on"] = pd.Timestamp("2024-07-24")
    after = capital_gains(gain)["tax"].iloc[0]
    assert before == pytest.approx(150_000.0)
    assert after == pytest.approx(200_000.0)


def test_ltcg_exemption_is_applied():
    t = TaxRates()
    df = pd.DataFrame([{"sold_on": pd.Timestamp("2023-06-01"),
                        "gain": 100_000.0, "holding_days": 400}])
    assert capital_gains(df, t)["tax"].iloc[0] == pytest.approx(0.0)


def test_losses_do_not_create_negative_tax():
    df = pd.DataFrame([{"sold_on": pd.Timestamp("2023-06-01"),
                        "gain": -500_000.0, "holding_days": 100}])
    assert capital_gains(df)["tax"].iloc[0] == 0.0


def test_empty_realised_gains():
    assert capital_gains(pd.DataFrame()).empty


# =====================================================================
# The portfolio engine
# =====================================================================
def _prices(n=300, n_sym=4, drift=0.0005, seed=0):
    idx = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f"S{i}": 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
         for i in range(n_sym)},
        index=idx,
    )


def test_buy_and_hold_trades_once_per_symbol():
    p = _prices()
    pos = pd.DataFrame(True, index=p.index, columns=p.columns)
    res = run_backtest(p, pos, capital=1_000_000.0)
    assert len(res["trades"]) == len(p.columns)
    assert (res["trades"]["side"] == "buy").all()
    assert res["realised"].empty          # never sells, so nothing realised


def test_exits_and_reentries_produce_round_trips():
    p = _prices()
    pos = pd.DataFrame(True, index=p.index, columns=p.columns)
    pos.iloc[100:150] = False
    res = run_backtest(p, pos, capital=1_000_000.0)
    assert (res["trades"]["side"] == "sell").sum() == len(p.columns)
    assert len(res["realised"]) == len(p.columns)


def test_costs_reduce_equity():
    p = _prices(drift=0.0)
    pos = pd.DataFrame(True, index=p.index, columns=p.columns)
    free = run_backtest(p, pos, 1_000_000.0, rates=RateCard(
        stt_pct=0, stamp_duty_pct=0, sebi_fee_pct=0, dp_charge_flat=0,
        base_slippage_pct=0))
    charged = run_backtest(p, pos, 1_000_000.0)
    assert charged["equity"].iloc[-1] < free["equity"].iloc[-1]
    assert charged["total_costs"] > 0


def test_terminal_unrealised_is_reported():
    """
    Buy-and-hold never sells, so on a realised basis it appears to pay no tax.
    That is deferral, not saving -- comparing against it without pricing the
    deferred liability would overstate the strategy's tax burden.
    """
    p = _prices(drift=0.002)
    pos = pd.DataFrame(True, index=p.index, columns=p.columns)
    res = run_backtest(p, pos, 1_000_000.0)

    assert res["total_tax"] == 0.0                 # nothing realised
    assert not res["terminal_unrealised"].empty
    assert res["deferred_tax"] > 0
    assert res["total_tax_liquidated"] == pytest.approx(res["deferred_tax"])


def test_sleeve_capital_never_goes_negative():
    p = _prices(n=400)
    rng = np.random.default_rng(1)
    pos = pd.DataFrame(rng.random((400, 4)) > 0.3, index=p.index, columns=p.columns)
    res = run_backtest(p, pos, 1_000_000.0)
    assert (res["equity"] > 0).all()


def test_missing_prices_are_skipped_not_treated_as_zero():
    p = _prices()
    p.iloc[50:60, 0] = np.nan
    pos = pd.DataFrame(True, index=p.index, columns=p.columns)
    res = run_backtest(p, pos, 1_000_000.0)
    assert np.isfinite(res["equity"]).all()
    assert (res["equity"] > 0).all()
