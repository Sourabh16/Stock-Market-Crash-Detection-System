"""
backtest.py
-----------
Phase 6: the cost model and the portfolio simulation.

What it does:   Applies the full NSE delivery cost stack plus Indian capital
                gains tax to the Phase 5 positions, and produces equity curves
                net of everything.
Why we do it:   A backtest without costs is a description of a market, not of a
                strategy. And for THIS strategy the interesting cost is not
                brokerage -- at 0.4 trades per symbol per year that is noise --
                it is tax.
Where:          backtest.py -> run_backtest()

THE TAX HYPOTHESIS, AND WHY IT DID NOT SURVIVE MEASUREMENT
----------------------------------------------------------
The expectation going in was that crash exits would convert long-term holdings
into short-term ones -- moving the Indian rate from 12.5% to 20% -- and that
drawdown reduction would therefore carry a hidden tax penalty that published
results ignore.

Measured on 2021-2026, that is not what happens:

  * Only 23.3% of the strategy's sales are short-term. It holds for years
    between trades, so most exits are already past the 12-month boundary.

  * Comparing realised tax alone is meaningless, because buy-and-hold never
    sells and so appears to pay NOTHING. That is deferral, not saving. Priced
    like-for-like by liquidating both books at the final close:

        strategy      Rs  27,295 paid + Rs 246,178 deferred = Rs 273,473
        buy & hold    Rs       0 paid + Rs 317,958 deferred = Rs 317,958

    The strategy pays LESS tax -- but mainly because it earned less, and a
    smaller gain carries a smaller liability. Lower tax on a lower return is
    not a benefit.

The honest conclusion is that the tax effect is real but small, and it is
swamped by the return difference. Reporting it as a headline contribution
would have been overclaiming. It is kept here because the measurement is the
point, and because the deferred-liability comparison is the part most
backtests genuinely do get wrong.

RATES CHANGE MID-BACKTEST
-------------------------
The window spans 2021-2026 and two of these rates moved inside it:

    2024-07-23   STCG 15% -> 20%; LTCG 10% -> 12.5%; exemption 1L -> 1.25L
    2024-10-01   NSE equity transaction charge 0.00325% -> 0.00297%

A single static rate card would be wrong for roughly half the window, so every
rate is a function of the trade date.

VERIFY BEFORE PUBLISHING. These are the rates as understood at the time of
writing and they are the kind of thing that changes with each budget. They are
isolated in RateCard so that checking them is a five-minute job.

SLIPPAGE IS NOT FLAT
--------------------
A constant slippage percentage is the wrong shape for this strategy
specifically. It trades ONLY on crash and rally days, which is precisely when
spreads widen. A flat figure understates cost exactly where every trade is, so
slippage scales with the day's realised volatility.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "RateCard",
    "TaxRates",
    "Lot",
    "leg_cost",
    "apply_slippage",
    "capital_gains",
    "financial_year",
    "run_backtest",
]

#: Indian financial year runs 1 April to 31 March.
FY_START_MONTH = 4

#: Budget 2024, effective 23 July 2024.
_CG_CHANGE = dt.date(2024, 7, 23)

#: NSE transaction charge revision, effective 1 October 2024.
_TXN_CHANGE = dt.date(2024, 10, 1)


@dataclass(frozen=True)
class RateCard:
    """
    NSE equity DELIVERY cost stack. All rates are fractions of turnover unless
    named `_flat`.

    Delivery, not intraday: STT applies to BOTH legs at 0.1%, which is the
    single largest component and the one most often modelled wrongly.
    """

    #: Zero at discount brokers for delivery. Set higher to model a full-service
    #: broker.
    brokerage_pct: float = 0.0

    sebi_fee_pct: float = 0.000001          # Rs 10 per crore
    stamp_duty_pct: float = 0.00015         # 0.015%, BUY side only
    stt_pct: float = 0.001                  # 0.1%, BOTH legs on delivery
    gst_pct: float = 0.18

    #: Depository charge per scrip on the sell side. Quoted GST-inclusive by
    #: most brokers, so it is added after GST rather than into its base.
    dp_charge_flat: float = 15.34

    #: Baseline slippage in quiet conditions, scaled up by realised volatility.
    base_slippage_pct: float = 0.0005       # 5 bps

    #: Slippage multiplier per unit of volatility ratio above 1. A day twice as
    #: volatile as normal pays roughly twice the baseline.
    slippage_vol_beta: float = 1.0

    #: Cap so a single wild day cannot produce an absurd fill.
    max_slippage_pct: float = 0.01

    def exchange_txn_pct(self, on: dt.date) -> float:
        """NSE transaction charge, revised 1 October 2024."""
        return 0.0000297 if on >= _TXN_CHANGE else 0.0000325


@dataclass(frozen=True)
class TaxRates:
    """
    Capital gains on listed equity with STT paid.

    Short-term is under 12 months (Section 111A); long-term is 12 months or more
    (Section 112A) and carries an annual exemption.
    """

    stcg_before: float = 0.15
    stcg_after: float = 0.20
    ltcg_before: float = 0.10
    ltcg_after: float = 0.125
    ltcg_exemption_before: float = 100_000.0
    ltcg_exemption_after: float = 125_000.0

    def stcg(self, on: dt.date) -> float:
        return self.stcg_after if on >= _CG_CHANGE else self.stcg_before

    def ltcg(self, on: dt.date) -> float:
        return self.ltcg_after if on >= _CG_CHANGE else self.ltcg_before

    def ltcg_exemption(self, on: dt.date) -> float:
        return (
            self.ltcg_exemption_after if on >= _CG_CHANGE
            else self.ltcg_exemption_before
        )


@dataclass
class Lot:
    """One purchase, tracked so holding period and gain can be computed on sale."""

    symbol: str
    qty: int
    price: float
    bought: pd.Timestamp


def financial_year(date) -> int:
    """Indian FY label: 2024 means FY 2024-25, starting 1 April 2024."""
    d = pd.Timestamp(date)
    return int(d.year if d.month >= FY_START_MONTH else d.year - 1)


def apply_slippage(
    price: float,
    side: str,
    vol_ratio: float = 1.0,
    rates: RateCard | None = None,
) -> float:
    """
    Fill price after slippage. Buy pays more, sell receives less.

    vol_ratio : today's volatility against the stock's own normal. 1.0 is a
                typical day; 3.0 is a crash day, where spreads genuinely widen.

    Scaling by volatility matters more here than in most backtests, because
    this strategy trades only on unusual days. Flat slippage would understate
    cost on every single trade it makes.
    """
    r = rates or RateCard()
    ratio = 1.0 if not np.isfinite(vol_ratio) else max(vol_ratio, 0.0)
    slip = r.base_slippage_pct * (1.0 + r.slippage_vol_beta * max(ratio - 1.0, 0.0))
    slip = min(slip, r.max_slippage_pct)

    if side == "buy":
        return price * (1.0 + slip)
    if side == "sell":
        return price * (1.0 - slip)
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def leg_cost(
    side: str,
    price: float,
    qty: int,
    on: dt.date,
    rates: RateCard | None = None,
) -> dict:
    """
    Full cost stack for one leg, in rupees.

    GST applies to brokerage, exchange charges and SEBI fees. Stamp duty is
    buy-side only; the DP charge is sell-side only and is added after GST
    because brokers quote it inclusive.
    """
    side = side.lower().strip()
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    r = rates or RateCard()
    on = pd.Timestamp(on).date() if not isinstance(on, dt.date) else on
    notional = price * qty

    brokerage = notional * r.brokerage_pct
    exchange = notional * r.exchange_txn_pct(on)
    sebi = notional * r.sebi_fee_pct
    gst = r.gst_pct * (brokerage + exchange + sebi)

    stamp = notional * r.stamp_duty_pct if side == "buy" else 0.0
    stt = notional * r.stt_pct                       # BOTH legs on delivery
    dp = r.dp_charge_flat if side == "sell" else 0.0

    total = brokerage + exchange + sebi + gst + stamp + stt + dp
    return {
        "side": side, "notional": notional, "brokerage": brokerage,
        "exchange": exchange, "sebi": sebi, "gst": gst, "stamp_duty": stamp,
        "stt": stt, "dp_charge": dp, "total_cost": total,
    }


def capital_gains(
    realised: pd.DataFrame,
    tax: TaxRates | None = None,
) -> pd.DataFrame:
    """
    Capital gains tax by financial year.

    Parameters
    ----------
    realised : one row per sale with columns
               sold_on, gain, holding_days.

    Short-term losses offset short-term gains, and long-term losses offset
    long-term gains, within a year. Carry-forward across years is NOT modelled
    -- it would understate the drag, so this errs on the conservative side.
    """
    t = tax or TaxRates()
    if realised.empty:
        return pd.DataFrame(columns=["fy", "stcg_gain", "ltcg_gain", "tax"])

    df = realised.copy()
    df["fy"] = df["sold_on"].map(financial_year)
    df["long_term"] = df["holding_days"] >= 365

    rows = []
    for fy, block in df.groupby("fy"):
        # Rate is set by when the gain was realised; use the year's last sale.
        as_of = pd.Timestamp(block["sold_on"].max()).date()
        st = block.loc[~block["long_term"], "gain"].sum()
        lt = block.loc[block["long_term"], "gain"].sum()

        st_tax = max(st, 0.0) * t.stcg(as_of)
        lt_taxable = max(lt - t.ltcg_exemption(as_of), 0.0)
        lt_tax = lt_taxable * t.ltcg(as_of)

        rows.append({"fy": fy, "stcg_gain": st, "ltcg_gain": lt,
                     "tax": st_tax + lt_tax})
    return pd.DataFrame(rows)


def run_backtest(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    capital: float = 1_000_000.0,
    vol_ratio: pd.DataFrame | None = None,
    rates: RateCard | None = None,
    tax: TaxRates | None = None,
) -> dict:
    """
    Sleeve-based portfolio simulation.

    Capital is split equally across symbols at the start, and each symbol keeps
    its own sleeve of cash and shares thereafter. On an exit the sleeve sells
    to cash; on a re-entry it buys back with whatever that sleeve holds.

    Why sleeves rather than daily rebalancing: a daily-rebalanced equal-weight
    portfolio trades every symbol every day, which would cost more in brokerage
    than the strategy could ever save in drawdown. Sleeves trade only when the
    signal changes, which is the whole point of a low-turnover design. The cost
    is some weight drift as sleeves grow apart -- accepted deliberately.

    Returns a dict with the equity curve, trade log, realised gains, tax by
    financial year, and a cost breakdown.
    """
    r = rates or RateCard()
    symbols = [s for s in positions.columns if s in prices.columns]
    dates = prices.index
    sleeve_capital = capital / len(symbols)

    cash = {s: sleeve_capital for s in symbols}
    lots: dict[str, list[Lot]] = {s: [] for s in symbols}
    trades: list[dict] = []
    realised: list[dict] = []
    equity = np.zeros(len(dates))

    def vol_at(sym, i):
        if vol_ratio is None or sym not in vol_ratio.columns:
            return 1.0
        v = vol_ratio.iat[i, vol_ratio.columns.get_loc(sym)]
        return float(v) if np.isfinite(v) else 1.0

    for i, date in enumerate(dates):
        day = date.date()
        for sym in symbols:
            price = prices.iat[i, prices.columns.get_loc(sym)]
            if not np.isfinite(price) or price <= 0:
                continue

            want = bool(positions.iat[i, positions.columns.get_loc(sym)])
            held = sum(lot.qty for lot in lots[sym])

            # ---- SELL -------------------------------------------------
            if held > 0 and not want:
                fill = apply_slippage(price, "sell", vol_at(sym, i), r)
                cost = leg_cost("sell", fill, held, day, r)
                proceeds = fill * held - cost["total_cost"]
                cash[sym] += proceeds

                for lot in lots[sym]:
                    gain = (fill - lot.price) * lot.qty
                    realised.append({
                        "symbol": sym, "sold_on": date, "qty": lot.qty,
                        "gain": gain,
                        "holding_days": (date - lot.bought).days,
                    })
                lots[sym] = []
                trades.append({"date": date, "symbol": sym, "side": "sell",
                               "qty": held, "price": fill, **cost})

            # ---- BUY --------------------------------------------------
            elif held == 0 and want and cash[sym] > 0:
                fill = apply_slippage(price, "buy", vol_at(sym, i), r)
                # Reserve headroom for costs so the sleeve cannot go negative.
                qty = int(cash[sym] * 0.995 // fill)
                if qty > 0:
                    cost = leg_cost("buy", fill, qty, day, r)
                    outlay = fill * qty + cost["total_cost"]
                    if outlay <= cash[sym]:
                        cash[sym] -= outlay
                        lots[sym].append(Lot(sym, qty, fill, date))
                        trades.append({"date": date, "symbol": sym, "side": "buy",
                                       "qty": qty, "price": fill, **cost})

        equity[i] = sum(cash.values()) + sum(
            sum(lot.qty for lot in lots[s])
            * (prices.iat[i, prices.columns.get_loc(s)]
               if np.isfinite(prices.iat[i, prices.columns.get_loc(s)]) else 0.0)
            for s in symbols
        )

    equity_series = pd.Series(equity, index=dates, name="equity")
    trade_log = pd.DataFrame(trades)
    realised_df = pd.DataFrame(realised)
    tax_table = capital_gains(realised_df, tax)

    # ---- terminal liquidation, so the comparison is like-for-like ------
    # Buy-and-hold never sells, so on a realised-gains basis it appears to pay
    # NO tax at all. That is not a saving, it is a deferral: it ends the period
    # holding a large unrealised liability that crystallises the moment anyone
    # actually wants the money.
    #
    # Comparing a strategy that realises gains against one that defers them,
    # without pricing the deferral, would overstate the strategy's tax penalty
    # by the entire buy-and-hold liability. Both books are therefore liquidated
    # at the final close and taxed on the result.
    final = dates[-1]
    terminal = []
    for sym, sym_lots in lots.items():
        price = prices.iat[-1, prices.columns.get_loc(sym)]
        if not np.isfinite(price):
            continue
        for lot in sym_lots:
            terminal.append({
                "symbol": sym, "sold_on": final, "qty": lot.qty,
                "gain": (price - lot.price) * lot.qty,
                "holding_days": (final - lot.bought).days,
            })
    terminal_df = pd.DataFrame(terminal)

    all_realised = pd.concat([realised_df, terminal_df], ignore_index=True) \
        if len(terminal_df) else realised_df
    tax_liquidated = capital_gains(all_realised, tax)

    total_tax = float(tax_table["tax"].sum()) if len(tax_table) else 0.0
    total_tax_liq = float(tax_liquidated["tax"].sum()) if len(tax_liquidated) else 0.0

    return {
        "equity": equity_series,
        "equity_after_tax": _apply_tax(equity_series, tax_table),
        "trades": trade_log,
        "realised": realised_df,
        "terminal_unrealised": terminal_df,
        "tax": tax_table,
        "tax_liquidated": tax_liquidated,
        "total_costs": float(trade_log["total_cost"].sum()) if len(trade_log) else 0.0,
        "total_tax": total_tax,
        "total_tax_liquidated": total_tax_liq,
        "deferred_tax": total_tax_liq - total_tax,
    }


def _apply_tax(equity: pd.Series, tax_table: pd.DataFrame) -> pd.Series:
    """
    Deduct each financial year's tax on 31 March, when it crystallises.

    Deducted as a step rather than smoothed, because that is when the cash
    actually leaves -- and a tax bill landing in one day is visible in the
    drawdown curve, which is exactly the effect worth showing.
    """
    out = equity.copy()
    if tax_table.empty:
        return out

    running = 0.0
    for _, row in tax_table.sort_values("fy").iterrows():
        pay_date = pd.Timestamp(dt.date(int(row["fy"]) + 1, 3, 31))
        running += float(row["tax"])
        out.loc[out.index >= pay_date] = equity.loc[out.index >= pay_date] - running
    return out
