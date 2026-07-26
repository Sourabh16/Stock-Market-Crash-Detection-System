"""
signals.py
----------
Phase 5: turning anomaly intensity into a position.

What it does:   Runs a per-stock state machine that decides whether to hold the
                stock or sit in cash, plus a market-wide de-risking overlay.
Why we do it:   Intensity says "today is unusual" and phase says "downward".
                Neither is a position. Something has to decide when to leave,
                when to come back, and how to avoid doing either too often.
Where:          signals.py -> generate_signals(), market_signal()

THE FRAMING: FAST REACTION, NOT PREDICTION
------------------------------------------
Phase 4 measured this directly. Against a random signal of identical firing
rate, the detector's early-warning recall was 0.88x -- no skill at a 15-day
horizon. The 2-3 day prediction claim is not supported.

But the same signal is enormously informative about the IMMEDIATE next few
days:

    P(drawdown <= -10% within H days), given a signal
      H=1     6.92%  vs   0.06% base   =  110x
      H=3    20.00%  vs   0.57% base   =   35x
      H=5    23.08%  vs   1.48% base   =   16x
      H=10   26.15%  vs   4.82% base   =    5x

Lift decays sharply with horizon, and the median same-day return on signal days
is -1.14% against +0.02% overall. The signal fires as a decline BEGINS.

So the strategy is not "predict the crash and step aside beforehand". It is
"recognise within a day that a decline has started, and leave before it
deepens". That is a weaker claim, and it is the one the evidence supports.

WHY RE-ENTRY IS THE HARD HALF
-----------------------------
Exiting is easy; the exit rule has a 110x edge at one day. The difficulty is
coming back. Crashes are followed by recoveries, and a strategy that exits well
but re-enters late loses more to the missed rebound than it ever saved on the
decline. That failure mode is the reason most crash-avoidance strategies
underperform buy-and-hold despite being right about the crash.

Re-entry rules are therefore configurable and measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "SignalConfig",
    "ReentryRule",
    "generate_signals",
    "market_signal",
    "equal_weight_equity",
    "performance",
    "signal_strength",
]


@dataclass(frozen=True)
class SignalConfig:
    """
    Thresholds and the whipsaw controls.

    The whipsaw controls are not polish. Every round trip pays brokerage, STT,
    stamp duty, GST and slippage, and converts a long-term holding into a
    short-term one for tax. A rule that is right slightly more often than it is
    wrong can still lose money if it trades enough.
    """

    #: Exit when intensity clears this AND the trend phase is an accelerating
    #: decline. 0.99 measured 110x lift at a one-day horizon; 0.95 fires ~7x
    #: more often for materially less edge.
    exit_intensity: float = 0.99

    #: Consecutive confirming days before acting. 1 = act immediately, which is
    #: what the fast-reaction framing wants -- the edge is concentrated at H=1
    #: and decays fast, so waiting for confirmation spends the entire advantage.
    exit_persistence: int = 1

    #: Re-entry threshold, used by the intensity-based rules.
    reentry_intensity: float = 0.95

    #: Minimum sessions held before an exit may fire. Stops an entry and exit
    #: landing on consecutive days.
    min_hold: int = 3

    #: Minimum sessions in cash before re-entry. The dominant whipsaw guard.
    cooldown: int = 3

    #: Hard cap: force re-entry after this many sessions in cash regardless of
    #: signal. Being long is the default state, and an indefinite cash position
    #: is a bet the model was never asked to make.
    max_cash_days: int = 20


class ReentryRule:
    """
    Candidate re-entry rules, kept separate so they can be compared.

    Which one is right is an empirical question, not a design opinion -- see
    scripts/run_all_phases.py phase 5.
    """

    #: Come back on a confirmed rally: unusual AND accelerating upward.
    RALLY_SIGNAL = "rally_signal"

    #: Come back when the decline stops accelerating. Earlier than a rally
    #: signal, so it captures more of the rebound but risks re-entering into a
    #: pause rather than a bottom.
    DECEL = "decel"

    #: Come back after a fixed number of sessions. The dumb baseline, and the
    #: one the others must beat to justify their complexity.
    TIME = "time"

    #: Come back as soon as the phase is no longer an accelerating decline.
    NOT_DECLINING = "not_declining"

    ALL = (RALLY_SIGNAL, DECEL, TIME, NOT_DECLINING)


def _reentry_mask(frame: pd.DataFrame, rule: str, config: SignalConfig) -> np.ndarray:
    """Boolean array: is the re-entry condition satisfied on each bar?"""
    phase = frame["phase"].to_numpy()
    intensity = frame["intensity"].to_numpy()

    if rule == ReentryRule.RALLY_SIGNAL:
        return (intensity >= config.reentry_intensity) & (phase == "AcceleratingAdvance")
    if rule == ReentryRule.DECEL:
        return phase == "DeceleratingDecline"
    if rule == ReentryRule.NOT_DECLINING:
        return phase != "AcceleratingDecline"
    if rule == ReentryRule.TIME:
        return np.ones(len(frame), dtype=bool)      # cooldown alone governs
    raise ValueError(f"unknown re-entry rule {rule!r}")


def generate_signals(
    frame: pd.DataFrame,
    config: SignalConfig | None = None,
    reentry: str = ReentryRule.NOT_DECLINING,
) -> pd.DataFrame:
    """
    Run the hold/cash state machine for one symbol.

    Parameters
    ----------
    frame : per-symbol frame indexed by date, with `intensity` and `phase`.

    Returns
    -------
    DataFrame with:
        exit_signal   the exit condition fired on this bar
        reentry_ok    the re-entry condition was satisfied on this bar
        in_position   whether the stock is HELD on this bar
        action        "EXIT", "ENTER", or "" on the bar the decision was taken

    CAUSALITY. A decision taken on bar t applies from bar t+1. The signal is
    computed at the close, so acting on the same bar's close would require a
    time machine. `in_position` therefore reflects the state you would actually
    have been in, and returns should be applied to it directly.
    """
    cfg = config or SignalConfig()
    n = len(frame)

    intensity = frame["intensity"].to_numpy()
    phase = frame["phase"].to_numpy()

    exit_cond = (intensity >= cfg.exit_intensity) & (phase == "AcceleratingDecline")

    # Require N consecutive confirming days.
    if cfg.exit_persistence > 1:
        series = pd.Series(exit_cond)
        exit_cond = (
            series.rolling(cfg.exit_persistence, min_periods=cfg.exit_persistence)
            .sum()
            .fillna(0)
            .to_numpy()
            >= cfg.exit_persistence
        )

    reentry_cond = _reentry_mask(frame, reentry, cfg)

    in_position = np.ones(n, dtype=bool)      # long is the default state
    action = np.array([""] * n, dtype=object)

    holding = True
    since_change = 10_000                     # large, so nothing is blocked at the start

    for t in range(n):
        in_position[t] = holding
        since_change += 1

        if holding:
            if exit_cond[t] and since_change > cfg.min_hold:
                holding = False
                since_change = 0
                action[t] = "EXIT"
        else:
            forced = since_change >= cfg.max_cash_days
            if since_change > cfg.cooldown and (reentry_cond[t] or forced):
                holding = True
                since_change = 0
                action[t] = "ENTER"

    # T+1 execution is already baked into the loop: in_position[t] is assigned
    # BEFORE the decision at t is processed, so an exit signalled on bar t
    # leaves in_position[t] True and in_position[t+1] False. Applying a further
    # .shift(1) here would delay every trade by a second day -- a bug that a
    # backtest would never complain about, it would just quietly report worse
    # numbers. Caught by test_exit_takes_effect_the_day_after_the_signal.
    return pd.DataFrame(
        {
            "exit_signal": exit_cond,
            "reentry_ok": reentry_cond,
            "in_position": in_position,
            "action": action,
        },
        index=frame.index,
    )


def market_signal(
    market: pd.DataFrame,
    breadth_threshold: float = 75.0,
    median_slope_threshold: float = -1.5,
) -> pd.Series:
    """
    Market-wide de-risking flag.

    Both conditions are required, and that is the whole point. Measured on the
    real universe, breadth alone does not distinguish a systemic crash from an
    ordinary pullback -- roughly 90% of stocks were declining on 2020-03-12
    (COVID) and on several mild pullbacks alike. Only the median slope
    separates them: -2.11 sigma against about -0.8.

        date         breadth   median slope   event
        2020-03-12     92.1%         -2.11    COVID crash
        2016-02-11     90.1%         -0.94    Feb-2016 selloff
        2022-09-26     87.2%         -0.67    Fed / GBP crisis
    """
    return (
        (market["breadth_decline"] >= breadth_threshold)
        & (market["median_slope_z"] <= median_slope_threshold)
    ).fillna(False)


def equal_weight_equity(
    returns: pd.DataFrame,
    positions: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Equity curve of an equal-weighted, daily-rebalanced portfolio.

    Parameters
    ----------
    returns   : dates x symbols SIMPLE returns (not log).
    positions : dates x symbols booleans -- held (True) or in cash (False).
                None means always fully invested, i.e. buy-and-hold.

    WHY NOT exp(mean(log equity))
    -----------------------------
    Averaging per-stock LOG equity curves and exponentiating gives the
    GEOMETRIC MEAN of individual stock outcomes, which is not a portfolio. It
    silently discards the diversification benefit -- by Jensen's inequality the
    mean of logs is below the log of the mean, and the gap IS the benefit.

    Measured on this universe over 2021-2026:

        geometric mean of stocks    CAGR 20.69%   maxDD -21.2%
        true equal-weight portfolio CAGR 25.44%   maxDD -20.0%

    A 4.75pp CAGR understatement, in the direction that makes results look
    worse. Portfolio return is the mean of SIMPLE returns each day, compounded.
    """
    weights = positions.astype(float) if positions is not None else 1.0
    daily = (returns * weights).mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + daily).cumprod()


def performance(equity: pd.Series, years: float) -> dict:
    """CAGR and maximum drawdown from an equity curve."""
    if equity.empty or years <= 0:
        return {"cagr": float("nan"), "max_drawdown": float("nan")}
    return {
        "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
    }


def signal_strength(frame: pd.DataFrame) -> pd.Series:
    """
    Rank score for choosing WHICH stocks to act on when several fire at once.

    Capital is finite, so on a broad selloff we cannot exit or re-enter
    everything simultaneously; positions have to be ranked. Strength combines
    how unusual the day is with how hard the stock is moving, both of which are
    already comparable across the universe -- intensity is a percentile and
    slope_z is in daily sigmas.
    """
    return frame["intensity"].fillna(0.0) * frame["slope_z"].abs().fillna(0.0)
