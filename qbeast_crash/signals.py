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

from qbeast_crash.config import DEFAULT_CONFIG, SignalConfig

import numpy as np
import pandas as pd

#: Every action that closes a position. Both paths sell -- a severe anomaly
#: acts the next day, a mild one sells from inside its watch -- and every count
#: of "trades" or "symbols traded" must include both.
#:
#: This exists because adding EXIT_WATCH silently broke eight separate counts
#: that each tested `action == "EXIT"`. Phase 6 reported 82 trades while Phase 8
#: reported 7 of 10 symbols never traded, from the same signal file.
SELL_ACTIONS = ("EXIT", "EXIT_WATCH")


def is_sell(actions) -> "pd.Series":
    """True where the action closed a position, by either path."""
    return actions.isin(SELL_ACTIONS)


__all__ = [
    "SELL_ACTIONS",
    "is_sell",
    "SignalConfig",
    "ReentryRule",
    "generate_signals",
    "market_signal",
    "equal_weight_equity",
    "performance",
    "signal_strength",
]


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
    Decide, for each day, whether to hold the stock or sit in cash.

    TWO WAYS A SIGNAL CAN FIRE
    --------------------------
    An anomaly score is computed every day. What happens next depends on how
    severe it is.

    SEVERE anomaly (intensity >= exit_intensity, default 0.99)
        Check today's slope and acceleration. If the stock is falling AND the
        fall is speeding up, act on the NEXT trading day. No waiting -- the
        edge is concentrated at a one-day horizon and decays fast, so a
        confirmation delay spends most of it.

    MILD anomaly (moderate_intensity <= intensity < exit_intensity)
        Do not act. Open a WATCH lasting up to watch_days sessions. On each
        day of the watch, compare the 1-day move against the 5-day slope and
        check the regime:

            1-day move worse than the 5-day slope  ->  the fall is
                                                       accelerating
            regime is Crashing or Falling          ->  the backdrop confirms

        If both agree, sell immediately rather than waiting out the watch.
        If the stock turns up instead, close the watch and do nothing.

    The mild path exists because most declines do not announce themselves with
    one dramatic day. They start as an ordinary-looking wobble that only
    becomes obviously bad several days in, by which point a severe-only rule
    has already missed most of the fall.

    Parameters
    ----------
    frame : per-symbol frame indexed by date. Needs `intensity` and `phase`;
            uses `ret_1d`, `slope_z` and `regime` when present.

    Returns
    -------
    DataFrame with:
        exit_signal   the severe exit condition fired today
        watching      a mild-anomaly watch was open today
        reentry_ok    the re-entry condition was satisfied today
        in_position   whether the stock is HELD today
        action        "EXIT", "EXIT_WATCH", "ENTER", or "" on the decision day

    CAUSALITY. A decision taken on day t applies from day t+1. The signal is
    computed at the close, so acting on that same close would require a time
    machine. `in_position` already reflects that, so returns can be applied to
    it directly.
    """
    cfg = config or DEFAULT_CONFIG.signals
    n = len(frame)

    intensity = frame["intensity"].to_numpy()
    phase = frame["phase"].to_numpy()

    # ---- severe: act next day on slope + acceleration -------------------
    severe = (intensity >= cfg.exit_intensity) & (phase == "AcceleratingDecline")
    if cfg.exit_persistence > 1:
        severe = (
            pd.Series(severe)
            .rolling(cfg.exit_persistence, min_periods=cfg.exit_persistence)
            .sum().fillna(0).to_numpy() >= cfg.exit_persistence
        )

    # ---- mild: open a watch ---------------------------------------------
    mild = (intensity >= cfg.moderate_intensity) & (intensity < cfg.exit_intensity) \
        & (phase.astype(str) != "AcceleratingAdvance")

    # Today's move against the 5-day trend. Both are already in daily sigmas,
    # so the comparison is like-for-like across stocks. Today being worse than
    # the trend means the decline is steepening.
    ret_1d = frame["ret_1d"].to_numpy() if "ret_1d" in frame else np.zeros(n)
    slope_z = frame["slope_z"].to_numpy() if "slope_z" in frame else np.zeros(n)
    vol = frame["vol"].to_numpy() if "vol" in frame else np.ones(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_z = np.where(np.isfinite(vol) & (vol > 1e-12), ret_1d / vol, 0.0)
    accelerating_down = np.nan_to_num(ret_z) < np.nan_to_num(slope_z)

    regime = (frame["regime"].to_numpy() if "regime" in frame
              else np.array(["Unknown"] * n, dtype=object))
    regime_bad = np.isin(regime, ("Crashing", "Falling"))
    # With no regime column the backdrop check cannot veto, so it passes.
    if "regime" not in frame:
        regime_bad = np.ones(n, dtype=bool)

    falling = np.nan_to_num(slope_z) < 0
    watch_confirms = accelerating_down & regime_bad & falling

    reentry_cond = _reentry_mask(frame, reentry, cfg)

    in_position = np.ones(n, dtype=bool)
    watching = np.zeros(n, dtype=bool)
    action = np.array([""] * n, dtype=object)

    holding = True
    since_change = 10_000
    watch_left = 0

    for t in range(n):
        in_position[t] = holding
        watching[t] = watch_left > 0
        since_change += 1

        if holding:
            if severe[t] and since_change > cfg.min_hold:
                holding = False
                since_change = 0
                watch_left = 0
                action[t] = "EXIT"

            elif watch_left > 0:
                if watch_confirms[t] and since_change > cfg.min_hold:
                    holding = False
                    since_change = 0
                    watch_left = 0
                    action[t] = "EXIT_WATCH"
                elif np.nan_to_num(slope_z[t]) > 0:
                    watch_left = 0          # it turned up; stand down
                else:
                    watch_left -= 1

            elif mild[t]:
                watch_left = cfg.watch_days

        else:
            forced = since_change >= cfg.max_cash_days
            if since_change > cfg.cooldown and (reentry_cond[t] or forced):
                holding = True
                since_change = 0
                action[t] = "ENTER"

    # T+1 execution is already baked in: in_position[t] is assigned BEFORE the
    # decision at t is processed, so an exit signalled on day t leaves
    # in_position[t] True and in_position[t+1] False. A further .shift(1) here
    # would delay every trade by a second day -- a bug a backtest would never
    # complain about, it would just report worse numbers forever.
    return pd.DataFrame(
        {
            "exit_signal": severe,
            "watching": watching,
            "reentry_ok": reentry_cond,
            "in_position": in_position,
            "action": action,
        },
        index=frame.index,
    )


def market_signal(
    market: pd.DataFrame,
    breadth_threshold: float | None = None,
    median_slope_threshold: float | None = None,
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
    cfg = DEFAULT_CONFIG.signals
    breadth = cfg.market_breadth_threshold if breadth_threshold is None else breadth_threshold
    slope = cfg.market_slope_threshold if median_slope_threshold is None else median_slope_threshold
    return (
        (market["breadth_decline"] >= breadth)
        & (market["median_slope_z"] <= slope)
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
