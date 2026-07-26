"""
config.py
---------
Single source of truth for the whole pipeline.

Every path, date window, threshold, rate and model parameter lives here and
nowhere else. If a number appears in a result, it is defined in this file.

WHY ONE FILE
------------
A threshold that lives in three places will eventually disagree with itself,
and a backtest that disagrees with itself is worse than no backtest -- it
produces a number that looks fine and is quietly wrong.

WHY THE COMMENTS ARE LONG
-------------------------
Most of these values were chosen by measurement, and several replaced an
earlier value that turned out to be wrong. The measurement is recorded next to
the number so nobody -- including us in three months -- has to re-litigate the
decision from memory, or worse, re-run the experiment to find out why 0.99 and
not 0.95.

Where a value is a judgement rather than a measurement, it says so.

WHAT TO VERIFY BEFORE PUBLISHING
--------------------------------
The statutory rates in CostConfig and TaxConfig change with each budget. They
are grouped together at the bottom so re-checking takes five minutes.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

# =====================================================================
# Paths
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
MODELS = PROJECT_ROOT / "models"

#: The index file doubles as the master trading calendar. It is the only series
#: guaranteed to have a bar on every session the exchange was open, which no
#: individual stock guarantees -- a stock can be suspended or halted.
INDEX_SYMBOL = "NIFTY100"

#: Raw CSVs use Indian day-first format. Parsing this as month-first succeeds
#: silently for the first twelve days of every month and corrupts the rest, so
#: the format is pinned explicitly and never inferred.
RAW_DATE_FORMAT = "%d-%m-%Y"


# =====================================================================
# Data cleaning
# =====================================================================
@dataclass(frozen=True)
class DataConfig:
    """Rules for turning raw CSVs into a panel that can be believed."""

    #: A gap longer than this many CALENDAR days is never a holiday. NSE has
    #: never closed for ten straight days, so such a gap means the vendor
    #: stitched sparse or monthly data in front of the real listing history.
    #: Everything before the LAST such gap is discarded.
    #:
    #: Validated against externally sourced NSE listing dates -- the rule
    #: recovers all four to the day:
    #:     VBL 2016-11-08, DMART 2017-03-21,
    #:     SBILIFE 2017-10-03, MAZDOCK 2020-10-12
    max_gap_days: int = 10

    #: Symbols with fewer usable bars than this cannot support a 60-day
    #: volatility warmup plus a meaningful backtest. Excludes ENRIN, TATACAP
    #: and TMCV.
    min_bars: int = 300

    #: A one-day LOG return beyond this is an unadjusted corporate action --
    #: a demerger or spin-off -- not a market move. `close` is back-adjusted
    #: for splits and bonuses but NOT for restructurings.
    #:
    #: 0.50 in log terms is roughly -39% / +65% simple. The margin is chosen so
    #: artefacts are caught and real crashes survive:
    #:
    #:     CGPOWER  2016-03-15  log -1.074  (-65.8%)  demerger    -> caught
    #:     TRENT    2026-06-02  log -0.384  (-31.9%)  market      -> kept
    #:     ADANIENT 2023-02-03  log -0.302  (-26.1%)  Hindenburg  -> kept
    #:
    #: The hard part is not filtering artefacts out, it is not filtering too
    #: much: a blanket outlier rule would delete the very events this project
    #: exists to detect.
    max_abs_log_return: float = 0.50

    #: Ragged vendor end-dates span 2026-06-03 to 2026-06-22. 98 of 100 symbols
    #: reach this date; trimming here keeps the cross-section square, which the
    #: market-wide breadth signal depends on.
    hard_end: dt.date = dt.date(2026, 6, 5)

    #: Bars where open == high == low == close. Real for illiquid names, a
    #: vendor artefact otherwise. Either way they carry zero intraday range.
    #: FLAGGED, never dropped: deleting a row silently shifts every rolling
    #: window that spans it, so a 5-day slope would quietly cover 6 days.
    flag_flat_bars: bool = True

    #: A 10-symbol subset for fast iteration. The full pipeline takes ~40s on
    #: 96 symbols; this runs in a few seconds, which changes how often you are
    #: willing to re-run it.
    #:
    #: Chosen to be REPRESENTATIVE, not arbitrary. A convenience slice
    #: (alphabetical, or the first ten files) would concentrate sectors and
    #: volatility profiles and give misleading results. These span:
    #:
    #:     annualised volatility 25.8% (HINDUNILVR) to 47.9% (ADANIENT)
    #:     against a universe median of 34.4%
    #:
    #:   sectors : energy, IT, banking, FMCG x2, pharma, infra, auto, metals
    #:   history : all have pre-2016 data, so the training window is full
    #:   stress  : ADANIENT carries a real -30.3% day (Hindenburg, Feb 2023)
    #:             and a -71.3% drawdown; TATASTEEL is the cyclical
    #:
    #: TWO CAVEATS, THE SECOND MORE SERIOUS THAN IT LOOKS.
    #:
    #: 1. Cross-sectional features are much noisier on ten names. Breadth moves
    #:    in 10% steps, and average pairwise correlation is estimated from 45
    #:    pairs instead of 4,560.
    #:
    #: 2. PORTFOLIO RESULTS ON A SMALL UNIVERSE ARE DOMINATED BY ONE NAME.
    #:    Measured on this exact subset over 2021-2026:
    #:
    #:        total edge over buy-and-hold   +442pp
    #:        ADANIENT alone                 +411pp   (93% of it)
    #:        six of ten symbols              0pp     (never traded at all)
    #:
    #:    The full 96-symbol run reports -4.4% of capital; this subset reports
    #:    +33%. Both are correct. The difference is entirely position weight --
    #:    ADANIENT is 10% of a ten-stock portfolio and about 1% of the full one,
    #:    so the same avoided crash is diluted tenfold.
    #:
    #: USE THE DEV SUBSET FOR: checking the pipeline runs, per-stock signal
    #: inspection, and iterating on features.
    #: DO NOT USE IT FOR: portfolio returns, drawdown comparisons, or any
    #: conclusion about whether the strategy works. Confirm those on the full
    #: universe, where no single name can carry the result.
    dev_universe: tuple[str, ...] = (
        "RELIANCE", "TCS", "HDFCBANK", "ITC", "ADANIENT",
        "SUNPHARMA", "LT", "MARUTI", "HINDUNILVR", "TATASTEEL",
    )

    #: Columns kept from the raw file. The underscore-prefixed metadata is
    #: retained deliberately -- `_source` records which vendor supplied each
    #: bar (kite vs upstox), which is two independent observations of one
    #: market. Dropping it is irreversible.
    keep_columns: tuple[str, ...] = (
        "open", "high", "low", "close", "volume",
        "_source", "_dq_score", "_gap_filled",
    )


# =====================================================================
# Date windows
# =====================================================================
@dataclass(frozen=True)
class WindowConfig:
    """
    Train/backtest split. Strict, and never overlapping.

    Any feature, threshold or scaler fitted on backtest-window data would leak
    the future into the past and inflate every number downstream.

    NOTE ON THIS PARTICULAR SPLIT: it places COVID inside TRAINING. That is
    what the project brief specifies, but it is worth understanding the
    consequence -- an Isolation Forest trained on a crisis learns that crises
    are ordinary. The Phase 5 stress test deliberately retrains on 2016-2019
    instead, so COVID is genuinely out of sample and the system can be tested
    against a crash it has never seen.
    """

    train_start: dt.date = dt.date(2016, 1, 1)
    train_end: dt.date = dt.date(2020, 12, 31)
    backtest_start: dt.date = dt.date(2021, 1, 1)
    backtest_end: dt.date = dt.date(2026, 6, 5)

    #: Used by the stress test, which needs a crash the model has never seen.
    stress_train_end: dt.date = dt.date(2019, 12, 31)

    def __post_init__(self) -> None:
        if self.train_end >= self.backtest_start:
            raise ValueError("train window must end before the backtest begins")

    @property
    def backtest_years(self) -> float:
        return (self.backtest_end - self.backtest_start).days / 365.25


# =====================================================================
# Features
# =====================================================================
@dataclass(frozen=True)
class FeatureConfig:
    """
    Slope, acceleration and the precursor block.

    THE DESIGN RULE: every model feature measures market STRESS rather than
    price MOVEMENT. Stress builds while price is still roughly flat, which is
    what makes it available before a break rather than during one.

    The same-day return is computed and returned for labelling and plotting but
    is NOT in the model's feature list -- see features.FEATURE_COLUMNS. Feeding
    it in would make the detector coincident by construction, and no downstream
    cleverness recovers lead time once the feature vector contains the answer.
    """

    slope_window: int = 5          # one trading week
    accel_window: int = 7          # a parabola needs more room than a line
    vol_window: int = 60           # ~3 months of trading

    #: Lag applied to the volatility denominator before normalising slope.
    #:
    #: Not cosmetic. A crash inflates its own denominator: once the trailing
    #: window fills with crash days the normaliser grows and the signal
    #: shrinks, precisely when it should be loudest. Measured on RELIANCE
    #: across March 2020, the COVID slope_z reads -1.67 at lag 0 but -3.01 at
    #: lag 20. Lagging means the denominator describes the regime the move is
    #: departing FROM, not the one it is creating.
    vol_lag: int = 20

    #: Deadbands in units of daily sigma. Below these, a reading is called Flat
    #: rather than directional. Without them, sign flips on numerical noise
    #: produce phase churn -- and phase churn becomes trade churn.
    slope_deadband: float = 0.10
    accel_deadband: float = 0.05

    short_vol: int = 5             # numerator of the volatility term structure
    med_window: int = 20           # semi-deviation, gaps, illiquidity
    gap_threshold: float = 0.01    # overnight move counted as a gap

    #: Cross-sectional block.
    corr_window: int = 60
    ma_window: int = 50
    min_symbols: int = 20          # below this, breadth is not meaningful


# =====================================================================
# Model
# =====================================================================
@dataclass(frozen=True)
class ModelConfig:
    """
    Isolation Forest settings.

    NOTE WHAT IS ABSENT: `contamination`. Measured across values from 0.001 to
    0.2, the raw anomaly scores are BIT-IDENTICAL -- it does not affect tree
    building at all, only an internal offset used to binarise scores in
    predict(), which this pipeline never calls.

    Setting contamination amounts to asserting what fraction of history was a
    crash, and then being scored against your own assertion. The
    training-percentile mapping in model.py replaces the assumption with a
    measurement.
    """

    n_estimators: int = 300

    #: The subsample each tree isolates from. Often described as the parameter
    #: that genuinely matters, in contrast to contamination. Measured across a
    #: 64x range on this data, lift moved only 2.18x to 2.29x -- it barely
    #: matters either. 256 is the library default and is kept on evidence.
    max_samples: int = 256

    random_state: int = 0

    #: Withhold the top-quantile of market turbulence from training.
    #: None disables purging, which is the BASELINE.
    #:
    #: The idea is sound in the setting that motivated it: Isolation Forest
    #: learns normal from what it is shown, so training on crises makes crises
    #: unremarkable. A detector trained on MARKET-level features across
    #: 2006-2020 -- containing both 2008 and COVID -- fired ZERO alerts across
    #: 1,344 out-of-sample days.
    #:
    #: But on the POOLED PER-STOCK cross-section it is counterproductive.
    #: Measured on 2021-2026, signal = intensity 0.99+ and AcceleratingDecline,
    #: against a base crash rate of 11.0%:
    #:
    #:     purge   signals   P(crash)   lift
    #:     none        125      38.4%   3.48x
    #:     0.99        205      29.3%   2.65x
    #:     0.95        420      24.8%   2.24x
    #:     0.90        444      24.5%   2.22x
    #:     0.75        645      20.8%   1.88x
    #:
    #: Monotone: purging buys coverage at the cost of precision. Pooling 96
    #: symbols over 2016-2020 gives a training distribution rich enough that no
    #: single crisis dominates, so the silence problem never arises and the
    #: cure is worse than the disease.
    #:
    #: Purging remains a Phase 7 retraining VARIANT, not the default.
    purge_quantile: float | None = None
    purge_column: str = "dispersion"

    #: Intensity bands. These are where the "5% of days are anomalous"
    #: intuition actually belongs -- not in contamination. A 0.99 cut implies
    #: an alert budget of roughly one day in a hundred.
    intensity_high: float = 0.99
    intensity_moderate: float = 0.95
    intensity_low: float = 0.90


# =====================================================================
# Evaluation labels
# =====================================================================
@dataclass(frozen=True)
class LabelConfig:
    """
    Crash and rally ground truth. EVALUATION ONLY -- never touches training.

    That separation is structural, not a convention. Isolation Forest is
    unsupervised; if the crash threshold were derived from the model, the model
    would be defining the event it is then scored on detecting, and the
    evaluation would be circular.

    WHY ABSOLUTE MAGNITUDE, NOT VOLATILITY-RELATIVE: drawdown is what hurts a
    portfolio, and it is absolute -- losing 30% hurts identically in a calm
    year and a wild one. The volatility-relative alternative was tested and
    rejected: Bollinger band position becomes LESS extreme as a crash deepens,
    because the band widens faster than price falls. Measured on NIFTY100 in
    2020, -9% from peak scored -2.85 SD but -38% scored only -2.01 SD.
    """

    horizon: int = 5

    #: Single-stock threshold. NOT the index threshold -- stocks are roughly
    #: twice as volatile, so an index-calibrated -5% is an ordinary pullback
    #: for a stock. Measured event counts, 2021-2026 across 96 symbols:
    #:
    #:     threshold   onsets   per stock per year
    #:        -5%       3,950         7.6      <- ordinary pullbacks
    #:        -8%       1,479         2.9
    #:       -10%         705         1.4      <- default
    #:       -12%         350         0.7
    #:       -20%          36         0.1
    crash_threshold: float = -0.10
    rally_threshold: float = 0.10

    #: The index is less volatile. -5% occurs on 5.44% of NIFTY100 days, which
    #: happens to align almost exactly with the 5% anomaly rate conventionally
    #: quoted for this kind of model -- two numbers arrived at from completely
    #: different directions agreeing is a useful cross-check on both.
    index_crash_threshold: float = -0.05

    #: Events closer together than this merge into one. Without it a single
    #: crash produces a run of daily "events" and every recall figure is
    #: inflated by counting the same episode many times.
    min_gap: int = 10

    #: How far back to search for a warning before an onset. Beyond this an
    #: alert is more likely coincidence than warning, given the base rate.
    max_lookback: int = 15


# =====================================================================
# Signals
# =====================================================================
@dataclass(frozen=True)
class SignalConfig:
    """
    The hold/cash state machine.

    THE FRAMING IS FAST REACTION, NOT PREDICTION. Phase 4 measured it: against
    a random signal of identical firing rate, early-warning recall was 0.88x --
    no skill at a 15-day horizon. But the same signal is enormously informative
    about the immediate next few days:

        P(drawdown <= -10% within H days), given a signal
          H=1     6.92%  vs 0.06% base  = 110x
          H=3    20.00%  vs 0.57% base  =  35x
          H=10   26.15%  vs 4.82% base  =   5x

    The median same-day return on signal days is -1.14% against +0.02%
    overall: the signal fires as a decline BEGINS, not before it.

    The whipsaw controls are load-bearing, not polish. Every round trip pays
    STT on both legs plus stamp duty, GST, DP and slippage, and converts a
    long-term holding into a short-term one for tax. A rule right slightly more
    often than it is wrong can still lose money if it trades enough.
    """

    #: Exit when intensity clears this AND the phase is an accelerating
    #: decline. 0.99 measured 110x lift at one day; 0.95 fires ~7x more often
    #: for materially less edge.
    exit_intensity: float = 0.99

    #: Consecutive confirming days before acting. 1 is deliberate: the edge is
    #: concentrated at H=1 and decays fast, so waiting for confirmation spends
    #: the entire advantage. Whipsaw is controlled by the cooldown instead.
    exit_persistence: int = 1

    reentry_intensity: float = 0.95

    #: Minimum sessions held before an exit may fire.
    min_hold: int = 3

    #: Minimum sessions in cash before re-entry. Measured across four re-entry
    #: rules, this dominates the outcome far more than the choice of rule --
    #: all four landed within 0.7pp of CAGR of each other.
    cooldown: int = 3

    #: Force re-entry after this many sessions in cash regardless of signal.
    #: Long is the default state, and an indefinite cash position is a bet the
    #: model was never asked to make.
    max_cash_days: int = 20

    #: Market-wide de-risking. BOTH conditions are required -- breadth alone
    #: cannot separate a systemic crash from a broad pullback. Around 90% of
    #: stocks were declining on 2020-03-12 (COVID) and on several mild
    #: pullbacks alike; only the median slope distinguishes them (-2.11 sigma
    #: against roughly -0.8).
    #:
    #: Fired ZERO times out of sample. That is arguably correct: a
    #: systemic-crash trigger should be rare, and tuning it to fire in a period
    #: containing no systemic crash would be fitting noise.
    market_breadth_threshold: float = 75.0
    market_slope_threshold: float = -1.5


# =====================================================================
# Portfolio
# =====================================================================
@dataclass(frozen=True)
class PortfolioConfig:
    """
    Capital and position sizing.

    Sleeves, not daily rebalancing: capital is split equally across symbols at
    the start and each symbol keeps its own sleeve of cash and shares. A
    daily-rebalanced equal-weight portfolio trades every symbol every day,
    which would cost more in brokerage than this strategy could ever save in
    drawdown. The cost of sleeves is some weight drift as they grow apart --
    accepted deliberately.
    """

    initial_capital: float = 1_000_000.0     # Rs 10 lakh

    #: Fraction of a sleeve's cash left unspent on a buy, so costs cannot push
    #: the sleeve negative.
    cash_buffer: float = 0.005

    #: Cap on concurrent positions when ranking by signal strength. None means
    #: no cap, which is the current behaviour -- the strategy holds everything
    #: by default and only steps out selectively.
    max_positions: int | None = None


# =====================================================================
# Costs -- NSE equity DELIVERY
# =====================================================================
@dataclass(frozen=True)
class CostConfig:
    """
    Transaction cost stack. All `_pct` fields are FRACTIONS of turnover:
    0.001 means 0.1%. Getting this wrong by a factor of 100 is the classic
    cost-model bug, and it produces results that look merely disappointing
    rather than obviously absurd.

    Measured round trip on Rs 1,00,000 notional: about 0.238%.
    """

    #: Zero for delivery at discount brokers. Full-service brokers charge
    #: 0.3%-0.5%; set that here to model one.
    brokerage_pct: float = 0.0

    sebi_fee_pct: float = 0.000001           # Rs 10 per crore
    stamp_duty_pct: float = 0.00015          # 0.015%, BUY side only
    gst_pct: float = 0.18                    # on brokerage + exchange + SEBI

    #: STT is 0.1% on DELIVERY and is charged on BOTH legs. Sell-side-only is
    #: INTRADAY. This is the largest single component and the one most often
    #: modelled wrongly -- halving it by mistake makes every backtest look
    #: better in a way nothing else flags.
    stt_pct: float = 0.001

    #: Depository charge per SCRIP per DAY, sell side only. Brokers advertise
    #: Rs 15.34 GST-INCLUSIVE, so this is that figure directly: backtest.py
    #: adds it AFTER GST rather than into the GST base.
    #:
    #: If you ever move it into the GST base, change this to 13.00 --
    #: 13.00 x 1.18 = 15.34 exactly.
    dp_charge_flat: float = 15.34

    #: Baseline one-way slippage on a quiet day. 5bp is reasonable for
    #: NIFTY 100 liquidity.
    base_slippage_pct: float = 0.0005

    #: Extra slippage per unit of volatility ratio above 1.
    #:
    #: Flat slippage is the wrong SHAPE for this strategy specifically. It
    #: trades ONLY on unusual days, which is exactly when spreads widen --
    #: often three to five times. A flat rate is simultaneously too high on the
    #: quiet days it never trades and too low on the volatile days it always
    #: does.
    slippage_vol_beta: float = 1.0

    #: Cap, so one wild day cannot produce an absurd fill.
    max_slippage_pct: float = 0.01

    #: NSE equity transaction charge, revised 1 October 2024. A static rate
    #: would be wrong on one side of that date.
    exchange_txn_pct_before: float = 0.0000325
    exchange_txn_pct_after: float = 0.0000297
    exchange_change_date: dt.date = dt.date(2024, 10, 1)

    def exchange_txn_pct(self, on: dt.date) -> float:
        return (
            self.exchange_txn_pct_after if on >= self.exchange_change_date
            else self.exchange_txn_pct_before
        )


# =====================================================================
# Capital gains tax
# =====================================================================
@dataclass(frozen=True)
class TaxConfig:
    """
    Capital gains on listed equity with STT paid.

    Cannot live in a per-leg cost function: it is computed per financial year
    across the whole portfolio, needs the holding period of every lot, and
    carries an annual exemption on the long-term side.

    Rates changed on 23 July 2024, INSIDE the backtest window, so a single
    static rate would be wrong for roughly half the period.

    A TRAP WORTH KNOWING: comparing a strategy's realised tax against
    buy-and-hold is misleading, because buy-and-hold never sells and appears to
    pay nothing. That is deferral, not saving -- it ends the period carrying an
    unrealised liability. Both books must be liquidated at the final close for
    the comparison to mean anything. Measured on this universe, doing so
    reverses the sign of the apparent tax penalty:
    Rs 273,473 for the strategy against Rs 317,958 for buy-and-hold.
    """

    stcg_before: float = 0.15                # short-term, under 12 months
    stcg_after: float = 0.20
    ltcg_before: float = 0.10                # long-term, 12 months or more
    ltcg_after: float = 0.125
    ltcg_exemption_before: float = 100_000.0
    ltcg_exemption_after: float = 125_000.0
    change_date: dt.date = dt.date(2024, 7, 23)

    #: Holding period at or beyond which a gain is long-term.
    long_term_days: int = 365

    def stcg(self, on: dt.date) -> float:
        return self.stcg_after if on >= self.change_date else self.stcg_before

    def ltcg(self, on: dt.date) -> float:
        return self.ltcg_after if on >= self.change_date else self.ltcg_before

    def ltcg_exemption(self, on: dt.date) -> float:
        return (
            self.ltcg_exemption_after if on >= self.change_date
            else self.ltcg_exemption_before
        )


# =====================================================================
# Retraining schemes (Phase 7)
# =====================================================================
@dataclass(frozen=True)
class RetrainConfig:
    """
    The four schemes to be compared.

    The comparison is only valid because intensity is a percentile of the
    TRAINING distribution rather than a raw score. A fixed cut on the raw score
    would mean four different things under four schemes, and the benchmark
    would be measuring the scoring scale instead of the schemes.
    """

    rolling_window_years: int = 3
    incremental_step_months: int = 1
    ewma_decay: float = 0.994                # per the project brief
    vol_purge_quantile: float = 0.90
    trading_days_per_year: int = 252


# =====================================================================
# Aggregate
# =====================================================================
@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    tax: TaxConfig = field(default_factory=TaxConfig)
    retrain: RetrainConfig = field(default_factory=RetrainConfig)


DEFAULT_CONFIG = Config()


def ensure_dirs() -> None:
    """Create output directories. Raw data is read-only and never created."""
    for path in (DATA_INTERIM, DATA_PROCESSED, REPORTS, REPORTS / "figures", MODELS):
        path.mkdir(parents=True, exist_ok=True)
