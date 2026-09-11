"""
Configuration objects for the corrected CCT engine.

Deliberately dataclasses rather than module-level globals: v1's
`_retry_with_relaxed_params` reached into `config` and mutated it mid-run, so the
settings printed in the report were not the settings that produced it. Here a run
owns an immutable-by-convention config object that is echoed into the result.

All distance thresholds are expressed in ATR multiples, not in absolute price.
v1 hard-coded things like SWEEP_MIN_PIPS = 0.10 and MIN_RECTANGLE_SIZE_PIPS = 0.15,
which mean completely different things on gold at 1800 vs 3600, and nothing at all
on EURUSD. ATR-relative thresholds transfer across symbols and across volatility
regimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace


@dataclass
class StrategyConfig:
    # --- timeframes ---
    tf_direction: str = "4h"
    tf_weakness: str = "15m"
    tf_entry: str = "1m"

    # --- step 1: 4H direction candle -------------------------------------- #
    direction_mode: str = "engulf_or_reclaim"
    """How the higher-timeframe bias is established.

    'engulf'  - v1's rule: sweep the prior bar's extreme AND close beyond its
                opposite extreme. Correct in spirit but fires on roughly 1.5% of
                4H bars, which is ~4 signals in six weeks. Not enough
                observations to measure anything, which is part of why v1's 47
                trades could not have told anyone whether the idea works.
    'reclaim' - sweep the prior bar's extreme and close back beyond the midpoint
                of its range: the same liquidity-grab logic with a lower bar.
    'engulf_or_reclaim' - either.
    """
    require_full_engulf: bool = True
    """Retained for the audit's v1 reproduction; only read when
    direction_mode == 'engulf'."""
    min_direction_body_atr: float = 0.15
    """Reject doji-ish 'engulfings' whose body is noise relative to 4H ATR.

    This single number decides the sample size, so it belongs in the walk-forward
    grid rather than in a hard-coded constant. Measured on gold-like bars, signal
    count as a share of 4H bars: 0.0 -> 10.8%, 0.15 -> 9.1%, 0.4 -> 6.3%,
    0.5 -> 5.4%, i.e. about 15 trades a month down to about 6."""
    direction_atr_period: int = 14
    use_trend_filter: bool = True
    trend_ema: int = 50
    """Only take longs whose 4H close is above the 4H EMA, and vice versa."""
    direction_valid_hours: float = 18.0
    """A direction signal expires this long after the 4H bar closes.

    This and `max_setups_per_direction` are set for observation count, not for
    return. Strict engulfing fires on ~5% of 4H bars; adding the reclaim variant
    roughly doubles that. See the power calculation printed with every result for
    what the resulting rate implies about how long confirmation takes."""
    max_active_directions: int = 1
    """Newer direction signals replace older ones rather than stacking."""

    # --- step 2: 15M weakness / liquidity sweep ---------------------------- #
    swing_k: int = 3
    """Fractal width for pivot detection on the weakness timeframe."""
    atr_period: int = 14
    sweep_min_atr: float = 0.15
    """How far beyond the level price must trade for it to count as a sweep."""
    sweep_max_atr: float = 2.5
    """A sweep deeper than this is a breakout, not a liquidity grab."""
    min_close_position: float = 0.55
    """Rejection quality: for a long, close must sit in the upper 55% of the range."""
    rect_min_atr: float = 0.20
    rect_max_atr: float = 1.60
    """Rectangle (future stop distance) must be a sane fraction of ATR."""
    level_max_age_bars: int = 96
    """Ignore pivots older than this many weakness bars (96 x 15m = 24h)."""
    level_dedupe_atr: float = 0.35
    """Levels closer together than this are treated as one level."""
    max_setups_per_direction: int = 3

    # --- step 3: 1M rectangle entry ---------------------------------------- #
    entry_window_min: int = 90
    """Setup expires if the breakout does not happen inside this many minutes."""
    entry_buffer_atr: float = 0.02
    """Close must clear the rectangle edge by this much (filters ties)."""
    invalidate_on_rect_breach: bool = True
    """THE fix for AUDIT #2: if price trades to the far side of the rectangle
    before the breakout, the setup is dead. v1 happily entered anyway."""
    cancel_on_opposing_direction: bool = True

    # --- exits ------------------------------------------------------------- #
    sl_buffer_atr: float = 0.10
    """Stop sits this far beyond the rectangle's far edge."""
    tp_r: float = 3.0
    """Fixed target in multiples of initial risk. 0 disables the fixed target."""
    use_partial: bool = True
    partial_at_r: float = 1.5
    partial_frac: float = 0.5
    move_to_breakeven_after_partial: bool = True
    use_trail: bool = True
    trail_start_r: float = 2.0
    trail_atr: float = 1.2
    """Trail distance in ATR of the weakness timeframe, applied from the next bar."""
    trail_r: float = 0.0
    """Used instead of `trail_atr` when `trail_atr` <= 0: trail distance in
    multiples of initial risk (this is how v1 trailed)."""
    disable_tp_when_trailing: bool = False
    """v1 stopped checking the fixed target once the trail armed, which made the
    bar where price touched both resolve however the code happened to order it."""
    max_hold_min: int = 480
    """Time stop. Gold's 4H structure is stale long before this."""

    # --- filters ----------------------------------------------------------- #
    sessions: tuple[tuple[int, int], ...] = ((7, 16), (12, 21))
    """UTC hour windows in which entries are allowed (London, New York).
    Empty tuple = trade around the clock."""
    vol_rank_min: float = 0.15
    vol_rank_max: float = 0.95
    """Skip the deadest and the most violent volatility regimes, ranked against a
    trailing window rather than the whole sample."""
    vol_rank_window: int = 480

    def with_(self, **kw) -> "StrategyConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CostModel:
    """
    Charged on both sides of every trade. v1 charged nothing on entry and filled
    exits exactly at the stop/target (AUDIT #6).
    """

    spread: float = 0.30
    """Full bid/ask spread in price units. Half is paid per side."""
    slippage: float = 0.05
    """Additional adverse price movement per side, in price units."""
    stop_slippage: float = 0.10
    """Extra adverse slippage when a stop is triggered (market order into a move)."""
    commission_per_lot: float = 3.5
    """Charged per lot per side."""
    contract_size: float = 100.0
    """Units of the underlying per lot. 100 oz for XAUUSD / GC."""
    lot_step: float = 0.01
    min_lot: float = 0.01
    max_lot: float = 200.0

    def per_side(self) -> float:
        return self.spread / 2.0 + self.slippage

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskConfig:
    initial_capital: float = 10_000.0
    risk_frac: float = 0.0075
    """Fraction of equity risked per trade. 0.75% default.

    v1 used 0.25. At its own claimed 48% win rate on 3:1, five losses in a row
    (~3.8% likely in any five-trade window) takes 10k to 2.4k. And that win rate
    was measured with 8 hours of look-ahead, so the true Kelly fraction is far
    below 25% and quite possibly negative."""
    compounding: bool = True
    max_risk_frac: float = 0.02
    """Hard ceiling, regardless of what an optimiser asks for."""
    max_concurrent: int = 2
    """Actually enforced here, unlike v1 where the check was dead code."""
    max_concurrent_same_direction: int = 1
    daily_loss_cap_frac: float = 0.03
    """Stop opening trades for the rest of the UTC day after this drawdown."""
    max_consecutive_losses: int = 4
    cooldown_hours_after_streak: float = 24.0
    max_leverage: float = 20.0
    """Notional cap: lots * contract_size * price <= equity * max_leverage."""
    ruin_floor_frac: float = 0.30
    """Stop the whole run if equity falls below this fraction of the start.
    A backtest that continues past a 70% drawdown is answering a question no
    account owner would ever get to ask."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BiasFlags:
    """
    Reintroduces v1's defects, one at a time, so `run.py audit` can price each of
    them on identical bars. Everything defaults to off; the honest engine never
    sets any of these.
    """

    lookahead_direction: bool = False
    """AUDIT #1: stamp the 4H signal at the bar's OPEN time (v1 behaviour)."""
    lookahead_swings: bool = False
    """Use pivots from the moment they exist in the array, before confirmation."""
    skip_invalidation: bool = False
    """AUDIT #2: enter even if price already traded through the future stop."""
    optimistic_intrabar: bool = False
    """AUDIT #5: resolve target-before-stop inside a bar, and trail using the
    current bar's own extreme before testing that same bar's opposite extreme."""
    zero_costs: bool = False
    """AUDIT #6: no spread, no slippage, no commission."""

    def any_on(self) -> bool:
        return any(vars(self).values())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunConfig:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    costs: CostModel = field(default_factory=CostModel)
    risk: RiskConfig = field(default_factory=RiskConfig)
    bias: BiasFlags = field(default_factory=BiasFlags)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.to_dict(),
            "costs": self.costs.to_dict(),
            "risk": self.risk.to_dict(),
            "bias": self.bias.to_dict(),
        }


# Presets ------------------------------------------------------------------- #

def gold_costs() -> CostModel:
    """Retail XAUUSD, ECN-ish. Widen if your broker is worse; results are sensitive."""
    return CostModel(
        spread=0.30,
        slippage=0.05,
        stop_slippage=0.10,
        commission_per_lot=3.5,
        contract_size=100.0,
    )


def v1_equivalent() -> RunConfig:
    """
    v1's parameters and v1's defects, for the audit baseline. This is what
    produced "775% monthly".
    """
    return RunConfig(
        strategy=StrategyConfig(
            direction_mode="engulf",
            use_trend_filter=True,
            min_direction_body_atr=0.0,
            sweep_min_atr=0.0,
            sweep_max_atr=99.0,
            min_close_position=0.0,
            rect_min_atr=0.0,
            rect_max_atr=99.0,
            entry_window_min=120,
            entry_buffer_atr=0.0,
            # Left ON so the `skip_invalidation` bias flag has something to switch
            # off. v1 effectively ran with it off.
            invalidate_on_rect_breach=True,
            cancel_on_opposing_direction=False,
            tp_r=3.0,
            use_partial=False,
            use_trail=True,
            trail_start_r=3.0,
            trail_atr=0.0,   # v1 trailed by R, not by ATR
            trail_r=2.5,
            disable_tp_when_trailing=True,
            max_hold_min=100_000,
            sessions=(),
            vol_rank_min=0.0,
            vol_rank_max=1.0,
            max_setups_per_direction=3,
            direction_valid_hours=24.0,
        ),
        costs=CostModel(
            spread=0.20,
            slippage=0.0,
            stop_slippage=0.0,
            commission_per_lot=0.0,
            contract_size=100.0,
        ),
        risk=RiskConfig(
            risk_frac=0.25,
            max_risk_frac=0.25,
            max_concurrent=99,
            max_concurrent_same_direction=99,
            daily_loss_cap_frac=1.0,
            max_consecutive_losses=99,
            max_leverage=1e9,
            ruin_floor_frac=0.10,
        ),
        bias=BiasFlags(
            lookahead_direction=True,
            lookahead_swings=True,
            skip_invalidation=True,
            optimistic_intrabar=True,
            zero_costs=False,
        ),
    )
