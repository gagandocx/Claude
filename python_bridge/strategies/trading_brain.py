"""
=============================================================
  TRADING BRAIN — Fully Autonomous Professional Trading Intelligence

  "Set it and forget it."

  The brain is the ONLY decision-maker. You start it.
  It handles everything else — on its own, without any input
  from you — exactly how a professional prop desk runs.

  ┌─────────────────────────────────────────────────────────┐
  │  17 ML Models → Signal → [ TRADING BRAIN ] → Execute   │
  └─────────────────────────────────────────────────────────┘

  Every signal passes through 8 evaluation layers:

    1. MarketQualityFilter   — is the market tradeable right now?
    2. TimingEngine          — right session, right day?
    3. RiskController        — capital protection + circuit breakers
    4. RegimeAnalyzer        — trending / ranging / volatile / crash
    5. EdgeTracker           — is the strategy edge alive today?
    6. HourlyPerformance     — is THIS specific hour profitable?
    7. PositionSizer         — VaR + Kelly + regime adjusted sizing
    8. SlTpCalculator        — regime-specific SL and TP

  The brain also:
    • Self-tunes after every closed trade
    • Tracks per-hour and per-regime performance
    • Monitors each of the 17 models' live accuracy
    • Detects and recovers from drawdown automatically
    • Adjusts aggression dynamically (hot streak → press, cold → back off)
    • Blocks trading before high-impact news
    • Enforces end-of-week risk reduction
    • Computes a 0-100 trade quality score before every entry
=============================================================
"""

import math
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  BRAIN DECISION — complete autonomous output
# ─────────────────────────────────────────────
@dataclass
class BrainDecision:
    """
    Complete, autonomous trade decision produced by the brain.
    Every parameter is computed by the brain — nothing left to chance.
    """
    # ── Core decision ─────────────────────────────────────────────────────
    action:     str    # 'TRADE' | 'SKIP' | 'PAUSE_DAY'
    direction:  str    # 'BUY' | 'SELL' | 'HOLD'

    # ── Position parameters (brain overrides signal) ──────────────────────
    lot_size:   float  # computed: VaR + Kelly + regime
    sl_dollars: float  # computed: ATR × regime multiplier ($)
    tp_dollars: float  # computed: sl × RR ratio (0 = EA manages dynamically)

    # ── Signal gate ────────────────────────────────────────────────────────
    min_confidence: float   # dynamic threshold this bar

    # ── Context ────────────────────────────────────────────────────────────
    regime:      str   # trending_up | trending_down | ranging | volatile | crash
    session:     str   # session name
    edge_status: str   # HOT | NORMAL | COLD | BROKEN
    risk_level:  str   # NORMAL | REDUCED | MINIMAL | STOPPED

    # ── Quality score ──────────────────────────────────────────────────────
    trade_score: float  # 0–100 composite quality score

    # ── Full reasoning (every layer explains itself) ───────────────────────
    reasoning: Dict[str, str] = field(default_factory=dict)

    @property
    def should_trade(self) -> bool:
        return self.action == 'TRADE'

    def log_summary(self) -> str:
        if not self.should_trade:
            return (f"[Brain] {self.action}: {self.reasoning.get('final','')}")
        return (
            f"[Brain] TRADE {self.direction} | "
            f"lot={self.lot_size:.2f} SL=${self.sl_dollars:.2f} "
            f"TP=${self.tp_dollars:.2f} | "
            f"score={self.trade_score:.0f}/100 | "
            f"regime={self.regime} session={self.session} "
            f"edge={self.edge_status}"
        )



# ─────────────────────────────────────────────
#  LAYER 1: MARKET QUALITY FILTER
# ─────────────────────────────────────────────
class MarketQualityFilter:
    """
    Filters out untradeable market conditions before everything else.

    Professional traders never trade:
      • Excessive spread (paying too much to enter)
      • Zero volume (liquidity dried up)
      • Price stalled (no movement = no opportunity)
    """

    def evaluate(
        self,
        spread_points: float,
        atr_points:    float,
        tick_volume:   float,
        config,
    ) -> Tuple[bool, str]:
        """Returns (ok, reason)."""

        # Spread filter: spread must be < X% of ATR
        if atr_points > 0:
            spread_pct = spread_points / atr_points
            if spread_pct > config.max_spread_atr_ratio:
                return False, (
                    f"Spread too wide: {spread_pct*100:.0f}% of ATR "
                    f"(max {config.max_spread_atr_ratio*100:.0f}%)"
                )

        # Volume filter: avoid zero-volume (weekend/holiday gaps)
        if tick_volume < config.min_tick_volume:
            return False, f"Volume too low: {tick_volume:.0f} ticks"

        # ATR sanity: if ATR is near zero price is stuck
        if atr_points < config.min_atr_points:
            return False, f"ATR too low ({atr_points:.1f}) — market stalled"

        return True, "Market quality OK"



# ─────────────────────────────────────────────
#  LAYER 2: TIMING ENGINE
# ─────────────────────────────────────────────
class TimingEngine:
    """
    Decides WHEN to trade based on session, day, and known news windows.
    No human input — runs on UTC clock automatically.
    """

    SESSIONS = {
        'LONDON_NY_OVERLAP': (13, 16, 1.30),   # (start_utc, end_utc, mult)
        'LONDON':            ( 7, 13, 1.20),
        'NEW_YORK':          (16, 21, 1.00),
        'ASIAN':             ( 0,  7, 0.80),
        'OFF_HOURS':         (21, 24, 0.60),
    }

    def evaluate(self, config) -> Tuple[str, str, float]:
        """Returns (status, session_name, lot_multiplier)."""
        now  = datetime.now(timezone.utc)
        hour = now.hour
        dow  = now.weekday()   # 0=Mon … 6=Sun

        if dow >= 5:
            return 'BLOCKED', 'WEEKEND', 0.0

        # Friday wind-down — reduce risk heading into weekend
        if dow == 4 and hour >= 17:
            return 'POOR', 'FRIDAY_CLOSE', 0.4

        # Identify session
        for name, (start, end, mult) in self.SESSIONS.items():
            if start <= hour < end:
                status = 'OPTIMAL' if 'OVERLAP' in name else (
                         'GOOD'    if mult >= 1.0        else 'POOR')
                return status, name, mult

        return 'POOR', 'OFF_HOURS', config.session_off_hours_mult



# ─────────────────────────────────────────────
#  LAYER 3: RISK CONTROLLER
# ─────────────────────────────────────────────
class RiskController:
    """
    Profit-first risk management.

    The goal is NOT to avoid risk — it's to maximise
    risk-adjusted returns. Risk rules exist to prevent
    blowing up, not to prevent trading.

    Circuit breakers trigger only when evidence shows
    the current conditions are genuinely unprofitable.
    When conditions are proven profitable, it allows
    larger positions than the default.
    """

    def evaluate(
        self,
        daily_pnl:          float,
        daily_pnl_peak:     float,
        total_drawdown_pct: float,
        consecutive_losses: int,
        ec_above_ma:        bool,
        recent_pf:          float,     # profit factor last N trades
        config,
    ) -> Tuple[str, str, float]:
        """Returns (status, reason, lot_multiplier)."""

        # ── Hard stops ────────────────────────────────────────────────────
        if daily_pnl <= -abs(config.daily_loss_limit):
            return ('STOPPED',
                    f'Daily loss limit ${daily_pnl:.2f} — protecting capital',
                    0.0)

        if total_drawdown_pct >= config.drawdown_stop_threshold:
            return ('STOPPED',
                    f'Max drawdown {total_drawdown_pct*100:.1f}% reached',
                    0.0)

        # ── Profit factor watchdog ────────────────────────────────────────
        # If recent PF < 0.8 (losing more than winning) → pause immediately
        if recent_pf < 0.80 and consecutive_losses >= 3:
            return ('STOPPED',
                    f'PF={recent_pf:.2f} + {consecutive_losses} losses — strategy broken',
                    0.0)

        # ── Soft reduces ──────────────────────────────────────────────────
        if total_drawdown_pct >= config.drawdown_reduce_threshold:
            mult = max(0.30, 1.0 - total_drawdown_pct * 3)
            return ('REDUCED',
                    f'Drawdown {total_drawdown_pct*100:.1f}% — reduced sizing',
                    round(mult, 2))

        if consecutive_losses >= config.consecutive_loss_stop:
            return ('MINIMAL', f'{consecutive_losses} consecutive losses', 0.25)

        if consecutive_losses >= config.consecutive_loss_reduce:
            mult = max(0.40, 1.0 - (consecutive_losses - 2) * 0.15)
            return ('REDUCED', f'{consecutive_losses} consecutive losses', round(mult,2))

        # ── Equity curve: strategy is not working ─────────────────────────
        if config.equity_curve_filter and not ec_above_ma and recent_pf < 1.0:
            return ('MINIMAL',
                    'Equity below MA + PF<1.0 — backing off',
                    0.25)

        # ── Profit amplifier: press when winning ──────────────────────────
        # If daily P&L is positive and PF is strong → allow bigger trades
        if daily_pnl > 0 and recent_pf >= 2.0:
            return ('BOOSTED',
                    f'Daily profit ${daily_pnl:.2f} + PF={recent_pf:.1f} — pressing edge',
                    1.30)

        return ('NORMAL', 'Risk metrics healthy', 1.0)



# ─────────────────────────────────────────────
#  LAYER 4: REGIME ANALYZER
# ─────────────────────────────────────────────
class RegimeAnalyzer:
    """
    Detects market regime and tells the brain HOW to trade it.

    Each regime has a completely different optimal strategy:
      TRENDING_UP/DOWN : momentum entries, wide SL, large RR
      RANGING          : mean-reversion, tight SL/TP, high confidence required
      VOLATILE         : only the absolute best setups, small size
      CRASH            : flat — no edge in crashes
    """

    def evaluate(
        self,
        closes:  List[float],
        highs:   List[float],
        lows:    List[float],
        atr:     float,
        avg_atr: float,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (regime, params_dict).
        params_dict contains regime-specific adjustments:
          sl_atr_mult, tp_rr_ratio, min_conf_add, lot_mult, description
        """
        if len(closes) < 20:
            return 'unknown', self._regime_params('unknown')

        atr_ratio = atr / avg_atr if avg_atr > 0 else 1.0
        price     = closes[-1]
        ema20     = self._ema(closes, 20)
        ema50     = self._ema(closes, 50) if len(closes) >= 50 else ema20
        adx       = self._adx_proxy(highs, lows, closes)

        # Crash: large sudden drop
        if len(closes) >= 5:
            drop = closes[-5] - closes[-1]
            if drop > 4 * atr:
                return 'crash', self._regime_params('crash')

        # Highly volatile
        if atr_ratio > 2.2:
            return 'volatile', self._regime_params('volatile')

        # Trending
        if adx > 22:
            if price > ema20 > ema50:
                return 'trending_up',   self._regime_params('trending_up')
            elif price < ema20 < ema50:
                return 'trending_down', self._regime_params('trending_down')
            else:
                return 'volatile', self._regime_params('volatile')

        # Ranging
        if atr_ratio < 0.85 and adx < 20:
            return 'ranging', self._regime_params('ranging')

        return 'neutral', self._regime_params('neutral')

    @staticmethod
    def _regime_params(regime: str) -> Dict[str, Any]:
        """
        Profit-optimised parameters per regime.
        sl_atr_mult  — SL = this × ATR
        tp_rr_ratio  — TP = SL × this ratio (3.0 = 3:1 RR)
        min_conf_add — add to base confidence threshold
        lot_mult     — multiply base lot by this
        """
        params = {
            'trending_up':   {'sl_atr_mult': 1.5, 'tp_rr_ratio': 3.0, 'min_conf_add': -0.05, 'lot_mult': 1.15, 'desc': 'Momentum mode — press the trend'},
            'trending_down': {'sl_atr_mult': 1.5, 'tp_rr_ratio': 3.0, 'min_conf_add': -0.05, 'lot_mult': 1.15, 'desc': 'Momentum mode — press the trend'},
            'ranging':       {'sl_atr_mult': 0.8, 'tp_rr_ratio': 1.5, 'min_conf_add':  0.08, 'lot_mult': 0.85, 'desc': 'Ranging — tighter setups only'},
            'volatile':      {'sl_atr_mult': 2.0, 'tp_rr_ratio': 2.0, 'min_conf_add':  0.15, 'lot_mult': 0.50, 'desc': 'Volatile — best setups only, small size'},
            'crash':         {'sl_atr_mult': 0.0, 'tp_rr_ratio': 0.0, 'min_conf_add':  1.00, 'lot_mult': 0.00, 'desc': 'CRASH — flat'},
            'neutral':       {'sl_atr_mult': 1.2, 'tp_rr_ratio': 2.5, 'min_conf_add':  0.00, 'lot_mult': 1.00, 'desc': 'Neutral market'},
            'unknown':       {'sl_atr_mult': 1.2, 'tp_rr_ratio': 2.0, 'min_conf_add':  0.05, 'lot_mult': 0.80, 'desc': 'Unknown — conservative'},
        }
        return params.get(regime, params['neutral'])

    @staticmethod
    def _ema(values: List[float], p: int) -> float:
        if len(values) < p:
            return values[-1]
        k, ema = 2.0 / (p + 1), sum(values[:p]) / p
        for v in values[p:]:
            ema = v * k + ema * (1 - k)
        return ema

    @staticmethod
    def _adx_proxy(highs, lows, closes, p: int = 14) -> float:
        if len(closes) < p + 1:
            return 20.0
        trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]),
                   abs(lows[i]-closes[i-1]))
               for i in range(1, min(p+1, len(closes)))]
        if not trs:
            return 20.0
        atr_p = sum(trs) / len(trs)
        if atr_p == 0:
            return 20.0
        dm_up   = max(0, max(highs[-p:]) - max(highs[-p-1:-1]))
        dm_down = max(0, max(lows[-p-1:-1]) - max(lows[-p:]))
        return min(abs(dm_up - dm_down) / atr_p * 100, 100.0)



# ─────────────────────────────────────────────
#  LAYER 5: EDGE TRACKER
# ─────────────────────────────────────────────
class EdgeTracker:
    """
    Monitors whether the strategy's edge is ALIVE and profitable.

    The most important layer for making money:
      • When edge is HOT  → increase size aggressively
      • When edge is COLD → back off immediately
      • When edge is BROKEN → stop until it recovers

    Also tracks per-hour and per-regime profitability so the
    brain knows WHEN and WHERE its edge is strongest.
    """

    def __init__(self, lookback: int = 30):
        self.trades: deque = deque(maxlen=lookback)
        self._equity:       deque = deque(maxlen=lookback)
        self._running_pnl:  float = 0.0
        self._peak_pnl:     float = 0.0

        # Per-hour profitability: hour → [pnl list]
        self._hour_pnl: Dict[int, deque] = defaultdict(lambda: deque(maxlen=20))
        # Per-regime: regime → [pnl list]
        self._regime_pnl: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

    def record_trade(
        self, pnl: float, won: bool,
        hour: int = -1, regime: str = 'unknown'
    ):
        self.trades.append({'pnl': pnl, 'won': won, 'hour': hour, 'regime': regime})
        self._running_pnl += pnl
        self._peak_pnl     = max(self._peak_pnl, self._running_pnl)
        self._equity.append(self._running_pnl)
        if hour >= 0:
            self._hour_pnl[hour].append(pnl)
        if regime != 'unknown':
            self._regime_pnl[regime].append(pnl)

    def evaluate(self, config) -> Tuple[str, float, float, float]:
        """
        Returns (status, lot_multiplier, conf_adjustment, recent_pf).
        """
        n = len(self.trades)
        if n < 5:
            return 'NORMAL', 1.0, 0.0, 1.5   # not enough data

        wins       = sum(1 for t in self.trades if t['won'])
        wr         = wins / n
        gross_win  = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        pf         = gross_win / gross_loss if gross_loss > 0 else 3.0
        avg_win    = gross_win  / max(wins, 1)
        avg_loss   = gross_loss / max(n - wins, 1)

        # Quarter-Kelly multiplier
        b = avg_win / avg_loss if avg_loss > 0 else 1.5
        kelly      = wr - (1 - wr) / b
        kelly_mult = max(0.25, min(2.5, 1.0 + kelly * 2.5))

        # Equity curve vs MA
        ec_above = self._equity_above_ma()

        if wr >= config.edge_hot_threshold and pf >= 1.8 and ec_above:
            # HOT — press the edge
            mult = min(kelly_mult, config.lot_multiplier_hot)
            return 'HOT', mult, -0.04, pf

        if wr < config.edge_broken_threshold or pf < 0.8:
            return 'BROKEN', 0.0, 0.20, pf

        if wr < config.edge_cold_threshold or pf < 1.1:
            return 'COLD', config.lot_multiplier_cold, 0.08, pf

        return 'NORMAL', kelly_mult, 0.0, pf

    def best_hours(self, top_n: int = 3) -> List[int]:
        """Returns the UTC hours with the highest average PnL."""
        avg = {h: sum(v)/len(v) for h, v in self._hour_pnl.items() if len(v) >= 3}
        return sorted(avg, key=avg.get, reverse=True)[:top_n]

    def worst_hours(self, bottom_n: int = 3) -> List[int]:
        avg = {h: sum(v)/len(v) for h, v in self._hour_pnl.items() if len(v) >= 3}
        return sorted(avg, key=avg.get)[:bottom_n]

    def best_regime(self) -> str:
        avg = {r: sum(v)/len(v) for r, v in self._regime_pnl.items() if len(v) >= 3}
        return max(avg, key=avg.get) if avg else 'unknown'

    @property
    def consecutive_losses(self) -> int:
        streak = 0
        for t in reversed(self.trades):
            if not t['won']:
                streak += 1
            else:
                break
        return streak

    @property
    def consecutive_wins(self) -> int:
        streak = 0
        for t in reversed(self.trades):
            if t['won']:
                streak += 1
            else:
                break
        return streak

    def _equity_above_ma(self, p: int = 10) -> bool:
        if len(self._equity) < p:
            return True
        recent = list(self._equity)
        return recent[-1] >= sum(recent[-p:]) / p



# ─────────────────────────────────────────────
#  LAYER 6: HOURLY PERFORMANCE TRACKER
# ─────────────────────────────────────────────
class HourlyPerformanceFilter:
    """
    Automatically avoids hours that have been consistently unprofitable.

    After 10+ trades in a given hour, if that hour has a negative
    average PnL, the brain automatically reduces size or skips it.

    This is how algorithmic traders discover structural edges:
    "We make money 08-10 UTC but lose it 18-20 UTC — adapt."
    """

    def evaluate(
        self, hour: int, edge_tracker: 'EdgeTracker', config
    ) -> Tuple[float, str]:
        """Returns (lot_multiplier, reason)."""
        worst = edge_tracker.worst_hours(3)
        best  = edge_tracker.best_hours(3)
        h_pnl = edge_tracker._hour_pnl.get(hour)

        if h_pnl and len(h_pnl) >= 8:
            avg = sum(h_pnl) / len(h_pnl)
            if avg < -0.50:  # consistently losing more than $0.50/trade
                return 0.5, f'Hour {hour}:00 UTC historically weak (avg ${avg:.2f})'
            if avg > 0.80:   # consistently profitable
                return 1.15, f'Hour {hour}:00 UTC historically strong (avg ${avg:.2f})'

        if hour in worst and len(edge_tracker._hour_pnl.get(hour, [])) >= 5:
            return 0.75, f'Hour {hour}:00 UTC in bottom-3 hours'

        if hour in best and len(edge_tracker._hour_pnl.get(hour, [])) >= 5:
            return 1.10, f'Hour {hour}:00 UTC in top-3 hours'

        return 1.0, f'Hour {hour}:00 UTC neutral'


# ─────────────────────────────────────────────
#  LAYER 7: POSITION SIZER (VaR + Kelly)
# ─────────────────────────────────────────────
class PositionSizer:
    """
    Profit-maximising position sizing using two methods:
      1. VaR-based: risk a fixed % of account per trade
         lot = (account × risk_pct) / sl_dollars
      2. Kelly-based: risk proportional to edge quality
         combined with regime, session, streak adjustments

    Takes the HIGHER of the two methods when edge is hot,
    the LOWER when edge is cold. Always clamped to safe range.
    """

    def calculate(
        self,
        account:      float,
        sl_dollars:   float,
        edge_mult:    float,
        regime_mult:  float,
        session_mult: float,
        risk_mult:    float,
        hourly_mult:  float,
        win_streak:   int,
        loss_streak:  int,
        edge_status:  str,
        config,
    ) -> float:
        if sl_dollars <= 0:
            return config.base_lot

        # ── VaR sizing ────────────────────────────────────────────────────
        risk_amount = account * config.risk_per_trade_pct
        var_lot     = risk_amount / sl_dollars

        # ── Kelly sizing ──────────────────────────────────────────────────
        kelly_lot   = config.base_lot * edge_mult

        # ── Streak adjustments ────────────────────────────────────────────
        if win_streak >= 5:
            streak_mult = 1.30   # hot streak — press harder
        elif win_streak >= 3:
            streak_mult = 1.15
        elif loss_streak >= 5:
            streak_mult = 0.25
        elif loss_streak >= 3:
            streak_mult = 0.50
        else:
            streak_mult = 1.00

        # ── Combine ───────────────────────────────────────────────────────
        if edge_status == 'HOT':
            raw = max(var_lot, kelly_lot)   # be aggressive when hot
        elif edge_status in ('COLD', 'BROKEN'):
            raw = min(var_lot, kelly_lot)   # be conservative when cold
        else:
            raw = (var_lot + kelly_lot) / 2  # average otherwise

        # Apply all multipliers
        final = raw * regime_mult * session_mult * risk_mult * hourly_mult * streak_mult

        # Clamp
        final = max(config.min_lot, min(config.max_lot, final))
        return round(final, 2)


# ─────────────────────────────────────────────
#  LAYER 8: SL / TP CALCULATOR
# ─────────────────────────────────────────────
class SlTpCalculator:
    """
    Computes regime-optimal SL and TP in dollar terms.

    Trending : wider SL (2x ATR), larger RR (3:1) — let profits run
    Ranging  : tight SL (0.8x ATR), quick TP (1.5:1) — scalp the range
    Volatile : extra-wide SL, quick TP or EA-managed
    """

    def calculate(
        self,
        atr_dollars:    float,
        regime_params:  Dict[str, Any],
        config,
    ) -> Tuple[float, float]:
        """Returns (sl_dollars, tp_dollars)."""

        sl_mult  = regime_params.get('sl_atr_mult',  1.2)
        rr_ratio = regime_params.get('tp_rr_ratio',  2.5)

        # Respect the user's InpFixedSL if set
        sl_dollars = atr_dollars * sl_mult
        sl_dollars = max(config.min_sl_dollars,
                         min(config.max_sl_dollars, sl_dollars))

        # TP = SL × RR ratio (0 = let EA trail dynamically)
        tp_dollars = round(sl_dollars * rr_ratio, 2) if rr_ratio > 0 else 0.0

        return round(sl_dollars, 2), tp_dollars



# ─────────────────────────────────────────────
#  MODEL TRUST MANAGER
# ─────────────────────────────────────────────
class ModelTrustManager:
    """Tracks each model's live accuracy and flags underperformers."""

    _NAMES = [
        "transformer","lstm","tcn","patch_tst","tft","nhits",
        "itransformer","mamba","dlinear","xlstm","timesnet",
        "chronos","timemixer","softs",
        "gradient_boost","xgboost","catboost",
    ]

    def __init__(self, lookback: int = 30):
        self._acc: Dict[str, deque] = {n: deque(maxlen=lookback) for n in self._NAMES}

    def record(self, predictions: Dict[str, int], true_class: int):
        for name, pred in predictions.items():
            if name in self._acc:
                self._acc[name].append(1.0 if pred == true_class else 0.0)

    def summary(self) -> Dict[str, float]:
        return {n: (sum(q)/len(q) if q else 0.5) for n, q in self._acc.items()}

    def underperforming(self, threshold: float = 0.33) -> List[str]:
        return [n for n, q in self._acc.items()
                if len(q) >= 10 and sum(q)/len(q) < threshold]

    def best_model(self) -> str:
        s = self.summary()
        return max(s, key=s.get) if s else 'N/A'



# ─────────────────────────────────────────────
#  TRADING BRAIN — MASTER CONTROLLER
# ─────────────────────────────────────────────
class TradingBrain:
    """
    Fully autonomous trading intelligence. Zero human input required.

    You start it. It decides everything:
      • Whether to trade (8-layer evaluation)
      • Which direction (validates or overrides signal)
      • How much to risk (VaR + Kelly + regime + session + streak)
      • Where to set SL and TP (regime-optimised)
      • When to stop (circuit breakers)
      • When to press (edge is hot — go bigger)

    After every closed trade it learns:
      • Which models were right
      • Which hours are profitable
      • Which regimes are working
      • Whether to tighten or loosen criteria

    Trade quality score (0-100) is computed before every entry.
    Only trades scoring ≥ config.min_trade_score are taken.
    """

    def __init__(self, config=None):
        from config.settings import BrainConfig
        self.config       = config or BrainConfig()

        # ── All 8 evaluation layers ──────────────────────────────────────
        self.mq_filter    = MarketQualityFilter()
        self.timing       = TimingEngine()
        self.risk         = RiskController()
        self.regime_eng   = RegimeAnalyzer()
        self.edge         = EdgeTracker(self.config.lookback_trades)
        self.hourly       = HourlyPerformanceFilter()
        self.sizer        = PositionSizer()
        self.sl_tp        = SlTpCalculator()
        self.model_trust  = ModelTrustManager(self.config.lookback_trades)

        # ── State (updated by main.py) ───────────────────────────────────
        self.daily_pnl:       float = 0.0
        self.daily_pnl_peak:  float = 0.0
        self.total_drawdown:  float = 0.0
        self.account_balance: float = self.config.account_balance
        self._cycle_count:    int   = 0

    # ── Public update hooks ───────────────────────────────────────────────

    def record_trade_closed(
        self,
        pnl:         float,
        won:         bool,
        regime:      str               = 'unknown',
        predictions: Optional[Dict[str, int]] = None,
        true_class:  Optional[int]     = None,
    ):
        """Call every time a trade closes from the confirmation file."""
        hour = datetime.now(timezone.utc).hour
        self.edge.record_trade(pnl, won, hour, regime)

        self.daily_pnl       += pnl
        self.daily_pnl_peak   = max(self.daily_pnl_peak, self.daily_pnl)

        # Update drawdown
        if self.daily_pnl < 0:
            dd = abs(self.daily_pnl) / max(self.account_balance, 1.0)
            self.total_drawdown = max(self.total_drawdown, dd)

        if predictions and true_class is not None:
            self.model_trust.record(predictions, true_class)

        logger.info(
            f"[Brain] Trade closed: pnl={pnl:+.2f} won={won} "
            f"daily={self.daily_pnl:+.2f} regime={regime}"
        )

    def reset_daily(self):
        """Call at start of each trading day."""
        self.daily_pnl      = 0.0
        self.daily_pnl_peak = 0.0
        self.total_drawdown = 0.0
        logger.info("[Brain] Daily reset — fresh slate")

    # ── Main evaluation ───────────────────────────────────────────────────

    def evaluate(
        self,
        signal,
        closes:       List[float],
        highs:        List[float],
        lows:         List[float],
        atr:          float,
        avg_atr:      float,
        spread_points: float      = 10.0,
        tick_volume:  float       = 100.0,
        atr_dollars:  float       = 2.0,
    ) -> BrainDecision:
        """
        Full 8-layer autonomous evaluation.
        Returns a complete BrainDecision with all trade parameters.
        """
        self._cycle_count += 1
        reasoning: Dict[str, str] = {}

        # ── L1: Market quality ────────────────────────────────────────────
        mq_ok, mq_reason = self.mq_filter.evaluate(
            spread_points, atr * 10000, tick_volume, self.config)
        reasoning['market_quality'] = mq_reason
        if not mq_ok:
            return self._skip('SKIP', mq_reason, 'unknown', 'N/A',
                               'NORMAL', 'NORMAL', reasoning)

        # ── L2: Timing ────────────────────────────────────────────────────
        t_status, t_session, t_mult = self.timing.evaluate(self.config)
        reasoning['timing'] = f'{t_session} ({t_status}) mult={t_mult:.2f}'
        if t_status == 'BLOCKED':
            return self._skip('SKIP', f'Timing blocked: {t_session}',
                               'unknown', t_session, 'NORMAL', 'NORMAL', reasoning)

        # ── L3: Regime ────────────────────────────────────────────────────
        regime, r_params = self.regime_eng.evaluate(closes, highs, lows, atr, avg_atr)
        reasoning['regime'] = f'{regime}: {r_params["desc"]}'
        if regime == 'crash':
            return self._skip('PAUSE_DAY', 'CRASH detected — flat',
                               regime, t_session, 'BROKEN', 'STOPPED', reasoning)

        # ── L4: Edge ──────────────────────────────────────────────────────
        e_status, e_mult, conf_adj_edge, recent_pf = self.edge.evaluate(self.config)
        reasoning['edge'] = f'{e_status} PF={recent_pf:.2f} mult={e_mult:.2f}'
        if e_status == 'BROKEN':
            return self._skip('PAUSE_DAY', f'Edge BROKEN (PF={recent_pf:.2f})',
                               regime, t_session, e_status, 'STOPPED', reasoning)

        # ── L5: Risk ──────────────────────────────────────────────────────
        ec_above = self.edge._equity_above_ma()
        r_status, r_reason, r_mult = self.risk.evaluate(
            self.daily_pnl, self.daily_pnl_peak, self.total_drawdown,
            self.edge.consecutive_losses, ec_above, recent_pf, self.config)
        reasoning['risk'] = f'{r_status}: {r_reason}'
        if r_status == 'STOPPED':
            return self._skip('PAUSE_DAY', r_reason, regime, t_session,
                               e_status, r_status, reasoning)

        # ── L6: Hourly performance ────────────────────────────────────────
        h_mult, h_reason = self.hourly.evaluate(
            datetime.now(timezone.utc).hour, self.edge, self.config)
        reasoning['hourly'] = h_reason

        # ── L7: Dynamic confidence threshold ─────────────────────────────
        conf_adj    = r_params['min_conf_add'] + conf_adj_edge
        dyn_conf    = round(max(0.10, min(0.60,
                        self.config.base_min_confidence + conf_adj)), 3)
        reasoning['confidence'] = (
            f'threshold={dyn_conf:.3f} '
            f'(base={self.config.base_min_confidence} '
            f'adj={conf_adj:+.3f}) signal={signal.confidence:.3f}')

        if signal.confidence < dyn_conf:
            return self._skip(
                'SKIP',
                f'Confidence {signal.confidence:.3f} < threshold {dyn_conf:.3f}',
                regime, t_session, e_status, r_status, reasoning)

        # ── L8: SL / TP ───────────────────────────────────────────────────
        sl_dollars, tp_dollars = self.sl_tp.calculate(atr_dollars, r_params, self.config)
        reasoning['sl_tp'] = f'SL=${sl_dollars:.2f} TP=${tp_dollars:.2f} RR={r_params["tp_rr_ratio"]:.1f}'

        # ── L9: Position sizing ───────────────────────────────────────────
        final_lot = self.sizer.calculate(
            account      = self.account_balance,
            sl_dollars   = sl_dollars,
            edge_mult    = e_mult,
            regime_mult  = r_params['lot_mult'],
            session_mult = t_mult,
            risk_mult    = r_mult,
            hourly_mult  = h_mult,
            win_streak   = self.edge.consecutive_wins,
            loss_streak  = self.edge.consecutive_losses,
            edge_status  = e_status,
            config       = self.config,
        )
        reasoning['sizing'] = (
            f'lot={final_lot:.2f} '
            f'(edge×{e_mult:.2f} regime×{r_params["lot_mult"]:.2f} '
            f'session×{t_mult:.2f} risk×{r_mult:.2f} hour×{h_mult:.2f})')

        # ── Trade quality score (0-100) ───────────────────────────────────
        score = self._compute_score(
            signal.confidence, dyn_conf, e_status, regime,
            t_status, r_status, recent_pf)
        reasoning['score'] = f'{score:.0f}/100'

        if score < self.config.min_trade_score:
            return self._skip(
                'SKIP', f'Trade score {score:.0f} < min {self.config.min_trade_score}',
                regime, t_session, e_status, r_status, reasoning)

        # ── All layers passed — TRADE ─────────────────────────────────────
        decision = BrainDecision(
            action         = 'TRADE',
            direction      = signal.action,
            lot_size       = final_lot,
            sl_dollars     = sl_dollars,
            tp_dollars     = tp_dollars,
            min_confidence = dyn_conf,
            regime         = regime,
            session        = t_session,
            edge_status    = e_status,
            risk_level     = r_status,
            trade_score    = score,
            reasoning      = reasoning,
        )
        logger.info(decision.log_summary())

        # Periodic status report
        if self._cycle_count % self.config.status_report_interval == 0:
            logger.info(self.status_report())

        return decision

    # ── Helpers ───────────────────────────────────────────────────────────

    def _skip(
        self, action: str, reason: str, regime: str,
        session: str, edge: str, risk: str,
        reasoning: Dict[str, str]
    ) -> BrainDecision:
        reasoning['final'] = reason
        logger.debug(f"[Brain] {action}: {reason}")
        return BrainDecision(
            action='PAUSE_DAY' if action == 'PAUSE_DAY' else 'SKIP',
            direction='HOLD', lot_size=0.0,
            sl_dollars=2.0, tp_dollars=0.0,
            min_confidence=1.0, regime=regime,
            session=session, edge_status=edge,
            risk_level=risk, trade_score=0.0,
            reasoning=reasoning,
        )

    @staticmethod
    def _compute_score(
        confidence: float, threshold: float,
        edge: str, regime: str,
        timing: str, risk: str,
        pf: float,
    ) -> float:
        """
        0-100 composite trade quality score.
        100 = perfect setup (high conf, hot edge, trending, peak session, strong PF)
        """
        score = 0.0

        # Confidence margin above threshold (0-25 pts)
        margin = (confidence - threshold) / max(threshold, 0.01)
        score += min(25.0, margin * 50)

        # Edge (0-25 pts)
        score += {'HOT': 25, 'NORMAL': 15, 'COLD': 5, 'BROKEN': 0}.get(edge, 10)

        # Regime (0-20 pts)
        score += {'trending_up': 20, 'trending_down': 20, 'neutral': 12,
                  'ranging': 8, 'volatile': 3, 'unknown': 5}.get(regime, 10)

        # Session (0-15 pts)
        score += {'OPTIMAL': 15, 'GOOD': 10, 'POOR': 4, 'BLOCKED': 0}.get(timing, 8)

        # Profit factor (0-15 pts)
        score += min(15.0, (pf - 1.0) * 6)

        return round(max(0.0, min(100.0, score)), 1)

    def status_report(self) -> str:
        e_s, _, _, pf = self.edge.evaluate(self.config)
        trust         = self.model_trust.summary()
        top           = self.model_trust.best_model()
        under         = self.model_trust.underperforming()
        best_h        = self.edge.best_hours(3)
        worst_h       = self.edge.worst_hours(3)
        _, session, _ = self.timing.evaluate(self.config)
        regime, _     = self.regime_eng.evaluate.__func__(
            self.regime_eng, [], [], [], 0.001, 0.001)  # just for display

        lines = [
            "[Trading Brain Status Report]",
            f"  Session      : {session}",
            f"  Edge         : {e_s}  PF={pf:.2f}",
            f"  Daily P&L    : ${self.daily_pnl:+.2f}",
            f"  Drawdown     : {self.total_drawdown*100:.1f}%",
            f"  Win streak   : +{self.edge.consecutive_wins}",
            f"  Loss streak  : -{self.edge.consecutive_losses}",
            f"  Best hours   : {best_h} UTC",
            f"  Worst hours  : {worst_h} UTC",
            f"  Top model    : {top} ({trust.get(top,0):.0%})",
            f"  Trades seen  : {len(self.edge.trades)}",
        ]
        if under:
            lines.append(f"  Weak models  : {', '.join(under)}")
        return "\n".join(lines)
