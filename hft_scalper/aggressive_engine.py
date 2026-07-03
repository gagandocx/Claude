"""
Aggressive backtesting engine with compounding position sizing.

Key features:
- Dynamic lot sizing based on current equity (compounding)
- Risk % per trade model (scales lots with equity growth)
- Trailing stop functionality
- Anti-drawdown circuit breaker
- Realistic execution: slippage, commission, spread
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class AggressiveConfig:
    """Configuration for the aggressive backtest engine."""
    initial_equity: float = 1000.0
    leverage: int = 500
    contract_size: float = 100.0  # 100 oz per standard lot
    slippage_points: float = 0.3
    commission_per_lot_rt: float = 7.0  # $7 per standard lot round-trip
    risk_pct: float = 0.03  # Risk 3% of equity per trade
    max_risk_pct: float = 0.05  # Maximum risk per trade
    min_lot: float = 0.01
    max_lot: float = 5.0
    # Trailing stop settings
    trail_activate_rr: float = 1.5  # Activate trailing after 1.5x risk in profit
    trail_step_rr: float = 0.5  # Trail by 0.5x risk increments
    # Drawdown circuit breaker
    dd_threshold: float = 0.08  # 8% drawdown triggers size reduction
    dd_reduction: float = 0.5  # Reduce size by 50% when DD threshold hit
    # Recovery: resume normal sizing when equity recovers to 95% of peak
    dd_recovery_pct: float = 0.95


@dataclass
class AggressiveTrade:
    """Record of a single trade."""
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    pnl: float
    pnl_points: float
    commission: float
    exit_reason: str
    equity_at_entry: float
    equity_at_exit: float


@dataclass
class AggressiveResult:
    """Complete backtest results."""
    trades: list = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_dollar: float = 0.0
    sharpe_ratio: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    max_consecutive_losses: int = 0
    avg_lot_size: float = 0.0
    total_commission: float = 0.0


def calculate_lot_size(
    equity: float,
    risk_pct: float,
    sl_distance: float,
    config: AggressiveConfig,
    in_drawdown: bool = False,
) -> float:
    """
    Calculate position size based on equity and risk parameters.
    
    This is the KEY to compounding: as equity grows, lot size grows proportionally.
    """
    if sl_distance <= 0:
        return config.min_lot
    
    # Dollar risk for this trade
    dollar_risk = equity * risk_pct
    
    # Apply drawdown reduction
    if in_drawdown:
        dollar_risk *= config.dd_reduction
    
    # lot_size = dollar_risk / (sl_distance * contract_size)
    # Because PnL = points * contract_size * lots
    lot_size = dollar_risk / (sl_distance * config.contract_size)
    
    # Clamp to limits
    lot_size = max(config.min_lot, min(config.max_lot, lot_size))
    
    # Round to 0.01
    lot_size = round(lot_size, 2)
    
    # Margin check
    # We need a price estimate - use a rough average (~4700)
    # margin_required = (price * contract_size * lot_size) / leverage
    # At 4700, 0.1 lot: (4700 * 100 * 0.1) / 500 = $94
    # We need at least 2x margin buffer
    estimated_price = 4700.0
    margin_required = (estimated_price * config.contract_size * lot_size) / config.leverage
    if margin_required > equity * 0.8:
        lot_size = (equity * 0.8 * config.leverage) / (estimated_price * config.contract_size)
        lot_size = max(config.min_lot, round(lot_size, 2))
    
    return lot_size


def run_aggressive_backtest(
    bars: pd.DataFrame,
    signals: np.ndarray,
    config: Optional[AggressiveConfig] = None,
) -> AggressiveResult:
    """
    Run backtest with compounding position sizing and trailing stops.
    
    Parameters
    ----------
    bars : pd.DataFrame
        OHLC bars with columns: open, high, low, close, avg_spread
    signals : np.ndarray
        Array of shape (n_bars, 3): [direction, sl_distance, tp_distance]
    config : AggressiveConfig
        Engine configuration
        
    Returns
    -------
    AggressiveResult
        Complete backtest results
    """
    if config is None:
        config = AggressiveConfig()
    
    n_bars = len(bars)
    assert len(signals) == n_bars, f"Signal length {len(signals)} != bar length {n_bars}"
    
    # Extract arrays for speed
    opens = bars["open"].values
    highs = bars["high"].values
    lows = bars["low"].values
    closes = bars["close"].values
    spreads = bars["avg_spread"].values if "avg_spread" in bars.columns else np.full(n_bars, 0.1)
    
    # Simulation state
    equity = config.initial_equity
    peak_equity = equity
    equity_curve = np.zeros(n_bars)
    trades: List[AggressiveTrade] = []
    
    # Position state
    position = 0  # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    initial_sl_distance = 0.0
    lot_size = 0.0
    entry_bar = 0
    equity_at_entry = 0.0
    trail_active = False
    
    # Drawdown state
    in_drawdown = False
    
    for i in range(n_bars):
        # Update peak equity and drawdown state
        if equity > peak_equity:
            peak_equity = equity
            in_drawdown = False
        elif peak_equity > 0:
            current_dd = (peak_equity - equity) / peak_equity
            if current_dd >= config.dd_threshold:
                in_drawdown = True
            elif equity >= peak_equity * config.dd_recovery_pct:
                in_drawdown = False
        
        # Stop-out check
        if equity < config.initial_equity * 0.1:
            equity_curve[i:] = equity
            break
        
        # Process open position
        if position != 0:
            # Trailing stop logic
            if not trail_active:
                if position == 1:
                    current_profit = highs[i] - entry_price
                elif position == -1:
                    current_profit = entry_price - lows[i]
                else:
                    current_profit = 0
                
                if current_profit >= initial_sl_distance * config.trail_activate_rr:
                    trail_active = True
                    # Move stop to breakeven + small buffer
                    if position == 1:
                        new_sl = entry_price + initial_sl_distance * 0.2
                        stop_loss = max(stop_loss, new_sl)
                    else:
                        new_sl = entry_price - initial_sl_distance * 0.2
                        stop_loss = min(stop_loss, new_sl)
            
            if trail_active:
                # Trail the stop
                trail_distance = initial_sl_distance * config.trail_step_rr
                if position == 1:
                    new_trail = highs[i] - trail_distance
                    if new_trail > stop_loss:
                        stop_loss = new_trail
                elif position == -1:
                    new_trail = lows[i] + trail_distance
                    if new_trail < stop_loss:
                        stop_loss = new_trail
            
            # Check SL/TP
            exit_price_val = 0.0
            exit_reason = ""
            
            if position == 1:
                sl_hit = lows[i] <= stop_loss
                tp_hit = highs[i] >= take_profit
                
                if sl_hit and tp_hit:
                    dist_sl = opens[i] - stop_loss
                    dist_tp = take_profit - opens[i]
                    if dist_sl <= dist_tp:
                        exit_price_val = stop_loss - config.slippage_points
                        exit_reason = "sl"
                    else:
                        exit_price_val = take_profit
                        exit_reason = "tp"
                elif sl_hit:
                    exit_price_val = stop_loss - config.slippage_points
                    exit_reason = "sl"
                elif tp_hit:
                    exit_price_val = take_profit
                    exit_reason = "tp"
                elif signals[i, 0] == -1:
                    exit_price_val = closes[i] - spreads[i] / 2 - config.slippage_points
                    exit_reason = "signal"
                    
            elif position == -1:
                sl_hit = highs[i] >= stop_loss
                tp_hit = lows[i] <= take_profit
                
                if sl_hit and tp_hit:
                    dist_sl = stop_loss - opens[i]
                    dist_tp = opens[i] - take_profit
                    if dist_sl <= dist_tp:
                        exit_price_val = stop_loss + config.slippage_points
                        exit_reason = "sl"
                    else:
                        exit_price_val = take_profit
                        exit_reason = "tp"
                elif sl_hit:
                    exit_price_val = stop_loss + config.slippage_points
                    exit_reason = "sl"
                elif tp_hit:
                    exit_price_val = take_profit
                    exit_reason = "tp"
                elif signals[i, 0] == 1:
                    exit_price_val = closes[i] + spreads[i] / 2 + config.slippage_points
                    exit_reason = "signal"
            
            if exit_reason:
                # Calculate PnL
                if position == 1:
                    pnl_points = exit_price_val - entry_price
                else:
                    pnl_points = entry_price - exit_price_val
                
                commission = config.commission_per_lot_rt * lot_size
                pnl_dollar = pnl_points * config.contract_size * lot_size - commission
                equity += pnl_dollar
                
                trades.append(AggressiveTrade(
                    entry_bar=entry_bar, exit_bar=i, direction=position,
                    entry_price=entry_price, exit_price=exit_price_val,
                    stop_loss=stop_loss, take_profit=take_profit,
                    lot_size=lot_size, pnl=pnl_dollar, pnl_points=pnl_points,
                    commission=commission, exit_reason=exit_reason,
                    equity_at_entry=equity_at_entry, equity_at_exit=equity,
                ))
                position = 0
                trail_active = False
        
        # Entry logic
        if position == 0 and signals[i, 0] != 0:
            direction = int(signals[i, 0])
            sl_dist = signals[i, 1]
            tp_dist = signals[i, 2]
            
            if sl_dist <= 0 or tp_dist <= 0:
                equity_curve[i] = equity
                continue
            
            # Calculate dynamic lot size
            lot_size = calculate_lot_size(equity, config.risk_pct, sl_dist, config, in_drawdown)
            
            # Entry price with spread and slippage
            if direction == 1:
                entry_price = closes[i] + spreads[i] / 2 + config.slippage_points
                stop_loss = entry_price - sl_dist
                take_profit = entry_price + tp_dist
            else:
                entry_price = closes[i] - spreads[i] / 2 - config.slippage_points
                stop_loss = entry_price + sl_dist
                take_profit = entry_price - tp_dist
            
            initial_sl_distance = sl_dist
            position = direction
            entry_bar = i
            equity_at_entry = equity
            trail_active = False
        
        equity_curve[i] = equity
    
    # Close any remaining position
    if position != 0:
        if position == 1:
            exit_price_val = closes[-1] - spreads[-1] / 2 - config.slippage_points
            pnl_points = exit_price_val - entry_price
        else:
            exit_price_val = closes[-1] + spreads[-1] / 2 + config.slippage_points
            pnl_points = entry_price - exit_price_val
        
        commission = config.commission_per_lot_rt * lot_size
        pnl_dollar = pnl_points * config.contract_size * lot_size - commission
        equity += pnl_dollar
        equity_curve[-1] = equity
        
        trades.append(AggressiveTrade(
            entry_bar=entry_bar, exit_bar=n_bars - 1, direction=position,
            entry_price=entry_price, exit_price=exit_price_val,
            stop_loss=stop_loss, take_profit=take_profit,
            lot_size=lot_size, pnl=pnl_dollar, pnl_points=pnl_points,
            commission=commission, exit_reason="eod",
            equity_at_entry=equity_at_entry, equity_at_exit=equity,
        ))
    
    # Compute metrics
    result = _compute_aggressive_metrics(trades, equity_curve, config)
    return result


def _compute_aggressive_metrics(
    trades: List[AggressiveTrade],
    equity_curve: np.ndarray,
    config: AggressiveConfig,
) -> AggressiveResult:
    """Compute performance metrics."""
    result = AggressiveResult()
    result.trades = trades
    result.equity_curve = equity_curve
    result.total_trades = len(trades)
    
    if result.total_trades == 0:
        result.final_equity = config.initial_equity
        return result
    
    # Final equity
    valid_eq = equity_curve[equity_curve > 0]
    result.final_equity = valid_eq[-1] if len(valid_eq) > 0 else config.initial_equity
    result.total_return_pct = ((result.final_equity - config.initial_equity) / config.initial_equity) * 100
    
    # Trade stats
    pnls = np.array([t.pnl for t in trades])
    result.winning_trades = int(np.sum(pnls > 0))
    result.losing_trades = int(np.sum(pnls <= 0))
    result.win_rate = result.winning_trades / result.total_trades
    
    # Profit factor
    gross_profit = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
    gross_loss = abs(float(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 0.001
    result.profit_factor = gross_profit / gross_loss
    
    # Averages
    result.avg_trade_pnl = float(np.mean(pnls))
    result.avg_winner = float(np.mean(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
    result.avg_loser = float(np.mean(pnls[pnls < 0])) if np.any(pnls < 0) else 0.0
    
    # Max consecutive losses
    streak = 0
    max_streak = 0
    for p in pnls:
        if p <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    result.max_consecutive_losses = max_streak
    
    # Drawdown
    if len(valid_eq) > 0:
        peak = np.maximum.accumulate(valid_eq)
        dd_pct = (valid_eq - peak) / peak * 100
        result.max_drawdown_pct = float(np.min(dd_pct))
        result.max_drawdown_dollar = float(np.min(valid_eq - peak))
    
    # Average lot size
    lot_sizes = [t.lot_size for t in trades]
    result.avg_lot_size = float(np.mean(lot_sizes))
    
    # Total commission
    result.total_commission = float(sum(t.commission for t in trades))
    
    # Sharpe ratio
    if len(valid_eq) > 1:
        eq_returns = np.diff(valid_eq) / valid_eq[:-1]
        eq_returns = eq_returns[np.isfinite(eq_returns)]
        if len(eq_returns) > 1 and np.std(eq_returns) > 0:
            bars_per_year = 252 * 390
            result.sharpe_ratio = float(
                np.mean(eq_returns) / np.std(eq_returns) * np.sqrt(bars_per_year)
            )
    
    return result
