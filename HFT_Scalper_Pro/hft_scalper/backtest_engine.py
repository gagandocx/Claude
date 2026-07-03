"""
High-performance vectorized backtest engine for 1-minute bars.

Simulates realistic execution with:
- Bid/ask spread modeling
- Configurable slippage
- Commission costs
- Maximum 1 position at a time
- Stop loss and take profit execution
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    """Configuration for the backtest engine."""
    slippage_points: float = 0.3  # Slippage in price points for gold
    commission_per_lot_rt: float = 0.7  # Round-trip commission (scaled for 0.1 lot)
    lot_size: float = 0.1  # 0.1 standard lot
    point_value: float = 1.0  # Dollar value per point per lot (gold = $1/point for 1 oz, $100/point for 100 oz)
    contract_size: float = 100.0  # 100 oz per standard lot (effective 10 oz for 0.1 lot)
    initial_equity: float = 1000.0  # Starting equity ($1000 deposit)
    leverage: int = 500  # 1:500 leverage


@dataclass
class Trade:
    """Record of a single trade."""
    entry_bar: int
    exit_bar: int
    direction: int  # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_points: float
    commission: float
    exit_reason: str  # 'tp', 'sl', 'signal', 'eod'


@dataclass
class BacktestResult:
    """Complete backtest results."""
    trades: list = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    max_consecutive_losses: int = 0
    avg_trade_duration: float = 0.0
    calmar_ratio: float = 0.0
    expectancy: float = 0.0


def run_backtest(bars: pd.DataFrame, signals: np.ndarray, config: Optional[BacktestConfig] = None) -> BacktestResult:
    """
    Run vectorized backtest on 1-minute bars with signals.

    Parameters
    ----------
    bars : pd.DataFrame
        OHLC bars with columns: open, high, low, close, avg_spread
    signals : np.ndarray
        Array of shape (n_bars, 3): [direction, sl_distance, tp_distance]
    config : BacktestConfig
        Backtest configuration

    Returns
    -------
    BacktestResult
        Complete backtest results with trades, equity curve, and metrics
    """
    if config is None:
        config = BacktestConfig()

    n_bars = len(bars)
    assert len(signals) == n_bars, f"Signal length {len(signals)} != bar length {n_bars}"

    # Extract bar data as numpy arrays for speed
    opens = bars["open"].values
    highs = bars["high"].values
    lows = bars["low"].values
    closes = bars["close"].values
    spreads = bars["avg_spread"].values if "avg_spread" in bars.columns else np.full(n_bars, 0.1)

    # Simulation state
    equity = config.initial_equity
    equity_curve = np.zeros(n_bars)
    trades = []
    position = 0  # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_bar = 0

    for i in range(n_bars):
        # Stop-out check: if equity drops below 20% of initial, stop trading
        if equity < config.initial_equity * 0.2:
            equity_curve[i:] = equity
            break

        # Check if we have an open position
        if position != 0:
            # Check stop loss and take profit hit within bar
            if position == 1:  # Long position
                sl_hit = lows[i] <= stop_loss
                tp_hit = highs[i] >= take_profit

                if sl_hit and tp_hit:
                    # Both SL and TP could be hit in the same bar.
                    # Resolve by checking which level the open is closer to:
                    # - Distance from open down to SL vs distance from open up to TP
                    # The shorter distance was likely hit first.
                    dist_to_sl = opens[i] - stop_loss
                    dist_to_tp = take_profit - opens[i]
                    if dist_to_sl <= dist_to_tp:
                        # SL was closer to open, hit first
                        exit_price = stop_loss - config.slippage_points
                        pnl_points = exit_price - entry_price
                        exit_reason = "sl"
                    else:
                        # TP was closer to open, hit first
                        exit_price = take_profit
                        pnl_points = exit_price - entry_price
                        exit_reason = "tp"
                    pnl_dollar = pnl_points * config.contract_size * config.lot_size - config.commission_per_lot_rt
                    equity += pnl_dollar
                    trades.append(Trade(
                        entry_bar=entry_bar, exit_bar=i, direction=1,
                        entry_price=entry_price, exit_price=exit_price,
                        stop_loss=stop_loss, take_profit=take_profit,
                        pnl=pnl_dollar, pnl_points=pnl_points,
                        commission=config.commission_per_lot_rt, exit_reason=exit_reason
                    ))
                    position = 0
                elif sl_hit:
                    # Stop loss hit (price went below SL)
                    exit_price = stop_loss - config.slippage_points
                    pnl_points = exit_price - entry_price
                    pnl_dollar = pnl_points * config.contract_size * config.lot_size - config.commission_per_lot_rt
                    equity += pnl_dollar
                    trades.append(Trade(
                        entry_bar=entry_bar, exit_bar=i, direction=1,
                        entry_price=entry_price, exit_price=exit_price,
                        stop_loss=stop_loss, take_profit=take_profit,
                        pnl=pnl_dollar, pnl_points=pnl_points,
                        commission=config.commission_per_lot_rt, exit_reason="sl"
                    ))
                    position = 0
                elif tp_hit:
                    # Take profit hit (price went above TP)
                    exit_price = take_profit
                    pnl_points = exit_price - entry_price
                    pnl_dollar = pnl_points * config.contract_size * config.lot_size - config.commission_per_lot_rt
                    equity += pnl_dollar
                    trades.append(Trade(
                        entry_bar=entry_bar, exit_bar=i, direction=1,
                        entry_price=entry_price, exit_price=exit_price,
                        stop_loss=stop_loss, take_profit=take_profit,
                        pnl=pnl_dollar, pnl_points=pnl_points,
                        commission=config.commission_per_lot_rt, exit_reason="tp"
                    ))
                    position = 0
                # Opposite signal - exit at close
                elif signals[i, 0] == -1:
                    exit_price = closes[i] - spreads[i] / 2 - config.slippage_points
                    pnl_points = exit_price - entry_price
                    pnl_dollar = pnl_points * config.contract_size * config.lot_size - config.commission_per_lot_rt
                    equity += pnl_dollar
                    trades.append(Trade(
                        entry_bar=entry_bar, exit_bar=i, direction=1,
                        entry_price=entry_price, exit_price=exit_price,
                        stop_loss=stop_loss, take_profit=take_profit,
                        pnl=pnl_dollar, pnl_points=pnl_points,
                        commission=config.commission_per_lot_rt, exit_reason="signal"
                    ))
                    position = 0

            elif position == -1:  # Short position
                sl_hit = highs[i] >= stop_loss
                tp_hit = lows[i] <= take_profit

                if sl_hit and tp_hit:
                    # Both SL and TP could be hit in the same bar.
                    # Resolve by checking which level the open is closer to:
                    # - Distance from open up to SL vs distance from open down to TP
                    dist_to_sl = stop_loss - opens[i]
                    dist_to_tp = opens[i] - take_profit
                    if dist_to_sl <= dist_to_tp:
                        # SL was closer to open, hit first
                        exit_price = stop_loss + config.slippage_points
                        pnl_points = entry_price - exit_price
                        exit_reason = "sl"
                    else:
                        # TP was closer to open, hit first
                        exit_price = take_profit
                        pnl_points = entry_price - exit_price
                        exit_reason = "tp"
                    pnl_dollar = pnl_points * config.contract_size * config.lot_size - config.commission_per_lot_rt
                    equity += pnl_dollar
                    trades.append(Trade(
                        entry_bar=entry_bar, exit_bar=i, direction=-1,
                        entry_price=entry_price, exit_price=exit_price,
                        stop_loss=stop_loss, take_profit=take_profit,
                        pnl=pnl_dollar, pnl_points=pnl_points,
                        commission=config.commission_per_lot_rt, exit_reason=exit_reason
                    ))
                    position = 0
                elif sl_hit:
                    # Stop loss hit (price went above SL)
                    exit_price = stop_loss + config.slippage_points
                    pnl_points = entry_price - exit_price
                    pnl_dollar = pnl_points * config.contract_size * config.lot_size - config.commission_per_lot_rt
                    equity += pnl_dollar
                    trades.append(Trade(
                        entry_bar=entry_bar, exit_bar=i, direction=-1,
                        entry_price=entry_price, exit_price=exit_price,
                        stop_loss=stop_loss, take_profit=take_profit,
                        pnl=pnl_dollar, pnl_points=pnl_points,
                        commission=config.commission_per_lot_rt, exit_reason="sl"
                    ))
                    position = 0
                elif tp_hit:
                    # Take profit hit (price went below TP)
                    exit_price = take_profit
                    pnl_points = entry_price - exit_price
                    pnl_dollar = pnl_points * config.contract_size * config.lot_size - config.commission_per_lot_rt
                    equity += pnl_dollar
                    trades.append(Trade(
                        entry_bar=entry_bar, exit_bar=i, direction=-1,
                        entry_price=entry_price, exit_price=exit_price,
                        stop_loss=stop_loss, take_profit=take_profit,
                        pnl=pnl_dollar, pnl_points=pnl_points,
                        commission=config.commission_per_lot_rt, exit_reason="tp"
                    ))
                    position = 0
                # Opposite signal - exit at close
                elif signals[i, 0] == 1:
                    exit_price = closes[i] + spreads[i] / 2 + config.slippage_points
                    pnl_points = entry_price - exit_price
                    pnl_dollar = pnl_points * config.contract_size * config.lot_size - config.commission_per_lot_rt
                    equity += pnl_dollar
                    trades.append(Trade(
                        entry_bar=entry_bar, exit_bar=i, direction=-1,
                        entry_price=entry_price, exit_price=exit_price,
                        stop_loss=stop_loss, take_profit=take_profit,
                        pnl=pnl_dollar, pnl_points=pnl_points,
                        commission=config.commission_per_lot_rt, exit_reason="signal"
                    ))
                    position = 0

        # If flat, check for new entry signal
        if position == 0 and signals[i, 0] != 0:
            direction = int(signals[i, 0])
            sl_dist = signals[i, 1]
            tp_dist = signals[i, 2]

            # Margin check: can we afford this position?
            # Margin required = (price * contract_size * lot_size) / leverage
            margin_required = (closes[i] * config.contract_size * config.lot_size) / config.leverage
            if equity < margin_required * 1.2:  # Need 120% of margin as buffer
                equity_curve[i] = equity
                continue

            if direction == 1:  # Buy
                entry_price = closes[i] + spreads[i] / 2 + config.slippage_points  # Enter at ask + slippage
                stop_loss = entry_price - sl_dist
                take_profit = entry_price + tp_dist
            else:  # Sell
                entry_price = closes[i] - spreads[i] / 2 - config.slippage_points  # Enter at bid - slippage
                stop_loss = entry_price + sl_dist
                take_profit = entry_price - tp_dist

            position = direction
            entry_bar = i

        equity_curve[i] = equity

    # Close any remaining position at end
    if position != 0:
        if position == 1:
            exit_price = closes[-1] - spreads[-1] / 2 - config.slippage_points
            pnl_points = exit_price - entry_price
        else:
            exit_price = closes[-1] + spreads[-1] / 2 + config.slippage_points
            pnl_points = entry_price - exit_price
        pnl_dollar = pnl_points * config.contract_size * config.lot_size - config.commission_per_lot_rt
        equity += pnl_dollar
        trades.append(Trade(
            entry_bar=entry_bar, exit_bar=n_bars - 1, direction=position,
            entry_price=entry_price, exit_price=exit_price,
            stop_loss=stop_loss, take_profit=take_profit,
            pnl=pnl_dollar, pnl_points=pnl_points,
            commission=config.commission_per_lot_rt, exit_reason="eod"
        ))
        equity_curve[-1] = equity

    # Compute metrics
    result = _compute_metrics(trades, equity_curve, config)
    return result


def _compute_metrics(trades: list, equity_curve: np.ndarray, config: BacktestConfig) -> BacktestResult:
    """Compute comprehensive performance metrics from trades and equity curve."""
    result = BacktestResult()
    result.trades = trades
    result.equity_curve = equity_curve
    result.total_trades = len(trades)

    if result.total_trades == 0:
        return result

    pnls = np.array([t.pnl for t in trades])
    result.total_pnl = float(np.sum(pnls))
    result.winning_trades = int(np.sum(pnls > 0))
    result.losing_trades = int(np.sum(pnls < 0))
    result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0.0

    # Profit factor
    gross_profit = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 0.001
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average trade metrics
    result.avg_trade_pnl = float(np.mean(pnls))
    result.avg_winner = float(np.mean(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
    result.avg_loser = float(np.mean(pnls[pnls < 0])) if np.any(pnls < 0) else 0.0

    # Max consecutive losses
    losses_streak = 0
    max_streak = 0
    for p in pnls:
        if p < 0:
            losses_streak += 1
            max_streak = max(max_streak, losses_streak)
        else:
            losses_streak = 0
    result.max_consecutive_losses = max_streak

    # Drawdown from equity curve
    valid_equity = equity_curve[equity_curve > 0]
    if len(valid_equity) > 0:
        peak = np.maximum.accumulate(valid_equity)
        drawdown = valid_equity - peak
        result.max_drawdown = float(np.min(drawdown))
        result.max_drawdown_pct = float(np.min(drawdown / peak)) * 100 if np.any(peak > 0) else 0.0

    # Sharpe ratio (annualized, assuming ~390 trading minutes per day, ~21 days/month)
    # Use bar-level returns of equity curve
    equity_returns = np.diff(valid_equity) / valid_equity[:-1] if len(valid_equity) > 1 else np.array([])
    if len(equity_returns) > 1 and np.std(equity_returns) > 0:
        bars_per_year = 252 * 390  # ~98,280 1-min bars per year
        result.sharpe_ratio = float(np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(bars_per_year))

        # Sortino ratio (downside deviation)
        downside = equity_returns[equity_returns < 0]
        if len(downside) > 0:
            downside_std = np.std(downside)
            if downside_std > 0:
                result.sortino_ratio = float(np.mean(equity_returns) / downside_std * np.sqrt(bars_per_year))

    # Average trade duration in bars
    durations = [t.exit_bar - t.entry_bar for t in trades]
    result.avg_trade_duration = float(np.mean(durations)) if durations else 0.0

    # Calmar ratio
    if result.max_drawdown_pct < 0:
        total_return_pct = (result.total_pnl / config.initial_equity) * 100
        result.calmar_ratio = abs(total_return_pct / result.max_drawdown_pct)

    # Expectancy
    result.expectancy = result.avg_trade_pnl

    return result
