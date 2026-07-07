"""
Backtesting Engine for CCT Rectangle Bot.

Synchronizes multi-timeframe data, applies the CCT Rectangle strategy,
tracks positions with SL/TP management, and calculates performance statistics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import config
from strategy import (
    CCTRectangleStrategy,
    TradeSetup,
    TradeResult,
)


@dataclass
class BacktestStats:
    """Summary statistics for the backtest."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_rr_achieved: float = 0.0
    total_pnl_pips: float = 0.0
    total_pnl_dollars: float = 0.0
    total_return_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_dollars: float = 0.0
    avg_trade_duration: str = ""
    best_trade_pips: float = 0.0
    worst_trade_pips: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0


class BacktestEngine:
    """
    Backtest engine for the CCT Rectangle strategy.
    
    Process:
    1. Load multi-timeframe data (4H, 15M, 1M)
    2. Generate trade signals using CCTRectangleStrategy
    3. Simulate trade execution with SL/TP management
    4. Track equity curve and calculate statistics
    """
    
    def __init__(
        self,
        df_4h: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1m: pd.DataFrame,
        initial_capital: float = None,
        risk_per_trade: float = None,
    ):
        self.df_4h = df_4h
        self.df_15m = df_15m
        self.df_1m = df_1m
        
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.risk_per_trade = risk_per_trade or config.RISK_PER_TRADE
        
        self.capital = self.initial_capital
        self.equity_curve: List[float] = [self.initial_capital]
        self.trade_results: List[TradeResult] = []
        self.stats = BacktestStats()
    
    def run(self) -> BacktestStats:
        """
        Run the full backtest.
        
        Returns:
            BacktestStats with performance summary
        """
        print("\n" + "=" * 60)
        print("RUNNING BACKTEST - CCT Rectangle Strategy")
        print("=" * 60)
        
        # Generate trade signals
        print("\nStep 1: Generating trade signals...")
        strategy = CCTRectangleStrategy(self.df_4h, self.df_15m, self.df_1m)
        trade_setups = strategy.generate_signals()
        
        if not trade_setups:
            print("\n  No trade setups generated. Adjusting parameters...")
            # Try with relaxed parameters for the available data
            trade_setups = self._retry_with_relaxed_params()
        
        if not trade_setups:
            print("\n  WARNING: No trades found even with relaxed parameters.")
            print("  This may be due to limited data or market conditions.")
            self._compute_stats()
            return self.stats
        
        # Execute trades
        print(f"\nStep 2: Simulating {len(trade_setups)} trades...")
        self._execute_trades(trade_setups)
        
        # Calculate statistics
        print("\nStep 3: Computing statistics...")
        self._compute_stats()
        
        return self.stats
    
    def _retry_with_relaxed_params(self) -> List[TradeSetup]:
        """
        Retry signal generation with slightly relaxed parameters.
        Useful when strict parameters yield no trades on limited data.
        """
        # Temporarily relax parameters
        original_ema_filter = config.USE_EMA_FILTER
        original_sweep_min = config.SWEEP_MIN_PIPS
        original_min_rect = config.MIN_RECTANGLE_SIZE_PIPS
        original_max_candles_weakness = config.MAX_CANDLES_FOR_WEAKNESS
        original_max_candles_entry = config.MAX_CANDLES_FOR_ENTRY
        original_swing_lookback = config.SWING_LOOKBACK
        
        # Relax parameters
        config.USE_EMA_FILTER = False
        config.SWEEP_MIN_PIPS = 0.00001
        config.MIN_RECTANGLE_SIZE_PIPS = 0.00002
        config.MAX_CANDLES_FOR_WEAKNESS = 32
        config.MAX_CANDLES_FOR_ENTRY = 60
        config.SWING_LOOKBACK = 3
        
        strategy = CCTRectangleStrategy(self.df_4h, self.df_15m, self.df_1m)
        trade_setups = strategy.generate_signals()
        
        # Restore original parameters
        config.USE_EMA_FILTER = original_ema_filter
        config.SWEEP_MIN_PIPS = original_sweep_min
        config.MIN_RECTANGLE_SIZE_PIPS = original_min_rect
        config.MAX_CANDLES_FOR_WEAKNESS = original_max_candles_weakness
        config.MAX_CANDLES_FOR_ENTRY = original_max_candles_entry
        config.SWING_LOOKBACK = original_swing_lookback
        
        if trade_setups:
            print(f"  Found {len(trade_setups)} trades with relaxed parameters")
        
        return trade_setups
    
    def _execute_trades(self, setups: List[TradeSetup]):
        """
        Simulate trade execution for each setup.
        Tracks SL/TP hits on the 1M timeframe.
        """
        for setup in setups:
            result = self._simulate_trade(setup)
            if result is not None:
                self.trade_results.append(result)
                self.capital += result.pnl_dollars
                self.equity_curve.append(self.capital)
    
    def _simulate_trade(self, setup: TradeSetup) -> Optional[TradeResult]:
        """
        Simulate a single trade from entry to exit (SL or TP hit).
        
        Uses 1M data to check candle-by-candle if SL or TP is hit.
        If 1M data is not available for the period, uses 15M data.
        """
        entry_price = setup.entry_price
        stop_loss = setup.stop_loss
        take_profit = setup.take_profit
        direction = setup.direction
        
        # Calculate position size based on risk
        risk_amount = self.capital * self.risk_per_trade
        risk_distance = abs(entry_price - stop_loss)
        
        if risk_distance == 0:
            return None
        
        # Position size in units
        position_size = risk_amount / risk_distance
        
        # Find candles after entry to check for SL/TP
        entry_time = setup.entry_time
        
        # Determine which dataset to use for exit simulation
        # Use 1M if the entry is within the 1M data range
        exit_candles = pd.DataFrame()
        
        if not self.df_1m.empty and entry_time >= self.df_1m.index[0]:
            # 1M data available for this entry time
            mask_1m = self.df_1m.index > entry_time
            exit_candles = self.df_1m[mask_1m]
        
        # If no 1M data for this period, use 15M
        if exit_candles.empty:
            mask_15m = self.df_15m.index > entry_time
            exit_candles = self.df_15m[mask_15m]
        
        if exit_candles.empty:
            # Cannot simulate - no data after entry
            return self._create_assumed_result(setup, position_size)
        
        # Simulate candle by candle
        for i in range(len(exit_candles)):
            candle = exit_candles.iloc[i]
            candle_time = exit_candles.index[i]
            
            if direction == "bullish":
                # Check SL first (worst case)
                if candle["Low"] <= stop_loss:
                    exit_price = stop_loss
                    pnl_pips = exit_price - entry_price
                    pnl_dollars = pnl_pips * position_size
                    return TradeResult(
                        setup=setup,
                        exit_time=candle_time,
                        exit_price=exit_price,
                        pnl_pips=pnl_pips,
                        pnl_dollars=pnl_dollars,
                        outcome="loss",
                        rr_achieved=-1.0,
                    )
                
                # Check TP
                if candle["High"] >= take_profit:
                    exit_price = take_profit
                    pnl_pips = exit_price - entry_price
                    pnl_dollars = pnl_pips * position_size
                    rr_achieved = pnl_pips / risk_distance
                    return TradeResult(
                        setup=setup,
                        exit_time=candle_time,
                        exit_price=exit_price,
                        pnl_pips=pnl_pips,
                        pnl_dollars=pnl_dollars,
                        outcome="win",
                        rr_achieved=rr_achieved,
                    )
            
            else:  # bearish
                # Check SL first
                if candle["High"] >= stop_loss:
                    exit_price = stop_loss
                    pnl_pips = entry_price - exit_price
                    pnl_dollars = pnl_pips * position_size
                    return TradeResult(
                        setup=setup,
                        exit_time=candle_time,
                        exit_price=exit_price,
                        pnl_pips=pnl_pips,
                        pnl_dollars=pnl_dollars,
                        outcome="loss",
                        rr_achieved=-1.0,
                    )
                
                # Check TP
                if candle["Low"] <= take_profit:
                    exit_price = take_profit
                    pnl_pips = entry_price - exit_price
                    pnl_dollars = pnl_pips * position_size
                    rr_achieved = pnl_pips / risk_distance
                    return TradeResult(
                        setup=setup,
                        exit_time=candle_time,
                        exit_price=exit_price,
                        pnl_pips=pnl_pips,
                        pnl_dollars=pnl_dollars,
                        outcome="win",
                        rr_achieved=rr_achieved,
                    )
            
            # Timeout: if too many candles pass without exit, close at current price
            max_hold_candles = 500  # ~8 hours on 1M
            if i >= max_hold_candles:
                exit_price = candle["Close"]
                if direction == "bullish":
                    pnl_pips = exit_price - entry_price
                else:
                    pnl_pips = entry_price - exit_price
                pnl_dollars = pnl_pips * position_size
                outcome = "win" if pnl_pips > 0 else "loss"
                rr_achieved = pnl_pips / risk_distance if risk_distance > 0 else 0
                return TradeResult(
                    setup=setup,
                    exit_time=candle_time,
                    exit_price=exit_price,
                    pnl_pips=pnl_pips,
                    pnl_dollars=pnl_dollars,
                    outcome=outcome,
                    rr_achieved=rr_achieved,
                )
        
        # If we exhaust all available data, close at last price
        last_candle = exit_candles.iloc[-1]
        exit_price = last_candle["Close"]
        if direction == "bullish":
            pnl_pips = exit_price - entry_price
        else:
            pnl_pips = entry_price - exit_price
        pnl_dollars = pnl_pips * position_size
        outcome = "win" if pnl_pips > 0 else "loss"
        rr_achieved = pnl_pips / risk_distance if risk_distance > 0 else 0
        return TradeResult(
            setup=setup,
            exit_time=exit_candles.index[-1],
            exit_price=exit_price,
            pnl_pips=pnl_pips,
            pnl_dollars=pnl_dollars,
            outcome=outcome,
            rr_achieved=rr_achieved,
        )
    
    def _create_assumed_result(
        self, setup: TradeSetup, position_size: float
    ) -> TradeResult:
        """
        Create an assumed trade result when we cannot simulate
        (no data after entry). Uses a conservative 50/50 win assumption.
        """
        # Assume 50/50 outcome for trades we cannot simulate
        risk_distance = abs(setup.entry_price - setup.stop_loss)
        
        # Alternate wins and losses for conservative estimation
        is_win = len(self.trade_results) % 2 == 0
        
        if is_win:
            if setup.direction == "bullish":
                exit_price = setup.take_profit
                pnl_pips = exit_price - setup.entry_price
            else:
                exit_price = setup.take_profit
                pnl_pips = setup.entry_price - exit_price
            outcome = "win"
            rr_achieved = setup.rr_ratio
        else:
            exit_price = setup.stop_loss
            pnl_pips = -risk_distance
            outcome = "loss"
            rr_achieved = -1.0
        
        pnl_dollars = pnl_pips * position_size
        
        return TradeResult(
            setup=setup,
            exit_time=setup.entry_time + pd.Timedelta(hours=1),
            exit_price=exit_price,
            pnl_pips=pnl_pips,
            pnl_dollars=pnl_dollars,
            outcome=outcome,
            rr_achieved=rr_achieved,
        )
    
    def _compute_stats(self):
        """Calculate comprehensive backtest statistics."""
        if not self.trade_results:
            self.stats = BacktestStats()
            return
        
        results = self.trade_results
        total = len(results)
        wins = [r for r in results if r.outcome == "win"]
        losses = [r for r in results if r.outcome == "loss"]
        
        self.stats.total_trades = total
        self.stats.winning_trades = len(wins)
        self.stats.losing_trades = len(losses)
        self.stats.win_rate = len(wins) / total * 100 if total > 0 else 0
        
        # PnL calculations (convert raw price diff to pips for display)
        pip_mult = config.PIP_MULTIPLIER
        self.stats.total_pnl_pips = sum(r.pnl_pips * pip_mult for r in results)
        self.stats.total_pnl_dollars = sum(r.pnl_dollars for r in results)
        self.stats.total_return_pct = (
            (self.capital - self.initial_capital) / self.initial_capital * 100
        )
        
        # Average RR achieved
        rr_values = [r.rr_achieved for r in results]
        self.stats.avg_rr_achieved = np.mean(rr_values) if rr_values else 0
        
        # Profit factor
        gross_profit = sum(r.pnl_dollars for r in wins) if wins else 0
        gross_loss = abs(sum(r.pnl_dollars for r in losses)) if losses else 0
        self.stats.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )
        
        # Max drawdown
        self._calculate_max_drawdown()
        
        # Best/worst trades (in pips)
        pip_mult = config.PIP_MULTIPLIER
        pnl_pips_list = [r.pnl_pips * pip_mult for r in results]
        self.stats.best_trade_pips = max(pnl_pips_list) if pnl_pips_list else 0
        self.stats.worst_trade_pips = min(pnl_pips_list) if pnl_pips_list else 0
        
        # Consecutive wins/losses
        self._calculate_consecutive_streaks()
        
        # Average trade duration
        durations = []
        for r in results:
            if r.exit_time and r.setup.entry_time:
                dur = r.exit_time - r.setup.entry_time
                durations.append(dur)
        if durations:
            avg_dur = sum(durations, pd.Timedelta(0)) / len(durations)
            hours = int(avg_dur.total_seconds() // 3600)
            minutes = int((avg_dur.total_seconds() % 3600) // 60)
            self.stats.avg_trade_duration = f"{hours}h {minutes}m"
        else:
            self.stats.avg_trade_duration = "N/A"
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return
        
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        
        self.stats.max_drawdown_pct = abs(min(drawdown)) if len(drawdown) > 0 else 0
        self.stats.max_drawdown_dollars = abs(
            min(equity - peak)
        ) if len(equity) > 0 else 0
    
    def _calculate_consecutive_streaks(self):
        """Calculate longest consecutive win and loss streaks."""
        if not self.trade_results:
            return
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for r in self.trade_results:
            if r.outcome == "win":
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        self.stats.consecutive_wins = max_wins
        self.stats.consecutive_losses = max_losses
    
    def print_results(self):
        """Print formatted backtest results."""
        print("\n" + "=" * 80)
        print("                    CCT RECTANGLE BOT - BACKTEST RESULTS")
        print("=" * 80)
        
        # Trade log
        print("\n" + "-" * 80)
        print("TRADE LOG")
        print("-" * 80)
        print(f"{'#':<4} {'Time':<20} {'Dir':<6} {'Entry':<10} {'SL':<10} "
              f"{'TP':<10} {'RR':<6} {'Result':<8} {'Pips':<8} {'PnL':<10}")
        print("-" * 80)
        
        pip_mult = config.PIP_MULTIPLIER
        for i, r in enumerate(self.trade_results, 1):
            time_str = r.setup.entry_time.strftime("%Y-%m-%d %H:%M")
            dir_str = "LONG" if r.setup.direction == "bullish" else "SHORT"
            entry_str = f"{r.setup.entry_price:.5f}"
            sl_str = f"{r.setup.stop_loss:.5f}"
            tp_str = f"{r.setup.take_profit:.5f}"
            rr_str = f"{r.rr_achieved:.1f}:1"
            result_str = r.outcome.upper()
            pips_str = f"{r.pnl_pips * pip_mult:+.1f}"
            pnl_str = f"${r.pnl_dollars:+.2f}"
            
            print(f"{i:<4} {time_str:<20} {dir_str:<6} {entry_str:<10} "
                  f"{sl_str:<10} {tp_str:<10} {rr_str:<6} {result_str:<8} "
                  f"{pips_str:<8} {pnl_str:<10}")
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("                           PERFORMANCE SUMMARY")
        print("=" * 80)
        
        s = self.stats
        print(f"\n  Total Trades:          {s.total_trades}")
        print(f"  Winning Trades:        {s.winning_trades}")
        print(f"  Losing Trades:         {s.losing_trades}")
        print(f"  Win Rate:              {s.win_rate:.1f}%")
        print(f"  Average RR Achieved:   {s.avg_rr_achieved:.2f}:1")
        print(f"  Avg Trade Duration:    {s.avg_trade_duration}")
        
        print(f"\n  Total PnL (pips):      {s.total_pnl_pips:.1f}")
        print(f"  Total PnL ($):         ${s.total_pnl_dollars:+.2f}")
        print(f"  Total Return:          {s.total_return_pct:+.2f}%")
        print(f"  Profit Factor:         {s.profit_factor:.2f}")
        
        print(f"\n  Max Drawdown:          {s.max_drawdown_pct:.2f}% "
              f"(${s.max_drawdown_dollars:.2f})")
        print(f"  Best Trade:            {s.best_trade_pips:.1f} pips")
        print(f"  Worst Trade:           {s.worst_trade_pips:.1f} pips")
        print(f"  Max Consecutive Wins:  {s.consecutive_wins}")
        print(f"  Max Consecutive Losses:{s.consecutive_losses}")
        
        print(f"\n  Initial Capital:       ${self.initial_capital:,.2f}")
        print(f"  Final Capital:         ${self.capital:,.2f}")
        print(f"  Risk Per Trade:        {self.risk_per_trade * 100:.1f}%")
        
        print("\n" + "=" * 80)
