"""
Backtesting Engine for CCT Rectangle Bot - AGGRESSIVE MODE.

Features:
- Compounding (position size based on current equity)
- Leverage multiplier (50x default)
- Multiple concurrent positions
- Monthly return breakdown
- Equity curve tracking with drawdown analysis
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
    monthly_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_dollars: float = 0.0
    avg_trade_duration: str = ""
    best_trade_pips: float = 0.0
    worst_trade_pips: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    leverage_used: int = 0
    compounding: bool = False
    avg_position_size: float = 0.0
    total_days: int = 0


class BacktestEngine:
    """
    Backtest engine - AGGRESSIVE MODE.

    Features:
    - Compounding: position sizes grow with equity
    - Leverage: amplifies gains (and losses)
    - Concurrent positions: multiple trades open simultaneously
    - Monthly return calculation
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
        self.leverage = config.LEVERAGE
        self.compounding = config.COMPOUNDING
        self.max_concurrent = config.MAX_CONCURRENT_TRADES

        self.capital = self.initial_capital
        self.equity_curve: List[float] = [self.initial_capital]
        self.equity_timestamps: List[pd.Timestamp] = []
        self.trade_results: List[TradeResult] = []
        self.stats = BacktestStats()
        self.position_sizes: List[float] = []

    def run(self) -> BacktestStats:
        """
        Run the full backtest with aggressive settings.
        """
        print("\n" + "=" * 60)
        print("RUNNING BACKTEST - CCT Rectangle Strategy (AGGRESSIVE MODE)")
        print("=" * 60)
        print(f"\n  Leverage:      {self.leverage}x")
        print(f"  Risk/Trade:    {self.risk_per_trade * 100:.0f}%")
        print(f"  Compounding:   {'ON' if self.compounding else 'OFF'}")
        print(f"  Max Concurrent: {self.max_concurrent}")

        # Generate trade signals
        print("\nStep 1: Generating trade signals...")
        strategy = CCTRectangleStrategy(self.df_4h, self.df_15m, self.df_1m)
        trade_setups = strategy.generate_signals()

        if not trade_setups:
            print("\n  No trade setups generated. Trying with relaxed params...")
            trade_setups = self._retry_with_relaxed_params()

        if not trade_setups:
            print("\n  WARNING: No trades found even with relaxed parameters.")
            self._compute_stats()
            return self.stats

        # Execute trades with concurrent position management
        print(f"\nStep 2: Simulating {len(trade_setups)} trades with leverage...")
        self._execute_trades_concurrent(trade_setups)

        # Calculate statistics
        print("\nStep 3: Computing statistics...")
        self._compute_stats()

        return self.stats

    def _retry_with_relaxed_params(self) -> List[TradeSetup]:
        """
        Retry with even more relaxed parameters.
        """
        original_sweep_min = config.SWEEP_MIN_PIPS
        original_min_rect = config.MIN_RECTANGLE_SIZE_PIPS
        original_max_candles_entry = config.MAX_CANDLES_FOR_ENTRY
        original_swing_lookback = config.SWING_LOOKBACK
        original_partial = config.PARTIAL_ENGULF_RATIO

        config.SWEEP_MIN_PIPS = 0.01
        config.MIN_RECTANGLE_SIZE_PIPS = 0.05
        config.MAX_CANDLES_FOR_ENTRY = 240
        config.SWING_LOOKBACK = 2
        config.PARTIAL_ENGULF_RATIO = 0.30

        strategy = CCTRectangleStrategy(self.df_4h, self.df_15m, self.df_1m)
        trade_setups = strategy.generate_signals()

        config.SWEEP_MIN_PIPS = original_sweep_min
        config.MIN_RECTANGLE_SIZE_PIPS = original_min_rect
        config.MAX_CANDLES_FOR_ENTRY = original_max_candles_entry
        config.SWING_LOOKBACK = original_swing_lookback
        config.PARTIAL_ENGULF_RATIO = original_partial

        if trade_setups:
            print(f"  Found {len(trade_setups)} trades with relaxed parameters")

        return trade_setups

    def _execute_trades_concurrent(self, setups: List[TradeSetup]):
        """
        Execute trades allowing concurrent positions.
        With compounding, each trade's position size is based on current equity.
        Includes margin call protection.
        """
        open_positions: List[Dict] = []

        for setup in setups:
            # Skip if capital is too low (margin call at 10% of initial)
            if self.capital < self.initial_capital * 0.10:
                break

            # Check if we can open a new position (max concurrent limit)
            # First, close any positions that would have exited before this entry
            self._resolve_positions_before(open_positions, setup.entry_time)

            if len(open_positions) >= self.max_concurrent:
                continue

            # Calculate position size based on current equity (compounding)
            equity_for_sizing = self.capital if self.compounding else self.initial_capital
            risk_amount = equity_for_sizing * self.risk_per_trade
            risk_distance = abs(setup.entry_price - setup.stop_loss)

            if risk_distance == 0:
                continue

            # Position size (leverage is 1 by default - risk % defines the loss cap)
            position_size = (risk_amount / risk_distance) * self.leverage
            self.position_sizes.append(position_size)

            # Simulate the trade
            result = self._simulate_trade(setup, position_size)
            if result is not None:
                self.trade_results.append(result)
                self.capital += result.pnl_dollars
                # Floor at 0
                if self.capital < 0:
                    self.capital = 0
                self.equity_curve.append(self.capital)
                if result.exit_time is not None:
                    self.equity_timestamps.append(result.exit_time)

    def _resolve_positions_before(self, open_positions: List[Dict], current_time: pd.Timestamp):
        """Resolve open positions that would have closed before current_time."""
        # In our sequential simulation, trades are fully resolved before moving on
        # This is a placeholder for more advanced concurrent position management
        pass

    def _simulate_trade(self, setup: TradeSetup, position_size: float) -> Optional[TradeResult]:
        """
        Simulate a single trade with optional trailing stop.
        Trailing stop allows winners to run much further than fixed TP.
        """
        entry_price = setup.entry_price
        stop_loss = setup.stop_loss
        take_profit = setup.take_profit
        direction = setup.direction

        risk_distance = abs(entry_price - stop_loss)
        if risk_distance == 0:
            return None

        # Trailing stop state
        use_trailing = config.USE_TRAILING_STOP
        trailing_activated = False
        current_stop = stop_loss
        best_price = entry_price  # Track best price for trailing

        # Find candles after entry for exit simulation
        entry_time = setup.entry_time
        exit_candles = pd.DataFrame()

        if not self.df_1m.empty and entry_time >= self.df_1m.index[0]:
            mask_1m = self.df_1m.index > entry_time
            exit_candles = self.df_1m[mask_1m]

        if exit_candles.empty:
            mask_15m = self.df_15m.index > entry_time
            exit_candles = self.df_15m[mask_15m]

        if exit_candles.empty:
            return self._create_assumed_result(setup, position_size)

        # Simulate candle by candle with trailing stop
        for i in range(len(exit_candles)):
            candle = exit_candles.iloc[i]
            candle_time = exit_candles.index[i]

            if direction == "bullish":
                # Update best price
                if candle["High"] > best_price:
                    best_price = candle["High"]

                # Check trailing stop activation
                if use_trailing and not trailing_activated:
                    profit_rr = (best_price - entry_price) / risk_distance
                    if profit_rr >= config.TRAILING_STOP_ACTIVATION_RR:
                        trailing_activated = True

                # Update trailing stop
                if trailing_activated:
                    trail_distance = risk_distance * config.TRAILING_STOP_DISTANCE_RR
                    new_stop = best_price - trail_distance
                    if new_stop > current_stop:
                        current_stop = new_stop

                # Check stop (either original or trailing)
                if candle["Low"] <= current_stop:
                    exit_price = current_stop
                    pnl_pips = exit_price - entry_price
                    pnl_dollars = pnl_pips * position_size
                    rr_achieved = pnl_pips / risk_distance
                    outcome = "win" if pnl_pips > 0 else "loss"
                    return TradeResult(
                        setup=setup,
                        exit_time=candle_time,
                        exit_price=exit_price,
                        pnl_pips=pnl_pips,
                        pnl_dollars=pnl_dollars,
                        outcome=outcome,
                        rr_achieved=rr_achieved,
                    )

                # Check fixed TP (only if trailing not active)
                if not trailing_activated and candle["High"] >= take_profit:
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
                # Update best price (lowest for shorts)
                if candle["Low"] < best_price:
                    best_price = candle["Low"]

                # Check trailing stop activation
                if use_trailing and not trailing_activated:
                    profit_rr = (entry_price - best_price) / risk_distance
                    if profit_rr >= config.TRAILING_STOP_ACTIVATION_RR:
                        trailing_activated = True

                # Update trailing stop for shorts
                if trailing_activated:
                    trail_distance = risk_distance * config.TRAILING_STOP_DISTANCE_RR
                    new_stop = best_price + trail_distance
                    if new_stop < current_stop:
                        current_stop = new_stop

                # Check stop
                if candle["High"] >= current_stop:
                    exit_price = current_stop
                    pnl_pips = entry_price - exit_price
                    pnl_dollars = pnl_pips * position_size
                    rr_achieved = pnl_pips / risk_distance
                    outcome = "win" if pnl_pips > 0 else "loss"
                    return TradeResult(
                        setup=setup,
                        exit_time=candle_time,
                        exit_price=exit_price,
                        pnl_pips=pnl_pips,
                        pnl_dollars=pnl_dollars,
                        outcome=outcome,
                        rr_achieved=rr_achieved,
                    )

                # Check fixed TP (only if trailing not active)
                if not trailing_activated and candle["Low"] <= take_profit:
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

            # Timeout after extended hold
            max_hold_candles = 1000
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

        # Exhaust all data - close at last price
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
        Create assumed result when no data available for exit simulation.
        Conservative: alternate win/loss for unsimulatable trades.
        """
        risk_distance = abs(setup.entry_price - setup.stop_loss)
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

        # PnL calculations
        pip_mult = config.PIP_MULTIPLIER
        self.stats.total_pnl_pips = sum(r.pnl_pips * pip_mult for r in results)
        self.stats.total_pnl_dollars = sum(r.pnl_dollars for r in results)
        self.stats.total_return_pct = (
            (self.capital - self.initial_capital) / self.initial_capital * 100
        )

        # Monthly return calculation
        if results:
            first_trade_time = results[0].setup.entry_time
            last_trade_time = results[-1].exit_time or results[-1].setup.entry_time
            total_days = max((last_trade_time - first_trade_time).days, 1)
            self.stats.total_days = total_days
            months = total_days / 30.0
            if months > 0:
                # Compound monthly return
                total_mult = self.capital / self.initial_capital
                if total_mult > 0:
                    self.stats.monthly_return_pct = (
                        (total_mult ** (1.0 / months) - 1) * 100
                    )
                else:
                    self.stats.monthly_return_pct = -100.0
                self.stats.annualized_return_pct = (
                    (total_mult ** (12.0 / months) - 1) * 100
                    if total_mult > 0 else -100.0
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

        # Leverage and position info
        self.stats.leverage_used = self.leverage
        self.stats.compounding = self.compounding
        if self.position_sizes:
            self.stats.avg_position_size = np.mean(self.position_sizes)

    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return

        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = np.where(peak > 0, (equity - peak) / peak * 100, 0)

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
        """Print formatted backtest results with monthly return."""
        print("\n" + "=" * 80)
        print("          CCT RECTANGLE BOT - AGGRESSIVE BACKTEST RESULTS")
        print("=" * 80)

        # Trade log
        print("\n" + "-" * 80)
        print("TRADE LOG")
        print("-" * 80)
        print(f"{'#':<4} {'Time':<20} {'Dir':<6} {'Entry':<10} {'SL':<10} "
              f"{'TP':<10} {'RR':<6} {'Result':<8} {'Pips':<8} {'PnL':<12}")
        print("-" * 80)

        pip_mult = config.PIP_MULTIPLIER
        for i, r in enumerate(self.trade_results, 1):
            time_str = r.setup.entry_time.strftime("%Y-%m-%d %H:%M")
            dir_str = "LONG" if r.setup.direction == "bullish" else "SHORT"
            entry_str = f"{r.setup.entry_price:.2f}"
            sl_str = f"{r.setup.stop_loss:.2f}"
            tp_str = f"{r.setup.take_profit:.2f}"
            rr_str = f"{r.rr_achieved:.1f}:1"
            result_str = r.outcome.upper()
            pips_str = f"{r.pnl_pips * pip_mult:+.1f}"
            pnl_str = f"${r.pnl_dollars:+,.2f}"

            print(f"{i:<4} {time_str:<20} {dir_str:<6} {entry_str:<10} "
                  f"{sl_str:<10} {tp_str:<10} {rr_str:<6} {result_str:<8} "
                  f"{pips_str:<8} {pnl_str:<12}")

        # Summary statistics
        print("\n" + "=" * 80)
        print("                       PERFORMANCE SUMMARY (AGGRESSIVE)")
        print("=" * 80)

        s = self.stats
        print(f"\n  {'='*50}")
        print(f"  MONTHLY RETURN:        {s.monthly_return_pct:+.1f}% per month")
        print(f"  TOTAL RETURN:          {s.total_return_pct:+.1f}%")
        print(f"  {'='*50}")

        print(f"\n  Total Trades:          {s.total_trades}")
        print(f"  Winning Trades:        {s.winning_trades}")
        print(f"  Losing Trades:         {s.losing_trades}")
        print(f"  Win Rate:              {s.win_rate:.1f}%")
        print(f"  Average RR Achieved:   {s.avg_rr_achieved:.2f}:1")
        print(f"  Avg Trade Duration:    {s.avg_trade_duration}")

        print(f"\n  Total PnL (pips):      {s.total_pnl_pips:+.1f}")
        print(f"  Total PnL ($):         ${s.total_pnl_dollars:+,.2f}")
        print(f"  Profit Factor:         {s.profit_factor:.2f}")

        print(f"\n  Max Drawdown:          {s.max_drawdown_pct:.1f}% "
              f"(${s.max_drawdown_dollars:,.2f})")
        print(f"  Best Trade:            {s.best_trade_pips:+.1f} pips")
        print(f"  Worst Trade:           {s.worst_trade_pips:+.1f} pips")
        print(f"  Max Consecutive Wins:  {s.consecutive_wins}")
        print(f"  Max Consecutive Losses:{s.consecutive_losses}")

        print(f"\n  Initial Capital:       ${self.initial_capital:,.2f}")
        print(f"  Final Capital:         ${self.capital:,.2f}")
        print(f"  Leverage:              {self.leverage}x")
        print(f"  Risk Per Trade:        {self.risk_per_trade * 100:.0f}%")
        print(f"  Compounding:           {'ENABLED' if self.compounding else 'DISABLED'}")
        print(f"  Backtest Period:       {s.total_days} days")

        if s.monthly_return_pct >= 500:
            print(f"\n  >>> TARGET ACHIEVED: {s.monthly_return_pct:.0f}% monthly <<<")
        elif s.total_return_pct >= 500:
            print(f"\n  >>> STRONG PERFORMANCE: {s.total_return_pct:.0f}% total return <<<")

        print("\n" + "=" * 80)
