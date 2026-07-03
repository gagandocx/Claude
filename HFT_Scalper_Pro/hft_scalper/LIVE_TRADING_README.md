# Live Trading Bot - Zero-Discrepancy from Backtest

This live trading bot uses the **EXACT same code** from `run_aggressive_backtest.py` to trade on MetaTrader 5. There is zero translation error between the backtest and live execution.

## Strategy Overview

**Two-Mode Adaptive Compounding Scalper on XAUUSD (1-minute bars)**

- **GROW mode** (near equity peak): risk_grow = 0.17 for aggressive compounding
- **PROTECT mode** (in drawdown): risk_protect = 0.025 for capital preservation
- **Transition**: Exponential scaling via `(equity/peak)^dd_power` with dd_power=13
- **Signals**: Dual RSI (periods 8 and 14) mean-reversion + 4-bar reversal pattern
- **Position Management**: Up to 2 simultaneous positions with independent SL/TP

### Backtest Results (on historical tick data)

| Metric | Value |
|--------|-------|
| Initial Equity | $1,000 |
| Final Equity | $20,110 |
| Total Return | 1,911% |
| Max Drawdown | -14.86% |
| Total Trades | 847 |
| Win Rate | 56.43% |
| Profit Factor | 2.31 |

## Prerequisites

1. **Windows OS** (MetaTrader5 Python package only works on Windows)
2. **MetaTrader 5 Terminal** installed and running, logged into a broker account
3. **Python 3.8+** (64-bit recommended)
4. **Python packages**:
   ```
   pip install MetaTrader5 numpy
   ```

## Installation

1. Ensure MT5 terminal is running and logged into your broker account
2. Enable "Algo Trading" in MT5 (Tools > Options > Expert Advisors > Allow Algo Trading)
3. Ensure XAUUSD (or your chosen symbol) is visible in Market Watch

## Usage

### Basic usage (default XAUUSD with best parameters):
```bash
python live_trader.py --symbol XAUUSD --magic 202401
```

### With custom config file:
```bash
python live_trader.py --config path/to/config.json --magic 100
```

### With custom log directory:
```bash
python live_trader.py --symbol XAUUSD --magic 202401 --log-dir C:\trading\logs
```

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--symbol` | XAUUSD | Trading symbol |
| `--magic` | 202401 | Magic number for order identification |
| `--config` | None | Path to JSON config file (overrides defaults) |
| `--log-dir` | Script directory | Directory for log and CSV files |

## Configuration

The bot uses the exact parameters from `aggressive_results.json` by default:

```json
{
    "rsi_entry": 25,
    "sl_mult": 2.0,
    "tp_mult": 3.0,
    "risk_grow": 0.17,
    "risk_protect": 0.025,
    "dd_power": 13,
    "cooldown": 3,
    "max_positions": 2,
    "use_4bar": true,
    "session_start": 7,
    "session_end": 20,
    "dd_halt": 0.149,
    "streak_n": 3,
    "streak_mult": 1.3,
    "max_risk_cap": 0.25
}
```

You can override any parameter by passing a JSON config file with `--config`.

## How It Maps to the Backtest

The live trader guarantees zero discrepancy from the backtest by:

1. **`compute_rsi()`** - Copied character-for-character from `run_aggressive_backtest.py`. Uses Wilder smoothing with the exact same loop structure.

2. **`compute_atr()`** - Copied character-for-character. True Range calculation with Wilder smoothing.

3. **Signal Generation** - Exact same conditional ordering:
   - Primary: RSI(8) < 25 -> buy, RSI(8) > 75 -> sell
   - Secondary: RSI(14) < 30 -> buy, RSI(14) > 70 -> sell
   - Tertiary: 4-bar reversal pattern (all 4 bars down -> buy, all up -> sell)

4. **Position Sizing** - Exact same two-mode formula:
   ```
   eq_ratio = equity / peak_equity
   dd_scale = eq_ratio ^ dd_power
   risk = risk_protect + (risk_grow - risk_protect) * dd_scale
   ```
   With streak boost: if consec_wins >= streak_n and dd_scale > 0.8, risk *= streak_mult

5. **Lot Calculation** - Same formula: `lot = (equity * risk) / (sl_dist * CONTRACT_SIZE)`

6. **Filters** - Same session hours (07-20 UTC), ATR minimum (0.5), DD halt (14.9%), cooldown between entries

## Output Files

- **`live_trader.log`** - Detailed log of all signals, orders, fills, and errors
- **`live_trades.csv`** - CSV trade log with columns: timestamp, symbol, direction, lot_size, entry_price, sl, tp, magic, signal_type, rsi_fast, rsi_slow, atr, equity, peak_equity, dd_scale, risk_pct, bar_count

## Graceful Shutdown

Press `Ctrl+C` or send SIGTERM to stop the bot gracefully. Open positions will remain with their SL/TP intact (they will be managed by the broker's server-side stops).

## Auto-Reconnection

If the MT5 connection drops, the bot will automatically retry with exponential backoff:
- Base delay: 5 seconds
- Max delay: 300 seconds (5 minutes)
- Max retries: 10

## Risk Warnings

- **This is a live trading system that risks real money.**
- Past backtest performance does not guarantee future results.
- Market conditions change; the strategy may not perform as backtested.
- Always start with a demo account to verify behavior matches expectations.
- Monitor the bot closely during initial live deployment.
- The DD halt (14.9%) will pause trading if drawdown exceeds the threshold.
- Ensure your broker supports the required order types and has reasonable spreads on XAUUSD.
- Slippage in live markets may differ from the 0.15pt assumption in the backtest.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "MetaTrader5 package not found" | Run `pip install MetaTrader5` (Windows only) |
| "Symbol not found" | Add the symbol to Market Watch in MT5 |
| Connection failures | Ensure MT5 terminal is running and logged in |
| Orders rejected | Check if Algo Trading is enabled in MT5 settings |
| "Insufficient funds" | Reduce initial risk or increase account balance |
