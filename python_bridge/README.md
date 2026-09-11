# Python Bridge for MetaTrader 5

A **simple, rule-based** trade signal generator for gold (XAUUSD) scalping on
MetaTrader 5. The bridge decides direction from short-term price **momentum**,
gates entries with a **rules-derived confidence** and a set of transparent
filters, then writes signals to a CSV file that the `.mq5` Expert Advisors read.

There are no machine-learning models, checkpoints, or training steps. Instead of
being fed pre-trained checkpoint data, the system **backtests locally so it can
tweak itself**: the `AutoOptimizer` records every closed trade and gradually
shifts the strategy parameters toward the values that perform best.

## How a signal is made

```
+-------------------+
|   Market Data     |   yfinance (live) or exported broker CSV (backtest)
|   (OHLCV bars)    |
+--------+----------+
         |
         v
+-------------------------------------------------------------------+
|  1. Momentum direction                                            |
|     close[-1] - close[-lookback] vs a $ threshold -> BUY/SELL/FLAT |
|     (volume-weighted + adaptive lookback when ATR is elevated)     |
+-------------------------------------------------------------------+
         |
         v
+-------------------------------------------------------------------+
|  2. Rules-derived confidence (momentum magnitude)                 |
|     confidence = 0.2 + 0.7 * (1 - exp(-abs_diff / 1.5)), cap 0.95  |
|     abs_diff = |close[-1] - close[-(lookback+1)]|                  |
+-------------------------------------------------------------------+
         |
         v
+-------------------------------------------------------------------+
|  3. Filters (adjust confidence / gate the entry)                  |
|     - RSI zone (BUY 25-70, SELL 30-75)                             |
|     - Session detection (asian/london/newyork/overlap/off)         |
|     - Price structure (higher-highs / lower-lows), EMA, S/R, ATR   |
|     - Regime detection (trending / ranging / volatile / crash)     |
|     - min_confidence gate                                          |
+-------------------------------------------------------------------+
         |
         v
+-------------------------------------------------------------------+
|  4. Risk + exits                                                  |
|     - Kelly-criterion position sizing, drawdown / daily-loss halts |
|     - ATR-based stop loss; progressive trailing stop (no fixed TP) |
+-------------------------------------------------------------------+
         |
         v
+-------------------------------------------------------------------+
|              CSV File Bridge (MT5 Common Files)                    |
|   python_bridge_signal.csv  -->  Python_Bridge_EA.mq5             |
|   python_bridge_confirm.csv <--  Python_Bridge_EA.mq5             |
+-------------------------------------------------------------------+
```

Direction always comes from momentum. Confidence is a pure function of momentum
magnitude (mirroring `backtest.py`'s `compute_momentum_magnitude`) and is only
used to gate entry **timing**; it never overrides the momentum direction.

## Quick Start

### 1. Install Dependencies

```bash
cd python_bridge
pip install -r requirements.txt
```

### 2. Run the Bridge (live signal loop)

```bash
# Paper trading mode (default)
python main.py --paper

# Live mode
python main.py --live

# Custom loop interval in seconds
python main.py --interval 10
```

There is no training step. The bridge starts generating rule-based signals
immediately.

### 3. Set Up MT5

1. Copy `Python_Bridge_EA.mq5` to your MT5 `Experts` folder
2. Compile in MetaEditor
3. Attach to an XAUUSD chart
4. Ensure the `Common Files` path is accessible to both Python and MT5

## Local backtest self-tuning (AutoOptimizer)

The self-tuning loop is what replaces checkpoint training. Run a local backtest
over historical data and the `AutoOptimizer` will adjust the strategy toward the
parameter values that historically performed best.

```bash
# Backtest on an MT5-exported CSV (real broker data + variable spread)
python backtest.py --data-file /path/to/XAUUSD_M1.csv

# Or backtest on downloaded data (yfinance)
python backtest.py --days 5 --interval 1m --verbose
```

How the tuning works (`strategies/auto_optimizer.py`):

- Every closed trade is recorded with full context (session, entry confidence,
  momentum lookback, SL distance, RSI at entry, trailing tier, realized P/L).
- After every `optimize_frequency` trades it groups trades by parameter value,
  measures win rate / average P/L per value, and shifts each parameter a single
  small step toward the winning value.
- All parameters are **clamped to configured ranges** and never jump more than
  one step per cycle.
- If performance drops more than `rollback_threshold` (default 20%) after a
  cycle, it **rolls back** to the previous parameters.
- State is persisted to **`auto_optimizer_state.json`** (written next to
  `backtest.py`/`main.py` in the `python_bridge/` directory), so tuning survives
  restarts. The file is a runtime artifact and is gitignored.

Tunable parameters include `sl_distance`, per-session multipliers,
`min_confidence`, `momentum_lookback`, RSI overbought/oversold levels, trailing
distances, cooldown seconds, and max positions. At runtime `main.py` wires the
optimizer into the signal generator via `set_auto_optimizer`, so the live loop
picks up the tuned values automatically.

## Configuration

Edit `config/settings.py` to customize:

- **DataConfig**: symbol, momentum lookback, RSI levels
- **SignalConfig**: confidence thresholds, cooldown, ATR SL/TP multipliers
- **SessionConfig / SpreadFilterConfig / AdaptiveMomentumConfig**: session and
  volatility-aware behavior
- **PriceStructureConfig / FVGConfig / LiquiditySweepConfig**: structure filters
- **RegimeConfig**: regime-detection thresholds and adjustments
- **RiskConfig**: risk limits (max drawdown, Kelly fraction, lot sizes)
- **AutoOptimizerConfig**: self-tuning frequency, shift rate, rollback threshold,
  parameter ranges, and the state file name
- **MainConfig**: loop interval and feature toggles

## Directory Structure

```
python_bridge/
  config/
    settings.py          - All configuration parameters
  data/
    market_data.py       - Market data fetching and indicator computation
    multi_timeframe.py   - Higher-timeframe bias helpers
    news_calendar.py     - Optional news-window filter
  strategies/
    signal_generator.py  - Rule-based signal generation + risk filters
    auto_optimizer.py    - Local-backtest self-tuning engine
    risk_manager.py      - Kelly criterion, drawdown, position sizing
    regime_detector.py   - Market regime classification
  signals/
    bridge.py            - CSV file bridge for MT5 communication
  dashboard/
    performance_tracker.py - Per-trade / per-regime performance stats
    dashboard_renderer.py  - Console / HTML dashboard rendering
  tests/
    test_signal_generator.py - Rule-based signal tests
    test_backtester.py       - Backtest engine + self-tuning wiring tests
    test_auto_optimizer.py   - Self-tuning loop tests
    test_dashboard.py        - Performance tracker tests
    test_bridge.py           - Bridge communication tests
    test_multi_timeframe.py  - Multi-timeframe helper tests
    test_news_calendar.py    - News filter tests
    test_smart_upgrades.py   - Structure / session filter tests
  main.py              - Live signal loop entry point
  main_multi.py        - Multi-pair variant of the live loop
  backtest.py          - Local backtester + AutoOptimizer integration
  requirements.txt     - Python dependencies
  README.md            - This file
```

## Signal Format (CSV)

The CSV signal contract is unchanged so the existing `.mq5` EAs keep working.
The `model_name` field carries the rule-based label `momentum`.

```csv
timestamp,symbol,action,confidence,sl_pips,tp_pips,lot_size,model_name,regime
2024-01-15 14:30:00,XAUUSD,BUY,0.8532,30.0,9999.0,0.10,momentum,trending
```

Fields:
- **timestamp**: Signal generation time (YYYY-MM-DD HH:MM:SS)
- **symbol**: Trading instrument (XAUUSD)
- **action**: BUY, SELL, or HOLD
- **confidence**: Rules-derived confidence (0.0 to 0.95)
- **sl_pips**: Stop loss in pips
- **tp_pips**: Take profit in pips (`9999` signals the EA to manage the exit with
  a dynamic trailing stop instead of a fixed take profit)
- **lot_size**: Position size (0 for HOLD)
- **model_name**: Always `momentum` (rule-based label)
- **regime**: Current market regime

## Risk Management

- **Kelly Criterion**: Optimal position sizing with quarter-Kelly safety
- **Max Drawdown**: Halts trading at a configurable threshold
- **Daily Loss Limit**: Stops after the daily loss exceeds the limit
- **Regime Adjustment**: Reduces position size in volatile/crash regimes
- **Correlation Filter**: Prevents over-exposure in one direction
- **Progressive Trailing Stops**: Break-even then tiered trailing as profit grows

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Rule-based signal + backtest self-tuning tests
python -m pytest tests/test_signal_generator.py -v
python -m pytest tests/test_backtester.py -v
python -m pytest tests/test_auto_optimizer.py -v
```

## Environment Variables

- `MT5_COMMON_PATH`: Override the MT5 Common Files path

## Requirements

- Python 3.9+
- pandas, numpy, ta (technical indicators)
- yfinance / requests / feedparser (data feeds)
- Internet access for live data feeds
- MetaTrader 5 (for trade execution)
