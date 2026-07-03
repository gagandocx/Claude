# Adaptive Multi-Currency Trading System

Advanced regime-aware trading system that dynamically detects market conditions, selects optimal strategies, adapts position sizing, and manages risk across multiple currency pairs simultaneously.

## Architecture Overview

```
Market Data (per symbol)
    |
    v
[Regime Detection] --> Classifies market into 6 regimes
    |
    v
[Strategy Selection] --> Routes regime to optimal strategy
    |
    v
[Signal Generation] --> Produces entry signals with SL/TP
    |
    v
[Position Sizing] --> Kelly + equity curve feedback + correlation penalty
    |
    v
[Risk Manager] --> Portfolio-level approval/rejection
    |
    v
[Execution] --> MT5 order placement (live) or simulation (backtest)
    |
    v
[Online Learner] --> Updates performance stats, adapts parameters
```

## Quick Start

### Demo (No data required)

```bash
python run_demo.py
```

This generates synthetic data with embedded regime shifts, runs the full adaptive pipeline on 3 symbols (XAUUSD, EURUSD, GBPJPY), and prints detailed results.

### Backtest with Real Data

```bash
python run_backtest.py --data-dir ./tick_data --symbols XAUUSD,EURUSD --timeframe 1min
```

### Live Trading

```bash
python live_trader.py --symbols XAUUSD,EURUSD,GBPJPY --magic 202501 --risk-profile balanced
```

## Market Regime Detection

The system classifies the market into 6 regimes using multiple indicators:

| Regime | Description | Key Indicators |
|--------|-------------|----------------|
| TRENDING_UP | Strong upward trend | ADX > 25, DI+ > DI-, positive EMA slope, Hurst > 0.6 |
| TRENDING_DOWN | Strong downward trend | ADX > 25, DI- > DI+, negative EMA slope, Hurst > 0.6 |
| RANGING_NARROW | Low volatility sideways | ADX < 18, low vol ratio, small EMA slope |
| RANGING_WIDE | Higher volatility sideways | ADX < 25, moderate vol ratio |
| VOLATILE_BREAKOUT | Volatility expansion | Vol ratio > 1.4, ATR spike, large moves |
| MEAN_REVERTING | Anti-persistent price action | Hurst < 0.4, low ADX, price oscillating |

Detection uses a weighted voting system across indicators with confidence scoring and regime persistence (hysteresis to prevent rapid switching).

## Strategies

### TrendFollower
- **When used**: TRENDING_UP, TRENDING_DOWN
- **Logic**: Fast/slow EMA crossover with ADX filter, enters on pullback to fast EMA
- **Risk/Reward**: 1.5 ATR stop, 3.0 ATR target

### MeanReversion
- **When used**: RANGING_WIDE, MEAN_REVERTING
- **Logic**: Bollinger Band touch + RSI confirmation, Keltner squeeze filter
- **Risk/Reward**: 1.2 ATR stop, target at middle band

### BreakoutTrader
- **When used**: VOLATILE_BREAKOUT, transitioning from RANGING_NARROW
- **Logic**: ATR compression detection + breakout above/below consolidation range
- **Risk/Reward**: 2.0 ATR stop, 4.0 ATR target (wide for breakout moves)

### ScalpMomentum
- **When used**: TRENDING_UP, TRENDING_DOWN (fast scalps with trend)
- **Logic**: Fast RSI + VWAP deviation + consecutive same-direction bars
- **Risk/Reward**: 0.8 ATR stop, 1.6 ATR target (tight scalps)

### FadeStrategy
- **When used**: VOLATILE_BREAKOUT (false breakouts), RANGING_WIDE
- **Logic**: Order flow imbalance detection (z-score), fades extreme moves
- **Risk/Reward**: 1.5 ATR stop, 2.0 ATR target

## Position Sizing

The adaptive position sizer uses multiple factors:

1. **Two-Mode Sizing (Grow/Protect)**:
   - At equity high: use `risk_grow` (aggressive, default 15%)
   - In drawdown: `risk_protect * (equity/peak)^dd_power` (exponential decay)

2. **Kelly Criterion**: Half-Kelly applied as upper bound based on estimated win rate and reward/risk ratio

3. **Equity Curve Feedback**: Reduces size automatically as drawdown deepens, increases at new highs

4. **Correlation Penalty**: Reduces position size when new trade is highly correlated with existing positions

5. **Streak Adjustment**: Boosts risk after losses (Martingale-lite), reduces after wins (profit lock)

6. **Risk Caps**: Per-symbol (5%) and total portfolio (15%) risk limits

## Online Learning

The system adapts in real-time without external ML libraries:

- **Strategy Performance Tracking**: Exponentially-weighted mean/variance of P&L per (strategy, regime) pair
- **Regime Transition Matrix**: Learns which regimes follow which, enabling anticipatory positioning
- **Parameter Adaptation**: Tracks which parameter values correlate with better outcomes
- **Market Profiling**: Session-based volatility/spread patterns that update continuously
- **Win Rate Estimation**: Per-strategy per-regime win rate with Bayesian confidence blending

## Configuration

### Risk Profiles

| Profile | Risk/Trade | Max Positions | DD Halt | Best For |
|---------|-----------|---------------|---------|----------|
| Conservative | 8% grow / 1% protect | 3 | 15% | Capital preservation |
| Balanced | 15% grow / 2% protect | 6 | 20% | Steady growth |
| Aggressive | 20% grow / 4% protect | 8 | 30% | Maximum growth |

### Configuration File

```json
{
  "risk_profile": "balanced",
  "position_sizing": {
    "initial_equity": 1000.0,
    "leverage": 500.0,
    "risk_grow": 0.15,
    "risk_protect": 0.02,
    "dd_power": 12.0,
    "dd_halt": 0.20,
    "use_kelly": true
  },
  "risk_management": {
    "max_concurrent_positions": 6,
    "max_total_risk": 0.15,
    "daily_loss_limit": 0.05
  },
  "symbols": {
    "XAUUSD": {
      "contract_size": 100,
      "pip_size": 0.01,
      "typical_spread": 0.30
    }
  }
}
```

Save to `config.json` and load with `--config config.json`.

## Symbol Configuration

| Symbol | Pip Size | Contract Size | Typical Spread | Description |
|--------|----------|---------------|----------------|-------------|
| XAUUSD | 0.01 | 100 | 0.30 | Gold vs USD |
| EURUSD | 0.0001 | 100,000 | 0.00012 | Euro vs USD |
| GBPJPY | 0.01 | 100,000 | 0.03 | Pound vs Yen |
| USDJPY | 0.01 | 100,000 | 0.015 | Dollar vs Yen |
| NAS100 | 0.01 | 1 | 1.50 | Nasdaq 100 |
| BTCUSD | 0.01 | 1 | 30.00 | Bitcoin vs USD |

## Live Trading Setup

### Requirements
- Windows OS with MetaTrader 5 terminal
- MT5 terminal connected to a broker account
- Python 3.8+ with numpy, pandas, MetaTrader5

### Installation
```bash
pip install numpy pandas MetaTrader5
```

### Running
```bash
python live_trader.py --symbols XAUUSD,EURUSD,GBPJPY --magic 202501 --risk-profile balanced
```

### State Persistence
The live trader automatically saves state (regime, learner data, portfolio) to `adaptive_state.json`. On restart, it picks up where it left off.

### Graceful Shutdown
Press Ctrl+C or send SIGTERM. The trader saves state and closes cleanly.

## Risk Management Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_risk_per_symbol | 5% | Maximum risk on any single symbol |
| max_total_risk | 15% | Maximum total portfolio risk |
| max_concurrent_positions | 6 | Maximum open positions across all symbols |
| dd_halt_threshold | 20% | Halt ALL trading at this drawdown |
| dd_reduce_threshold | 10% | Start reducing size at this drawdown |
| daily_loss_limit | 5% | Stop trading after this daily loss |
| max_correlation | 0.7 | Reduce size when correlation exceeds this |

## File Structure

```
adaptive_system/
  __init__.py          - Package init with version
  config.py            - Configuration management
  data_loader.py       - Data loading and synthetic generation
  backtest_engine.py   - Multi-symbol backtester
  live_trader.py       - MT5 live trading
  run_demo.py          - Self-contained demo
  run_backtest.py      - Production backtest script
  README.md            - This documentation
  core/
    __init__.py        - Core module exports
    indicators.py      - Technical indicators (RSI, ATR, EMA, BB, ADX, etc.)
    regime_detector.py - Market regime classification
    strategies.py      - 5 strategy implementations
    strategy_selector.py - Regime-to-strategy routing
    position_sizer.py  - Adaptive position sizing
    online_learner.py  - Online statistical learning
    risk_manager.py    - Portfolio risk management
    portfolio_manager.py - Multi-symbol coordination
```
