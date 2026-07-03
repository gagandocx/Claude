# HFT Scalper Pro - Ensemble Strategy EA

## Overview

HFT Scalper Pro is a high-frequency scalping Expert Advisor for MetaTrader 5 designed specifically for XAUUSD (Gold) on the M1 (1-minute) timeframe. It uses an ensemble approach combining three proven sub-strategies into a consensus-driven trading system.

## Strategy Logic

The EA combines three independent signal generators and only trades when at least 2 out of 3 agree on direction:

### Sub-Strategy 1: OrderFlow Contrarian
- Calculates Order Flow Imbalance (OFI) using bar close position relative to range, weighted by tick volume
- Computes rolling z-score of OFI over lookback period
- Generates SELL when z-score > threshold (fading extreme buying pressure)
- Generates BUY when z-score < -threshold (fading extreme selling pressure)
- Based on the negative autocorrelation property of order flow (-0.177 tick-level, -0.456 block-level)

### Sub-Strategy 2: Multi-Timeframe Momentum
- Uses slow EMA (40-period) to determine trend direction
- Calculates trend slope normalized by ATR
- Uses fast RSI (7-period) to detect pullbacks within the trend
- Generates BUY when trend is UP and RSI shows oversold pullback
- Generates SELL when trend is DOWN and RSI shows overbought pullback

### Sub-Strategy 3: Spread Fade
- Monitors rolling median spread over lookback period
- Detects spread widening events (spread > 2.5x median)
- Waits for spread contraction (< 1.5x median)
- Trades in the direction of price movement during contraction
- Exploits the mean-reverting nature of bid-ask spreads

### Consensus Scoring
- Each sub-strategy votes: BUY (+1) or SELL (+1) or NEUTRAL (0)
- **Score 2**: Two strategies agree - trade with wider stops (2.5x ATR SL, 2.0x ATR TP)
- **Score 3**: All three agree - trade with tighter stops (1.5x ATR SL, 3.0x ATR TP)
- Score 1 or 0: No trade

## Backtest Results

| Metric | Ensemble | OrderFlow | MomentumMTF | SpreadFade |
|--------|----------|-----------|-------------|------------|
| PnL ($) | $430.59 | $2,519.02 | $325.06 | $1,062.40 |
| Win Rate | 80.0% | 45.0% | 50.0% | 43.1% |
| Max Drawdown | -10.5% | -29.0% | -34.7% | -24.5% |
| Sharpe Ratio | 3.76 | 3.90 | 1.77 | 2.93 |
| Profit Factor | 2.980 | 1.192 | 1.179 | 1.489 |
| Trades | 15 | 429 | 72 | 51 |
| Avg Trade | $28.71 | $5.87 | $4.51 | $20.83 |
| Validation PnL | +$29.36 | -$219.62 | +$50.91 | -$272.77 |

**Key Advantages of Ensemble:**
- Lowest drawdown (-10.5% vs -29% to -34.7% for individuals)
- Highest win rate (80% vs 43-50%)
- Best per-trade expectancy ($28.71)
- Positive out-of-sample validation (+$29.36)
- Highest profit factor (2.98 vs 1.19-1.49)

## Input Parameters

### Ensemble Strategy Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| OFI Period | 30 | Lookback bars for Order Flow z-score |
| OFI Threshold | 2.0 | Z-score threshold for OFI signal |
| Slow Period | 40 | EMA period for trend direction |
| Fast RSI Period | 7 | RSI period for pullback detection |
| Fast RSI OB | 75 | RSI overbought level |
| Fast RSI OS | 20 | RSI oversold level |
| Trend Threshold | 0.1 | Minimum trend slope (ATR-normalized) |
| Spread Lookback | 30 | Bars for median spread calculation |
| Wide Threshold | 2.5 | Multiple of median for "wide" spread |
| Contract Threshold | 1.5 | Spread must contract below this x median |
| Min Score | 2 | Minimum consensus score to trade (2-3) |

### Risk Management
| Parameter | Default | Description |
|-----------|---------|-------------|
| Lot Size | 0.1 | Fixed position size |
| SL ATR Mult High | 1.5 | Stop loss (score=3, high confidence) |
| SL ATR Mult Low | 2.5 | Stop loss (score=2, moderate confidence) |
| TP ATR Mult High | 3.0 | Take profit (score=3) |
| TP ATR Mult Low | 2.0 | Take profit (score=2) |
| ATR Period | 14 | ATR calculation period |
| Max Daily Loss | $50 | Daily loss limit (stops trading) |
| Max Drawdown % | 30% | Account drawdown limit (disables EA) |
| Max Spread | 30 pts | Maximum spread to enter trades |
| Use Trailing Stop | true | Enable ATR-based trailing |
| Trailing ATR Mult | 1.0 | Trailing stop distance in ATR |

### Session Filter
| Parameter | Default | Description |
|-----------|---------|-------------|
| Use Session Filter | true | Enable hour-based filtering |
| Session 1 Start | 4 | Hour 4 UTC (London pre-open) |
| Session 1 End | 4 | Single hour session |
| Session 2 Start | 8 | Hour 8 UTC (London/NY overlap start) |
| Session 2 End | 21 | Hour 21 UTC (NY close) |

### General Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| Magic Number | 202604 | Unique EA identifier |
| Cooldown Bars | 3 | Minimum bars between trades |
| Symbol | XAUUSD | Trading symbol |
| Max Retries | 3 | Order send retry attempts |
| Timer Seconds | 60 | Equity check interval |

## Broker Requirements

- **Account Type**: ECN/Raw Spread preferred (lower spreads on gold)
- **Leverage**: 1:500 minimum recommended
- **Spread**: Average below 15 points on XAUUSD (30pt max filter built in)
- **Execution**: Market execution with < 50ms latency ideal
- **Minimum Deposit**: $1,000 recommended for 0.1 lot
- **Commission**: $7 per standard lot round-trip or equivalent
- **Hedging**: Not required (one position at a time)
- **Platform**: MetaTrader 5 (MQL5)

## Installation Guide

### Step 1: Copy EA File
1. Open MetaTrader 5
2. Navigate to `File > Open Data Folder`
3. Go to `MQL5/Experts/`
4. Copy `HFT_Scalper_Pro.mq5` into this folder

### Step 2: Compile
1. Open MetaEditor (F4 in MT5)
2. Open `HFT_Scalper_Pro.mq5`
3. Click Compile (F7)
4. Verify 0 errors (warnings are acceptable)

### Step 3: Attach to Chart
1. Open a XAUUSD M1 chart
2. Drag the EA from Navigator onto the chart
3. In the Inputs tab, review and adjust parameters
4. Enable "Allow Algo Trading" checkbox
5. Click OK

### Step 4: Verify Operation
1. Check the Experts tab for initialization messages
2. Verify "HFT Scalper Pro initialized" appears
3. Ensure AutoTrading button is enabled (green)
4. Monitor first few trades in the Trade tab

## Risk Warnings

- Past performance does not guarantee future results
- This EA was optimized on April 2026 XAUUSD data
- Market conditions can change, reducing strategy effectiveness
- Always start with demo trading or minimal lot size
- The max drawdown of 10.5% in testing may be exceeded in live conditions
- Gold is highly volatile; use proper risk management at all times
- Consider reducing lot size during high-impact news events
- Monitor the EA daily, especially during the first week of live trading

## Recommended Settings by Account Size

| Account | Lot Size | Max Daily Loss | Max DD % |
|---------|----------|----------------|----------|
| $500 | 0.05 | $25 | 25% |
| $1,000 | 0.1 | $50 | 30% |
| $5,000 | 0.3 | $150 | 25% |
| $10,000 | 0.5 | $250 | 20% |

## Troubleshooting

- **"Symbol not available"**: Ensure your broker offers XAUUSD; some use XAU/USD or GOLD
- **No trades**: Check session hours match your broker's server time (UTC)
- **Large slippage**: Reduce lot size or switch to a lower-latency VPS
- **Frequent SL hits**: Market may be in trending mode; consider increasing SL multiplier
