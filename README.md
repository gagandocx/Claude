# Claude - AI Trading Systems

A collection of Expert Advisors (EAs) for MetaTrader 5, plus a rule-based Python signal bridge with local-backtest self-tuning and automated trade execution.

## Expert Advisors

- **AI_Adaptive_EA.mq5** - Self-learning EA with 19 AI/ML systems including ensemble, MCTS, multi-timeframe analysis, and sentiment-based decision making.
- **Python_Bridge_EA.mq5** - Signal executor that reads trade signals from the rule-based Python bridge via CSV file communication.
- **GaganEA_v3.mq5** - Gold scalping EA with tick analysis.
- **ICT_EA.mq5** - ICT (Inner Circle Trader) concepts EA.
- **XAU_M1_EA.mq5** - XAUUSD M1 timeframe EA.

## Python Signal Bridge

The `python_bridge/` directory contains a **simple, rule-based** signal generator
(no machine-learning models or training). It decides direction from short-term
price momentum and gates entries with a rules-derived confidence plus transparent
filters, then writes signals to a CSV file the EAs read:

- Momentum direction (BUY/SELL/FLAT) with an adaptive, volume-weighted lookback
- Rules-derived confidence gate: `0.2 + 0.7 * (1 - exp(-abs_diff / 1.5))`, capped 0.95
- RSI zone, session, price-structure, EMA, S/R, and ATR filters
- Regime detection (trending / ranging / volatile / crash)
- Kelly criterion risk management with progressive trailing stops
- **Local-backtest self-tuning**: instead of pre-trained checkpoints, the
  `AutoOptimizer` backtests locally and tweaks its own parameters toward the
  values that perform best (persisted to `auto_optimizer_state.json`)
- CSV-based MT5 communication bridge (unchanged signal contract; `model_name=momentum`)

Run a local backtest to self-tune: `python backtest.py --data-file <MT5_export.csv>`.

See [python_bridge/README.md](python_bridge/README.md) for full documentation.

## Utilities

- **analyze_ticks.py** - Tick data analyzer for XAUUSD CSV files, generates statistical reports for EA development.