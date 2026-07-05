# Claude - AI Trading Systems

A collection of AI-powered Expert Advisors (EAs) for MetaTrader 5, featuring deep learning models, adaptive algorithms, and automated trade execution.

## Expert Advisors

- **AI_Adaptive_EA.mq5** - Self-learning EA with 19 AI/ML systems including ensemble, MCTS, multi-timeframe analysis, and sentiment-based decision making.
- **Python_Bridge_EA.mq5** - Signal executor that reads trade signals from the Python ML Bridge system via CSV file communication.
- **GaganEA_v3.mq5** - Gold scalping EA with tick analysis.
- **ICT_EA.mq5** - ICT (Inner Circle Trader) concepts EA.
- **XAU_M1_EA.mq5** - XAUUSD M1 timeframe EA.

## Python ML Bridge

The `python_bridge/` directory contains a complete deep learning pipeline for generating trade signals:

- PyTorch Transformer (8-head, 4-layer, 256-dim) for market pattern recognition
- Bidirectional LSTM with attention for sequential analysis
- FinBERT sentiment analysis on financial news
- Alternative data feeds (VIX, DXY, yields, oil)
- Ensemble meta-learner combining all models
- Kelly criterion risk management
- CSV-based MT5 communication bridge

See [python_bridge/README.md](python_bridge/README.md) for full documentation.

## Utilities

- **analyze_ticks.py** - Tick data analyzer for XAUUSD CSV files, generates statistical reports for EA development.