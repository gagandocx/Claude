# Python ML Bridge for MetaTrader 5

A deep learning trade signal generation system that combines PyTorch Transformer/LSTM models, NLP sentiment analysis, and alternative data to generate high-confidence trade signals for gold (XAUUSD) trading via MetaTrader 5.

## Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   Market Data     |     |   Sentiment       |     |  Alternative Data |
|   (yfinance)      |     |   (FinBERT)       |     |  (VIX/DXY/Yields) |
+--------+----------+     +--------+----------+     +--------+----------+
         |                          |                          |
         v                          v                          v
+-------------------------------------------------------------------+
|                    Feature Engineering (50+ features)               |
+-------------------------------------------------------------------+
         |                          |                          |
         v                          v                          v
+-------------------+     +-------------------+     +-------------------+
|   Transformer     |     |   BiLSTM +        |     |  Gradient         |
|   (8 heads,       |     |   Attention       |     |  Boosting         |
|    4 layers,      |     |   (3 layers,      |     |  (sklearn)        |
|    256-dim)       |     |    128 hidden)     |     |                   |
+--------+----------+     +--------+----------+     +--------+----------+
         |                          |                          |
         v                          v                          v
+-------------------------------------------------------------------+
|              Ensemble Meta-Learner (Stacking)                      |
+-------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------+
|              Signal Generator + Risk Filters                       |
|   - Regime detection (trending/ranging/volatile/crash)             |
|   - Kelly criterion position sizing                                |
|   - Drawdown monitoring                                            |
|   - Confidence thresholds                                          |
+-------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------+
|              CSV File Bridge (MT5 Common Files)                    |
|   python_bridge_signal.csv  -->  Python_Bridge_EA.mq5             |
|   python_bridge_confirm.csv <--  Python_Bridge_EA.mq5             |
+-------------------------------------------------------------------+
```

## Quick Start

### 1. Install Dependencies

```bash
cd python_bridge
pip install -r requirements.txt
```

### 2. Train Models (Optional - uses historical data)

```bash
python train.py
```

This downloads 2 years of gold price data, trains the Transformer, LSTM, and Gradient Boosting models, fits the meta-learner, and saves checkpoints.

### 3. Run the Bridge

```bash
# Paper trading mode (default)
python main.py --paper

# Live mode
python main.py --live

# Custom interval (5 minutes)
python main.py --interval 300
```

### 4. Set Up MT5

1. Copy `Python_Bridge_EA.mq5` to your MT5 `Experts` folder
2. Compile in MetaEditor
3. Attach to an XAUUSD chart
4. Ensure the `Common Files` path is accessible to both Python and MT5

## Configuration

Edit `config/settings.py` to customize:

- **MT5_COMMON_PATH**: Path to MT5 Common Files folder
- **TransformerConfig**: Model architecture (heads, layers, dimensions)
- **LSTMConfig**: LSTM parameters (layers, hidden size, attention)
- **RiskConfig**: Risk limits (max drawdown, Kelly fraction, lot sizes)
- **SignalConfig**: Signal thresholds (confidence, cooldown, ATR multipliers)

## Directory Structure

```
python_bridge/
  config/
    settings.py          - All configuration parameters
  data/
    market_data.py       - Market data fetching and feature engineering
    sentiment.py         - FinBERT sentiment analysis
    alternative_data.py  - VIX, DXY, yields, oil data
  models/
    transformer_model.py - PyTorch Transformer (8-head, 4-layer)
    lstm_model.py        - Bidirectional LSTM with attention
    ensemble.py          - Ensemble manager with meta-learner
  strategies/
    signal_generator.py  - Signal generation with risk filters
    risk_manager.py      - Kelly criterion, drawdown, position sizing
    regime_detector.py   - Market regime classification
  signals/
    bridge.py            - CSV file bridge for MT5 communication
  tests/
    test_models.py       - Model unit tests
    test_signal_generator.py - Signal generation tests
    test_bridge.py       - Bridge communication tests
  main.py              - Main entry point (continuous loop)
  train.py             - Model training script
  requirements.txt     - Python dependencies
  README.md            - This file
```

## Signal Format (CSV)

```csv
timestamp,symbol,action,confidence,sl_pips,tp_pips,lot_size,model_name,regime
2024-01-15 14:30:00,XAUUSD,BUY,0.8532,150.5,251.0,0.10,transformer,trending
```

Fields:
- **timestamp**: Signal generation time (YYYY-MM-DD HH:MM:SS)
- **symbol**: Trading instrument (XAUUSD)
- **action**: BUY, SELL, or HOLD
- **confidence**: Model confidence (0.0 to 1.0)
- **sl_pips**: Stop loss in pips
- **tp_pips**: Take profit in pips
- **lot_size**: Position size
- **model_name**: Primary contributing model
- **regime**: Current market regime

## Models

### Transformer
- 8 attention heads for parallel pattern recognition
- 4 encoder layers with GELU activation
- 256-dimensional hidden representation
- Positional encoding for temporal awareness
- Separate confidence head

### Bidirectional LSTM
- 3 stacked BiLSTM layers
- 128 hidden units per direction (256 effective)
- Scaled dot-product attention over hidden states
- Orthogonal initialization for stable gradients

### Gradient Boosting
- 100 trees, max depth 5
- Serves as non-neural baseline
- Captures linear and feature-interaction patterns

### Ensemble
- Stacking meta-learner (Logistic Regression)
- Dynamic weight adjustment based on recent accuracy
- Model disagreement as uncertainty indicator

## Risk Management

- **Kelly Criterion**: Optimal position sizing with quarter-Kelly safety
- **Max Drawdown**: Halts trading at configurable threshold (default 10%)
- **Daily Loss Limit**: Stops after daily loss exceeds limit (default 5%)
- **Regime Adjustment**: Reduces position size in volatile/crash regimes
- **Correlation Filter**: Prevents over-exposure in one direction
- **Time Filter**: Avoids trading during illiquid hours

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_models.py -v
python -m pytest tests/test_signal_generator.py -v
python -m pytest tests/test_bridge.py -v
```

## Environment Variables

- `MT5_COMMON_PATH`: Override the MT5 Common Files path

## Requirements

- Python 3.9+
- PyTorch 2.0+
- HuggingFace Transformers 4.30+
- Internet access for data feeds (yfinance, RSS)
- MetaTrader 5 (for trade execution)
