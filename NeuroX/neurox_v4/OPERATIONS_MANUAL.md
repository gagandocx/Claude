# NeuroX v7.5 - Operations Manual
# XAUUSD M1 Scalping System | Fusion Markets MT5

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [System Requirements](#2-system-requirements)
3. [Architecture](#3-architecture)
4. [File Paths Reference](#4-file-paths-reference)
5. [First-Time Setup](#5-first-time-setup)
6. [Daily Operations](#6-daily-operations)
7. [Update Procedures](#7-update-procedures)
8. [CSV Bridge Protocol](#8-csv-bridge-protocol)
9. [Configuration Reference](#9-configuration-reference)
10. [Model Training (Colab/Kaggle)](#10-model-training-colabkaggle)
11. [Troubleshooting](#11-troubleshooting)
12. [Emergency Procedures](#12-emergency-procedures)
13. [Feature Reference](#13-feature-reference)

---

## 1. System Overview

| Field | Value |
|-------|-------|
| **Product** | NeuroX v7.5 |
| **Strategy** | XAUUSD M1 Scalping |
| **Broker** | Fusion Markets (MT5) |
| **Execution** | 17-Model AI Ensemble + Institutional-Grade Risk |
| **Architecture** | Python ML Bridge + MQL5 EA (CSV IPC) |
| **GitHub** | gagandocx/Claude, branch: feature/5-model-ensemble-tcn-lgbm |

### How It Works

1. **Python** generates trade signals using a 17-model ensemble
2. Python writes signals to `python_bridge_signal.csv`
3. **MT5 EA** reads signals every tick and executes trades
4. EA manages trailing stops, breakeven, and partial closes
5. EA writes confirmations to `python_bridge_confirm.csv`
6. Python polls confirmations every 10ms for instant position sync
7. Trading Brain manages risk, sizing, and edge tracking in real-time

---
## 2. System Requirements

### Hardware

| Component | Specification |
|-----------|---------------|
| CPU | Intel i7-4720HQ (or equivalent) |
| RAM | 24 GB |
| GPU | GTX 950M (CPU-only inference in production) |
| Storage | SSD recommended for CSV I/O speed |
| Network | Stable internet for MT5 connection |

### Software

| Software | Version / Details |
|----------|-------------------|
| OS | Windows 10/11 |
| Python | 3.14 |
| MetaTrader 5 | Fusion Markets terminal |
| MetaEditor | C:\\Program Files\\Fusion Markets MetaTrader 5\\MetaEditor64.exe |
| scikit-learn | 1.9.0 (MUST match training environment) |
| PyTorch | >= 2.0.0 (CPU build recommended for VPS) |

### Python Dependencies

Install with:
```bash
pip install -r requirements.txt
```

Key packages: torch, scikit-learn==1.9.0, lightgbm, catboost, xgboost, yfinance, ta, pandas, numpy, scipy, hmmlearn, joblib, transformers, feedparser, beautifulsoup4, requests, websocket-client, schedule

> **CRITICAL**: scikit-learn version MUST be 1.9.0 to match the training environment. Model loading will fail with version mismatch.

---
## 3. Architecture

```
+------------------+        CSV Files        +------------------+
|                  | -----------------------> |                  |
|   Python ML      |  python_bridge_signal    |   MT5 EA         |
|   Bridge         |  python_bridge_brain     |   (Executor)     |
|   (17 Models)    | <----------------------- |                  |
|                  |  python_bridge_confirm    |   Trailing SL    |
|                  |  python_bridge_tick_data  |   Breakeven      |
|                  |  python_bridge_spread     |   Partial Close  |
|                  |  python_bridge_balance    |                  |
+------------------+                          +------------------+
```

### Data Flow

| Direction | File | Frequency | Purpose |
|-----------|------|-----------|---------|
| Python -> EA | python_bridge_signal.csv | On signal | Trade entry orders |
| Python -> EA | python_bridge_brain_settings.csv | Every cycle | Brain SL/trailing params |
| Python -> EA | python_bridge_heartbeat.txt | Every 1s | Python alive status |
| EA -> Python | python_bridge_confirm.csv | On fill/close | Execution confirmations |
| EA -> Python | python_bridge_tick_data.csv | Every tick | Order flow features |
| EA -> Python | python_bridge_spread.csv | Every 500ms | Current spread |
| EA -> Python | python_bridge_balance.csv | On trade close | Account balance/equity |
| EA -> Python | mt5_bridge_heartbeat.txt | Every 2s | EA alive status |

### 17-Model Ensemble

| # | Model | Type | Weight |
|---|-------|------|--------|
| 1 | Transformer | Deep Learning | 6% |
| 2 | BiLSTM | Deep Learning | 5% |
| 3 | TCN | Deep Learning | 5% |
| 4 | PatchTST | Deep Learning | 8% |
| 5 | TFT | Deep Learning | 8% |
| 6 | N-HiTS | Deep Learning | 5% |
| 7 | iTransformer | Deep Learning | 8% |
| 8 | Mamba | Deep Learning | 7% |
| 9 | DLinear | Deep Learning | 3% |
| 10 | xLSTM | Deep Learning | 8% |
| 11 | TimesNet | Deep Learning | 6% |
| 12 | Chronos | Foundation Model | 9% |
| 13 | TimeMixer | Deep Learning | 7% |
| 14 | SOFTS | Deep Learning | 5% |
| 15 | GradBoost/LightGBM | Tree-Based | 4% |
| 16 | XGBoost/LightGBM | Tree-Based | 3% |
| 17 | CatBoost | Tree-Based | 3% |
| - | Meta-Learner | Stacking | Top layer |

Weights are dynamically adjusted using Sharpe-ratio attribution (20-trade reweight cycle, 0.02 min floor).

---
## 4. File Paths Reference

### MT5 Paths

| Purpose | Path |
|---------|------|
| MT5 Terminal ID | EE1261C89A64D41685651B738DC52A84 |
| MT5 Common Files | C:\\Users\\gagan\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\\ |
| MT5 Experts | C:\\Users\\gagan\\AppData\\Roaming\\MetaQuotes\\Terminal\\EE1261C89A64D41685651B738DC52A84\\MQL5\\Experts\\Advisors\\ |
| MetaEditor | C:\\Program Files\\Fusion Markets MetaTrader 5\\MetaEditor64.exe |

### Local Paths

| Purpose | Path |
|---------|------|
| EA Folder (v4 ACTIVE) | D:\\Automation\\EA Testing\\NeuroX\\NeuroX_v4\\ |
| Python Bridge | D:\\Automation\\EA Testing\\NeuroX\\NeuroX_v4\\python_bridge_v4\\ |
| Model Checkpoints | D:\\Automation\\EA Testing\\NeuroX\\NeuroX_v4\\python_bridge_v4\\checkpoints\\ |
| Logs | D:\\Automation\\EA Testing\\NeuroX\\NeuroX_v4\\python_bridge_v4\\logs\\ |

### Bridge CSV Files (in MT5 Common Files)

| File | Writer | Reader |
|------|--------|--------|
| python_bridge_signal.csv | Python | EA |
| python_bridge_confirm.csv | EA | Python |
| python_bridge_brain_settings.csv | Python | EA |
| python_bridge_tick_data.csv | EA | Python |
| python_bridge_spread.csv | EA | Python |
| python_bridge_balance.csv | EA | Python |
| python_bridge_heartbeat.txt | Python | EA |
| mt5_bridge_heartbeat.txt | EA | Python |
| python_bridge_exit.csv | Python | EA |
| python_bridge_status.txt | Python | EA |

---
## 5. First-Time Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/gagandocx/Claude.git
cd Claude
git checkout feature/5-model-ensemble-tcn-lgbm
```

### Step 2: Set Up Local Folder

```cmd
mkdir "D:\Automation\EA Testing\NeuroX\NeuroX_v4"
cd /d "D:\Automation\EA Testing\NeuroX\NeuroX_v4"
```

### Step 3: Run the Update/Install Script

```cmd
cd /d "D:\Automation\EA Testing\NeuroX\NeuroX_v4"
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_run_v4.bat?t=' + str(__import__('time').time()),'neurox_run_v4.bat'); print('downloaded')" 
neurox_run_v4.bat
```

This script will:
- Download all Python source files from GitHub
- Create the directory structure (python_bridge_v4/, config/, models/, strategies/, etc.)
- Copy the EA file to the MT5 Experts folder
- Compile the EA using MetaEditor
- Start the Python bridge

### Step 4: Install Python Dependencies

```cmd
cd /d "D:\Automation\EA Testing\NeuroX\NeuroX_v4\python_bridge_v4"
pip install -r requirements.txt
```

For CPU-only PyTorch (recommended on trading machines):
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 5: Install EA in MT5

1. Open MetaTrader 5 (Fusion Markets)
2. EA file is auto-copied to the MT5 Experts\\Advisors folder by neurox_run_v4.bat
3. In MT5, press F7 or right-click Navigator > Refresh
4. Drag **NeuroX_EA_v4** onto the XAUUSD M1 chart
5. Enable AutoTrading (green button in toolbar)
6. In EA settings, ensure:
   - Allow DLL imports: Yes
   - Allow live trading: Yes

### Step 6: Place Model Checkpoints

Copy trained model files to:
```
D:\Automation\EA Testing\NeuroX\NeuroX_v4\python_bridge_v4\checkpoints\
```

Required files:
- transformer_model.pt
- lstm_model.pt
- tcn_model.pt
- patch_tst_model.pt
- tft_model.pt
- nhits_model.pt
- itransformer_model.pt
- mamba_model.pt
- dlinear_model.pt
- xlstm_model.pt
- timesnet_model.pt
- chronos_model.pt
- timemixer_model.pt
- softs_model.pt
- gradient_boost_model.joblib
- xgboost_model.joblib
- catboost_model.cbm
- meta_learner.joblib

### Step 7: Verify Installation

```cmd
cd /d "D:\Automation\EA Testing\NeuroX\NeuroX_v4\python_bridge_v4"
python verify_models.py
```

---
## 6. Daily Operations

### Starting the System

**Option A: Standard Launch (recommended)**
```cmd
cd /d "D:\Automation\EA Testing\NeuroX\NeuroX_v4\python_bridge_v4"
python -u main.py --live
```

**Option B: With Watchdog (auto-restart on crash)**
```cmd
cd /d "D:\Automation\EA Testing\NeuroX\NeuroX_v4"
neurox_watchdog.bat
```

**Option C: Full Update + Launch**
```cmd
cd /d "D:\Automation\EA Testing\NeuroX\NeuroX_v4"
neurox_run_v4.bat
```

### Pre-Session Checklist

1. Confirm MT5 is running and connected to Fusion Markets
2. Confirm AutoTrading is enabled (green button)
3. Confirm EA is attached to XAUUSD M1 chart
4. Start Python bridge (Option A, B, or C above)
5. Watch for startup log confirming all models loaded:
   ```
   [INFO] NeuroX v7.5 starting...
   [INFO] Loaded 17/17 models
   [INFO] Trading Brain: ACTIVE
   [INFO] Heartbeat: OK
   ```
6. Verify heartbeat files are being updated (both directions)

### Monitoring During Trading

The system runs autonomously. Monitor via:

- **Console output**: Real-time signal generation, trade execution, P&L
- **Dashboard**: HTML report at python_bridge_v4/dashboard/report.html
- **Log files**: python_bridge_v4/logs/

### Stopping the System

1. Press Ctrl+C in the Python console
2. Optionally disable AutoTrading in MT5 (red button)
3. The EA will continue managing open positions even if Python stops

### Main Loop Cycle (Every 2 Seconds)

1. Fetch market data (yfinance M1 candles)
2. Validate data (NaN check, price sanity, staleness)
3. Compute 46 features (technical + sentiment + order flow)
4. Run 17-model ensemble inference
5. Apply Platt calibration (regime-conditional)
6. Generate signal (momentum + model confidence gate)
7. Trading Brain evaluation (8 layers: edge, regime, risk, sizing)
8. Write signal CSV if approved
9. Poll confirmations (10ms background thread)
10. Update Sharpe weights, metrics, and dashboard

---
## 7. Update Procedures

### Quick Update (One-Liner)

```cmd
cd /d "D:\Automation\EA Testing\NeuroX\NeuroX_v4" && python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_run_v4.bat?t=' + str(__import__('time').time()),'neurox_run_v4.bat'); print('updated')" && neurox_run_v4.bat
```

### Manual Update

1. Stop the Python bridge (Ctrl+C)
2. Run neurox_run_v4.bat - it downloads all latest files from GitHub
3. The script auto-compiles the EA and starts Python

### Updating Models Only

Copy new checkpoint files to python_bridge_v4/checkpoints/ and restart Python.

### Verify After Update

```cmd
cd /d "D:\Automation\EA Testing\NeuroX\NeuroX_v4\python_bridge_v4"
python verify_models.py
python -c "import main; print(main.VERSION)"
```

---

## 8. CSV Bridge Protocol

### Signal File (Python -> EA)

**File**: python_bridge_signal.csv

| Column | Type | Description |
|--------|------|-------------|
| timestamp | string | ISO format datetime |
| symbol | string | Always XAUUSD |
| action | string | BUY or SELL |
| confidence | float | 0.0 - 1.0 model confidence |
| sl_pips | float | Stop loss in dollar distance |
| tp_pips | float | Take profit (0 = EA manages) |
| lot_size | float | Position size |
| model_name | string | Winning model or ensemble |
| regime | string | Current market regime |

### Confirmation File (EA -> Python)

**File**: python_bridge_confirm.csv

| Column | Type | Description |
|--------|------|-------------|
| timestamp | string | ISO format datetime |
| ticket | int | MT5 position ticket |
| symbol | string | XAUUSD |
| action | string | BUY or SELL |
| lot_size | float | Filled lot size |
| open_price | float | Actual fill price |
| sl | float | Stop loss price |
| tp | float | Take profit price |
| status | string | FILLED, CLOSED, or FAILED |
| profit | float | P&L in dollars (CLOSED only) |
| slippage | float | Fill slippage in price units |

### Brain Settings File (Python -> EA)

**File**: python_bridge_brain_settings.csv

Written every cycle. EA reads every 1 second. Contains dynamic SL, trailing stop tiers, and position management parameters.

### Tick Data File (EA -> Python)

**File**: python_bridge_tick_data.csv

Real-time tick stream for order flow features (bid/ask imbalance, volume delta, trade flow direction).

### Spread File (EA -> Python)

**File**: python_bridge_spread.csv

Current bid-ask spread. Python uses this for spread gating (blocks entries when spread > 1.5x average or > 80 points absolute).

### Balance File (EA -> Python)

**File**: python_bridge_balance.csv

Written after each trade close. Contains current account balance and equity for live position sizing.

### Heartbeat Protocol

- **Python -> EA**: Writes python_bridge_heartbeat.txt every 1 second
- **EA -> Python**: Writes mt5_bridge_heartbeat.txt every 2 seconds
- **Timeout**: Connection considered OFFLINE after 5 seconds without heartbeat
- **Recovery**: Auto-reconnects when heartbeat resumes

---
## 9. Configuration Reference

### Trading Parameters (v7.5 Active)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Fixed SL | $2.00 | InpFixedSL in EA |
| Breakeven | $0.70 | Move SL to entry at this profit |
| Trail Tier 1 | $0.70/$0.20 | Activate at $0.70, trail $0.20 behind |
| Trail Tier 2 | $0.40 | Second trail distance |
| Momentum Threshold | $0.60 | Min price move for momentum |
| Momentum Lookback | 8 bars | M1 bars for momentum detection |
| Cooldown | 2 cycles | Between signals (~4 seconds) |
| Max Hold | 20 bars | Maximum position duration |
| Max Positions | 1 | Single position at a time |

### Entry Timing

| Parameter | Value |
|-----------|-------|
| Timeout | 10s (adaptive based on volatility) |
| Pullback Target | $0.30 |
| Breakout Override | $1.00 |
| Adaptive Min | 3s |
| Adaptive Max | 30s |

### Model Confidence

| Parameter | Value |
|-----------|-------|
| Min Confidence | 0.25 |
| Strong Confidence | 0.40 |
| Model Override Threshold | 0.60 (regime-adaptive) |
| Platt Calibration Window | 200 samples |
| Platt Min Samples | 50 |
| Regime-Conditional | Yes (per-regime A/B parameters) |

### Risk Management

| Parameter | Value |
|-----------|-------|
| Kelly Fraction | 0.25x |
| Kelly Min Trades | 20 |
| Kelly Max Lot | 0.10 |
| Monte Carlo Simulations | 1000 |
| Max Ruin Probability | 5% |
| Serial Correlation | 0.15 |
| Daily Loss Limit | $50.00 |
| Drawdown Reduce Threshold | 8% |
| Drawdown Stop Threshold | 15% |
| Consecutive Loss Reduce | 3 losses |
| Consecutive Loss Stop | 6 losses |

### Sharpe Weights

| Parameter | Value |
|-----------|-------|
| Reweight Cycle | Every 20 trades |
| Min Weight Floor | 0.02 per model |
| Lookback Trades | 100 |
| Min Trades per Model | 20 |

### Session Multipliers

| Session | Hours (UTC) | Multiplier |
|---------|-------------|-----------|
| London/NY Overlap | 13:00-16:00 | 1.3x |
| London | 08:00-16:00 | 1.2x |
| New York | 13:00-21:00 | 1.0x |
| Asian | 00:00-08:00 | 0.8x |
| Off Hours | 21:00-00:00 | 0.6x |

### Brain Configuration

| Parameter | Value |
|-----------|-------|
| Base Lot | 0.01 |
| Min Lot | 0.01 |
| Max Lot | 0.05 |
| Edge HOT Threshold | Win rate > 62% |
| Edge COLD Threshold | Win rate < 42% |
| Edge BROKEN Threshold | Win rate < 30% |
| HOT Multiplier | 1.5x |
| COLD Multiplier | 0.5x |
| Volatile Multiplier | 0.4x |
| Min Trade Score | 30/100 |
| Min Win Probability | 52% (Bayesian posterior) |
| Risk per Trade | 0.5% of account |
| Probe Interval | 300s (when BROKEN) |

---
## 10. Model Training (Colab/Kaggle)

### Training Environment

| Setting | Value |
|---------|-------|
| Platform | Google Colab / Kaggle T4 GPU |
| Python | 3.10+ |
| scikit-learn | 1.9.0 (REQUIRED - must match inference) |
| PyTorch | Latest stable |
| Training Data | XAUUSD M1 candles, 1+ year |
| Input Features | 46 per sample |
| Sequence Length | 64 bars |
| Output Classes | 3 (BUY, SELL, HOLD) |

### Training Procedure

1. Upload train_colab.py to Colab/Kaggle
2. Ensure GPU runtime is selected (T4 minimum)
3. Install dependencies:
   ```python
   !pip install scikit-learn==1.9.0 lightgbm catboost xgboost torch ta yfinance hmmlearn
   ```
4. Run training:
   ```python
   !python train_colab.py
   ```
5. Download checkpoint files from output
6. Copy to local checkpoints/ folder

### Walk-Forward Retraining (Automatic)

| Parameter | Value |
|-----------|-------|
| Schedule | Weekly (Saturday) |
| Gate | New model must be > 2% better |
| Validation Window | 500 bars walk-forward |
| Max Duration | 30 minutes |
| Min Trades Required | 20 (for outcome incorporation) |
| Min Days Between | 7 |

### Meta-Learner Auto-Retrain

- Accumulates stacked predictions during live trading (17 models x 3 classes = 51 features)
- Auto-retrains when meta_learner.joblib is missing and > 500 samples exist
- Data stored in checkpoints/meta_learner_data.npz
- Also callable during walk-forward retraining cycle

### Online Learning (Between Retrains)

- Lightweight gradient updates on neural model classification heads
- Frozen backbone, trainable head only
- Batch size: 10 labeled samples minimum
- Learning rate: 1e-4
- Bridges gap between weekly full retrains

---
## 11. Troubleshooting

### Python Startup Issues

| Symptom | Solution |
|---------|----------|
| ModuleNotFoundError | Run pip install -r requirements.txt |
| scikit-learn version error | pip install scikit-learn==1.9.0 |
| Model file not found | Place checkpoints in python_bridge_v4/checkpoints/ |
| Permission denied on CSV | Close MT5, delete stale CSV files, restart |
| CUDA not found error | Install CPU PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu |

### EA Not Executing Signals

| Symptom | Solution |
|---------|----------|
| No trades firing | Check AutoTrading is enabled (green button in MT5) |
| Signal file not found | Verify MT5 Common Files path matches config |
| Heartbeat timeout | Restart Python bridge |
| OFFLINE in EA comment | Python is not running or heartbeat stale |
| EA not on chart | Drag NeuroX_EA_v4 onto XAUUSD M1 chart |

### Connection Issues

| Symptom | Solution |
|---------|----------|
| Heartbeat OFFLINE | Check both Python and EA are running |
| CSV file locked | Another process has the file open; restart both |
| Stale tick data | EA not writing ticks; check EA is attached to chart |
| Spread file missing | EA not generating; verify EA input settings |

### Performance Issues

| Symptom | Solution |
|---------|----------|
| Slow inference (>2s) | Ensure CPU PyTorch build (not GPU polling on CPU-only machine) |
| High memory usage | Check max_ticks in TickDataConfig (default 5000) |
| Missed signals | Check console for errors; reduce cycle load |
| Models degrading | Check feature_monitor logs for degraded features |
| Frequent BROKEN edge | Market conditions unfavorable; wait for recovery |

### Common Error Messages

**RuntimeError: Found no NVIDIA driver**

Solution: Install CPU-only PyTorch:
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**ValueError: X has N features, but model expects 46**

Feature schema mismatch. Re-download latest code or retrain models with current feature set.

**FileNotFoundError: python_bridge_signal.csv**

MT5 Common Files path is incorrect. Set environment variable:
```cmd
set MT5_COMMON_PATH=C:\Users\gagan\AppData\Roaming\MetaQuotes\Terminal\Common\Files
```

**OSError: [WinError 32] The process cannot access the file**

CSV file is locked by another process. Stop both Python and EA, delete CSV files, restart.

---
## 12. Emergency Procedures

### Immediate Stop (Kill Switch)

1. **Disable AutoTrading** in MT5 (click the green button to turn red)
2. Press Ctrl+C in Python console
3. EA will stop opening new positions but continues managing existing ones

### Close All Positions Immediately

1. In MT5: Right-click any open position > Close All
2. Or: Use MT5 Trade panel > Close All button
3. The EA will write CLOSED confirmations back to Python

### System Crash Recovery

1. Check if Python is running. If not, restart with watchdog:
   ```cmd
   cd /d "D:\Automation\EA Testing\NeuroX\NeuroX_v4"
   neurox_watchdog.bat
   ```
2. The watchdog auto-restarts with exponential backoff (5s, 10s, 30s, 60s max)
3. EA continues managing existing positions independently of Python
4. Check MT5 Journal tab for any EA errors

### Data Corruption

If CSV files become corrupted or locked:

1. Stop Python bridge (Ctrl+C)
2. Disable AutoTrading in MT5
3. Delete all bridge CSV files in MT5 Common Files folder:
   ```cmd
   del "C:\Users\gagan\AppData\Roaming\MetaQuotes\Terminal\Common\Files\python_bridge_*.csv"
   del "C:\Users\gagan\AppData\Roaming\MetaQuotes\Terminal\Common\Files\python_bridge_*.txt"
   del "C:\Users\gagan\AppData\Roaming\MetaQuotes\Terminal\Common\Files\mt5_bridge_*.txt"
   ```
4. Re-enable AutoTrading
5. Restart Python bridge
6. Files will be recreated automatically on next cycle

### Edge BROKEN (Trading Auto-Paused)

When win rate drops below 30%, the brain enters BROKEN state:

1. Trading is paused automatically (no new signals)
2. One probe trade allowed every 5 minutes to test recovery
3. System auto-recovers when edge improves above COLD threshold (42%)
4. If BROKEN persists > 1 hour, consider stopping for the session
5. Check for unusual market conditions (major news, flash crash)

### Daily Loss Limit Hit ($50)

When daily loss reaches the $50 limit:

1. System automatically stops generating new signals for the day
2. Existing positions are NOT force-closed (EA continues managing them)
3. System resets at midnight via soft_reset (preserves rolling metrics)
4. Review trades in dashboard before next session
5. Consider reducing position size if limit is hit frequently

### Drawdown Circuit Breakers

| Level | Threshold | Action |
|-------|-----------|--------|
| Warning | 8% drawdown | Position size reduced to 50% |
| Critical | 15% drawdown | Full trading stop |
| Loss Streak | 3 consecutive | Reduce to 50% |
| Severe Streak | 6 consecutive | Full pause |

### MT5 Disconnection

1. Python will detect missing mt5_bridge_heartbeat.txt (timeout: 5s)
2. System logs OFFLINE warning and pauses signal generation
3. When MT5 reconnects and EA resumes heartbeat, system auto-recovers
4. No manual intervention required unless disconnection persists

---
## 13. Feature Reference

All 22 features enabled in v7.5:

| # | Feature | Config Flag | Description |
|---|---------|-------------|-------------|
| 1 | Platt Scaling | enable_platt_calibration | Regime-conditional confidence calibration (200-sample window) |
| 2 | Micro-Pullback Entry | enable_entry_timing | Adaptive timeout entry timing with $0.30 pullback target |
| 3 | Sharpe Weights | enable_sharpe_weights | Dynamic model weights based on Sharpe-ratio attribution |
| 4 | Order Flow | enable_tick_data | Bid/ask imbalance, volume delta from tick stream |
| 5 | Regime Routing | enable_regime_routing | Route model subsets by detected market regime |
| 6 | Walk-Forward Retrain | enable_walk_forward | Weekly auto-retrain with gated deployment |
| 7 | Adversarial Filter | enable_adversarial_filter | Skip signals similar to recent losers |
| 8 | Spread Gate | enable_spread_gate | Block entries when spread > 1.5x average |
| 9 | Microstructure | enable_microstructure | Tick rate, bounce rate, large order detection |
| 10 | Correlation Regime | enable_correlation_regime | DXY/bond correlation state monitoring |
| 11 | Adaptive Threshold | enable_adaptive_threshold | Dynamic confidence threshold based on accuracy |
| 12 | Disagreement Signal | enable_disagreement_signal | Reduce position size on model disagreement |
| 13 | Kelly Sizing | enable_kelly_sizing | 0.25x fractional Kelly criterion |
| 14 | Monte Carlo Risk | enable_monte_carlo_risk | 1000 sims with serial correlation |
| 15 | Data Validation | enable_data_validation | NaN, gap, price sanity, staleness checks |
| 16 | Pipeline Threading | enable_pipeline | Overlap I/O with model compute |
| 17 | Account Sync | enable_account_sync | Live balance/equity sync from MT5 |
| 18 | Slippage Tracking | enable_slippage_tracker | Fill quality and execution monitoring |
| 19 | Feature Monitor | enable_feature_monitor | Permutation importance degradation alerts |
| 20 | Online Learning | enable_online_learning | Between-retrain gradient updates on heads |
| 21 | Equity Curve Trading | enable_equity_curve_trading | Lot reduction when equity below EMA |
| 22 | A/B Testing | enable_ab_testing | Parameter variant comparison (disabled by default) |

---

## Commands Quick Reference

| Task | Command |
|------|---------|
| **Daily Run** | cd /d "D:\\Automation\\EA Testing\\NeuroX\\NeuroX_v4\\python_bridge_v4" && python -u main.py --live |
| **Watchdog** | cd /d "D:\\Automation\\EA Testing\\NeuroX\\NeuroX_v4" && neurox_watchdog.bat |
| **Verify Models** | cd /d "D:\\Automation\\EA Testing\\NeuroX\\NeuroX_v4\\python_bridge_v4" && python verify_models.py |
| **Check Version** | python -c "import main; print(main.VERSION)" |

### Full Update Command

```cmd
cd /d "D:\Automation\EA Testing\NeuroX\NeuroX_v4" && python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_run_v4.bat?t=' + str(__import__('time').time()),'neurox_run_v4.bat'); print('updated')" && neurox_run_v4.bat
```

---

## Directory Structure

```
D:\Automation\EA Testing\NeuroX\NeuroX_v4\
+-- neurox_run_v4.bat          # Update + compile + launch script
+-- neurox_watchdog.bat         # Watchdog launcher
+-- NeuroX_EA_v4.mq5           # MT5 Expert Advisor source
+-- python_bridge_v4/
    +-- main.py                # Main entry point
    +-- watchdog.py            # Process supervisor
    +-- verify_models.py       # Model verification utility
    +-- train.py               # Local training script
    +-- train_colab.py         # Colab/Kaggle training script
    +-- backtest.py            # Backtesting engine
    +-- requirements.txt       # Python dependencies
    +-- config/
    |   +-- settings.py        # All configuration dataclasses
    +-- models/
    |   +-- ensemble.py        # 17-model ensemble manager
    |   +-- transformer_model.py
    |   +-- lstm_model.py
    |   +-- tcn_model.py
    |   +-- patch_tst.py
    |   +-- tft_model.py
    |   +-- nhits_model.py
    |   +-- itransformer.py
    |   +-- mamba_model.py
    |   +-- dlinear_model.py
    |   +-- xlstm_model.py
    |   +-- timesnet_model.py
    |   +-- chronos_model.py
    |   +-- timemixer_model.py
    |   +-- softs_model.py
    |   +-- gradient_boost_extra.py
    |   +-- catboost_model.py
    +-- strategies/
    |   +-- signal_generator.py
    |   +-- trading_brain.py
    |   +-- confidence_calibrator.py
    |   +-- entry_timing.py
    |   +-- risk_manager.py
    |   +-- monte_carlo.py
    |   +-- kelly_sizing.py
    |   +-- regime_detector.py
    |   +-- regime_router.py
    |   +-- walk_forward.py
    |   +-- adversarial_filter.py
    |   +-- adaptive_threshold.py
    |   +-- correlation_regime.py
    |   +-- disagreement_signal.py
    |   +-- smart_exits.py
    |   +-- slippage_tracker.py
    |   +-- feature_monitor.py
    |   +-- ab_testing.py
    |   +-- auto_optimizer.py
    +-- data/
    |   +-- market_data.py
    |   +-- tick_data.py
    |   +-- spread_monitor.py
    |   +-- microstructure.py
    |   +-- data_validator.py
    |   +-- pipeline.py
    |   +-- sentiment.py
    |   +-- alternative_data.py
    |   +-- multi_timeframe.py
    |   +-- news_calendar.py
    +-- signals/
    |   +-- bridge.py          # MT5 CSV bridge I/O
    +-- dashboard/
    |   +-- dashboard_renderer.py
    |   +-- performance_tracker.py
    +-- training/
    |   +-- auto_retrain.py
    +-- checkpoints/           # Model weight files
    +-- logs/                  # Runtime logs
    +-- tests/                 # pytest test suite
```

---

*NeuroX v7.5 - Institutional-Grade AI Trading System*
*Last updated: 2025*
