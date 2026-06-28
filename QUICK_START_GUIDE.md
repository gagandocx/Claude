# PYTHON ML BRIDGE - QUICK START GUIDE
# All Commands You Need (Save This File!)

---

## SETUP (One Time)

### Install Python Dependencies
```bash
cd (your folder)\python_bridge
pip install -r requirements.txt
```

### Train the Models (3-4 hours, run overnight)
```bash
cd (your folder)\python_bridge
python train.py
```

---

## RUNNING THE SYSTEM

### Gold (XAUUSD) - Main EA
```bash
cd (your folder)\python_bridge
python main.py --live
```

### EURUSD (or any forex pair)
```bash
cd (your folder)\python_bridge
python main_multi.py --live --symbol EURUSD
```

### Other Pairs
```bash
python main_multi.py --live --symbol GBPUSD
python main_multi.py --live --symbol USDJPY
python main_multi.py --live --symbol AUDUSD
python main_multi.py --live --symbol USDCAD
python main_multi.py --live --symbol NZDUSD
```

---

## METATRADER 5 SETUP

### EA Files (copy to MT5 Experts folder)
- `Python_Bridge_EA.mq5` → For XAUUSD (Gold)
- `Python_Bridge_EA_Multi.mq5` → For any other pair
- `AI_Adaptive_EA.mq5` → Standalone (no Python needed)

### Find your MT5 Experts folder:
```
File → Open Data Folder → MQL5 → Experts
```

### Steps:
1. Open MetaEditor (press F4 in MT5)
2. Open the .mq5 file → Compile (F7)
3. Go back to MT5
4. Open chart (XAUUSD M1 or EURUSD M1)
5. Drag EA from Navigator onto chart
6. Enable AutoTrading (green button in toolbar)
7. Check "Allow Algo Trading" in EA settings

---

## HOW TO VERIFY CONNECTION

### Python side should show:
```
Loaded model checkpoints from ...
Starting main loop...
--- Cycle 1 ---
[SignalGen] SIGNAL GENERATED: BUY XAUUSD conf=0.35 lot=0.01
```

### MT5 Experts tab should show:
```
[PythonBridge] EA initialized. Magic=20240115
[PythonBridge] Signal file: python_bridge_signal.csv
[PythonBridge] Parsed signal - Action=BUY
```

### Check signal files exist at:
```
C:\Users\gagan\AppData\Roaming\MetaQuotes\Terminal\Common\Files\
```
Files: `python_bridge_signal.csv`, `python_bridge_heartbeat.txt`

---

## IMPORTANT NOTES

### Checkpoints Folder
- Location: `python_bridge\checkpoints\`
- Contains trained model weights (took 3-4 hours to train)
- ALWAYS copy this folder when updating to new version
- Without it, models are untrained and useless

### Files in Checkpoints:
- `transformer.pth` - Transformer model
- `lstm.pth` - LSTM model
- `gradient_boost.joblib` - Gradient Boosting
- `meta_learner.joblib` - Ensemble combiner

### Auto-Optimizer State
- File: `python_bridge\auto_optimizer_state.json`
- Contains learned parameter settings
- Also copy this when updating (preserves what it learned)

---

## SYSTEM BEHAVIOR

### What Happens Automatically:
- Every 10 seconds: Scans market, generates signal if conditions align
- Every tick: Manages open positions (trailing SL, momentum exit)
- Every 50 trades: Auto-optimizer tunes all parameters
- Every weekend: Auto-retrains models on latest data
- Always: RL agent learns from every closed trade

### Self-Tuning Parameters (auto-adjusts):
- SL distance ($1-$5)
- Session sizing (Asian/London/NY)
- Confidence threshold
- Momentum lookback (3-7 bars)
- RSI levels
- Trailing distances
- Cooldown between trades
- Max open positions

---

## SAFETY FEATURES

- $50 daily loss cap → stops trading for the day
- Max 5 positions at a time
- Emergency close-all if floating loss > $50
- News filter blocks trading during NFP/FOMC/CPI
- Lose streak detection → halves lot size after 3 losses
- Auto-rollback if new parameters perform 20% worse

---

## TROUBLESHOOTING

### "No model checkpoints found"
→ Copy your `checkpoints/` folder into `python_bridge/`

### "Loaded checkpoints" but size mismatch error
→ Need to retrain: `python train.py`

### Signals generating but no trades in MT5
→ Check: AutoTrading enabled? EA attached? Experts tab for errors?

### "NEWS FILTER: Skipping cycle"
→ Normal! High-impact news event nearby. Will resume in 2-5 min.

### EA says "SELL FAILED" or "BUY FAILED"
→ Check: Sufficient margin? Market open? Correct symbol on chart?

### Python terminal shows only "--- Cycle N ---" with no signals
→ Weekend/market closed OR momentum is flat (no clear direction)

---

## DOWNLOAD LINKS

### Full System (ZIP):
https://github.com/gagandocx/Claude/archive/refs/heads/feature/ai-adaptive-ea.zip

### Individual Files:
- AI_Adaptive_EA.mq5: https://github.com/gagandocx/Claude/raw/feature/ai-adaptive-ea/AI_Adaptive_EA.mq5
- Python_Bridge_EA.mq5: https://github.com/gagandocx/Claude/raw/feature/ai-adaptive-ea/Python_Bridge_EA.mq5
- Python_Bridge_EA_Multi.mq5: https://github.com/gagandocx/Claude/raw/feature/ai-adaptive-ea/Python_Bridge_EA_Multi.mq5

---

## RECOMMENDED WORKFLOW

1. Start `python train.py` overnight (one time, 3-4 hours)
2. Next day: `python main.py --live` on DEMO account
3. Let it trade 50+ trades (auto-optimizer needs data)
4. After 1-2 weeks profitable on demo → go live with 0.01 lots
5. Never touch parameters — the system tunes itself
6. Keep Python terminal running 24/5 (use a VPS for 24/7)

---

## MAGIC NUMBERS (Don't Change Unless You Know Why)
- Gold EA: 20240115
- Multi EA: 20240116
- AI Adaptive EA: 777888

Each EA uses a unique magic number so they don't interfere with each other.
You can run all 3 simultaneously on different charts.
