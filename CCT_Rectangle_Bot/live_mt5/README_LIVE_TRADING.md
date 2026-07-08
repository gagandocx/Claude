# CCT Rectangle Bot - Live MT5 Trading Guide

## Overview

This module enables the CCT Rectangle Bot to trade live on MetaTrader 5 (MT5). It reuses the proven strategy logic from the backtester and adds real-time data fetching, order execution, risk management, and safety controls.

**IMPORTANT: Always start with a demo account. Only switch to live trading after extensive demo testing.**

---

## Prerequisites

- **Operating System:** Windows 10 or Windows 11 (MT5 Python package is Windows-only)
- **Python:** Version 3.9, 3.10, or 3.11 (recommended: 3.11)
- **MetaTrader 5 Terminal:** Installed and running
- **Broker Account:** Demo or live account with a broker that supports MT5

---

## Step 1: Install MetaTrader 5 Terminal

1. Download MT5 from [MetaQuotes](https://www.metatrader5.com/en/download) or your broker's website
2. Install and launch the terminal
3. Create a **demo account** first:
   - File > Open an Account
   - Select your broker's demo server
   - Choose "Open a demo account"
   - Note your account number, password, and server name

---

## Step 2: Set Up Python Environment

```bash
# Create a virtual environment (recommended)
python -m venv mt5_env
mt5_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Required packages:
- `MetaTrader5>=5.0.45` - Python interface to MT5 terminal
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.23.0` - Numerical operations
- `schedule>=1.2.0` - Task scheduling
- `ta>=0.10.0` - Technical analysis indicators

---

## Step 3: Configure the Bot

Edit `mt5_config.py` with your account details:

```python
# Your MT5 credentials
MT5_ACCOUNT = 12345678          # Your account number
MT5_PASSWORD = "your_password"  # Your account password
MT5_SERVER = "YourBroker-Demo"  # Your broker's server name

# Optional: path to MT5 terminal if not in default location
MT5_PATH = ""  # e.g., "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
```

### Key Settings to Review

| Setting | Default | Description |
|---------|---------|-------------|
| `DEMO_MODE` | `True` | Safety flag - keep True until ready for live |
| `SYMBOL` | `XAUUSD` | Trading symbol (Gold) |
| `LOT_SIZE` | `0.01` | Base lot size (micro lot) |
| `RISK_PER_TRADE` | `0.01` | 1% risk per trade |
| `MAX_DAILY_LOSS_PCT` | `0.03` | Stop trading after 3% daily loss |
| `MAX_TRADES_PER_DAY` | `5` | Maximum 5 trades per day |
| `MAX_DRAWDOWN_PCT` | `0.05` | Halt if 5% drawdown from peak |

---

## Step 4: Run on Demo Account

1. Make sure MT5 terminal is running and logged in
2. Ensure `DEMO_MODE = True` in `mt5_config.py`
3. Run the bot:

```bash
cd CCT_Rectangle_Bot/live_mt5
python mt5_trader.py
```

4. The bot will:
   - Connect to your MT5 terminal
   - Verify the symbol is available
   - Start scanning for signals every 60 seconds
   - Display a dashboard with account info and status

5. Let it run on demo for at least 1-2 weeks to verify:
   - Signals are detected correctly
   - Orders are executed properly
   - Risk management works as expected
   - No unexpected errors occur

---

## Step 5: Switch to Live Trading

**Only after successful demo testing:**

1. Change credentials in `mt5_config.py` to your live account
2. Set `DEMO_MODE = False`
3. Review and possibly reduce risk settings:
   - Consider `RISK_PER_TRADE = 0.005` (0.5%) for live
   - Keep `MAX_DAILY_LOSS_PCT` at 3% or lower
4. Run the bot - it will ask for confirmation before starting in live mode

---

## Architecture

```
CCT_Rectangle_Bot/
|-- strategy.py          <-- Strategy logic (imported by live trader)
|-- config.py            <-- Backtest config (NOT used for live)
|-- utils.py             <-- Utility functions (imported by strategy)
|-- live_mt5/
    |-- mt5_trader.py    <-- Main loop (entry point)
    |-- mt5_connector.py <-- MT5 connection & order execution
    |-- mt5_data_feed.py <-- Live OHLCV data from MT5
    |-- mt5_config.py    <-- Live trading configuration
    |-- risk_manager.py  <-- Safety controls & position sizing
    |-- dashboard.py     <-- Console status display
    |-- logger_setup.py  <-- Logging configuration
    |-- logs/            <-- Log files (auto-created)
```

### How It Works

1. **mt5_trader.py** starts the main loop (60-second polling interval)
2. Each cycle fetches fresh 4H, 15M, and 1M candle data via **mt5_data_feed.py**
3. Data is passed to `CCTRectangleStrategy` from the parent `strategy.py`
4. If a new signal is detected, **risk_manager.py** checks all safety limits
5. If approved, **mt5_connector.py** executes the market order on MT5
6. Existing positions are monitored for trailing stop activation
7. **dashboard.py** displays current status on the console

---

## Troubleshooting

### Common Issues

**"MetaTrader5 package not available"**
- You must run this on Windows. The MT5 Python package does not work on macOS or Linux.
- Install with: `pip install MetaTrader5`

**"MT5 initialization failed"**
- Make sure the MT5 terminal is running
- Check if `MT5_PATH` is correct (if specified)
- Try restarting the MT5 terminal

**"MT5 login failed"**
- Verify account number, password, and server in `mt5_config.py`
- Make sure you can log in manually via the MT5 terminal
- Check if your broker's server is available

**"Symbol XAUUSD not found"**
- The symbol name varies by broker (could be `GOLD`, `XAUUSDm`, etc.)
- Check Market Watch in MT5 for the exact symbol name
- Update `SYMBOL` in `mt5_config.py`

**"Spread too high"**
- Spread exceeds `MAX_SPREAD_POINTS` in config
- This is a safety feature - trades are skipped during high spread periods
- Increase the limit if needed, but be cautious

**"Order failed: Invalid stops"**
- Stop loss or take profit is too close to current price
- Some brokers have minimum stop distance requirements
- Check your broker's contract specifications for the symbol

**Connection drops**
- The bot has automatic reconnection with exponential backoff
- Check your internet connection
- Make sure MT5 terminal stays running (disable auto-sleep)

### Log Files

Check `live_mt5/logs/` for detailed logs:
- Log files are named `cct_live_YYYYMMDD.log`
- Contains all trading decisions, errors, and status updates
- Rotates at 10MB with 5 backup files kept

---

## Safety Warnings

1. **Trading involves significant risk of loss.** Only trade with money you can afford to lose.
2. **Past backtest performance does not guarantee future results.** Market conditions change.
3. **Always start with demo.** Never go live without extensive demo testing.
4. **Monitor the bot.** Do not leave it unattended for extended periods initially.
5. **Use conservative settings.** The default 1% risk is appropriate for live trading. The 25% used in backtesting is for simulation only.
6. **Keep the MT5 terminal running.** The bot requires an active MT5 terminal connection.
7. **Network stability matters.** Use a reliable internet connection. Consider a VPS for 24/7 operation.

---

## Running as a Service (Advanced)

For unattended operation, consider:

1. **Windows Task Scheduler** - Set up auto-restart on crash
2. **VPS** - Run on a Windows VPS close to your broker's servers
3. **Watchdog script** - Monitor the bot process and restart if it dies

Example watchdog (save as `watchdog.bat`):
```batch
@echo off
:loop
python mt5_trader.py
echo Bot stopped. Restarting in 10 seconds...
timeout /t 10
goto loop
```

---

## Support

- Strategy documentation: See parent `CCT_Rectangle_Bot/` directory
- MT5 Python docs: https://www.mql5.com/en/docs/python_metatrader5
- MetaTrader 5 help: https://www.metatrader5.com/en/terminal/help
