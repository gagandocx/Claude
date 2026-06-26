@echo off
title EA Update and Run
color 0A
setlocal

set REPO=https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm
set PB=python_bridge

echo ============================================
echo   EA Update and Run Script
echo ============================================
echo.

cd /d "%~dp0"
echo Working directory: %CD%
echo.

REM ── Step 1: Self-update this batch file ──────────────────────────────
echo [0/7] Self-updating script...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO%/update_and_run.bat','update_and_run.bat')" 2>nul
echo   update_and_run.bat refreshed

REM ── Step 2: Download all updated Python files ────────────────────────
echo [1/7] Trading Brain...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO%/%PB%/strategies/trading_brain.py','%PB%/strategies/trading_brain.py'); print('  OK')"

echo [2/7] Settings...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO%/%PB%/config/settings.py','%PB%/config/settings.py'); print('  OK')"

echo [3/7] Main...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO%/%PB%/main.py','%PB%/main.py'); print('  OK')"

echo [4/7] TFT model fix...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO%/%PB%/models/tft_model.py','%PB%/models/tft_model.py'); print('  OK')"

echo [5/7] LightGBM model...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO%/%PB%/models/gradient_boost_extra.py','%PB%/models/gradient_boost_extra.py'); print('  OK')"

echo [6/7] CatBoost model...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO%/%PB%/models/catboost_model.py','%PB%/models/catboost_model.py'); print('  OK')"

echo [7/7] Ensemble...
python -c "import urllib.request; urllib.request.urlretrieve('%REPO%/%PB%/models/ensemble.py','%PB%/models/ensemble.py'); print('  OK')"

echo.
echo All files updated. Starting EA...
echo ============================================
echo.

echo Verifying Trading Brain...
python -c "import sys; sys.path.insert(0,'python_bridge'); from strategies.trading_brain import TradingBrain; from config.settings import BrainConfig; b=TradingBrain(BrainConfig()); print('  Brain: OK ^| min_conf=' + str(b.config.base_min_confidence) + ' ^| min_win_prob=' + str(b.config.min_win_probability) + ' ^| daily_limit=$' + str(b.config.daily_loss_limit))"
echo.

cd %PB%
python main.py --live

pause
