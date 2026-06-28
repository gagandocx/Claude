@echo off
color 0B
title NeuroX v8.0
SET PYTHONUNBUFFERED=1
SET PYTHONIOENCODING=utf-8

echo ============================================
echo   NeuroX v8.0
echo   Institutional-Grade Trading System
echo   Update and Run
echo ============================================
echo.

cd /d "%~dp0"
echo Dir: %CD%
echo.

echo [1/4] Setting up neurox_v3...

if not exist neurox_v3 (
    echo   Cloning neurox folder to neurox_v3...
    xcopy /E /I neurox neurox_v3 >nul
    echo   Clone complete - all checkpoints and models carried over
) else (
    echo   neurox_v3 already exists - skipping clone
)
echo.

echo [2/4] Downloading v3-specific files ^(new + modified only^)...

REM --- New institutional modules (don't exist in neurox/) ---

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/data/tick_data.py','neurox_v3/data/tick_data.py'); print('  data/tick_data.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/data/microstructure.py','neurox_v3/data/microstructure.py'); print('  data/microstructure.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/data/spread_monitor.py','neurox_v3/data/spread_monitor.py'); print('  data/spread_monitor.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/strategies/regime_router.py','neurox_v3/strategies/regime_router.py'); print('  strategies/regime_router.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/strategies/walk_forward.py','neurox_v3/strategies/walk_forward.py'); print('  strategies/walk_forward.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/strategies/adversarial_filter.py','neurox_v3/strategies/adversarial_filter.py'); print('  strategies/adversarial_filter.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/strategies/adaptive_threshold.py','neurox_v3/strategies/adaptive_threshold.py'); print('  strategies/adaptive_threshold.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/strategies/correlation_regime.py','neurox_v3/strategies/correlation_regime.py'); print('  strategies/correlation_regime.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/strategies/disagreement_signal.py','neurox_v3/strategies/disagreement_signal.py'); print('  strategies/disagreement_signal.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/strategies/kelly_sizing.py','neurox_v3/strategies/kelly_sizing.py'); print('  strategies/kelly_sizing.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/strategies/monte_carlo.py','neurox_v3/strategies/monte_carlo.py'); print('  strategies/monte_carlo.py OK')"

REM --- Modified files (v3 versions overwrite the cloned v1 copies) ---

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/main.py','neurox_v3/main.py'); print('  main.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/config/settings.py','neurox_v3/config/settings.py'); print('  config/settings.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/strategies/signal_generator.py','neurox_v3/strategies/signal_generator.py'); print('  strategies/signal_generator.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/strategies/trading_brain.py','neurox_v3/strategies/trading_brain.py'); print('  strategies/trading_brain.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/models/ensemble.py','neurox_v3/models/ensemble.py'); print('  models/ensemble.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/signals/bridge.py','neurox_v3/signals/bridge.py'); print('  signals/bridge.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v3/requirements.txt','neurox_v3/requirements.txt'); print('  requirements.txt OK')"

echo.
echo [3/4] Downloading EA file...

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/NeuroX_EA_v3.mq5','NeuroX_EA_v3.mq5'); print('  NeuroX_EA_v3.mq5 OK  -- recompile in MetaEditor if updated')"

echo.
echo [4/4] Installing dependencies...

cd neurox_v3
pip install -r requirements.txt --quiet 2>nul
echo   dependencies OK

echo.
echo ============================================
echo   All files updated. Starting NeuroX v8.0...
echo ============================================
echo.

python -u main.py --live

echo.
echo Bridge stopped. Press any key to exit...
pause >nul
