@echo off
color 0D
title NeuroX v7.4
SET PYTHONUNBUFFERED=1
SET PYTHONIOENCODING=utf-8

echo ============================================
echo   NeuroX v7.4 - Update, Compile, Run
echo   Platt scaling, micro-pullback, Sharpe wts
echo ============================================
echo.

cd /d "%~dp0"
echo Dir: %CD%
echo.

:: Create target folders if they don't exist
if not exist "python_bridge_v4" mkdir python_bridge_v4
if not exist "python_bridge_v4\config" mkdir python_bridge_v4\config
if not exist "python_bridge_v4\models" mkdir python_bridge_v4\models
if not exist "python_bridge_v4\strategies" mkdir python_bridge_v4\strategies
if not exist "python_bridge_v4\data" mkdir python_bridge_v4\data
if not exist "python_bridge_v4\signals" mkdir python_bridge_v4\signals

:: ══════════════════════════════════════════════
:: STEP 1: Download latest files from GitHub
:: ══════════════════════════════════════════════
echo [1/3] Downloading latest files...
echo.

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/main.py','python_bridge_v4/main.py'); print('  main.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/config/settings.py','python_bridge_v4/config/settings.py'); print('  config/settings.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/ensemble.py','python_bridge_v4/models/ensemble.py'); print('  models/ensemble.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/tft_model.py','python_bridge_v4/models/tft_model.py'); print('  models/tft_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/chronos_model.py','python_bridge_v4/models/chronos_model.py'); print('  models/chronos_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/gradient_boost_extra.py','python_bridge_v4/models/gradient_boost_extra.py'); print('  models/gradient_boost_extra.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/catboost_model.py','python_bridge_v4/models/catboost_model.py'); print('  models/catboost_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/signal_generator.py','python_bridge_v4/strategies/signal_generator.py'); print('  strategies/signal_generator.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/trading_brain.py','python_bridge_v4/strategies/trading_brain.py'); print('  strategies/trading_brain.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/confidence_calibrator.py','python_bridge_v4/strategies/confidence_calibrator.py'); print('  strategies/confidence_calibrator.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/entry_timing.py','python_bridge_v4/strategies/entry_timing.py'); print('  strategies/entry_timing.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/data/market_data.py','python_bridge_v4/data/market_data.py'); print('  data/market_data.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/data/multi_timeframe.py','python_bridge_v4/data/multi_timeframe.py'); print('  data/multi_timeframe.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/signals/bridge.py','python_bridge_v4/signals/bridge.py'); print('  signals/bridge.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/NeuroX_EA_v4.mq5','NeuroX_EA_v4.mq5'); print('  NeuroX_EA_v4.mq5 OK')"

echo.
echo   All files downloaded.
echo.

:: Apply patches
python -c "f=open('python_bridge_v4/strategies/trading_brain.py','r',encoding='utf-8').read(); f=f.replace('rp[\"rr\"]','rp.get(\"tp_rr\",rp.get(\"rr\",0.0))'); open('python_bridge_v4/strategies/trading_brain.py','w',encoding='utf-8').write(f)"

:: ══════════════════════════════════════════════
:: STEP 2: Compile EA in MetaEditor
:: ══════════════════════════════════════════════
echo [2/3] Compiling EA...

set "MT5_EXPERTS=C:\Users\gagan\AppData\Roaming\MetaQuotes\Terminal\EE1261C89A64D41685651B738DC52A84\MQL5\Experts\Advisors"
set "MT5_IMAGES=C:\Users\gagan\AppData\Roaming\MetaQuotes\Terminal\EE1261C89A64D41685651B738DC52A84\MQL5\Images"
set "METAEDITOR="

if exist "C:\Program Files\Fusion Markets MetaTrader 5\metaeditor64.exe" (
    set "METAEDITOR=C:\Program Files\Fusion Markets MetaTrader 5\metaeditor64.exe"
) else if exist "C:\Program Files (x86)\Fusion Markets MetaTrader 5\metaeditor64.exe" (
    set "METAEDITOR=C:\Program Files (x86)\Fusion Markets MetaTrader 5\metaeditor64.exe"
)

if defined METAEDITOR (
    copy /Y "%~dp0NeuroX_EA_v4.mq5" "%MT5_EXPERTS%\" >nul
    if exist "%~dp0neurox_logo.bmp" copy /Y "%~dp0neurox_logo.bmp" "%MT5_IMAGES%\" >nul
    "%METAEDITOR%" /compile:"%MT5_EXPERTS%\NeuroX_EA_v4.mq5" /log
    echo   EA compiled. Refresh Navigator in MT5.
) else (
    echo   MetaEditor not found - skipping compile.
    echo   Manually compile NeuroX_EA_v4.mq5 in MetaEditor.
)
echo.

:: ══════════════════════════════════════════════
:: STEP 3: Start Python bridge
:: ══════════════════════════════════════════════
echo [3/3] Starting NeuroX v7.4...
echo ============================================
echo.

cd python_bridge_v4
python -u main.py --live

pause
