@echo off
setlocal enabledelayedexpansion
title NeuroX - New PC Setup
color 0B

echo.
echo  ================================================================
echo     NEUROX - ONE-CLICK SETUP
echo     Fresh Windows PC Configuration
echo  ================================================================
echo.

:: ============================================================
:: SECTION 1: Check Python Installation
:: ============================================================
color 0E
echo  [1/5] CHECKING PYTHON INSTALLATION...
echo  ----------------------------------------------------------------

python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo.
    echo  ERROR: Python is not installed or not in PATH!
    echo.
    echo  Please install Python 3.14 from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo  Found: %PYVER%
echo  Python is ready!
echo.

:: ============================================================
:: SECTION 2: Create Folder Structure
:: ============================================================
color 0B
echo  [2/5] CREATING FOLDER STRUCTURE...
echo  ----------------------------------------------------------------

cd /d "%~dp0"
echo  Working directory: %CD%
echo.

:: Create all needed directories
if not exist "neurox" mkdir "neurox"
if not exist "neurox\config" mkdir "neurox\config"
if not exist "neurox\data" mkdir "neurox\data"
if not exist "neurox\models" mkdir "neurox\models"
if not exist "neurox\signals" mkdir "neurox\signals"
if not exist "neurox\strategies" mkdir "neurox\strategies"
if not exist "neurox\training" mkdir "neurox\training"
if not exist "neurox\checkpoints" mkdir "neurox\checkpoints"
if not exist "neurox\dashboard" mkdir "neurox\dashboard"
if not exist "neurox\tests" mkdir "neurox\tests"

echo  Created: neurox/
echo  Created: neurox/config/
echo  Created: neurox/data/
echo  Created: neurox/models/
echo  Created: neurox/signals/
echo  Created: neurox/strategies/
echo  Created: neurox/training/
echo  Created: neurox/checkpoints/
echo  Created: neurox/dashboard/
echo  Created: neurox/tests/
echo.

:: Create MT5 Common Files directory
echo  Creating MT5 Common Files directory...
set "MT5_PATH=C:\Users\%USERNAME%\AppData\Roaming\MetaQuotes\Terminal\Common\Files"
if not exist "%MT5_PATH%" (
    mkdir "%MT5_PATH%"
    echo  Created: %MT5_PATH%
) else (
    echo  Already exists: %MT5_PATH%
)
echo.

:: ============================================================
:: SECTION 3: Install Python Dependencies
:: ============================================================
color 0D
echo  [3/5] INSTALLING PYTHON DEPENDENCIES...
echo  ----------------------------------------------------------------
echo.

:: Install CPU-only PyTorch first (special index URL)
echo  Installing PyTorch (CPU-only)...
pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    color 0E
    echo  WARNING: PyTorch install had issues, continuing...
    color 0D
)
echo.

:: Install transformers
echo  Installing transformers...
pip install transformers>=4.30.0
if errorlevel 1 (
    echo  WARNING: transformers install had issues, continuing...
)

:: Install scikit-learn (pinned version - MUST match training)
echo  Installing scikit-learn==1.9.0 (pinned to match training)...
pip install scikit-learn==1.9.0
if errorlevel 1 (
    echo  WARNING: scikit-learn install had issues, continuing...
)

:: Install remaining dependencies one by one so failures don't block others
echo  Installing remaining dependencies...
echo.

for %%P in (
    "pandas>=1.5.0"
    "numpy>=1.24.0"
    "scipy>=1.10.0"
    "lightgbm>=4.0.0"
    "catboost>=1.2.0"
    "xgboost>=2.0.0"
    "hmmlearn>=0.3.0"
    "joblib>=1.3.0"
    "ta>=0.10.0"
    "yfinance>=0.2.0"
    "feedparser>=6.0.0"
    "beautifulsoup4>=4.12.0"
    "requests>=2.28.0"
    "websocket-client>=1.5.0"
    "schedule>=1.2.0"
    "pytest>=7.3.0"
) do (
    echo  Installing %%~P...
    pip install %%~P >nul 2>&1
    if errorlevel 1 (
        echo    WARNING: %%~P had issues
    ) else (
        echo    OK
    )
)
echo.

:: ============================================================
:: SECTION 4: Download EA Files from GitHub
:: ============================================================
color 0A
echo  [4/5] DOWNLOADING EA FILES FROM GITHUB...
echo  ----------------------------------------------------------------
echo.

set "BASE_URL=https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX"

:: Root-level files
echo  --- Root Files ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox_run.bat','neurox_run.bat'); print('    neurox_run.bat OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/NeuroX_EA.mq5','NeuroX_EA.mq5'); print('    NeuroX_EA.mq5 OK')"
echo.

:: neurox root files
echo  --- neurox/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/main.py','neurox/main.py'); print('    main.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/main_multi.py','neurox/main_multi.py'); print('    main_multi.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/backtest.py','neurox/backtest.py'); print('    backtest.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/train.py','neurox/train.py'); print('    train.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/train_colab.py','neurox/train_colab.py'); print('    train_colab.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/__init__.py','neurox/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/requirements.txt','neurox/requirements.txt'); print('    requirements.txt OK')"
echo.

:: config/
echo  --- neurox/config/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/config/__init__.py','neurox/config/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/config/settings.py','neurox/config/settings.py'); print('    settings.py OK')"
echo.

:: data/
echo  --- neurox/data/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/data/__init__.py','neurox/data/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/data/market_data.py','neurox/data/market_data.py'); print('    market_data.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/data/multi_timeframe.py','neurox/data/multi_timeframe.py'); print('    multi_timeframe.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/data/news_calendar.py','neurox/data/news_calendar.py'); print('    news_calendar.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/data/sentiment.py','neurox/data/sentiment.py'); print('    sentiment.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/data/alternative_data.py','neurox/data/alternative_data.py'); print('    alternative_data.py OK')"
echo.

:: models/
echo  --- neurox/models/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/__init__.py','neurox/models/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/ensemble.py','neurox/models/ensemble.py'); print('    ensemble.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/tft_model.py','neurox/models/tft_model.py'); print('    tft_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/chronos_model.py','neurox/models/chronos_model.py'); print('    chronos_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/gradient_boost_extra.py','neurox/models/gradient_boost_extra.py'); print('    gradient_boost_extra.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/catboost_model.py','neurox/models/catboost_model.py'); print('    catboost_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/rl_agent.py','neurox/models/rl_agent.py'); print('    rl_agent.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/tcn_model.py','neurox/models/tcn_model.py'); print('    tcn_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/lstm_model.py','neurox/models/lstm_model.py'); print('    lstm_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/transformer_model.py','neurox/models/transformer_model.py'); print('    transformer_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/dlinear_model.py','neurox/models/dlinear_model.py'); print('    dlinear_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/nhits_model.py','neurox/models/nhits_model.py'); print('    nhits_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/patch_tst.py','neurox/models/patch_tst.py'); print('    patch_tst.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/itransformer.py','neurox/models/itransformer.py'); print('    itransformer.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/timemixer_model.py','neurox/models/timemixer_model.py'); print('    timemixer_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/timesnet_model.py','neurox/models/timesnet_model.py'); print('    timesnet_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/mamba_model.py','neurox/models/mamba_model.py'); print('    mamba_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/softs_model.py','neurox/models/softs_model.py'); print('    softs_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/models/xlstm_model.py','neurox/models/xlstm_model.py'); print('    xlstm_model.py OK')"
echo.

:: signals/
echo  --- neurox/signals/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/signals/__init__.py','neurox/signals/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/signals/bridge.py','neurox/signals/bridge.py'); print('    bridge.py OK')"
echo.

:: strategies/
echo  --- neurox/strategies/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/strategies/__init__.py','neurox/strategies/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/strategies/signal_generator.py','neurox/strategies/signal_generator.py'); print('    signal_generator.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/strategies/trading_brain.py','neurox/strategies/trading_brain.py'); print('    trading_brain.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/strategies/smart_exits.py','neurox/strategies/smart_exits.py'); print('    smart_exits.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/strategies/risk_manager.py','neurox/strategies/risk_manager.py'); print('    risk_manager.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/strategies/regime_detector.py','neurox/strategies/regime_detector.py'); print('    regime_detector.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/strategies/auto_optimizer.py','neurox/strategies/auto_optimizer.py'); print('    auto_optimizer.py OK')"
echo.

:: training/
echo  --- neurox/training/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/training/__init__.py','neurox/training/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/training/auto_retrain.py','neurox/training/auto_retrain.py'); print('    auto_retrain.py OK')"
echo.

:: dashboard/
echo  --- neurox/dashboard/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/dashboard/__init__.py','neurox/dashboard/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/dashboard/dashboard_renderer.py','neurox/dashboard/dashboard_renderer.py'); print('    dashboard_renderer.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/neurox/dashboard/performance_tracker.py','neurox/dashboard/performance_tracker.py'); print('    performance_tracker.py OK')"
echo.

:: ============================================================
:: SECTION 5: Final Summary
:: ============================================================
color 0A
echo  [5/5] SETUP SUMMARY
echo  ================================================================
echo.
echo  FOLDER STRUCTURE:
echo    neurox/
echo      config/settings.py
echo      data/market_data.py, multi_timeframe.py, news_calendar.py, sentiment.py
echo      models/ensemble.py, tft_model.py, chronos_model.py, tcn_model.py, ...
echo      signals/bridge.py
echo      strategies/signal_generator.py, trading_brain.py, smart_exits.py
echo      training/auto_retrain.py
echo      dashboard/dashboard_renderer.py, performance_tracker.py
echo      checkpoints/  (YOUR MODELS GO HERE)
echo      main.py
echo      requirements.txt
echo.
echo  MT5 SIGNAL PATH:
echo    %MT5_PATH%
echo.
echo  ================================================================
color 0C
echo.
echo  !!! IMPORTANT - ACTION REQUIRED !!!
echo.
echo  You MUST copy your trained model checkpoints from the training
echo  machine into:
echo.
echo    neurox\checkpoints\
echo.
echo  Required files (from your training PC):
echo    - best_model.pt (main ensemble weights)
echo    - lgbm_model.txt (LightGBM model)
echo    - catboost_model.cbm (CatBoost model)
echo    - xgb_model.json (XGBoost model)
echo    - feature_scaler.pkl (scikit-learn scaler)
echo    - Any other .pt/.pkl/.txt/.cbm files from training
echo.
echo  Without these files, the EA will not generate predictions!
echo.
color 0A
echo  ================================================================
echo.
echo  SETUP COMPLETE! 
echo.
echo  Next steps:
echo    1. Copy your model checkpoints (see above)
echo    2. Open MetaTrader 5
echo    3. Compile NeuroX_EA.mq5 in MetaEditor
echo    4. Run: neurox_run.bat to start trading
echo.
echo  ================================================================
echo.

pause
