@echo off
setlocal enabledelayedexpansion
title AI Trading EA - New PC Setup
color 0B

echo.
echo  ================================================================
echo     AI TRADING EA - ONE-CLICK SETUP
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
if not exist "python_bridge" mkdir "python_bridge"
if not exist "python_bridge\config" mkdir "python_bridge\config"
if not exist "python_bridge\data" mkdir "python_bridge\data"
if not exist "python_bridge\models" mkdir "python_bridge\models"
if not exist "python_bridge\signals" mkdir "python_bridge\signals"
if not exist "python_bridge\strategies" mkdir "python_bridge\strategies"
if not exist "python_bridge\training" mkdir "python_bridge\training"
if not exist "python_bridge\checkpoints" mkdir "python_bridge\checkpoints"
if not exist "python_bridge\dashboard" mkdir "python_bridge\dashboard"
if not exist "python_bridge\tests" mkdir "python_bridge\tests"

echo  Created: python_bridge/
echo  Created: python_bridge/config/
echo  Created: python_bridge/data/
echo  Created: python_bridge/models/
echo  Created: python_bridge/signals/
echo  Created: python_bridge/strategies/
echo  Created: python_bridge/training/
echo  Created: python_bridge/checkpoints/
echo  Created: python_bridge/dashboard/
echo  Created: python_bridge/tests/
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

set "BASE_URL=https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm"

:: Root-level files
echo  --- Root Files ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/update_and_run.bat','update_and_run.bat'); print('    update_and_run.bat OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/Python_Bridge_EA.mq5','Python_Bridge_EA.mq5'); print('    Python_Bridge_EA.mq5 OK')"
echo.

:: python_bridge root files
echo  --- python_bridge/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/main.py','python_bridge/main.py'); print('    main.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/main_multi.py','python_bridge/main_multi.py'); print('    main_multi.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/backtest.py','python_bridge/backtest.py'); print('    backtest.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/train.py','python_bridge/train.py'); print('    train.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/train_colab.py','python_bridge/train_colab.py'); print('    train_colab.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/__init__.py','python_bridge/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/requirements.txt','python_bridge/requirements.txt'); print('    requirements.txt OK')"
echo.

:: config/
echo  --- python_bridge/config/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/config/__init__.py','python_bridge/config/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/config/settings.py','python_bridge/config/settings.py'); print('    settings.py OK')"
echo.

:: data/
echo  --- python_bridge/data/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/data/__init__.py','python_bridge/data/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/data/market_data.py','python_bridge/data/market_data.py'); print('    market_data.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/data/multi_timeframe.py','python_bridge/data/multi_timeframe.py'); print('    multi_timeframe.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/data/news_calendar.py','python_bridge/data/news_calendar.py'); print('    news_calendar.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/data/sentiment.py','python_bridge/data/sentiment.py'); print('    sentiment.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/data/alternative_data.py','python_bridge/data/alternative_data.py'); print('    alternative_data.py OK')"
echo.

:: models/
echo  --- python_bridge/models/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/__init__.py','python_bridge/models/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/ensemble.py','python_bridge/models/ensemble.py'); print('    ensemble.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/tft_model.py','python_bridge/models/tft_model.py'); print('    tft_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/chronos_model.py','python_bridge/models/chronos_model.py'); print('    chronos_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/gradient_boost_extra.py','python_bridge/models/gradient_boost_extra.py'); print('    gradient_boost_extra.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/catboost_model.py','python_bridge/models/catboost_model.py'); print('    catboost_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/rl_agent.py','python_bridge/models/rl_agent.py'); print('    rl_agent.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/tcn_model.py','python_bridge/models/tcn_model.py'); print('    tcn_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/lstm_model.py','python_bridge/models/lstm_model.py'); print('    lstm_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/transformer_model.py','python_bridge/models/transformer_model.py'); print('    transformer_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/dlinear_model.py','python_bridge/models/dlinear_model.py'); print('    dlinear_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/nhits_model.py','python_bridge/models/nhits_model.py'); print('    nhits_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/patch_tst.py','python_bridge/models/patch_tst.py'); print('    patch_tst.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/itransformer.py','python_bridge/models/itransformer.py'); print('    itransformer.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/timemixer_model.py','python_bridge/models/timemixer_model.py'); print('    timemixer_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/timesnet_model.py','python_bridge/models/timesnet_model.py'); print('    timesnet_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/mamba_model.py','python_bridge/models/mamba_model.py'); print('    mamba_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/softs_model.py','python_bridge/models/softs_model.py'); print('    softs_model.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/models/xlstm_model.py','python_bridge/models/xlstm_model.py'); print('    xlstm_model.py OK')"
echo.

:: signals/
echo  --- python_bridge/signals/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/signals/__init__.py','python_bridge/signals/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/signals/bridge.py','python_bridge/signals/bridge.py'); print('    bridge.py OK')"
echo.

:: strategies/
echo  --- python_bridge/strategies/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/strategies/__init__.py','python_bridge/strategies/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/strategies/signal_generator.py','python_bridge/strategies/signal_generator.py'); print('    signal_generator.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/strategies/trading_brain.py','python_bridge/strategies/trading_brain.py'); print('    trading_brain.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/strategies/smart_exits.py','python_bridge/strategies/smart_exits.py'); print('    smart_exits.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/strategies/risk_manager.py','python_bridge/strategies/risk_manager.py'); print('    risk_manager.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/strategies/regime_detector.py','python_bridge/strategies/regime_detector.py'); print('    regime_detector.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/strategies/auto_optimizer.py','python_bridge/strategies/auto_optimizer.py'); print('    auto_optimizer.py OK')"
echo.

:: training/
echo  --- python_bridge/training/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/training/__init__.py','python_bridge/training/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/training/auto_retrain.py','python_bridge/training/auto_retrain.py'); print('    auto_retrain.py OK')"
echo.

:: dashboard/
echo  --- python_bridge/dashboard/ ---
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/dashboard/__init__.py','python_bridge/dashboard/__init__.py'); print('    __init__.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/dashboard/dashboard_renderer.py','python_bridge/dashboard/dashboard_renderer.py'); print('    dashboard_renderer.py OK')"
python -c "import urllib.request; urllib.request.urlretrieve('%BASE_URL%/python_bridge/dashboard/performance_tracker.py','python_bridge/dashboard/performance_tracker.py'); print('    performance_tracker.py OK')"
echo.

:: ============================================================
:: SECTION 5: Final Summary
:: ============================================================
color 0A
echo  [5/5] SETUP SUMMARY
echo  ================================================================
echo.
echo  FOLDER STRUCTURE:
echo    python_bridge/
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
echo    python_bridge\checkpoints\
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
echo    3. Compile Python_Bridge_EA.mq5 in MetaEditor
echo    4. Run: update_and_run.bat to start trading
echo.
echo  ================================================================
echo.

pause
