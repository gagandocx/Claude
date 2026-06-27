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
if not exist "python_bridge_v4\dashboard" mkdir python_bridge_v4\dashboard
if not exist "python_bridge_v4\training" mkdir python_bridge_v4\training
if not exist "python_bridge_v4\tests" mkdir python_bridge_v4\tests

:: ══════════════════════════════════════════════
:: STEP 1: Download latest files from GitHub
:: ══════════════════════════════════════════════
echo [1/3] Downloading latest files...
echo.

:: --- Root files ---
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/__init__.py','python_bridge_v4/__init__.py'); print('  __init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/main.py','python_bridge_v4/main.py'); print('  main.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/main_multi.py','python_bridge_v4/main_multi.py'); print('  main_multi.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/backtest.py','python_bridge_v4/backtest.py'); print('  backtest.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/train.py','python_bridge_v4/train.py'); print('  train.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/train_colab.py','python_bridge_v4/train_colab.py'); print('  train_colab.py OK')"

:: --- config/ ---
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/config/__init__.py','python_bridge_v4/config/__init__.py'); print('  config/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/config/settings.py','python_bridge_v4/config/settings.py'); print('  config/settings.py OK')"

:: --- data/ ---
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/data/__init__.py','python_bridge_v4/data/__init__.py'); print('  data/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/data/alternative_data.py','python_bridge_v4/data/alternative_data.py'); print('  data/alternative_data.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/data/market_data.py','python_bridge_v4/data/market_data.py'); print('  data/market_data.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/data/multi_timeframe.py','python_bridge_v4/data/multi_timeframe.py'); print('  data/multi_timeframe.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/data/news_calendar.py','python_bridge_v4/data/news_calendar.py'); print('  data/news_calendar.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/data/sentiment.py','python_bridge_v4/data/sentiment.py'); print('  data/sentiment.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/data/tick_data.py','python_bridge_v4/data/tick_data.py'); print('  data/tick_data.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/data/spread_monitor.py','python_bridge_v4/data/spread_monitor.py'); print('  data/spread_monitor.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/data/microstructure.py','python_bridge_v4/data/microstructure.py'); print('  data/microstructure.py OK')"

:: --- models/ ---
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/__init__.py','python_bridge_v4/models/__init__.py'); print('  models/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/catboost_model.py','python_bridge_v4/models/catboost_model.py'); print('  models/catboost_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/chronos_model.py','python_bridge_v4/models/chronos_model.py'); print('  models/chronos_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/dlinear_model.py','python_bridge_v4/models/dlinear_model.py'); print('  models/dlinear_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/ensemble.py','python_bridge_v4/models/ensemble.py'); print('  models/ensemble.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/gradient_boost_extra.py','python_bridge_v4/models/gradient_boost_extra.py'); print('  models/gradient_boost_extra.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/itransformer.py','python_bridge_v4/models/itransformer.py'); print('  models/itransformer.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/lstm_model.py','python_bridge_v4/models/lstm_model.py'); print('  models/lstm_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/mamba_model.py','python_bridge_v4/models/mamba_model.py'); print('  models/mamba_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/nhits_model.py','python_bridge_v4/models/nhits_model.py'); print('  models/nhits_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/patch_tst.py','python_bridge_v4/models/patch_tst.py'); print('  models/patch_tst.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/rl_agent.py','python_bridge_v4/models/rl_agent.py'); print('  models/rl_agent.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/softs_model.py','python_bridge_v4/models/softs_model.py'); print('  models/softs_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/tcn_model.py','python_bridge_v4/models/tcn_model.py'); print('  models/tcn_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/tft_model.py','python_bridge_v4/models/tft_model.py'); print('  models/tft_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/timemixer_model.py','python_bridge_v4/models/timemixer_model.py'); print('  models/timemixer_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/timesnet_model.py','python_bridge_v4/models/timesnet_model.py'); print('  models/timesnet_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/transformer_model.py','python_bridge_v4/models/transformer_model.py'); print('  models/transformer_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/models/xlstm_model.py','python_bridge_v4/models/xlstm_model.py'); print('  models/xlstm_model.py OK')"

:: --- strategies/ ---
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/__init__.py','python_bridge_v4/strategies/__init__.py'); print('  strategies/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/auto_optimizer.py','python_bridge_v4/strategies/auto_optimizer.py'); print('  strategies/auto_optimizer.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/confidence_calibrator.py','python_bridge_v4/strategies/confidence_calibrator.py'); print('  strategies/confidence_calibrator.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/entry_timing.py','python_bridge_v4/strategies/entry_timing.py'); print('  strategies/entry_timing.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/regime_detector.py','python_bridge_v4/strategies/regime_detector.py'); print('  strategies/regime_detector.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/risk_manager.py','python_bridge_v4/strategies/risk_manager.py'); print('  strategies/risk_manager.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/signal_generator.py','python_bridge_v4/strategies/signal_generator.py'); print('  strategies/signal_generator.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/smart_exits.py','python_bridge_v4/strategies/smart_exits.py'); print('  strategies/smart_exits.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/trading_brain.py','python_bridge_v4/strategies/trading_brain.py'); print('  strategies/trading_brain.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/regime_router.py','python_bridge_v4/strategies/regime_router.py'); print('  strategies/regime_router.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/walk_forward.py','python_bridge_v4/strategies/walk_forward.py'); print('  strategies/walk_forward.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/adversarial_filter.py','python_bridge_v4/strategies/adversarial_filter.py'); print('  strategies/adversarial_filter.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/correlation_regime.py','python_bridge_v4/strategies/correlation_regime.py'); print('  strategies/correlation_regime.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/adaptive_threshold.py','python_bridge_v4/strategies/adaptive_threshold.py'); print('  strategies/adaptive_threshold.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/disagreement_signal.py','python_bridge_v4/strategies/disagreement_signal.py'); print('  strategies/disagreement_signal.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/kelly_sizing.py','python_bridge_v4/strategies/kelly_sizing.py'); print('  strategies/kelly_sizing.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/strategies/monte_carlo.py','python_bridge_v4/strategies/monte_carlo.py'); print('  strategies/monte_carlo.py OK')"

:: --- signals/ ---
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/signals/__init__.py','python_bridge_v4/signals/__init__.py'); print('  signals/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/signals/bridge.py','python_bridge_v4/signals/bridge.py'); print('  signals/bridge.py OK')"

:: --- dashboard/ ---
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/dashboard/__init__.py','python_bridge_v4/dashboard/__init__.py'); print('  dashboard/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/dashboard/dashboard_renderer.py','python_bridge_v4/dashboard/dashboard_renderer.py'); print('  dashboard/dashboard_renderer.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/dashboard/performance_tracker.py','python_bridge_v4/dashboard/performance_tracker.py'); print('  dashboard/performance_tracker.py OK')"

:: --- training/ ---
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/training/__init__.py','python_bridge_v4/training/__init__.py'); print('  training/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/training/auto_retrain.py','python_bridge_v4/training/auto_retrain.py'); print('  training/auto_retrain.py OK')"

:: --- tests/ ---
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/tests/__init__.py','python_bridge_v4/tests/__init__.py'); print('  tests/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/tests/test_auto_optimizer.py','python_bridge_v4/tests/test_auto_optimizer.py'); print('  tests/test_auto_optimizer.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/tests/test_auto_retrain.py','python_bridge_v4/tests/test_auto_retrain.py'); print('  tests/test_auto_retrain.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/tests/test_backtester.py','python_bridge_v4/tests/test_backtester.py'); print('  tests/test_backtester.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/tests/test_bridge.py','python_bridge_v4/tests/test_bridge.py'); print('  tests/test_bridge.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/tests/test_dashboard.py','python_bridge_v4/tests/test_dashboard.py'); print('  tests/test_dashboard.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/tests/test_models.py','python_bridge_v4/tests/test_models.py'); print('  tests/test_models.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/tests/test_multi_timeframe.py','python_bridge_v4/tests/test_multi_timeframe.py'); print('  tests/test_multi_timeframe.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/tests/test_news_calendar.py','python_bridge_v4/tests/test_news_calendar.py'); print('  tests/test_news_calendar.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/tests/test_rl_agent.py','python_bridge_v4/tests/test_rl_agent.py'); print('  tests/test_rl_agent.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/tests/test_signal_generator.py','python_bridge_v4/tests/test_signal_generator.py'); print('  tests/test_signal_generator.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/tests/test_smart_exits.py','python_bridge_v4/tests/test_smart_exits.py'); print('  tests/test_smart_exits.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/tests/test_smart_upgrades.py','python_bridge_v4/tests/test_smart_upgrades.py'); print('  tests/test_smart_upgrades.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/tests/test_training.py','python_bridge_v4/tests/test_training.py'); print('  tests/test_training.py OK')"

:: --- requirements.txt ---
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_v4/requirements.txt','python_bridge_v4/requirements.txt'); print('  requirements.txt OK')"

:: --- EA file ---
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
