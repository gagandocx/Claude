@echo off
color 0B
title Python ML Bridge v3 - Update and Run
SET PYTHONUNBUFFERED=1
SET PYTHONIOENCODING=utf-8

echo ============================================
echo   Python ML Bridge v3 (v8.0)
echo   Institutional-Grade Trading System
echo   Update and Run
echo ============================================
echo.

cd /d "%~dp0"
echo Dir: %CD%
echo.

echo [1/4] Setting up directories...

if not exist python_bridge_v3 mkdir python_bridge_v3
if not exist python_bridge_v3\config mkdir python_bridge_v3\config
if not exist python_bridge_v3\data mkdir python_bridge_v3\data
if not exist python_bridge_v3\models mkdir python_bridge_v3\models
if not exist python_bridge_v3\signals mkdir python_bridge_v3\signals
if not exist python_bridge_v3\strategies mkdir python_bridge_v3\strategies
if not exist python_bridge_v3\training mkdir python_bridge_v3\training
if not exist python_bridge_v3\dashboard mkdir python_bridge_v3\dashboard
echo   directories OK

echo.
echo [2/4] Downloading latest files...

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/__init__.py','python_bridge_v3/__init__.py'); print('  __init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/main.py','python_bridge_v3/main.py'); print('  main.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/main_multi.py','python_bridge_v3/main_multi.py'); print('  main_multi.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/backtest.py','python_bridge_v3/backtest.py'); print('  backtest.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/train.py','python_bridge_v3/train.py'); print('  train.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/train_colab.py','python_bridge_v3/train_colab.py'); print('  train_colab.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/config/__init__.py','python_bridge_v3/config/__init__.py'); print('  config/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/config/settings.py','python_bridge_v3/config/settings.py'); print('  config/settings.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/data/__init__.py','python_bridge_v3/data/__init__.py'); print('  data/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/data/market_data.py','python_bridge_v3/data/market_data.py'); print('  data/market_data.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/data/multi_timeframe.py','python_bridge_v3/data/multi_timeframe.py'); print('  data/multi_timeframe.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/data/alternative_data.py','python_bridge_v3/data/alternative_data.py'); print('  data/alternative_data.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/data/news_calendar.py','python_bridge_v3/data/news_calendar.py'); print('  data/news_calendar.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/data/sentiment.py','python_bridge_v3/data/sentiment.py'); print('  data/sentiment.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/data/tick_data.py','python_bridge_v3/data/tick_data.py'); print('  data/tick_data.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/data/microstructure.py','python_bridge_v3/data/microstructure.py'); print('  data/microstructure.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/data/spread_monitor.py','python_bridge_v3/data/spread_monitor.py'); print('  data/spread_monitor.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/__init__.py','python_bridge_v3/models/__init__.py'); print('  models/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/ensemble.py','python_bridge_v3/models/ensemble.py'); print('  models/ensemble.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/transformer_model.py','python_bridge_v3/models/transformer_model.py'); print('  models/transformer_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/lstm_model.py','python_bridge_v3/models/lstm_model.py'); print('  models/lstm_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/tcn_model.py','python_bridge_v3/models/tcn_model.py'); print('  models/tcn_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/patch_tst.py','python_bridge_v3/models/patch_tst.py'); print('  models/patch_tst.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/tft_model.py','python_bridge_v3/models/tft_model.py'); print('  models/tft_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/nhits_model.py','python_bridge_v3/models/nhits_model.py'); print('  models/nhits_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/itransformer.py','python_bridge_v3/models/itransformer.py'); print('  models/itransformer.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/mamba_model.py','python_bridge_v3/models/mamba_model.py'); print('  models/mamba_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/dlinear_model.py','python_bridge_v3/models/dlinear_model.py'); print('  models/dlinear_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/xlstm_model.py','python_bridge_v3/models/xlstm_model.py'); print('  models/xlstm_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/timesnet_model.py','python_bridge_v3/models/timesnet_model.py'); print('  models/timesnet_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/chronos_model.py','python_bridge_v3/models/chronos_model.py'); print('  models/chronos_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/timemixer_model.py','python_bridge_v3/models/timemixer_model.py'); print('  models/timemixer_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/softs_model.py','python_bridge_v3/models/softs_model.py'); print('  models/softs_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/gradient_boost_extra.py','python_bridge_v3/models/gradient_boost_extra.py'); print('  models/gradient_boost_extra.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/catboost_model.py','python_bridge_v3/models/catboost_model.py'); print('  models/catboost_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/models/rl_agent.py','python_bridge_v3/models/rl_agent.py'); print('  models/rl_agent.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/signals/__init__.py','python_bridge_v3/signals/__init__.py'); print('  signals/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/signals/bridge.py','python_bridge_v3/signals/bridge.py'); print('  signals/bridge.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/__init__.py','python_bridge_v3/strategies/__init__.py'); print('  strategies/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/signal_generator.py','python_bridge_v3/strategies/signal_generator.py'); print('  strategies/signal_generator.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/trading_brain.py','python_bridge_v3/strategies/trading_brain.py'); print('  strategies/trading_brain.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/auto_optimizer.py','python_bridge_v3/strategies/auto_optimizer.py'); print('  strategies/auto_optimizer.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/risk_manager.py','python_bridge_v3/strategies/risk_manager.py'); print('  strategies/risk_manager.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/regime_detector.py','python_bridge_v3/strategies/regime_detector.py'); print('  strategies/regime_detector.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/smart_exits.py','python_bridge_v3/strategies/smart_exits.py'); print('  strategies/smart_exits.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/regime_router.py','python_bridge_v3/strategies/regime_router.py'); print('  strategies/regime_router.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/walk_forward.py','python_bridge_v3/strategies/walk_forward.py'); print('  strategies/walk_forward.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/adversarial_filter.py','python_bridge_v3/strategies/adversarial_filter.py'); print('  strategies/adversarial_filter.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/adaptive_threshold.py','python_bridge_v3/strategies/adaptive_threshold.py'); print('  strategies/adaptive_threshold.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/correlation_regime.py','python_bridge_v3/strategies/correlation_regime.py'); print('  strategies/correlation_regime.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/disagreement_signal.py','python_bridge_v3/strategies/disagreement_signal.py'); print('  strategies/disagreement_signal.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/kelly_sizing.py','python_bridge_v3/strategies/kelly_sizing.py'); print('  strategies/kelly_sizing.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/strategies/monte_carlo.py','python_bridge_v3/strategies/monte_carlo.py'); print('  strategies/monte_carlo.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/training/__init__.py','python_bridge_v3/training/__init__.py'); print('  training/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/training/auto_retrain.py','python_bridge_v3/training/auto_retrain.py'); print('  training/auto_retrain.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/dashboard/__init__.py','python_bridge_v3/dashboard/__init__.py'); print('  dashboard/__init__.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/dashboard/dashboard_renderer.py','python_bridge_v3/dashboard/dashboard_renderer.py'); print('  dashboard/dashboard_renderer.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/dashboard/performance_tracker.py','python_bridge_v3/dashboard/performance_tracker.py'); print('  dashboard/performance_tracker.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge_v3/requirements.txt','python_bridge_v3/requirements.txt'); print('  requirements.txt OK')"

echo.
echo [3/4] Downloading EA file...

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/Python_Bridge_EA_v3.mq5','Python_Bridge_EA_v3.mq5'); print('  Python_Bridge_EA_v3.mq5 OK  -- recompile in MetaEditor if updated')"

echo.
echo [4/4] Installing dependencies...

cd python_bridge_v3
pip install -r requirements.txt --quiet 2>nul
echo   dependencies OK

echo.
echo ============================================
echo   All files updated. Starting v3 bridge...
echo ============================================
echo.

python -u main.py --live

echo.
echo Bridge stopped. Press any key to exit...
pause >nul
