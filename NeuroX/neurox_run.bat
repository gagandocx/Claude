@echo off
color 0A
title NeuroX v7.1
SET PYTHONUNBUFFERED=1
SET PYTHONIOENCODING=utf-8

echo ============================================
echo   NeuroX v7.1
echo ============================================
echo.

cd /d "%~dp0"
echo Dir: %CD%
echo.

echo Downloading latest files...

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox_run.bat','neurox_run_new.bat')" 2>nul
if exist neurox_run_new.bat (
    copy /y neurox_run_new.bat neurox_run.bat >nul
    del neurox_run_new.bat >nul
    echo   self-updated
)

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox/strategies/trading_brain.py','neurox/strategies/trading_brain.py'); print('  trading_brain.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox/strategies/signal_generator.py','neurox/strategies/signal_generator.py'); print('  signal_generator.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox/config/settings.py','neurox/config/settings.py'); print('  settings.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox/main.py','neurox/main.py'); print('  main.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox/models/tft_model.py','neurox/models/tft_model.py'); print('  tft_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox/models/chronos_model.py','neurox/models/chronos_model.py'); print('  chronos_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox/models/gradient_boost_extra.py','neurox/models/gradient_boost_extra.py'); print('  gradient_boost_extra.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox/models/catboost_model.py','neurox/models/catboost_model.py'); print('  catboost_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox/models/ensemble.py','neurox/models/ensemble.py'); print('  ensemble.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox/data/market_data.py','neurox/data/market_data.py'); print('  market_data.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox/data/multi_timeframe.py','neurox/data/multi_timeframe.py'); print('  multi_timeframe.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/neurox/signals/bridge.py','neurox/signals/bridge.py'); print('  bridge.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/NeuroX/NeuroX_EA.mq5','NeuroX_EA.mq5'); print('  NeuroX_EA.mq5 OK  <-- recompile in MetaEditor if updated')"

echo.
echo All files updated. Applying patches...
python -c "f=open('neurox/strategies/trading_brain.py','r',encoding='utf-8').read(); f=f.replace('rp[\"rr\"]','rp.get(\"tp_rr\",rp.get(\"rr\",0.0))'); open('neurox/strategies/trading_brain.py','w',encoding='utf-8').write(f)"
echo Starting NeuroX...
echo ============================================
echo.

cd neurox
python -u main.py --live

pause
