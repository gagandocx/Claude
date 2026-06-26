@echo off
color 0A
title EA Update and Run
SET PYTHONUNBUFFERED=1
SET PYTHONIOENCODING=utf-8

echo ============================================
echo   EA Update and Run
echo ============================================
echo.

cd /d "%~dp0"
echo Dir: %CD%
echo.

echo Downloading latest files...

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/update_and_run.bat','update_and_run_new.bat')" 2>nul
if exist update_and_run_new.bat (
    copy /y update_and_run_new.bat update_and_run.bat >nul
    del update_and_run_new.bat >nul
    echo   self-updated
)

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge/strategies/trading_brain.py','python_bridge/strategies/trading_brain.py'); print('  trading_brain.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge/config/settings.py','python_bridge/config/settings.py'); print('  settings.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge/main.py','python_bridge/main.py'); print('  main.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge/models/tft_model.py','python_bridge/models/tft_model.py'); print('  tft_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge/models/chronos_model.py','python_bridge/models/chronos_model.py'); print('  chronos_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge/models/gradient_boost_extra.py','python_bridge/models/gradient_boost_extra.py'); print('  gradient_boost_extra.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge/models/catboost_model.py','python_bridge/models/catboost_model.py'); print('  catboost_model.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge/models/ensemble.py','python_bridge/models/ensemble.py'); print('  ensemble.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge/data/market_data.py','python_bridge/data/market_data.py'); print('  market_data.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge/data/multi_timeframe.py','python_bridge/data/multi_timeframe.py'); print('  multi_timeframe.py OK')"

python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge/signals/bridge.py','python_bridge/signals/bridge.py'); print('  bridge.py OK')"

echo.
echo All files updated. Applying patches...
python -c "f=open('python_bridge/strategies/trading_brain.py','r',encoding='utf-8').read(); f=f.replace('rp[\"rr\"]','rp.get(\"tp_rr\",rp.get(\"rr\",0.0))'); open('python_bridge/strategies/trading_brain.py','w',encoding='utf-8').write(f)"
echo Starting EA...
echo ============================================
echo.

cd python_bridge
python -u main.py --live

pause
