@echo off
title EA Update and Run
color 0A

echo ============================================
echo   EA Update and Run Script
echo ============================================
echo.

REM Navigate to the EA folder (edit this path if needed)
cd /d "%~dp0"
echo Working directory: %CD%
echo.

REM Download latest files from GitHub
echo [1/4] Downloading Trading Brain...
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge/strategies/trading_brain.py','python_bridge/strategies/trading_brain.py'); print('  trading_brain.py updated')"

echo [2/4] Downloading settings...
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge/config/settings.py','python_bridge/config/settings.py'); print('  settings.py updated')"

echo [3/4] Downloading main.py...
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/gagandocx/Claude/feature/5-model-ensemble-tcn-lgbm/python_bridge/main.py','python_bridge/main.py'); print('  main.py updated')"

echo.
echo [4/4] Starting EA...
echo ============================================
echo.
cd python_bridge
python main.py --live

pause
