@echo off
REM ============================================
REM  NeuroX v7.2-rc1 - Update and Run
REM  This runs the v2 system (python_bridge_v2/)
REM  The original system in python_bridge/ is untouched
REM ============================================

echo ============================================
echo  NeuroX v7.2-rc1 - Starting
echo ============================================

cd /d "%~dp0"

REM Pull latest code from git
echo [1/3] Pulling latest code...
git pull origin feature/5-model-ensemble-tcn-lgbm 2>nul
if %errorlevel% neq 0 (
    echo Warning: git pull failed - continuing with local code
)

REM Install/update dependencies
echo [2/3] Checking dependencies...
cd python_bridge_v2
pip install -r requirements.txt --quiet 2>nul

REM Run the v2 bridge
echo [3/3] Starting NeuroX v7.2-rc1...
echo ============================================
python -u main.py --live

REM If Python exits, pause so user can see errors
echo.
echo Bridge stopped. Press any key to exit...
pause >nul
