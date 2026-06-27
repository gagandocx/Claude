@echo off
color 0D
title NeuroX Watchdog - Process Supervisor
SET PYTHONUNBUFFERED=1
SET PYTHONIOENCODING=utf-8

echo ============================================
echo   NeuroX Watchdog - Auto Restart
echo   Will restart NeuroX if it crashes
echo ============================================
echo.

cd /d "%~dp0"

:: Navigate to the Python bridge directory
if exist "python_bridge_v4" (
    cd python_bridge_v4
) else if exist "neurox_v4" (
    cd neurox_v4
) else (
    echo ERROR: Cannot find python_bridge_v4 or neurox_v4 directory
    pause
    exit /b 1
)

echo Dir: %CD%
echo.
echo Starting NeuroX with watchdog supervisor...
echo Press Ctrl+C to stop.
echo.

python -u watchdog.py --live

pause
