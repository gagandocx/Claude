@echo off
setlocal EnableDelayedExpansion

:: ------------------------------------------------------------
::  Adaptive Trading System - Auto Update Script
::  Downloads latest adaptive system files from GitHub
::  Target: F:\Automation\EA Testing\Adaptive_System
:: ------------------------------------------------------------

title Adaptive Trading System - Update
color 0A

echo.
echo ============================================================
echo   Adaptive Trading System - Auto Update
echo   Downloads latest files from GitHub
echo ============================================================
echo.

:: ------------------------------------------------------------
:: CONFIGURATION
:: ------------------------------------------------------------
set "BASE_URL=https://raw.githubusercontent.com/gagandocx/Claude/main/HFT_Scalper_Pro/adaptive_system"
set "TARGET_DIR=F:\Automation\EA Testing\Adaptive_System"

:: Create target directories if they don't exist
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
if not exist "%TARGET_DIR%\core" mkdir "%TARGET_DIR%\core"
if not exist "%TARGET_DIR%\tick_data" mkdir "%TARGET_DIR%\tick_data"
if not exist "%TARGET_DIR%\results" mkdir "%TARGET_DIR%\results"

echo   Source:  %BASE_URL%
echo   Target:  %TARGET_DIR%
echo.

:: ------------------------------------------------------------
:: STEP 0: Self-update AdaptiveUpdate.bat
:: ------------------------------------------------------------
echo [0/2] Self-updating...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%BASE_URL%/AdaptiveUpdate.bat' -OutFile '%TARGET_DIR%\AdaptiveUpdate.bat' -UseBasicParsing; exit 0 } catch { exit 1 }"
if !ERRORLEVEL! equ 0 (
    echo        [OK] AdaptiveUpdate.bat
) else (
    echo        [FAIL] AdaptiveUpdate.bat
)

echo.

:: ------------------------------------------------------------
:: STEP 1: Download root-level Python files
:: ------------------------------------------------------------
echo [1/2] Downloading root-level Python files...
echo.

set "ROOT_FILES=__init__.py config.py data_loader.py backtest_engine.py live_trader.py run_demo.py run_backtest.py run_optimized_backtest.py run_optimized_local.py README.md"

set "DOWNLOAD_COUNT=0"
set "ERROR_COUNT=0"

for %%F in (%ROOT_FILES%) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%BASE_URL%/%%F' -OutFile '%TARGET_DIR%\%%F' -UseBasicParsing; exit 0 } catch { exit 1 }"
    if !ERRORLEVEL! equ 0 (
        echo        [OK] %%F
        set /a DOWNLOAD_COUNT+=1
    ) else (
        echo        [FAIL] %%F
        set /a ERROR_COUNT+=1
    )
)

echo.

:: ------------------------------------------------------------
:: STEP 2: Download core module files
:: ------------------------------------------------------------
echo [2/2] Downloading core module files...
echo.

set "CORE_FILES=__init__.py indicators.py regime_detector.py strategies.py strategy_selector.py position_sizer.py online_learner.py risk_manager.py portfolio_manager.py"

for %%F in (%CORE_FILES%) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%BASE_URL%/core/%%F' -OutFile '%TARGET_DIR%\core\%%F' -UseBasicParsing; exit 0 } catch { exit 1 }"
    if !ERRORLEVEL! equ 0 (
        echo        [OK] core/%%F
        set /a DOWNLOAD_COUNT+=1
    ) else (
        echo        [FAIL] core/%%F
        set /a ERROR_COUNT+=1
    )
)

echo.

:: ------------------------------------------------------------
:: SUMMARY
:: ------------------------------------------------------------
echo ============================================================
if !ERROR_COUNT! equ 0 (
    echo   Update Complete! All !DOWNLOAD_COUNT! files downloaded.
) else (
    echo   Update finished with errors.
    echo   Downloaded: !DOWNLOAD_COUNT! files
    echo   Failed:     !ERROR_COUNT! files
)
echo.
echo   Files saved to: %TARGET_DIR%
echo ============================================================
echo.

pause
exit /b 0
