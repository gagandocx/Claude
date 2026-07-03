@echo off
setlocal EnableDelayedExpansion

:: ------------------------------------------------------------
::  Adaptive Multi-Currency System - Auto Update Script
::  Downloads latest adaptive system files from GitHub
::  Target: F:\Automation\EA Testing\HFT\HFT_v2\adaptive_system
:: ------------------------------------------------------------

title Adaptive System - Update
color 0B

echo.
echo ============================================================
echo   Adaptive Multi-Currency System - Auto Update
echo   Downloads latest files from GitHub
echo ============================================================
echo.

:: ------------------------------------------------------------
:: CONFIGURATION
:: ------------------------------------------------------------
set "BASE_URL=https://raw.githubusercontent.com/gagandocx/Claude/main/HFT_Scalper_Pro/adaptive_system"
set "TARGET_DIR=F:\Automation\EA Testing\HFT\HFT_v2\adaptive_system"

:: Create target directories if they don't exist
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
if not exist "%TARGET_DIR%\core" mkdir "%TARGET_DIR%\core"

echo   Source:  %BASE_URL%
echo   Target:  %TARGET_DIR%
echo.

:: ------------------------------------------------------------
:: STEP 0: Self-update AdaptiveUpdate.bat
:: ------------------------------------------------------------
echo [0/4] Self-updating...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/gagandocx/Claude/main/HFT_Scalper_Pro/AdaptiveUpdate.bat' -OutFile 'F:\Automation\EA Testing\HFT\HFT_v2\AdaptiveUpdate.bat' -UseBasicParsing; exit 0 } catch { exit 1 }"
if !ERRORLEVEL! equ 0 (
    echo        [OK] AdaptiveUpdate.bat
) else (
    echo        [FAIL] AdaptiveUpdate.bat
)

echo.

:: ------------------------------------------------------------
:: STEP 1: Download core engine modules
:: ------------------------------------------------------------
echo [1/4] Downloading core engine modules...
echo.

set "CORE_FILES=__init__.py indicators.py regime_detector.py strategies.py strategy_selector.py position_sizer.py online_learner.py risk_manager.py portfolio_manager.py"

set "DOWNLOAD_COUNT=0"
set "ERROR_COUNT=0"

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
:: STEP 2: Download application-level Python files
:: ------------------------------------------------------------
echo [2/4] Downloading application files...
echo.

set "APP_FILES=__init__.py config.py data_loader.py backtest_engine.py live_trader.py run_demo.py run_backtest.py"

for %%F in (%APP_FILES%) do (
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
:: STEP 3: Download documentation
:: ------------------------------------------------------------
echo [3/4] Downloading documentation...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%BASE_URL%/README.md' -OutFile '%TARGET_DIR%\README.md' -UseBasicParsing; exit 0 } catch { exit 1 }"
if !ERRORLEVEL! equ 0 (
    echo        [OK] README.md
    set /a DOWNLOAD_COUNT+=1
) else (
    echo        [FAIL] README.md
    set /a ERROR_COUNT+=1
)

echo.

:: ------------------------------------------------------------
:: STEP 4: Verify installation
:: ------------------------------------------------------------
echo [4/4] Verifying installation...
echo.

:: Check if Python is available and files compile
where python >nul 2>&1
if !ERRORLEVEL! equ 0 (
    python -c "import sys; sys.path.insert(0, '%TARGET_DIR%'); from core import RegimeDetector; print('        [OK] Core imports verified')" 2>nul
    if !ERRORLEVEL! neq 0 (
        echo        [WARN] Could not verify imports - check Python path
    )
) else (
    echo        [SKIP] Python not found in PATH - manual verification needed
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
echo.
echo   Quick Test:
echo     cd "%TARGET_DIR%"
echo     python run_demo.py
echo.
echo   Live Trading:
echo     python live_trader.py --symbols XAUUSD,EURUSD,GBPJPY --magic 202501
echo ============================================================
echo.

pause
exit /b 0
