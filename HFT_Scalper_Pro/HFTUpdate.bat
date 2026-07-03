@echo off
setlocal EnableDelayedExpansion

:: ------------------------------------------------------------
::  HFT Scalper Pro - Auto Update Script
::  Downloads latest HFT system files from GitHub
::  Target: F:\Automation\EA Testing\HFT\HFT_v2
:: ------------------------------------------------------------

title HFT Scalper Pro - Update
color 0A

echo.
echo ============================================================
echo   HFT Scalper Pro - Auto Update
echo   Downloads latest files from GitHub
echo ============================================================
echo.

:: ------------------------------------------------------------
:: CONFIGURATION
:: ------------------------------------------------------------
set "BASE_URL=https://raw.githubusercontent.com/gagandocx/Claude/main/HFT_Scalper_Pro/hft_scalper"
set "TARGET_DIR=F:\Automation\EA Testing\HFT\HFT_v2"

:: Create target directories if they don't exist
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
if not exist "%TARGET_DIR%\strategies" mkdir "%TARGET_DIR%\strategies"
if not exist "%TARGET_DIR%\output" mkdir "%TARGET_DIR%\output"
if not exist "%TARGET_DIR%\results" mkdir "%TARGET_DIR%\results"
if not exist "%TARGET_DIR%\tick_data" mkdir "%TARGET_DIR%\tick_data"

echo   Source:  %BASE_URL%
echo   Target:  %TARGET_DIR%
echo.

:: ------------------------------------------------------------
:: STEP 0: Download account config sample
:: ------------------------------------------------------------
echo [0/4] Downloading account config...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%BASE_URL%/account_config.sample.json' -OutFile '%TARGET_DIR%\account_config.sample.json' -UseBasicParsing; exit 0 } catch { exit 1 }"
if !ERRORLEVEL! equ 0 (
    echo        [OK] account_config.sample.json
) else (
    echo        [FAIL] account_config.sample.json
)

:: Copy to account_config.json only if it doesn't already exist
if not exist "%TARGET_DIR%\account_config.json" (
    copy "%TARGET_DIR%\account_config.sample.json" "%TARGET_DIR%\account_config.json" >nul 2>&1
    echo        [OK] Created account_config.json from sample
) else (
    echo        [SKIP] account_config.json already exists
)

echo.

:: ------------------------------------------------------------
:: STEP 1: Download core Python files
:: ------------------------------------------------------------
echo [1/4] Downloading core Python files...
echo.

set "CORE_FILES=__init__.py analyze.py backtest_engine.py data_loader.py live_trader.py microstructure_analysis.py optimizer.py run_aggressive_backtest.py run_backtest.py run_ensemble_backtest.py run_quick_test.py LIVE_TRADING_README.md"

set "DOWNLOAD_COUNT=0"
set "ERROR_COUNT=0"

for %%F in (%CORE_FILES%) do (
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
:: STEP 2: Download strategy files
:: ------------------------------------------------------------
echo [2/4] Downloading strategy files...
echo.

set "STRATEGY_FILES=__init__.py base.py ensemble.py mean_reversion.py momentum_mtf.py order_flow.py spread_fade.py volatility_breakout.py"

for %%F in (%STRATEGY_FILES%) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%BASE_URL%/strategies/%%F' -OutFile '%TARGET_DIR%\strategies\%%F' -UseBasicParsing; exit 0 } catch { exit 1 }"
    if !ERRORLEVEL! equ 0 (
        echo        [OK] strategies/%%F
        set /a DOWNLOAD_COUNT+=1
    ) else (
        echo        [FAIL] strategies/%%F
        set /a ERROR_COUNT+=1
    )
)

echo.

:: ------------------------------------------------------------
:: STEP 3: Download output files (EA + docs)
:: ------------------------------------------------------------
echo [3/4] Downloading output files...
echo.

set "OUTPUT_FILES=HFT_Scalper_Pro.mq5 EA_README.md backtest_summary.md"

for %%F in (%OUTPUT_FILES%) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%BASE_URL%/output/%%F' -OutFile '%TARGET_DIR%\output\%%F' -UseBasicParsing; exit 0 } catch { exit 1 }"
    if !ERRORLEVEL! equ 0 (
        echo        [OK] output/%%F
        set /a DOWNLOAD_COUNT+=1
    ) else (
        echo        [FAIL] output/%%F
        set /a ERROR_COUNT+=1
    )
)

echo.

:: ------------------------------------------------------------
:: STEP 4: Download result files
:: ------------------------------------------------------------
echo [4/4] Downloading result files...
echo.

set "RESULT_FILES=aggressive_results.json ensemble_results.json equity_curves.json microstructure_report.json strategy_comparison.json winner_trade_log.json"

for %%F in (%RESULT_FILES%) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%BASE_URL%/results/%%F' -OutFile '%TARGET_DIR%\results\%%F' -UseBasicParsing; exit 0 } catch { exit 1 }"
    if !ERRORLEVEL! equ 0 (
        echo        [OK] results/%%F
        set /a DOWNLOAD_COUNT+=1
    ) else (
        echo        [FAIL] results/%%F
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
