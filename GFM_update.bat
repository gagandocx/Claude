@echo off
setlocal enabledelayedexpansion

:: ------------------------------------------------------------
::  GFM EA - T-Spot Model Update
::  1. Download latest GFM_EA.mq5 from GitHub
::  2. Copy EA to local folder + MT5 Experts\Advisors
::  3. Compile EA using MetaEditor
:: ------------------------------------------------------------

title GFM EA - T-Spot Model Update
color 0B

echo.
echo ============================================================
echo   GFM EA - T-Spot Model Update
echo   Downloads latest from GitHub and sets up for MT5
echo ============================================================
echo.

:: ------------------------------------------------------------
:: CONFIGURATION
:: ------------------------------------------------------------
set "EA_FILE=GFM_EA.mq5"
set "EA_URL=https://raw.githubusercontent.com/gagandocx/Claude/GFM/GFM_EA.mq5"
set "TARGET_DIR=F:\Automation\EA Testing\GFM\v1"

:: MT5 Terminal: EA runs here
set "MT5_TERMINAL_ID=930119AA53207C8778B41171FBFFB46F"
set "MT5_BASE=C:\Users\gagan\AppData\Roaming\MetaQuotes\Terminal\%MT5_TERMINAL_ID%"
set "MT5_EXPERTS=%MT5_BASE%\MQL5\Experts\Advisors"

:: Find MetaEditor
set "METAEDITOR="
for %%P in (
    "C:\Program Files\Fusion Markets MetaTrader 5\metaeditor64.exe"
    "C:\Program Files (x86)\Fusion Markets MetaTrader 5\metaeditor64.exe"
    "C:\Program Files\MetaTrader 5\metaeditor64.exe"
) do (
    if exist %%P set "METAEDITOR=%%~P"
)

:: ------------------------------------------------------------
:: STEP 1: Download EA from GitHub
:: ------------------------------------------------------------
echo [1/3] Downloading latest %EA_FILE% from GitHub...
echo        URL: %EA_URL%
echo.

:: Create target directory if it doesn't exist
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

:: Download using PowerShell
echo        Downloading...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%EA_URL%' -OutFile '%TARGET_DIR%\%EA_FILE%' -UseBasicParsing; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }"
if !ERRORLEVEL! neq 0 (
    echo        [ERROR] Failed to download %EA_FILE%. Check your internet connection.
    goto :error_exit
)

if not exist "%TARGET_DIR%\%EA_FILE%" (
    echo        [ERROR] %EA_FILE% not found after download.
    goto :error_exit
)

echo        [OK] %EA_FILE% downloaded to: %TARGET_DIR%
echo.

:: ------------------------------------------------------------
:: STEP 2: Copy EA to MT5 Experts\Advisors
:: ------------------------------------------------------------
echo [2/3] Copying %EA_FILE% to MT5 terminal...
echo.

:: Create MT5 Experts directory if it doesn't exist
if not exist "%MT5_EXPERTS%" mkdir "%MT5_EXPERTS%"

:: Copy EA to MT5
copy /Y "%TARGET_DIR%\%EA_FILE%" "%MT5_EXPERTS%\" >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo        [ERROR] Failed to copy %EA_FILE% to MT5 Experts folder.
    goto :error_exit
)

echo        [OK] %EA_FILE% copied to: %MT5_EXPERTS%
echo.

:: ------------------------------------------------------------
:: STEP 3: Compile EA
:: ------------------------------------------------------------
echo [3/3] Compiling...
echo.

if not defined METAEDITOR (
    echo        [INFO] MetaEditor not found. Open MetaEditor and press F7 to compile manually.
    goto :done
)

:: Compile GFM EA
if exist "%MT5_EXPERTS%\%EA_FILE%" (
    set "COMPILE_TARGET=%MT5_EXPERTS%\%EA_FILE%"
    echo        Compiling: !COMPILE_TARGET!
    "%METAEDITOR%" /compile:"!COMPILE_TARGET!" /log >nul 2>&1
    timeout /t 8 /nobreak >nul

    set "LOG_FILE=%MT5_EXPERTS%\GFM_EA.log"
    if exist "!LOG_FILE!" (
        findstr /i " error " "!LOG_FILE!" >nul
        if !errorlevel! equ 0 (
            echo        [WARNING] %EA_FILE% compilation has errors. Check log.
        ) else (
            echo        [OK] %EA_FILE% compiled successfully.
        )
    ) else (
        echo        [OK] %EA_FILE% compile complete.
    )
) else (
    echo        [WARNING] %EA_FILE% not in Experts folder. Skipping compile.
)

:done
echo.
echo ============================================================
echo   Update Complete!
echo   %EA_FILE% downloaded and installed.
echo   Attach EA to chart and trade.
echo ============================================================
echo.
pause
exit /b 0

:error_exit
echo.
echo ============================================================
echo   Update Failed! Check errors above.
echo ============================================================
echo.
pause
exit /b 1
