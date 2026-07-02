@echo off
setlocal enabledelayedexpansion

:: ------------------------------------------------------------
::  NeuroX v9.4 - Force Update Script
::  1. Force download: always downloads latest ZIP from GitHub
::  2. Copy EAs + includes to MT5 terminals
::  3. Compile EAs (NeuroX_EA_v9 + NeuroX_Standalone_v9)
:: ------------------------------------------------------------

title NeuroX v9.4 - Force Update
color 0B

echo.
echo ============================================================
echo   NeuroX v9.4 - Force Update
echo   Always downloads latest from GitHub (replaces all files)
echo ============================================================
echo.

:: ------------------------------------------------------------
:: CONFIGURATION
:: ------------------------------------------------------------
set "REPO_DIR=%~dp0"
if "!REPO_DIR:~-1!"=="\" set "REPO_DIR=!REPO_DIR:~0,-1!"
set "EA_FILE=NeuroX_EA_v9.mq5"
set "STANDALONE_FILE=NeuroX_Standalone_v9.mq5"
set "ZIP_URL=https://github.com/gagandocx/NeuroX-v9/archive/refs/heads/main.zip"
set "ZIP_FILE=%TEMP%\neurox_update.zip"
set "ZIP_EXTRACT=%TEMP%\neurox_update"

:: MT5 Terminal: EA runs here
set "MT5_TERMINAL_ID=930119AA53207C8778B41171FBFFB46F"
set "MT5_BASE=C:\Users\gagan\AppData\Roaming\MetaQuotes\Terminal\%MT5_TERMINAL_ID%"
set "MT5_EXPERTS=%MT5_BASE%\MQL5\Experts\Advisors"
set "MT5_INCLUDE=%MT5_BASE%\MQL5\Include\NeuroX"
set "MT5_SCRIPTS=%MT5_BASE%\MQL5\Scripts"

:: ExportRealTicks configuration
set "EXPORT_TICKS_URL=https://raw.githubusercontent.com/gagandocx/Claude/main/ExportRealTicks.mq5"
set "EXPORT_TICKS_FILE=ExportRealTicks.mq5"

:: MetaEditor include resolution terminal (copy includes here too)
set "MT5_EDITOR_ID=D0E8209F77C8CF37AD8BF550E51FF075"
set "MT5_EDITOR_BASE=C:\Users\gagan\AppData\Roaming\MetaQuotes\Terminal\%MT5_EDITOR_ID%"
set "MT5_EDITOR_INCLUDE=%MT5_EDITOR_BASE%\MQL5\Include\NeuroX"

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
:: STEP 1: Force Download ZIP from GitHub
:: ------------------------------------------------------------
echo [1/3] Force downloading latest version from GitHub...
echo        URL: %ZIP_URL%
echo.

cd /d "%REPO_DIR%"

:: Back up .env if it exists
set "ENV_BACKED_UP=0"
if exist "%REPO_DIR%\.env" (
    copy /Y "%REPO_DIR%\.env" "%TEMP%\neurox_env_backup" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "ENV_BACKED_UP=1"
        echo        [OK] .env file backed up.
    )
)

:: Clean up any previous temp files
if exist "%ZIP_FILE%" del /f /q "%ZIP_FILE%" >nul 2>&1
if exist "%ZIP_EXTRACT%" rd /s /q "%ZIP_EXTRACT%" >nul 2>&1

:: Download ZIP using PowerShell
echo        Downloading...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%ZIP_URL%' -OutFile '%ZIP_FILE%' -UseBasicParsing; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }"
if !ERRORLEVEL! neq 0 (
    echo        [ERROR] Failed to download ZIP. Check your internet connection.
    goto :error_exit
)

if not exist "%ZIP_FILE%" (
    echo        [ERROR] ZIP file not found after download.
    goto :error_exit
)

echo        [OK] ZIP downloaded successfully.

:: Extract ZIP using PowerShell
echo        Extracting...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%ZIP_EXTRACT%' -Force; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }"
if !ERRORLEVEL! neq 0 (
    echo        [ERROR] Failed to extract ZIP file.
    goto :error_exit
)

:: Verify extracted folder exists
if not exist "%ZIP_EXTRACT%\NeuroX-v9-main" (
    echo        [ERROR] Expected folder 'NeuroX-v9-main' not found in ZIP.
    goto :error_exit
)

echo        [OK] ZIP extracted successfully.

:: Copy all files from extracted folder to REPO_DIR (force overwrite)
echo        Copying files (force overwrite)...
xcopy "%ZIP_EXTRACT%\NeuroX-v9-main\*" "%REPO_DIR%\" /E /Y /Q >nul
if !ERRORLEVEL! neq 0 (
    echo        [ERROR] Failed to copy files from ZIP.
    goto :error_exit
)

echo        [OK] All files replaced with latest from GitHub.

:: Restore .env if it was backed up
if "!ENV_BACKED_UP!"=="1" (
    copy /Y "%TEMP%\neurox_env_backup" "%REPO_DIR%\.env" >nul 2>&1
    echo        [OK] .env file restored.
    del /f /q "%TEMP%\neurox_env_backup" >nul 2>&1
)

:: Clean up temp files
if exist "%ZIP_FILE%" del /f /q "%ZIP_FILE%" >nul 2>&1
if exist "%ZIP_EXTRACT%" rd /s /q "%ZIP_EXTRACT%" >nul 2>&1

echo.

:: ------------------------------------------------------------
:: STEP 2: Copy EAs + Includes to MT5
:: ------------------------------------------------------------
echo [2/3] Copying files to MT5 terminals...
echo.

set "EA_SOURCE=%REPO_DIR%\%EA_FILE%"
set "STANDALONE_SOURCE=%REPO_DIR%\%STANDALONE_FILE%"

:: Create directories
if not exist "%MT5_EXPERTS%" mkdir "%MT5_EXPERTS%"
if not exist "%MT5_INCLUDE%" mkdir "%MT5_INCLUDE%"
if not exist "%MT5_EDITOR_INCLUDE%" mkdir "%MT5_EDITOR_INCLUDE%"
if not exist "%MT5_SCRIPTS%" mkdir "%MT5_SCRIPTS%"

:: Copy NeuroX EA to MT5
if exist "%EA_SOURCE%" (
    copy /Y "%EA_SOURCE%" "%MT5_EXPERTS%\" >nul 2>&1
    echo        [OK] %EA_FILE% copied to: %MT5_EXPERTS%
) else (
    echo        [WARNING] %EA_FILE% not found. Skipping.
)

:: Copy NeuroX Standalone to MT5
if exist "%STANDALONE_SOURCE%" (
    copy /Y "%STANDALONE_SOURCE%" "%MT5_EXPERTS%\" >nul 2>&1
    echo        [OK] %STANDALONE_FILE% copied to: %MT5_EXPERTS%
) else (
    echo        [WARNING] %STANDALONE_FILE% not found. Skipping.
)

:: Copy includes to EA terminal
if exist "%REPO_DIR%\Include\*.mqh" (
    copy /Y "%REPO_DIR%\Include\*.mqh" "%MT5_INCLUDE%\" >nul 2>&1
    echo        [OK] Includes copied to: %MT5_INCLUDE%
)

:: Copy includes to MetaEditor resolution terminal
if exist "%REPO_DIR%\Include\*.mqh" (
    copy /Y "%REPO_DIR%\Include\*.mqh" "%MT5_EDITOR_INCLUDE%\" >nul 2>&1
    echo        [OK] Includes copied to: %MT5_EDITOR_INCLUDE% (MetaEditor)
)

:: Download and copy ExportRealTicks.mq5 to MT5 Scripts folder
echo.
echo        Downloading ExportRealTicks.mq5...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%EXPORT_TICKS_URL%' -OutFile '%MT5_SCRIPTS%\%EXPORT_TICKS_FILE%' -UseBasicParsing; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }"
if !ERRORLEVEL! neq 0 (
    echo        [WARNING] Failed to download ExportRealTicks.mq5. Skipping.
) else (
    if exist "%MT5_SCRIPTS%\%EXPORT_TICKS_FILE%" (
        echo        [OK] %EXPORT_TICKS_FILE% downloaded to: %MT5_SCRIPTS%
    ) else (
        echo        [WARNING] %EXPORT_TICKS_FILE% not found after download. Skipping.
    )
)

echo.

:: ------------------------------------------------------------
:: STEP 3: Compile EAs
:: ------------------------------------------------------------
echo [3/3] Compiling...
echo.

if not defined METAEDITOR (
    echo        [INFO] MetaEditor not found. Open MetaEditor and press F7 to compile manually.
    goto :done
)

:: Compile NeuroX EA
if exist "%MT5_EXPERTS%\%EA_FILE%" (
    set "COMPILE_TARGET=%MT5_EXPERTS%\%EA_FILE%"
    echo        Compiling: !COMPILE_TARGET!
    "%METAEDITOR%" /compile:"!COMPILE_TARGET!" /log >nul 2>&1
    timeout /t 8 /nobreak >nul

    set "LOG_FILE=%MT5_EXPERTS%\NeuroX_EA_v9.log"
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

echo.

:: Compile NeuroX Standalone
if exist "%MT5_EXPERTS%\%STANDALONE_FILE%" (
    set "COMPILE_TARGET=%MT5_EXPERTS%\%STANDALONE_FILE%"
    echo        Compiling: !COMPILE_TARGET!
    "%METAEDITOR%" /compile:"!COMPILE_TARGET!" /log >nul 2>&1
    timeout /t 8 /nobreak >nul

    set "LOG_FILE=%MT5_EXPERTS%\NeuroX_Standalone_v9.log"
    if exist "!LOG_FILE!" (
        findstr /i " error " "!LOG_FILE!" >nul
        if !errorlevel! equ 0 (
            echo        [WARNING] %STANDALONE_FILE% compilation has errors. Check log.
        ) else (
            echo        [OK] %STANDALONE_FILE% compiled successfully.
        )
    ) else (
        echo        [OK] %STANDALONE_FILE% compile complete.
    )
) else (
    echo        [WARNING] %STANDALONE_FILE% not in Experts folder. Skipping compile.
)

:done
echo.
echo ============================================================
echo   Update Complete!
echo   All files replaced with latest v9.4 from GitHub.
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
