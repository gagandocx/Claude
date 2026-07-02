@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo   ExportRealTicks.mq5 - Auto Updater
echo ============================================================
echo.

:: MT5 Terminal configuration
set "MT5_BASE=C:\Users\gagan\AppData\Roaming\MetaQuotes\Terminal\930119AA53207C8778B41171FBFFB46F"
set "SCRIPTS_DIR=%MT5_BASE%\MQL5\Scripts"
set "MQL5_DIR=%MT5_BASE%\MQL5"
set "TARGET_FILE=%SCRIPTS_DIR%\ExportRealTicks.mq5"
set "DOWNLOAD_URL=https://raw.githubusercontent.com/gagandocx/Claude/main/ExportRealTicks.mq5"

:: Check if Scripts folder exists
if not exist "%SCRIPTS_DIR%" (
    echo ERROR: Scripts folder not found:
    echo   %SCRIPTS_DIR%
    echo.
    echo Please check your MT5 terminal ID.
    pause
    exit /b 1
)

echo [1/3] Downloading latest ExportRealTicks.mq5...
echo   From: %DOWNLOAD_URL%
echo   To:   %TARGET_FILE%
echo.

:: Download using PowerShell with TLS 1.2
powershell -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%DOWNLOAD_URL%' -OutFile '%TARGET_FILE%'"

if %ERRORLEVEL% neq 0 (
    echo ERROR: Download failed! Check your internet connection.
    pause
    exit /b 1
)

echo   Download complete!
echo.

:: Find MetaEditor
echo [2/3] Locating MetaEditor for compilation...

set "METAEDITOR="

:: Check common MetaEditor locations
if exist "C:\Program Files\Fusion Markets MetaTrader 5\metaeditor64.exe" (
    set "METAEDITOR=C:\Program Files\Fusion Markets MetaTrader 5\metaeditor64.exe"
)
if exist "C:\Program Files\MetaTrader 5\metaeditor64.exe" (
    set "METAEDITOR=C:\Program Files\MetaTrader 5\metaeditor64.exe"
)
if exist "C:\Program Files (x86)\MetaTrader 5\metaeditor64.exe" (
    set "METAEDITOR=C:\Program Files (x86)\MetaTrader 5\metaeditor64.exe"
)
if exist "D:\Program Files\MetaTrader 5\metaeditor64.exe" (
    set "METAEDITOR=D:\Program Files\MetaTrader 5\metaeditor64.exe"
)
if exist "C:\Program Files\Fusion Markets MetaTrader 5 Terminal 2\metaeditor64.exe" (
    set "METAEDITOR=C:\Program Files\Fusion Markets MetaTrader 5 Terminal 2\metaeditor64.exe"
)

:: Search more broadly if not found
if "!METAEDITOR!"=="" (
    for /f "delims=" %%F in ('where /r "C:\Program Files" metaeditor64.exe 2^>nul') do (
        set "METAEDITOR=%%F"
        goto :found_editor
    )
    for /f "delims=" %%F in ('where /r "C:\Program Files (x86)" metaeditor64.exe 2^>nul') do (
        set "METAEDITOR=%%F"
        goto :found_editor
    )
    for /f "delims=" %%F in ('where /r "D:\Program Files" metaeditor64.exe 2^>nul') do (
        set "METAEDITOR=%%F"
        goto :found_editor
    )
)

:found_editor
if "!METAEDITOR!"=="" (
    echo WARNING: MetaEditor not found! File downloaded but not compiled.
    echo   Please compile manually in MetaEditor or check your MT5 installation path.
    echo.
    echo   The file is here: %TARGET_FILE%
    pause
    exit /b 0
)

echo   Found: !METAEDITOR!
echo.
echo [3/3] Compiling ExportRealTicks.mq5...

:: Compile with /inc pointing to MQL5 folder for includes
"!METAEDITOR!" /compile:"%TARGET_FILE%" /inc:"%MQL5_DIR%" /log

if %ERRORLEVEL% neq 0 (
    echo.
    echo WARNING: Compilation may have issues. Check MetaEditor log.
    echo   The .mq5 source file is still saved and can be compiled manually.
) else (
    echo   Compilation successful!
)

echo.
echo ============================================================
echo   Good to go!
echo ============================================================
echo.
echo   File location: %TARGET_FILE%
echo   Ready to use in MT5 - Navigator ^> Scripts ^> ExportRealTicks
echo.

pause
exit /b 0
