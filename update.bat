@echo off
setlocal

set "REPO_URL=https://raw.githubusercontent.com/gagandocx/Claude/main/ExportRealTicks.mq5"
set "FILENAME=ExportRealTicks.mq5"

REM Attempt to find the MT5 Scripts folder
set "MT5_SCRIPTS="
if exist "%APPDATA%\MetaQuotes\Terminal" (
    for /d %%D in ("%APPDATA%\MetaQuotes\Terminal\*") do (
        if exist "%%D\MQL5\Scripts" (
            set "MT5_SCRIPTS=%%D\MQL5\Scripts"
        )
    )
)

echo Downloading %FILENAME% from GitHub...

REM Download using PowerShell Invoke-WebRequest
powershell -Command "Invoke-WebRequest -Uri '%REPO_URL%' -OutFile '%FILENAME%'" 2>nul
if %ERRORLEVEL% neq 0 (
    echo PowerShell download failed, trying curl...
    curl -L -o "%FILENAME%" "%REPO_URL%"
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Download failed. Check your internet connection.
        pause
        exit /b 1
    )
)

echo Downloaded %FILENAME% to current directory.

REM Copy to MT5 Scripts folder if found
if defined MT5_SCRIPTS (
    copy /Y "%FILENAME%" "%MT5_SCRIPTS%\%FILENAME%" >nul
    echo Copied %FILENAME% to %MT5_SCRIPTS%
) else (
    echo MT5 Scripts folder not found. File saved to current directory only.
    echo You can manually copy it to: %%APPDATA%%\MetaQuotes\Terminal\[ID]\MQL5\Scripts\
)

echo.
echo Done!
pause
