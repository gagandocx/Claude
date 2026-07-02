@echo off
setlocal

set "REPO_URL=https://raw.githubusercontent.com/gagandocx/Claude/main/ExportRealTicks.mq5"
set "DEST_FOLDER=F:\Automation\EA Testing\NeuroX\NeuroX v9.0"
set "FILENAME=ExportRealTicks.mq5"
set "DEST_FILE=%DEST_FOLDER%\%FILENAME%"

echo ============================================
echo   ExportRealTicks.mq5 Updater
echo ============================================
echo.
echo Destination: %DEST_FILE%
echo.

REM Create destination folder if it doesn't exist
if not exist "%DEST_FOLDER%" (
    mkdir "%DEST_FOLDER%"
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Could not create folder: %DEST_FOLDER%
        pause
        exit /b 1
    )
)

echo Downloading %FILENAME% from GitHub...
echo.

REM Method 1: PowerShell with TLS 1.2 and ExecutionPolicy Bypass
echo Trying PowerShell...
powershell -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%REPO_URL%' -OutFile '%DEST_FILE%' -UseBasicParsing"
if %ERRORLEVEL% equ 0 (
    if exist "%DEST_FILE%" (
        echo SUCCESS: Downloaded via PowerShell.
        goto :done
    )
)

echo PowerShell failed, trying curl...
echo.

REM Method 2: curl.exe (built into Windows 10+)
curl.exe -L --ssl-reqd --tls-max 1.2 -o "%DEST_FILE%" "%REPO_URL%"
if %ERRORLEVEL% equ 0 (
    if exist "%DEST_FILE%" (
        echo SUCCESS: Downloaded via curl.
        goto :done
    )
)

echo.
echo ERROR: Download failed with both methods.
echo Please check your internet connection and try again.
pause
exit /b 1

:done
echo.
echo ============================================
echo   File saved to:
echo   %DEST_FILE%
echo ============================================
echo.
pause
