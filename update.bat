@echo off
setlocal enabledelayedexpansion

REM ============================================
REM   NeuroX v9.4 Update Script
REM   Downloads and installs NeuroX EA + ExportRealTicks
REM ============================================

set "DEST_FOLDER=F:\Automation\EA Testing\NeuroX\NeuroX v9.0"

REM --- MT5 Terminal Configuration ---
set "MT5_TERMINAL_ID=930119AA53207C8778B41171FBFFB46F"
set "MT5_BASE=C:\Users\gagan\AppData\Roaming\MetaQuotes\Terminal\%MT5_TERMINAL_ID%"
set "MT5_EXPERTS=%MT5_BASE%\MQL5\Experts\Advisors"
set "MT5_INCLUDE=%MT5_BASE%\MQL5\Include\NeuroX"
set "MT5_SCRIPTS=%MT5_BASE%\MQL5\Scripts"
set "METAEDITOR=%MT5_BASE%\MetaEditor64.exe"

REM --- Download URLs ---
set "NEUROX_REPO=https://github.com/gagandocx/NeuroX-v9/archive/refs/heads/main.zip"
set "NEUROX_ZIP=%DEST_FOLDER%\NeuroX-v9.zip"
set "NEUROX_EXTRACT=%DEST_FOLDER%\NeuroX-v9-main"
set "EXPORT_TICKS_URL=https://raw.githubusercontent.com/gagandocx/Claude/main/ExportRealTicks.mq5"
set "EXPORT_TICKS_FILE=ExportRealTicks.mq5"

echo ============================================
echo   NeuroX v9.4 Updater
echo ============================================
echo.
echo MT5 Terminal: %MT5_BASE%
echo Destination:  %DEST_FOLDER%
echo.

REM --- Step 1: Create directories if needed ---
echo [Step 1] Preparing directories...
if not exist "%DEST_FOLDER%" (
    mkdir "%DEST_FOLDER%"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Could not create folder: %DEST_FOLDER%
        pause
        exit /b 1
    )
)
if not exist "%MT5_EXPERTS%" (
    mkdir "%MT5_EXPERTS%"
)
if not exist "%MT5_INCLUDE%" (
    mkdir "%MT5_INCLUDE%"
)
if not exist "%MT5_SCRIPTS%" (
    mkdir "%MT5_SCRIPTS%"
)
echo   Directories ready.
echo.

REM --- Step 2: Download NeuroX v9 ZIP ---
echo [Step 2] Downloading NeuroX v9 from GitHub...
if exist "%NEUROX_ZIP%" del "%NEUROX_ZIP%"
powershell -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%NEUROX_REPO%' -OutFile '%NEUROX_ZIP%' -UseBasicParsing"
if !ERRORLEVEL! neq 0 (
    echo   PowerShell failed, trying curl...
    curl.exe -L --ssl-reqd -o "%NEUROX_ZIP%" "%NEUROX_REPO%"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to download NeuroX v9 ZIP.
        pause
        exit /b 1
    )
)
if not exist "%NEUROX_ZIP%" (
    echo ERROR: NeuroX ZIP file not found after download.
    pause
    exit /b 1
)
echo   NeuroX v9 ZIP downloaded.
echo.

REM --- Step 2b: Download ExportRealTicks.mq5 ---
echo [Step 2b] Downloading ExportRealTicks.mq5 from GitHub...
powershell -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%EXPORT_TICKS_URL%' -OutFile '%DEST_FOLDER%\%EXPORT_TICKS_FILE%' -UseBasicParsing"
if !ERRORLEVEL! neq 0 (
    echo   PowerShell failed, trying curl...
    curl.exe -L --ssl-reqd -o "%DEST_FOLDER%\%EXPORT_TICKS_FILE%" "%EXPORT_TICKS_URL%"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to download ExportRealTicks.mq5.
        pause
        exit /b 1
    )
)
if not exist "%DEST_FOLDER%\%EXPORT_TICKS_FILE%" (
    echo ERROR: ExportRealTicks.mq5 not found after download.
    pause
    exit /b 1
)
echo   ExportRealTicks.mq5 downloaded.
echo.

REM --- Step 3: Extract NeuroX ZIP ---
echo [Step 3] Extracting NeuroX v9 ZIP...
if exist "%NEUROX_EXTRACT%" rmdir /s /q "%NEUROX_EXTRACT%"
powershell -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%NEUROX_ZIP%' -DestinationPath '%DEST_FOLDER%' -Force"
if !ERRORLEVEL! neq 0 (
    echo ERROR: Failed to extract NeuroX ZIP.
    pause
    exit /b 1
)
echo   Extracted successfully.
echo.

REM --- Step 4: Copy EAs to MT5 Experts folder ---
echo [Step 4] Copying Expert Advisors to MT5...
if exist "%NEUROX_EXTRACT%\Experts\*" (
    xcopy "%NEUROX_EXTRACT%\Experts\*" "%MT5_EXPERTS%\" /s /y /q
    echo   EAs copied to %MT5_EXPERTS%
) else (
    echo   WARNING: No Experts folder found in extracted files.
)
echo.

REM --- Step 4b: Copy Include files to MT5 ---
echo [Step 4b] Copying Include files to MT5...
if exist "%NEUROX_EXTRACT%\Include\*" (
    xcopy "%NEUROX_EXTRACT%\Include\*" "%MT5_INCLUDE%\" /s /y /q
    echo   Include files copied to %MT5_INCLUDE%
) else (
    echo   WARNING: No Include folder found in extracted files.
)
echo.

REM --- Step 4c: Copy ExportRealTicks.mq5 to MT5 Scripts folder ---
echo [Step 4c] Copying ExportRealTicks.mq5 to MT5 Scripts...
copy /y "%DEST_FOLDER%\%EXPORT_TICKS_FILE%" "%MT5_SCRIPTS%\%EXPORT_TICKS_FILE%"
if !ERRORLEVEL! neq 0 (
    echo ERROR: Failed to copy ExportRealTicks.mq5 to MT5 Scripts folder.
    pause
    exit /b 1
)
echo   ExportRealTicks.mq5 copied to %MT5_SCRIPTS%
echo.

REM --- Step 5: Compile EAs using MetaEditor ---
echo [Step 5] Compiling Expert Advisors...
if exist "%METAEDITOR%" (
    for %%f in ("%MT5_EXPERTS%\*.mq5") do (
        echo   Compiling %%~nxf ...
        "%METAEDITOR%" /compile:"%%f" /log /inc:"%MT5_BASE%\MQL5"
    )
    echo   Compilation complete.
) else (
    echo   WARNING: MetaEditor64.exe not found at %METAEDITOR%
    echo   Skipping compilation. Please compile manually from MT5.
)
echo.

REM --- Step 6: Cleanup ---
echo [Step 6] Cleaning up temporary files...
if exist "%NEUROX_ZIP%" del "%NEUROX_ZIP%"
if exist "%NEUROX_EXTRACT%" rmdir /s /q "%NEUROX_EXTRACT%"
echo   Cleanup done.
echo.

REM --- Done ---
echo ============================================
echo   NeuroX v9.4 Update Complete!
echo ============================================
echo.
echo   EAs installed to:       %MT5_EXPERTS%
echo   Includes installed to:  %MT5_INCLUDE%
echo   Scripts installed to:   %MT5_SCRIPTS%
echo.
echo   ExportRealTicks.mq5 is ready in MT5 Scripts.
echo ============================================
echo.
pause
