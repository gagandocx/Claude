@echo off
color 0B
title NeuroX EA Compiler
echo ============================================
echo        NeuroX EA - One-Click Compiler
echo ============================================
echo.

:: --- Configuration ---
set "EA_SOURCE=%~dp0NeuroX_EA.mq5"
set "LOGO_SOURCE=%~dp0neurox_logo.bmp"
set "MT5_EXPERTS=C:\Users\gagan\AppData\Roaming\MetaQuotes\Terminal\EE1261C89A64D41685651B738DC52A84\MQL5\Experts\Advisors\"
set "MT5_IMAGES=C:\Users\gagan\AppData\Roaming\MetaQuotes\Terminal\EE1261C89A64D41685651B738DC52A84\MQL5\Images\"
set "METAEDITOR="

:: --- Find MetaEditor ---
echo [1/4] Locating MetaEditor...
if exist "C:\Program Files\Fusion Markets MetaTrader 5\metaeditor64.exe" (
    set "METAEDITOR=C:\Program Files\Fusion Markets MetaTrader 5\metaeditor64.exe"
) else if exist "C:\Program Files (x86)\Fusion Markets MetaTrader 5\metaeditor64.exe" (
    set "METAEDITOR=C:\Program Files (x86)\Fusion Markets MetaTrader 5\metaeditor64.exe"
) else (
    echo [ERROR] MetaEditor not found in expected locations!
    echo         Checked: C:\Program Files\Fusion Markets MetaTrader 5\
    echo         Checked: C:\Program Files ^(x86^)\Fusion Markets MetaTrader 5\
    echo.
    pause
    exit /b 1
)
echo         Found: %METAEDITOR%
echo.

:: --- Copy EA source ---
echo [2/4] Copying NeuroX_EA.mq5 to MT5 Experts folder...
if not exist "%EA_SOURCE%" (
    echo [ERROR] NeuroX_EA.mq5 not found in current folder!
    pause
    exit /b 1
)
copy /Y "%EA_SOURCE%" "%MT5_EXPERTS%" >nul
if %errorlevel% equ 0 (
    echo         Done.
) else (
    echo [ERROR] Failed to copy EA file!
    pause
    exit /b 1
)
echo.

:: --- Copy logo BMP if it exists ---
echo [3/4] Copying neurox_logo.bmp to MT5 Images folder...
if exist "%LOGO_SOURCE%" (
    if not exist "%MT5_IMAGES%" mkdir "%MT5_IMAGES%"
    copy /Y "%LOGO_SOURCE%" "%MT5_IMAGES%" >nul
    if %errorlevel% equ 0 (
        echo         Done.
    ) else (
        echo [WARNING] Failed to copy logo file.
    )
) else (
    echo         Skipped (neurox_logo.bmp not found).
)
echo.

:: --- Compile EA ---
echo [4/4] Compiling NeuroX_EA.mq5...
set "COMPILE_TARGET=%MT5_EXPERTS%NeuroX_EA.mq5"
set "LOG_FILE=%MT5_EXPERTS%NeuroX_EA.log"

"%METAEDITOR%" /compile:"%COMPILE_TARGET%" /log
echo.

:: --- Check compilation result ---
echo ============================================
if exist "%LOG_FILE%" (
    findstr /i "error" "%LOG_FILE%" >nul
    if %errorlevel% equ 0 (
        echo    COMPILATION FAILED - Errors detected:
        echo ============================================
        echo.
        findstr /i "error" "%LOG_FILE%"
    ) else (
        echo    COMPILATION SUCCESSFUL
        echo ============================================
        echo.
        echo    NeuroX EA is ready to use in MT5!
    )
) else (
    echo    WARNING: Log file not found.
    echo    Compilation status unknown.
    echo ============================================
)

echo.
pause
