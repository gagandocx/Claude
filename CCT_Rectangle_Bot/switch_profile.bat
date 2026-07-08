@echo off
title CCT Rectangle Bot - Profile Switcher
cls
echo.
echo  ============================================================
echo   CCT Rectangle Bot - Trading Profile Switcher
echo  ============================================================
echo.
echo   Select a trading profile:
echo.
echo   [1] SAFE           - 1%% risk, 3:1 RR, 1 concurrent, 3%% daily limit
echo   [2] MODERATE       - 5%% risk, 3:1 RR, 2 concurrent, 10%% daily limit
echo   [3] AGGRESSIVE     - 10%% risk, 2:1 RR, 3 concurrent, 25%% daily limit
echo   [4] ULTRA          - 25%% risk, 2:1 RR, 3 concurrent, 50%% daily limit
echo.
echo  ============================================================
echo.
set /p choice="  Enter choice (1-4): "

if "%choice%"=="1" goto SAFE
if "%choice%"=="2" goto MODERATE
if "%choice%"=="3" goto AGGRESSIVE
if "%choice%"=="4" goto ULTRA

echo.
echo   Invalid choice. Please enter 1, 2, 3, or 4.
echo.
pause
goto :eof

:SAFE
echo SAFE> "%~dp0live_mt5\active_profile.txt"
echo.
echo   Profile switched to SAFE
echo   (1%% risk, 3:1 RR, 1 concurrent trade, 3%% daily loss limit)
echo.
goto DONE

:MODERATE
echo MODERATE> "%~dp0live_mt5\active_profile.txt"
echo.
echo   Profile switched to MODERATE
echo   (5%% risk, 3:1 RR, 2 concurrent trades, 10%% daily loss limit)
echo.
goto DONE

:AGGRESSIVE
echo AGGRESSIVE> "%~dp0live_mt5\active_profile.txt"
echo.
echo   Profile switched to AGGRESSIVE
echo   (10%% risk, 2:1 RR, 3 concurrent trades, 25%% daily loss limit)
echo.
goto DONE

:ULTRA
echo ULTRA> "%~dp0live_mt5\active_profile.txt"
echo.
echo   Profile switched to ULTRA AGGRESSIVE
echo   (25%% risk, 2:1 RR, 3 concurrent trades, 50%% daily loss limit, Asia session ON)
echo.
goto DONE

:DONE
echo   Next trade will use the new profile automatically.
echo.
pause
