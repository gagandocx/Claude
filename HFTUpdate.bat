@echo off
echo ============================================
echo   HFT Scalper v2 - File Updater
echo ============================================
echo.

set "TARGET=F:\Automation\EA Testing\HFT\HFT_v2"
set "BASE_URL=https://raw.githubusercontent.com/gagandocx/Claude/hft-scalper-v2"

echo Target folder: %TARGET%
echo Source: %BASE_URL%
echo.

:: Create subdirectories
echo Creating directories...
mkdir "%TARGET%\hft_scalper" 2>nul
mkdir "%TARGET%\hft_scalper\strategies" 2>nul
mkdir "%TARGET%\hft_scalper\output" 2>nul
mkdir "%TARGET%\hft_scalper\results" 2>nul
echo Directories ready.
echo.

:: Download hft_scalper root files
echo Downloading hft_scalper/__init__.py...
curl -sL "%BASE_URL%/hft_scalper/__init__.py" -o "%TARGET%\hft_scalper\__init__.py"

echo Downloading hft_scalper/analyze.py...
curl -sL "%BASE_URL%/hft_scalper/analyze.py" -o "%TARGET%\hft_scalper\analyze.py"

echo Downloading hft_scalper/backtest_engine.py...
curl -sL "%BASE_URL%/hft_scalper/backtest_engine.py" -o "%TARGET%\hft_scalper\backtest_engine.py"

echo Downloading hft_scalper/data_loader.py...
curl -sL "%BASE_URL%/hft_scalper/data_loader.py" -o "%TARGET%\hft_scalper\data_loader.py"

echo Downloading hft_scalper/live_trader.py...
curl -sL "%BASE_URL%/hft_scalper/live_trader.py" -o "%TARGET%\hft_scalper\live_trader.py"

echo Downloading hft_scalper/microstructure_analysis.py...
curl -sL "%BASE_URL%/hft_scalper/microstructure_analysis.py" -o "%TARGET%\hft_scalper\microstructure_analysis.py"

echo Downloading hft_scalper/optimizer.py...
curl -sL "%BASE_URL%/hft_scalper/optimizer.py" -o "%TARGET%\hft_scalper\optimizer.py"

echo Downloading hft_scalper/run_aggressive_backtest.py...
curl -sL "%BASE_URL%/hft_scalper/run_aggressive_backtest.py" -o "%TARGET%\hft_scalper\run_aggressive_backtest.py"

echo Downloading hft_scalper/run_backtest.py...
curl -sL "%BASE_URL%/hft_scalper/run_backtest.py" -o "%TARGET%\hft_scalper\run_backtest.py"

echo Downloading hft_scalper/run_ensemble_backtest.py...
curl -sL "%BASE_URL%/hft_scalper/run_ensemble_backtest.py" -o "%TARGET%\hft_scalper\run_ensemble_backtest.py"

:: Download strategies
echo Downloading hft_scalper/strategies/__init__.py...
curl -sL "%BASE_URL%/hft_scalper/strategies/__init__.py" -o "%TARGET%\hft_scalper\strategies\__init__.py"

echo Downloading hft_scalper/strategies/base.py...
curl -sL "%BASE_URL%/hft_scalper/strategies/base.py" -o "%TARGET%\hft_scalper\strategies\base.py"

echo Downloading hft_scalper/strategies/ensemble.py...
curl -sL "%BASE_URL%/hft_scalper/strategies/ensemble.py" -o "%TARGET%\hft_scalper\strategies\ensemble.py"

echo Downloading hft_scalper/strategies/mean_reversion.py...
curl -sL "%BASE_URL%/hft_scalper/strategies/mean_reversion.py" -o "%TARGET%\hft_scalper\strategies\mean_reversion.py"

echo Downloading hft_scalper/strategies/momentum_mtf.py...
curl -sL "%BASE_URL%/hft_scalper/strategies/momentum_mtf.py" -o "%TARGET%\hft_scalper\strategies\momentum_mtf.py"

echo Downloading hft_scalper/strategies/order_flow.py...
curl -sL "%BASE_URL%/hft_scalper/strategies/order_flow.py" -o "%TARGET%\hft_scalper\strategies\order_flow.py"

echo Downloading hft_scalper/strategies/spread_fade.py...
curl -sL "%BASE_URL%/hft_scalper/strategies/spread_fade.py" -o "%TARGET%\hft_scalper\strategies\spread_fade.py"

echo Downloading hft_scalper/strategies/volatility_breakout.py...
curl -sL "%BASE_URL%/hft_scalper/strategies/volatility_breakout.py" -o "%TARGET%\hft_scalper\strategies\volatility_breakout.py"

:: Download output files
echo Downloading hft_scalper/output/HFT_Scalper_Pro.mq5...
curl -sL "%BASE_URL%/hft_scalper/output/HFT_Scalper_Pro.mq5" -o "%TARGET%\hft_scalper\output\HFT_Scalper_Pro.mq5"

echo Downloading hft_scalper/output/EA_README.md...
curl -sL "%BASE_URL%/hft_scalper/output/EA_README.md" -o "%TARGET%\hft_scalper\output\EA_README.md"

echo Downloading hft_scalper/output/backtest_summary.md...
curl -sL "%BASE_URL%/hft_scalper/output/backtest_summary.md" -o "%TARGET%\hft_scalper\output\backtest_summary.md"

:: Download other files
echo Downloading hft_scalper/LIVE_TRADING_README.md...
curl -sL "%BASE_URL%/hft_scalper/LIVE_TRADING_README.md" -o "%TARGET%\hft_scalper\LIVE_TRADING_README.md"

echo Downloading hft_scalper/results/aggressive_results.json...
curl -sL "%BASE_URL%/hft_scalper/results/aggressive_results.json" -o "%TARGET%\hft_scalper\results\aggressive_results.json"

echo.
echo ============================================
echo   Update complete! All files downloaded to F:\Automation\EA Testing\HFT\HFT_v2
echo ============================================
echo.
pause
