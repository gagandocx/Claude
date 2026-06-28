"""
=============================================================
  Test Suite: Performance Dashboard
  
  Comprehensive tests for the prop trading desk dashboard:
  - PerformanceTracker metric computations
  - Edge cases (no trades, all wins, all losses)
  - Per-model and per-regime breakdowns
  - DashboardRenderer output validation
  - Integration with trade recording pipeline
=============================================================
"""

import math
import os
import sys
import tempfile
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard.performance_tracker import PerformanceTracker, TradeRecord
from dashboard.dashboard_renderer import DashboardRenderer


# ─────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def tracker():
    """Fresh PerformanceTracker instance."""
    return PerformanceTracker(min_trades_for_stats=3)


@pytest.fixture
def populated_tracker():
    """Tracker with a realistic trade sequence."""
    t = PerformanceTracker(min_trades_for_stats=3)
    trades = [
        ("T001", 50.0, "transformer", "trending", "BUY"),
        ("T002", -20.0, "lstm", "ranging", "SELL"),
        ("T003", 30.0, "gradient_boost", "trending", "BUY"),
        ("T004", 80.0, "transformer", "volatile", "BUY"),
        ("T005", -40.0, "lstm", "crash", "SELL"),
        ("T006", 25.0, "gradient_boost", "ranging", "BUY"),
        ("T007", -15.0, "transformer", "ranging", "SELL"),
        ("T008", 60.0, "ensemble", "trending", "BUY"),
        ("T009", -10.0, "lstm", "volatile", "SELL"),
        ("T010", 45.0, "transformer", "trending", "BUY"),
    ]
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    for i, (tid, pnl, model, regime, direction) in enumerate(trades):
        entry = base_time + timedelta(hours=i * 2)
        exit_ = entry + timedelta(hours=1, minutes=30)
        trade = TradeRecord(
            trade_id=tid,
            entry_time=entry.isoformat(),
            exit_time=exit_.isoformat(),
            direction=direction,
            pnl=pnl,
            model=model,
            regime=regime,
        )
        t.record_trade(trade)
    return t


@pytest.fixture
def all_wins_tracker():
    """Tracker where every trade is a winner."""
    t = PerformanceTracker(min_trades_for_stats=3)
    for i in range(5):
        t.record_trade_simple(
            trade_id=f"W{i:03d}",
            pnl=float(20 + i * 5),
            model="transformer",
            regime="trending",
        )
    return t


@pytest.fixture
def all_losses_tracker():
    """Tracker where every trade is a loser."""
    t = PerformanceTracker(min_trades_for_stats=3)
    for i in range(5):
        t.record_trade_simple(
            trade_id=f"L{i:03d}",
            pnl=float(-15 - i * 3),
            model="lstm",
            regime="crash",
        )
    return t


# ─────────────────────────────────────────────
#  TEST: EDGE CASES
# ─────────────────────────────────────────────

class TestEdgeCases:
    """Test behavior with no trades and edge conditions."""

    def test_no_trades_win_rate(self, tracker):
        assert tracker.win_rate() == 0.0

    def test_no_trades_profit_factor(self, tracker):
        assert tracker.profit_factor() == 0.0

    def test_no_trades_sharpe(self, tracker):
        assert tracker.sharpe_ratio() == 0.0

    def test_no_trades_sortino(self, tracker):
        assert tracker.sortino_ratio() == 0.0

    def test_no_trades_max_drawdown(self, tracker):
        assert tracker.max_drawdown() == 0.0

    def test_no_trades_expectancy(self, tracker):
        assert tracker.expectancy() == 0.0

    def test_no_trades_avg_hold_time(self, tracker):
        assert tracker.avg_hold_time() == 0.0

    def test_no_trades_best_worst(self, tracker):
        assert tracker.best_trade() == 0.0
        assert tracker.worst_trade() == 0.0

    def test_no_trades_total_trades(self, tracker):
        assert tracker.total_trades == 0

    def test_no_trades_net_profit(self, tracker):
        assert tracker.net_profit() == 0.0

    def test_single_trade(self, tracker):
        tracker.record_trade_simple("T1", 100.0, "transformer", "trending")
        assert tracker.total_trades == 1
        assert tracker.win_rate() == 1.0
        assert tracker.net_profit() == 100.0
        assert tracker.best_trade() == 100.0

    def test_single_losing_trade(self, tracker):
        tracker.record_trade_simple("T1", -50.0, "lstm", "crash")
        assert tracker.win_rate() == 0.0
        assert tracker.profit_factor() == 0.0
        assert tracker.max_drawdown() == 50.0


# ─────────────────────────────────────────────
#  TEST: ALL WINS
# ─────────────────────────────────────────────

class TestAllWins:
    """Test behavior when every trade is profitable."""

    def test_win_rate_is_100(self, all_wins_tracker):
        assert all_wins_tracker.win_rate() == 1.0

    def test_profit_factor_is_inf(self, all_wins_tracker):
        assert all_wins_tracker.profit_factor() == float('inf')

    def test_no_drawdown(self, all_wins_tracker):
        assert all_wins_tracker.max_drawdown() == 0.0

    def test_consecutive_wins(self, all_wins_tracker):
        assert all_wins_tracker.consecutive_wins() == 5

    def test_consecutive_losses_zero(self, all_wins_tracker):
        assert all_wins_tracker.consecutive_losses() == 0

    def test_avg_loss_zero(self, all_wins_tracker):
        assert all_wins_tracker.avg_loss() == 0.0

    def test_sortino_inf(self, all_wins_tracker):
        # No downside returns = infinite Sortino
        assert all_wins_tracker.sortino_ratio() == float('inf')

    def test_recovery_factor_inf(self, all_wins_tracker):
        # No drawdown = infinite recovery factor
        assert all_wins_tracker.recovery_factor() == float('inf')


# ─────────────────────────────────────────────
#  TEST: ALL LOSSES
# ─────────────────────────────────────────────

class TestAllLosses:
    """Test behavior when every trade is a loss."""

    def test_win_rate_is_zero(self, all_losses_tracker):
        assert all_losses_tracker.win_rate() == 0.0

    def test_profit_factor_is_zero(self, all_losses_tracker):
        assert all_losses_tracker.profit_factor() == 0.0

    def test_drawdown_equals_total_loss(self, all_losses_tracker):
        total_loss = abs(sum(t.pnl for t in all_losses_tracker._trades))
        assert all_losses_tracker.max_drawdown() == total_loss

    def test_consecutive_losses(self, all_losses_tracker):
        assert all_losses_tracker.consecutive_losses() == 5

    def test_consecutive_wins_zero(self, all_losses_tracker):
        assert all_losses_tracker.consecutive_wins() == 0

    def test_avg_win_zero(self, all_losses_tracker):
        assert all_losses_tracker.avg_win() == 0.0

    def test_net_profit_negative(self, all_losses_tracker):
        assert all_losses_tracker.net_profit() < 0


# ─────────────────────────────────────────────
#  TEST: CORE METRICS COMPUTATION
# ─────────────────────────────────────────────

class TestCoreMetrics:
    """Test correctness of core metric calculations."""

    def test_win_rate_computation(self, populated_tracker):
        # 10 trades: T001(+), T002(-), T003(+), T004(+), T005(-),
        #            T006(+), T007(-), T008(+), T009(-), T010(+)
        # Winners: 6, Losers: 4
        assert populated_tracker.win_rate() == 0.6

    def test_profit_factor_computation(self, populated_tracker):
        # Gross wins: 50+30+80+25+60+45 = 290
        # Gross losses: |(-20)+(-40)+(-15)+(-10)| = 85
        # PF = 290 / 85 = 3.4117...
        expected_pf = 290.0 / 85.0
        assert abs(populated_tracker.profit_factor() - expected_pf) < 0.001

    def test_net_profit(self, populated_tracker):
        # Sum of all PnLs: 50-20+30+80-40+25-15+60-10+45 = 205
        assert abs(populated_tracker.net_profit() - 205.0) < 0.001

    def test_avg_win(self, populated_tracker):
        # Winning PnLs: 50, 30, 80, 25, 60, 45 => avg = 290/6
        expected = 290.0 / 6.0
        assert abs(populated_tracker.avg_win() - expected) < 0.001

    def test_avg_loss(self, populated_tracker):
        # Losing PnLs: -20, -40, -15, -10 => avg = -85/4
        expected = -85.0 / 4.0
        assert abs(populated_tracker.avg_loss() - expected) < 0.001

    def test_expectancy(self, populated_tracker):
        # E = (0.6 * 290/6) + (0.4 * -85/4) = 29.0 + (-8.5) = 20.5
        wr = 0.6
        avg_w = 290.0 / 6.0
        avg_l = -85.0 / 4.0
        expected = (wr * avg_w) + ((1 - wr) * avg_l)
        assert abs(populated_tracker.expectancy() - expected) < 0.001

    def test_best_trade(self, populated_tracker):
        assert populated_tracker.best_trade() == 80.0

    def test_worst_trade(self, populated_tracker):
        assert populated_tracker.worst_trade() == -40.0

    def test_total_trades(self, populated_tracker):
        assert populated_tracker.total_trades == 10

    def test_total_wins_losses(self, populated_tracker):
        assert populated_tracker.total_wins() == 6
        assert populated_tracker.total_losses() == 4

    def test_payoff_ratio(self, populated_tracker):
        # avg_win / abs(avg_loss) = (290/6) / (85/4) = 48.33 / 21.25 = 2.274...
        expected = (290.0 / 6.0) / (85.0 / 4.0)
        assert abs(populated_tracker.payoff_ratio() - expected) < 0.01

    def test_max_drawdown(self, populated_tracker):
        # Equity sequence: 0, 50, 30, 60, 140, 100, 125, 110, 170, 160, 205
        # Peak at 140, drops to 100 = dd 40
        # Then peak at 170, drops to 160 = dd 10
        # Max dd = 40
        assert abs(populated_tracker.max_drawdown() - 40.0) < 0.001

    def test_recovery_factor(self, populated_tracker):
        # net_profit / max_dd = 205 / 40 = 5.125
        expected = 205.0 / 40.0
        assert abs(populated_tracker.recovery_factor() - expected) < 0.001

    def test_consecutive_wins(self, populated_tracker):
        # T001(+) -> 1 win streak
        # T003(+), T004(+) -> 2 win streak
        # T006(+) -> 1
        # T008(+) -> 1
        # T010(+) -> 1
        # Max consecutive wins = 2
        assert populated_tracker.consecutive_wins() == 2

    def test_consecutive_losses(self, populated_tracker):
        # Only individual losses: T002(-), T005(-), T007(-), T009(-)
        # None are consecutive (T005 follows T004 which is a win)
        # Wait... T002 is single, T005 is single, T007 is single, T009 is single
        assert populated_tracker.consecutive_losses() == 1

    def test_avg_hold_time(self, populated_tracker):
        # Each trade has 1.5 hours hold time
        assert abs(populated_tracker.avg_hold_time() - 1.5) < 0.01

    def test_sharpe_ratio_is_positive_for_winning_system(self, populated_tracker):
        # Winning system should have positive Sharpe
        assert populated_tracker.sharpe_ratio() > 0

    def test_sortino_ratio_is_positive_for_winning_system(self, populated_tracker):
        # Winning system should have positive Sortino
        assert populated_tracker.sortino_ratio() > 0

    def test_sharpe_ratio_formula(self):
        """Verify Sharpe calculation matches manual computation."""
        tracker = PerformanceTracker(min_trades_for_stats=2)
        # Simple series: [10, 20, 30]
        for i, pnl in enumerate([10.0, 20.0, 30.0]):
            tracker.record_trade_simple(f"T{i}", pnl, "transformer", "trending")

        returns = [10.0, 20.0, 30.0]
        mean_r = sum(returns) / 3
        # Population std
        variance = sum((x - mean_r) ** 2 for x in returns) / 3
        std_r = math.sqrt(variance)

        trades_per_year = 252 * 3
        rf_per_trade = 0.05 / trades_per_year
        expected_sharpe = (mean_r - rf_per_trade) / std_r * math.sqrt(trades_per_year)

        assert abs(tracker.sharpe_ratio() - expected_sharpe) < 0.01


# ─────────────────────────────────────────────
#  TEST: PER-MODEL BREAKDOWN
# ─────────────────────────────────────────────

class TestPerModelBreakdown:
    """Test per-model performance tracking."""

    def test_transformer_stats(self, populated_tracker):
        stats = populated_tracker.get_model_stats("transformer")
        # Transformer trades: T001(+50), T004(+80), T007(-15), T010(+45)
        assert stats["trade_count"] == 4
        assert stats["win_rate"] == 0.75  # 3/4
        gross_wins = 50 + 80 + 45  # 175
        gross_losses = 15
        assert abs(stats["profit_factor"] - gross_wins / gross_losses) < 0.001
        assert abs(stats["total_pnl"] - (50 + 80 - 15 + 45)) < 0.001

    def test_lstm_stats(self, populated_tracker):
        stats = populated_tracker.get_model_stats("lstm")
        # LSTM trades: T002(-20), T005(-40), T009(-10)
        assert stats["trade_count"] == 3
        assert stats["win_rate"] == 0.0
        assert stats["profit_factor"] == 0.0

    def test_gradient_boost_stats(self, populated_tracker):
        stats = populated_tracker.get_model_stats("gradient_boost")
        # GB trades: T003(+30), T006(+25)
        assert stats["trade_count"] == 2
        assert stats["win_rate"] == 1.0
        assert stats["profit_factor"] == float('inf')

    def test_ensemble_stats(self, populated_tracker):
        stats = populated_tracker.get_model_stats("ensemble")
        # Ensemble trades: T008(+60)
        assert stats["trade_count"] == 1
        assert stats["win_rate"] == 1.0
        assert stats["total_pnl"] == 60.0

    def test_nonexistent_model(self, populated_tracker):
        stats = populated_tracker.get_model_stats("random_forest")
        assert stats["trade_count"] == 0
        assert stats["win_rate"] == 0.0

    def test_all_model_stats(self, populated_tracker):
        all_stats = populated_tracker.get_all_model_stats()
        assert "transformer" in all_stats
        assert "lstm" in all_stats
        assert "gradient_boost" in all_stats
        assert "ensemble" in all_stats


# ─────────────────────────────────────────────
#  TEST: PER-REGIME BREAKDOWN
# ─────────────────────────────────────────────

class TestPerRegimeBreakdown:
    """Test per-regime performance tracking."""

    def test_trending_stats(self, populated_tracker):
        stats = populated_tracker.get_regime_stats("trending")
        # Trending trades: T001(+50), T003(+30), T008(+60), T010(+45)
        assert stats["trade_count"] == 4
        assert stats["win_rate"] == 1.0
        assert abs(stats["total_pnl"] - 185.0) < 0.001

    def test_ranging_stats(self, populated_tracker):
        stats = populated_tracker.get_regime_stats("ranging")
        # Ranging trades: T002(-20), T006(+25), T007(-15)
        assert stats["trade_count"] == 3
        expected_wr = 1.0 / 3.0
        assert abs(stats["win_rate"] - expected_wr) < 0.001

    def test_volatile_stats(self, populated_tracker):
        stats = populated_tracker.get_regime_stats("volatile")
        # Volatile trades: T004(+80), T009(-10)
        assert stats["trade_count"] == 2
        assert stats["win_rate"] == 0.5

    def test_crash_stats(self, populated_tracker):
        stats = populated_tracker.get_regime_stats("crash")
        # Crash trades: T005(-40)
        assert stats["trade_count"] == 1
        assert stats["win_rate"] == 0.0

    def test_nonexistent_regime(self, populated_tracker):
        stats = populated_tracker.get_regime_stats("sideways")
        assert stats["trade_count"] == 0

    def test_all_regime_stats(self, populated_tracker):
        all_stats = populated_tracker.get_all_regime_stats()
        assert "trending" in all_stats
        assert "ranging" in all_stats
        assert "volatile" in all_stats
        assert "crash" in all_stats


# ─────────────────────────────────────────────
#  TEST: DASHBOARD RENDERER
# ─────────────────────────────────────────────

class TestDashboardRenderer:
    """Test dashboard rendering in all formats."""

    def test_console_render_no_trades(self, tracker):
        renderer = DashboardRenderer(tracker, use_colors=False)
        output = renderer.render_console()
        assert "HF SCALPER DASHBOARD" in output
        assert "Net Profit" in output
        assert "Win Rate" in output

    def test_console_render_with_trades(self, populated_tracker):
        renderer = DashboardRenderer(populated_tracker, use_colors=False)
        output = renderer.render_console()
        assert "PROFIT & LOSS" in output
        assert "RISK ANALYTICS" in output
        assert "MODEL ALPHA" in output
        assert "REGIME BREAKDOWN" in output
        assert "Sharpe Ratio" in output
        assert "Sortino Ratio" in output
        assert "Profit Factor" in output
        assert "Max Drawdown" in output

    def test_console_render_with_colors(self, populated_tracker):
        renderer = DashboardRenderer(populated_tracker, use_colors=True)
        output = renderer.render_console()
        # Should contain ANSI escape codes
        assert "\033[" in output

    def test_console_render_without_colors(self, populated_tracker):
        renderer = DashboardRenderer(populated_tracker, use_colors=False)
        output = renderer.render_console()
        # Should NOT contain ANSI escape codes
        assert "\033[" not in output

    def test_html_render_creates_file(self, populated_tracker):
        renderer = DashboardRenderer(populated_tracker, use_colors=False)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name
        try:
            result = renderer.render_html(output_path)
            assert result == output_path
            assert os.path.exists(output_path)
            with open(output_path, 'r') as f:
                html = f.read()
            assert "<!DOCTYPE html>" in html
            assert "HF Scalper" in html
            assert "Sharpe Ratio" in html
            assert "Model Performance" in html
            assert "Regime Performance" in html
        finally:
            os.unlink(output_path)

    def test_html_render_creates_directory(self, populated_tracker):
        renderer = DashboardRenderer(populated_tracker, use_colors=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "sub", "dir", "report.html")
            result = renderer.render_html(output_path)
            assert os.path.exists(output_path)

    def test_html_contains_equity_curve_data(self, populated_tracker):
        renderer = DashboardRenderer(populated_tracker, use_colors=False)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name
        try:
            renderer.render_html(output_path)
            with open(output_path, 'r') as f:
                html = f.read()
            # Equity curve should be JSON embedded in the HTML
            assert "Equity Curve Data" in html
            # First point is always 0.0
            assert "0.0" in html
        finally:
            os.unlink(output_path)

    def test_log_render(self, populated_tracker):
        renderer = DashboardRenderer(populated_tracker, use_colors=False)
        log_data = renderer.render_log()
        assert log_data["type"] == "performance_snapshot"
        assert "metrics" in log_data
        assert "per_model" in log_data
        assert "per_regime" in log_data
        metrics = log_data["metrics"]
        assert metrics["total_trades"] == 10
        assert metrics["win_rate"] == 0.6
        assert metrics["net_profit"] == 205.0

    def test_log_render_no_trades(self, tracker):
        renderer = DashboardRenderer(tracker, use_colors=False)
        log_data = renderer.render_log()
        assert log_data["metrics"]["total_trades"] == 0
        assert log_data["metrics"]["net_profit"] == 0.0


# ─────────────────────────────────────────────
#  TEST: RESET AND FULL SUMMARY
# ─────────────────────────────────────────────

class TestTrackerUtilities:
    """Test utility methods and data access."""

    def test_reset(self, populated_tracker):
        assert populated_tracker.total_trades == 10
        populated_tracker.reset()
        assert populated_tracker.total_trades == 0
        assert populated_tracker.net_profit() == 0.0
        assert populated_tracker.max_drawdown() == 0.0
        assert populated_tracker.win_rate() == 0.0

    def test_full_summary_keys(self, populated_tracker):
        summary = populated_tracker.get_full_summary()
        expected_keys = [
            "total_trades", "net_profit", "win_rate", "loss_rate",
            "profit_factor", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "recovery_factor", "expectancy",
            "avg_win", "avg_loss", "payoff_ratio", "best_trade",
            "worst_trade", "consecutive_wins", "consecutive_losses",
            "avg_hold_time_hours", "total_wins", "total_losses",
            "equity_curve", "per_model", "per_regime",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_get_pnl_series(self, populated_tracker):
        pnls = populated_tracker.get_pnl_series()
        assert len(pnls) == 10
        assert pnls[0] == 50.0
        assert pnls[1] == -20.0

    def test_has_sufficient_data(self, tracker):
        assert not tracker.has_sufficient_data
        for i in range(3):
            tracker.record_trade_simple(f"T{i}", 10.0, "transformer", "trending")
        assert tracker.has_sufficient_data

    def test_record_trade_simple(self, tracker):
        tracker.record_trade_simple(
            "T001", 42.0, "transformer", "trending",
            direction="BUY", hold_bars=5, confidence=0.8
        )
        assert tracker.total_trades == 1
        assert tracker.net_profit() == 42.0
        trade = tracker._trades[0]
        assert trade.model == "transformer"
        assert trade.regime == "trending"
        assert trade.direction == "BUY"
        assert trade.hold_bars == 5
        assert trade.confidence == 0.8

    def test_equity_curve_tracking(self, tracker):
        tracker.record_trade_simple("T1", 100.0, "transformer", "trending")
        tracker.record_trade_simple("T2", -30.0, "lstm", "ranging")
        tracker.record_trade_simple("T3", 50.0, "gradient_boost", "trending")
        # Equity: [0, 100, 70, 120]
        assert tracker._equity_curve == [0.0, 100.0, 70.0, 120.0]


# ─────────────────────────────────────────────
#  TEST: DRAWDOWN SCENARIOS
# ─────────────────────────────────────────────

class TestDrawdownScenarios:
    """Test max drawdown calculation in various scenarios."""

    def test_simple_drawdown(self, tracker):
        tracker.record_trade_simple("T1", 100.0, "transformer", "trending")
        tracker.record_trade_simple("T2", -60.0, "lstm", "crash")
        # Peak at 100, trough at 40 => dd = 60
        assert tracker.max_drawdown() == 60.0

    def test_multiple_drawdowns_takes_max(self, tracker):
        # First run up then small dd
        tracker.record_trade_simple("T1", 100.0, "transformer", "trending")
        tracker.record_trade_simple("T2", -20.0, "lstm", "ranging")
        # New peak then bigger dd
        tracker.record_trade_simple("T3", 50.0, "transformer", "trending")
        tracker.record_trade_simple("T4", -80.0, "lstm", "crash")
        # Equity: [0, 100, 80, 130, 50]
        # DD1: 100-80 = 20
        # DD2: 130-50 = 80
        assert tracker.max_drawdown() == 80.0

    def test_drawdown_with_recovery(self, tracker):
        tracker.record_trade_simple("T1", 100.0, "transformer", "trending")
        tracker.record_trade_simple("T2", -50.0, "lstm", "crash")
        tracker.record_trade_simple("T3", 200.0, "transformer", "trending")
        # Peak at 100, trough at 50 => dd 50
        # Then new peak at 250
        assert tracker.max_drawdown() == 50.0


# ─────────────────────────────────────────────
#  TEST: CONFIG IMPORT
# ─────────────────────────────────────────────

class TestDashboardConfig:
    """Test DashboardConfig import and defaults."""

    def test_import(self):
        from config.settings import DashboardConfig
        config = DashboardConfig()
        assert config.update_interval_trades == 1
        assert config.min_trades_for_stats == 10
        assert config.console_refresh_seconds == 60
        assert config.track_per_model is True
        assert config.track_per_regime is True
        assert config.enable_console is True
        assert config.enable_html_report is True
        assert config.enable_log_output is True
        assert config.use_colors is True
        assert config.html_output_path == "dashboard/report.html"
