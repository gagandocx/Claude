"""
=============================================================
  Dashboard Renderer - Professional Trading Desk Display
  
  Renders performance metrics in multiple formats:
  - Console (ASCII formatted like an institutional trading terminal)
  - HTML (standalone report with tables and equity curve data)
  - Log (structured output for monitoring systems)
  
  Styled after prop trading desk dashboards where clarity and
  information density are critical for real-time decision making.
=============================================================
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional

from .performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


# ANSI color codes for terminal output
class Colors:
    """Terminal color codes for futuristic neon display."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    BG_DARK = "\033[40m"
    # Extended neon palette
    NEON_GREEN = "\033[38;5;46m"
    NEON_CYAN = "\033[38;5;51m"
    NEON_PINK = "\033[38;5;198m"
    NEON_ORANGE = "\033[38;5;208m"
    NEON_PURPLE = "\033[38;5;141m"
    BG_NAVY = "\033[48;5;17m"


class DashboardRenderer:
    """
    Professional trading desk dashboard renderer.
    
    Outputs performance data in formats optimized for different contexts:
    - Console: real-time monitoring during live trading sessions
    - HTML: end-of-day/week reports for strategy review
    - Log: structured data for automated monitoring pipelines
    """

    # Dashboard width in characters
    CONSOLE_WIDTH = 76

    def __init__(self, tracker: PerformanceTracker, use_colors: bool = True):
        """
        Initialize renderer with a performance tracker reference.
        
        Args:
            tracker: PerformanceTracker instance to pull metrics from
            use_colors: Whether to use ANSI colors in console output
        """
        self._tracker = tracker
        self._use_colors = use_colors

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self._use_colors:
            return text
        return f"{color}{text}{Colors.RESET}"

    def _pnl_color(self, value: float) -> str:
        """Color a PnL value: green for positive, red for negative."""
        formatted = f"{value:+.2f}"
        if value > 0:
            return self._color(formatted, Colors.GREEN)
        elif value < 0:
            return self._color(formatted, Colors.RED)
        return formatted

    def _pct_color(self, value: float) -> str:
        """Color a percentage value."""
        formatted = f"{value:.1f}%"
        if value > 50:
            return self._color(formatted, Colors.GREEN)
        elif value < 40:
            return self._color(formatted, Colors.RED)
        return self._color(formatted, Colors.YELLOW)

    def _ratio_color(self, value: float, good_threshold: float = 1.5) -> str:
        """Color a ratio value based on quality threshold."""
        if value == float('inf'):
            return self._color("INF", Colors.GREEN)
        formatted = f"{value:.2f}"
        if value >= good_threshold:
            return self._color(formatted, Colors.GREEN)
        elif value >= 1.0:
            return self._color(formatted, Colors.YELLOW)
        else:
            return self._color(formatted, Colors.RED)

    def _header_line(self, title: str) -> str:
        """Create a futuristic formatted header line with neon border."""
        w = self.CONSOLE_WIDTH
        top_border = self._color("+" + "=" * (w - 2) + "+", Colors.NEON_CYAN)
        padding = (w - len(title) - 6) // 2
        header = f"|{'>' * padding} {title} {'<' * (w - padding - len(title) - 6)}|"
        return top_border + "\n" + self._color(header, Colors.NEON_CYAN)

    def _section_line(self, title: str) -> str:
        """Create a futuristic section separator with arrows."""
        w = self.CONSOLE_WIDTH
        padding = (w - len(title) - 8) // 2
        return self._color(
            f"  {'>' * 3} [ {title} ] {'<' * 3}{' ' * (w - padding - len(title) - 12)}",
            Colors.NEON_PURPLE
        )

    def _metric_line(self, label: str, value: str, width: int = 30) -> str:
        """Format a single metric as label : value."""
        dots = "." * (width - len(label))
        return f"  {label} {self._color(dots, Colors.DIM)} {value}"

    def render_console(self) -> str:
        """
        Render the full performance dashboard as futuristic ASCII art.
        
        Returns a multi-line string formatted for terminal display,
        styled like a next-gen prop trading desk HUD with neon accents.
        """
        summary = self._tracker.get_full_summary()
        lines = []
        w = self.CONSOLE_WIDTH

        # Top border with futuristic design
        lines.append("")
        lines.append(self._color("+" + "=" * (w - 2) + "+", Colors.NEON_CYAN))
        lines.append(self._color("|" + " " * (w - 2) + "|", Colors.NEON_CYAN))
        title = "PYTHON ML BRIDGE // HF SCALPER DASHBOARD"
        pad = (w - len(title) - 4) // 2
        lines.append(self._color(
            f"|{' ' * pad}{title}{' ' * (w - pad - len(title) - 2)}|", Colors.NEON_CYAN
        ))
        subtitle = f"[LIVE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | M1 XAUUSD"
        pad2 = (w - len(subtitle) - 4) // 2
        lines.append(self._color(
            f"|{' ' * pad2}{subtitle}{' ' * (w - pad2 - len(subtitle) - 2)}|", Colors.DIM
        ))
        lines.append(self._color("|" + " " * (w - 2) + "|", Colors.NEON_CYAN))
        lines.append(self._color("+" + "=" * (w - 2) + "+", Colors.NEON_CYAN))
        lines.append("")

        # Core P&L section
        lines.append(self._section_line("PROFIT & LOSS"))
        lines.append(self._metric_line(
            "Net Profit", self._pnl_color(summary["net_profit"])
        ))
        lines.append(self._metric_line(
            "Total Trades", self._color(str(summary["total_trades"]), Colors.NEON_CYAN)
        ))
        lines.append(self._metric_line(
            "Winners / Losers",
            self._color(str(summary['total_wins']), Colors.GREEN) + " / " +
            self._color(str(summary['total_losses']), Colors.RED)
        ))
        lines.append(self._metric_line(
            "Win Rate", self._pct_color(summary["win_rate"] * 100)
        ))
        lines.append(self._metric_line(
            "Expectancy/Trade", self._pnl_color(summary["expectancy"])
        ))
        lines.append("")

        # Risk Metrics
        lines.append(self._section_line("RISK ANALYTICS"))
        lines.append(self._metric_line(
            "Sharpe Ratio", self._ratio_color(summary["sharpe_ratio"], 1.0)
        ))
        lines.append(self._metric_line(
            "Sortino Ratio", self._ratio_color(summary["sortino_ratio"], 1.5)
        ))
        lines.append(self._metric_line(
            "Profit Factor", self._ratio_color(summary["profit_factor"], 1.5)
        ))
        lines.append(self._metric_line(
            "Payoff Ratio", self._ratio_color(summary["payoff_ratio"], 1.0)
        ))
        lines.append(self._metric_line(
            "Max Drawdown", self._color(f"${summary['max_drawdown']:.2f}", Colors.RED)
        ))
        lines.append(self._metric_line(
            "Recovery Factor", self._ratio_color(summary["recovery_factor"], 2.0)
        ))
        lines.append("")

        # Trade Statistics
        lines.append(self._section_line("TRADE STATS"))
        lines.append(self._metric_line(
            "Avg Win", self._color(f"${summary['avg_win']:.2f}", Colors.GREEN)
        ))
        lines.append(self._metric_line(
            "Avg Loss", self._color(f"${abs(summary['avg_loss']):.2f}", Colors.RED)
        ))
        lines.append(self._metric_line(
            "Best Trade", self._pnl_color(summary["best_trade"])
        ))
        lines.append(self._metric_line(
            "Worst Trade", self._pnl_color(summary["worst_trade"])
        ))
        lines.append(self._metric_line(
            "Max Consec Wins", self._color(str(summary["consecutive_wins"]), Colors.NEON_GREEN)
        ))
        lines.append(self._metric_line(
            "Max Consec Losses", self._color(str(summary["consecutive_losses"]), Colors.RED)
        ))
        lines.append(self._metric_line(
            "Avg Hold Time", self._color(f"{summary['avg_hold_time_hours']:.1f}h", Colors.NEON_ORANGE)
        ))
        lines.append("")

        # Per-Model Breakdown
        lines.append(self._section_line("MODEL ALPHA"))
        model_stats = summary["per_model"]
        model_header = f"  {'Model':<16} {'Trades':>7} {'Win%':>7} {'PF':>7} {'PnL':>10}"
        lines.append(self._color(model_header, Colors.BOLD))
        lines.append(self._color(f"  {'.' * 52}", Colors.DIM))
        for model_name, stats in model_stats.items():
            if stats["trade_count"] > 0:
                wr_str = f"{stats['win_rate']*100:.1f}%"
                pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "INF"
                pnl_str = f"{stats['total_pnl']:+.2f}"
                line = f"  {model_name:<16} {stats['trade_count']:>7} {wr_str:>7} {pf_str:>7} {pnl_str:>10}"
                lines.append(line)
        lines.append("")

        # Per-Regime Breakdown
        lines.append(self._section_line("REGIME BREAKDOWN"))
        regime_stats = summary["per_regime"]
        regime_header = f"  {'Regime':<16} {'Trades':>7} {'Win%':>7} {'PF':>7} {'PnL':>10}"
        lines.append(self._color(regime_header, Colors.BOLD))
        lines.append(self._color(f"  {'.' * 52}", Colors.DIM))
        for regime_name, stats in regime_stats.items():
            if stats["trade_count"] > 0:
                wr_str = f"{stats['win_rate']*100:.1f}%"
                pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "INF"
                pnl_str = f"{stats['total_pnl']:+.2f}"
                line = f"  {regime_name:<16} {stats['trade_count']:>7} {wr_str:>7} {pf_str:>7} {pnl_str:>10}"
                lines.append(line)
        lines.append("")

        # Bottom border
        lines.append(self._color("+" + "=" * (w - 2) + "+", Colors.NEON_CYAN))
        status_line = "  [HF SCALPER] ATR Cap: $5 | SL: ~$1 | TP: ~$1.50 | Cycle: 10s"
        lines.append(self._color(status_line, Colors.DIM))
        lines.append("")

        return "\n".join(lines)

    def render_html(self, output_path: str) -> str:
        """
        Generate a standalone HTML performance report.
        
        Creates a professional-looking HTML file with:
        - Summary metrics table
        - Per-model performance comparison
        - Per-regime breakdown
        - Equity curve data (JSON for chart rendering)
        
        Args:
            output_path: File path to write the HTML report
            
        Returns:
            The output file path
        """
        summary = self._tracker.get_full_summary()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Serialize equity curve for JavaScript chart
        equity_json = json.dumps(summary["equity_curve"])

        # Build model rows
        model_rows = ""
        for model_name, stats in summary["per_model"].items():
            if stats["trade_count"] > 0:
                pf = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "INF"
                model_rows += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{stats['trade_count']}</td>
                    <td class="{'positive' if stats['win_rate'] > 0.5 else 'negative'}">{stats['win_rate']*100:.1f}%</td>
                    <td>{pf}</td>
                    <td class="{'positive' if stats['total_pnl'] > 0 else 'negative'}">${stats['total_pnl']:.2f}</td>
                    <td>${stats['expectancy']:.2f}</td>
                </tr>"""

        # Build regime rows
        regime_rows = ""
        for regime_name, stats in summary["per_regime"].items():
            if stats["trade_count"] > 0:
                pf = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "INF"
                regime_rows += f"""
                <tr>
                    <td>{regime_name}</td>
                    <td>{stats['trade_count']}</td>
                    <td class="{'positive' if stats['win_rate'] > 0.5 else 'negative'}">{stats['win_rate']*100:.1f}%</td>
                    <td>{pf}</td>
                    <td class="{'positive' if stats['total_pnl'] > 0 else 'negative'}">${stats['total_pnl']:.2f}</td>
                </tr>"""

        # Format metrics for display
        pf_display = f"{summary['profit_factor']:.2f}" if summary['profit_factor'] != float('inf') else "INF"
        sortino_display = f"{summary['sortino_ratio']:.2f}" if summary['sortino_ratio'] != float('inf') else "INF"
        recovery_display = f"{summary['recovery_factor']:.2f}" if summary['recovery_factor'] != float('inf') else "INF"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Python ML Bridge // HF Scalper Dashboard</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Orbitron:wght@400;700;900&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            background: #050a15;
            color: #c8d6e5;
            padding: 24px;
            line-height: 1.7;
            min-height: 100vh;
            background-image:
                radial-gradient(ellipse at 20% 50%, rgba(0, 255, 200, 0.03) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(100, 80, 255, 0.03) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(255, 50, 100, 0.02) 0%, transparent 50%);
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            text-align: center;
            padding: 40px 0 30px;
            position: relative;
        }}
        .header::before {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 10%;
            right: 10%;
            height: 1px;
            background: linear-gradient(90deg, transparent, #00ffc8, #6450ff, #00ffc8, transparent);
        }}
        .header h1 {{
            font-family: 'Orbitron', sans-serif;
            font-size: 2rem;
            font-weight: 900;
            background: linear-gradient(135deg, #00ffc8 0%, #6450ff 50%, #ff3264 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: 3px;
            text-transform: uppercase;
        }}
        .header .subtitle {{
            font-family: 'Orbitron', sans-serif;
            color: #4a5568;
            font-size: 0.7rem;
            margin-top: 8px;
            letter-spacing: 4px;
            text-transform: uppercase;
        }}
        .header .timestamp {{
            color: #00ffc8;
            font-size: 0.75rem;
            margin-top: 12px;
            opacity: 0.7;
        }}
        .header .mode-badge {{
            display: inline-block;
            margin-top: 12px;
            padding: 4px 14px;
            border: 1px solid #00ffc8;
            border-radius: 20px;
            font-size: 0.65rem;
            color: #00ffc8;
            letter-spacing: 2px;
            text-transform: uppercase;
            animation: pulse-glow 2s ease-in-out infinite;
        }}
        @keyframes pulse-glow {{
            0%, 100% {{ box-shadow: 0 0 5px rgba(0,255,200,0.3); }}
            50% {{ box-shadow: 0 0 15px rgba(0,255,200,0.6); }}
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(145deg, #0d1526 0%, #0a1020 100%);
            border: 1px solid rgba(100, 80, 255, 0.2);
            border-radius: 12px;
            padding: 18px 14px;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }}
        .metric-card:hover {{
            border-color: rgba(0, 255, 200, 0.5);
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 255, 200, 0.1);
        }}
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #6450ff, transparent);
        }}
        .metric-card .label {{
            font-size: 0.65rem;
            color: #5a6a7a;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 500;
        }}
        .metric-card .value {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 8px;
        }}
        .positive {{ color: #00ffc8; }}
        .negative {{ color: #ff3264; }}
        .neutral {{ color: #ffaa00; }}
        .section {{
            background: linear-gradient(145deg, #0d1526 0%, #080e1c 100%);
            border: 1px solid rgba(100, 80, 255, 0.15);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            position: relative;
        }}
        .section::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 20px;
            right: 20px;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(100, 80, 255, 0.4), transparent);
        }}
        .section h2 {{
            font-family: 'Orbitron', sans-serif;
            color: #6450ff;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 18px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(100, 80, 255, 0.1);
            letter-spacing: 2px;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            text-align: left;
            padding: 12px 10px;
            color: #5a6a7a;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 1px solid rgba(100, 80, 255, 0.15);
        }}
        td {{
            padding: 12px 10px;
            border-bottom: 1px solid rgba(100, 80, 255, 0.05);
            font-size: 0.85rem;
        }}
        tr:hover {{ background: rgba(0, 255, 200, 0.03); }}
        .equity-section {{ margin-top: 20px; }}
        .footer {{
            text-align: center;
            color: #3a4a5a;
            font-size: 0.65rem;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid rgba(100, 80, 255, 0.1);
            letter-spacing: 2px;
            text-transform: uppercase;
        }}
        pre {{
            background: #050a12;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.75rem;
            color: #00ffc8;
            margin-top: 12px;
            border: 1px solid rgba(0, 255, 200, 0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Python ML Bridge</h1>
            <div class="subtitle">High-Frequency Scalper // Performance Analytics</div>
            <div class="timestamp">Generated: {timestamp}</div>
            <div class="mode-badge">Live M1 XAUUSD</div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Net Profit</div>
                <div class="value {'positive' if summary['net_profit'] >= 0 else 'negative'}">${summary['net_profit']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Trades</div>
                <div class="value" style="color: #6450ff;">{summary['total_trades']}</div>
            </div>
            <div class="metric-card">
                <div class="label">Win Rate</div>
                <div class="value {'positive' if summary['win_rate'] > 0.5 else 'negative'}">{summary['win_rate']*100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="label">Sharpe Ratio</div>
                <div class="value {'positive' if summary['sharpe_ratio'] > 1 else 'negative'}">{summary['sharpe_ratio']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Sortino Ratio</div>
                <div class="value {'positive' if summary['sortino_ratio'] > 1.5 else 'neutral'}">{sortino_display}</div>
            </div>
            <div class="metric-card">
                <div class="label">Profit Factor</div>
                <div class="value {'positive' if summary['profit_factor'] > 1.5 else 'negative'}">{pf_display}</div>
            </div>
            <div class="metric-card">
                <div class="label">Max Drawdown</div>
                <div class="value negative">${summary['max_drawdown']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Recovery Factor</div>
                <div class="value {'positive' if summary['recovery_factor'] > 2 else 'neutral'}">{recovery_display}</div>
            </div>
            <div class="metric-card">
                <div class="label">Expectancy</div>
                <div class="value {'positive' if summary['expectancy'] > 0 else 'negative'}">${summary['expectancy']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg Win</div>
                <div class="value positive">${summary['avg_win']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg Loss</div>
                <div class="value negative">${abs(summary['avg_loss']):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Payoff Ratio</div>
                <div class="value" style="color: #ffaa00;">{summary['payoff_ratio']:.2f}</div>
            </div>
        </div>

        <div class="section">
            <h2>Model Performance Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>Profit Factor</th>
                        <th>Total PnL</th>
                        <th>Expectancy</th>
                    </tr>
                </thead>
                <tbody>
                    {model_rows}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Regime Performance Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Regime</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>Profit Factor</th>
                        <th>Total PnL</th>
                    </tr>
                </thead>
                <tbody>
                    {regime_rows}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Trade Statistics</h2>
            <table>
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>
                    <tr><td>Best Trade</td><td class="positive">${summary['best_trade']:.2f}</td></tr>
                    <tr><td>Worst Trade</td><td class="negative">${summary['worst_trade']:.2f}</td></tr>
                    <tr><td>Max Consecutive Wins</td><td>{summary['consecutive_wins']}</td></tr>
                    <tr><td>Max Consecutive Losses</td><td>{summary['consecutive_losses']}</td></tr>
                    <tr><td>Avg Hold Time</td><td>{summary['avg_hold_time_hours']:.1f} hours</td></tr>
                </tbody>
            </table>
        </div>

        <div class="section equity-section">
            <h2>Equity Curve Data</h2>
            <p style="color: #5a6a7a; font-size: 0.75rem; letter-spacing: 0.5px;">
                Equity curve data embedded as JSON for charting libraries (D3.js, Chart.js, Lightweight Charts)
            </p>
            <pre>{json.dumps(summary['equity_curve'], indent=2)}</pre>
        </div>

        <div class="footer">
            Python ML Bridge // Prop Trading Analytics Engine // HF Scalper v2.0
        </div>
    </div>
</body>
</html>"""

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"[Dashboard] HTML report generated: {output_path}")
        return output_path

    def render_log(self) -> Dict:
        """
        Render structured performance data for logging systems.
        
        Returns a dict suitable for JSON serialization and ingestion
        by monitoring pipelines (ELK, Grafana, Datadog, etc.).
        """
        summary = self._tracker.get_full_summary()

        # Remove equity curve from log output (too verbose)
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "performance_snapshot",
            "metrics": {
                "total_trades": summary["total_trades"],
                "net_profit": round(summary["net_profit"], 2),
                "win_rate": round(summary["win_rate"], 4),
                "profit_factor": round(summary["profit_factor"], 4) if summary["profit_factor"] != float('inf') else "inf",
                "sharpe_ratio": round(summary["sharpe_ratio"], 4),
                "sortino_ratio": round(summary["sortino_ratio"], 4) if summary["sortino_ratio"] != float('inf') else "inf",
                "max_drawdown": round(summary["max_drawdown"], 2),
                "recovery_factor": round(summary["recovery_factor"], 4) if summary["recovery_factor"] != float('inf') else "inf",
                "expectancy": round(summary["expectancy"], 4),
                "avg_win": round(summary["avg_win"], 2),
                "avg_loss": round(summary["avg_loss"], 2),
                "payoff_ratio": round(summary["payoff_ratio"], 4) if summary["payoff_ratio"] != float('inf') else "inf",
                "best_trade": round(summary["best_trade"], 2),
                "worst_trade": round(summary["worst_trade"], 2),
                "consecutive_wins": summary["consecutive_wins"],
                "consecutive_losses": summary["consecutive_losses"],
                "avg_hold_time_hours": round(summary["avg_hold_time_hours"], 2),
            },
            "per_model": {},
            "per_regime": {},
        }

        # Serialize per-model (only models with trades)
        for model_name, stats in summary["per_model"].items():
            if stats["trade_count"] > 0:
                log_data["per_model"][model_name] = {
                    "trades": stats["trade_count"],
                    "win_rate": round(stats["win_rate"], 4),
                    "profit_factor": round(stats["profit_factor"], 4) if stats["profit_factor"] != float('inf') else "inf",
                    "total_pnl": round(stats["total_pnl"], 2),
                }

        # Serialize per-regime (only regimes with trades)
        for regime_name, stats in summary["per_regime"].items():
            if stats["trade_count"] > 0:
                log_data["per_regime"][regime_name] = {
                    "trades": stats["trade_count"],
                    "win_rate": round(stats["win_rate"], 4),
                    "profit_factor": round(stats["profit_factor"], 4) if stats["profit_factor"] != float('inf') else "inf",
                    "total_pnl": round(stats["total_pnl"], 2),
                }

        logger.info(f"[Dashboard] Performance snapshot: trades={summary['total_trades']} "
                    f"pnl={summary['net_profit']:.2f} wr={summary['win_rate']*100:.1f}% "
                    f"sharpe={summary['sharpe_ratio']:.2f} pf={summary['profit_factor']:.2f}")

        return log_data
