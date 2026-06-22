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
    """Terminal color codes for professional display."""
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


class DashboardRenderer:
    """
    Professional trading desk dashboard renderer.
    
    Outputs performance data in formats optimized for different contexts:
    - Console: real-time monitoring during live trading sessions
    - HTML: end-of-day/week reports for strategy review
    - Log: structured data for automated monitoring pipelines
    """

    # Dashboard width in characters
    CONSOLE_WIDTH = 72

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
        """Create a formatted header line."""
        w = self.CONSOLE_WIDTH
        line = "=" * w
        padding = (w - len(title) - 4) // 2
        header = f"{'=' * padding}[ {title} ]{'=' * (w - padding - len(title) - 4)}"
        return self._color(header, Colors.CYAN)

    def _section_line(self, title: str) -> str:
        """Create a section separator."""
        w = self.CONSOLE_WIDTH
        padding = (w - len(title) - 4) // 2
        return self._color(
            f"{'-' * padding}[ {title} ]{'-' * (w - padding - len(title) - 4)}",
            Colors.DIM
        )

    def _metric_line(self, label: str, value: str, width: int = 30) -> str:
        """Format a single metric as label : value."""
        dots = "." * (width - len(label))
        return f"  {label} {self._color(dots, Colors.DIM)} {value}"

    def render_console(self) -> str:
        """
        Render the full performance dashboard as ASCII art.
        
        Returns a multi-line string formatted for terminal display,
        similar to a Bloomberg terminal or prop desk monitor.
        """
        summary = self._tracker.get_full_summary()
        lines = []
        w = self.CONSOLE_WIDTH

        # Top border
        lines.append("")
        lines.append(self._header_line("PYTHON ML BRIDGE - LIVE PERFORMANCE"))
        lines.append(
            self._color(f"  Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", Colors.DIM)
        )
        lines.append("")

        # ── Core P&L ──
        lines.append(self._section_line("PROFIT & LOSS"))
        lines.append(self._metric_line(
            "Net Profit", self._pnl_color(summary["net_profit"])
        ))
        lines.append(self._metric_line(
            "Total Trades", str(summary["total_trades"])
        ))
        lines.append(self._metric_line(
            "Winners / Losers",
            f"{summary['total_wins']} / {summary['total_losses']}"
        ))
        lines.append(self._metric_line(
            "Win Rate", self._pct_color(summary["win_rate"] * 100)
        ))
        lines.append(self._metric_line(
            "Expectancy/Trade", self._pnl_color(summary["expectancy"])
        ))
        lines.append("")

        # ── Risk Metrics ──
        lines.append(self._section_line("RISK METRICS"))
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

        # ── Trade Statistics ──
        lines.append(self._section_line("TRADE STATISTICS"))
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
            "Max Consec Wins", self._color(str(summary["consecutive_wins"]), Colors.GREEN)
        ))
        lines.append(self._metric_line(
            "Max Consec Losses", self._color(str(summary["consecutive_losses"]), Colors.RED)
        ))
        lines.append(self._metric_line(
            "Avg Hold Time", f"{summary['avg_hold_time_hours']:.1f}h"
        ))
        lines.append("")

        # ── Per-Model Breakdown ──
        lines.append(self._section_line("MODEL PERFORMANCE"))
        model_stats = summary["per_model"]
        model_header = f"  {'Model':<16} {'Trades':>7} {'Win%':>7} {'PF':>7} {'PnL':>10}"
        lines.append(self._color(model_header, Colors.BOLD))
        lines.append(f"  {'-' * 50}")
        for model_name, stats in model_stats.items():
            if stats["trade_count"] > 0:
                wr_str = f"{stats['win_rate']*100:.1f}%"
                pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "INF"
                pnl_str = f"{stats['total_pnl']:+.2f}"
                line = f"  {model_name:<16} {stats['trade_count']:>7} {wr_str:>7} {pf_str:>7} {pnl_str:>10}"
                lines.append(line)
        lines.append("")

        # ── Per-Regime Breakdown ──
        lines.append(self._section_line("REGIME PERFORMANCE"))
        regime_stats = summary["per_regime"]
        regime_header = f"  {'Regime':<16} {'Trades':>7} {'Win%':>7} {'PF':>7} {'PnL':>10}"
        lines.append(self._color(regime_header, Colors.BOLD))
        lines.append(f"  {'-' * 50}")
        for regime_name, stats in regime_stats.items():
            if stats["trade_count"] > 0:
                wr_str = f"{stats['win_rate']*100:.1f}%"
                pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "INF"
                pnl_str = f"{stats['total_pnl']:+.2f}"
                line = f"  {regime_name:<16} {stats['trade_count']:>7} {wr_str:>7} {pf_str:>7} {pnl_str:>10}"
                lines.append(line)
        lines.append("")

        # Bottom border
        lines.append(self._color("=" * w, Colors.CYAN))
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
    <title>Python ML Bridge - Performance Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            background: #0a0e17;
            color: #e0e6ed;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid #1e3a5f;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 1.8rem;
            color: #4ecdc4;
            letter-spacing: 2px;
        }}
        .header .timestamp {{
            color: #6b7b8d;
            font-size: 0.85rem;
            margin-top: 8px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #141b2d;
            border: 1px solid #1e3a5f;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .metric-card .label {{
            font-size: 0.75rem;
            color: #6b7b8d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-card .value {{
            font-size: 1.5rem;
            font-weight: bold;
            margin-top: 5px;
        }}
        .positive {{ color: #4ecdc4; }}
        .negative {{ color: #ff6b6b; }}
        .neutral {{ color: #ffa726; }}
        .section {{
            background: #141b2d;
            border: 1px solid #1e3a5f;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .section h2 {{
            color: #4ecdc4;
            font-size: 1.1rem;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #1e3a5f;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            text-align: left;
            padding: 10px;
            color: #6b7b8d;
            font-size: 0.8rem;
            text-transform: uppercase;
            border-bottom: 1px solid #1e3a5f;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #0d1421;
            font-size: 0.9rem;
        }}
        tr:hover {{ background: #1a2235; }}
        .equity-section {{
            margin-top: 20px;
        }}
        .footer {{
            text-align: center;
            color: #6b7b8d;
            font-size: 0.75rem;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #1e3a5f;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>PYTHON ML BRIDGE - PERFORMANCE REPORT</h1>
            <div class="timestamp">Generated: {timestamp}</div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Net Profit</div>
                <div class="value {'positive' if summary['net_profit'] >= 0 else 'negative'}">${summary['net_profit']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Trades</div>
                <div class="value">{summary['total_trades']}</div>
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
                <div class="value">{summary['payoff_ratio']:.2f}</div>
            </div>
        </div>

        <div class="section">
            <h2>MODEL PERFORMANCE BREAKDOWN</h2>
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
            <h2>REGIME PERFORMANCE BREAKDOWN</h2>
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
            <h2>TRADE STATISTICS</h2>
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
            <h2>EQUITY CURVE DATA</h2>
            <p style="color: #6b7b8d; font-size: 0.85rem;">
                Equity curve data embedded as JSON for charting libraries (D3.js, Chart.js, etc.)
            </p>
            <pre style="background: #0d1421; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 0.8rem; color: #4ecdc4; margin-top: 10px;">
{json.dumps(summary['equity_curve'], indent=2)}
            </pre>
        </div>

        <div class="footer">
            Python ML Bridge Performance Dashboard | Prop Trading Analytics Engine
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
