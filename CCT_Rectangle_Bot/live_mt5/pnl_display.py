"""
Giant Floating PnL Display - Real-Time ASCII Art Terminal Display.

Renders the current floating PnL as a MASSIVE ASCII art number that
fills most of the terminal width. Designed to be readable from across
the room - like a giant price board.

Uses pyfiglet for large ASCII text rendering and ANSI escape codes
for color (green = profit, red = loss, white = zero).

The display OVERWRITES in place using ANSI cursor repositioning.
On first render: clears the screen. On subsequent renders: moves cursor
to home position and overwrites all content. No scrolling.
"""

import os
import sys
import time
from typing import Optional, Any

try:
    import pyfiglet
except ImportError:
    pyfiglet = None

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init()
except ImportError:
    # Fallback: define color codes manually
    class Fore:
        GREEN = "\033[32m"
        RED = "\033[31m"
        WHITE = "\033[37m"
        YELLOW = "\033[33m"
        CYAN = "\033[36m"
        RESET = "\033[0m"

    class Style:
        BRIGHT = "\033[1m"
        RESET_ALL = "\033[0m"

    def colorama_init():
        pass


# ANSI escape sequences for screen control
CURSOR_HOME = "\033[H"          # Move cursor to top-left (home position)
CLEAR_SCREEN = "\033[2J"        # Clear entire screen (used once at startup)
CLEAR_LINE = "\033[K"           # Clear from cursor to end of line
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
RESET_COLOR = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Color codes
GREEN = "\033[32m"
RED = "\033[31m"
WHITE = "\033[37m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_RED = "\033[91m"
BRIGHT_WHITE = "\033[97m"


class GiantPnLDisplay:
    """
    Renders a giant floating PnL number in the terminal using ASCII art.

    The PnL number fills most of the terminal width and is colored:
    - GREEN when positive (profit)
    - RED when negative (loss)
    - WHITE when zero

    Below the giant number, a compact status line shows account info
    and signal state.

    DISPLAY METHOD: Overwrites in place using ANSI escape codes.
    - First render: clear screen + cursor home
    - Subsequent renders: cursor home only (overwrite existing content)
    - Each line ends with clear-to-EOL to remove leftover characters

    This creates a static dashboard that refreshes in place with no scrolling.

    Usage:
        display = GiantPnLDisplay()
        display.render(
            pnl=1234.56,
            balance=50000.0,
            equity=51234.56,
            spread=15,
            signal_state="ARMED",
        )
    """

    # Preferred pyfiglet fonts for readability at large size
    PREFERRED_FONTS = ["banner3", "banner", "big", "block", "doom", "standard"]

    def __init__(self):
        """Initialize the GiantPnLDisplay."""
        self._font_name: str = "banner3"
        self._figlet: Optional[Any] = None
        self._last_render_time: float = 0.0
        self._terminal_width: int = 80
        self._terminal_height: int = 24
        self._first_render: bool = True

        # Initialize pyfiglet with the best available font
        self._init_figlet()

        # Hide cursor and clear screen on first init
        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.flush()

    def _init_figlet(self):
        """Initialize pyfiglet with the best available font."""
        if pyfiglet is None:
            return

        # Try preferred fonts in order
        for font in self.PREFERRED_FONTS:
            try:
                self._figlet = pyfiglet.Figlet(font=font)
                # Test render to make sure it works
                self._figlet.renderText("$0.00")
                self._font_name = font
                break
            except Exception:
                continue

        # Fallback to default font
        if self._figlet is None:
            try:
                self._figlet = pyfiglet.Figlet(font="standard")
                self._font_name = "standard"
            except Exception:
                self._figlet = None

    def _get_terminal_size(self):
        """Update terminal dimensions."""
        try:
            size = os.get_terminal_size()
            self._terminal_width = size.columns
            self._terminal_height = size.lines
        except (OSError, ValueError):
            self._terminal_width = 80
            self._terminal_height = 24

    def render(
        self,
        pnl: float = 0.0,
        balance: float = 0.0,
        equity: float = 0.0,
        spread: int = 0,
        signal_state: str = "IDLE",
        positions_count: int = 0,
        daily_pnl: float = 0.0,
        daily_trades: int = 0,
        profile_name: str = "SMART-HYBRID",
    ):
        """
        Render the giant PnL display to the terminal, overwriting in place.

        On first render: clears screen and moves cursor home.
        On subsequent renders: moves cursor to home position only (no clear).
        Each line ends with CLEAR_LINE to remove leftover characters.

        This creates a static dashboard that updates in place - no scrolling.

        Args:
            pnl: Current floating PnL (unrealized profit/loss).
            balance: Account balance.
            equity: Account equity.
            spread: Current spread in points.
            signal_state: Current signal state (IDLE/SCANNING/ARMED).
            positions_count: Number of open positions.
            daily_pnl: Today's realized PnL.
            daily_trades: Number of trades today.
            profile_name: Active trading profile name (e.g. AGGRESSIVE, SAFE).
        """
        self._get_terminal_size()

        # Format PnL string
        pnl_str = self._format_pnl(pnl)

        # Get color based on PnL
        bright_color = self._get_bright_color(pnl)

        # Build all output lines
        lines = []

        # Add some top padding
        top_padding = max(1, (self._terminal_height - 15) // 4)
        for _ in range(top_padding):
            lines.append("")

        # Render giant PnL with pyfiglet
        giant_text = self._render_giant_text(pnl_str)
        if giant_text:
            # Apply color to the giant text
            for line in giant_text.split("\n"):
                lines.append(f"{BOLD}{bright_color}{line}{RESET_COLOR}")
        else:
            # Fallback: large text without pyfiglet
            for line in self._render_fallback_large(pnl_str).split("\n"):
                lines.append(f"{BOLD}{bright_color}{line}{RESET_COLOR}")

        # Spacer
        lines.append("")

        # Status line separator
        lines.append(f"{DIM}{'=' * self._terminal_width}{RESET_COLOR}")

        # Status info line 1: Balance | Equity | Spread | Signal
        lines.append(self._build_status_line(
            balance=balance,
            equity=equity,
            spread=spread,
            signal_state=signal_state,
        ))

        # Status info line 2: Positions | Daily PnL | Daily Trades | Mode
        lines.append(self._build_detail_line(
            positions_count=positions_count,
            daily_pnl=daily_pnl,
            daily_trades=daily_trades,
            profile_name=profile_name,
        ))

        # Bottom separator
        lines.append(f"{DIM}{'=' * self._terminal_width}{RESET_COLOR}")

        # Footer
        lines.append(f"{DIM}  CCT Rectangle Bot | {profile_name} Mode | Ctrl+C to stop{RESET_COLOR}")

        # Build final output with in-place overwrite
        output_parts = []

        # First render: clear screen then home. Subsequent: just home.
        if self._first_render:
            output_parts.append(CLEAR_SCREEN)
            output_parts.append(CURSOR_HOME)
            self._first_render = False
        else:
            output_parts.append(CURSOR_HOME)

        # Write each line followed by clear-to-end-of-line to remove artifacts
        for line in lines:
            output_parts.append(line)
            output_parts.append(CLEAR_LINE)
            output_parts.append("\n")

        # Clear any remaining lines below (in case terminal was resized smaller)
        # Fill remaining lines with empty + clear
        remaining_lines = self._terminal_height - len(lines) - 1
        for _ in range(max(0, remaining_lines)):
            output_parts.append(CLEAR_LINE)
            output_parts.append("\n")

        # Single write + flush for minimal flicker
        sys.stdout.write("".join(output_parts))
        sys.stdout.flush()

        self._last_render_time = time.time()

    def _format_pnl(self, pnl: float) -> str:
        """
        Format PnL as a display string with sign and dollar amount.

        Args:
            pnl: The PnL value.

        Returns:
            Formatted string like '+$1,234.56' or '-$567.89'.
        """
        if pnl >= 0:
            return f"+${pnl:,.2f}"
        else:
            return f"-${abs(pnl):,.2f}"

    def _get_bright_color(self, pnl: float) -> str:
        """Get bright ANSI color code for PnL value."""
        if pnl > 0:
            return BRIGHT_GREEN
        elif pnl < 0:
            return BRIGHT_RED
        else:
            return BRIGHT_WHITE

    def _render_giant_text(self, text: str) -> Optional[str]:
        """
        Render text as giant ASCII art using pyfiglet.

        Automatically adjusts font to fit terminal width.

        Args:
            text: The text to render large.

        Returns:
            Multi-line ASCII art string, or None if pyfiglet unavailable.
        """
        if self._figlet is None:
            return None

        try:
            rendered = self._figlet.renderText(text)

            # Check if it fits the terminal width
            lines = rendered.split("\n")
            max_line_width = max(len(line) for line in lines if line.strip())

            if max_line_width > self._terminal_width:
                # Try a smaller font
                for font in ["big", "doom", "standard", "small"]:
                    try:
                        smaller_figlet = pyfiglet.Figlet(font=font)
                        rendered = smaller_figlet.renderText(text)
                        lines = rendered.split("\n")
                        max_line_width = max(
                            len(line) for line in lines if line.strip()
                        )
                        if max_line_width <= self._terminal_width:
                            break
                    except Exception:
                        continue

            # Center the text
            centered_lines = []
            for line in rendered.split("\n"):
                if line.strip():
                    padding = max(0, (self._terminal_width - len(line)) // 2)
                    centered_lines.append(" " * padding + line)
                else:
                    centered_lines.append("")

            return "\n".join(centered_lines)

        except Exception:
            return None

    def _render_fallback_large(self, text: str) -> str:
        """
        Fallback large text rendering without pyfiglet.

        Uses simple text repetition and padding for visibility.

        Args:
            text: The text to render.

        Returns:
            Multi-line large text string.
        """
        # Center the text with large padding
        padding = max(0, (self._terminal_width - len(text)) // 2)
        padded = " " * padding + text

        # Repeat the line a few times for visual size
        lines = []
        lines.append("")
        lines.append("")
        lines.append(padded)
        lines.append("")
        lines.append(" " * padding + "=" * len(text))
        lines.append("")
        lines.append("")

        return "\n".join(lines)

    def _build_status_line(
        self,
        balance: float,
        equity: float,
        spread: int,
        signal_state: str,
    ) -> str:
        """
        Build the compact status line below the giant PnL.

        Shows: Balance | Equity | Spread | Signal State
        """
        # Color the signal state
        state_color = WHITE
        if signal_state == "ARMED":
            state_color = BRIGHT_GREEN
        elif signal_state == "SCANNING":
            state_color = YELLOW
        elif signal_state == "IDLE":
            state_color = DIM + WHITE

        parts = [
            f"  {CYAN}Balance:{RESET_COLOR} ${balance:,.2f}",
            f"  {CYAN}Equity:{RESET_COLOR} ${equity:,.2f}",
            f"  {CYAN}Spread:{RESET_COLOR} {spread} pts",
            f"  {CYAN}Signal:{RESET_COLOR} {state_color}{BOLD}{signal_state}{RESET_COLOR}",
        ]

        return "  |".join(parts)

    def _build_detail_line(
        self,
        positions_count: int,
        daily_pnl: float,
        daily_trades: int,
        profile_name: str = "SMART-HYBRID",
    ) -> str:
        """
        Build the secondary detail line.

        Shows: Positions | Daily PnL | Daily Trades | Mode
        """
        daily_color = BRIGHT_GREEN if daily_pnl >= 0 else BRIGHT_RED

        parts = [
            f"  {CYAN}Positions:{RESET_COLOR} {positions_count}",
            f"  {CYAN}Daily PnL:{RESET_COLOR} {daily_color}{daily_pnl:+,.2f}{RESET_COLOR}",
            f"  {CYAN}Trades:{RESET_COLOR} {daily_trades}",
            f"  {CYAN}Mode:{RESET_COLOR} {BRIGHT_GREEN}{profile_name}{RESET_COLOR}",
        ]

        return "  |".join(parts)

    def cleanup(self):
        """Restore terminal state on shutdown."""
        # Show cursor
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.write(RESET_COLOR)
        sys.stdout.flush()

    def __del__(self):
        """Destructor to restore terminal state."""
        try:
            self.cleanup()
        except Exception:
            pass
