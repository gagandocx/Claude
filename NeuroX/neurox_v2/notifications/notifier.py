"""
=============================================================
  Notification System - Telegram & Discord Alerts

  Opt-in notification system for trade events. Disabled by default.
  Events:
    - Trade opened (with direction, lot, entry price)
    - Trade closed (with P&L, duration)
    - Daily summary (total P&L, win rate, trade count)
    - Drawdown alert (when drawdown exceeds threshold)
    - Error alerts (critical system errors)

  Configuration via NotificationConfig in settings.
  Sends notifications via Telegram bot API and/or Discord webhook.
=============================================================
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class NotificationConfig:
    """Notification system configuration. Disabled by default (opt-in)."""
    # Master switch
    enabled: bool = False

    # Telegram settings
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Discord settings
    discord_enabled: bool = False
    discord_webhook_url: str = ""

    # Event filters (which events to notify)
    notify_trade_opened: bool = True
    notify_trade_closed: bool = True
    notify_daily_summary: bool = True
    notify_drawdown_alert: bool = True
    notify_error_alert: bool = True

    # Thresholds
    drawdown_alert_threshold: float = 0.05   # 5% drawdown triggers alert
    error_cooldown_seconds: int = 300        # Min 5 min between error alerts
    daily_summary_hour: int = 21             # UTC hour for daily summary

    # Rate limiting
    max_messages_per_minute: int = 10
    max_messages_per_hour: int = 60


class NotificationManager:
    """
    Manages sending notifications to Telegram and Discord.

    Thread-safe: notifications are queued and sent asynchronously
    to avoid blocking the main trading loop.
    """

    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig()
        self._message_queue: deque = deque(maxlen=100)
        self._send_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_error_time: float = 0.0
        self._messages_this_minute: int = 0
        self._messages_this_hour: int = 0
        self._minute_reset_time: float = time.time()
        self._hour_reset_time: float = time.time()
        self._daily_trades: List[Dict] = []
        self._daily_pnl: float = 0.0

        if self.config.enabled:
            self._start_sender()
            logger.info("[Notifier] Notification system enabled")
            if self.config.telegram_enabled:
                logger.info("[Notifier]   Telegram: ON")
            if self.config.discord_enabled:
                logger.info("[Notifier]   Discord: ON")
        else:
            logger.info("[Notifier] Notification system disabled (opt-in)")

    def _start_sender(self):
        """Start the background message sender thread."""
        self._running = True
        self._send_thread = threading.Thread(
            target=self._sender_loop, daemon=True, name="NotificationSender"
        )
        self._send_thread.start()

    def _sender_loop(self):
        """Background loop that sends queued messages."""
        while self._running:
            try:
                if self._message_queue:
                    msg = self._message_queue.popleft()
                    if self._check_rate_limit():
                        self._send_message(msg)
                time.sleep(0.5)
            except Exception as e:
                logger.debug(f"[Notifier] Sender error: {e}")
                time.sleep(2.0)

    def _check_rate_limit(self) -> bool:
        """Check if we can send another message (rate limiting)."""
        now = time.time()

        # Reset minute counter
        if now - self._minute_reset_time >= 60:
            self._messages_this_minute = 0
            self._minute_reset_time = now

        # Reset hour counter
        if now - self._hour_reset_time >= 3600:
            self._messages_this_hour = 0
            self._hour_reset_time = now

        if self._messages_this_minute >= self.config.max_messages_per_minute:
            return False
        if self._messages_this_hour >= self.config.max_messages_per_hour:
            return False

        self._messages_this_minute += 1
        self._messages_this_hour += 1
        return True

    def _send_message(self, message: str):
        """Send a message to all enabled channels."""
        if self.config.telegram_enabled and self.config.telegram_bot_token:
            self._send_telegram(message)
        if self.config.discord_enabled and self.config.discord_webhook_url:
            self._send_discord(message)

    def _send_telegram(self, message: str):
        """Send a message via Telegram Bot API."""
        try:
            import requests
            url = (
                f"https://api.telegram.org/bot{self.config.telegram_bot_token}"
                f"/sendMessage"
            )
            payload = {
                "chat_id": self.config.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code != 200:
                logger.debug(
                    f"[Notifier] Telegram send failed: {resp.status_code}"
                )
        except Exception as e:
            logger.debug(f"[Notifier] Telegram error: {e}")

    def _send_discord(self, message: str):
        """Send a message via Discord webhook."""
        try:
            import requests
            payload = {"content": message}
            resp = requests.post(
                self.config.discord_webhook_url,
                json=payload,
                timeout=10
            )
            if resp.status_code not in (200, 204):
                logger.debug(
                    f"[Notifier] Discord send failed: {resp.status_code}"
                )
        except Exception as e:
            logger.debug(f"[Notifier] Discord error: {e}")

    # ── Public notification methods ──────────────────────────────────────

    def notify_trade_opened(self, direction: str, lot_size: float,
                            entry_price: float, sl: float = 0.0,
                            tp: float = 0.0, confidence: float = 0.0):
        """Notify that a trade was opened."""
        if not self.config.enabled or not self.config.notify_trade_opened:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = (
            f"*TRADE OPENED*\n"
            f"Direction: {direction}\n"
            f"Lot: {lot_size:.2f}\n"
            f"Entry: ${entry_price:.2f}\n"
            f"SL: ${sl:.2f} | TP: ${tp:.2f}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Time: {timestamp}"
        )
        self._message_queue.append(msg)

    def notify_trade_closed(self, direction: str, pnl: float,
                            entry_price: float, exit_price: float,
                            lot_size: float = 0.0, duration_s: float = 0.0):
        """Notify that a trade was closed."""
        if not self.config.enabled or not self.config.notify_trade_closed:
            return

        result = "WIN" if pnl > 0 else "LOSS"
        emoji = "+" if pnl > 0 else ""
        duration_min = duration_s / 60.0 if duration_s > 0 else 0.0
        timestamp = datetime.now().strftime("%H:%M:%S")

        msg = (
            f"*TRADE CLOSED - {result}*\n"
            f"Direction: {direction} {lot_size:.2f} lot\n"
            f"Entry: ${entry_price:.2f} -> Exit: ${exit_price:.2f}\n"
            f"P&L: {emoji}${pnl:.2f}\n"
            f"Duration: {duration_min:.1f} min\n"
            f"Time: {timestamp}"
        )
        self._message_queue.append(msg)

        # Track for daily summary
        self._daily_trades.append({
            "direction": direction, "pnl": pnl,
            "entry": entry_price, "exit": exit_price
        })
        self._daily_pnl += pnl

    def notify_drawdown_alert(self, drawdown_pct: float,
                              daily_pnl: float, account_balance: float):
        """Notify about excessive drawdown."""
        if not self.config.enabled or not self.config.notify_drawdown_alert:
            return
        if drawdown_pct < self.config.drawdown_alert_threshold:
            return

        msg = (
            f"*DRAWDOWN ALERT*\n"
            f"Drawdown: {drawdown_pct:.1%}\n"
            f"Daily P&L: ${daily_pnl:+.2f}\n"
            f"Balance: ${account_balance:.2f}"
        )
        self._message_queue.append(msg)

    def notify_error(self, error_msg: str):
        """Notify about a critical error (rate-limited)."""
        if not self.config.enabled or not self.config.notify_error_alert:
            return

        now = time.time()
        if now - self._last_error_time < self.config.error_cooldown_seconds:
            return
        self._last_error_time = now

        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = (
            f"*ERROR ALERT*\n"
            f"{error_msg[:200]}\n"
            f"Time: {timestamp}"
        )
        self._message_queue.append(msg)

    def send_daily_summary(self):
        """Send the daily trading summary."""
        if not self.config.enabled or not self.config.notify_daily_summary:
            return
        if not self._daily_trades:
            return

        wins = sum(1 for t in self._daily_trades if t["pnl"] > 0)
        losses = len(self._daily_trades) - wins
        win_rate = wins / len(self._daily_trades) if self._daily_trades else 0.0
        date_str = datetime.now().strftime("%Y-%m-%d")

        msg = (
            f"*DAILY SUMMARY - {date_str}*\n"
            f"Trades: {len(self._daily_trades)}\n"
            f"Wins: {wins} | Losses: {losses}\n"
            f"Win Rate: {win_rate:.1%}\n"
            f"Total P&L: ${self._daily_pnl:+.2f}"
        )
        self._message_queue.append(msg)

        # Reset daily tracking
        self._daily_trades.clear()
        self._daily_pnl = 0.0

    def stop(self):
        """Stop the notification sender thread."""
        self._running = False
        if self._send_thread and self._send_thread.is_alive():
            self._send_thread.join(timeout=2.0)
