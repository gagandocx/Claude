"""
MT5 Connection and Order Execution Module.

Handles:
- MT5 terminal initialization and login
- Market order execution (buy/sell) with SL and TP
- Position management (get, modify, close)
- Closed-position reconciliation (detecting fills via history)
- Reconnection with exponential backoff
- Proper error handling with MT5 error codes
"""

import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # Will be checked at runtime on Windows

import mt5_config


@dataclass
class OrderResult:
    """Result of an order execution attempt."""
    success: bool
    ticket: Optional[int] = None
    error_code: Optional[int] = None
    error_message: str = ""
    volume: float = 0.0
    price: float = 0.0


class MT5Connection:
    """
    Manages connection to MetaTrader 5 terminal.

    Provides methods for:
    - Connecting and disconnecting from MT5
    - Sending market orders with SL/TP
    - Modifying open positions
    - Closing positions
    - Automatic reconnection with exponential backoff
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize MT5Connection.

        Args:
            logger: Optional logger instance. Creates one if not provided.
        """
        self.logger = logger or logging.getLogger("mt5_connector")
        self._connected = False
        self._retry_count = 0

    def connect(self) -> bool:
        """
        Initialize MT5 terminal and login to account.

        Returns:
            True if connection successful, False otherwise.
        """
        if mt5 is None:
            self.logger.error("MetaTrader5 package not available. Windows required.")
            return False

        # Initialize MT5
        init_kwargs = {}
        if mt5_config.MT5_PATH:
            init_kwargs["path"] = mt5_config.MT5_PATH

        if not mt5.initialize(**init_kwargs):
            error = mt5.last_error()
            self.logger.error(f"MT5 initialization failed: {error}")
            return False

        # Login to account
        authorized = mt5.login(
            login=mt5_config.MT5_ACCOUNT,
            password=mt5_config.MT5_PASSWORD,
            server=mt5_config.MT5_SERVER,
        )

        if not authorized:
            error = mt5.last_error()
            self.logger.error(f"MT5 login failed: {error}")
            mt5.shutdown()
            return False

        self._connected = True
        self._retry_count = 0

        account_info = mt5.account_info()
        if account_info:
            self.logger.info(
                f"Connected to MT5: Account #{account_info.login}, "
                f"Balance: {account_info.balance:.2f} {account_info.currency}, "
                f"Server: {account_info.server}"
            )
            if mt5_config.DEMO_MODE and account_info.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
                self.logger.warning(
                    "WARNING: DEMO_MODE is True but account is NOT a demo account! "
                    "Please verify your configuration."
                )

        return True

    def disconnect(self):
        """Shut down MT5 connection cleanly."""
        if mt5 is not None and self._connected:
            mt5.shutdown()
            self._connected = False
            self.logger.info("Disconnected from MT5")

    def is_connected(self) -> bool:
        """
        Check if MT5 connection is alive.

        Returns:
            True if connected and terminal is responsive.
        """
        if mt5 is None or not self._connected:
            return False

        # Try to get terminal info as a connectivity check
        info = mt5.terminal_info()
        if info is None:
            self._connected = False
            return False

        return info.connected

    def reconnect(self) -> bool:
        """
        Attempt to reconnect with exponential backoff.

        Returns:
            True if reconnection successful, False if max retries exceeded.
        """
        max_retries = mt5_config.RECONNECT_MAX_RETRIES
        base_delay = mt5_config.RECONNECT_BASE_DELAY
        max_delay = mt5_config.RECONNECT_MAX_DELAY

        for attempt in range(1, max_retries + 1):
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            self.logger.warning(
                f"Reconnection attempt {attempt}/{max_retries} "
                f"(waiting {delay:.1f}s)..."
            )
            time.sleep(delay)

            # Disconnect cleanly first
            self.disconnect()

            if self.connect():
                self.logger.info(f"Reconnected successfully on attempt {attempt}")
                return True

        self.logger.error(f"Failed to reconnect after {max_retries} attempts")
        return False

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current account information.

        Returns:
            Dictionary with account details or None if unavailable.
        """
        if not self.is_connected():
            return None

        info = mt5.account_info()
        if info is None:
            return None

        return {
            "login": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit,
            "currency": info.currency,
            "leverage": info.leverage,
            "server": info.server,
            "trade_mode": info.trade_mode,
        }

    def get_symbol_info(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get symbol information (tick size, spread, etc.).

        Args:
            symbol: Symbol name. Defaults to config SYMBOL.

        Returns:
            Dictionary with symbol details or None if unavailable.
        """
        symbol = symbol or mt5_config.SYMBOL
        info = mt5.symbol_info(symbol)
        if info is None:
            self.logger.error(f"Symbol {symbol} not found")
            return None

        # Ensure symbol is visible in Market Watch
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                self.logger.error(f"Failed to select symbol {symbol}")
                return None

        return {
            "name": info.name,
            "bid": info.bid,
            "ask": info.ask,
            "spread": info.spread,
            "point": info.point,
            "digits": info.digits,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_mode": info.trade_mode,
            "trade_tick_value": info.trade_tick_value,
        }

    def send_order(
        self,
        direction: str,
        volume: float,
        stop_loss: float,
        take_profit: float,
        symbol: Optional[str] = None,
        comment: str = "CCT_Rectangle",
    ) -> OrderResult:
        """
        Send a market order (buy or sell).

        Args:
            direction: 'buy' or 'sell'
            volume: Lot size
            stop_loss: Stop loss price
            take_profit: Take profit price
            symbol: Symbol to trade (defaults to config SYMBOL)
            comment: Order comment

        Returns:
            OrderResult with execution details.
        """
        symbol = symbol or mt5_config.SYMBOL

        if not self.is_connected():
            return OrderResult(
                success=False,
                error_message="Not connected to MT5",
            )

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(
                success=False,
                error_message=f"Failed to get tick for {symbol}",
            )

        # Determine order type and price
        if direction.lower() == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        elif direction.lower() == "sell":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            return OrderResult(
                success=False,
                error_message=f"Invalid direction: {direction}",
            )

        # Check spread
        spread_points = tick.ask - tick.bid
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info and symbol_info.point > 0:
            spread_in_points = int(spread_points / symbol_info.point)
            if spread_in_points > mt5_config.MAX_SPREAD_POINTS:
                return OrderResult(
                    success=False,
                    error_message=f"Spread too high: {spread_in_points} points "
                                  f"(max: {mt5_config.MAX_SPREAD_POINTS})",
                )

        # Build order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": mt5_config.SLIPPAGE_POINTS,
            "magic": mt5_config.MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        self.logger.info(
            f"Sending {direction.upper()} order: {volume} lots @ {price:.5f}, "
            f"SL={stop_loss:.5f}, TP={take_profit:.5f}"
        )

        # Send order
        result = mt5.order_send(request)

        if result is None:
            error = mt5.last_error()
            return OrderResult(
                success=False,
                error_code=error[0] if error else None,
                error_message=f"Order send returned None: {error}",
            )

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                success=False,
                error_code=result.retcode,
                error_message=f"Order failed: {result.comment} (code: {result.retcode})",
            )

        self.logger.info(
            f"Order executed: Ticket #{result.order}, "
            f"{direction.upper()} {volume} lots @ {result.price:.5f}"
        )

        return OrderResult(
            success=True,
            ticket=result.order,
            volume=result.volume,
            price=result.price,
        )

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open positions for the symbol.

        Args:
            symbol: Symbol filter. Defaults to config SYMBOL.

        Returns:
            List of position dictionaries.
        """
        symbol = symbol or mt5_config.SYMBOL

        if not self.is_connected():
            return []

        positions = mt5.positions_get(symbol=symbol)

        if positions is None or len(positions) == 0:
            return []

        result = []
        for pos in positions:
            # Only include positions opened by this bot
            if pos.magic != mt5_config.MAGIC_NUMBER:
                continue

            result.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "buy" if pos.type == mt5.POSITION_TYPE_BUY else "sell",
                "volume": pos.volume,
                "open_price": pos.price_open,
                "current_price": pos.price_current,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "swap": pos.swap,
                "time": pos.time,
                "magic": pos.magic,
                "comment": pos.comment,
            })

        return result

    def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """
        Modify SL/TP of an open position.

        Args:
            ticket: Position ticket number
            stop_loss: New stop loss (None to keep current)
            take_profit: New take profit (None to keep current)

        Returns:
            True if modification successful.
        """
        if not self.is_connected():
            return False

        # Get current position info
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            self.logger.error(f"Position {ticket} not found")
            return False

        pos = position[0]
        new_sl = stop_loss if stop_loss is not None else pos.sl
        new_tp = take_profit if take_profit is not None else pos.tp

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": pos.symbol,
            "sl": new_sl,
            "tp": new_tp,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = result.comment if result else "Unknown error"
            self.logger.error(f"Failed to modify position {ticket}: {error_msg}")
            return False

        self.logger.info(
            f"Position {ticket} modified: SL={new_sl:.5f}, TP={new_tp:.5f}"
        )
        return True

    def close_position(self, ticket: int) -> bool:
        """
        Close an open position by ticket number.

        Args:
            ticket: Position ticket to close.

        Returns:
            True if position closed successfully.
        """
        if not self.is_connected():
            return False

        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            self.logger.error(f"Position {ticket} not found for closing")
            return False

        pos = position[0]

        # Determine close direction (opposite of position type)
        if pos.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(pos.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(pos.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": mt5_config.SLIPPAGE_POINTS,
            "magic": mt5_config.MAGIC_NUMBER,
            "comment": "CCT_Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = result.comment if result else "Unknown error"
            self.logger.error(f"Failed to close position {ticket}: {error_msg}")
            return False

        self.logger.info(f"Position {ticket} closed at {price:.5f}")
        return True

    def get_closed_positions(self, since_seconds: int = 300) -> List[Dict[str, Any]]:
        """
        Get positions closed within the last N seconds (via deal history).

        Uses mt5.history_deals_get() to detect positions that were closed
        by SL, TP, or manual close since the last check.

        Args:
            since_seconds: Look back this many seconds for closed deals.

        Returns:
            List of closed-deal dictionaries with P&L information.
        """
        if not self.is_connected():
            return []

        now = datetime.now(timezone.utc)
        from_time = now - timedelta(seconds=since_seconds)

        deals = mt5.history_deals_get(from_time, now)
        if deals is None or len(deals) == 0:
            return []

        closed = []
        for deal in deals:
            # Only include exit deals from this bot (DEAL_ENTRY_OUT = 1)
            if deal.magic != mt5_config.MAGIC_NUMBER:
                continue
            if deal.entry != 1:  # 1 = DEAL_ENTRY_OUT (closing a position)
                continue

            closed.append({
                "ticket": deal.position_id,
                "deal_ticket": deal.ticket,
                "symbol": deal.symbol,
                "type": "buy" if deal.type == 0 else "sell",  # deal close type
                "volume": deal.volume,
                "price": deal.price,
                "profit": deal.profit,
                "swap": deal.swap,
                "commission": deal.commission,
                "time": deal.time,
                "comment": deal.comment,
            })

        return closed

    def get_current_price(self, symbol: Optional[str] = None) -> Optional[Dict[str, float]]:
        """
        Get current bid/ask price.

        Args:
            symbol: Symbol name. Defaults to config SYMBOL.

        Returns:
            Dictionary with 'bid' and 'ask' or None.
        """
        symbol = symbol or mt5_config.SYMBOL

        if not self.is_connected():
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "time": tick.time,
        }
