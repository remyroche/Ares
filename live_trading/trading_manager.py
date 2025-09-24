"""
Live Trading Manager

This is the core module for live trading that handles:
- Order placement and management
- Data streaming and processing
- Exchange integration
- Real-time trading operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass

from src.interfaces.base_interfaces import (
    MarketData,
    TradeDecision,
    AnalysisResult,
    StrategyResult
)
from ..exchange.factory import ExchangeFactory
from .order_manager import OrderManager
from .data_receiver import DataReceiver
from .trade_executor import TradeExecutor


@dataclass
class TradingConfig:
    """Configuration for live trading operations"""
    exchange_name: str
    symbols: List[str]
    max_position_size: float
    max_daily_trades: int
    risk_per_trade: float
    enable_data_streaming: bool = True
    enable_order_execution: bool = True
    api_key: str = ""
    api_secret: str = ""


class TradingManager:
    """
    Main trading manager for live trading operations.

    This class coordinates all live trading activities including:
    - Order placement and management
    - Market data streaming
    - Risk management
    - Position tracking
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.exchange = None
        self.order_manager = None
        self.data_receiver = None
        self.trade_executor = None

        # Trading state
        self.is_running = False
        self.active_positions = {}
        self.open_orders = {}
        self.daily_trade_count = 0
        self.last_trade_time = None

        # Event callbacks
        self.on_order_update: Optional[Callable] = None
        self.on_data_update: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None

    async def initialize(self) -> bool:
        """Initialize the trading manager and all components."""
        try:
            self.logger.info(f"Initializing TradingManager for {self.config.exchange_name}")

            # Initialize exchange
            self.exchange = ExchangeFactory.get_exchange(self.config.exchange_name)
            if not self.exchange:
                raise ValueError(f"Failed to initialize exchange: {self.config.exchange_name}")

            # Initialize components
            self.order_manager = OrderManager(self.exchange, self.config)
            self.data_receiver = DataReceiver(self.exchange, self.config.symbols)
            self.trade_executor = TradeExecutor(self.exchange, self.config)

            # Set up event handlers
            await self._setup_event_handlers()

            self.logger.info("✅ TradingManager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize TradingManager: {e}")
            return False

    async def _setup_event_handlers(self):
        """Set up event handlers for order and data updates."""
        # Order update handler
        self.order_manager.on_order_update = self._handle_order_update

        # Data update handler
        self.data_receiver.on_data_update = self._handle_data_update

        # Position update handler
        self.trade_executor.on_position_update = self._handle_position_update

    async def start(self) -> bool:
        """Start live trading operations."""
        try:
            if self.is_running:
                self.logger.warning("TradingManager is already running")
                return True

            self.logger.info("Starting TradingManager...")

            # Initialize if not already done
            if not self.exchange:
                await self.initialize()

            # Start data streaming
            if self.config.enable_data_streaming:
                await self.data_receiver.start()

            # Start order monitoring
            if self.config.enable_order_execution:
                await self.order_manager.start()

            self.is_running = True
            self.logger.info("✅ TradingManager started successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to start TradingManager: {e}")
            return False

    async def stop(self) -> None:
        """Stop live trading operations."""
        try:
            self.logger.info("Stopping TradingManager...")

            self.is_running = False

            # Stop components
            if self.data_receiver:
                await self.data_receiver.stop()

            if self.order_manager:
                await self.order_manager.stop()

            if self.exchange:
                await self.exchange.close()

            self.logger.info("✅ TradingManager stopped successfully")

        except Exception as e:
            self.logger.error(f"❌ Error stopping TradingManager: {e}")

    async def place_order(self, trade_decision: TradeDecision) -> Optional[Dict[str, Any]]:
        """
        Place a trading order.

        Args:
            trade_decision: The trade decision to execute

        Returns:
            Order result dictionary or None if failed
        """
        try:
            if not self.is_running:
                self.logger.warning("Cannot place order: TradingManager not running")
                return None

            # Check daily trade limits
            if not self._check_trade_limits():
                self.logger.warning("Daily trade limit exceeded")
                return None

            # Check position limits
            if not self._check_position_limits(trade_decision):
                self.logger.warning("Position limit exceeded")
                return None

            # Execute the order
            result = await self.trade_executor.execute_trade(trade_decision)

            if result:
                self.daily_trade_count += 1
                self.last_trade_time = datetime.now()

                # Update position tracking
                await self._update_position_tracking(trade_decision, result)

                self.logger.info(f"✅ Order placed successfully: {result.get('orderId', 'N/A')}")
            else:
                self.logger.error("❌ Failed to place order")

            return result

        except Exception as e:
            self.logger.error(f"❌ Error placing order: {e}")
            return None

    async def cancel_order(self, symbol: str, order_id: Any) -> bool:
        """
        Cancel an open order.

        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel

        Returns:
            True if successfully cancelled, False otherwise
        """
        try:
            if not self.order_manager:
                return False

            result = await self.order_manager.cancel_order(symbol, order_id)
            if result:
                self.logger.info(f"✅ Order cancelled: {order_id}")
            else:
                self.logger.error(f"❌ Failed to cancel order: {order_id}")

            return result is not None

        except Exception as e:
            self.logger.error(f"❌ Error cancelling order: {e}")
            return False

    async def get_market_data(self, symbol: str, interval: str = "1m", limit: int = 100) -> List[MarketData]:
        """
        Get market data for a symbol.

        Args:
            symbol: Trading symbol
            interval: Data interval
            limit: Number of data points

        Returns:
            List of MarketData objects
        """
        try:
            if not self.exchange:
                return []

            return await self.exchange.get_klines(symbol, interval, limit)

        except Exception as e:
            self.logger.error(f"❌ Error getting market data: {e}")
            return []

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            if not self.exchange:
                return {}

            return await self.exchange.get_account_info()

        except Exception as e:
            self.logger.error(f"❌ Error getting account info: {e}")
            return {}

    async def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        try:
            if not self.exchange:
                return {}

            positions = {}
            for symbol in self.config.symbols:
                try:
                    position = await self.exchange.get_position_risk(symbol)
                    if position:
                        positions[symbol] = position
                except Exception:
                    continue

            return positions

        except Exception as e:
            self.logger.error(f"❌ Error getting positions: {e}")
            return {}

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        try:
            if not self.exchange:
                return []

            return await self.exchange.get_open_orders(symbol)

        except Exception as e:
            self.logger.error(f"❌ Error getting open orders: {e}")
            return []

    def _check_trade_limits(self) -> bool:
        """Check if daily trade limits are exceeded."""
        if self.daily_trade_count >= self.config.max_daily_trades:
            return False

        # Reset daily count if it's a new day
        now = datetime.now()
        if self.last_trade_time:
            if (now - self.last_trade_time).days >= 1:
                self.daily_trade_count = 0

        return True

    def _check_position_limits(self, trade_decision: TradeDecision) -> bool:
        """Check if position limits are exceeded."""
        try:
            # Get current positions
            positions = asyncio.run(self.get_positions())

            # Check if adding this position would exceed limits
            current_exposure = sum(
                abs(float(pos.get('positionAmt', 0))) * float(pos.get('markPrice', 0))
                for pos in positions.values()
                if pos.get('positionAmt', 0) != 0
            )

            new_exposure = trade_decision.quantity * trade_decision.price
            total_exposure = current_exposure + new_exposure

            return total_exposure <= self.config.max_position_size

        except Exception as e:
            self.logger.error(f"❌ Error checking position limits: {e}")
            return False

    async def _update_position_tracking(self, trade_decision: TradeDecision, order_result: Dict[str, Any]):
        """Update internal position tracking."""
        symbol = trade_decision.symbol
        if symbol not in self.active_positions:
            self.active_positions[symbol] = []

        self.active_positions[symbol].append({
            'timestamp': datetime.now(),
            'order_id': order_result.get('orderId'),
            'quantity': trade_decision.quantity,
            'price': trade_decision.price,
            'side': trade_decision.action
        })

    async def _handle_order_update(self, order_update: Dict[str, Any]):
        """Handle order status updates."""
        if self.on_order_update:
            await self.on_order_update(order_update)

        # Update internal tracking
        symbol = order_update.get('symbol')
        order_id = order_update.get('orderId')

        if symbol and order_id:
            self.logger.info(f"Order update: {symbol} {order_id} - {order_update.get('status', 'UNKNOWN')}")

    async def _handle_data_update(self, market_data: MarketData):
        """Handle new market data."""
        if self.on_data_update:
            await self.on_data_update(market_data)

        self.logger.debug(f"Market data update: {market_data.symbol} @ {market_data.close}")

    async def _handle_position_update(self, position_update: Dict[str, Any]):
        """Handle position updates."""
        if self.on_position_update:
            await self.on_position_update(position_update)

        symbol = position_update.get('symbol')
        if symbol:
            self.logger.info(f"Position update: {symbol} - {position_update.get('positionAmt', 0)}")

    # Configuration methods
    def update_config(self, new_config: TradingConfig):
        """Update trading configuration."""
        self.config = new_config
        self.logger.info("Trading configuration updated")

    def set_order_update_callback(self, callback: Callable):
        """Set callback for order updates."""
        self.on_order_update = callback

    def set_data_update_callback(self, callback: Callable):
        """Set callback for data updates."""
        self.on_data_update = callback

    def set_position_update_callback(self, callback: Callable):
        """Set callback for position updates."""
        self.on_position_update = callback