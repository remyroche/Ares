"""
Mode-Aware Exchange Interface

This module provides a mode-aware wrapper around the ExchangeInterface that can route orders
to either real exchanges (TRADE mode) or the simulator (PAPER mode) based on configuration flags.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass

from .base_exchange.exchange_interface import IExchange, OrderRequest, OrderResponse, OrderSide, OrderType, OrderStatus
from .order_router import OrderRouter
from .exchange_registry import ExchangeRegistry
from simulator import OrderSimulator, PortfolioSimulator


class TradingMode(Enum):
    """Trading mode enumeration"""
    TRADE = "trade"
    PAPER = "paper"


@dataclass
class ModeAwareConfig:
    """Configuration for mode-aware exchange interface"""
    mode: TradingMode
    initial_balance: float = 100000.0
    enable_order_book_simulation: bool = True
    simulation_commission_rate: float = 0.001
    log_trades: bool = True


class ModeAwareExchangeInterface:
    """
    Mode-aware exchange interface that routes orders based on trading mode.
    
    In TRADE mode: Orders are sent to real exchanges
    In PAPER mode: Orders are sent to the simulator
    """

    def __init__(
        self,
        config: ModeAwareConfig,
        exchange_registry: Optional[ExchangeRegistry] = None,
        order_router: Optional[OrderRouter] = None
    ):
        """
        Initialize the mode-aware exchange interface
        
        Args:
            config: Configuration for the interface
            exchange_registry: Registry for real exchanges (required for TRADE mode)
            order_router: Router for real exchange orders (required for TRADE mode)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Real exchange components (for TRADE mode)
        self.exchange_registry = exchange_registry
        self.order_router = order_router
        
        # Simulator components (for PAPER mode)
        self.simulator: Optional[OrderSimulator] = None
        self.portfolio_simulator: Optional[PortfolioSimulator] = None
        
        # Initialize based on mode
        if self.config.mode == TradingMode.PAPER:
            self._initialize_simulator()
        elif self.config.mode == TradingMode.TRADE:
            self._validate_trade_mode_components()
        
        self.logger.info(f"Mode-aware exchange interface initialized in {self.config.mode.value} mode")

    def _initialize_simulator(self) -> None:
        """Initialize simulator components for PAPER mode"""
        try:
            self.simulator = OrderSimulator(initial_balance=self.config.initial_balance)
            self.portfolio_simulator = PortfolioSimulator(initial_balance=self.config.initial_balance)
            self.logger.info("Simulator components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize simulator: {e}")
            raise

    def _validate_trade_mode_components(self) -> None:
        """Validate that required components are available for TRADE mode"""
        if self.exchange_registry is None:
            raise ValueError("ExchangeRegistry is required for TRADE mode")
        if self.order_router is None:
            raise ValueError("OrderRouter is required for TRADE mode")

    async def initialize(self) -> None:
        """Initialize the mode-aware interface"""
        try:
            if self.config.mode == TradingMode.PAPER and self.simulator:
                await self.simulator.initialize()
                self.logger.info("Simulator initialized")
            elif self.config.mode == TradingMode.TRADE:
                if self.order_router:
                    await self.order_router.start()
                    self.logger.info("Order router started")
        except Exception as e:
            self.logger.error(f"Failed to initialize mode-aware interface: {e}")
            raise

    async def close(self) -> None:
        """Close the mode-aware interface"""
        try:
            if self.config.mode == TradingMode.PAPER and self.simulator:
                await self.simulator.close()
                self.logger.info("Simulator closed")
            elif self.config.mode == TradingMode.TRADE and self.order_router:
                await self.order_router.stop()
                self.logger.info("Order router stopped")
        except Exception as e:
            self.logger.error(f"Error closing mode-aware interface: {e}")

    async def create_order(
        self,
        exchange: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create an order (mode-aware)
        
        Args:
            exchange: Exchange name (for TRADE mode) or ignored (for PAPER mode)
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET/LIMIT/etc)
            quantity: Order quantity
            price: Order price (for limit orders)
            **kwargs: Additional order parameters
            
        Returns:
            Order response dictionary
        """
        try:
            if self.config.mode == TradingMode.PAPER:
                return await self._create_simulated_order(
                    symbol, side, order_type, quantity, price, **kwargs
                )
            elif self.config.mode == TradingMode.TRADE:
                return await self._create_real_order(
                    exchange, symbol, side, order_type, quantity, price, **kwargs
                )
            else:
                raise ValueError(f"Unsupported trading mode: {self.config.mode}")
                
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            return {
                "success": False,
                "error": str(e),
                "mode": self.config.mode.value
            }

    async def cancel_order(self, order_id: str, exchange: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel an order (mode-aware)
        
        Args:
            order_id: Order ID to cancel
            exchange: Exchange name (for TRADE mode) or ignored (for PAPER mode)
            
        Returns:
            Cancel response dictionary
        """
        try:
            if self.config.mode == TradingMode.PAPER:
                return await self._cancel_simulated_order(order_id)
            elif self.config.mode == TradingMode.TRADE:
                return await self._cancel_real_order(order_id, exchange)
            else:
                raise ValueError(f"Unsupported trading mode: {self.config.mode}")
                
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return {
                "success": False,
                "error": str(e),
                "mode": self.config.mode.value
            }

    async def get_order_status(self, order_id: str, exchange: Optional[str] = None) -> Dict[str, Any]:
        """
        Get order status (mode-aware)
        
        Args:
            order_id: Order ID to check
            exchange: Exchange name (for TRADE mode) or ignored (for PAPER mode)
            
        Returns:
            Order status dictionary
        """
        try:
            if self.config.mode == TradingMode.PAPER:
                return await self._get_simulated_order_status(order_id)
            elif self.config.mode == TradingMode.TRADE:
                return await self._get_real_order_status(order_id, exchange)
            else:
                raise ValueError(f"Unsupported trading mode: {self.config.mode}")
                
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {
                "success": False,
                "error": str(e),
                "mode": self.config.mode.value
            }

    async def get_open_orders(self, exchange: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders (mode-aware)
        
        Args:
            exchange: Exchange name (for TRADE mode) or ignored (for PAPER mode)
            symbol: Symbol filter (optional)
            
        Returns:
            List of open orders
        """
        try:
            if self.config.mode == TradingMode.PAPER:
                return await self._get_simulated_open_orders(symbol)
            elif self.config.mode == TradingMode.TRADE:
                return await self._get_real_open_orders(exchange, symbol)
            else:
                raise ValueError(f"Unsupported trading mode: {self.config.mode}")
                
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            return []

    async def get_positions(self, exchange: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get positions (mode-aware)
        
        Args:
            exchange: Exchange name (for TRADE mode) or ignored (for PAPER mode)
            symbol: Symbol filter (optional)
            
        Returns:
            List of positions
        """
        try:
            if self.config.mode == TradingMode.PAPER:
                return await self._get_simulated_positions(symbol)
            elif self.config.mode == TradingMode.TRADE:
                return await self._get_real_positions(exchange, symbol)
            else:
                raise ValueError(f"Unsupported trading mode: {self.config.mode}")
                
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []

    async def get_account_info(self, exchange: Optional[str] = None) -> Dict[str, Any]:
        """
        Get account information (mode-aware)
        
        Args:
            exchange: Exchange name (for TRADE mode) or ignored (for PAPER mode)
            
        Returns:
            Account information dictionary
        """
        try:
            if self.config.mode == TradingMode.PAPER:
                return await self._get_simulated_account_info()
            elif self.config.mode == TradingMode.TRADE:
                return await self._get_real_account_info(exchange)
            else:
                raise ValueError(f"Unsupported trading mode: {self.config.mode}")
                
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {
                "success": False,
                "error": str(e),
                "mode": self.config.mode.value
            }

    async def update_market_data(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """
        Update market data for simulation (PAPER mode only)
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
        """
        if self.config.mode == TradingMode.PAPER and self.simulator:
            await self.simulator.update_market_data(symbol, market_data)
        else:
            self.logger.warning("Market data updates only supported in PAPER mode")

    async def process_order_book(self, symbol: str, order_book: Dict[str, Any]) -> None:
        """
        Process order book data for simulation (PAPER mode only)
        
        Args:
            symbol: Trading symbol
            order_book: Order book data
        """
        if self.config.mode == TradingMode.PAPER and self.simulator:
            await self.simulator.process_order_book(symbol, order_book)
        else:
            self.logger.warning("Order book processing only supported in PAPER mode")

    # Simulator methods (PAPER mode)
    async def _create_simulated_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a simulated order"""
        if not self.simulator:
            raise RuntimeError("Simulator not initialized")
        
        result = await self.simulator.create_order(
            symbol, side, order_type, quantity, price, **kwargs
        )
        
        # Log trade if enabled
        if self.config.log_trades and result.get("success"):
            self.logger.info(f"SIMULATED ORDER: {side} {quantity} {symbol} @ {price or 'MARKET'}")
        
        return result

    async def _cancel_simulated_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a simulated order"""
        if not self.simulator:
            raise RuntimeError("Simulator not initialized")
        
        return await self.simulator.cancel_order(order_id)

    async def _get_simulated_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get simulated order status"""
        if not self.simulator:
            raise RuntimeError("Simulator not initialized")
        
        return await self.simulator.get_order_status(order_id)

    async def _get_simulated_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get simulated open orders"""
        if not self.simulator:
            raise RuntimeError("Simulator not initialized")
        
        return await self.simulator.get_open_orders(symbol)

    async def _get_simulated_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get simulated positions"""
        if not self.simulator:
            raise RuntimeError("Simulator not initialized")
        
        return await self.simulator.get_positions(symbol)

    async def _get_simulated_account_info(self) -> Dict[str, Any]:
        """Get simulated account info"""
        if not self.simulator:
            raise RuntimeError("Simulator not initialized")
        
        portfolio_value = await self.simulator.get_portfolio_value()
        
        return {
            "success": True,
            "mode": "PAPER",
            "account_type": "simulated",
            "can_trade": True,
            "can_withdraw": False,
            "can_deposit": False,
            "portfolio": portfolio_value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # Real exchange methods (TRADE mode)
    async def _create_real_order(
        self,
        exchange: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a real order"""
        if not self.order_router:
            raise RuntimeError("Order router not initialized")
        
        result = await self.order_router.route_order(
            exchange, symbol, side, order_type, quantity, price, **kwargs
        )
        
        # Log trade if enabled
        if self.config.log_trades and result.get("success"):
            self.logger.info(f"REAL ORDER: {side} {quantity} {symbol} @ {price or 'MARKET'} on {exchange}")
        
        return result

    async def _cancel_real_order(self, order_id: str, exchange: Optional[str] = None) -> Dict[str, Any]:
        """Cancel a real order"""
        if not self.order_router:
            raise RuntimeError("Order router not initialized")
        
        return await self.order_router.cancel_order(order_id)

    async def _get_real_order_status(self, order_id: str, exchange: Optional[str] = None) -> Dict[str, Any]:
        """Get real order status"""
        if not self.order_router:
            raise RuntimeError("Order router not initialized")
        
        return await self.order_router.get_order_status(order_id)

    async def _get_real_open_orders(self, exchange: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get real open orders"""
        if not self.order_router:
            raise RuntimeError("Order router not initialized")
        
        return await self.order_router.get_active_orders(exchange, symbol)

    async def _get_real_positions(self, exchange: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get real positions"""
        if not self.exchange_registry:
            raise RuntimeError("Exchange registry not initialized")
        
        positions = []
        
        if exchange:
            # Get positions from specific exchange
            exchange_instance = await self.exchange_registry.get_exchange(exchange)
            if exchange_instance and hasattr(exchange_instance, 'get_positions'):
                try:
                    exchange_positions = await exchange_instance.get_positions(symbol)
                    positions.extend(exchange_positions)
                except Exception as e:
                    self.logger.error(f"Error getting positions from {exchange}: {e}")
        else:
            # Get positions from all exchanges
            all_exchanges = await self.exchange_registry.get_all_exchanges()
            for exchange_name, exchange_instance in all_exchanges.items():
                if hasattr(exchange_instance, 'get_positions'):
                    try:
                        exchange_positions = await exchange_instance.get_positions(symbol)
                        positions.extend(exchange_positions)
                    except Exception as e:
                        self.logger.error(f"Error getting positions from {exchange_name}: {e}")
        
        return positions

    async def _get_real_account_info(self, exchange: Optional[str] = None) -> Dict[str, Any]:
        """Get real account info"""
        if not self.exchange_registry:
            raise RuntimeError("Exchange registry not initialized")
        
        if exchange:
            # Get account info from specific exchange
            exchange_instance = await self.exchange_registry.get_exchange(exchange)
            if exchange_instance and hasattr(exchange_instance, 'get_account_info'):
                try:
                    return await exchange_instance.get_account_info()
                except Exception as e:
                    self.logger.error(f"Error getting account info from {exchange}: {e}")
                    return {"success": False, "error": str(e)}
        else:
            # Get account info from all exchanges
            all_exchanges = await self.exchange_registry.get_all_exchanges()
            account_info = {}
            
            for exchange_name, exchange_instance in all_exchanges.items():
                if hasattr(exchange_instance, 'get_account_info'):
                    try:
                        exchange_account = await exchange_instance.get_account_info()
                        account_info[exchange_name] = exchange_account
                    except Exception as e:
                        self.logger.error(f"Error getting account info from {exchange_name}: {e}")
                        account_info[exchange_name] = {"success": False, "error": str(e)}
            
            return {
                "success": True,
                "mode": "TRADE",
                "exchanges": account_info,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_mode(self) -> TradingMode:
        """Get current trading mode"""
        return self.config.mode

    def set_mode(self, mode: TradingMode) -> None:
        """Set trading mode (requires reinitialization)"""
        self.config.mode = mode
        self.logger.info(f"Trading mode changed to {mode.value}")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get interface statistics"""
        try:
            stats = {
                "mode": self.config.mode.value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if self.config.mode == TradingMode.PAPER and self.simulator:
                portfolio_value = await self.simulator.get_portfolio_value()
                stats.update({
                    "simulator": portfolio_value,
                    "initial_balance": self.config.initial_balance
                })
            elif self.config.mode == TradingMode.TRADE and self.order_router:
                router_stats = await self.order_router.get_statistics()
                stats.update({
                    "order_router": router_stats
                })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {
                "mode": self.config.mode.value,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }