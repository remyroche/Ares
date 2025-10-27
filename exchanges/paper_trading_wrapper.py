"""
Paper Trading Wrapper for ExchangeInterface

This module provides a wrapper around ExchangeInterface that redirects
trading operations to a paper trading simulator when in PAPER mode.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .base_exchange.exchange_interface import (
    IExchange, OrderSide, OrderType, OrderStatus, ExchangeStatus
)
from .paper_trading_simulator import PaperTradingSimulator, SimulatorConfig


class PaperTradingWrapper:
    """
    Wrapper that intercepts trading operations and redirects them to a simulator
    when in PAPER mode, while allowing normal exchange operations for data fetching.
    """
    
    def __init__(self, exchange: IExchange, trading_mode: str = "TRADE", 
                 simulator_config: Optional[SimulatorConfig] = None):
        """
        Initialize the paper trading wrapper.
        
        Args:
            exchange: The underlying exchange interface
            trading_mode: "TRADE" for live trading, "PAPER" for simulation
            simulator_config: Configuration for the paper trading simulator
        """
        self.exchange = exchange
        self.trading_mode = trading_mode.upper()
        self.logger = logging.getLogger(__name__)
        
        # Initialize simulator if in paper mode
        if self.trading_mode == "PAPER":
            self.simulator = PaperTradingSimulator(simulator_config)
            self.logger.info("Paper trading mode enabled - orders will be simulated")
        else:
            self.simulator = None
            self.logger.info("Live trading mode enabled - orders will be sent to exchange")
    
    def set_trading_mode(self, mode: str) -> None:
        """Set trading mode (TRADE or PAPER)"""
        self.trading_mode = mode.upper()
        
        if self.trading_mode == "PAPER" and self.simulator is None:
            self.simulator = PaperTradingSimulator()
            self.logger.info("Switched to paper trading mode")
        elif self.trading_mode == "TRADE":
            self.logger.info("Switched to live trading mode")
    
    def is_paper_mode(self) -> bool:
        """Check if currently in paper trading mode"""
        return self.trading_mode == "PAPER"
    
    def is_trade_mode(self) -> bool:
        """Check if currently in live trading mode"""
        return self.trading_mode == "TRADE"
    
    # Delegate all non-trading methods to the underlying exchange
    async def initialize(self) -> None:
        """Initialize the exchange connection"""
        await self.exchange.initialize()
    
    async def close(self) -> None:
        """Close the exchange connection"""
        await self.exchange.close()
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def get_status(self) -> ExchangeStatus:
        """Get current exchange status"""
        return await self.exchange.get_status()
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        if self.is_paper_mode():
            # Return simulated account info
            balance = self.simulator.get_balance()
            positions = self.simulator.get_positions()
            performance = self.simulator.get_performance_metrics()
            
            return {
                "account_type": "paper_trading",
                "balance": balance,
                "positions": positions,
                "performance": performance,
                "simulator_status": self.simulator.get_simulator_status()
            }
        else:
            return await self.exchange.get_account_info()
    
    async def get_balance(self, currency: str) -> Dict[str, Any]:
        """Get balance for a specific currency"""
        if self.is_paper_mode():
            balance = self.simulator.get_balance()
            return {
                "currency": currency,
                "free": balance.get(currency, 0.0),
                "locked": 0.0,
                "total": balance.get(currency, 0.0)
            }
        else:
            return await self.exchange.get_balance(currency)
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker information"""
        # Always use real exchange for market data
        return await self.exchange.get_ticker(symbol)
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get kline/candlestick data"""
        # Always use real exchange for market data
        return await self.exchange.get_klines(symbol, interval, limit)
    
    # Trading methods - redirect to simulator in PAPER mode
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new order.
        In PAPER mode, redirects to simulator.
        In TRADE mode, sends to real exchange.
        """
        if self.is_paper_mode():
            self.logger.info(f"Paper trading: Creating {side.value} order for {quantity} {symbol}")
            
            # Update simulator with current market price
            try:
                ticker = await self.get_ticker(symbol)
                current_price = float(ticker.get("last", 0))
                if current_price > 0:
                    self.simulator.update_market_price(symbol, current_price)
            except Exception as e:
                self.logger.warning(f"Could not update market price for {symbol}: {e}")
            
            # Create order in simulator
            return await self.simulator.create_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                **kwargs
            )
        else:
            self.logger.info(f"Live trading: Creating {side.value} order for {quantity} {symbol}")
            return await self.exchange.create_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                **kwargs
            )
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order.
        In PAPER mode, cancels in simulator.
        In TRADE mode, cancels on real exchange.
        """
        if self.is_paper_mode():
            self.logger.info(f"Paper trading: Cancelling order {order_id}")
            return await self.simulator.cancel_order(order_id)
        else:
            self.logger.info(f"Live trading: Cancelling order {order_id}")
            return await self.exchange.cancel_order(order_id)
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of an order.
        In PAPER mode, gets status from simulator.
        In TRADE mode, gets status from real exchange.
        """
        if self.is_paper_mode():
            return await self.simulator.get_order_status(order_id)
        else:
            return await self.exchange.get_order_status(order_id)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders.
        In PAPER mode, gets orders from simulator.
        In TRADE mode, gets orders from real exchange.
        """
        if self.is_paper_mode():
            return await self.simulator.get_open_orders(symbol)
        else:
            return await self.exchange.get_open_orders(symbol)
    
    # Additional methods for paper trading
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions (paper trading only)"""
        if self.is_paper_mode():
            return self.simulator.get_positions()
        else:
            return []
    
    def get_trade_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history (paper trading only)"""
        if self.is_paper_mode():
            return self.simulator.get_trade_history(symbol, limit)
        else:
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics (paper trading only)"""
        if self.is_paper_mode():
            return self.simulator.get_performance_metrics()
        else:
            return {}
    
    def get_simulator_status(self) -> Dict[str, Any]:
        """Get simulator status (paper trading only)"""
        if self.is_paper_mode():
            return self.simulator.get_simulator_status()
        else:
            return {"mode": "live_trading", "simulator_available": False}
    
    def update_market_price(self, symbol: str, price: float) -> None:
        """Update market price in simulator (paper trading only)"""
        if self.is_paper_mode():
            self.simulator.update_market_price(symbol, price)
    
    def reset_daily_counters(self) -> None:
        """Reset daily trade counters (paper trading only)"""
        if self.is_paper_mode():
            self.simulator.reset_daily_counters()
    
    # Method to get the underlying exchange for direct access if needed
    def get_underlying_exchange(self) -> IExchange:
        """Get the underlying exchange interface"""
        return self.exchange
    
    # Method to get the simulator for direct access if needed
    def get_simulator(self) -> Optional[PaperTradingSimulator]:
        """Get the paper trading simulator"""
        return self.simulator


def create_paper_trading_wrapper(
    exchange: IExchange, 
    trading_mode: str = "TRADE",
    simulator_config: Optional[SimulatorConfig] = None
) -> PaperTradingWrapper:
    """
    Factory function to create a paper trading wrapper.
    
    Args:
        exchange: The underlying exchange interface
        trading_mode: "TRADE" for live trading, "PAPER" for simulation
        simulator_config: Configuration for the paper trading simulator
        
    Returns:
        PaperTradingWrapper instance
    """
    return PaperTradingWrapper(exchange, trading_mode, simulator_config)