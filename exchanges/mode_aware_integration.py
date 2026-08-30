"""
Mode-Aware Integration

Simple integration script for using the mode-aware exchange interface.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from .mode_aware_exchange_interface import ModeAwareExchangeInterface, ModeAwareConfig, TradingMode
from .trading_mode_config import get_trading_mode_config


class ModeAwareTradingClient:
    """
    Simple client for mode-aware trading
    
    This client automatically handles the mode switching and provides
    a simple interface for both paper and live trading.
    """

    def __init__(self, config: Optional[ModeAwareConfig] = None):
        """
        Initialize the trading client
        
        Args:
            config: Optional configuration. If None, loads from environment.
        """
        self.config = config or get_trading_mode_config()
        self.interface: Optional[ModeAwareExchangeInterface] = None
        self.logger = logging.getLogger(__name__)

    async def start(self) -> None:
        """Start the trading client"""
        try:
            self.interface = ModeAwareExchangeInterface(config=self.config)
            await self.interface.initialize()
            self.logger.info(f"Trading client started in {self.config.mode.value} mode")
        except Exception as e:
            self.logger.error(f"Failed to start trading client: {e}")
            raise

    async def stop(self) -> None:
        """Stop the trading client"""
        if self.interface:
            await self.interface.close()
            self.logger.info("Trading client stopped")

    async def buy(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        exchange: str = "default"
    ) -> Dict[str, Any]:
        """
        Place a buy order
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            quantity: Order quantity
            price: Order price (None for market order)
            exchange: Exchange name (ignored in paper mode)
            
        Returns:
            Order result dictionary
        """
        if not self.interface:
            raise RuntimeError("Trading client not started")
        
        order_type = "LIMIT" if price else "MARKET"
        
        return await self.interface.create_order(
            exchange=exchange,
            symbol=symbol,
            side="BUY",
            order_type=order_type,
            quantity=quantity,
            price=price
        )

    async def sell(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        exchange: str = "default"
    ) -> Dict[str, Any]:
        """
        Place a sell order
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            quantity: Order quantity
            price: Order price (None for market order)
            exchange: Exchange name (ignored in paper mode)
            
        Returns:
            Order result dictionary
        """
        if not self.interface:
            raise RuntimeError("Trading client not started")
        
        order_type = "LIMIT" if price else "MARKET"
        
        return await self.interface.create_order(
            exchange=exchange,
            symbol=symbol,
            side="SELL",
            order_type=order_type,
            quantity=quantity,
            price=price
        )

    async def cancel_order(self, order_id: str, exchange: str = "default") -> Dict[str, Any]:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            exchange: Exchange name (ignored in paper mode)
            
        Returns:
            Cancel result dictionary
        """
        if not self.interface:
            raise RuntimeError("Trading client not started")
        
        return await self.interface.cancel_order(order_id, exchange)

    async def get_order_status(self, order_id: str, exchange: str = "default") -> Dict[str, Any]:
        """
        Get order status
        
        Args:
            order_id: Order ID to check
            exchange: Exchange name (ignored in paper mode)
            
        Returns:
            Order status dictionary
        """
        if not self.interface:
            raise RuntimeError("Trading client not started")
        
        return await self.interface.get_order_status(order_id, exchange)

    async def get_positions(self, symbol: Optional[str] = None, exchange: str = "default") -> list:
        """
        Get current positions
        
        Args:
            symbol: Symbol filter (optional)
            exchange: Exchange name (ignored in paper mode)
            
        Returns:
            List of positions
        """
        if not self.interface:
            raise RuntimeError("Trading client not started")
        
        return await self.interface.get_positions(exchange, symbol)

    async def get_balance(self, exchange: str = "default") -> Dict[str, Any]:
        """
        Get account balance
        
        Args:
            exchange: Exchange name (ignored in paper mode)
            
        Returns:
            Balance information
        """
        if not self.interface:
            raise RuntimeError("Trading client not started")
        
        return await self.interface.get_account_info(exchange)

    async def update_market_data(self, symbol: str, price: float, **kwargs) -> None:
        """
        Update market data for simulation (paper mode only)
        
        Args:
            symbol: Trading symbol
            price: Current price
            **kwargs: Additional market data
        """
        if not self.interface:
            raise RuntimeError("Trading client not started")
        
        market_data = {"price": price, **kwargs}
        await self.interface.update_market_data(symbol, market_data)

    def is_paper_mode(self) -> bool:
        """Check if in paper trading mode"""
        return self.config.mode == TradingMode.PAPER

    def is_live_mode(self) -> bool:
        """Check if in live trading mode"""
        return self.config.mode == TradingMode.TRADE

    async def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics"""
        if not self.interface:
            raise RuntimeError("Trading client not started")
        
        return await self.interface.get_statistics()


# Convenience functions for quick usage
async def create_trading_client(config: Optional[ModeAwareConfig] = None) -> ModeAwareTradingClient:
    """Create and start a trading client"""
    client = ModeAwareTradingClient(config)
    await client.start()
    return client


async def quick_buy(
    symbol: str,
    quantity: float,
    price: Optional[float] = None,
    config: Optional[ModeAwareConfig] = None
) -> Dict[str, Any]:
    """Quick buy function"""
    client = await create_trading_client(config)
    try:
        return await client.buy(symbol, quantity, price)
    finally:
        await client.stop()


async def quick_sell(
    symbol: str,
    quantity: float,
    price: Optional[float] = None,
    config: Optional[ModeAwareConfig] = None
) -> Dict[str, Any]:
    """Quick sell function"""
    client = await create_trading_client(config)
    try:
        return await client.sell(symbol, quantity, price)
    finally:
        await client.stop()


# Example usage
async def example_usage():
    """Example of how to use the mode-aware trading client"""
    
    # Create client (automatically loads config from environment)
    client = await create_trading_client()
    
    try:
        # Update market data for simulation (paper mode only)
        if client.is_paper_mode():
            await client.update_market_data("BTCUSDT", 50000.0)
        
        # Place a buy order
        buy_result = await client.buy("BTCUSDT", 0.1, 49000.0)  # Limit buy
        print(f"Buy order result: {buy_result}")
        
        # Place a market sell order
        sell_result = await client.sell("BTCUSDT", 0.05)  # Market sell
        print(f"Sell order result: {sell_result}")
        
        # Get positions
        positions = await client.get_positions("BTCUSDT")
        print(f"Positions: {positions}")
        
        # Get balance
        balance = await client.get_balance()
        print(f"Balance: {balance}")
        
        # Get statistics
        stats = await client.get_statistics()
        print(f"Statistics: {stats}")
        
    finally:
        await client.stop()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())