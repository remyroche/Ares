"""
Mode-Aware Trading Example

This example demonstrates how to use the mode-aware exchange interface
for both paper trading and live trading modes.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any

# Import the mode-aware components
from exchanges.mode_aware_exchange_interface import ModeAwareExchangeInterface, ModeAwareConfig, TradingMode
from exchanges.trading_mode_config import get_trading_mode_config, is_paper_trading
from exchanges.exchange_registry import ExchangeRegistry
from exchanges.order_router import OrderRouter


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_exchange_interface() -> ModeAwareExchangeInterface:
    """Setup the mode-aware exchange interface"""
    
    # Get configuration
    config = get_trading_mode_config()
    
    # Initialize components based on mode
    exchange_registry = None
    order_router = None
    
    if config.mode == TradingMode.TRADE:
        # Setup real exchange components for live trading
        exchange_registry = ExchangeRegistry()
        order_router = OrderRouter(exchange_registry)
        
        # Register exchanges (example)
        # await exchange_registry.register_exchange("binance", binance_exchange)
        # await exchange_registry.register_exchange("okx", okx_exchange)
    
    # Create mode-aware interface
    interface = ModeAwareExchangeInterface(
        config=config,
        exchange_registry=exchange_registry,
        order_router=order_router
    )
    
    # Initialize
    await interface.initialize()
    
    return interface


async def paper_trading_example():
    """Example of paper trading mode"""
    logger.info("=== PAPER TRADING EXAMPLE ===")
    
    # Setup interface
    interface = await setup_exchange_interface()
    
    try:
        # Update market data for simulation
        market_data = {
            "price": 50000.0,
            "bid": 49999.0,
            "ask": 50001.0,
            "volume": 1000.0
        }
        await interface.update_market_data("BTCUSDT", market_data)
        
        # Create a market buy order
        logger.info("Creating market buy order...")
        order_result = await interface.create_order(
            exchange="simulator",  # Ignored in paper mode
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.1
        )
        logger.info(f"Order result: {order_result}")
        
        # Create a limit sell order
        logger.info("Creating limit sell order...")
        limit_order_result = await interface.create_order(
            exchange="simulator",
            symbol="BTCUSDT", 
            side="SELL",
            order_type="LIMIT",
            quantity=0.05,
            price=51000.0
        )
        logger.info(f"Limit order result: {limit_order_result}")
        
        # Get open orders
        open_orders = await interface.get_open_orders(symbol="BTCUSDT")
        logger.info(f"Open orders: {open_orders}")
        
        # Get positions
        positions = await interface.get_positions(symbol="BTCUSDT")
        logger.info(f"Positions: {positions}")
        
        # Get account info
        account_info = await interface.get_account_info()
        logger.info(f"Account info: {account_info}")
        
        # Get statistics
        stats = await interface.get_statistics()
        logger.info(f"Statistics: {stats}")
        
    finally:
        await interface.close()


async def live_trading_example():
    """Example of live trading mode"""
    logger.info("=== LIVE TRADING EXAMPLE ===")
    
    # Setup interface
    interface = await setup_exchange_interface()
    
    try:
        # Create a market buy order on Binance
        logger.info("Creating market buy order on Binance...")
        order_result = await interface.create_order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.001  # Small amount for safety
        )
        logger.info(f"Order result: {order_result}")
        
        # Get order status
        if order_result.get("success") and order_result.get("order_id"):
            order_status = await interface.get_order_status(
                order_id=order_result["order_id"],
                exchange="binance"
            )
            logger.info(f"Order status: {order_status}")
        
        # Get open orders
        open_orders = await interface.get_open_orders(exchange="binance")
        logger.info(f"Open orders: {open_orders}")
        
        # Get positions
        positions = await interface.get_positions(exchange="binance")
        logger.info(f"Positions: {positions}")
        
        # Get account info
        account_info = await interface.get_account_info(exchange="binance")
        logger.info(f"Account info: {account_info}")
        
    finally:
        await interface.close()


async def mode_switching_example():
    """Example of switching between modes"""
    logger.info("=== MODE SWITCHING EXAMPLE ===")
    
    # Start in paper mode
    config = ModeAwareConfig(mode=TradingMode.PAPER)
    interface = ModeAwareExchangeInterface(config=config)
    await interface.initialize()
    
    logger.info(f"Current mode: {interface.get_mode().value}")
    
    # Create a paper trade
    order_result = await interface.create_order(
        exchange="simulator",
        symbol="ETHUSDT",
        side="BUY",
        order_type="MARKET",
        quantity=1.0
    )
    logger.info(f"Paper trade result: {order_result}")
    
    # Switch to trade mode (requires reinitialization)
    interface.set_mode(TradingMode.TRADE)
    logger.info(f"Switched to mode: {interface.get_mode().value}")
    
    # Note: In a real application, you would reinitialize the interface
    # with proper exchange registry and order router for live trading
    
    await interface.close()


async def order_book_simulation_example():
    """Example of order book simulation for accurate fills"""
    logger.info("=== ORDER BOOK SIMULATION EXAMPLE ===")
    
    # Setup paper trading interface
    config = ModeAwareConfig(mode=TradingMode.PAPER, enable_order_book_simulation=True)
    interface = ModeAwareExchangeInterface(config=config)
    await interface.initialize()
    
    try:
        # Simulate order book data
        order_book = {
            "bids": [
                [49950.0, 0.5],
                [49900.0, 1.0],
                [49850.0, 2.0]
            ],
            "asks": [
                [50050.0, 0.5],
                [50100.0, 1.0],
                [50150.0, 2.0]
            ]
        }
        
        # Process order book for accurate simulation
        await interface.process_order_book("BTCUSDT", order_book)
        
        # Create a limit order that should fill
        logger.info("Creating limit buy order at 49950...")
        order_result = await interface.create_order(
            exchange="simulator",
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.3,
            price=49950.0
        )
        logger.info(f"Order result: {order_result}")
        
        # Wait a moment for order processing
        await asyncio.sleep(1)
        
        # Check order status
        if order_result.get("success") and order_result.get("order_id"):
            order_status = await interface.get_order_status(order_result["order_id"])
            logger.info(f"Order status: {order_status}")
        
    finally:
        await interface.close()


async def main():
    """Main example function"""
    logger.info("Mode-Aware Trading Examples")
    logger.info("=" * 50)
    
    # Check current mode
    if is_paper_trading():
        logger.info("Running in PAPER trading mode")
        await paper_trading_example()
    else:
        logger.info("Running in LIVE trading mode")
        await live_trading_example()
    
    # Additional examples
    await mode_switching_example()
    await order_book_simulation_example()
    
    logger.info("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())