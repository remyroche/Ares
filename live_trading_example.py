#!/usr/bin/env python3
"""
Live Trading System Example

This example demonstrates how to use the complete live trading system
including the trading manager, order receiver, and exchange integration.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from src.interfaces.base_interfaces import TradeDecision
from live_trading.trading_manager import TradingManager, TradingConfig
from exchange.order_receiver import ExchangeOrderReceiver
from live_trading.event_system import EventBus, TradingEventPublisher, EventHandlers
from live_trading.data_receiver import DataReceiver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingBot:
    """Example trading bot using the live trading system"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.trading_manager = None
        self.event_bus = None
        self.event_publisher = None

    async def initialize(self):
        """Initialize the trading bot."""
        logger.info("Initializing Trading Bot...")

        # Initialize event bus
        self.event_bus = EventBus()
        await self.event_bus.start()

        # Initialize event publisher
        self.event_publisher = TradingEventPublisher(self.event_bus)

        # Set up event handlers
        self._setup_event_handlers()

        # Initialize trading manager
        self.trading_manager = TradingManager(self.config)

        # Set up callbacks
        self.trading_manager.set_order_update_callback(self.on_order_update)
        self.trading_manager.set_data_update_callback(self.on_data_update)
        self.trading_manager.set_position_update_callback(self.on_position_update)

        # Initialize trading manager
        success = await self.trading_manager.initialize()
        if not success:
            raise Exception("Failed to initialize trading manager")

        # Publish system startup event
        await self.event_publisher.publish_system_startup({
            'exchange': self.config.exchange_name,
            'symbols': self.config.symbols
        })

        logger.info("✅ Trading Bot initialized successfully")

    async def start(self):
        """Start the trading bot."""
        logger.info("Starting Trading Bot...")

        # Start trading manager
        success = await self.trading_manager.start()
        if not success:
            raise Exception("Failed to start trading manager")

        # Start trading loop
        await self.trading_loop()

    async def stop(self):
        """Stop the trading bot."""
        logger.info("Stopping Trading Bot...")

        if self.trading_manager:
            await self.trading_manager.stop()

        if self.event_bus:
            await self.event_bus.stop()

        logger.info("✅ Trading Bot stopped")

    async def trading_loop(self):
        """Main trading loop."""
        logger.info("Starting trading loop...")

        while True:
            try:
                # Get market data
                for symbol in self.config.symbols:
                    try:
                        market_data = await self.trading_manager.get_market_data(symbol, "1m", 10)
                        if market_data:
                            logger.info(f"📊 {symbol}: Last price = {market_data[-1].close}")

                            # Simple trading logic (example)
                            await self._make_trading_decision(symbol, market_data)

                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")

                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)

    async def _make_trading_decision(self, symbol: str, market_data: list):
        """Make a simple trading decision based on market data."""
        try:
            if len(market_data) < 5:
                return

            # Simple moving average strategy
            recent_prices = [data.close for data in market_data[-5:]]
            sma = sum(recent_prices) / len(recent_prices)
            current_price = market_data[-1].close

            # Buy signal if price above SMA
            if current_price > sma * 1.001:  # 0.1% above SMA
                await self._place_buy_order(symbol, current_price)

            # Sell signal if price below SMA
            elif current_price < sma * 0.999:  # 0.1% below SMA
                await self._place_sell_order(symbol, current_price)

        except Exception as e:
            logger.error(f"Error making trading decision for {symbol}: {e}")

    async def demonstrate_position_management(self):
        """Demonstrate position management features."""
        try:
            symbol = "BTCUSDT"
            logger.info(f"📊 Demonstrating position management for {symbol}")

            # Get asset data (formatted as klines)
            asset_data = await self.trading_manager.get_asset_data(symbol, "1m", 10)
            if asset_data:
                logger.info(f"✅ Retrieved {len(asset_data)} data points for {symbol}")
                logger.info(f"Latest price: {asset_data[-1].close}")

            # Open a futures position with leverage
            logger.info("🔓 Opening perpetual futures position with leverage...")
            position_result = await self.trading_manager.open_position(
                symbol=symbol,
                side="BUY",
                quantity=0.001,
                leverage=5.0,  # 5x leverage
                order_type="MARKET"
            )

            if position_result and position_result.get("success"):
                trade_id = position_result.get("trade_id")
                logger.info(f"✅ Position opened with trade ID: {trade_id}")

                # Get trade information
                trade_info = await self.trading_manager.get_trade_info(symbol, trade_id)
                if trade_info:
                    logger.info(f"📋 Trade info: {trade_info}")

                # Wait a bit and then close the position
                await asyncio.sleep(5)  # Wait 5 seconds

                logger.info("🔒 Closing position...")
                close_result = await self.trading_manager.close_position(symbol, trade_id)

                if close_result and close_result.get("success"):
                    logger.info(f"✅ Position closed successfully. PnL: {close_result.get('pnl', 0)}")
                else:
                    logger.error("❌ Failed to close position")

            else:
                logger.error("❌ Failed to open position")

        except Exception as e:
            logger.error(f"Error in position management demo: {e}")

    async def _place_buy_order(self, symbol: str, price: float):
        """Place a buy order."""
        try:
            quantity = 0.001  # Small test amount

            trade_decision = TradeDecision(
                timestamp=datetime.now(),
                symbol=symbol,
                action="BUY",
                quantity=quantity,
                price=price,
                leverage=1.0,
                stop_loss=price * 0.98,  # 2% stop loss
                take_profit=price * 1.05,  # 5% take profit
                confidence=0.7,
                risk_score=0.1
            )

            result = await self.trading_manager.place_order(trade_decision)

            if result:
                logger.info(f"✅ Buy order placed: {symbol} @ {price}")
            else:
                logger.error(f"❌ Failed to place buy order: {symbol}")

        except Exception as e:
            logger.error(f"Error placing buy order for {symbol}: {e}")

    async def _place_sell_order(self, symbol: str, price: float):
        """Place a sell order."""
        try:
            quantity = 0.001  # Small test amount

            trade_decision = TradeDecision(
                timestamp=datetime.now(),
                symbol=symbol,
                action="SELL",
                quantity=quantity,
                price=price,
                leverage=1.0,
                stop_loss=price * 1.02,  # 2% stop loss
                take_profit=price * 0.95,  # 5% take profit
                confidence=0.7,
                risk_score=0.1
            )

            result = await self.trading_manager.place_order(trade_decision)

            if result:
                logger.info(f"✅ Sell order placed: {symbol} @ {price}")
            else:
                logger.error(f"❌ Failed to place sell order: {symbol}")

        except Exception as e:
            logger.error(f"Error placing sell order for {symbol}: {e}")

    def _setup_event_handlers(self):
        """Set up event handlers."""
        # Order events
        self.event_bus.subscribe(
            "order_created",
            EventHandlers.log_event_handler,
            priority=1
        )

        self.event_bus.subscribe(
            "order_filled",
            self._on_order_filled,
            priority=2
        )

        # Market data events
        self.event_bus.subscribe(
            "market_data_update",
            self._on_market_data_update,
            priority=1
        )

        # Risk events
        self.event_bus.subscribe(
            "risk_limit_exceeded",
            self._on_risk_warning,
            priority=3
        )

    def _on_order_filled(self, event):
        """Handle order filled event."""
        logger.info(f"🎉 Order filled: {event.data}")

    def _on_market_data_update(self, event):
        """Handle market data update event."""
        logger.debug(f"📈 Market data update: {event.data}")

    def _on_risk_warning(self, event):
        """Handle risk warning event."""
        logger.warning(f"⚠️ Risk warning: {event.data}")

    # Event callback methods
    async def on_order_update(self, order_update: Dict[str, Any]):
        """Handle order updates."""
        await self.event_publisher.publish_order_filled(order_update)

    async def on_data_update(self, market_data):
        """Handle data updates."""
        await self.event_publisher.publish_market_data_update({
            'symbol': market_data.symbol,
            'price': market_data.close,
            'timestamp': market_data.timestamp.isoformat()
        })

    async def on_position_update(self, position_update: Dict[str, Any]):
        """Handle position updates."""
        await self.event_publisher.publish_position_update(position_update)


async def main():
    """Main function to run the trading bot."""
    try:
        # Configuration - Replace with your actual API keys
        config = TradingConfig(
            exchange_name="binance",  # or "okx", "gateio", "mexc"
            symbols=["BTCUSDT", "ETHUSDT"],
            max_position_size=10000.0,
            max_daily_trades=20,
            risk_per_trade=0.02,
            enable_data_streaming=True,
            enable_order_execution=True,
            api_key="YOUR_API_KEY",  # Replace with actual API key
            api_secret="YOUR_API_SECRET"  # Replace with actual API secret
        )

        # Create and run trading bot
        bot = TradingBot(config)

        await bot.initialize()

        # Start trading
        await bot.start()

        # Demonstrate position management features
        await bot.demonstrate_position_management()

    except KeyboardInterrupt:
        logger.info("Shutting down trading bot...")
    except Exception as e:
        logger.error(f"Error running trading bot: {e}")
    finally:
        if 'bot' in locals():
            await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())