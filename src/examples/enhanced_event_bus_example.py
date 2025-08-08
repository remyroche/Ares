# src/examples/enhanced_event_bus_example.py

"""
Example usage of the Enhanced Event Bus with event sourcing, versioning, and persistence.

This example demonstrates:
1. Basic event publishing and subscribing
2. Event persistence and replay
3. Event versioning and migration
4. Snapshot creation and state reconstruction
5. Audit trail and correlation tracking
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from src.config.system import get_event_bus_config
from src.interfaces.enhanced_event_bus import (
    EnhancedEventBus,
    Event,
    EventType,
    setup_enhanced_event_bus,
)
from src.utils.logger import setup_logging, system_logger

# Configure logging
setup_logging()
logger = system_logger.getChild("EventBusExample")


class TradingBot:
    """Example trading bot that uses the enhanced event bus"""
    
    def __init__(self, event_bus: EnhancedEventBus):
        self.event_bus = event_bus
        self.portfolio_balance = 10000.0
        self.positions = {}
        self.trade_count = 0
        
        # Subscribe to relevant events
        self._setup_event_subscriptions()
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions"""
        self.event_bus.subscribe(EventType.MARKET_DATA_RECEIVED, self.handle_market_data)
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self.handle_trade_executed)
        self.event_bus.subscribe(EventType.RISK_ALERT, self.handle_risk_alert)
    
    async def handle_market_data(self, event: Event):
        """Handle market data events"""
        data = event.data
        symbol = data.get("symbol")
        price = data.get("price")
        
        logger.info(f"📊 Received market data for {symbol}: ${price}")
        
        # Simple trading logic: buy if price drops below a threshold
        if price < 50000 and symbol == "BTCUSDT":
            await self.place_trade(symbol, "buy", 0.1, price, event.metadata.correlation_id)
    
    async def handle_trade_executed(self, event: Event):
        """Handle trade execution events"""
        data = event.data
        symbol = data.get("symbol")
        side = data.get("side")
        quantity = data.get("quantity")
        price = data.get("price")
        
        logger.info(f"✅ Trade executed: {side} {quantity} {symbol} @ ${price}")
        
        # Update portfolio
        if side == "buy":
            self.portfolio_balance -= quantity * price
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        else:
            self.portfolio_balance += quantity * price
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
        
        self.trade_count += 1
        
        # Publish performance update
        await self.event_bus.publish(
            EventType.PERFORMANCE_UPDATE,
            {
                "portfolio_balance": self.portfolio_balance,
                "positions": self.positions.copy(),
                "trade_count": self.trade_count
            },
            source="TradingBot",
            correlation_id=event.metadata.correlation_id
        )
    
    async def handle_risk_alert(self, event: Event):
        """Handle risk alert events"""
        data = event.data
        alert_type = data.get("type")
        message = data.get("message")
        
        logger.warning(f"⚠️ Risk Alert [{alert_type}]: {message}")
        
        # Take protective action if needed
        if alert_type == "portfolio_loss":
            await self.close_all_positions()
    
    async def place_trade(self, symbol: str, side: str, quantity: float, price: float, correlation_id: str = None):
        """Place a trade order"""
        trade_data = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_id": f"order_{self.trade_count + 1}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Publish trade decision event
        await self.event_bus.publish(
            EventType.TRADE_DECISION_MADE,
            trade_data,
            source="TradingBot",
            correlation_id=correlation_id,
            aggregate_id="trader_bot_1"
        )
        
        # Simulate trade execution (in real system, this would go to exchange)
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Publish trade executed event
        await self.event_bus.publish(
            EventType.TRADE_EXECUTED,
            trade_data,
            source="ExchangeAPI",
            correlation_id=correlation_id,
            aggregate_id="trader_bot_1"
        )
    
    async def close_all_positions(self):
        """Close all open positions"""
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                # Get current price (simplified)
                current_price = 49000  # Mock current price
                await self.place_trade(symbol, "sell", quantity, current_price)


class MarketDataProvider:
    """Example market data provider"""
    
    def __init__(self, event_bus: EnhancedEventBus):
        self.event_bus = event_bus
        self.is_running = False
    
    async def start_streaming(self):
        """Start streaming market data"""
        self.is_running = True
        logger.info("📡 Starting market data stream...")
        
        # Simulate market data stream
        base_price = 50000
        while self.is_running:
            # Simulate price movement
            import random
            price_change = random.uniform(-1000, 1000)
            current_price = base_price + price_change
            
            # Publish market data event
            await self.event_bus.publish(
                EventType.MARKET_DATA_RECEIVED,
                {
                    "symbol": "BTCUSDT",
                    "price": current_price,
                    "volume": random.uniform(1, 10),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "bid": current_price - 1,
                    "ask": current_price + 1
                },
                source="MarketDataProvider",
                aggregate_id="market_data_btcusdt"
            )
            
            await asyncio.sleep(2)  # Send data every 2 seconds
    
    def stop_streaming(self):
        """Stop streaming market data"""
        self.is_running = False
        logger.info("🛑 Stopped market data stream")


class RiskManager:
    """Example risk manager"""
    
    def __init__(self, event_bus: EnhancedEventBus):
        self.event_bus = event_bus
        self.initial_balance = 10000.0
        
        # Subscribe to performance updates
        self.event_bus.subscribe(EventType.PERFORMANCE_UPDATE, self.monitor_performance)
    
    async def monitor_performance(self, event: Event):
        """Monitor portfolio performance and generate risk alerts"""
        data = event.data
        current_balance = data.get("portfolio_balance", 0)
        
        # Calculate loss percentage
        loss_percentage = ((self.initial_balance - current_balance) / self.initial_balance) * 100
        
        if loss_percentage > 10:  # More than 10% loss
            await self.event_bus.publish(
                EventType.RISK_ALERT,
                {
                    "type": "portfolio_loss",
                    "message": f"Portfolio loss of {loss_percentage:.2f}% detected",
                    "current_balance": current_balance,
                    "initial_balance": self.initial_balance,
                    "loss_amount": self.initial_balance - current_balance
                },
                source="RiskManager",
                correlation_id=event.metadata.correlation_id
            )


async def demonstrate_basic_usage():
    """Demonstrate basic event bus usage"""
    logger.info("🚀 Starting Enhanced Event Bus Example")
    
    # Setup event bus with configuration
    config = get_event_bus_config()
    event_bus = await setup_enhanced_event_bus({"event_bus": config})
    
    if not event_bus:
        logger.error("Failed to setup event bus")
        return
    
    # Start the event bus
    event_bus_task = asyncio.create_task(event_bus.run())
    
    try:
        # Create components
        trading_bot = TradingBot(event_bus)
        market_data_provider = MarketDataProvider(event_bus)
        risk_manager = RiskManager(event_bus)
        
        # Start market data streaming
        data_task = asyncio.create_task(market_data_provider.start_streaming())
        
        # Let the system run for a while
        await asyncio.sleep(20)
        
        # Stop market data
        market_data_provider.stop_streaming()
        await data_task
        
        # Show some statistics
        status = event_bus.get_status()
        logger.info(f"📈 Event Bus Status: {status}")
        
        metrics = event_bus.get_metrics()
        logger.info(f"📊 Event Bus Metrics: {metrics}")
        
    finally:
        # Stop event bus
        await event_bus.stop()
        event_bus_task.cancel()
        
        try:
            await event_bus_task
        except asyncio.CancelledError:
            pass


async def demonstrate_event_replay():
    """Demonstrate event replay capabilities"""
    logger.info("🔄 Demonstrating Event Replay")
    
    # Setup event bus
    event_bus = await setup_enhanced_event_bus()
    if not event_bus:
        return
    
    # Start event bus
    event_bus_task = asyncio.create_task(event_bus.run())
    
    try:
        # Publish some historical events
        for i in range(5):
            await event_bus.publish(
                EventType.TRADE_EXECUTED,
                {
                    "symbol": "BTCUSDT",
                    "side": "buy",
                    "quantity": 0.1 * (i + 1),
                    "price": 50000 + i * 100,
                    "order_id": f"historical_order_{i}"
                },
                source="HistoricalData",
                aggregate_id="trader_bot_1"
            )
        
        # Process events
        await asyncio.sleep(2)
        
        # Replay events for specific aggregate
        logger.info("📼 Replaying events for trader_bot_1...")
        replayed_events = await event_bus.replay_events(
            aggregate_id="trader_bot_1",
            from_sequence=0
        )
        
        logger.info(f"🎬 Replayed {len(replayed_events)} events:")
        for event in replayed_events:
            logger.info(f"  - {event.event_type.value}: {event.data}")
        
        # Rebuild state from events
        logger.info("🏗️ Rebuilding state from events...")
        state = await event_bus.rebuild_from_events("trader_bot_1")
        logger.info(f"📊 Rebuilt state: {state}")
        
    finally:
        await event_bus.stop()
        event_bus_task.cancel()
        
        try:
            await event_bus_task
        except asyncio.CancelledError:
            pass


async def demonstrate_event_versioning():
    """Demonstrate event versioning and migration"""
    logger.info("🔄 Demonstrating Event Versioning")
    
    # Setup event bus
    event_bus = await setup_enhanced_event_bus()
    if not event_bus:
        return
    
    # Test event migration
    from src.interfaces.enhanced_event_bus import EventMetadata
    
    # Create an old version event
    old_metadata = EventMetadata(schema_version="1.0.0")
    old_event = Event(
        event_type=EventType.MARKET_DATA_RECEIVED,
        data={"symbol": "BTCUSDT", "price": 50000, "volume": 100},
        metadata=old_metadata
    )
    
    logger.info(f"📜 Original event (v1.0.0): {old_event.data}")
    
    # Migrate to new version
    migrated_event = event_bus.version_manager.migrate_event(old_event, "1.1.0")
    logger.info(f"🔄 Migrated event (v1.1.0): {migrated_event.data}")
    
    # Validate schemas
    is_valid_old = event_bus.version_manager.validate_event_schema(old_event)
    is_valid_new = event_bus.version_manager.validate_event_schema(migrated_event)
    
    logger.info(f"✅ Old version valid: {is_valid_old}")
    logger.info(f"✅ New version valid: {is_valid_new}")


async def main():
    """Main example function"""
    try:
        # Run different demonstrations
        await demonstrate_basic_usage()
        await asyncio.sleep(1)
        
        await demonstrate_event_replay()
        await asyncio.sleep(1)
        
        await demonstrate_event_versioning()
        
        logger.info("🎉 Enhanced Event Bus examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())