"""
Live Trading System Example

This example demonstrates how to use the live trading system with:
1. Exchange-agnostic trading receiver
2. Live trading engine with order management
3. Risk management and data streaming
"""

import asyncio
import logging
from datetime import datetime

# Import trading system components
from live_trading import TradingEngine, TradingConfig, TradingMode
from exchanges import TradingReceiver
from exchanges.factory import ExchangeFactory
from src.interfaces.base_interfaces import TradeDecision

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function demonstrating live trading system usage."""
    
    print("🚀 Live Trading System Example")
    print("=" * 50)
    
    # 1. Configure trading parameters
    config = TradingConfig(
        mode=TradingMode.PAPER,  # Start with paper trading
        exchange_name="binance",
        symbols=["BTCUSDT", "ETHUSDT"],
        max_position_size=1000.0,
        max_daily_loss=100.0,
        max_leverage=5.0,
        data_update_interval=2.0,
        trade_log_enabled=True,
        metrics_enabled=True
    )
    
    print(f"📊 Configuration: {config.mode.value} trading on {config.exchange_name}")
    print(f"📈 Symbols: {config.symbols}")
    print(f"💰 Max position size: ${config.max_position_size}")
    print()
    
    # 2. Create exchange client
    exchange_client = ExchangeFactory.get_exchange(config.exchange_name)
    print(f"🔌 Exchange client created: {config.exchange_name}")
    
    # 3. Create trading engine
    trading_engine = TradingEngine(config, exchange_client)
    print("⚙️ Trading engine created")
    
    # 4. Create exchange-agnostic receiver
    receiver_config = {
        "exchanges": {
            "binance": {
                "api_key": "your_api_key_here",
                "api_secret": "your_api_secret_here"
            }
        }
    }
    trading_receiver = TradingReceiver(receiver_config)
    print("📡 Exchange-agnostic receiver created")
    
    try:
        # 5. Start all components
        print("\n🔄 Starting trading system...")
        await trading_engine.start()
        await trading_receiver.start()
        print("✅ Trading system started successfully")
        
        # 6. Demonstrate order execution through receiver
        print("\n📝 Demonstrating order execution...")
        
        # Create a sample trade decision
        trade_decision = TradeDecision(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            action="buy",
            quantity=0.001,
            price=0.0,  # Market order
            leverage=1.0,
            stop_loss=45000.0,
            take_profit=55000.0,
            confidence=0.8,
            risk_score=0.3
        )
        
        # Execute trade through trading engine
        order = await trading_engine.execute_trade_decision(trade_decision)
        if order:
            print(f"✅ Order executed: {order.id} - {order.symbol} {order.side.value} {order.quantity}")
        else:
            print("❌ Order execution failed or was rejected")
        
        # 7. Demonstrate data streaming
        print("\n📊 Demonstrating data streaming...")
        
        # Register data handlers
        async def on_ticker_data(data):
            print(f"📈 Ticker: {data['symbol']} @ ${data['data']['last_price']}")
        
        async def on_trade_data(data):
            print(f"💱 Trade: {data['symbol']} - {data['data']['quantity']} @ ${data['data']['price']}")
        
        # Register handlers
        trading_engine.data_streamer.register_handler("ticker", on_ticker_data)
        trading_engine.data_streamer.register_handler("trade", on_trade_data)
        
        # Let it run for a few seconds to demonstrate streaming
        print("🔄 Streaming data for 10 seconds...")
        await asyncio.sleep(10)
        
        # 8. Demonstrate receiver functionality
        print("\n🔄 Demonstrating exchange-agnostic receiver...")
        
        # Send order through receiver
        response = await trading_receiver.send_order(
            exchange="binance",
            symbol="ETHUSDT",
            side="buy",
            order_type="market",
            quantity=0.01
        )
        
        if response["success"]:
            print(f"✅ Receiver order: {response['order_id']} - {response['status']}")
        else:
            print(f"❌ Receiver order failed: {response['error']}")
        
        # Request data through receiver
        data_response = await trading_receiver.request_data(
            exchange="binance",
            symbol="BTCUSDT",
            data_type="ticker"
        )
        
        if data_response["success"]:
            print(f"📊 Receiver data: {data_response['data']}")
        else:
            print(f"❌ Receiver data failed: {data_response['error']}")
        
        # 9. Get system status and performance
        print("\n📊 System Status and Performance:")
        print("-" * 30)
        
        # Trading engine status
        status = await trading_engine.get_trading_status()
        print(f"🔄 Engine running: {status['running']}")
        print(f"📈 Trading active: {status['trading_active']}")
        print(f"📊 Total trades: {status['total_trades']}")
        
        # Position summary
        positions = await trading_engine.get_position_summary()
        for symbol, position in positions.items():
            print(f"💰 {symbol}: Position={position['current_position']}, PnL=${position['daily_pnl']:.2f}")
        
        # Performance metrics
        performance = await trading_engine.get_performance_metrics()
        print(f"📈 Win rate: {performance['win_rate']:.2%}")
        print(f"💰 Total PnL: ${performance['total_pnl']:.2f}")
        print(f"📊 Average trade size: {performance['average_trade_size']:.4f}")
        
        # Receiver statistics
        receiver_stats = await trading_receiver.get_statistics()
        print(f"📡 Receiver messages: {receiver_stats['statistics']['total_received']}")
        print(f"✅ Successful: {receiver_stats['statistics']['total_processed']}")
        print(f"❌ Errors: {receiver_stats['statistics']['total_errors']}")
        
        print("\n🎉 Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        logger.exception("Error in live trading example")
    
    finally:
        # 10. Clean shutdown
        print("\n🛑 Shutting down trading system...")
        await trading_engine.stop()
        await trading_receiver.stop()
        print("✅ Trading system stopped")


async def demonstrate_risk_management():
    """Demonstrate risk management features."""
    print("\n🛡️ Risk Management Demonstration")
    print("-" * 40)
    
    config = TradingConfig(
        mode=TradingMode.PAPER,
        exchange_name="binance",
        symbols=["BTCUSDT"],
        max_position_size=100.0,
        max_daily_loss=50.0,
        max_leverage=2.0
    )
    
    exchange_client = ExchangeFactory.get_exchange(config.exchange_name)
    trading_engine = TradingEngine(config, exchange_client)
    
    try:
        await trading_engine.start()
        
        # Create risky trade decision
        risky_decision = TradeDecision(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            action="buy",
            quantity=1000.0,  # Exceeds max position size
            price=0.0,
            leverage=10.0,  # Exceeds max leverage
            stop_loss=0.0,
            take_profit=0.0,
            confidence=0.9,
            risk_score=0.9  # High risk
        )
        
        print("🚨 Attempting risky trade...")
        order = await trading_engine.execute_trade_decision(risky_decision)
        
        if order:
            print(f"⚠️ Risky order executed: {order.id}")
        else:
            print("✅ Risky order rejected by risk management")
        
        # Get risk summary
        risk_summary = await trading_engine.risk_manager.get_risk_summary()
        print(f"📊 Risk violations: {risk_summary['risk_violations_count']}")
        print(f"💰 Total exposure: ${risk_summary['total_exposure']:.2f}")
        
    finally:
        await trading_engine.stop()


async def demonstrate_multi_exchange():
    """Demonstrate multi-exchange trading."""
    print("\n🌐 Multi-Exchange Trading Demonstration")
    print("-" * 45)
    
    receiver_config = {
        "exchanges": {
            "binance": {
                "api_key": "binance_key",
                "api_secret": "binance_secret"
            },
            "okx": {
                "api_key": "okx_key",
                "api_secret": "okx_secret",
                "password": "okx_passphrase"
            }
        }
    }
    
    trading_receiver = TradingReceiver(receiver_config)
    
    try:
        await trading_receiver.start()
        
        # Send orders to different exchanges
        exchanges = ["binance", "okx"]
        
        for exchange in exchanges:
            print(f"📤 Sending order to {exchange}...")
            
            response = await trading_receiver.send_order(
                exchange=exchange,
                symbol="BTCUSDT",
                side="buy",
                order_type="market",
                quantity=0.001
            )
            
            if response["success"]:
                print(f"✅ {exchange}: Order {response['order_id']} submitted")
            else:
                print(f"❌ {exchange}: {response['error']}")
        
        # Get aggregated data from multiple exchanges
        print("\n📊 Getting aggregated data...")
        
        aggregated_response = await trading_receiver.request_data(
            exchange="binance",  # This would use data aggregator internally
            symbol="BTCUSDT",
            data_type="ticker"
        )
        
        if aggregated_response["success"]:
            print(f"📈 Aggregated ticker data: {aggregated_response['data']}")
        
    finally:
        await trading_receiver.stop()


if __name__ == "__main__":
    print("🎯 Live Trading System Examples")
    print("=" * 60)
    
    # Run main example
    asyncio.run(main())
    
    # Run risk management example
    asyncio.run(demonstrate_risk_management())
    
    # Run multi-exchange example
    asyncio.run(demonstrate_multi_exchange())
    
    print("\n🎉 All examples completed!")
    print("\n📚 Key Features Demonstrated:")
    print("  ✅ Exchange-agnostic trading receiver")
    print("  ✅ Live trading engine with order management")
    print("  ✅ Real-time data streaming")
    print("  ✅ Risk management and validation")
    print("  ✅ Multi-exchange support")
    print("  ✅ Performance monitoring and metrics")
    print("  ✅ Paper and live trading modes")
    print("  ✅ Comprehensive error handling")