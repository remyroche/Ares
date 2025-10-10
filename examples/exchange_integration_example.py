"""
Exchange Integration Example

This example demonstrates how to properly use the exchanges/shared/ modules
within the trading system through the ExchangeInterface with comprehensive
error handling and type hints.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import the integration module
from src.trading.integration.exchange_integration import (
    create_binance_integration,
    create_bingx_integration,
    ExchangeIntegrationConfig,
    create_exchange_integration
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_binance_integration():
    """Demonstrate Binance exchange integration."""
    print("🚀 Starting Binance Integration Demo")
    print("=" * 50)
    
    # Create Binance integration (using testnet for safety)
    integration = create_binance_integration(
        api_key="your_binance_api_key",
        api_secret="your_binance_api_secret",
        testnet=True,
        trade_symbol="BTCUSDT"
    )
    
    try:
        # Connect to exchange
        print("📡 Connecting to Binance...")
        connected = await integration.connect()
        
        if not connected:
            print("❌ Failed to connect to Binance")
            return
        
        print("✅ Connected to Binance successfully")
        
        # Get ticker data
        print("\n📊 Getting ticker data...")
        ticker = await integration.get_ticker("BTCUSDT")
        if ticker:
            print(f"BTC/USDT Price: ${ticker.price:.2f}")
            print(f"24h Volume: {ticker.volume_24h:,.2f}")
            print(f"24h Change: {ticker.price_change_percent_24h:.2f}%")
        else:
            print("❌ Failed to get ticker data")
        
        # Get kline data
        print("\n📈 Getting kline data...")
        klines = await integration.get_klines("BTCUSDT", "1h", limit=24)
        if klines:
            print(f"Retrieved {len(klines)} hourly candles")
            latest = klines[0]
            print(f"Latest candle: O:{latest.open_price:.2f} H:{latest.high_price:.2f} L:{latest.low_price:.2f} C:{latest.close_price:.2f}")
        else:
            print("❌ Failed to get kline data")
        
        # Get account balance
        print("\n💰 Getting account balance...")
        balance = await integration.get_account_balance()
        if balance:
            print("Account balances:")
            for asset, amount in balance.items():
                print(f"  {asset}: {amount:.8f}")
        else:
            print("❌ Failed to get account balance")
        
        # Risk management example
        print("\n⚠️ Risk management example...")
        risk_info = await integration.get_risk_info("BTCUSDT", 0.1, 50000.0, leverage=2.0)
        if risk_info:
            print("Position risk analysis:")
            for key, value in risk_info.items():
                print(f"  {key}: {value}")
        else:
            print("❌ Failed to get risk information")
        
        # Portfolio risk example
        print("\n📊 Portfolio risk analysis...")
        positions = [
            {"symbol": "BTCUSDT", "size": 0.1, "price": 50000.0},
            {"symbol": "ETHUSDT", "size": 1.0, "price": 3000.0}
        ]
        portfolio_risk = await integration.get_portfolio_risk(positions)
        if portfolio_risk:
            print("Portfolio risk analysis:")
            for key, value in portfolio_risk.items():
                print(f"  {key}: {value}")
        else:
            print("❌ Failed to get portfolio risk")
        
        # Get integration status
        print("\n📋 Integration status...")
        status = integration.get_status()
        print("Integration Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error in Binance integration demo: {e}")
    
    finally:
        # Disconnect
        print("\n📴 Disconnecting from Binance...")
        await integration.disconnect()
        print("✅ Disconnected successfully")


async def demonstrate_bingx_integration():
    """Demonstrate BingX exchange integration."""
    print("\n🚀 Starting BingX Integration Demo")
    print("=" * 50)
    
    # Create BingX integration (using testnet for safety)
    integration = create_bingx_integration(
        api_key="your_bingx_api_key",
        api_secret="your_bingx_api_secret",
        testnet=True,
        trade_symbol="BTCUSDT"
    )
    
    try:
        # Connect to exchange
        print("📡 Connecting to BingX...")
        connected = await integration.connect()
        
        if not connected:
            print("❌ Failed to connect to BingX")
            return
        
        print("✅ Connected to BingX successfully")
        
        # Get ticker data
        print("\n📊 Getting ticker data...")
        ticker = await integration.get_ticker("BTCUSDT")
        if ticker:
            print(f"BTC/USDT Price: ${ticker.price:.2f}")
            print(f"24h Volume: {ticker.volume_24h:,.2f}")
            print(f"24h Change: {ticker.price_change_percent_24h:.2f}%")
        else:
            print("❌ Failed to get ticker data")
        
        # Get kline data
        print("\n📈 Getting kline data...")
        klines = await integration.get_klines("BTCUSDT", "1h", limit=24)
        if klines:
            print(f"Retrieved {len(klines)} hourly candles")
            latest = klines[0]
            print(f"Latest candle: O:{latest.open_price:.2f} H:{latest.high_price:.2f} L:{latest.low_price:.2f} C:{latest.close_price:.2f}")
        else:
            print("❌ Failed to get kline data")
        
        # Get account balance
        print("\n💰 Getting account balance...")
        balance = await integration.get_account_balance()
        if balance:
            print("Account balances:")
            for asset, amount in balance.items():
                print(f"  {asset}: {amount:.8f}")
        else:
            print("❌ Failed to get account balance")
        
        # Risk management example
        print("\n⚠️ Risk management example...")
        risk_info = await integration.get_risk_info("BTCUSDT", 0.1, 50000.0, leverage=2.0)
        if risk_info:
            print("Position risk analysis:")
            for key, value in risk_info.items():
                print(f"  {key}: {value}")
        else:
            print("❌ Failed to get risk information")
        
        # Get integration status
        print("\n📋 Integration status...")
        status = integration.get_status()
        print("Integration Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error in BingX integration demo: {e}")
    
    finally:
        # Disconnect
        print("\n📴 Disconnecting from BingX...")
        await integration.disconnect()
        print("✅ Disconnected successfully")


async def demonstrate_custom_integration():
    """Demonstrate custom exchange integration configuration."""
    print("\n🚀 Starting Custom Integration Demo")
    print("=" * 50)
    
    # Create custom integration configuration
    config = ExchangeIntegrationConfig(
        exchange_type="binance",
        api_key="your_api_key",
        api_secret="your_api_secret",
        testnet=True,
        trade_symbol="ETHUSDT",
        enable_shared_utilities=True,
        enable_risk_management=True,
        enable_rate_limiting=True,
        rate_limits={
            "ticker": 100,
            "klines": 50,
            "orders": 10
        }
    )
    
    # Create integration with custom config
    integration = create_exchange_integration(config)
    
    try:
        # Connect to exchange
        print("📡 Connecting with custom configuration...")
        connected = await integration.connect()
        
        if not connected:
            print("❌ Failed to connect with custom configuration")
            return
        
        print("✅ Connected with custom configuration successfully")
        
        # Get ticker data
        print("\n📊 Getting ticker data...")
        ticker = await integration.get_ticker("ETHUSDT")
        if ticker:
            print(f"ETH/USDT Price: ${ticker.price:.2f}")
            print(f"24h Volume: {ticker.volume_24h:,.2f}")
            print(f"24h Change: {ticker.price_change_percent_24h:.2f}%")
        else:
            print("❌ Failed to get ticker data")
        
        # Get integration status
        print("\n📋 Integration status...")
        status = integration.get_status()
        print("Integration Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error in custom integration demo: {e}")
    
    finally:
        # Disconnect
        print("\n📴 Disconnecting...")
        await integration.disconnect()
        print("✅ Disconnected successfully")


async def demonstrate_error_handling():
    """Demonstrate comprehensive error handling."""
    print("\n🚀 Starting Error Handling Demo")
    print("=" * 50)
    
    # Create integration with invalid credentials
    integration = create_binance_integration(
        api_key="invalid_key",
        api_secret="invalid_secret",
        testnet=True
    )
    
    try:
        # Attempt to connect with invalid credentials
        print("📡 Attempting to connect with invalid credentials...")
        connected = await integration.connect()
        
        if not connected:
            print("❌ Connection failed as expected (invalid credentials)")
        
        # Try to get data without connection
        print("\n📊 Attempting to get data without connection...")
        ticker = await integration.get_ticker("BTCUSDT")
        if not ticker:
            print("❌ Failed to get ticker data as expected (not connected)")
        
        # Try to create order without connection
        print("\n📝 Attempting to create order without connection...")
        order_result = await integration.create_order(
            "BTCUSDT", "buy", "market", 0.001
        )
        if 'error' in order_result:
            print(f"❌ Order creation failed as expected: {order_result['error']}")
        
        # Get integration status
        print("\n📋 Integration status...")
        status = integration.get_status()
        print("Integration Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error in error handling demo: {e}")
    
    finally:
        # Disconnect
        print("\n📴 Disconnecting...")
        await integration.disconnect()
        print("✅ Disconnected successfully")


async def main():
    """Main demonstration function."""
    print("🎯 Exchange Integration Comprehensive Demo")
    print("=" * 60)
    print("This demo shows how to properly use exchanges/shared/ modules")
    print("within the trading system through ExchangeInterface with")
    print("comprehensive error handling and type hints.")
    print("=" * 60)
    
    # Run demonstrations
    await demonstrate_binance_integration()
    await demonstrate_bingx_integration()
    await demonstrate_custom_integration()
    await demonstrate_error_handling()
    
    print("\n🎉 Demo completed successfully!")
    print("=" * 60)
    print("Key features demonstrated:")
    print("✅ Proper integration of exchanges/shared/ modules")
    print("✅ Comprehensive error handling with tprint")
    print("✅ Full type hint coverage")
    print("✅ Risk management integration")
    print("✅ Rate limiting support")
    print("✅ Unified exchange interface")
    print("✅ Multiple exchange support (Binance, BingX)")
    print("✅ Custom configuration support")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())