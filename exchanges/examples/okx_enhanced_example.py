"""
Enhanced OKX Exchange Usage Example

This example demonstrates how to use the enhanced OKX exchange implementation
with all the shared utilities and comprehensive functionality.
"""

import asyncio
import os
from datetime import datetime, timedelta

# Import the enhanced OKX exchange
from ..okx_enhanced import create_enhanced_okx_exchange


async def main():
    """Main example function."""
    print("🚀 Enhanced OKX Exchange Example")
    print("=" * 50)
    
    # Initialize the enhanced exchange
    exchange = create_enhanced_okx_exchange(
        api_key=os.getenv("OKX_API_KEY", ""),
        api_secret=os.getenv("OKX_API_SECRET", ""),
        password=os.getenv("OKX_PASSPHRASE", ""),
        trade_symbol="BTCUSDT",
        use_testnet=True  # Use testnet for safety
    )
    
    try:
        # Initialize the exchange
        print("📡 Initializing exchange...")
        await exchange._initialize_exchange()
        print("✅ Exchange initialized successfully!")
        
        # 1. Market Data Operations
        print("\n📊 Market Data Operations")
        print("-" * 30)
        
        # Get current price
        price = await exchange.get_price("BTCUSDT")
        print(f"Current BTC/USDT price: ${price:,.2f}" if price else "Price not available")
        
        # Get OHLCV data
        print("\n📈 Getting OHLCV data...")
        ohlcv_data = await exchange.get_ohlcv("BTCUSDT", "1h", 24)
        print(f"Retrieved {len(ohlcv_data)} hourly candles")
        
        if ohlcv_data:
            latest = ohlcv_data[-1]
            print(f"Latest candle: O={latest.open:.2f}, H={latest.high:.2f}, L={latest.low:.2f}, C={latest.close:.2f}")
        
        # 2. Account & Balance Operations
        print("\n💰 Account & Balance Operations")
        print("-" * 30)
        
        # Get USDT balance
        usdt_balance = await exchange.get_balance("USDT")
        print(f"USDT Balance: {usdt_balance:.2f}")
        
        # Get BTC balance
        btc_balance = await exchange.get_balance("BTC")
        print(f"BTC Balance: {btc_balance:.8f}")
        
        # 3. Order Management
        print("\n📋 Order Management")
        print("-" * 30)
        
        # Create a small test order (market buy)
        print("Creating test market order...")
        order_result = await exchange.create_order_enhanced(
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            quantity=0.001,  # Very small amount for testing
            client_order_id=f"test_order_{int(datetime.now().timestamp())}"
        )
        
        if order_result:
            print(f"✅ Order created successfully!")
            print(f"Order ID: {order_result['order_id']}")
            print(f"Client Order ID: {order_result['client_order_id']}")
            print(f"Status: {order_result['status']}")
        else:
            print("❌ Order creation failed")
        
        # 4. Position & Risk Management
        print("\n⚠️ Position & Risk Management")
        print("-" * 30)
        
        # Get current positions
        positions = await exchange.get_positions()
        print(f"Current positions: {len(positions)}")
        
        for position in positions:
            symbol = position.get("instId", "Unknown")
            size = position.get("pos", "0")
            unrealized_pnl = position.get("upl", "0")
            print(f"  {symbol}: Size={size}, Unrealized PnL={unrealized_pnl}")
        
        # Get liquidation risk for BTCUSDT
        liquidation_risk = await exchange.get_liquidation_risk("BTCUSDT")
        if liquidation_risk:
            print(f"\nLiquidation Risk for BTCUSDT:")
            print(f"  Margin Ratio: {liquidation_risk['margin_ratio']:.2%}")
            print(f"  Liquidation Price: ${liquidation_risk['liquidation_price']:,.2f}")
            print(f"  Risk Level: {liquidation_risk['risk_level']}")
            print(f"  Unrealized PnL: ${liquidation_risk['unrealized_pnl']:,.2f}")
        
        # 5. Market Metadata & Instruments
        print("\n📋 Market Metadata & Instruments")
        print("-" * 30)
        
        # Get instrument information
        instrument = exchange.market_metadata.get_instrument("BTCUSDT")
        if instrument:
            print(f"BTCUSDT Instrument Info:")
            print(f"  Base Currency: {instrument.base_currency}")
            print(f"  Quote Currency: {instrument.quote_currency}")
            print(f"  Tick Size: {instrument.tick_size}")
            print(f"  Lot Size: {instrument.lot_size}")
            print(f"  Min Notional: {instrument.min_notional}")
            print(f"  Max Leverage: {instrument.max_leverage}")
            print(f"  Price Precision: {instrument.price_precision}")
            print(f"  Quantity Precision: {instrument.quantity_precision}")
        
        # 6. Precision & Risk Tier Information
        print("\n🎯 Precision & Risk Information")
        print("-" * 30)
        
        # Get precision configuration
        precision_config = exchange.precision_helper.get_precision_config("BTCUSDT")
        if precision_config:
            print(f"Precision Config for BTCUSDT:")
            print(f"  Price Precision: {precision_config.price_precision}")
            print(f"  Quantity Precision: {precision_config.quantity_precision}")
            print(f"  Tick Size: {precision_config.tick_size}")
            print(f"  Lot Size: {precision_config.lot_size}")
        
        # Get risk tier information
        risk_profile = exchange.risk_tier_manager.get_symbol_risk_profile("BTCUSDT")
        if risk_profile:
            print(f"\nRisk Profile for BTCUSDT:")
            print(f"  Risk Tier: {risk_profile.risk_tier.value}")
            print(f"  Max Leverage: {risk_profile.max_leverage}")
            print(f"  Max Position Size: {risk_profile.max_position_size:,.2f}")
            print(f"  Margin Ratio: {risk_profile.margin_ratio:.2%}")
            print(f"  Liquidation Ratio: {risk_profile.liquidation_ratio:.2%}")
        
        # 7. Authentication & Time Sync Status
        print("\n🔐 Authentication & Time Sync Status")
        print("-" * 30)
        
        auth_status = exchange.auth_manager.get_authentication_status()
        print(f"Authentication Status:")
        print(f"  Is Authenticated: {auth_status['is_authenticated']}")
        print(f"  Time Synced: {auth_status['time_synced']}")
        print(f"  Clock Skew: {auth_status['clock_skew']}ms")
        print(f"  Permissions: {auth_status['permissions']}")
        
        # 8. Rate Limiting Status
        print("\n⏱️ Rate Limiting Status")
        print("-" * 30)
        
        rate_limit_stats = exchange.rate_limit_manager.get_rate_limit_statistics()
        print(f"Rate Limiting Statistics:")
        print(f"  Total Requests: {rate_limit_stats['total_requests']}")
        print(f"  Requests Last Minute: {rate_limit_stats['requests_last_minute']}")
        print(f"  Configured Rate Limits: {rate_limit_stats['configured_rate_limits']}")
        
        # 9. Order Statistics
        print("\n📊 Order Statistics")
        print("-" * 30)
        
        order_stats = exchange.order_manager.get_order_statistics()
        print(f"Order Statistics:")
        print(f"  Total Orders: {order_stats['total_orders']}")
        print(f"  Open Orders: {order_stats['open_orders']}")
        print(f"  Filled Orders: {order_stats['filled_orders']}")
        print(f"  Cancelled Orders: {order_stats['cancelled_orders']}")
        
        # 10. Balance Statistics
        print("\n💳 Balance Statistics")
        print("-" * 30)
        
        balance_stats = exchange.balance_manager.get_balance_statistics()
        print(f"Balance Statistics:")
        print(f"  Total Currencies: {balance_stats['total_currencies']}")
        print(f"  Total Account Types: {balance_stats['total_account_types']}")
        
        if 'account_summaries' in balance_stats:
            for account_type, summary in balance_stats['account_summaries'].items():
                print(f"  {account_type}:")
                print(f"    Total Equity: {summary['total_equity']:,.2f}")
                print(f"    Available Equity: {summary['available_equity']:,.2f}")
                print(f"    Currency Count: {summary['currency_count']}")
        
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during example: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close the exchange
        print("\n🔌 Closing exchange connection...")
        await exchange.close()
        print("✅ Exchange connection closed")


async def demonstrate_advanced_features():
    """Demonstrate advanced features of the enhanced exchange."""
    print("\n🚀 Advanced Features Demonstration")
    print("=" * 50)
    
    exchange = create_enhanced_okx_exchange(
        api_key=os.getenv("OKX_API_KEY", ""),
        api_secret=os.getenv("OKX_API_SECRET", ""),
        password=os.getenv("OKX_PASSPHRASE", ""),
        trade_symbol="BTCUSDT",
        use_testnet=True
    )
    
    try:
        await exchange._initialize_exchange()
        
        # 1. Idempotency Key Management
        print("\n🔑 Idempotency Key Management")
        print("-" * 30)
        
        # Create an idempotency key for order creation
        idempotency_key = exchange.idempotency_manager.create_order_key(
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            quantity=0.001,
            client_order_id="test_idempotent_order"
        )
        print(f"Generated idempotency key: {idempotency_key}")
        
        # Check if operation is duplicate
        is_duplicate = exchange.idempotency_manager.is_operation_duplicate(
            operation_type="create_order",
            parameters={
                "symbol": "BTCUSDT",
                "side": "buy",
                "order_type": "market",
                "quantity": 0.001
            }
        )
        print(f"Is duplicate operation: {is_duplicate is not None}")
        
        # 2. Risk Calculation
        print("\n⚠️ Risk Calculation")
        print("-" * 30)
        
        # Calculate position risk
        position_risk = exchange.risk_calculator.calculate_position_risk(
            symbol="BTCUSDT",
            position_size=0.1,  # 0.1 BTC
            entry_price=50000.0,
            current_price=51000.0,
            leverage=2.0
        )
        
        print(f"Position Risk Calculation:")
        print(f"  Symbol: {position_risk.symbol}")
        print(f"  Position Size: {position_risk.position_size}")
        print(f"  Entry Price: ${position_risk.entry_price:,.2f}")
        print(f"  Current Price: ${position_risk.current_price:,.2f}")
        print(f"  Leverage: {position_risk.leverage}x")
        print(f"  Margin Used: ${position_risk.margin_used:,.2f}")
        print(f"  Unrealized PnL: ${position_risk.unrealized_pnl:,.2f}")
        print(f"  Margin Ratio: {position_risk.margin_ratio:.2%}")
        print(f"  Liquidation Price: ${position_risk.liquidation_price:,.2f}")
        print(f"  Risk Level: {position_risk.risk_level.value}")
        
        # Validate position risk
        is_safe, warnings = exchange.risk_calculator.validate_position_risk(position_risk)
        print(f"  Is Safe: {is_safe}")
        if warnings:
            print(f"  Warnings: {warnings}")
        
        # 3. Precision Validation
        print("\n🎯 Precision Validation")
        print("-" * 30)
        
        # Test price validation
        is_valid_price, price_error = exchange.precision_helper.validate_price(50000.123456789, "BTCUSDT")
        print(f"Price 50000.123456789 valid: {is_valid_price}")
        if not is_valid_price:
            print(f"  Error: {price_error}")
        
        # Test quantity validation
        is_valid_qty, qty_error = exchange.precision_helper.validate_quantity(0.001, "BTCUSDT")
        print(f"Quantity 0.001 valid: {is_valid_qty}")
        if not is_valid_qty:
            print(f"  Error: {qty_error}")
        
        # Round prices and quantities
        rounded_price = exchange.precision_helper.round_price(50000.123456789, "BTCUSDT")
        rounded_qty = exchange.precision_helper.round_quantity(0.00123456789, "BTCUSDT")
        print(f"Rounded price: {rounded_price}")
        print(f"Rounded quantity: {rounded_qty}")
        
        # 4. Market Data Aggregation
        print("\n📊 Market Data Aggregation")
        print("-" * 30)
        
        # Get multiple price sources
        ticker_price = await exchange.price_manager.get_price("BTCUSDT", exchange.price_manager.PriceSource.TICKER)
        orderbook_price = await exchange.price_manager.get_price("BTCUSDT", exchange.price_manager.PriceSource.ORDER_BOOK)
        
        print(f"Ticker Price: ${ticker_price.price if ticker_price else 'N/A'}")
        print(f"Order Book Price: ${orderbook_price.price if orderbook_price else 'N/A'}")
        
        # Get best price from multiple sources
        best_price = await exchange.price_manager.get_best_price("BTCUSDT")
        print(f"Best Price: ${best_price.price if best_price else 'N/A'}")
        
        # 5. Technical Indicators
        print("\n📈 Technical Indicators")
        print("-" * 30)
        
        # Get OHLCV data for technical analysis
        ohlcv_data = await exchange.get_ohlcv("BTCUSDT", "1h", 100)
        if ohlcv_data:
            # Calculate technical indicators
            indicators = exchange.ohlcv_manager.calculate_technical_indicators(
                "BTCUSDT", 
                exchange.ohlcv_manager.Timeframe.HOUR_1,
                ["sma_20", "ema_20", "rsi_14", "macd"]
            )
            
            print(f"Technical Indicators (last 5 values):")
            for indicator_name, values in indicators.items():
                if values:
                    last_values = values[-5:] if len(values) >= 5 else values
                    print(f"  {indicator_name}: {last_values}")
        
        print("\n✅ Advanced features demonstration completed!")
        
    except Exception as e:
        print(f"❌ Error during advanced features demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await exchange.close()


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Run the advanced features demonstration
    asyncio.run(demonstrate_advanced_features())