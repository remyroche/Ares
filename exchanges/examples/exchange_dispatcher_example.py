"""
Exchange Dispatcher Usage Example

This example demonstrates how to use the exchange dispatcher
for exchange-agnostic trading operations.
"""

import asyncio
import os
from datetime import datetime

# Import the exchange dispatcher
from ..exchange_dispatcher import create_okx_dispatcher, create_exchange_dispatcher, ExchangeConfig, ExchangeType


async def main():
    """Main example function."""
    print("🚀 Exchange Dispatcher Example")
    print("=" * 50)
    
    # Method 1: Using convenience function for OKX
    print("\n📡 Method 1: Using OKX convenience function")
    print("-" * 30)
    
    okx_dispatcher = create_okx_dispatcher(
        api_key=os.getenv("OKX_API_KEY", ""),
        api_secret=os.getenv("OKX_API_SECRET", ""),
        password=os.getenv("OKX_PASSPHRASE", ""),
        use_testnet=True,
        trade_symbol="BTCUSDT"
    )
    
    try:
        # Initialize the dispatcher
        print("📡 Initializing OKX dispatcher...")
        success = await okx_dispatcher.initialize()
        
        if success:
            print("✅ OKX dispatcher initialized successfully!")
            
            # Get current price
            price = await okx_dispatcher.get_price("BTCUSDT")
            print(f"Current BTC/USDT price: ${price:,.2f}" if price else "Price not available")
            
            # Get balance
            balance = await okx_dispatcher.get_balance("USDT")
            print(f"USDT Balance: {balance:.2f}")
            
            # Get exchange status
            status = await okx_dispatcher.get_exchange_status()
            print(f"Exchange Status: {status}")
            
        else:
            print("❌ Failed to initialize OKX dispatcher")
    
    except Exception as e:
        print(f"❌ Error with OKX dispatcher: {e}")
    
    finally:
        await okx_dispatcher.close()
    
    # Method 2: Using generic dispatcher with configuration
    print("\n📡 Method 2: Using generic dispatcher")
    print("-" * 30)
    
    config = ExchangeConfig(
        exchange_type=ExchangeType.OKX,
        api_key=os.getenv("OKX_API_KEY", ""),
        api_secret=os.getenv("OKX_API_SECRET", ""),
        password=os.getenv("OKX_PASSPHRASE", ""),
        use_testnet=True,
        trade_symbol="BTCUSDT"
    )
    
    dispatcher = create_exchange_dispatcher(config)
    
    try:
        # Initialize the dispatcher
        print("📡 Initializing generic dispatcher...")
        success = await dispatcher.initialize()
        
        if success:
            print("✅ Generic dispatcher initialized successfully!")
            
            # Get instrument information
            instrument_info = await dispatcher.get_instrument_info("BTCUSDT")
            if instrument_info:
                print(f"Instrument Info for BTCUSDT:")
                print(f"  Base Currency: {instrument_info['base_currency']}")
                print(f"  Quote Currency: {instrument_info['quote_currency']}")
                print(f"  Tick Size: {instrument_info['tick_size']}")
                print(f"  Lot Size: {instrument_info['lot_size']}")
                print(f"  Min Notional: {instrument_info['min_notional']}")
                print(f"  Max Leverage: {instrument_info['max_leverage']}")
            
            # Get positions
            positions = await dispatcher.get_positions()
            print(f"Current positions: {len(positions)}")
            
            # Get liquidation risk
            risk = await dispatcher.get_liquidation_risk("BTCUSDT")
            if risk:
                print(f"Liquidation Risk for BTCUSDT:")
                print(f"  Margin Ratio: {risk['margin_ratio']:.2%}")
                print(f"  Liquidation Price: ${risk['liquidation_price']:,.2f}")
                print(f"  Risk Level: {risk['risk_level']}")
            
        else:
            print("❌ Failed to initialize generic dispatcher")
    
    except Exception as e:
        print(f"❌ Error with generic dispatcher: {e}")
    
    finally:
        await dispatcher.close()
    
    print("\n✅ Exchange dispatcher example completed!")


async def demonstrate_order_operations():
    """Demonstrate order operations through the dispatcher."""
    print("\n📋 Order Operations Demonstration")
    print("=" * 50)
    
    dispatcher = create_okx_dispatcher(
        api_key=os.getenv("OKX_API_KEY", ""),
        api_secret=os.getenv("OKX_API_SECRET", ""),
        password=os.getenv("OKX_PASSPHRASE", ""),
        use_testnet=True,
        trade_symbol="BTCUSDT"
    )
    
    try:
        await dispatcher.initialize()
        
        # Create a small test order
        print("📝 Creating test order...")
        order_result = await dispatcher.create_order(
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            quantity=0.001,  # Very small amount for testing
            client_order_id=f"test_order_{int(datetime.now().timestamp())}"
        )
        
        if order_result:
            print(f"✅ Order created successfully!")
            print(f"Order ID: {order_result.get('order_id', 'N/A')}")
            print(f"Client Order ID: {order_result.get('client_order_id', 'N/A')}")
            print(f"Status: {order_result.get('status', 'N/A')}")
            
            # Get order status
            if order_result.get('order_id'):
                print("\n📊 Getting order status...")
                order_status = await dispatcher.get_order_status(
                    "BTCUSDT", 
                    order_result['order_id']
                )
                if order_status:
                    print(f"Order Status: {order_status}")
            
        else:
            print("❌ Order creation failed")
        
        # Get open orders
        print("\n📋 Getting open orders...")
        open_orders = await dispatcher.get_open_orders()
        print(f"Open orders: {len(open_orders)}")
        
        for order in open_orders[:3]:  # Show first 3 orders
            print(f"  Order: {order.get('symbol')} {order.get('side')} {order.get('quantity')} @ {order.get('price', 'market')}")
    
    except Exception as e:
        print(f"❌ Error in order operations: {e}")
    
    finally:
        await dispatcher.close()


async def demonstrate_risk_management():
    """Demonstrate risk management through the dispatcher."""
    print("\n⚠️ Risk Management Demonstration")
    print("=" * 50)
    
    dispatcher = create_okx_dispatcher(
        api_key=os.getenv("OKX_API_KEY", ""),
        api_secret=os.getenv("OKX_API_SECRET", ""),
        password=os.getenv("OKX_PASSPHRASE", ""),
        use_testnet=True,
        trade_symbol="BTCUSDT"
    )
    
    try:
        await dispatcher.initialize()
        
        # Calculate position risk
        print("📊 Calculating position risk...")
        position_risk = await dispatcher.calculate_position_risk(
            symbol="BTCUSDT",
            position_size=0.1,  # 0.1 BTC
            entry_price=50000.0,
            current_price=51000.0,
            leverage=2.0
        )
        
        if position_risk:
            print(f"Position Risk Calculation:")
            print(f"  Symbol: {position_risk['symbol']}")
            print(f"  Margin Ratio: {position_risk['margin_ratio']:.2%}")
            print(f"  Liquidation Price: ${position_risk['liquidation_price']:,.2f}")
            print(f"  Risk Level: {position_risk['risk_level']}")
            print(f"  Unrealized PnL: ${position_risk['unrealized_pnl']:,.2f}")
            print(f"  Margin Used: ${position_risk['margin_used']:,.2f}")
        
        # Get liquidation risk for current positions
        print("\n⚠️ Getting liquidation risk...")
        liquidation_risk = await dispatcher.get_liquidation_risk("BTCUSDT")
        if liquidation_risk:
            print(f"Liquidation Risk:")
            print(f"  Margin Ratio: {liquidation_risk['margin_ratio']:.2%}")
            print(f"  Liquidation Price: ${liquidation_risk['liquidation_price']:,.2f}")
            print(f"  Risk Level: {liquidation_risk['risk_level']}")
            print(f"  Unrealized PnL: ${liquidation_risk['unrealized_pnl']:,.2f}")
    
    except Exception as e:
        print(f"❌ Error in risk management: {e}")
    
    finally:
        await dispatcher.close()


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Run additional demonstrations
    asyncio.run(demonstrate_order_operations())
    asyncio.run(demonstrate_risk_management())