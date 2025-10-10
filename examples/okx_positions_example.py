#!/usr/bin/env python3
"""
OKX Position Management Example

This example demonstrates the comprehensive OKX position fetching suite
including all position-related methods and utilities.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchanges.okx import OkxExchange


async def main():
    """Main example function demonstrating OKX position functionality."""
    
    # Initialize OKX exchange
    # Note: Replace with your actual API credentials
    api_key = os.getenv("OKX_API_KEY", "")
    api_secret = os.getenv("OKX_API_SECRET", "")
    password = os.getenv("OKX_PASSWORD", "")
    trade_symbol = "BTCUSDT"
    
    if not api_key or not api_secret:
        print("⚠️  Please set OKX_API_KEY and OKX_API_SECRET environment variables")
        print("   For demo purposes, using mock credentials...")
        api_key = "demo_key"
        api_secret = "demo_secret"
    
    exchange = OkxExchange(api_key, api_secret, trade_symbol, password)
    
    try:
        # Initialize the exchange
        await exchange._initialize_exchange()
        print("✅ OKX Exchange initialized successfully")
        
        # Example 1: Get all positions
        print("\n📊 Fetching all positions...")
        all_positions = await exchange.get_all_positions("SPOT")
        print(f"Found {len(all_positions)} positions")
        
        if all_positions:
            print("Sample position data:")
            for i, position in enumerate(all_positions[:3]):  # Show first 3
                print(f"  Position {i+1}: {position['symbol']} - Size: {position['size']} - PnL: {position['unrealizedPnl']}")
        
        # Example 2: Get position for specific symbol
        print(f"\n🎯 Fetching position for {trade_symbol}...")
        btc_position = await exchange.get_position_by_symbol(trade_symbol, "SPOT")
        if btc_position:
            print(f"BTC Position: {btc_position}")
        else:
            print("No BTC position found")
        
        # Example 3: Get position history
        print("\n📈 Fetching position history...")
        history = await exchange.get_position_history(
            symbol=trade_symbol,
            inst_type="SPOT",
            limit=10
        )
        print(f"Found {len(history)} historical records")
        
        # Example 4: Get position margin information
        print("\n💰 Fetching position margin...")
        margin_info = await exchange.get_position_margin(trade_symbol)
        if margin_info:
            print(f"Margin info: {margin_info}")
        else:
            print("No margin information available")
        
        # Example 5: Get funding information
        print("\n💸 Fetching funding information...")
        funding_info = await exchange.get_position_funding(trade_symbol)
        print(f"Found {len(funding_info)} funding records")
        
        # Example 6: Get comprehensive risk metrics
        print("\n⚠️  Calculating risk metrics...")
        risk_metrics = await exchange.get_position_risk_metrics()
        if risk_metrics:
            print(f"Portfolio Risk Score: {risk_metrics.get('portfolioRiskScore', 0):.2%}")
            print(f"Total Positions: {risk_metrics.get('totalPositions', 0)}")
            print(f"Total Unrealized PnL: ${risk_metrics.get('totalUnrealizedPnl', 0):.2f}")
            print(f"Max Leverage: {risk_metrics.get('maxLeverage', 0)}x")
            print(f"High Risk Positions: {len(risk_metrics.get('highRiskPositions', []))}")
        
        # Example 7: Get position summary
        print("\n📋 Getting position summary...")
        summary = await exchange.get_position_summary("SPOT")
        print(f"Total Value: ${summary.get('totalValue', 0):.2f}")
        print(f"Total Unrealized PnL: ${summary.get('totalUnrealizedPnl', 0):.2f}")
        print(f"Long Positions: {summary.get('longPositions', 0)}")
        print(f"Short Positions: {summary.get('shortPositions', 0)}")
        print(f"Average Leverage: {summary.get('averageLeverage', 0):.2f}x")
        
        # Example 8: Get position alerts
        print("\n🚨 Checking position alerts...")
        alerts = await exchange.get_position_alerts(risk_threshold=0.05)  # 5% threshold
        print(f"Found {len(alerts)} alerts")
        for alert in alerts:
            print(f"  Alert: {alert['message']}")
        
        # Example 9: Calculate position size
        print("\n🧮 Calculating position size...")
        position_calc = await exchange.calculate_position_size(
            symbol="BTCUSDT",
            risk_amount=1000,  # Risk $1000
            entry_price=50000,  # Entry at $50k
            stop_loss_price=48000,  # Stop loss at $48k
            leverage=2.0  # 2x leverage
        )
        if "error" not in position_calc:
            print(f"Position Size: {position_calc['position_size']:.6f} BTC")
            print(f"Position Value: ${position_calc['position_value']:.2f}")
            print(f"Leveraged Size: {position_calc['leveraged_size']:.6f} BTC")
            print(f"Margin Required: ${position_calc['margin_required']:.2f}")
        else:
            print(f"Error: {position_calc['error']}")
        
        # Example 10: Position streaming (demo with polling)
        print("\n📡 Starting position streaming (5 seconds)...")
        
        async def position_callback(positions):
            print(f"  📊 Position update: {len(positions)} positions")
            if positions:
                total_pnl = sum(float(p.get('unrealizedPnl', 0)) for p in positions)
                print(f"  💰 Total PnL: ${total_pnl:.2f}")
        
        # Start streaming task
        stream_task = asyncio.create_task(
            exchange.get_positions_stream(position_callback, "SPOT")
        )
        
        # Let it run for 5 seconds
        await asyncio.sleep(5)
        stream_task.cancel()
        
        print("\n✅ Position streaming demo completed")
        
        # Example 11: Different instrument types
        print("\n🔧 Testing different instrument types...")
        for inst_type in ["SPOT", "MARGIN", "SWAP", "FUTURES"]:
            positions = await exchange.get_all_positions(inst_type)
            print(f"  {inst_type}: {len(positions)} positions")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Close the exchange connection
        await exchange.close()
        print("\n🔌 Exchange connection closed")


async def demo_position_risk_monitoring():
    """Demo function showing real-time position risk monitoring."""
    print("\n" + "="*60)
    print("🎯 POSITION RISK MONITORING DEMO")
    print("="*60)
    
    # Initialize exchange
    exchange = OkxExchange("demo", "demo", "BTCUSDT")
    await exchange._initialize_exchange()
    
    try:
        # Monitor positions for 10 seconds
        print("Monitoring positions for 10 seconds...")
        
        async def risk_callback(risk_data):
            if risk_data:
                print(f"\n📊 Risk Update at {datetime.now().strftime('%H:%M:%S')}")
                print(f"  Portfolio Risk: {risk_data.get('portfolioRiskScore', 0):.2%}")
                print(f"  Total PnL: ${risk_data.get('totalUnrealizedPnl', 0):.2f}")
                print(f"  High Risk Positions: {len(risk_data.get('highRiskPositions', []))}")
        
        # Start risk monitoring
        risk_task = asyncio.create_task(
            exchange.get_position_risk_stream("BTCUSDT", risk_callback)
        )
        
        await asyncio.sleep(10)
        risk_task.cancel()
        
    except Exception as e:
        print(f"❌ Error in risk monitoring: {e}")
    
    finally:
        await exchange.close()


async def demo_position_alerts():
    """Demo function showing position alert system."""
    print("\n" + "="*60)
    print("🚨 POSITION ALERTS DEMO")
    print("="*60)
    
    exchange = OkxExchange("demo", "demo", "BTCUSDT")
    await exchange._initialize_exchange()
    
    try:
        # Check for alerts with different thresholds
        thresholds = [0.01, 0.05, 0.10, 0.20]  # 1%, 5%, 10%, 20%
        
        for threshold in thresholds:
            print(f"\nChecking alerts with {threshold:.0%} risk threshold...")
            alerts = await exchange.get_position_alerts(threshold)
            
            if alerts:
                print(f"  Found {len(alerts)} alerts:")
                for alert in alerts:
                    print(f"    - {alert['type']}: {alert['message']}")
            else:
                print("  No alerts found")
    
    except Exception as e:
        print(f"❌ Error in alerts demo: {e}")
    
    finally:
        await exchange.close()


if __name__ == "__main__":
    print("🚀 OKX Position Management Suite Demo")
    print("="*60)
    
    # Run main demo
    asyncio.run(main())
    
    # Run additional demos
    asyncio.run(demo_position_risk_monitoring())
    asyncio.run(demo_position_alerts())
    
    print("\n🎉 Demo completed successfully!")
    print("\n📚 Available OKX Position Methods:")
    print("  - get_all_positions(inst_type)")
    print("  - get_position_by_symbol(symbol, inst_type)")
    print("  - get_position_history(symbol, inst_type, after, before, limit)")
    print("  - get_position_margin(symbol)")
    print("  - get_position_funding(symbol)")
    print("  - get_position_risk_metrics(symbol)")
    print("  - get_position_summary(inst_type)")
    print("  - get_position_alerts(risk_threshold)")
    print("  - calculate_position_size(symbol, risk_amount, entry_price, stop_loss_price, leverage)")
    print("  - get_positions_stream(callback, inst_type)")
    print("  - get_position_risk_stream(symbol, callback)")