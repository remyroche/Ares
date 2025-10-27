#!/usr/bin/env python3
"""
Simple Test for Paper Trading Components

This script tests the paper trading components without importing the full exchange module.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only the specific modules we need
from exchanges.paper_trading_simulator import PaperTradingSimulator, SimulatorConfig, OrderSide, OrderType


async def test_simulator_basic():
    """Test basic simulator functionality"""
    print("🧪 Testing Paper Trading Simulator (Basic)...")
    
    # Create simulator
    simulator = PaperTradingSimulator()
    
    # Test order creation
    print("Creating test orders...")
    
    # Buy order
    buy_result = await simulator.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1
    )
    print(f"✅ Buy order: {buy_result['order_id']}")
    print(f"   Executed: {buy_result['executed_quantity']} @ {buy_result['executed_price']}")
    print(f"   Commission: {buy_result['commission']} {buy_result['commission_asset']}")
    print(f"   Slippage: {buy_result['slippage']}")
    
    # Sell order
    sell_result = await simulator.create_order(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=0.05
    )
    print(f"✅ Sell order: {sell_result['order_id']}")
    print(f"   Executed: {sell_result['executed_quantity']} @ {sell_result['executed_price']}")
    print(f"   Commission: {sell_result['commission']} {sell_result['commission_asset']}")
    print(f"   Slippage: {sell_result['slippage']}")
    
    # Check positions
    positions = simulator.get_positions()
    print(f"\n📊 Positions: {len(positions)}")
    for pos in positions:
        print(f"   {pos['symbol']}: {pos['quantity']} @ {pos['entry_price']} (P&L: {pos['unrealized_pnl']:.2f})")
    
    # Check balance
    balance = simulator.get_balance()
    print(f"\n💰 Balance: {balance}")
    
    # Check performance metrics
    metrics = simulator.get_performance_metrics()
    print(f"\n📈 Performance:")
    print(f"   Trades: {metrics['total_trades']}")
    print(f"   Volume: {metrics['total_volume']:.2f}")
    print(f"   Commission: {metrics['total_commission']:.2f}")
    print(f"   Portfolio: {metrics['current_portfolio_value']:.2f}")
    
    return True


async def test_simulator_advanced():
    """Test advanced simulator functionality"""
    print("\n🧪 Testing Paper Trading Simulator (Advanced)...")
    
    # Create simulator with custom config
    config = SimulatorConfig(
        maker_fee_rate=0.002,  # 0.2% maker fee
        taker_fee_rate=0.002,  # 0.2% taker fee
        base_slippage_rate=0.0005,  # 0.05% base slippage
        max_slippage_rate=0.02,  # 2% max slippage
        execution_delay_ms=(50, 200),  # 50-200ms delay
        partial_fill_probability=0.2,  # 20% partial fill chance
        rejection_probability=0.005  # 0.5% rejection chance
    )
    
    simulator = PaperTradingSimulator(config)
    
    # Test multiple orders
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    orders = []
    
    for symbol in symbols:
        # Buy order
        buy_result = await simulator.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        orders.append(buy_result)
        print(f"✅ {symbol} buy: {buy_result['executed_quantity']} @ {buy_result['executed_price']}")
        
        # Sell order
        sell_result = await simulator.create_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.05
        )
        orders.append(sell_result)
        print(f"✅ {symbol} sell: {sell_result['executed_quantity']} @ {sell_result['executed_price']}")
    
    # Test limit orders
    print("\nTesting limit orders...")
    limit_buy = await simulator.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.1,
        price=45000.0  # Lower than current price
    )
    print(f"✅ Limit buy: {limit_buy['status']} - {limit_buy.get('metadata', {}).get('cancellation_reason', 'N/A')}")
    
    # Test order cancellation
    print("\nTesting order cancellation...")
    pending_order = await simulator.create_order(
        symbol="ETHUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        price=2500.0  # Much lower than current price
    )
    
    if pending_order['status'] == 'pending':
        cancel_result = await simulator.cancel_order(pending_order['order_id'])
        print(f"✅ Order cancelled: {cancel_result['status']}")
    
    # Check final state
    positions = simulator.get_positions()
    balance = simulator.get_balance()
    metrics = simulator.get_performance_metrics()
    
    print(f"\n📊 Final State:")
    print(f"   Positions: {len(positions)}")
    print(f"   Balance: {balance}")
    print(f"   Total trades: {metrics['total_trades']}")
    print(f"   Portfolio value: {metrics['current_portfolio_value']:.2f}")
    
    return True


async def test_launcher_mode_switching():
    """Test launcher mode switching"""
    print("\n🧪 Testing Launcher Mode Switching...")
    
    # Test the launcher mode switching without full initialization
    from src.launcher.enhanced_trading_launcher import EnhancedTradingLauncher
    
    # Create minimal config
    config = {
        "enhanced_trading_launcher": {
            "enable_paper_trading": True,
            "enable_live_trading": False,
            "enable_backtesting": False,
            "trading_mode": "TRADE"
        }
    }
    
    launcher = EnhancedTradingLauncher(config)
    
    # Test mode switching
    print(f"Initial mode: {launcher.get_trading_mode()}")
    
    # Switch to PAPER mode
    success = launcher.set_trading_mode("PAPER")
    print(f"Switch to PAPER: {success} - Mode: {launcher.get_trading_mode()}")
    print(f"Is paper mode: {launcher.is_paper_mode()}")
    print(f"Is trade mode: {launcher.is_trade_mode()}")
    
    # Switch to TRADE mode
    success = launcher.set_trading_mode("TRADE")
    print(f"Switch to TRADE: {success} - Mode: {launcher.get_trading_mode()}")
    print(f"Is paper mode: {launcher.is_paper_mode()}")
    print(f"Is trade mode: {launcher.is_trade_mode()}")
    
    # Test invalid mode
    success = launcher.set_trading_mode("INVALID")
    print(f"Invalid mode: {success} - Mode: {launcher.get_trading_mode()}")
    
    # Test launcher status
    status = launcher.get_launcher_status()
    print(f"Launcher status: {status}")
    
    return True


async def main():
    """Run all tests"""
    print("🚀 Starting Simple Paper Trading Tests...")
    
    try:
        # Test basic simulator
        await test_simulator_basic()
        
        # Test advanced simulator
        await test_simulator_advanced()
        
        # Test launcher mode switching
        await test_launcher_mode_switching()
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)