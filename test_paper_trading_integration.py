#!/usr/bin/env python3
"""
Test Paper Trading Integration

This script tests the paper trading integration functionality.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exchanges.paper_trading_simulator import PaperTradingSimulator, SimulatorConfig
from exchanges.paper_trading_wrapper import PaperTradingWrapper, create_paper_trading_wrapper
from exchanges.paper_trading_integration import PaperTradingIntegration, setup_paper_trading_integration
from exchanges.base_exchange.exchange_interface import OrderSide, OrderType


class MockExchange:
    """Mock exchange for testing"""
    
    def __init__(self, name: str = "mock"):
        self.name = name
        self.connected = False
    
    async def initialize(self):
        self.connected = True
        print(f"✅ Mock exchange '{self.name}' initialized")
    
    async def close(self):
        self.connected = False
        print(f"✅ Mock exchange '{self.name}' closed")
    
    async def get_status(self):
        return "connected" if self.connected else "disconnected"
    
    async def get_ticker(self, symbol: str):
        # Return mock ticker data
        return {
            "symbol": symbol,
            "last": 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 1.0,
            "bid": 49950.0 if "BTC" in symbol else 2995.0 if "ETH" in symbol else 0.99,
            "ask": 50050.0 if "BTC" in symbol else 3005.0 if "ETH" in symbol else 1.01,
            "volume": 1000.0,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 100):
        # Return mock kline data
        base_price = 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 1.0
        klines = []
        for i in range(limit):
            price = base_price * (1 + (i - limit/2) * 0.001)  # Slight price movement
            klines.append({
                "timestamp": datetime.now().timestamp() - (limit - i) * 60,
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price * 1.005,
                "volume": 100.0
            })
        return klines
    
    async def create_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                          quantity: float, price: float = None, **kwargs):
        # Mock order creation - would normally send to real exchange
        return {
            "order_id": f"REAL_{symbol}_{side.value}_{quantity}",
            "symbol": symbol,
            "side": side.value,
            "order_type": order_type.value,
            "quantity": quantity,
            "price": price,
            "status": "filled",
            "executed_at": datetime.now().isoformat()
        }
    
    async def cancel_order(self, order_id: str):
        return {"order_id": order_id, "status": "cancelled"}
    
    async def get_order_status(self, order_id: str):
        return {"order_id": order_id, "status": "filled"}
    
    async def get_open_orders(self, symbol: str = None):
        return []
    
    async def get_account_info(self):
        return {"account_type": "live", "balance": {"USDT": 10000.0}}
    
    async def get_balance(self, currency: str):
        return {"currency": currency, "free": 10000.0, "locked": 0.0, "total": 10000.0}


async def test_simulator():
    """Test the paper trading simulator"""
    print("\n🧪 Testing Paper Trading Simulator...")
    
    # Create simulator with custom config
    config = SimulatorConfig(
        maker_fee_rate=0.001,
        taker_fee_rate=0.001,
        base_slippage_rate=0.0001,
        max_slippage_rate=0.01
    )
    
    simulator = PaperTradingSimulator(config)
    
    # Test order creation
    print("Creating test orders...")
    
    # Buy order
    buy_result = await simulator.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1
    )
    print(f"✅ Buy order created: {buy_result['order_id']}")
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
    print(f"✅ Sell order created: {sell_result['order_id']}")
    print(f"   Executed: {sell_result['executed_quantity']} @ {sell_result['executed_price']}")
    print(f"   Commission: {sell_result['commission']} {sell_result['commission_asset']}")
    print(f"   Slippage: {sell_result['slippage']}")
    
    # Check positions
    positions = simulator.get_positions()
    print(f"\n📊 Current positions: {len(positions)}")
    for pos in positions:
        print(f"   {pos['symbol']}: {pos['quantity']} @ {pos['entry_price']} (P&L: {pos['unrealized_pnl']:.2f})")
    
    # Check balance
    balance = simulator.get_balance()
    print(f"\n💰 Balance: {balance}")
    
    # Check performance metrics
    metrics = simulator.get_performance_metrics()
    print(f"\n📈 Performance metrics:")
    print(f"   Total trades: {metrics['total_trades']}")
    print(f"   Total volume: {metrics['total_volume']:.2f}")
    print(f"   Total commission: {metrics['total_commission']:.2f}")
    print(f"   Portfolio value: {metrics['current_portfolio_value']:.2f}")
    
    return True


async def test_wrapper():
    """Test the paper trading wrapper"""
    print("\n🧪 Testing Paper Trading Wrapper...")
    
    # Create mock exchange
    mock_exchange = MockExchange("test_exchange")
    
    # Create wrapper in PAPER mode
    wrapper = create_paper_trading_wrapper(
        exchange=mock_exchange,
        trading_mode="PAPER"
    )
    
    # Initialize wrapper
    await wrapper.initialize()
    
    # Test market data (should use real exchange)
    print("Testing market data access...")
    ticker = await wrapper.get_ticker("BTCUSDT")
    print(f"✅ Ticker data: {ticker['symbol']} @ {ticker['last']}")
    
    # Test order creation (should use simulator)
    print("Testing order creation...")
    order_result = await wrapper.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1
    )
    print(f"✅ Order created: {order_result['order_id']}")
    print(f"   Status: {order_result['status']}")
    print(f"   Executed: {order_result['executed_quantity']} @ {order_result['executed_price']}")
    
    # Test position tracking
    positions = wrapper.get_positions()
    print(f"\n📊 Positions: {len(positions)}")
    for pos in positions:
        print(f"   {pos['symbol']}: {pos['quantity']} @ {pos['entry_price']}")
    
    # Test switching to live mode
    print("\nTesting mode switching...")
    wrapper.set_trading_mode("TRADE")
    print(f"✅ Switched to: {wrapper.get_trading_mode()}")
    
    # Test live order (would go to real exchange)
    live_order = await wrapper.create_order(
        symbol="ETHUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0
    )
    print(f"✅ Live order created: {live_order['order_id']}")
    
    await wrapper.close()
    return True


async def test_integration():
    """Test the paper trading integration"""
    print("\n🧪 Testing Paper Trading Integration...")
    
    # Create configuration
    config = {
        "paper_trading": {
            "maker_fee_rate": 0.001,
            "taker_fee_rate": 0.001,
            "base_slippage_rate": 0.0001,
            "max_slippage_rate": 0.01
        }
    }
    
    # Create integration
    integration = PaperTradingIntegration(config)
    
    # Register mock exchanges
    integration.register_exchange("binance", MockExchange("binance"))
    integration.register_exchange("okx", MockExchange("okx"))
    
    # Initialize integration
    await integration.initialize()
    
    # Test mode switching
    print("Testing mode switching...")
    integration.set_trading_mode("PAPER")
    print(f"✅ Mode set to: {integration.get_trading_mode()}")
    
    # Test trade execution
    print("Testing trade execution...")
    trade_result = await integration.execute_trade(
        exchange_name="binance",
        symbol="BTCUSDT",
        side="buy",
        quantity=0.1,
        price=50000.0,
        order_type="market"
    )
    print(f"✅ Trade executed: {trade_result['order_id']}")
    
    # Test portfolio summary
    portfolio = integration.get_portfolio_summary()
    print(f"\n📊 Portfolio summary:")
    print(f"   Mode: {portfolio['trading_mode']}")
    print(f"   Portfolio value: {portfolio['total_portfolio_value']:.2f}")
    print(f"   Active positions: {portfolio['active_positions']}")
    print(f"   Total trades: {portfolio['total_trades']}")
    
    # Test comprehensive report
    print("\nGenerating comprehensive report...")
    report = await integration.generate_comprehensive_report("test")
    print(f"✅ Report generated with {len(report)} sections")
    
    # Close integration
    await integration.close()
    return True


async def test_launcher_integration():
    """Test the enhanced trading launcher integration"""
    print("\n🧪 Testing Enhanced Trading Launcher Integration...")
    
    # Create configuration
    config = {
        "enhanced_trading_launcher": {
            "enable_paper_trading": True,
            "enable_live_trading": False,
            "enable_backtesting": False,
            "trading_mode": "PAPER"
        },
        "paper_trading": {
            "maker_fee_rate": 0.001,
            "taker_fee_rate": 0.001,
            "base_slippage_rate": 0.0001,
            "max_slippage_rate": 0.01
        }
    }
    
    # Import and create launcher
    from src.launcher.enhanced_trading_launcher import EnhancedTradingLauncher
    
    launcher = EnhancedTradingLauncher(config)
    
    # Initialize launcher
    success = await launcher.initialize()
    if not success:
        print("❌ Failed to initialize launcher")
        return False
    
    print(f"✅ Launcher initialized in mode: {launcher.get_trading_mode()}")
    
    # Test mode switching
    launcher.set_trading_mode("PAPER")
    print(f"✅ Mode switched to: {launcher.get_trading_mode()}")
    
    # Test trade execution (this would require registered exchanges)
    print("Note: Trade execution requires registered exchanges")
    print("Launcher status:", launcher.get_launcher_status())
    
    return True


async def main():
    """Run all tests"""
    print("🚀 Starting Paper Trading Integration Tests...")
    
    try:
        # Test individual components
        await test_simulator()
        await test_wrapper()
        await test_integration()
        await test_launcher_integration()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    asyncio.run(main())