#!/usr/bin/env python3
"""
Test Paper Trading Integration

This script demonstrates the paper trading functionality by:
1. Creating a paper trading launcher
2. Executing simulated orders
3. Tracking positions and P&L
4. Displaying performance metrics
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.launcher.trade_launcher import create_paper_trading_launcher
from src.utils.tprint import tprint_info, tprint_success, tprint_error

async def test_paper_trading():
    """Test paper trading functionality."""
    try:
        tprint_info("🚀 Starting Paper Trading Test...")
        
        # Configuration for paper trading
        exchange_config = {
            'exchange_type': 'simulated',  # Use simulated exchange for testing
            'trading_mode': 'paper',
            'initial_balance': 10000.0,
            'maker_fee': 0.001,  # 0.1%
            'taker_fee': 0.001,  # 0.1%
            'max_slippage': 0.005,  # 0.5%
            'slippage_model': 'linear'
        }
        
        paper_config = {
            'initial_balance': 10000.0,
            'risk_limits': {
                'max_position_size': 0.1,  # 10% of portfolio per position
                'max_daily_loss': 0.05,    # 5% max daily loss
                'stop_loss_pct': 0.02,     # 2% stop loss
                'take_profit_pct': 0.05    # 5% take profit
            }
        }
        
        # Create paper trading launcher
        launcher = create_paper_trading_launcher(exchange_config, paper_config)
        
        # Initialize launcher
        tprint_info("📋 Initializing paper trading launcher...")
        success = await launcher.initialize()
        if not success:
            tprint_error("❌ Failed to initialize launcher")
            return
        
        # Start launcher
        tprint_info("▶️ Starting paper trading launcher...")
        success = await launcher.start()
        if not success:
            tprint_error("❌ Failed to start launcher")
            return
        
        tprint_success("✅ Paper trading launcher started successfully!")
        
        # Test 1: Check initial balance
        tprint_info("\n📊 Test 1: Checking initial balance...")
        balance = await launcher.get_account_balance()
        tprint_info(f"Initial balance: {balance}")
        
        # Test 2: Execute a buy order
        tprint_info("\n📈 Test 2: Executing buy order...")
        buy_result = await launcher.execute_order(
            symbol='BTCUSDT',
            side='buy',
            order_type='market',
            quantity=0.1
        )
        tprint_info(f"Buy order result: {buy_result}")
        
        # Test 3: Check balance after buy
        tprint_info("\n💰 Test 3: Checking balance after buy...")
        balance_after_buy = await launcher.get_account_balance()
        tprint_info(f"Balance after buy: {balance_after_buy}")
        
        # Test 4: Check positions
        tprint_info("\n📋 Test 4: Checking positions...")
        positions = await launcher.get_positions()
        tprint_info(f"Current positions: {positions}")
        
        # Test 5: Execute a sell order
        tprint_info("\n📉 Test 5: Executing sell order...")
        sell_result = await launcher.execute_order(
            symbol='BTCUSDT',
            side='sell',
            order_type='market',
            quantity=0.05
        )
        tprint_info(f"Sell order result: {sell_result}")
        
        # Test 6: Check final balance and positions
        tprint_info("\n📊 Test 6: Checking final state...")
        final_balance = await launcher.get_account_balance()
        final_positions = await launcher.get_positions()
        tprint_info(f"Final balance: {final_balance}")
        tprint_info(f"Final positions: {final_positions}")
        
        # Test 7: Get performance metrics
        tprint_info("\n📈 Test 7: Getting performance metrics...")
        metrics = await launcher.get_performance_metrics()
        tprint_info(f"Performance metrics: {metrics}")
        
        # Test 8: Test with ETH
        tprint_info("\n🪙 Test 8: Testing with ETH...")
        eth_buy = await launcher.execute_order(
            symbol='ETHUSDT',
            side='buy',
            order_type='market',
            quantity=1.0
        )
        tprint_info(f"ETH buy result: {eth_buy}")
        
        # Check final state
        final_balance = await launcher.get_account_balance()
        final_positions = await launcher.get_positions()
        tprint_info(f"Final balance after ETH trade: {final_balance}")
        tprint_info(f"Final positions after ETH trade: {final_positions}")
        
        # Stop launcher
        tprint_info("\n⏹️ Stopping paper trading launcher...")
        await launcher.stop()
        
        tprint_success("✅ Paper trading test completed successfully!")
        
    except Exception as e:
        tprint_error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_live_vs_paper_comparison():
    """Test comparison between live and paper trading modes."""
    try:
        tprint_info("\n🔄 Testing Live vs Paper Trading Comparison...")
        
        # Paper trading configuration
        paper_exchange_config = {
            'exchange_type': 'simulated',
            'trading_mode': 'paper',
            'initial_balance': 10000.0
        }
        
        paper_config = {
            'initial_balance': 10000.0
        }
        
        # Live trading configuration (simulated for testing)
        live_exchange_config = {
            'exchange_type': 'simulated',
            'trading_mode': 'trade',
            'initial_balance': 10000.0
        }
        
        # Create both launchers
        paper_launcher = create_paper_trading_launcher(paper_exchange_config, paper_config)
        live_launcher = create_live_trading_launcher(live_exchange_config)
        
        # Initialize both
        await paper_launcher.initialize()
        await live_launcher.initialize()
        
        # Start both
        await paper_launcher.start()
        await live_launcher.start()
        
        # Execute same order on both
        symbol = 'BTCUSDT'
        side = 'buy'
        order_type = 'market'
        quantity = 0.1
        
        tprint_info(f"Executing {side} {quantity} {symbol} on both launchers...")
        
        paper_result = await paper_launcher.execute_order(symbol, side, order_type, quantity)
        live_result = await live_launcher.execute_order(symbol, side, order_type, quantity)
        
        tprint_info(f"Paper trading result: {paper_result}")
        tprint_info(f"Live trading result: {live_result}")
        
        # Compare balances
        paper_balance = await paper_launcher.get_account_balance()
        live_balance = await live_launcher.get_account_balance()
        
        tprint_info(f"Paper trading balance: {paper_balance}")
        tprint_info(f"Live trading balance: {live_balance}")
        
        # Stop both
        await paper_launcher.stop()
        await live_launcher.stop()
        
        tprint_success("✅ Live vs Paper comparison completed!")
        
    except Exception as e:
        tprint_error(f"❌ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_paper_trading())
    asyncio.run(test_live_vs_paper_comparison())