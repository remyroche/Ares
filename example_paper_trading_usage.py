#!/usr/bin/env python3
"""
Paper Trading Usage Example

This example demonstrates how to use the paper trading functionality
in your trading strategies.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.launcher.trade_launcher import create_paper_trading_launcher, create_live_trading_launcher
from src.utils.tprint import tprint_info, tprint_success, tprint_error

async def example_paper_trading_strategy():
    """Example of using paper trading in a simple strategy."""
    
    # Configuration for paper trading
    exchange_config = {
        'exchange_type': 'simulated',  # or 'binance', 'coinbase', etc.
        'trading_mode': 'paper',       # This enables paper trading mode
        'initial_balance': 10000.0,    # Starting with $10,000
        'maker_fee': 0.001,            # 0.1% maker fee
        'taker_fee': 0.001,            # 0.1% taker fee
        'max_slippage': 0.005,         # 0.5% maximum slippage
    }
    
    paper_config = {
        'initial_balance': 10000.0,
        'risk_limits': {
            'max_position_size': 0.2,  # Max 20% of portfolio per position
            'max_daily_loss': 0.05,    # Max 5% daily loss
            'stop_loss_pct': 0.02,     # 2% stop loss
            'take_profit_pct': 0.05    # 5% take profit
        }
    }
    
    # Create paper trading launcher
    launcher = create_paper_trading_launcher(exchange_config, paper_config)
    
    try:
        # Initialize and start
        await launcher.initialize()
        await launcher.start()
        
        tprint_success("🚀 Paper trading launcher started!")
        
        # Example trading strategy
        await run_simple_strategy(launcher)
        
    except Exception as e:
        tprint_error(f"❌ Strategy failed: {e}")
    finally:
        # Always stop the launcher
        await launcher.stop()

async def run_simple_strategy(launcher):
    """Run a simple trading strategy."""
    
    # Strategy parameters
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    position_size = 0.1  # 10% of portfolio per trade
    
    tprint_info("📈 Running simple trading strategy...")
    
    for symbol in symbols:
        try:
            # Get current balance
            balance = await launcher.get_account_balance()
            usdt_balance = balance.get('USDT', 0)
            
            if usdt_balance < 100:  # Minimum balance check
                tprint_info(f"⚠️ Insufficient balance for {symbol}: ${usdt_balance:.2f}")
                continue
            
            # Calculate position size
            position_value = usdt_balance * position_size
            
            # Get current price (this would be from your strategy)
            # For this example, we'll use a simple market order
            tprint_info(f"🔄 Executing buy order for {symbol}...")
            
            # Execute buy order
            buy_result = await launcher.execute_order(
                symbol=symbol,
                side='buy',
                order_type='market',
                quantity=position_value / 50000  # Approximate BTC price
            )
            
            if 'error' not in buy_result:
                tprint_success(f"✅ Buy order executed for {symbol}")
                
                # Wait a bit (simulate strategy logic)
                await asyncio.sleep(1)
                
                # Execute sell order (simulate quick profit taking)
                tprint_info(f"🔄 Executing sell order for {symbol}...")
                
                sell_result = await launcher.execute_order(
                    symbol=symbol,
                    side='sell',
                    order_type='market',
                    quantity=position_value / 50000
                )
                
                if 'error' not in sell_result:
                    tprint_success(f"✅ Sell order executed for {symbol}")
                else:
                    tprint_error(f"❌ Sell order failed for {symbol}: {sell_result['error']}")
            else:
                tprint_error(f"❌ Buy order failed for {symbol}: {buy_result['error']}")
                
        except Exception as e:
            tprint_error(f"❌ Error trading {symbol}: {e}")
    
    # Get final performance metrics
    tprint_info("\n📊 Final Performance Report:")
    metrics = await launcher.get_performance_metrics()
    
    for key, value in metrics.items():
        if isinstance(value, float):
            tprint_info(f"  {key}: {value:.4f}")
        else:
            tprint_info(f"  {key}: {value}")
    
    # Get final positions
    positions = await launcher.get_positions()
    if positions:
        tprint_info("\n📋 Final Positions:")
        for pos in positions:
            tprint_info(f"  {pos['symbol']}: {pos['quantity']:.4f} @ ${pos['entry_price']:.2f} (P&L: ${pos['unrealized_pnl']:.2f})")
    else:
        tprint_info("📋 No open positions")

async def example_live_trading():
    """Example of using live trading (for comparison)."""
    
    # Configuration for live trading
    exchange_config = {
        'exchange_type': 'simulated',  # or 'binance', 'coinbase', etc.
        'trading_mode': 'trade',       # This enables live trading mode
        'api_key': 'your_api_key',     # Your exchange API key
        'api_secret': 'your_api_secret', # Your exchange API secret
        'testnet': True,               # Use testnet for safety
    }
    
    # Create live trading launcher
    launcher = create_live_trading_launcher(exchange_config)
    
    try:
        # Initialize and start
        await launcher.initialize()
        await launcher.start()
        
        tprint_success("🚀 Live trading launcher started!")
        
        # In live trading mode, orders go directly to the exchange
        # This is the same interface as paper trading, but with real money
        
        # Example: Execute a real order (be careful!)
        # result = await launcher.execute_order(
        #     symbol='BTCUSDT',
        #     side='buy',
        #     order_type='market',
        #     quantity=0.001  # Small amount for safety
        # )
        
        tprint_info("⚠️ Live trading example - orders would go to real exchange")
        
    except Exception as e:
        tprint_error(f"❌ Live trading failed: {e}")
    finally:
        await launcher.stop()

def print_usage_instructions():
    """Print usage instructions."""
    print("""
📚 Paper Trading Usage Instructions:

1. Paper Trading Mode:
   - Set 'trading_mode': 'paper' in exchange_config
   - Orders are simulated with realistic pricing, slippage, and fees
   - No real money is at risk
   - Perfect for strategy testing and development

2. Live Trading Mode:
   - Set 'trading_mode': 'trade' in exchange_config
   - Orders go directly to the exchange
   - Real money is at risk
   - Use with caution and proper risk management

3. Key Features:
   - Real-time price fetching from exchange
   - Realistic slippage calculation
   - Fee calculation (maker/taker fees)
   - Position tracking with P&L
   - Risk management limits
   - Performance metrics

4. Configuration Options:
   - initial_balance: Starting balance for paper trading
   - maker_fee/taker_fee: Trading fees
   - max_slippage: Maximum slippage percentage
   - risk_limits: Risk management settings

5. Usage:
   - Create launcher with appropriate config
   - Initialize and start launcher
   - Execute orders using execute_order()
   - Monitor positions and performance
   - Stop launcher when done
""")

if __name__ == "__main__":
    print_usage_instructions()
    
    # Run the paper trading example
    asyncio.run(example_paper_trading_strategy())
    
    # Uncomment to run live trading example (be careful!)
    # asyncio.run(example_live_trading())