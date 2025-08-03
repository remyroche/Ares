#!/usr/bin/env python3
"""
Test the updated MEXC implementation with proper pagination.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from exchange.mexc_updated import MexcExchangeUpdated

async def test_mexc_pagination():
    """Test MEXC pagination with detailed logging."""
    print("🔍 DEBUG: Starting MEXC pagination test")
    
    try:
        # Create MEXC client
        print("🔍 DEBUG: Creating MEXC client...")
        client = MexcExchangeUpdated("", "", "ETHUSDT")  # No API keys needed for public data
        
        # Test parameters - 1 day for testing
        symbol = "ETHUSDT"
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)  # 1 day for testing
        start_time_ms = int(start_time.timestamp() * 1000)
        end_time_ms = int(end_time.timestamp() * 1000)
        
        print(f"🔍 DEBUG: Test parameters:")
        print(f"   Symbol: {symbol}")
        print(f"   Start time: {start_time}")
        print(f"   End time: {end_time}")
        print(f"   Time range: {(end_time - start_time).days} days")
        print(f"   Start time ms: {start_time_ms}")
        print(f"   End time ms: {end_time_ms}")
        
        # Test the method
        print("🔍 DEBUG: Calling get_historical_agg_trades with pagination...")
        trades = await client.get_historical_agg_trades(
            symbol=symbol,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            limit=1000
        )
        
        print(f"🔍 DEBUG: Method completed")
        print(f"🔍 DEBUG: Returned {len(trades)} trades")
        
        if trades:
            print(f"🔍 DEBUG: First trade: {trades[0]}")
            print(f"🔍 DEBUG: Last trade: {trades[-1]}")
            print(f"🔍 DEBUG: Time range of data:")
            first_time = datetime.fromtimestamp(trades[0]['T'] / 1000)
            last_time = datetime.fromtimestamp(trades[-1]['T'] / 1000)
            print(f"   First trade time: {first_time}")
            print(f"   Last trade time: {last_time}")
            print(f"   Data spans: {(last_time - first_time).total_seconds() / 3600:.1f} hours")
            
            # Calculate trades per hour
            hours_span = (last_time - first_time).total_seconds() / 3600
            trades_per_hour = len(trades) / hours_span if hours_span > 0 else 0
            print(f"   📈 Trades per hour: {trades_per_hour:.1f}")
            
            # Show sample trades
            print(f"   📋 Sample trades:")
            for i, trade in enumerate(trades[:5]):
                trade_time = datetime.fromtimestamp(trade['T'] / 1000)
                print(f"     {i+1}. Time: {trade_time}, Price: {trade['p']}, Qty: {trade['q']}")
        else:
            print("🔍 DEBUG: No trades returned")
            
    except Exception as e:
        print(f"🔍 DEBUG: Exception occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 DEBUG: Starting pagination test script")
    asyncio.run(test_mexc_pagination())
    print("🔍 DEBUG: Pagination test script completed") 