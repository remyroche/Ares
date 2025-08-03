#!/usr/bin/env python3
"""
Test MEXC aggregated trades for a longer time range (2 years).
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from exchange.factory import ExchangeFactory

async def test_mexc_longer_range():
    """Test MEXC aggregated trades for a longer time range."""
    print("🔍 DEBUG: Starting MEXC longer range test")
    
    try:
        # Create MEXC client
        print("🔍 DEBUG: Creating MEXC client...")
        client = ExchangeFactory.get_exchange('mexc')
        
        # Test parameters - 2 years back
        symbol = "ETHUSDT"
        end_time = datetime.now()
        start_time = end_time - timedelta(days=730)  # 2 years
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
        print("🔍 DEBUG: Calling get_historical_agg_trades for 2 years...")
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
            print(f"   Data spans: {(last_time - first_time).days} days")
        else:
            print("🔍 DEBUG: No trades returned")
            
    except Exception as e:
        print(f"🔍 DEBUG: Exception occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 DEBUG: Starting longer range test script")
    asyncio.run(test_mexc_longer_range())
    print("🔍 DEBUG: Longer range test script completed") 