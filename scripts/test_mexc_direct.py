#!/usr/bin/env python3
"""
Direct test of MEXC aggregated trades functionality with print statements.
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

async def test_mexc_agg_trades():
    """Test MEXC aggregated trades directly."""
    print("🔍 DEBUG: Starting MEXC test")
    
    try:
        # Create MEXC client
        print("🔍 DEBUG: Creating MEXC client...")
        client = ExchangeFactory.get_exchange('mexc')
        print(f"🔍 DEBUG: Client type: {type(client).__name__}")
        print(f"🔍 DEBUG: Client methods: {[m for m in dir(client) if not m.startswith('_')]}")
        
        # Test parameters
        symbol = "ETHUSDT"
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)  # Just 1 hour for testing
        start_time_ms = int(start_time.timestamp() * 1000)
        end_time_ms = int(end_time.timestamp() * 1000)
        
        print(f"🔍 DEBUG: Test parameters:")
        print(f"   Symbol: {symbol}")
        print(f"   Start time: {start_time}")
        print(f"   End time: {end_time}")
        print(f"   Start time ms: {start_time_ms}")
        print(f"   End time ms: {end_time_ms}")
        
        # Test the method
        print("🔍 DEBUG: Calling get_historical_agg_trades...")
        trades = await client.get_historical_agg_trades(
            symbol=symbol,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            limit=1000
        )
        
        print(f"🔍 DEBUG: Method completed")
        print(f"🔍 DEBUG: Returned {len(trades)} trades")
        
        if trades:
            print(f"🔍 DEBUG: Sample trade: {trades[0]}")
            print(f"🔍 DEBUG: Trade keys: {list(trades[0].keys())}")
        else:
            print("🔍 DEBUG: No trades returned")
            
    except Exception as e:
        print(f"🔍 DEBUG: Exception occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 DEBUG: Starting test script")
    asyncio.run(test_mexc_agg_trades())
    print("🔍 DEBUG: Test script completed") 