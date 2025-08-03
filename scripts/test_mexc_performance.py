#!/usr/bin/env python3
"""
Performance test script to compare original vs optimized MEXC implementation.
"""

import asyncio
import sys
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from exchange.mexc_updated import MexcExchangeUpdated
from exchange.mexc_optimized import MexcExchangeOptimized

async def test_performance():
    """Test performance of both implementations."""
    print("🔍 DEBUG: Starting MEXC performance test")
    
    # Test parameters - 6 hours for comparison
    symbol = "ETHUSDT"
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=6)  # 6 hours for testing
    start_time_ms = int(start_time.timestamp() * 1000)
    end_time_ms = int(end_time.timestamp() * 1000)
    
    print(f"🔍 DEBUG: Test parameters:")
    print(f"   Symbol: {symbol}")
    print(f"   Start time: {start_time}")
    print(f"   End time: {end_time}")
    print(f"   Time range: {(end_time - start_time).total_seconds() / 3600:.1f} hours")
    
    # Test original implementation
    print("\n🔍 DEBUG: Testing ORIGINAL implementation...")
    original_client = MexcExchangeUpdated("", "", "ETHUSDT")
    
    start_time_orig = time.time()
    original_trades = await original_client.get_historical_agg_trades(
        symbol=symbol,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        limit=1000
    )
    end_time_orig = time.time()
    
    original_duration = end_time_orig - start_time_orig
    print(f"🔍 DEBUG: Original implementation:")
    print(f"   Duration: {original_duration:.2f} seconds")
    print(f"   Trades: {len(original_trades)}")
    print(f"   Trades per second: {len(original_trades) / original_duration:.1f}")
    
    await original_client.close()
    
    # Test optimized implementation
    print("\n🔍 DEBUG: Testing OPTIMIZED implementation...")
    optimized_client = MexcExchangeOptimized("", "", "ETHUSDT")
    
    start_time_opt = time.time()
    optimized_trades = await optimized_client.get_historical_agg_trades(
        symbol=symbol,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        limit=1000
    )
    end_time_opt = time.time()
    
    optimized_duration = end_time_opt - start_time_opt
    print(f"🔍 DEBUG: Optimized implementation:")
    print(f"   Duration: {optimized_duration:.2f} seconds")
    print(f"   Trades: {len(optimized_trades)}")
    print(f"   Trades per second: {len(optimized_trades) / optimized_duration:.1f}")
    
    await optimized_client.close()
    
    # Performance comparison
    speedup = original_duration / optimized_duration if optimized_duration > 0 else 0
    print(f"\n🔍 DEBUG: Performance comparison:")
    print(f"   Speedup: {speedup:.2f}x faster")
    print(f"   Time saved: {original_duration - optimized_duration:.2f} seconds")
    print(f"   Efficiency improvement: {((original_duration - optimized_duration) / original_duration * 100):.1f}%")
    
    # Data validation
    print(f"\n🔍 DEBUG: Data validation:")
    print(f"   Original trades: {len(original_trades)}")
    print(f"   Optimized trades: {len(optimized_trades)}")
    print(f"   Trade count difference: {abs(len(original_trades) - len(optimized_trades))}")
    
    if len(original_trades) > 0 and len(optimized_trades) > 0:
        print(f"   Sample original trade: {original_trades[0]}")
        print(f"   Sample optimized trade: {optimized_trades[0]}")
        print(f"   Format matches: {original_trades[0].keys() == optimized_trades[0].keys()}")

if __name__ == "__main__":
    print("🔍 DEBUG: Starting performance test script")
    asyncio.run(test_performance())
    print("🔍 DEBUG: Performance test script completed") 