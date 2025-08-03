#!/usr/bin/env python3
"""
Test MEXC aggregated trades for different time periods to see trade counts.
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

async def test_mexc_trade_counts():
    """Test MEXC aggregated trades for different time periods."""
    print("🔍 DEBUG: Starting MEXC trade count test")
    
    try:
        # Create MEXC client
        print("🔍 DEBUG: Creating MEXC client...")
        client = ExchangeFactory.get_exchange('mexc')
        
        symbol = "ETHUSDT"
        end_time = datetime.now()
        
        # Test different time periods
        time_periods = [
            ("1 hour", timedelta(hours=1)),
            ("6 hours", timedelta(hours=6)),
            ("12 hours", timedelta(hours=12)),
            ("1 day", timedelta(days=1)),
            ("3 days", timedelta(days=3)),
            ("1 week", timedelta(weeks=1)),
        ]
        
        for period_name, delta in time_periods:
            start_time = end_time - delta
            start_time_ms = int(start_time.timestamp() * 1000)
            end_time_ms = int(end_time.timestamp() * 1000)
            
            print(f"\n🔍 DEBUG: Testing {period_name} period:")
            print(f"   Start time: {start_time}")
            print(f"   End time: {end_time}")
            print(f"   Duration: {delta}")
            
            try:
                trades = await client.get_historical_agg_trades(
                    symbol=symbol,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                    limit=1000
                )
                
                print(f"   ✅ Got {len(trades)} trades")
                
                if trades:
                    first_time = datetime.fromtimestamp(trades[0]['T'] / 1000)
                    last_time = datetime.fromtimestamp(trades[-1]['T'] / 1000)
                    print(f"   📊 Data time range: {first_time} to {last_time}")
                    print(f"   📊 Actual data span: {(last_time - first_time).total_seconds() / 3600:.1f} hours")
                    
                    # Calculate trades per hour
                    hours_span = (last_time - first_time).total_seconds() / 3600
                    trades_per_hour = len(trades) / hours_span if hours_span > 0 else 0
                    print(f"   📈 Trades per hour: {trades_per_hour:.1f}")
                    
                    # Show sample trades
                    print(f"   📋 Sample trades:")
                    for i, trade in enumerate(trades[:3]):
                        trade_time = datetime.fromtimestamp(trade['T'] / 1000)
                        print(f"     {i+1}. Time: {trade_time}, Price: {trade['p']}, Qty: {trade['q']}")
                else:
                    print(f"   ⚠️ No trades returned")
                    
            except Exception as e:
                print(f"   ❌ Error: {type(e).__name__}: {e}")
                
    except Exception as e:
        print(f"🔍 DEBUG: Exception occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 DEBUG: Starting trade count test script")
    asyncio.run(test_mexc_trade_counts())
    print("🔍 DEBUG: Trade count test script completed") 