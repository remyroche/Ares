#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exchange.mexc import MexcExchange
from datetime import datetime, timedelta

async def test_mexc_real_data():
    """Test what real historical data MEXC can provide"""
    print("🔍 Testing MEXC real historical data availability...")
    
    # Initialize with dummy credentials for testing
    exchange = MexcExchange(api_key="test", api_secret="test", trade_symbol="ETHUSDT")
    
    # Test different time periods
    test_periods = [
        ("Recent (1 week ago)", datetime.now() - timedelta(days=7)),
        ("1 month ago", datetime.now() - timedelta(days=30)),
        ("3 months ago", datetime.now() - timedelta(days=90)),
        ("6 months ago", datetime.now() - timedelta(days=180)),
        ("1 year ago", datetime.now() - timedelta(days=365)),
        ("2 years ago", datetime.now() - timedelta(days=730)),
    ]
    
    for period_name, test_date in test_periods:
        start_ms = int(test_date.timestamp() * 1000)
        end_ms = start_ms + (24 * 60 * 60 * 1000)  # 24 hours
        
        print(f"\n📊 Testing {period_name} ({test_date.strftime('%Y-%m-%d')}):")
        
        # Test klines
        try:
            klines = await exchange.get_historical_klines('ETHUSDT', '1m', start_ms, end_ms)
            print(f"   ✅ Klines: {len(klines)} records")
        except Exception as e:
            print(f"   ❌ Klines error: {e}")
        
        # Test aggregated trades
        try:
            aggtrades = await exchange.get_historical_agg_trades('ETHUSDT', start_ms, end_ms)
            print(f"   ✅ Aggregated trades: {len(aggtrades)} records")
        except Exception as e:
            print(f"   ❌ Aggregated trades error: {e}")
    
    await exchange.close()
    print("\n✅ MEXC real data test completed")

if __name__ == "__main__":
    asyncio.run(test_mexc_real_data()) 