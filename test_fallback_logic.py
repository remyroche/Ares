#!/usr/bin/env python3
"""
Test script to verify the updated fallback logic:
ETHUSDT is primary, BTCUSDT is secondary
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.parquet_utils import ParquetUtils

def test_fallback_logic():
    """Test that the updated fallback logic works correctly."""
    print("🧪 Testing updated fallback logic...")
    print("Expected behavior: ETHUSDT (primary) -> BTCUSDT (secondary)")

    exchange = 'binance'
    parquet_utils = ParquetUtils()

    # Test 1: ETHUSDT should be available (primary)
    eth_path = f'data/training/unified/{exchange.upper()}/ETHUSDT/1m/exchange={exchange.upper()}/symbol=ETHUSDT/timeframe=1m'
    print(f"\n🔍 Testing ETHUSDT (primary): {eth_path}")
    print(f"📁 ETHUSDT path exists: {os.path.exists(eth_path)}")

    if os.path.exists(eth_path):
        data = parquet_utils.safe_read_parquet(eth_path)
        if data is not None and not data.empty:
            print(f"✅ ETHUSDT loaded successfully: {data.shape}")
        else:
            print("❌ ETHUSDT exists but failed to load")
    else:
        print("❌ ETHUSDT path does not exist")

    # Test 2: BTCUSDT should not be available (secondary fallback)
    btc_path = f'data/training/unified/{exchange.upper()}/BTCUSDT/1m/exchange={exchange.upper()}/symbol=BTCUSDT/timeframe=1m'
    print(f"\n🔍 Testing BTCUSDT (secondary): {btc_path}")
    print(f"📁 BTCUSDT path exists: {os.path.exists(btc_path)}")

    if os.path.exists(btc_path):
        data = parquet_utils.safe_read_parquet(btc_path)
        if data is not None and not data.empty:
            print(f"✅ BTCUSDT loaded successfully: {data.shape}")
        else:
            print("❌ BTCUSDT exists but failed to load")
    else:
        print("❌ BTCUSDT path does not exist (expected - this is the fallback)")

    # Test 3: Simulate the logic flow
    print("\n🔄 Simulating fallback logic flow:")
    data = None

    # Step 1: Try ETHUSDT (primary)
    try:
        data = parquet_utils.safe_read_parquet(eth_path)
        if data is not None and not data.empty:
            print(f"✅ Step 1: ETHUSDT loaded successfully ({len(data)} rows) - PRIMARY SUCCESS")
        else:
            print("⚠️ Step 1: ETHUSDT returned empty data")
    except Exception as e:
        print(f"⚠️ Step 1: ETHUSDT failed to load: {e}")

    # Step 2: Try BTCUSDT if ETHUSDT failed
    if data is None or data.empty:
        try:
            data = parquet_utils.safe_read_parquet(btc_path)
            if data is not None and not data.empty:
                print(f"✅ Step 2: BTCUSDT loaded successfully ({len(data)} rows) - SECONDARY FALLBACK SUCCESS")
            else:
                print("❌ Step 2: BTCUSDT also failed or returned empty data")
        except Exception as e:
            print(f"❌ Step 2: BTCUSDT failed to load: {e}")

    print("\n📋 Summary:")
    print("- ETHUSDT is the PRIMARY data source")
    print("- BTCUSDT is the SECONDARY fallback")
    print("- This ensures the most reliable data is used first")

if __name__ == "__main__":
    test_fallback_logic()
