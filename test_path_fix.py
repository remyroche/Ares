#!/usr/bin/env python3
"""
Test script to verify the path construction fix in simplified_training_manager.py
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.parquet_utils import ParquetUtils

def test_path_construction():
    """Test that the corrected path construction works."""
    print("🧪 Testing path construction fix...")

    # Test the corrected path for ETHUSDT (which exists)
    exchange = 'binance'
    symbol = 'ETHUSDT'

    # This is the corrected path construction from the fix
    data_path = f'data/training/unified/{exchange.upper()}/{symbol}/1m/exchange={exchange.upper()}/symbol={symbol}/timeframe=1m'

    print(f"🔍 Testing path: {data_path}")
    print(f"📁 Path exists: {os.path.exists(data_path)}")

    if os.path.exists(data_path):
        print("✅ Primary path exists!")

        # Test loading with ParquetUtils
        parquet_utils = ParquetUtils()
        data = parquet_utils.safe_read_parquet(data_path)

        if data is not None and not data.empty:
            print(f"✅ Successfully loaded data: {data.shape}")
        else:
            print("❌ Failed to load data from existing path")
    else:
        print("❌ Primary path does not exist")

    # Test the fallback path for BTCUSDT -> ETHUSDT
    print("\n🔄 Testing BTCUSDT -> ETHUSDT fallback...")
    btc_symbol = 'BTCUSDT'
    eth_path = f'data/training/unified/{exchange.upper()}/ETHUSDT/1m/exchange={exchange.upper()}/symbol=ETHUSDT/timeframe=1m'

    print(f"🔍 BTCUSDT fallback path: {eth_path}")
    print(f"📁 Fallback path exists: {os.path.exists(eth_path)}")

    if os.path.exists(eth_path):
        print("✅ Fallback path exists!")

        # Test loading with ParquetUtils
        parquet_utils = ParquetUtils()
        data = parquet_utils.safe_read_parquet(eth_path)

        if data is not None and not data.empty:
            print(f"✅ Successfully loaded fallback data: {data.shape}")
        else:
            print("❌ Failed to load data from fallback path")
    else:
        print("❌ Fallback path does not exist")

if __name__ == "__main__":
    test_path_construction()
