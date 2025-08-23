#!/usr/bin/env python3
"""
Quick Test for Mock Data Generation

This script tests the mock data generation functionality to ensure it works correctly
before running the full pipeline tests.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_mock_data_generation():
    """Test mock data generation functionality."""
    print("🧪 Testing Mock Data Generation")
    print("=" * 50)
    
    # Create data directories
    data_cache_path = Path("data_cache")
    data_cache_path.mkdir(parents=True, exist_ok=True)
    
    # Generate 7 days of data for quick testing
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Generate klines data
    print("📊 Generating klines data...")
    klines_timestamps = pd.date_range(start=start_date, end=end_date, freq='1min')
    klines_data = []
    
    np.random.seed(42)
    base_price = 3000.0
    price = base_price
    
    for timestamp in klines_timestamps:
        price_change = np.random.normal(0, 0.001)
        price = max(price * (1 + price_change), 100)
        
        spread = price * 0.0005
        open_price = price + np.random.uniform(-spread, spread)
        high_price = max(open_price, price + np.random.uniform(0, spread))
        low_price = min(open_price, price - np.random.uniform(0, spread))
        close_price = price + np.random.uniform(-spread, spread)
        volume = np.random.uniform(10, 1000)
        
        klines_data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2),
            'quote_asset_volume': round(volume * close_price, 2),
            'number_of_trades': np.random.randint(1, 100),
            'taker_buy_base_asset_volume': round(volume * 0.6, 2),
            'taker_buy_quote_asset_volume': round(volume * close_price * 0.6, 2),
        })
    
    klines_df = pd.DataFrame(klines_data)
    print(f"✅ Generated {len(klines_df)} klines records")
    
    # Generate aggtrades data
    print("📊 Generating aggtrades data...")
    aggtrades_timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
    aggtrades_data = []
    
    for timestamp in aggtrades_timestamps:
        num_trades = np.random.randint(1, 10)
        for _ in range(num_trades):
            trade_price = base_price + np.random.normal(0, 50)
            quantity = np.random.uniform(0.1, 10.0)
            
            aggtrades_data.append({
                'timestamp': timestamp,
                'price': round(trade_price, 2),
                'quantity': round(quantity, 4),
                'first_trade_id': np.random.randint(1000000, 9999999),
                'last_trade_id': np.random.randint(1000000, 9999999),
                'trade_time': int(timestamp.timestamp() * 1000),
                'is_buyer_maker': np.random.choice([True, False]),
            })
    
    aggtrades_df = pd.DataFrame(aggtrades_data)
    print(f"✅ Generated {len(aggtrades_df)} aggtrades records")
    
    # Generate futures data
    print("📊 Generating futures data...")
    futures_timestamps = pd.date_range(start=start_date, end=end_date, freq='1min')
    futures_data = []
    
    for timestamp in futures_timestamps:
        mark_price = base_price + np.random.normal(0, 30)
        funding_rate = np.random.uniform(-0.001, 0.001)
        
        futures_data.append({
            'timestamp': timestamp,
            'symbol': 'ETHUSDT',
            'mark_price': round(mark_price, 2),
            'index_price': round(mark_price + np.random.normal(0, 5), 2),
            'funding_rate': round(funding_rate, 6),
            'next_funding_time': int((timestamp + timedelta(hours=8)).timestamp() * 1000),
        })
    
    futures_df = pd.DataFrame(futures_data)
    print(f"✅ Generated {len(futures_df)} futures records")
    
    # Save files
    print("💾 Saving data files...")
    
    klines_file = data_cache_path / "klines_BINANCE_ETHUSDT_1m_consolidated.parquet"
    klines_df.to_parquet(klines_file, index=False)
    print(f"✅ Saved klines: {klines_file}")
    
    aggtrades_file = data_cache_path / "aggtrades_BINANCE_ETHUSDT_consolidated.parquet"
    aggtrades_df.to_parquet(aggtrades_file, index=False)
    print(f"✅ Saved aggtrades: {aggtrades_file}")
    
    futures_file = data_cache_path / "futures_BINANCE_ETHUSDT_consolidated.parquet"
    futures_df.to_parquet(futures_file, index=False)
    print(f"✅ Saved futures: {futures_file}")
    
    # Verify files exist and have content
    print("\n🔍 Verifying generated files...")
    
    files_to_check = [
        klines_file,
        aggtrades_file,
        futures_file
    ]
    
    all_files_exist = True
    for file_path in files_to_check:
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"✅ {file_path.name}: {file_size} bytes")
        else:
            print(f"❌ {file_path.name}: File not found")
            all_files_exist = False
    
    # Show data summary
    print("\n📊 Data Summary:")
    print(f"Klines: {len(klines_df)} records, {len(klines_df.columns)} columns")
    print(f"Aggtrades: {len(aggtrades_df)} records, {len(aggtrades_df.columns)} columns")
    print(f"Futures: {len(futures_df)} records, {len(futures_df.columns)} columns")
    
    print(f"\nKlines columns: {list(klines_df.columns)}")
    print(f"Aggtrades columns: {list(aggtrades_df.columns)}")
    print(f"Futures columns: {list(futures_df.columns)}")
    
    if all_files_exist:
        print("\n🎉 Mock data generation test PASSED!")
        return True
    else:
        print("\n💥 Mock data generation test FAILED!")
        return False

if __name__ == "__main__":
    success = test_mock_data_generation()
    sys.exit(0 if success else 1)