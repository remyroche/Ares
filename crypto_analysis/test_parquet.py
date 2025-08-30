#!/usr/bin/env python3
"""
Test script to verify Parquet functionality and demonstrate data storage
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

def test_parquet_functionality():
    """Test Parquet read/write functionality"""
    print("🧪 Testing Parquet Functionality")
    print("=" * 50)
    
    # Create sample data similar to our crypto data
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='15T')
    symbols = ['ETHUSDT', 'BTCUSDT', 'ADAUSDT']
    
    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'open_time': date,
                'open': np.random.uniform(100, 50000),
                'high': np.random.uniform(100, 50000),
                'low': np.random.uniform(100, 50000),
                'close': np.random.uniform(100, 50000),
                'volume': np.random.uniform(1000, 100000),
                'quote_asset_volume': np.random.uniform(100000, 10000000),
                'number_of_trades': np.random.randint(100, 10000),
                'symbol': symbol
            })
    
    df = pd.DataFrame(data)
    df.set_index('open_time', inplace=True)
    
    print(f"Created test data: {len(df):,} records")
    print(f"Data types: {df.dtypes.to_dict()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Test Parquet write/read
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        parquet_path = tmp_file.name
    
    try:
        # Write to Parquet
        print(f"\n📝 Writing to Parquet: {parquet_path}")
        df.to_parquet(parquet_path, compression='snappy', engine='pyarrow', index=True)
        
        # Check file size
        file_size = os.path.getsize(parquet_path) / 1024 / 1024
        print(f"Parquet file size: {file_size:.2f} MB")
        print(f"Compression ratio: {df.memory_usage(deep=True).sum() / (file_size * 1024 * 1024):.1f}x")
        
        # Read from Parquet
        print(f"\n📖 Reading from Parquet...")
        loaded_df = pd.read_parquet(parquet_path)
        
        # Verify data integrity
        print(f"Loaded data: {len(loaded_df):,} records")
        print(f"Data types match: {df.dtypes.equals(loaded_df.dtypes)}")
        print(f"Data content match: {df.equals(loaded_df)}")
        
        # Test performance
        print(f"\n⚡ Performance test:")
        import time
        
        start_time = time.time()
        df.to_parquet(parquet_path, compression='snappy', engine='pyarrow', index=True)
        write_time = time.time() - start_time
        
        start_time = time.time()
        loaded_df = pd.read_parquet(parquet_path)
        read_time = time.time() - start_time
        
        print(f"Write time: {write_time:.3f} seconds")
        print(f"Read time: {read_time:.3f} seconds")
        
        # Test with different compression
        print(f"\n🗜️  Compression comparison:")
        compressions = ['snappy', 'gzip', 'brotli']
        
        for compression in compressions:
            test_file = f"test_{compression}.parquet"
            start_time = time.time()
            df.to_parquet(test_file, compression=compression, engine='pyarrow', index=True)
            write_time = time.time() - start_time
            
            file_size = os.path.getsize(test_file) / 1024 / 1024
            compression_ratio = df.memory_usage(deep=True).sum() / (file_size * 1024 * 1024)
            
            print(f"  {compression:8}: {file_size:6.2f} MB, {compression_ratio:5.1f}x compression, {write_time:.3f}s")
            
            # Clean up test file
            os.remove(test_file)
        
        print(f"\n✅ Parquet functionality test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing Parquet: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(parquet_path):
            os.remove(parquet_path)
    
    return True

def test_data_structure():
    """Test the expected data structure for our crypto analysis"""
    print(f"\n📊 Testing Expected Data Structure")
    print("=" * 50)
    
    # Create sample data with our expected structure
    dates = pd.date_range('2023-01-01', '2023-01-02', freq='15T')
    symbols = ['ETHUSDT', 'BTCUSDT']
    
    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'open_time': date,
                'open': 2000.0 + np.random.normal(0, 50),
                'high': 2000.0 + np.random.normal(0, 50),
                'low': 2000.0 + np.random.normal(0, 50),
                'close': 2000.0 + np.random.normal(0, 50),
                'volume': 1000.0 + np.random.normal(0, 200),
                'quote_asset_volume': 2000000.0 + np.random.normal(0, 100000),
                'number_of_trades': np.random.randint(100, 1000),
                'symbol': symbol
            })
    
    df = pd.DataFrame(data)
    df.set_index('open_time', inplace=True)
    
    # Ensure proper data types
    df['number_of_trades'] = df['number_of_trades'].astype('int64')
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']:
        df[col] = df[col].astype('float64')
    
    print("Expected columns and data types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col:20}: {dtype}")
    
    print(f"\nSample data:")
    print(df.head())
    
    print(f"\nData info:")
    print(f"  Records: {len(df):,}")
    print(f"  Symbols: {df['symbol'].nunique()}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    return True

if __name__ == "__main__":
    print("🚀 Parquet Functionality Test Suite")
    print("=" * 60)
    
    # Run tests
    test_parquet_functionality()
    test_data_structure()
    
    print(f"\n🎉 All tests completed!")
    print(f"\n💡 The data downloader will save data in this exact format.")
    print(f"   - Efficient Parquet compression (typically 3-5x smaller than CSV)")
    print(f"   - Fast read/write performance")
    print(f"   - Preserves data types and datetime index")
    print(f"   - Compatible with pandas, Apache Arrow, and other tools")