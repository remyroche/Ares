import pandas as pd
import numpy as np
from pathlib import Path

def debug_aggregation_process():
    """Debug the exact aggregation process that's failing."""

    # Load the raw aggregated trades data
    aggtrades_file = Path("/Users/remyroche/Documents/Ares/data/training/aggtrades_binance_ETHUSDT_raw.parquet")

    if not aggtrades_file.exists():
        print(f"❌ File not found: {aggtrades_file}")
        return

    data = pd.read_parquet(aggtrades_file)
    print(f"🔍 Loaded {len(data)} raw aggregated trades")
    print(f"📊 Columns: {list(data.columns)}")

    # Check if we have kline_timestamp column (needed for aggregation)
    if 'kline_timestamp' not in data.columns:
        print("❌ Missing 'kline_timestamp' column needed for aggregation!")

        # Check what timestamp columns we do have
        timestamp_cols = [col for col in data.columns if 'timestamp' in col.lower()]
        print(f"Available timestamp columns: {timestamp_cols}")

        # Maybe we need to use 'timestamp' as 'kline_timestamp'
        if 'timestamp' in data.columns:
            print("🔧 Using 'timestamp' as 'kline_timestamp' for aggregation")
            data['kline_timestamp'] = data['timestamp']
    else:
        print("✅ Found 'kline_timestamp' column")

    # Check what data is actually in the aggregation columns
    print("\n🔍 Pre-aggregation data check:")    sample_data = data.head(20)

    # Check if quantity and price columns have real data
    if 'quantity' in data.columns:
        print(f"quantity stats: min={data['quantity'].min():.6f}, max={data['quantity'].max():.6f}, mean={data['quantity'].mean():.6f}")
        print(f"quantity non-zero count: {(data['quantity'] != 0).sum()}")

    if 'price' in data.columns:
        print(f"price stats: min={data['price'].min():.6f}, max={data['price'].max():.6f}, mean={data['price'].mean():.6f}")
        print(f"price non-zero count: {(data['price'] != 0).sum()}")

    if 'close' in data.columns:
        print(f"close stats: min={data['close'].min():.6f}, max={data['close'].max():.6f}, mean={data['close'].mean():.6f}")
        print(f"close non-zero count: {(data['close'] != 0).sum()}")

    # Show sample of raw data before aggregation
    print("\n🔍 Sample raw data before aggregation:")    print(sample_data[['timestamp', 'kline_timestamp' if 'kline_timestamp' in sample_data.columns else 'timestamp', 'quantity', 'price' if 'price' in sample_data.columns else 'close']].head(10))

    # Now try the exact aggregation logic from the data converter
    print("\n🔧 Attempting aggregation...")    try:
        # This is the exact logic from the data converter
        if 'price' in data.columns:
            basic_stats = data.groupby('kline_timestamp').agg({
                'quantity': 'sum',
                'price': ['mean', 'min', 'max']
            }).reset_index()
        else:
            basic_stats = data.groupby('kline_timestamp').agg({
                'quantity': 'sum',
                'close': ['mean', 'min', 'max']
            }).reset_index()

        # Flatten column names
        if 'price' in data.columns:
            basic_stats.columns = ['timestamp', 'trade_volume', 'avg_price', 'min_price', 'max_price']
        else:
            basic_stats.columns = ['timestamp', 'trade_volume', 'avg_price', 'min_price', 'max_price']

        # Add trade_count
        trade_counts = data.groupby('kline_timestamp').size().reset_index(name='trade_count')
        basic_stats = basic_stats.merge(trade_counts, on='timestamp')

        print(f"✅ Aggregation successful: {len(basic_stats)} aggregated timestamps")

        # Check the results
        print("
📊 Aggregation results:"        print(f"trade_volume stats: min={basic_stats['trade_volume'].min():.6f}, max={basic_stats['trade_volume'].max():.6f}, mean={basic_stats['trade_volume'].mean():.6f}")
        print(f"trade_count stats: min={basic_stats['trade_count'].min()}, max={basic_stats['trade_count'].max()}, mean={basic_stats['trade_count'].mean():.1f}")
        print(f"avg_price stats: min={basic_stats['avg_price'].min():.6f}, max={basic_stats['avg_price'].max():.6f}, mean={basic_stats['avg_price'].mean():.6f}")

        # Show sample of aggregated data
        print("
🔍 Sample aggregated data:"        print(basic_stats.head(10))

    except Exception as e:
        print(f"❌ Aggregation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_aggregation_process()
