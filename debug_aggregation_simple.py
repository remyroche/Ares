import pandas as pd
import numpy as np
from pathlib import Path

def debug_aggregation():
    """Simple debug of aggregation process."""

    aggtrades_file = Path("/Users/remyroche/Documents/Ares/data/training/aggtrades_binance_ETHUSDT_raw.parquet")

    if not aggtrades_file.exists():
        print(f"❌ File not found: {aggtrades_file}")
        return

    data = pd.read_parquet(aggtrades_file)
    print(f"🔍 Loaded {len(data)} raw trades")
    print(f"📊 Columns: {list(data.columns)}")

    # Check if we have the right columns
    print("\n🔍 Checking key columns:")
    print(f"Has 'kline_timestamp': {'kline_timestamp' in data.columns}")
    print(f"Has 'price': {'price' in data.columns}")
    print(f"Has 'close': {'close' in data.columns}")
    print(f"Has 'quantity': {'quantity' in data.columns}")

    # Check data quality
    print("\n📊 Data quality:")
    if 'quantity' in data.columns:
        print(f"quantity > 0: {(data['quantity'] > 0).sum()}")
        print(f"quantity == 0: {(data['quantity'] == 0).sum()}")

    if 'close' in data.columns:
        print(f"close > 0: {(data['close'] > 0).sum()}")
        print(f"close == 0: {(data['close'] == 0).sum()}")

    # Show sample
    print("\n🔍 Sample data:")
    sample_cols = ['timestamp']
    if 'kline_timestamp' in data.columns:
        sample_cols.append('kline_timestamp')
    if 'quantity' in data.columns:
        sample_cols.append('quantity')
    if 'close' in data.columns:
        sample_cols.append('close')

    print(data[sample_cols].head(10))

    # Try simple aggregation
    print("\n🔧 Testing aggregation...")

    # For this data, we need to group by timestamp since there's no kline_timestamp
    if 'timestamp' in data.columns and 'quantity' in data.columns and 'close' in data.columns:
        try:
            agg_result = data.groupby('timestamp').agg({
                'quantity': 'sum',
                'close': ['mean', 'min', 'max']
            }).reset_index()

            agg_result.columns = ['timestamp', 'trade_volume', 'avg_price', 'min_price', 'max_price']

            trade_counts = data.groupby('timestamp').size().reset_index(name='trade_count')
            agg_result = agg_result.merge(trade_counts, on='timestamp')

            print(f"✅ Aggregation successful: {len(agg_result)} groups")

            print("\n📊 Aggregation results:")
            print(f"trade_volume range: {agg_result['trade_volume'].min():.6f} - {agg_result['trade_volume'].max():.6f}")
            print(f"trade_count range: {agg_result['trade_count'].min()} - {agg_result['trade_count'].max()}")
            print(f"avg_price range: {agg_result['avg_price'].min():.6f} - {agg_result['avg_price'].max():.6f}")

            # Check if results are still zero
            zero_trade_volume = (agg_result['trade_volume'] == 0).sum()
            zero_trade_count = (agg_result['trade_count'] == 0).sum()

            print(f"\n❌ Zero results check:")
            print(f"Zero trade_volume: {zero_trade_volume}/{len(agg_result)}")
            print(f"Zero trade_count: {zero_trade_count}/{len(agg_result)}")

            if zero_trade_volume > 0 or zero_trade_count > 0:
                print("🚨 AGGREGATION IS PRODUCING ZEROS - THIS IS THE BUG!")
            else:
                print("✅ Aggregation producing non-zero results")

        except Exception as e:
            print(f"❌ Aggregation failed: {e}")

if __name__ == "__main__":
    debug_aggregation()
