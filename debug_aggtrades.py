import pandas as pd
import numpy as np
from pathlib import Path

def analyze_aggtrades_data():
    """Analyze the raw aggregated trades data to find the root cause."""

    # Check the problematic file
    aggtrades_file = Path("/Users/remyroche/Documents/Ares/data/training/aggtrades_binance_ETHUSDT_raw.parquet")

    if not aggtrades_file.exists():
        print(f"❌ File not found: {aggtrades_file}")
        return

    print(f"🔍 Analyzing aggregated trades file: {aggtrades_file}")

    try:
        # Read the data
        data = pd.read_parquet(aggtrades_file)
        print(f"✅ Loaded data with {len(data)} rows and {len(data.columns)} columns")
        print(f"📊 Columns: {list(data.columns)}")

        # Check basic statistics
        print(f"\n📈 Basic Statistics:")
        print(f"Timestamp range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"Unique timestamps: {data['timestamp'].nunique()}")

        # Check trade-related columns
        trade_cols = ['trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price', 'quantity']
        print(f"\n🔧 Trade-related columns analysis:")

        for col in trade_cols:
            if col in data.columns:
                unique_vals = data[col].nunique()
                std_val = data[col].std()
                min_val = data[col].min()
                max_val = data[col].max()
                non_zero_count = (data[col] != 0).sum()

                print(f"{col:15} | Unique: {unique_vals:6} | Std: {std_val:.2e} | Min: {min_val:.2e} | Max: {max_val:.2e} | Non-zero: {non_zero_count}")
            else:
                print(f"{col:15} | NOT FOUND")

        # Check for empty/null data
        print(f"\n🔍 Data quality check:")
        null_counts = data.isnull().sum()
        if null_counts.any():
            print("Null values found:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"  {col}: {count} nulls")

        zero_counts = (data == 0).sum()
        if zero_counts.any():
            print("Zero values found:")
            for col, count in zero_counts[zero_counts > 0].items():
                pct_zero = (count / len(data)) * 100
                print(".1f")

        # Check if this is aggregated data or raw trades
        if 'kline_timestamp' in data.columns:
            print(f"\n📊 This appears to be aggregated data:")
            print(f"Unique kline_timestamps: {data['kline_timestamp'].nunique()}")
            print(f"Kline timestamp range: {data['kline_timestamp'].min()} to {data['kline_timestamp'].max()}")

            # Check aggregation quality
            agg_stats = data.groupby('kline_timestamp').agg({
                'quantity': ['count', 'sum'],
                'price': ['mean', 'min', 'max'] if 'price' in data.columns else []
            })
            print(f"Aggregation results: {len(agg_stats)} unique timestamps")
            if len(agg_stats) > 0:
                print(f"Trades per timestamp: min={agg_stats[('quantity', 'count')].min()}, max={agg_stats[('quantity', 'count')].max()}, mean={agg_stats[('quantity', 'count')].mean():.1f}")

        # Sample some rows
        print(f"\n🔍 Sample rows:")
        print(data.head(10).to_string())

    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_aggtrades_data()
