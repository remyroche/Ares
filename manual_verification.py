import pandas as pd
import numpy as np
from pathlib import Path

def check_constant_features(file_path):
    """Check for constant features in the data file."""
    print(f"🔍 Checking file: {file_path}")

    try:
        # Read the parquet file
        data = pd.read_parquet(file_path)
        print(f"✅ Loaded data with {len(data)} rows and {len(data.columns)} columns")

        # Show all columns first
        print("\n📋 All columns in the data:")
        print(f"Total columns: {len(data.columns)}")
        print("First 20 columns:", list(data.columns[:20]))
        print("Last 20 columns:", list(data.columns[-20:]))

        # Check specific columns mentioned in the error
        trade_stat_cols = ['trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price', 'price_std']
        funding_cols = ['funding_rate']

        print("\n📊 Checking trade and funding features:")
        print("=" * 60)

        constant_features = []

        for col in trade_stat_cols + funding_cols:
            if col in data.columns:
                unique_vals = data[col].nunique()
                std_val = data[col].std()
                min_val = data[col].min()
                max_val = data[col].max()

                print(f"{col:15} | Unique: {unique_vals:6} | Std: {std_val:.2e} | Min: {min_val:.6f} | Max: {max_val:.6f}")

                # Check if constant (same logic as the sub_pipeline)
                if unique_vals <= 2 or (not pd.isna(std_val) and std_val < 1e-10):
                    constant_features.append(f"{col}(unique={unique_vals}, std={std_val:.2e})")
                    print(f"  🚨 CONSTANT: {col}")
                else:
                    print(f"  ✅ Variable: {col}")
            else:
                print(f"{col:15} | NOT FOUND")

        print(f"\n📋 Constant features detected: {constant_features}")

        # Show some sample values for constant features
        if constant_features:
            print("\n🔍 Sample values for constant features:")
            for feature_desc in constant_features:
                col_name = feature_desc.split('(')[0]
                if col_name in data.columns:
                    sample_values = data[col_name].dropna().head(10).tolist()
                    print(f"{col_name}: {sample_values}")

        return constant_features

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return []

if __name__ == "__main__":
    # Check recent parquet files
    recent_files = [
        "/Users/remyroche/Documents/Ares/data/training/hmm_clusters/hmm_composite_clusters_binance_ETHUSDT_1m.parquet",
        "/Users/remyroche/Documents/Ares/data/training/aggtrades_binance_ETHUSDT_raw.parquet"
    ]

    for file_path in recent_files:
        file_obj = Path(file_path)
        if file_obj.exists():
            print(f"\n📁 Checking file: {file_path}")
            check_constant_features(file_obj)

    # Check the most recent vectorized features file
    vectorized_dir = Path("/Users/remyroche/Documents/Ares/data/vectorized_features")
    if vectorized_dir.exists():
        parquet_files = list(vectorized_dir.glob("*.parquet"))
        if parquet_files:
            # Sort by modification time (most recent first)
            most_recent = max(parquet_files, key=lambda f: f.stat().st_mtime)
            print(f"\n📁 Checking most recent vectorized file: {most_recent}")
            check_constant_features(most_recent)

    # Also check if there's a specific consolidated file
    consolidated_file = Path("/Users/remyroche/Documents/Ares/data/training/features_binance_ETHUSDT_consolidated.parquet")
    if consolidated_file.exists():
        print(f"\n📁 Checking consolidated file: {consolidated_file}")
        check_constant_features(consolidated_file)
    else:
        print(f"\n❌ Consolidated file not found: {consolidated_file}")

    # Search for any file with the problematic columns
    print("\n🔍 Searching for files containing problematic columns...")
    import os
    for root, dirs, files in os.walk("/Users/remyroche/Documents/Ares/data"):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                try:
                    # Quick check if file contains the problematic columns
                    data = pd.read_parquet(file_path, columns=['trade_volume', 'trade_count', 'avg_price']).head(1)
                    if len(data.columns) > 0:
                        print(f"📋 Found file with problematic columns: {file_path}")
                        check_constant_features(file_path)
                        break
                except:
                    continue