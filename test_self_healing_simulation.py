import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, '/Users/remyroche/Documents/Ares/src')

def simulate_constant_feature_detection(data):
    """Simulate the _check_for_constant_features method from the sub_pipeline."""
    constant_features = []
    trade_stat_cols = ['trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price', 'price_std']
    funding_cols = []

    # Check critical trade and funding features
    for col in trade_stat_cols + funding_cols:
        if col in data.columns:
            unique_vals = data[col].nunique()
            std_val = data[col].std()
            if unique_vals <= 2 or (not pd.isna(std_val) and std_val < 1e-10):
                constant_features.append(f"{col}(unique={unique_vals}, std={std_val:.2e})")

    return constant_features

def simulate_self_healing_hook():
    """Simulate the complete self-healing workflow."""

    print("🔧 SIMULATING SELF-HEALING HOOK FOR HMM REGIME DISCOVERY")
    print("=" * 60)

    # Step 1: Load the problematic data (simulating HMM regime discovery loading data)
    print("📥 Step 1: HMM Regime Discovery - Loading data...")
    data_file = "/Users/remyroche/Documents/Ares/data/training/aggtrades_binance_ETHUSDT_raw.parquet"

    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        return False

    data = pd.read_parquet(data_file)
    print(f"✅ Loaded data with {len(data)} records and {len(data.columns)} features")

    # Step 2: Check for constant features (simulating the FAST-FAIL check)
    print("\n🔍 Step 2: Checking for constant features...")
    constant_features = simulate_constant_feature_detection(data)

    if constant_features:
        print("🚨 CONSTANT FEATURES DETECTED!")
        for feature in constant_features:
            print(f"   - {feature}")
    else:
        print("✅ No constant features detected")
        return True

    # Step 3: Simulate the self-healing hook activation
    print("\n🔧 Step 3: SELF-HEALING HOOK ACTIVATED")
    print("🔄 Attempting automatic fix: Triggering data converter...")

    # Step 4: Simulate the data converter fix (our core logic)
    print("\n📊 Step 4: Data Converter - Detecting zero aggregated columns...")
    zero_cols = []
    for col in ['trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price']:
        if col in data.columns:
            unique_vals = data[col].nunique()
            non_zero_count = (data[col] != 0).sum()
            if unique_vals <= 1 and non_zero_count == 0:
                zero_cols.append(col)
                print(f"   ✅ Detected zero column: {col}")

    print(f"📋 Found {len(zero_cols)} zero aggregated columns: {zero_cols}")

    if zero_cols:
        print("\n🔄 Step 5: Removing zero columns and recalculating...")

        # Remove zero columns
        data_clean = data.drop(columns=zero_cols, errors='ignore')
        print(f"✅ Removed zero columns, remaining columns: {list(data_clean.columns)}")

        # Determine column names
        timestamp_col = 'timestamp'  # Based on our analysis
        price_col = 'close'  # Based on our analysis

        print(f"🔧 Using timestamp_col='{timestamp_col}', price_col='{price_col}'")

        # Perform fresh aggregation (simulating the fixed _calculate_proper_trade_statistics method)
        print("🔧 Performing fresh aggregation from raw trade data...")

        try:
            agg_result = data_clean.groupby(timestamp_col).agg({
                'quantity': ['sum', 'count'],
                price_col: ['mean', 'min', 'max', 'std']
            }).reset_index()

            # Flatten columns
            agg_result.columns = ['timestamp', 'trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price', 'price_std']

            print(f"✅ Fresh aggregation successful: {len(agg_result)} groups")

            # Step 6: Verify the fix worked
            print("\n📊 Step 6: Verifying the fix...")
            success = True
            for col in ['trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price']:
                if col in agg_result.columns:
                    unique_vals = agg_result[col].nunique()
                    non_zero = (agg_result[col] != 0).sum()
                    std_val = agg_result[col].std()
                    print(".6f")
                    if unique_vals <= 1:
                        print(f"   ❌ {col} still has constant values!")
                        success = False
                    else:
                        print(f"   ✅ {col} has proper variation")

            # Step 7: Simulate saving the fixed data
            if success:
                print("\n💾 Step 7: Saving fixed consolidated features file...")
                output_file = "/Users/remyroche/Documents/Ares/data/training/features_binance_ETHUSDT_consolidated_fixed.parquet"
                agg_result.to_parquet(output_file, index=False)
                print(f"✅ Saved fixed data to: {output_file}")

                # Verify file was created
                if Path(output_file).exists():
                    file_size = Path(output_file).stat().st_size
                    print(f"✅ File created successfully: {file_size} bytes")
                else:
                    print("❌ File creation failed")

            return success

        except Exception as e:
            print(f"❌ Fresh aggregation failed: {e}")
            return False
    else:
        print("❌ No zero columns detected - unexpected scenario")
        return False

if __name__ == "__main__":
    print("🧪 SELF-HEALING HOOK SIMULATION")
    print("This simulates the complete workflow of:")
    print("1. HMM regime discovery detecting constant features")
    print("2. Self-healing hook triggering data converter")
    print("3. Data converter fixing the aggregated statistics")
    print("4. HMM regime discovery proceeding with fixed data")
    print()

    success = simulate_self_healing_hook()

    print("\n" + "=" * 60)
    print("🎯 SIMULATION RESULTS:")
    print(f"   Status: {'✅ SUCCESS' if success else '❌ FAILED'}")

    if success:
        print("   🎉 Self-healing hook successfully resolved constant features!")
        print("   📁 Fixed consolidated features file created")
        print("   🚀 HMM regime discovery can now proceed")
    else:
        print("   ⚠️ Self-healing hook failed to resolve constant features")

    print("\n🔄 Next step: Test HMM regime discovery with the fixed data file")
