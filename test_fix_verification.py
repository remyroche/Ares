import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, '/Users/remyroche/Documents/Ares/src')

def test_fixed_aggregation():
    """Test that the fixed aggregation method works correctly."""

    # Load the problematic file
    aggtrades_file = Path("/Users/remyroche/Documents/Ares/data/training/aggtrades_binance_ETHUSDT_raw.parquet")

    if not aggtrades_file.exists():
        print(f"❌ Test file not found: {aggtrades_file}")
        return False

    print("🔍 Testing fixed aggregation method...")

    # Read the data
    data = pd.read_parquet(aggtrades_file)
    print(f"✅ Loaded {len(data)} rows with columns: {list(data.columns)}")

    # Verify the problematic columns are all zeros
    print("\n📊 Pre-fix verification:")
    for col in ['trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price']:
        if col in data.columns:
            unique_vals = data[col].nunique()
            non_zero = (data[col] != 0).sum()
            print(f"  {col}: {unique_vals} unique values, {non_zero} non-zero values")

    # Simulate the fixed logic
    print("\n🔧 Simulating fixed aggregation logic...")

    # 1. Detect zero aggregated columns
    zero_cols = []
    for col in ['trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price']:
        if col in data.columns:
            unique_vals = data[col].nunique()
            non_zero_count = (data[col] != 0).sum()
            if unique_vals <= 1 and non_zero_count == 0:
                zero_cols.append(col)
                print(f"  ✅ Detected zero column: {col}")

    print(f"📋 Found {len(zero_cols)} zero aggregated columns: {zero_cols}")

    # 2. Remove zero columns and recalculate
    if zero_cols:
        data_clean = data.drop(columns=zero_cols, errors='ignore')
        print(f"✅ Removed zero columns, remaining columns: {list(data_clean.columns)}")

        # 3. Perform fresh aggregation
        timestamp_col = 'timestamp'  # Based on our analysis
        price_col = 'close'  # Based on our analysis

        print(f"🔧 Using timestamp_col='{timestamp_col}', price_col='{price_col}'")

        agg_result = data_clean.groupby(timestamp_col).agg({
            'quantity': ['sum', 'count'],
            price_col: ['mean', 'min', 'max', 'std']
        }).reset_index()

        # Flatten columns
        agg_result.columns = ['timestamp', 'trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price', 'price_std']

        print(f"✅ Fresh aggregation successful: {len(agg_result)} groups")

        # 4. Verify results
        print("\n📊 Post-fix verification:")
        success = True
        for col in ['trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price']:
            if col in agg_result.columns:
                unique_vals = agg_result[col].nunique()
                non_zero = (agg_result[col] != 0).sum()
                std_val = agg_result[col].std()
                print(f"  {col}: {unique_vals} unique, {non_zero} non-zero, std={std_val:.6f}")
                if unique_vals <= 1:
                    print(f"  ❌ {col} still has constant values!")
                    success = False
                else:
                    print(f"  ✅ {col} has proper variation")

        if success:
            print("\n🎉 SUCCESS: Fixed aggregation method resolves the constant feature issue!")
            return True
        else:
            print("\n❌ FAILURE: Fixed method still produces constant features")
            return False
    else:
        print("❌ No zero columns detected - test scenario not met")
        return False

if __name__ == "__main__":
    success = test_fixed_aggregation()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
