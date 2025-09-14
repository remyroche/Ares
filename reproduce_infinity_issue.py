#!/usr/bin/env python3
"""
Script to reproduce the infinity value issue that occurs around row 4581
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_problematic_data_scenario():
    """Create a data scenario that would generate infinity values like those seen in the logs"""

    print("🔧 Creating problematic data scenario to reproduce infinity values...")

    # Load real market data to understand the structure
    try:
        data_file = "/Users/remyroche/Documents/Ares/historical_data/binance/ethusdt/processed/ethusdt_1m/features_ethusdt_1m_consolidated.parquet"
        df = pd.read_parquet(data_file)
        print(f"📊 Loaded real data: {df.shape}")

        # Create a copy for testing
        test_df = df.copy()

        # Simulate problematic conditions that could cause infinity values

        # 1. Create zero or near-zero values that would cause division by zero
        print("⚠️ Introducing problematic conditions...")

        # Make some close prices zero around row 4581 (this would cause price_range_pct to be infinite)
        problematic_rows = list(range(4580, 4590))
        test_df.loc[problematic_rows, 'close'] = 0.0

        # Make some volume values zero (this would cause volume ratios to be infinite)
        test_df.loc[problematic_rows, 'volume'] = 0.0

        # Create extreme values that could cause overflow
        test_df.loc[4595:4600, 'high'] = test_df.loc[4595:4600, 'high'] * 1e10
        test_df.loc[4595:4600, 'low'] = test_df.loc[4595:4600, 'low'] * -1e10

        print("✅ Created problematic data scenario")
        return test_df

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def simulate_feature_engineering_with_infinity(df):
    """Simulate feature engineering operations that would generate infinity values"""

    print("🔧 Simulating feature engineering operations that could generate infinity...")

    if df is None:
        return None

    try:
        # Create features that commonly generate infinity values

        # 1. Price range percentage (division by zero when close = 0)
        print("   📊 Creating price_range_pct feature...")
        df['price_range'] = df['high'] - df['low']
        df['price_range_pct'] = df['price_range'] / df['close']  # This will be infinite when close = 0

        # 2. Volatility percentage (division by zero)
        print("   📊 Creating volatility_pct feature...")
        df['volatility'] = df['close'].rolling(20).std()
        df['volatility_pct'] = df['volatility'] / df['close']  # This will be infinite when close = 0

        # 3. Volume ratios (division by zero)
        print("   📊 Creating volume_ratio feature...")
        df['avg_volume'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['avg_volume']  # This will be infinite when avg_volume = 0

        # 4. Momentum features (extreme values from pct_change)
        print("   📊 Creating momentum features...")
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)

        # 5. Complex ratios and interactions
        print("   📊 Creating complex ratio features...")
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']

        # 6. Rolling calculations that might fail
        print("   📊 Creating rolling calculation features...")
        df['rolling_std_5'] = df['close'].rolling(5).std()
        df['rolling_mean_5'] = df['close'].rolling(5).mean()

        # 7. Advanced features that might generate infinity
        print("   📊 Creating advanced features...")
        df['volatility_adjusted_return'] = df['momentum_5'] / df['volatility_pct']
        df['momentum_volatility_interaction'] = df['momentum_10'] * df['volatility_pct']

        print("✅ Feature engineering simulation completed")
        return df

    except Exception as e:
        print(f"❌ Error in feature engineering simulation: {e}")
        return None

def analyze_infinity_values(df):
    """Analyze the infinity values generated"""

    print("🔍 Analyzing infinity values...")

    if df is None:
        return

    # Check for infinity values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"📊 Checking {len(numeric_cols)} numeric columns for infinity values...")

    total_inf = 0
    features_with_inf = {}

    for col in numeric_cols:
        try:
            col_data = df[col].astype(float)
            inf_mask = np.isinf(col_data.values)
            inf_count = np.sum(inf_mask)

            if inf_count > 0:
                total_inf += inf_count
                features_with_inf[col] = inf_count
                print(f"   ⚠️ {col}: {inf_count} infinity values")

                # Show specific rows with infinity around 4581
                inf_rows = df.index[inf_mask]
                relevant_rows = [r for r in inf_rows if 4575 <= r <= 4605]

                if relevant_rows:
                    print(f"      🎯 Rows around 4581: {relevant_rows[:5]}{'...' if len(relevant_rows) > 5 else ''}")

                    # Show context for first few infinity values
                    for row_idx in relevant_rows[:3]:
                        if row_idx > 0 and row_idx < len(df) - 1:
                            prev_val = df[col].iloc[row_idx - 1] if pd.notna(df[col].iloc[row_idx - 1]) else 'NaN'
                            curr_val = df[col].iloc[row_idx]
                            next_val = df[col].iloc[row_idx + 1] if pd.notna(df[col].iloc[row_idx + 1]) else 'NaN'
                            print(f"         Row {row_idx}: prev={prev_val}, current={curr_val}, next={next_val}")
        except Exception as e:
            print(f"   ❌ Error checking {col}: {e}")

    print("\n📊 Summary:")
    print(f"   Total infinity values: {total_inf}")
    print(f"   Features with infinity: {len(features_with_inf)}")
    print(f"   Most problematic features: {sorted(features_with_inf.items(), key=lambda x: x[1], reverse=True)[:5]}")

    return features_with_inf

def main():
    """Main function to reproduce and analyze the infinity value issue"""

    print("🚀 Reproducing Infinity Value Issue Around Row 4581")
    print("=" * 60)

    # Create problematic data scenario
    test_df = create_problematic_data_scenario()

    # Simulate feature engineering
    test_df = simulate_feature_engineering_with_infinity(test_df)

    # Analyze infinity values
    features_with_inf = analyze_infinity_values(test_df)

    print("\n" + "=" * 60)
    print("🎯 CONCLUSION:")
    print("The infinity values around row 4581 are likely caused by:")
    print("1. Division by zero in percentage/ratio calculations")
    print("2. Extreme values from pct_change() on zero/near-zero prices")
    print("3. Rolling window calculations on constant data segments")
    print("4. Complex mathematical operations on problematic input data")

    if features_with_inf:
        print(f"\nMost problematic features: {list(features_with_inf.keys())[:5]}")

    print("\n💡 RECOMMENDATIONS:")
    print("1. Add data validation before feature engineering")
    print("2. Use safe_divide() for all division operations")
    print("3. Handle edge cases in percentage change calculations")
    print("4. Validate rolling window operations have sufficient data")
    print("5. Add infinity detection and replacement in preprocessing")

if __name__ == "__main__":
    main()
