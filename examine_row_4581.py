#!/usr/bin/env python3
"""
Script to examine the actual data values around row 4581 to understand
what specific mathematical operation is causing infinity values.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def examine_row_4581():
    """Examine data around row 4581 to find the cause of infinity values"""

    print("🔍 Examining data around row 4581 for infinity value causes...")

    # Find the feature file
    data_file = "/Users/remyroche/Documents/Ares/historical_data/binance/ethusdt/processed/ethusdt_1m/features_ethusdt_1m_consolidated.parquet"

    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        return

    try:
        print(f"📊 Loading data from: {data_file}")
        df = pd.read_parquet(data_file)
        print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")

        # Focus on rows around 4581
        start_row = max(0, 4581 - 10)
        end_row = min(len(df), 4581 + 10)
        subset = df.iloc[start_row:end_row]

        print(f"\n📋 Examining rows {start_row} to {end_row} (around row 4581)")

        # Check basic price data
        price_cols = ['open', 'high', 'low', 'close']
        print("\n💰 Basic Price Data:")
        for col in price_cols:
            if col in subset.columns:
                values = subset[col]
                print(f"   {col}: min={values.min():.8f}, max={values.max():.8f}, mean={values.mean():.8f}")

                # Check for very small values
                small_mask = (values > 0) & (values < 1e-8)
                if small_mask.any():
                    print(f"      ⚠️ Very small values found: {values[small_mask].head(3).tolist()}")

                # Check for zero values
                zero_mask = values == 0
                if zero_mask.any():
                    print(f"      ❌ Zero values found at rows: {values[zero_mask].index.tolist()}")

        # Check volume data
        volume_cols = ['volume', 'quote_volume', 'trades']
        print("\n📊 Volume Data:")
        for col in volume_cols:
            if col in subset.columns:
                values = subset[col]
                print(f"   {col}: min={values.min():.8f}, max={values.max():.8f}, mean={values.mean():.8f}")

                # Check for very small values
                small_mask = (values > 0) & (values < 1e-12)
                if small_mask.any():
                    print(f"      ⚠️ Very small values found: {values[small_mask].head(3).tolist()}")

                # Check for zero values
                zero_mask = values == 0
                if zero_mask.any():
                    print(f"      ❌ Zero values found at rows: {values[zero_mask].index.tolist()}")

        # Check if we can compute the problematic features manually
        print("\n🔬 Manual Feature Computation:")
        if all(col in subset.columns for col in ['high', 'low', 'close', 'volume']):
            try:
                # Compute price_range_pct manually
                price_range = subset['high'] - subset['low']
                price_range_pct = price_range / subset['close']
                print(f"   price_range_pct: {price_range_pct.describe()}")

                # Check for infinity in manual computation
                inf_mask = np.isinf(price_range_pct)
                if inf_mask.any():
                    print(f"      ❌ Manual price_range_pct has infinity at rows: {price_range_pct[inf_mask].index.tolist()}")
                    print(f"         Values: close={subset['close'][inf_mask].values}, range={price_range[inf_mask].values}")

                # Compute rolling volatility manually
                rolling_std = subset['close'].rolling(20, min_periods=1).std()
                volatility_pct = rolling_std / subset['close']
                print(f"   volatility_pct: {volatility_pct.describe()}")

                # Check for infinity in volatility_pct
                inf_mask_vol = np.isinf(volatility_pct)
                if inf_mask_vol.any():
                    print(f"      ❌ Manual volatility_pct has infinity at rows: {volatility_pct[inf_mask_vol].index.tolist()}")

                # Compute volume ratio manually
                rolling_volume = subset['volume'].rolling(20, min_periods=1).mean()
                volume_ratio = subset['volume'] / rolling_volume
                print(f"   volume_ratio: {volume_ratio.describe()}")

                # Check for infinity in volume_ratio
                inf_mask_ratio = np.isinf(volume_ratio)
                if inf_mask_ratio.any():
                    print(f"      ❌ Manual volume_ratio has infinity at rows: {volume_ratio[inf_mask_ratio].index.tolist()}")

            except Exception as e:
                print(f"   ❌ Error in manual computation: {e}")

        # Check for data type issues
        print("\n🔧 Data Type Analysis:")
        for col in subset.columns:
            dtype = subset[col].dtype
            if dtype in ['float64', 'float32', 'int64', 'int32']:
                nan_count = subset[col].isna().sum()
                inf_count = np.isinf(subset[col]).sum()
                if nan_count > 0 or inf_count > 0:
                    print(f"   {col}: dtype={dtype}, NaN={nan_count}, Inf={inf_count}")

        # Check for extreme values that might cause overflow
        print("\n⚠️ Extreme Value Analysis:")
        numeric_cols = subset.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            values = subset[col]
            finite_mask = np.isfinite(values)

            if finite_mask.any():
                finite_values = values[finite_mask]
                if len(finite_values) > 0:
                    very_large = finite_values > 1e15
                    very_small = (finite_values > 0) & (finite_values < 1e-15)

                    if very_large.any():
                        print(f"   {col}: Very large values > 1e15 at rows: {finite_values[very_large].index.tolist()[:3]}")
                    if very_small.any():
                        print(f"   {col}: Very small values < 1e-15 at rows: {finite_values[very_small].index.tolist()[:3]}")

    except Exception as e:
        print(f"❌ Error examining data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    examine_row_4581()
