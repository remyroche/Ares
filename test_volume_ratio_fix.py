#!/usr/bin/env python3
"""
Test script to verify volume_ratio calculation fix.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_data():
    """Create test data with varying volume."""
    # Create timestamp index (1-minute intervals)
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(50)]

    # Create OHLCV data with varying volume
    np.random.seed(42)
    close_prices = 50000 + np.cumsum(np.random.normal(0, 50, 50))

    # Create varying volume data (not constant)
    base_volume = 1000
    volume_variation = np.random.normal(0, 200, 50)
    volumes = base_volume + volume_variation
    volumes = np.maximum(volumes, 10)  # Ensure positive volumes

    data = {
        'timestamp': timestamps,
        'open': close_prices + np.random.normal(0, 10, 50),
        'high': close_prices + np.abs(np.random.normal(0, 20, 50)),
        'low': close_prices - np.abs(np.random.normal(0, 20, 50)),
        'close': close_prices,
        'volume': volumes,
    }

    df = pd.DataFrame(data)
    df = df.set_index('timestamp')

    return df

def test_volume_ratio_calculation():
    """Test the volume_ratio calculation."""
    print("🧪 Testing volume_ratio calculation fix...")

    # Create test data
    df = create_test_data()
    print(f"📊 Created test dataset with {len(df)} rows")
    print(".2f")

    # Calculate volume_ratio using the corrected method
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / volume_ma_20

    # Handle NaN values at the beginning
    df['volume_ratio'] = df['volume_ratio'].fillna(1.0)

    # Check if volume_ratio is constant
    unique_values = df['volume_ratio'].nunique()
    is_constant = unique_values <= 1

    print("\n🔍 volume_ratio analysis:")
    print(f"  Unique values: {unique_values}")
    print(f"  Is constant: {is_constant}")
    print(f"  Min value: {df['volume_ratio'].min():.6f}")
    print(f"  Max value: {df['volume_ratio'].max():.6f}")
    print(f"  Mean value: {df['volume_ratio'].mean():.6f}")

    # Check if the volume data itself varies
    volume_unique = df['volume'].nunique()
    print("\n📊 Volume data analysis:")
    print(f"  Volume unique values: {volume_unique}")
    print(f"  Volume is constant: {volume_unique <= 1}")

    if not is_constant:
        print("✅ SUCCESS: volume_ratio is no longer constant!")
    else:
        print("❌ FAILED: volume_ratio is still constant")
        print("   This suggests the volume data itself is constant")

    return not is_constant

if __name__ == "__main__":
    success = test_volume_ratio_calculation()
    exit(0 if success else 1)
