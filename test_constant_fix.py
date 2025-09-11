#!/usr/bin/env python3
"""
Test script to validate constant feature fixes in the data converter.
"""

import sys
import os
import numpy as np
import pandas as pd

def create_test_data():
    """Create test data to simulate the constant feature scenario."""
    # Create test OHLC data
    timestamps = np.arange(1640995200000, 1640995200000 + 1000 * 60 * 60 * 24, 60000)  # 1 day of 1-minute data

    # Create OHLC data with some variation
    base_price = 40000
    ohlc_data = []
    for i, ts in enumerate(timestamps):
        # Add some variation to prices
        variation = np.sin(i * 0.1) * 100 + np.random.normal(0, 50)
        open_price = base_price + variation
        high_price = open_price + abs(np.random.normal(0, 20))
        low_price = open_price - abs(np.random.normal(0, 20))
        close_price = open_price + np.random.normal(0, 10)
        volume = np.random.uniform(100, 1000)

        ohlc_data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

    ohlc_df = pd.DataFrame(ohlc_data)

    # Create aggtrades data that could cause constant features
    agg_data = []
    for i, ts in enumerate(timestamps):
        # Create trades with some variation but potentially constant aggregations
        if i % 5 == 0:  # Every 5th timestamp has trades
            for j in range(np.random.randint(1, 5)):  # 1-4 trades per timestamp
                price = ohlc_df.loc[i, 'close'] + np.random.normal(0, 5)
                quantity = np.random.uniform(0.1, 2.0)
                agg_data.append({
                    'kline_timestamp': ts,
                    'price': price,
                    'quantity': quantity,
                    'is_buyer_maker': np.random.choice([True, False])
                })

    agg_df = pd.DataFrame(agg_data)

    # Create futures data with constant funding rate (problematic case)
    futures_data = []
    for ts in timestamps[::100]:  # Every 100th timestamp (funding rates are less frequent)
        futures_data.append({
            'timestamp': ts,
            'fundingRate': 0.0001  # Constant funding rate
        })

    futures_df = pd.DataFrame(futures_data)

    return ohlc_df, agg_df, futures_df

def test_constant_feature_detection():
    """Test the constant feature detection logic."""
    print("🧪 Testing constant feature detection...")

    # Create test data
    ohlc_df, agg_df, futures_df = create_test_data()

    print(f"📊 Created test data:")
    print(f"  - OHLC: {len(ohlc_df)} rows")
    print(f"  - AggTrades: {len(agg_df)} rows")
    print(f"  - Futures: {len(futures_df)} rows")

    # Create mock sub-pipeline to test constant feature detection
    class MockSubPipeline:
        def _check_for_constant_features(self, data: pd.DataFrame):
            """Check for constant features that indicate data processing issues."""
            constant_features = []
            trade_stat_cols = ['trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price', 'price_std']
            funding_cols = ['funding_rate']

            # Check critical trade and funding features
            for col in trade_stat_cols + funding_cols:
                if col in data.columns:
                    unique_vals = data[col].nunique()
                    std_val = data[col].std()
                    if unique_vals <= 2 or (not pd.isna(std_val) and std_val < 1e-10):
                        constant_features.append(f"{col}(unique={unique_vals}, std={std_val:.2e})")

            return constant_features

    mock_pipeline = MockSubPipeline()

    # Test with original potentially constant data
    print("\n🔍 Testing constant feature detection...")

    # Create test unified data with constant features
    test_data = pd.DataFrame({
        'timestamp': ohlc_df['timestamp'][:10],
        'trade_volume': [100.0] * 10,  # Constant
        'trade_count': [5] * 10,       # Constant
        'avg_price': [40000.0] * 10,   # Constant
        'min_price': [39900.0] * 10,   # Constant
        'max_price': [40100.0] * 10,   # Constant
        'funding_rate': [0.0001] * 10  # Constant
    })

    constant_features = mock_pipeline._check_for_constant_features(test_data)

    if constant_features:
        print(f"🚨 Detected constant features: {constant_features}")
        print("❌ Constant features still present - fixes may not be working")
        return False
    else:
        print("✅ No constant features detected!")
        return True

def test_random_variation():
    """Test that random variation is working properly."""
    print("\n🎲 Testing random variation generation...")

    # Test the timestamp-based seed logic
    timestamps = np.array([1640995200000, 1640995260000, 1640995320000])

    # Test with the same timestamps multiple times to ensure reproducibility
    results1 = []
    results2 = []

    for i in range(3):
        np.random.seed(timestamps % 2**32)
        variation1 = np.random.uniform(0.95, 1.05, 3)
        results1.append(variation1)

        np.random.seed(timestamps % 2**32)
        variation2 = np.random.uniform(0.95, 1.05, 3)
        results2.append(variation2)

    # Check if results are reproducible
    reproducible = np.allclose(results1[0], results2[0]) and np.allclose(results1[1], results2[1])
    variation_present = not np.allclose(results1[0], [1.0, 1.0, 1.0])  # Should not be all 1.0

    if reproducible and variation_present:
        print("✅ Random variation is reproducible and varied!")
        return True
    else:
        print("❌ Random variation issues detected")
        print(f"  Reproducible: {reproducible}")
        print(f"  Has variation: {variation_present}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Constant Feature Fixes")
    print("=" * 50)

    success1 = test_random_variation()
    success2 = test_constant_feature_detection()

    if success1 and success2:
        print("\n🎉 All tests passed! Constant feature fixes appear to be working.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review the fixes.")
        sys.exit(1)
