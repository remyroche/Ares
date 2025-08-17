#!/usr/bin/env python3
"""
Test script to verify VectorizedAdvancedFeatureEngineering works properly.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from training.steps.vectorized_advanced_feature_engineering import (
    VectorizedAdvancedFeatureEngineering,
)


async def test_advanced_features():
    """Test the advanced feature engineering system."""
    print("🧪 Testing VectorizedAdvancedFeatureEngineering...")

    # Create sample data
    dates = pd.date_range(start="2025-01-01", end="2025-01-10", freq="1min")
    n_samples = len(dates)

    # Create sample OHLCV data
    np.random.seed(42)
    base_price = 100.0
    price_data = pd.DataFrame(
        {
            "open": base_price + np.random.randn(n_samples) * 0.1,
            "high": base_price + np.random.randn(n_samples) * 0.15,
            "low": base_price + np.random.randn(n_samples) * 0.15,
            "close": base_price + np.random.randn(n_samples) * 0.1,
            "volume": np.random.randint(100, 1000, n_samples),
        },
        index=dates,
    )

    volume_data = pd.DataFrame({"volume": price_data["volume"]}, index=dates)

    print(f"📊 Created sample data: {price_data.shape}")
    print(f"📊 Price data columns: {list(price_data.columns)}")
    print(f"📊 Volume data columns: {list(volume_data.columns)}")

    # Initialize feature engineering
    fe_config = {
        "enable_meta_labeling": False,
        "vectorized_advanced_features": {
            "enable_explicit_meta_labels": False,
            "enable_technical_indicators": True,
            "enable_volatility_features": True,
            "enable_momentum_features": True,
            "enable_volume_features": True,
            "enable_microstructure_features": True,
            "enable_wavelet_features": False,  # Disable for testing
        },
    }

    fe = VectorizedAdvancedFeatureEngineering(fe_config)

    try:
        # Initialize
        print("🔧 Initializing feature engineering...")
        success = await fe.initialize()
        if not success:
            print("❌ Failed to initialize feature engineering")
            return False

        print("✅ Feature engineering initialized successfully")

        # Engineer features
        print("🔧 Engineering features...")
        features_dict = await fe.engineer_features(price_data, volume_data)

        if not features_dict:
            print("❌ No features were generated")
            return False

        print(f"✅ Generated {len(features_dict)} features")

        # Convert to DataFrame for analysis
        features_df = pd.DataFrame(features_dict, index=price_data.index)

        # Check for NaN values
        nan_counts = features_df.isna().sum()
        total_nan = nan_counts.sum()

        print(f"📊 Feature statistics:")
        print(f"   - Total features: {len(features_df.columns)}")
        print(f"   - Total rows: {len(features_df)}")
        print(f"   - Total NaN values: {total_nan}")
        print(f"   - Features with NaN: {nan_counts[nan_counts > 0].count()}")

        if total_nan > 0:
            print("⚠️ Warning: Some features contain NaN values")
            print("   Features with NaN:")
            for col, count in nan_counts[nan_counts > 0].items():
                print(f"     - {col}: {count} NaN values")
        else:
            print("✅ No NaN values found in features")

        # Show some feature names
        feature_names = list(features_df.columns)
        print(f"📋 Sample features: {feature_names[:10]}")

        return True

    except Exception as e:
        print(f"❌ Error testing advanced features: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_advanced_features())
    if success:
        print("✅ Advanced features test passed!")
    else:
        print("❌ Advanced features test failed!")
        sys.exit(1)
