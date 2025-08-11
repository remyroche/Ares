#!/usr/bin/env python3
"""
Validate Multicollinearity Fix

This script validates that the multicollinearity issue has been fixed by testing
the feature engineering with sample data and checking for perfect correlations.

Usage:
    python scripts/validate_multicollinearity_fix.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Import after setting up the path
try:
    from training.steps.vectorized_advanced_feature_engineering import (
        VectorizedAdvancedFeatureEngineering,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("⚠️ Make sure you're running this from the project root directory")
    print("⚠️ Try: PYTHONPATH=src python scripts/validate_multicollinearity_fix.py")
    sys.exit(1)


async def validate_multicollinearity_fix():
    """Validate that the multicollinearity issue has been fixed."""

    print("🔍 Validating multicollinearity fix...")

    # Create sample data
    print("📊 Creating sample data...")
    dates = pd.date_range("2024-01-01", periods=1000, freq="1min")

    np.random.seed(42)
    base_price = 100 + np.cumsum(np.random.randn(1000) * 0.1)

    price_data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": base_price + np.random.randn(1000) * 0.5,
            "high": base_price + np.random.randn(1000) * 0.8,
            "low": base_price - np.random.randn(1000) * 0.8,
            "close": base_price + np.random.randn(1000) * 0.5,
            "volume": np.random.randint(100, 1000, 1000),
        }
    )

    volume_data = pd.DataFrame(
        {"timestamp": dates, "volume": price_data["volume"].copy()}
    )

    # Set timestamp as index
    price_data.set_index("timestamp", inplace=True)
    volume_data.set_index("timestamp", inplace=True)

    # Initialize feature engineering
    config = {
        "vectorized_advanced_feature_engineering": {
            "enable_multi_timeframe_features": True,
            "timeframes": ["1m", "5m", "15m", "30m"],
            "enable_microstructure_features": True,
            "enable_adaptive_indicators": True,
            "enable_wavelet_features": False,  # Disable for faster testing
        }
    }

    try:
        feature_eng = VectorizedAdvancedFeatureEngineering(config)
        await feature_eng.initialize()

        # Engineer features
        print("🔧 Engineering features...")
        features = await feature_eng.engineer_features(price_data, volume_data)

        # Convert features to DataFrame
        feature_df = pd.DataFrame()
        for feature_name, feature_series in features.items():
            if isinstance(feature_series, pd.Series):
                feature_df[feature_name] = feature_series

        print(f"📊 Generated {len(feature_df.columns)} features")

        # Check for perfect correlations
        print("🔍 Checking for perfect correlations...")
        correlation_matrix = feature_df.corr()

        # Find perfect correlations (r >= 0.9999)
        perfect_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value >= 0.9999:
                    feature1 = correlation_matrix.columns[i]
                    feature2 = correlation_matrix.columns[j]
                    perfect_correlations.append((feature1, feature2, corr_value))

        if perfect_correlations:
            print(f"❌ Found {len(perfect_correlations)} perfect correlations:")
            for feature1, feature2, corr_value in perfect_correlations:
                print(f"   {feature1} ↔ {feature2} (r={corr_value:.6f})")
            return False
        else:
            print("✅ No perfect correlations found!")

        # Check specific problematic features
        problematic_features = [
            "1m_price_change",
            "5m_price_change",
            "15m_price_change",
            "30m_price_change",
            "1m_volume_change",
            "5m_volume_change",
            "15m_volume_change",
            "30m_volume_change",
        ]

        print("🔍 Checking specific problematic features...")
        for feature in problematic_features:
            if feature in feature_df.columns:
                print(f"   ✅ Found {feature}")
            else:
                print(f"   ⚠️ Missing {feature}")

        # Check that the features are different
        if all(f in feature_df.columns for f in ["1m_price_change", "5m_price_change"]):
            correlation = feature_df["1m_price_change"].corr(
                feature_df["5m_price_change"]
            )
            print(f"📊 1m vs 5m price change correlation: {correlation:.6f}")

            if abs(correlation) >= 0.9999:
                print("❌ 1m and 5m price changes are still perfectly correlated!")
                return False
            else:
                print("✅ 1m and 5m price changes are properly differentiated")

        print("✅ Multicollinearity fix validation completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return False


def main():
    """Main function to run the validation."""
    import asyncio

    success = asyncio.run(validate_multicollinearity_fix())
    if success:
        print("\n🎉 MULTICOLLINEARITY FIX VALIDATION PASSED!")
        print("✅ Your feature engineering is now working correctly.")
    else:
        print("\n❌ MULTICOLLINEARITY FIX VALIDATION FAILED!")
        print("❌ There are still issues with the feature engineering.")
        sys.exit(1)


if __name__ == "__main__":
    main()
