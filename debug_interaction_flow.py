#!/usr/bin/env python3
"""
Debug script to trace the exact flow of interaction features through the system
"""

from src.training.steps.vectorized_advanced_feature_engineering import (VectorizedAdvancedFeatureEngineering)
import asyncio
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def debug_interaction_flow():
    """Debug the exact flow of interaction features through the system"""
    print("🔍 Debugging interaction features flow...")

    # Create a mock instance
    config, {
        "symbol": "ETHUSDT",
        "exchange": "BINANCE",
        "timeframe": "1m",
        "cache_dir": "data/feature_cache",
    }

    # Create instance with mock logger

    class MockLogger:
        def info(self, msg):
            print(f"INFO: {msg}")

        def warning(self, msg):
            print(f"WARNING: {msg}")

        def error(self, msg):
            print(f"ERROR: {msg}")

        def debug(self, msg):
            print(f"DEBUG: {msg}")

    feature_eng=VectorizedAdvancedFeatureEngineering(config)
    feature_eng.logger=MockLogger()

    # Create mock price data
    price_data=pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [102, 103, 104, 105, 106],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
    )

    # Create mock features
    features={
        "rsi": pd.Series([0.5, 0.6, 0.7, 0.8, 0.9]),
        "volume": pd.Series([100, 110, 120, 130, 140]),
        "price_momentum": pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]),
        "macd": pd.Series([0.05, 0.15, 0.25, 0.35, 0.45]),
        "volatility": pd.Series([0.02, 0.03, 0.04, 0.05, 0.06]),
    }

    print(f"\n📊 Initial features: {len(features)}")
    print("Feature names:", list(features.keys()))

    # Test the interaction features generation method directly
    print("\n🔍 Testing _generate_interaction_features method directly...")

    # Create some enhanced features (difference/acceleration features)
    enhanced_features={
        "rsi_diff_1": pd.Series([0.1, 0.1, 0.1, 0.1, 0.1]),
        "rsi_diff_3": pd.Series([0.2, 0.2, 0.2, 0.2, 0.2]),
        "volume_diff_1": pd.Series([10, 10, 10, 10, 10]),
        "volume_diff_3": pd.Series([20, 20, 20, 20, 20]),
        "rsi_accel_1": pd.Series([0.01, 0.01, 0.01, 0.01, 0.01]),
        "rsi_accel_3": pd.Series([0.02, 0.02, 0.02, 0.02, 0.02]),
        "price_momentum_diff_1": pd.Series([0.1, 0.1, 0.1, 0.1, 0.1]),
        "price_momentum_diff_3": pd.Series([0.2, 0.2, 0.2, 0.2, 0.2]),
        "macd_diff_1": pd.Series([0.1, 0.1, 0.1, 0.1, 0.1]),
        "macd_diff_3": pd.Series([0.2, 0.2, 0.2, 0.2, 0.2]),
        "volatility_diff_1": pd.Series([0.01, 0.01, 0.01, 0.01, 0.01]),
        "volatility_diff_3": pd.Series([0.02, 0.02, 0.02, 0.02, 0.02]),
    }

    print(f"Enhanced features before interaction generation: {len(enhanced_features)}")
    print("Enhanced feature names:", list(enhanced_features.keys()))

    # Call the interaction features generation method
    interaction_features=await feature_eng._generate_interaction_features(
        enhanced_features = features,
        price_data=price_data)

    print(f"\n📊 Interaction features generated: {len(interaction_features)}")
    print("Interaction feature names:", list(interaction_features.keys()))

    # Now test the full flow by calling the difference/acceleration method
    print("\n🔍 Testing full difference/acceleration flow...")

    # Create a larger set of features to simulate the real scenario
    large_features={}
    for i in range(100):  # Create 100 features to trigger capping
        large_features[f"feature_{i}"] = pd.Series([i] * 5)

    # Add some specific features that should trigger interaction generation
    large_features.update(
        {
            "rsi": pd.Series([0.5, 0.6, 0.7, 0.8, 0.9]),
            "volume": pd.Series([100, 110, 120, 130, 140]),
            "price_momentum": pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]),
            "macd": pd.Series([0.05, 0.15, 0.25, 0.35, 0.45]),
            "volatility": pd.Series([0.02, 0.03, 0.04, 0.05, 0.06]),
        }
    )

    print(f"Large features set: {len(large_features)}")

    # Call the full difference/acceleration method
    result=await feature_eng._engineer_difference_and_acceleration_features(
        large_features = price_data,
    )

    print(f"\n📊 Final result: {len(result)}")

    # Check for interaction features in the result
    interaction_features_in_result=[f for f in result.keys() if "_x_" in f]
    print(
        f"Interaction features in final result: {len(interaction_features_in_result)}"
    )
    print("Interaction feature names in result:", interaction_features_in_result)

    # Test the summary logging method
    print("\n🔍 Testing summary logging method...")
    feature_eng._log_feature_engineering_summary(result=result)


if __name__== "__main__":
    asyncio.run(debug_interaction_flow())
