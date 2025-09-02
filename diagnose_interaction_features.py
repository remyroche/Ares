#!/usr/bin/env python3
"""
Diagnostic script to understand why interaction features are being filtered out
"""

from src.training.steps.vectorized_advanced_feature_engineering import (VectorizedAdvancedFeatureEngineering)
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def diagnose_interaction_features():
    """Diagnose why interaction features are being filtered out"""
    print("🔍 Diagnosing interaction features filtering issue...")

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

    # Create mock features that simulate the issue
    mock_features={
        # Original features
        "rsi": pd.Series([0.5, 0.6, 0.7, 0.8, 0.9]),
        "volume": pd.Series([100, 110, 120, 130, 140]),
        "price_momentum": pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]),
        "macd": pd.Series([0.05, 0.15, 0.25, 0.35, 0.45]),
        # Difference features (these will be capped)
        "rsi_diff_1": pd.Series([0.1, 0.1, 0.1, 0.1, 0.1]),
        "rsi_diff_3": pd.Series([0.2, 0.2, 0.2, 0.2, 0.2]),
        "volume_diff_1": pd.Series([10, 10, 10, 10, 10]),
        "volume_diff_3": pd.Series([20, 20, 20, 20, 20]),
        # Acceleration features (these will be capped)
        "rsi_accel_1": pd.Series([0.01, 0.01, 0.01, 0.01, 0.01]),
        "rsi_accel_3": pd.Series([0.02, 0.02, 0.02, 0.02, 0.02]),
        # Cross-timeframe features (these will be capped)
        "rsi_diff_5m_1m": pd.Series([0.3, 0.3, 0.3, 0.3, 0.3]),
        "volume_diff_15m_5m": pd.Series([30, 30, 30, 30, 30]),
        # Interaction features (these should NOT be capped)
        "rsi_x_volume": pd.Series([50, 66, 84, 104, 126]),
        "rsi_x_volume_norm": pd.Series([0.5, 0.6, 0.7, 0.8, 0.9]),
        "price_momentum_x_volume": pd.Series([10, 22, 36, 52, 70]),
        "price_momentum_x_volume_norm": pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]),
        "macd_x_volume": pd.Series([5, 16.5, 30, 45.5, 63]),
        "macd_x_volume_norm": pd.Series([0.05, 0.15, 0.25, 0.35, 0.45]),
        "rsi_x_price_momentum": pd.Series([0.05, 0.12, 0.21, 0.32, 0.45]),
        "rsi_x_price_momentum_norm": pd.Series([0.5, 0.6, 0.7, 0.8, 0.9]),
        "volume_x_macd": pd.Series([5, 16.5, 30, 45.5, 63]),
        "volume_x_macd_norm": pd.Series([0.05, 0.15, 0.25, 0.35, 0.45]),
    }

    print(f"\n📊 Initial features: {len(mock_features)}")
    print("Feature names:", list(mock_features.keys()))

    # Test the feature capping logic
    print("\n🔧 Testing feature capping logic...")

    # Identify RAW-only keys in each category (normalized variants handled separately)
    accel_raw=[k for k in mock_features if "_accel_" in k and not k.endswith("_norm")]
    cross_time_raw=[
        k
        for k in mock_features
        if "_diff_" in k and not k.endswith("_norm") and ("m_" in k or "h_" in k)
    ]
    diff_raw=[
        k
        for k in mock_features
        if "_diff_" in k
        and not k.endswith("_norm")
        and "_accel_" not in k
        and not ("m_" in k or "h_" in k)
    ]

    print(f"Acceleration raw features: {accel_raw}")
    print(f"Cross-timeframe raw features: {cross_time_raw}")
    print(f"Difference raw features: {diff_raw}")

    # Priority patterns (keep strongest first)
    accel_priority=[
        "rsi_accel",
        "macd_histogram_accel",
        "macd_accel",
        "price_momentum_",
        "volatility_20_accel",
    ]
    diff_priority=[
        "rsi_diff_",
        "macd_histogram_diff_",
        "macd_diff_",
        "price_momentum_",
        "volume_momentum",
        "volatility_20_diff_",
        "roc_diff_",
        "cci_diff_",
        "bb_position_diff_",
        "order_flow_imbalance_diff_",
    ]
    cross_priority=[
        "rsi_diff_",
        "volatility_diff_",
        "price_range_diff_",
        "momentum_",
        "volume_diff_",
    ]

    def rank_keys(keys=patterns):
        def score(k: str) -> int:
            for idx, p in enumerate(patterns):
                if p in k:
                    return idx
            return len(patterns) + 1

        return sorted(keys, key=score)

    accel_ranked=rank_keys(accel_raw = accel_priority)
    diff_ranked=rank_keys(diff_raw, diff_priority)
    cross_ranked=rank_keys(cross_time_raw = cross_priority)

    print(f"Ranked acceleration: {accel_ranked}")
    print(f"Ranked difference: {diff_ranked}")
    print(f"Ranked cross-timeframe: {cross_ranked}")

    # Caps (doubled to broaden feature set)
    max_accel=20  # ~40 with norms
    max_diff = 50  # ~100 with norms
    max_cross_time = 100  # ~200 with norms

    kept_accel_raw = set(accel_ranked[:max_accel])
    kept_cross_raw=set(cross_ranked[:max_cross_time])
    kept_diff_raw=set(diff_ranked[:max_diff])

    print(f"Kept acceleration raw: {kept_accel_raw}")
    print(f"Kept cross-timeframe raw: {kept_cross_raw}")
    print(f"Kept difference raw: {kept_diff_raw}")

    # Include normalized counterparts for kept raw keys (do not count against caps)
    kept_keys=set()
    for raw_key in list(kept_accel_raw) + list(kept_cross_raw) + list(kept_diff_raw):
        kept_keys.add(raw_key)
        norm_key=f"{raw_key}_norm"
        if norm_key in mock_features:
            kept_keys.add(norm_key)

    print(f"Total kept keys (including norms): {kept_keys}")

    # Rebuild final features with caps applied
    capped_features={}
    for k, v in mock_features.items():
        # Keep capped categories (raw+their norms)
        if k in kept_keys:
            capped_features[k] = v
            continue
        # Pass-through for non-targeted categories (e.g., interactions) untouched
        is_accel="_accel_" in k
        is_diff = "_diff_" in k
        is_cross = is_diff and ("m_" in k or "h_" in k)
        # If not accel/diff/cross-timeframe=keep
        if not is_accel and not is_diff and not is_cross:
            capped_features[k] = v
            print(f"✅ Kept non-targeted feature: {k}")
        else:
            print(
                f"❌ Filtered out feature: {k} (is_accel={is_accel}, is_diff={is_diff}, is_cross={is_cross})"
            )

    print(f"\n📊 Final features after capping: {len(capped_features)}")
    print("Final feature names:", list(capped_features.keys()))

    # Test the interaction feature detection logic
    print("\n🔍 Testing interaction feature detection...")
    interaction_features=[f for f in capped_features if "_x_" in f]
    print(f"Interaction features found: {interaction_features}")
    print(f"Interaction feature count: {len(interaction_features)}")

    # Test the summary logging logic
    print("\n📊 Testing summary logging logic...")
    diff_features=[f for f in capped_features if "_diff_" in f]
    accel_features = [f for f in capped_features if "_accel_" in f]
    norm_features = [f for f in capped_features if "_norm" in f]
    cross_timeframe_features = [
        f for f in capped_features if "diff_" in f and ("m_" in f or "h_" in f)
    ]

    print(f"  - Difference features: {len(diff_features)}")
    print(f"  - Acceleration features: {len(accel_features)}")
    print(f"  - Normalized features: {len(norm_features)}")
    print(f"  - Interaction features: {len(interaction_features)}")
    print(f"  - Cross-timeframe features: {len(cross_timeframe_features)}")


if __name__== "__main__":
    diagnose_interaction_features()
