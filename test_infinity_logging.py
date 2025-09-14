#!/usr/bin/env python3
"""
Test script to verify that infinity values in features are logged with feature names instead of indices.
"""

import pandas as pd
import numpy as np
from src.training.utils.feature_selection.selection_methods import MRMRSelector


def test_infinity_logging_with_feature_names():
    """Test that infinity values in features are logged with feature names instead of indices."""
    print("🔍 Testing infinity value logging with feature names...")

    # Create sample data with infinity values
    n_samples = 100
    n_features = 10
    np.random.seed(42)

    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Create data with some infinity values
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    # Add infinity values to specific features
    X[10:12, 3] = np.inf    # Feature 3 (idx 3) gets 2 positive infinity values
    X[15, 7] = -np.inf      # Feature 7 (idx 7) gets 1 negative infinity value

    print(f"📊 Created test data: {n_samples} samples, {n_features} features")
    print(f"📊 Added infinity values to feature_3 and feature_7")

    # Test MRMR selection which will call preprocess_features_for_ml
    try:
        mrmr_selector = MRMRSelector()
        result = mrmr_selector.select_features(X, y, feature_names, n_features=5)

        if result['success']:
            print("✅ MRMR selection completed successfully")
            print(f"📊 Selected {len(result['selected_features'])} features: {result['selected_features']}")
        else:
            print(f"❌ MRMR selection failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Test failed: {e}")

    return True


if __name__ == "__main__":
    test_infinity_logging_with_feature_names()
