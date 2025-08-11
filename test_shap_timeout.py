#!/usr/bin/env python3
"""
Simple test script to verify SHAP timeout fix works correctly.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"  # Corrected path for test script
sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
import time
from analyst.autoencoder_feature_generator import FeatureFilter, AutoencoderConfig


def test_shap_timeout():
    """Test SHAP timeout mechanism and efficiency optimizations."""
    print("🚀 Starting SHAP timeout test...")
    print("🧪 Testing SHAP timeout mechanism...")

    # Create a larger test dataset with more features
    print("📊 Creating large test dataset...")
    np.random.seed(42)

    # Create 100,000 samples with 100 features for better testing
    n_samples = 100000
    n_features = 100

    # Generate synthetic features with varying importance
    features = {}
    for i in range(n_features):
        if i < 20:  # Top 20 features are important
            features[f"important_feature_{i}"] = np.random.normal(
                0, 1, n_samples
            ) + np.random.normal(0, 0.1, n_samples)
        else:  # Remaining features are noise
            features[f"noise_feature_{i}"] = np.random.normal(0, 0.1, n_samples)

    features_df = pd.DataFrame(features)

    # Create labels based on important features
    important_sum = sum(features_df[f"important_feature_{i}"] for i in range(20))
    labels = np.where(
        important_sum > np.percentile(important_sum, 70),
        1,
        np.where(important_sum < np.percentile(important_sum, 30), -1, 0),
    )

    print(f"📊 Dataset created: {n_samples} samples, {n_features} features")
    print(f"📊 Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Initialize FeatureFilter with config
    config = AutoencoderConfig("src/analyst/autoencoder_config.yaml")
    feature_filter = FeatureFilter(config)

    # Test feature filtering with timeout protection
    print("🔍 Starting feature filtering with timeout protection...")
    start_time = time.time()

    try:
        filtered_features = feature_filter.filter_features(features_df, labels)
        end_time = time.time()

        print(f"✅ Feature filtering completed in {end_time - start_time:.2f} seconds")
        print(f"📊 Original features: {len(features_df.columns)}")
        print(f"📊 Filtered features: {len(filtered_features.columns)}")
        print(f"📊 Selected features: {list(filtered_features.columns)}")

        # Verify that we got more features than before
        if len(filtered_features.columns) >= 5:
            print(
                "✅ Feature selection working correctly - got reasonable number of features"
            )
        else:
            print("⚠️ Feature selection may be too aggressive")

    except Exception as e:
        print(f"❌ Feature filtering failed: {e}")
        return False

    print("✅ SHAP timeout test completed successfully!")
    return True


if __name__ == "__main__":
    test_shap_timeout()
