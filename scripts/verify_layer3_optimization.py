
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.training.steps.labeling.layer3.feature_engineering import enhance_layer3_features_optimized, hierarchical_feature_filtering, downcast_float
from src.training.steps.labeling.layer3.core import apply_mild_mp_clustering, integrate_entropy_bars_into_layer3
from src.utils.tprint import tprint_info, tprint_success, tprint_error

def test_feature_engineering():
    tprint_info("Testing Feature Engineering optimizations...")

    # Create dummy data
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1min')
    df = pd.DataFrame(index=dates)
    market_data = pd.DataFrame({
        'open': np.random.randn(1000) + 100,
        'high': np.random.randn(1000) + 101,
        'low': np.random.randn(1000) + 99,
        'close': np.random.randn(1000) + 100,
        'volume': np.random.rand(1000) * 1000,
        'volatility_1d': np.random.rand(1000) * 0.01
    }, index=dates)

    layer1_weight = np.random.rand(1000)

    # Test enhance_layer3_features_optimized
    try:
        df_enhanced = enhance_layer3_features_optimized(df, market_data, layer1_weight, fast_mode=True)
        tprint_success(f"Enhanced features (Fast Mode): {df_enhanced.shape}")

        df_enhanced_full = enhance_layer3_features_optimized(df, market_data, layer1_weight, fast_mode=False)
        tprint_success(f"Enhanced features (Full Mode): {df_enhanced_full.shape}")

        # Check dtypes
        float_cols = df_enhanced_full.select_dtypes(include=['float64']).columns
        if len(float_cols) == 0:
             tprint_success("All float columns are float32 (Downcasting successful)")
        else:
             tprint_error(f"Float64 columns found: {float_cols}")

    except Exception as e:
        tprint_error(f"Enhance features failed: {e}")
        raise e

    # Test hierarchical_feature_filtering
    try:
        # Create correlated features
        X = pd.DataFrame(np.random.randn(100, 10).astype(np.float32), columns=[f'f{i}' for i in range(10)])
        X['f10'] = X['f0'] * 0.99 + np.random.randn(100) * 0.01 # Highly correlated
        y = pd.Series(np.random.randint(0, 2, 100))
        base_avg = pd.Series(np.random.rand(100))

        X_filtered = hierarchical_feature_filtering(X, y, base_avg)
        tprint_success(f"Filtered features: {X_filtered.shape}")

    except Exception as e:
        tprint_error(f"Hierarchical filtering failed: {e}")
        raise e

def test_core_clustering():
    tprint_info("Testing Core Clustering optimizations...")
    try:
        # Create highly correlated feature matrix
        X = pd.DataFrame(np.random.randn(100, 50).astype(np.float32))
        # Add perfectly correlated columns
        X[50] = X[0]
        X[51] = X[1]

        X_clustered = apply_mild_mp_clustering(X, threshold=0.98)
        tprint_success(f"Clustered features: {X_clustered.shape} (Should be <= 50)")

        if X_clustered.shape[1] > 50:
             tprint_error("Clustering failed to remove duplicates")
        else:
             tprint_success("Clustering removed duplicates")

    except Exception as e:
        tprint_error(f"Clustering failed: {e}")
        raise e

if __name__ == "__main__":
    test_feature_engineering()
    test_core_clustering()
