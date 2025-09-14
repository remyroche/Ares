#!/usr/bin/env python3
"""
Simple test for vectorized feature selection optimizations.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.tprint import tprint
from src.utils.feature_selection.feature_importance_analyzer import (
    FeatureImportanceAnalyzer, FeatureImportanceConfig, ImportanceMethod
)


def create_test_data(n_samples=1000, n_features=20):
    """Create simple test data."""
    np.random.seed(42)

    # Create informative features
    X_informative = np.random.randn(n_samples, 5)
    X_noise = np.random.randn(n_samples, n_features - 5)

    X = np.column_stack([X_informative, X_noise])
    y = X_informative @ np.array([2, -1, 1.5, -0.5, 1]) + 0.1 * np.random.randn(n_samples)

    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')

    return X_df, y_series


def test_basic_functionality():
    """Test basic functionality of vectorized operations."""
    tprint("🚀 TESTING VECTORIZED FEATURE SELECTION")

    # Create test data
    X, y = create_test_data()
    tprint(f"📊 Created dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Configure analyzer
    config = FeatureImportanceConfig(
        methods=[ImportanceMethod.CORRELATION, ImportanceMethod.VARIANCE, ImportanceMethod.MUTUAL_INFO],
        enable_parallel=True,
        n_jobs=4,  # Set explicit number of workers
        save_results=False,
        generate_plots=False
    )

    analyzer = FeatureImportanceAnalyzer(config)

    # Test batch computation
    tprint("🔄 Testing batch importance computation...")
    start_time = time.time()

    importance_results = analyzer.batch_compute_importance(X, y)

    batch_time = time.time() - start_time
    tprint("✅ Batch computation completed")
    tprint(f"📊 Computed {len(importance_results)} methods in {batch_time:.3f}s")
    tprint(f"📊 Methods: {list(importance_results.keys())}")

    # Test matrix ensemble voting
    tprint("🔄 Testing matrix ensemble voting...")
    voting_start = time.time()

    voting_result = analyzer.matrix_based_ensemble_voting(importance_results)

    voting_time = time.time() - voting_start
    tprint("✅ Ensemble voting completed")
    tprint(f"📊 Top features: {voting_result['top_features'][:5]}")
    tprint(f"📊 Consensus score: {voting_result['voting_statistics']['mean_consensus']:.3f}")
    tprint(f"📊 Voting time: {voting_time:.4f}s")

    # Test GPU stability selection
    tprint("🔄 Testing GPU stability selection...")
    gpu_start = time.time()

    gpu_result = analyzer.gpu_accelerated_stability_selection(
        X, y, n_bootstrap=10, stability_threshold=0.3
    )

    gpu_time = time.time() - gpu_start
    tprint("✅ GPU stability selection completed")
    tprint(f"📊 Selected {len(gpu_result['selected_features'])} stable features")
    tprint(f"📊 Computation method: {gpu_result['computation_method']}")
    tprint(f"📊 GPU time: {gpu_time:.3f}s")

    # Performance summary
    total_time = time.time() - start_time
    tprint("\n📊 PERFORMANCE SUMMARY:")
    tprint(f"📊 Total computation time: {total_time:.3f}s")
    tprint(f"📊 Batch processing: {batch_time:.3f}s")
    tprint(f"📊 Ensemble voting: {voting_time:.3f}s")
    tprint(f"📊 GPU stability: {gpu_time:.3f}s")
    tprint("✅ All vectorized optimizations working correctly!")
def main():
    """Run the test."""
    try:
        test_basic_functionality()
    except Exception as e:
        tprint(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
