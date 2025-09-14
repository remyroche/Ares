#!/usr/bin/env python3
"""
Test script for vectorized Bayesian optimization improvements.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_vectorized_optimization():
    """Test the vectorized optimization functionality."""
    print("🧪 Testing Vectorized Bayesian Optimization")

    try:
        from src.utils.hmm_composite_manager import EnhancedHMMCompositeManager
        from src.utils.logger import system_logger

        # Create test data
        np.random.seed(42)
        n_samples = 1000
        n_features = 4

        # Generate synthetic time series data
        data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples).cumsum(),
            'feature_2': np.random.randn(n_samples).cumsum(),
            'feature_3': np.random.randn(n_samples).cumsum(),
            'feature_4': np.random.randn(n_samples).cumsum()
        })

        print(f"📊 Generated test data: {data.shape[0]} rows, {data.shape[1]} columns")

        # Initialize manager
        manager = EnhancedHMMCompositeManager()

        # Test vectorized optimization
        print("🚀 Testing vectorized optimization...")
        result_vec = manager.optimize_hmm_parameters(
            data, use_vectorized=True, mode='light', use_adaptive=False
        )

        print(f"✅ Vectorized result: success={result_vec.get('success', False)}")
        if result_vec.get('success'):
            print(f"   📊 Best score: {result_vec.get('best_score', 'N/A')}")
            print(f"   🔧 Best params: {result_vec.get('best_params', {})}")

        # Test sequential optimization for comparison
        print("📊 Testing sequential optimization...")
        result_seq = manager.optimize_hmm_parameters(
            data, use_vectorized=False, mode='light', use_adaptive=False
        )

        print(f"✅ Sequential result: success={result_seq.get('success', False)}")
        if result_seq.get('success'):
            print(f"   📊 Best score: {result_seq.get('best_score', 'N/A')}")
            print(f"   🔧 Best params: {result_seq.get('best_params', {})}")

        # Test benchmark functionality
        print("🏁 Testing benchmark functionality...")
        benchmark_result = manager.benchmark_optimization_performance(
            data, n_trials=3, mode='light'
        )

        print("✅ Benchmark completed")
        if 'comparison' in benchmark_result:
            comp = benchmark_result['comparison']
            if comp.get('speedup_ratio', float('inf')) != float('inf'):
                print(f"🚀 Speedup ratio: {comp['speedup_ratio']:.2f}x")
            if not np.isnan(comp.get('score_improvement', float('nan'))):
                print(f"📊 Score improvement: {comp['score_improvement']:+.1f}%")

        print("🎉 All tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vectorized_optimization()
    sys.exit(0 if success else 1)
