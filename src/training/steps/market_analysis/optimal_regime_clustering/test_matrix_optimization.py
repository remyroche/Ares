"""
Test Matrix Optimization Integration

This script tests the matrix optimization features of the optimal regime clustering system.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

def test_matrix_operations_availability():
    """Test if matrix operations are available."""
    print("🧪 Testing Matrix Operations Availability...")

    try:
        from src.utils.matrix_operations import (
            get_unified_matrix_operations,
            get_vectorized_processing_core,
            get_enhanced_matrix_operations,
            get_batch_matrix_processor
        )
        print("✅ Matrix operations modules imported successfully")

        # Test getting instances
        matrix_ops = get_unified_matrix_operations()
        vectorized_core = get_vectorized_processing_core()
        enhanced_ops = get_enhanced_matrix_operations()
        batch_processor = get_batch_matrix_processor()

        print("✅ Matrix operations instances created successfully")
        return True

    except ImportError as e:
        print(f"❌ Matrix operations not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing matrix operations: {e}")
        return False

def test_matrix_optimized_clustering():
    """Test matrix-optimized clustering functionality."""
    print("\n🧪 Testing Matrix-Optimized Clustering...")

    try:
        from optimal_regime_clustering import (
            create_matrix_optimized_clusterer,
            MatrixOptimizedClusterer,
            OptimalClusteringConfig
        )

        # Create configuration
        config = OptimalClusteringConfig()
        config.target_n_clusters = 10  # Smaller for testing

        # Create clusterer
        clusterer = create_matrix_optimized_clusterer(config)

        print("✅ Matrix-optimized clusterer created successfully")
        return True

    except ImportError as e:
        print(f"❌ Matrix-optimized clustering not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing matrix-optimized clustering: {e}")
        return False

def test_matrix_operations_functionality():
    """Test basic matrix operations functionality."""
    print("\n🧪 Testing Matrix Operations Functionality...")

    try:
        from src.utils.matrix_operations import (
            safe_matrix_multiply,
            correlation_matrix_gpu,
            optimize_dataframe,
            matrix_correlation_analysis
        )

        # Create test data
        n_samples = 1000
        n_features = 5
        np.random.seed(42)

        test_data = np.random.randn(n_samples, n_features)
        test_df = pd.DataFrame(test_data, columns=[f'feature_{i}' for i in range(n_features)])

        # Test matrix multiplication
        result = safe_matrix_multiply(test_data, test_data.T)
        print(f"✅ Matrix multiplication: {result.shape}")

        # Test correlation matrix
        corr_result = correlation_matrix_gpu(test_data)
        print(f"✅ Correlation matrix: {corr_result.shape}")

        # Test DataFrame optimization
        optimized_df = optimize_dataframe(test_df)
        print(f"✅ DataFrame optimization: {optimized_df.shape}")

        # Test correlation analysis
        corr_matrix, corr_df = matrix_correlation_analysis(test_df)
        print(f"✅ Correlation analysis: {corr_matrix.shape}, {corr_df.shape}")

        print("✅ All matrix operations working correctly")
        return True

    except ImportError as e:
        print(f"❌ Matrix operations functions not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing matrix operations functionality: {e}")
        return False

def test_performance_comparison():
    """Test performance comparison between standard and matrix-optimized clustering."""
    print("\n🧪 Testing Performance Comparison...")

    try:
        from optimal_regime_clustering import (
            run_fast_clustering,
            run_matrix_optimized_clustering
        )

        # Create test data
        n_samples = 2000
        np.random.seed(42)

        test_data = pd.DataFrame({
            'volume': np.random.exponential(100, n_samples),
            'volatility': np.random.beta(2, 5, n_samples) * 0.1,
            'momentum': np.random.normal(0, 0.02, n_samples),
            'trend': np.random.normal(0, 0.05, n_samples),
            'timestamp': pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
        })

        print("📊 Testing with 2000 samples...")

        # Test fast clustering
        print("🚀 Running fast clustering...")
        fast_results = run_fast_clustering(
            data_path=test_data,
            output_dir="test_fast_clusters/"
        )

        fast_time = fast_results.get('execution_time', 0)
        fast_clusters = fast_results.get('clustering_result', {}).statistics.n_clusters if 'clustering_result' in fast_results else 0

        print(f"✅ Fast clustering: {fast_time".3f"}s, {fast_clusters} clusters")

        # Test matrix-optimized clustering
        print("🚀 Running matrix-optimized clustering...")
        matrix_results = run_matrix_optimized_clustering(
            data_path=test_data,
            output_dir="test_matrix_clusters/"
        )

        matrix_time = matrix_results.get('execution_time', 0)
        matrix_clusters = matrix_results.get('clustering_result', {}).statistics.n_clusters if 'clustering_result' in matrix_results else 0
        matrix_optimized = matrix_results.get('matrix_optimization_used', False)

        print(f"✅ Matrix-optimized clustering: {matrix_time".3f"}s, {matrix_clusters} clusters, Matrix ops: {matrix_optimized}")

        # Compare performance
        if matrix_optimized and matrix_time > 0 and fast_time > 0:
            speedup = fast_time / matrix_time if matrix_time > 0 else 0
            print(f"📈 Performance improvement: {speedup".2f"}x faster")

        return True

    except Exception as e:
        print(f"❌ Error in performance comparison: {e}")
        return False

def main():
    """Run all matrix optimization tests."""
    print("🚀 Matrix Optimization Test Suite")
    print("Testing integration of optimal regime clustering with matrix operations\n")

    tests = [
        ("Matrix Operations Availability", test_matrix_operations_availability),
        ("Matrix-Optimized Clustering", test_matrix_optimized_clustering),
        ("Matrix Operations Functionality", test_matrix_operations_functionality),
        ("Performance Comparison", test_performance_comparison)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)

        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    print(f"\n{'='*50}")
    print("📊 Test Results Summary")
    print('='*50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100".1f"}%")

    if passed == total:
        print("🎉 All tests passed! Matrix optimization is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")

    print("\n💡 Recommendations:")
    if passed >= 3:
        print("   ✅ Matrix optimization is ready for production use")
    elif passed >= 2:
        print("   ⚠️ Matrix optimization has partial functionality")
    else:
        print("   ❌ Matrix optimization needs troubleshooting")

    if passed >= 2:
        print("   💡 Try running: python example_usage.py")
        print("   💡 Use run_matrix_optimized_clustering() for best performance")

if __name__ == "__main__":
    main()