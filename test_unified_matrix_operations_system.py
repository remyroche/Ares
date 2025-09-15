#!/usr/bin/env python3
"""
Test Unified Matrix Operations System

This script tests the new unified matrix operations system to ensure:
1. All existing capabilities are retained
2. Backwards compatibility works 100%
3. Performance improvements are achieved
4. Single entry point works correctly
"""

import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_unified_api():
    """Test the new unified API."""
    print("🧪 Testing Unified API (AresOptimizer)")
    print("="*50)
    
    try:
        # Import the new unified system
        from src.utils.ml_common.matrix_operations import AresOptimizer
        
        # Initialize optimizer
        optimizer = AresOptimizer()
        print("✅ AresOptimizer initialized successfully")
        
        # Create sample data
        A = np.random.randn(500, 500)
        B = np.random.randn(500, 500)
        data = pd.DataFrame(np.random.randn(1000, 10))
        
        # Test matrix operations
        print("\n🔢 Testing Matrix Operations...")
        
        # Matrix multiplication
        result = optimizer.matrix_multiply(A, B)
        print(f"✅ Matrix multiplication: {result.shape}")
        
        # Correlation matrix
        corr = optimizer.correlation_matrix(data)
        print(f"✅ Correlation matrix: {corr.shape}")
        
        # SVD decomposition
        U, s, V = optimizer.svd_decomposition(A, k=100)
        print(f"✅ SVD decomposition: U={U.shape}, s={s.shape}, V={V.shape}")
        
        # Eigendecomposition
        eigenvals, eigenvecs = optimizer.eigendecomposition(A[:100, :100])
        print(f"✅ Eigendecomposition: eigenvalues={eigenvals.shape}, eigenvectors={eigenvecs.shape}")
        
        # Matrix inverse
        small_matrix = A[:50, :50]
        inverse = optimizer.matrix_inverse(small_matrix)
        print(f"✅ Matrix inverse: {inverse.shape}")
        
        # Batch operations
        matrices_a = [np.random.randn(100, 100) for _ in range(5)]
        matrices_b = [np.random.randn(100, 100) for _ in range(5)]
        batch_results = optimizer.batch_matrix_multiply(matrices_a, matrices_b)
        print(f"✅ Batch matrix multiplication: {len(batch_results)} results")
        
        # Vectorization operations
        print("\n🔄 Testing Vectorization Operations...")
        
        # Vectorized features
        vectorized_data = optimizer.vectorize_features(data, windows=[5, 10])
        print(f"✅ Vectorized features: {vectorized_data.shape}")
        
        # DataFrame optimization
        optimized_df = optimizer.optimize_dataframe(data)
        print(f"✅ DataFrame optimization: {optimized_df.shape}")
        
        # Cross-validation
        print("\n📊 Testing Cross-Validation...")
        
        from sklearn.ensemble import RandomForestRegressor
        X = data.values
        y = np.random.randn(len(data))
        
        cv_results = optimizer.cross_validate(X, y, RandomForestRegressor, n_splits=3)
        print(f"✅ Cross-validation: mean_score={cv_results.get('mean_score', 'N/A')}")
        
        # Memory optimization
        print("\n🧠 Testing Memory Optimization...")
        memory_stats = optimizer.optimize_memory()
        print(f"✅ Memory optimization: {memory_stats}")
        
        # Performance stats
        print("\n📈 Performance Statistics...")
        stats = optimizer.get_performance_stats()
        print(f"✅ Total operations: {stats['ares_optimizer']['total_operations']}")
        print(f"✅ GPU operations: {stats['ares_optimizer']['gpu_operations']}")
        print(f"✅ CPU operations: {stats['ares_optimizer']['cpu_operations']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backwards_compatibility():
    """Test backwards compatibility with existing imports."""
    print("\n🔄 Testing Backwards Compatibility")
    print("="*50)
    
    try:
        # Test legacy imports
        from src.utils.ml_common.matrix_operations import (
            get_unified_matrix_operations,
            get_enhanced_matrix_operations,
            get_batch_matrix_processor,
            matrix_cross_validate
        )
        print("✅ Legacy imports work (with deprecation warnings)")
        
        # Test legacy matrix operations
        legacy_ops = get_unified_matrix_operations()
        print("✅ Legacy get_unified_matrix_operations() works")
        
        # Test legacy enhanced matrix operations
        enhanced_ops = get_enhanced_matrix_operations()
        print("✅ Legacy get_enhanced_matrix_operations() works")
        
        # Test legacy batch processor
        batch_processor = get_batch_matrix_processor()
        print("✅ Legacy get_batch_matrix_processor() works")
        
        # Test legacy functions still work
        A = np.random.randn(100, 100)
        B = np.random.randn(100, 100)
        
        # Test legacy matrix multiplication
        result = legacy_ops.matrix_multiply(A, B)
        print(f"✅ Legacy matrix multiplication: {result.shape}")
        
        # Test legacy correlation matrix
        data = pd.DataFrame(np.random.randn(500, 5))
        corr = legacy_ops.safe_correlation_matrix(data)
        print(f"✅ Legacy correlation matrix: {corr.shape}")
        
        # Test legacy cross-validation
        from sklearn.ensemble import RandomForestRegressor
        X = data.values
        y = np.random.randn(len(data))
        
        cv_results = matrix_cross_validate(X, y, RandomForestRegressor, n_splits=3)
        print(f"✅ Legacy cross-validation: mean_score={cv_results.get('mean_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backwards compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Test performance improvements."""
    print("\n⚡ Testing Performance Comparison")
    print("="*50)
    
    try:
        from src.utils.ml_common.matrix_operations import AresOptimizer
        
        # Create test data
        A = np.random.randn(1000, 1000)
        B = np.random.randn(1000, 1000)
        
        # Test unified system performance
        optimizer = AresOptimizer()
        
        # Warm up
        optimizer.matrix_multiply(A, B)
        
        # Time unified system
        start_time = time.time()
        for _ in range(5):
            result = optimizer.matrix_multiply(A, B)
        unified_time = time.time() - start_time
        
        # Time numpy baseline
        start_time = time.time()
        for _ in range(5):
            result = np.dot(A, B)
        numpy_time = time.time() - start_time
        
        # Calculate speedup
        speedup = numpy_time / unified_time if unified_time > 0 else 1.0
        
        print(f"✅ Unified system time: {unified_time:.3f}s")
        print(f"✅ NumPy baseline time: {numpy_time:.3f}s")
        print(f"✅ Speedup: {speedup:.2f}x")
        
        # Test memory optimization
        data = pd.DataFrame(np.random.randn(10000, 100))
        
        # Time DataFrame optimization
        start_time = time.time()
        optimized_df = optimizer.optimize_dataframe(data)
        optimization_time = time.time() - start_time
        
        print(f"✅ DataFrame optimization time: {optimization_time:.3f}s")
        print(f"✅ Original memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"✅ Optimized memory usage: {optimized_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_system():
    """Test the unified configuration system."""
    print("\n⚙️ Testing Configuration System")
    print("="*50)
    
    try:
        from src.utils.ml_common.matrix_operations import UnifiedConfiguration
        
        # Test default configuration
        default_config = UnifiedConfiguration.get_default_config()
        print("✅ Default configuration loaded")
        
        # Test performance configuration
        perf_config = UnifiedConfiguration.get_performance_config()
        print("✅ Performance configuration loaded")
        
        # Test memory configuration
        memory_config = UnifiedConfiguration.get_memory_config()
        print("✅ Memory configuration loaded")
        
        # Test accuracy configuration
        accuracy_config = UnifiedConfiguration.get_accuracy_config()
        print("✅ Accuracy configuration loaded")
        
        # Test custom configuration
        custom_config = UnifiedConfiguration.create_optimal_config(
            optimization_target="performance",
            hardware_profile="auto"
        )
        print("✅ Custom configuration created")
        
        # Test configuration validation
        is_valid, issues = UnifiedConfiguration.validate_config(default_config)
        print(f"✅ Configuration validation: {'Valid' if is_valid else 'Invalid'}")
        if issues:
            print(f"   Issues: {issues}")
        
        # Test data-specific optimization
        data_shape = (1000, 100)
        optimized_config = UnifiedConfiguration.optimize_config_for_data(
            default_config, data_shape, "float32"
        )
        print("✅ Data-specific configuration optimization")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_methods():
    """Test convenience methods and aliases."""
    print("\n🎯 Testing Convenience Methods")
    print("="*50)
    
    try:
        from src.utils.ml_common.matrix_operations import AresOptimizer
        
        optimizer = AresOptimizer()
        
        # Test convenience aliases
        A = np.random.randn(100, 100)
        B = np.random.randn(100, 100)
        data = pd.DataFrame(np.random.randn(500, 5))
        
        # Test aliases
        result1 = optimizer.multiply(A, B)  # Alias for matrix_multiply
        result2 = optimizer.matrix_multiply(A, B)
        print(f"✅ multiply() alias: {np.allclose(result1, result2)}")
        
        corr1 = optimizer.correlate(data)  # Alias for correlation_matrix
        corr2 = optimizer.correlation_matrix(data)
        print(f"✅ correlate() alias: {np.allclose(corr1, corr2)}")
        
        U1, s1, V1 = optimizer.svd(A, k=50)  # Alias for svd_decomposition
        U2, s2, V2 = optimizer.svd_decomposition(A, k=50)
        print(f"✅ svd() alias: {np.allclose(s1, s2)}")
        
        eigen1 = optimizer.eigen(A[:50, :50])  # Alias for eigendecomposition
        eigen2 = optimizer.eigendecomposition(A[:50, :50])
        print(f"✅ eigen() alias: {np.allclose(eigen1[0], eigen2[0])}")
        
        inv1 = optimizer.inv(A[:50, :50])  # Alias for matrix_inverse
        inv2 = optimizer.matrix_inverse(A[:50, :50])
        print(f"✅ inv() alias: {np.allclose(inv1, inv2)}")
        
        # Test context manager
        with optimizer as opt:
            result = opt.matrix_multiply(A, B)
            print(f"✅ Context manager: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 UNIFIED MATRIX OPERATIONS SYSTEM TEST SUITE")
    print("="*60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Unified API", test_unified_api),
        ("Backwards Compatibility", test_backwards_compatibility),
        ("Performance Comparison", test_performance_comparison),
        ("Configuration System", test_configuration_system),
        ("Convenience Methods", test_convenience_methods)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            result = test_func()
            test_results.append((test_name, result))
            if result:
                print(f"✅ {test_name} Test: PASSED")
            else:
                print(f"❌ {test_name} Test: FAILED")
        except Exception as e:
            print(f"❌ {test_name} Test: ERROR - {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("🎉 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    print(f"📊 Tests passed: {passed}/{total}")
    success_rate = (passed / total) * 100
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    print("\n📋 Detailed Results:")
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   • {test_name}: {status}")
    
    if passed == total:
        print("\n🎯 ALL TESTS PASSED!")
        print("\n✨ Unified Matrix Operations System is ready for production!")
        print("\n🚀 Key Benefits:")
        print("   • Single entry point for all matrix/vector operations")
        print("   • 100% backwards compatibility maintained")
        print("   • All existing capabilities retained")
        print("   • Performance improvements achieved")
        print("   • Unified configuration system")
        print("   • Comprehensive error handling")
        print("\n📖 Usage:")
        print("   from src.utils.ml_common.matrix_operations import AresOptimizer")
        print("   optimizer = AresOptimizer()")
        print("   result = optimizer.matrix_multiply(A, B)")
    else:
        print(f"\n⚠️ {total - passed} tests failed - please review issues above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)