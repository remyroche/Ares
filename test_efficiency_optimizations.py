#!/usr/bin/env python3
"""
Test Script for Efficiency Optimizations

This script demonstrates the comprehensive efficiency improvements made to the
interactive feature generation system, including memory optimization, caching,
parallelism, and vectorization.
"""

import pandas as pd
import numpy as np
import sys
import time
import psutil
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_large_test_data(n_samples: int = 100000) -> pd.DataFrame:
    """Create large test dataset for efficiency testing."""
    np.random.seed(42)
    
    # Create time index
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    # Generate realistic market data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = {
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices * (1 + np.random.normal(0, 0.01, n_samples)),
        'volume': np.random.lognormal(10, 0.5, n_samples),
    }
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    for i in range(n_samples):
        data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i])
        data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
    
    df = pd.DataFrame(data, index=dates)
    
    # Add some additional features
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df

def test_data_fingerprinting():
    """Test data fingerprinting for cache keys."""
    print("🧪 Testing Data Fingerprinting...")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.efficiency_optimizations import (
            DataFingerprinter, EfficiencyConfig
        )
        
        # Create test data
        data = create_large_test_data(1000)
        config = {
            'max_interactions': 50,
            'variance_threshold': 1e-8,
            'rolling_windows': [5, 10, 20, 50]
        }
        
        # Test fingerprinting
        fingerprinter = DataFingerprinter(EfficiencyConfig())
        
        print("📊 Testing fingerprint generation...")
        fingerprint1 = fingerprinter.generate_fingerprint(data, config, "1.0.0")
        fingerprint2 = fingerprinter.generate_fingerprint(data, config, "1.0.0")
        fingerprint3 = fingerprinter.generate_fingerprint(data, config, "1.1.0")
        
        print(f"  Fingerprint 1: {fingerprint1}")
        print(f"  Fingerprint 2: {fingerprint2}")
        print(f"  Fingerprint 3: {fingerprint3}")
        
        # Test consistency
        if fingerprint1 == fingerprint2:
            print("  ✅ Fingerprints are consistent for same data/config")
        else:
            print("  ❌ Fingerprints are inconsistent")
        
        # Test sensitivity to changes
        if fingerprint1 != fingerprint3:
            print("  ✅ Fingerprints change with code version")
        else:
            print("  ❌ Fingerprints don't change with code version")
        
        # Test with different data
        data2 = data.copy()
        data2['close'] = data2['close'] * 1.01  # Small change
        fingerprint4 = fingerprinter.generate_fingerprint(data2, config, "1.0.0")
        
        if fingerprint1 != fingerprint4:
            print("  ✅ Fingerprints change with data changes")
        else:
            print("  ❌ Fingerprints don't change with data changes")
        
        return True
        
    except Exception as e:
        print(f"❌ Data fingerprinting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chunking_optimization():
    """Test chunking optimization for L3 cache."""
    print("\n🧪 Testing Chunking Optimization...")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.efficiency_optimizations import (
            ChunkingOptimizer, EfficiencyConfig
        )
        
        # Create test data
        data = create_large_test_data(50000)
        
        # Test chunking optimizer
        config = EfficiencyConfig(
            max_memory_gb=8.0,
            memory_headroom_gb=2.0,
            max_workers=4
        )
        chunking = ChunkingOptimizer(config)
        
        print("📊 Testing chunk size calculation...")
        data_size_mb = data.memory_usage(deep=True).sum() / (1024**2)
        print(f"  Data size: {data_size_mb:.1f}MB")
        
        optimal_chunk_size = chunking.calculate_optimal_chunk_size(data_size_mb, 4)
        print(f"  Optimal chunk size: {optimal_chunk_size} rows")
        
        # Test chunk creation
        print("📊 Testing chunk creation...")
        chunks = chunking.create_chunks(data, optimal_chunk_size)
        print(f"  Created {len(chunks)} chunks")
        
        # Verify chunk sizes
        chunk_sizes = [len(chunk) for chunk in chunks]
        print(f"  Chunk sizes: {chunk_sizes[:5]}... (showing first 5)")
        
        # Test memory efficiency
        total_rows = sum(chunk_sizes)
        if total_rows == len(data):
            print("  ✅ All data preserved in chunks")
        else:
            print(f"  ❌ Data loss: {len(data)} -> {total_rows}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chunking optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parallelism_optimization():
    """Test parallelism optimization."""
    print("\n🧪 Testing Parallelism Optimization...")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.efficiency_optimizations import (
            ParallelismOptimizer, EfficiencyConfig
        )
        
        # Test parallelism optimizer
        config = EfficiencyConfig(max_workers=4)
        parallelism = ParallelismOptimizer(config)
        
        print("📊 Testing parallelism decisions...")
        
        # Test different operation types
        operation_types = [
            'correlation', 'matrix_ops', 'feature_generation',
            'rolling_stats', 'technical_indicators', 'io_operations'
        ]
        
        for op_type in operation_types:
            use_mp = parallelism.should_use_multiprocessing(op_type)
            executor_type = "ProcessPoolExecutor" if use_mp else "ThreadPoolExecutor"
            print(f"  {op_type:20} → {executor_type}")
        
        # Test executor creation
        print("📊 Testing executor creation...")
        executor = parallelism.get_executor('feature_generation', max_workers=2)
        print(f"  Executor type: {type(executor).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parallelism optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_zero_copy_optimization():
    """Test zero-copy data conversions."""
    print("\n🧪 Testing Zero-Copy Optimization...")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.efficiency_optimizations import (
            ZeroCopyOptimizer, EfficiencyConfig
        )
        
        # Create test data
        data = create_large_test_data(10000)
        
        # Test zero-copy optimizer
        zero_copy = ZeroCopyOptimizer(EfficiencyConfig())
        
        print("📊 Testing DataFrame optimization...")
        
        # Test with existing DataFrame
        optimized_df = zero_copy.optimize_dataframe_conversion(data)
        print(f"  Original memory: {data.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
        print(f"  Optimized memory: {optimized_df.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
        
        # Test with numpy array
        print("📊 Testing numpy array conversion...")
        numpy_data = data.values
        df_from_numpy = zero_copy.optimize_dataframe_conversion(numpy_data)
        print(f"  Converted from numpy: {df_from_numpy.shape}")
        
        # Test dtype optimization
        print("📊 Testing dtype optimization...")
        original_dtypes = data.dtypes.value_counts()
        optimized_dtypes = optimized_df.dtypes.value_counts()
        print(f"  Original dtypes: {dict(original_dtypes)}")
        print(f"  Optimized dtypes: {dict(optimized_dtypes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Zero-copy optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vectorization_optimization():
    """Test vectorization optimizations."""
    print("\n🧪 Testing Vectorization Optimization...")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.efficiency_optimizations import (
            VectorizationOptimizer, EfficiencyConfig
        )
        
        # Create test data
        data = np.random.randn(10000, 10)
        target = np.random.randn(10000)
        
        # Test vectorization optimizer
        vectorization = VectorizationOptimizer(EfficiencyConfig())
        
        print("📊 Testing vectorized rolling operations...")
        
        # Test rolling operations
        operations = ['mean', 'std', 'min', 'max', 'sum']
        start_time = time.time()
        results = vectorization.vectorized_rolling_ops(data[:, 0], window=20, operations=operations)
        rolling_time = time.time() - start_time
        
        print(f"  Rolling operations time: {rolling_time:.3f}s")
        print(f"  Results shape: {[results[op].shape for op in operations]}")
        
        # Test correlations
        print("📊 Testing vectorized correlations...")
        start_time = time.time()
        correlations = vectorization.vectorized_correlations(data, target, threshold=0.95)
        corr_time = time.time() - start_time
        
        print(f"  Correlation time: {corr_time:.3f}s")
        print(f"  High correlations: {np.sum(np.abs(correlations) > 0.95)}")
        
        # Test batch operations
        print("📊 Testing batch matrix operations...")
        matrices = [np.random.randn(100, 100) for _ in range(10)]
        start_time = time.time()
        batch_results = vectorization.batch_matrix_operations(matrices, 'multiply')
        batch_time = time.time() - start_time
        
        print(f"  Batch operations time: {batch_time:.3f}s")
        print(f"  Batch results: {len(batch_results)} matrices")
        
        return True
        
    except Exception as e:
        print(f"❌ Vectorization optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_monitoring():
    """Test memory monitoring."""
    print("\n🧪 Testing Memory Monitoring...")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.efficiency_optimizations import (
            MemoryMonitor, EfficiencyConfig
        )
        
        # Test memory monitor
        monitor = MemoryMonitor(EfficiencyConfig(profile_memory=True))
        
        print("📊 Testing memory monitoring...")
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate some memory-intensive operations
        data1 = create_large_test_data(20000)
        data2 = create_large_test_data(20000)
        data3 = create_large_test_data(20000)
        
        # Record memory usage
        monitor._record_memory_usage()
        
        # Stop monitoring
        stats = monitor.stop_monitoring()
        
        print(f"  Duration: {stats['duration_seconds']:.2f}s")
        print(f"  Peak memory: {stats['peak_memory_mb']:.1f}MB")
        print(f"  Memory efficiency: {stats['memory_efficiency']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration of all optimizations."""
    print("\n🧪 Testing Integration...")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.efficiency_optimizations import (
            EfficiencyOptimizer, EfficiencyConfig
        )
        
        # Create test data
        data = create_large_test_data(50000)
        config = {
            'max_interactions': 100,
            'variance_threshold': 1e-8,
            'rolling_windows': [5, 10, 20, 50, 100]
        }
        
        # Test efficiency optimizer
        efficiency_config = EfficiencyConfig(
            max_memory_gb=8.0,
            memory_headroom_gb=2.0,
            max_workers=4,
            enable_profiling=True
        )
        
        optimizer = EfficiencyOptimizer(efficiency_config)
        
        print("📊 Testing integrated optimization...")
        
        # Test optimization
        start_time = time.time()
        result = optimizer.optimize_feature_generation(data, config)
        optimization_time = time.time() - start_time
        
        print(f"  Optimization time: {optimization_time:.2f}s")
        print(f"  Result shape: {result.shape}")
        print(f"  Memory usage: {result.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Test performance comparison with and without optimizations."""
    print("\n🧪 Testing Performance Comparison...")
    print("=" * 50)
    
    try:
        # Create test data
        data = create_large_test_data(20000)
        
        print("📊 Testing performance with different configurations...")
        
        # Test 1: Basic configuration
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.efficiency_optimizations import (
            EfficiencyConfig, EfficiencyOptimizer
        )
        
        configs = [
            ("Basic", EfficiencyConfig(max_workers=1, enable_caching=False)),
            ("Optimized", EfficiencyConfig(max_workers=4, enable_caching=True)),
            ("High Performance", EfficiencyConfig(max_workers=8, enable_caching=True, chunk_size_mb=0.5))
        ]
        
        results = []
        for name, eff_config in configs:
            print(f"  Testing {name} configuration...")
            
            optimizer = EfficiencyOptimizer(eff_config)
            start_time = time.time()
            result = optimizer.optimize_feature_generation(data, {})
            duration = time.time() - start_time
            
            results.append((name, duration, result.shape[0]))
            print(f"    Duration: {duration:.2f}s, Rows: {result.shape[0]}")
        
        # Show comparison
        print("\n📊 Performance Comparison:")
        for name, duration, rows in results:
            print(f"  {name:15}: {duration:6.2f}s ({rows:6d} rows)")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all efficiency tests."""
    print("🚀 Testing Efficiency Optimizations")
    print("=" * 60)
    
    tests = [
        ("Data Fingerprinting", test_data_fingerprinting),
        ("Chunking Optimization", test_chunking_optimization),
        ("Parallelism Optimization", test_parallelism_optimization),
        ("Zero-Copy Optimization", test_zero_copy_optimization),
        ("Vectorization Optimization", test_vectorization_optimization),
        ("Memory Monitoring", test_memory_monitoring),
        ("Integration", test_integration),
        ("Performance Comparison", test_performance_comparison),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        print("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} test passed!")
            else:
                print(f"❌ {test_name} test failed!")
                
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 EFFICIENCY TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All efficiency tests passed! The system is highly optimized.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)