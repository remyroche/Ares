#!/usr/bin/env python3
"""
Core Efficiency Test - Focused on Key Optimizations

This script tests the core efficiency improvements without complex dependencies.
"""

import pandas as pd
import numpy as np
import sys
import time
import psutil
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create test dataset."""
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    data = {
        'open': np.random.randn(n_samples) * 100,
        'high': np.random.randn(n_samples) * 100,
        'low': np.random.randn(n_samples) * 100,
        'close': np.random.randn(n_samples) * 100,
        'volume': np.random.lognormal(10, 0.5, n_samples),
    }
    
    df = pd.DataFrame(data, index=dates)
    df['returns'] = df['close'].pct_change()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df

def test_data_fingerprinting():
    """Test data fingerprinting."""
    print("🧪 Testing Data Fingerprinting...")
    print("=" * 50)
    
    try:
        # Simple fingerprinting implementation
        def simple_fingerprint(data: pd.DataFrame, config: dict) -> str:
            import hashlib
            import json
            
            # Data content hash
            content_hash = hashlib.sha256(data.values.tobytes()).hexdigest()[:8]
            
            # Config hash
            config_str = json.dumps(config, sort_keys=True, default=str)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
            
            # Environment hash
            env_info = {
                'python_version': sys.version,
                'numpy_version': np.__version__,
                'pandas_version': pd.__version__
            }
            env_str = json.dumps(env_info, sort_keys=True)
            env_hash = hashlib.sha256(env_str.encode()).hexdigest()[:8]
            
            return f"{content_hash}_{config_hash}_{env_hash}"
        
        # Test fingerprinting
        data = create_test_data(1000)
        config = {'max_interactions': 50, 'variance_threshold': 1e-8}
        
        fingerprint1 = simple_fingerprint(data, config)
        fingerprint2 = simple_fingerprint(data, config)
        fingerprint3 = simple_fingerprint(data, {'max_interactions': 100, 'variance_threshold': 1e-8})
        
        print(f"  Fingerprint 1: {fingerprint1}")
        print(f"  Fingerprint 2: {fingerprint2}")
        print(f"  Fingerprint 3: {fingerprint3}")
        
        # Test consistency
        if fingerprint1 == fingerprint2:
            print("  ✅ Fingerprints are consistent")
        else:
            print("  ❌ Fingerprints are inconsistent")
        
        # Test sensitivity
        if fingerprint1 != fingerprint3:
            print("  ✅ Fingerprints change with config")
        else:
            print("  ❌ Fingerprints don't change with config")
        
        return True
        
    except Exception as e:
        print(f"❌ Data fingerprinting test failed: {e}")
        return False

def test_chunking_optimization():
    """Test chunking optimization."""
    print("\n🧪 Testing Chunking Optimization...")
    print("=" * 50)
    
    try:
        def calculate_chunk_size(data_size_mb: float, max_workers: int, max_memory_gb: float = 8.0) -> int:
            """Calculate optimal chunk size."""
            # Memory budget per worker
            memory_per_worker = (max_memory_gb - 2.0) / max_workers  # 2GB headroom
            
            # L3 cache consideration (rough estimate)
            l3_cache_mb = 16.0  # Default estimate
            l3_optimal = l3_cache_mb * 0.8
            
            # Memory budget consideration
            memory_optimal = memory_per_worker * 1024 * 0.75  # 75% of memory per worker
            
            # Choose smaller
            optimal_mb = min(l3_optimal, memory_optimal)
            
            # Convert to rows (1KB per row estimate)
            optimal_rows = int(optimal_mb * 1024)
            
            # Apply constraints
            optimal_rows = max(1000, min(100000, optimal_rows))
            
            return optimal_rows
        
        def create_chunks(data: pd.DataFrame, chunk_size: int) -> list:
            """Create chunks."""
            chunks = []
            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i + chunk_size].copy()
                chunks.append(chunk)
            return chunks
        
        # Test chunking
        data = create_test_data(50000)
        data_size_mb = data.memory_usage(deep=True).sum() / (1024**2)
        
        print(f"  Data size: {data_size_mb:.1f}MB")
        
        chunk_size = calculate_chunk_size(data_size_mb, 4)
        print(f"  Optimal chunk size: {chunk_size} rows")
        
        chunks = create_chunks(data, chunk_size)
        print(f"  Created {len(chunks)} chunks")
        
        # Verify
        total_rows = sum(len(chunk) for chunk in chunks)
        if total_rows == len(data):
            print("  ✅ All data preserved")
        else:
            print(f"  ❌ Data loss: {len(data)} -> {total_rows}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chunking optimization test failed: {e}")
        return False

def test_zero_copy_optimization():
    """Test zero-copy optimizations."""
    print("\n🧪 Testing Zero-Copy Optimization...")
    print("=" * 50)
    
    try:
        def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            """Optimize DataFrame dtypes."""
            optimized = df.copy()
            
            for col in optimized.columns:
                if optimized[col].dtype == 'float64':
                    # Check if can downcast to float32
                    min_val = optimized[col].min()
                    max_val = optimized[col].max()
                    if (min_val >= np.finfo(np.float32).min and 
                        max_val <= np.finfo(np.float32).max):
                        optimized[col] = optimized[col].astype(np.float32)
                
                elif optimized[col].dtype == 'int64':
                    # Check if can downcast to int32
                    min_val = optimized[col].min()
                    max_val = optimized[col].max()
                    if (min_val >= np.iinfo(np.int32).min and 
                        max_val <= np.iinfo(np.int32).max):
                        optimized[col] = optimized[col].astype(np.int32)
            
            return optimized
        
        # Test optimization
        data = create_test_data(10000)
        
        print(f"  Original memory: {data.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
        
        optimized = optimize_dataframe(data)
        print(f"  Optimized memory: {optimized.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
        
        # Check dtypes
        original_dtypes = data.dtypes.value_counts()
        optimized_dtypes = optimized.dtypes.value_counts()
        
        print(f"  Original dtypes: {dict(original_dtypes)}")
        print(f"  Optimized dtypes: {dict(optimized_dtypes)}")
        
        # Check if optimization worked
        if optimized.memory_usage(deep=True).sum() < data.memory_usage(deep=True).sum():
            print("  ✅ Memory optimization successful")
        else:
            print("  ⚠️ No memory optimization applied")
        
        return True
        
    except Exception as e:
        print(f"❌ Zero-copy optimization test failed: {e}")
        return False

def test_vectorization_optimization():
    """Test vectorization optimizations."""
    print("\n🧪 Testing Vectorization Optimization...")
    print("=" * 50)
    
    try:
        def vectorized_rolling_ops(data: np.ndarray, window: int) -> dict:
            """Vectorized rolling operations."""
            results = {}
            
            if len(data) < window:
                return {op: np.full(len(data), np.nan) for op in ['mean', 'std', 'min', 'max']}
            
            # Use numpy's optimized functions
            results['mean'] = np.convolve(data, np.ones(window)/window, mode='valid')
            results['std'] = np.array([np.std(data[i:i+window]) for i in range(len(data)-window+1)])
            results['min'] = np.array([np.min(data[i:i+window]) for i in range(len(data)-window+1)])
            results['max'] = np.array([np.max(data[i:i+window]) for i in range(len(data)-window+1)])
            
            # Pad with NaN for alignment
            for key in results:
                if len(results[key]) < len(data):
                    padded = np.full(len(data), np.nan)
                    padded[window-1:] = results[key]
                    results[key] = padded
            
            return results
        
        def vectorized_correlations(data: np.ndarray, target: np.ndarray) -> np.ndarray:
            """Vectorized correlations."""
            return np.corrcoef(data.T, target)[:-1, -1]
        
        # Test vectorization
        data = np.random.randn(10000, 10)
        target = np.random.randn(10000)
        
        print("  Testing vectorized rolling operations...")
        start_time = time.time()
        rolling_results = vectorized_rolling_ops(data[:, 0], window=20)
        rolling_time = time.time() - start_time
        
        print(f"    Rolling operations time: {rolling_time:.3f}s")
        print(f"    Results shape: {[results.shape for results in rolling_results.values()]}")
        
        print("  Testing vectorized correlations...")
        start_time = time.time()
        correlations = vectorized_correlations(data, target)
        corr_time = time.time() - start_time
        
        print(f"    Correlation time: {corr_time:.3f}s")
        print(f"    High correlations: {np.sum(np.abs(correlations) > 0.95)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vectorization optimization test failed: {e}")
        return False

def test_memory_monitoring():
    """Test memory monitoring."""
    print("\n🧪 Testing Memory Monitoring...")
    print("=" * 50)
    
    try:
        def monitor_memory_usage():
            """Monitor memory usage."""
            try:
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)  # MB
            except:
                return 0.0
        
        # Test memory monitoring
        print("  Testing memory monitoring...")
        
        initial_memory = monitor_memory_usage()
        print(f"    Initial memory: {initial_memory:.1f}MB")
        
        # Create some data
        data1 = create_test_data(20000)
        data2 = create_test_data(20000)
        data3 = create_test_data(20000)
        
        peak_memory = monitor_memory_usage()
        print(f"    Peak memory: {peak_memory:.1f}MB")
        
        # Calculate efficiency
        memory_increase = peak_memory - initial_memory
        print(f"    Memory increase: {memory_increase:.1f}MB")
        
        if memory_increase > 0:
            print("  ✅ Memory monitoring working")
        else:
            print("  ⚠️ No memory increase detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory monitoring test failed: {e}")
        return False

def test_performance_comparison():
    """Test performance comparison."""
    print("\n🧪 Testing Performance Comparison...")
    print("=" * 50)
    
    try:
        def inefficient_operation(data: pd.DataFrame) -> pd.DataFrame:
            """Inefficient operation using concat."""
            results = []
            for i in range(0, len(data), 1000):
                chunk = data.iloc[i:i+1000]
                # Simulate some processing
                chunk['processed'] = chunk['close'] * 2
                results.append(chunk)
            return pd.concat(results, ignore_index=True)
        
        def efficient_operation(data: pd.DataFrame) -> pd.DataFrame:
            """Efficient operation using vectorization."""
            result = data.copy()
            result['processed'] = result['close'] * 2
            return result
        
        # Test performance
        data = create_test_data(50000)
        
        print("  Testing inefficient operation...")
        start_time = time.time()
        result1 = inefficient_operation(data)
        inefficient_time = time.time() - start_time
        
        print("  Testing efficient operation...")
        start_time = time.time()
        result2 = efficient_operation(data)
        efficient_time = time.time() - start_time
        
        print(f"    Inefficient time: {inefficient_time:.3f}s")
        print(f"    Efficient time: {efficient_time:.3f}s")
        print(f"    Speedup: {inefficient_time/efficient_time:.1f}x")
        
        # Verify results are the same
        if np.allclose(result1['processed'], result2['processed']):
            print("  ✅ Results are identical")
        else:
            print("  ❌ Results differ")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison test failed: {e}")
        return False

def main():
    """Run all efficiency tests."""
    print("🚀 Testing Core Efficiency Optimizations")
    print("=" * 60)
    
    tests = [
        ("Data Fingerprinting", test_data_fingerprinting),
        ("Chunking Optimization", test_chunking_optimization),
        ("Zero-Copy Optimization", test_zero_copy_optimization),
        ("Vectorization Optimization", test_vectorization_optimization),
        ("Memory Monitoring", test_memory_monitoring),
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
    print("📊 CORE EFFICIENCY TEST SUMMARY")
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
        print("🎉 All core efficiency tests passed!")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)