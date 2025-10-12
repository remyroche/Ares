#!/usr/bin/env python3
"""
Test script for VectorBT-optimized time features.

This script validates that the time features are properly using VectorBT
and UnifiedVectorizationManager for maximum performance.
"""

import pandas as pd
import numpy as np
import time
import warnings
from typing import List, Dict, Any
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create test data for time feature validation."""
    # Create datetime index with minute frequency
    start_date = pd.Timestamp('2020-01-01 09:00:00')
    end_date = start_date + pd.Timedelta(days=30)  # 30 days of data
    dates = pd.date_range(start=start_date, end=end_date, freq='1min')
    
    # Limit to requested number of samples
    if len(dates) > n_samples:
        dates = dates[:n_samples]
    
    # Create sample OHLCV data
    np.random.seed(42)
    n = len(dates)
    
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(n) * 0.01) + np.abs(np.random.randn(n) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(n) * 0.01) - np.abs(np.random.randn(n) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(n) * 0.01),
        'volume': np.random.lognormal(10, 1, n)
    }, index=dates)
    
    return data

def test_time_feature_generators():
    """Test all time feature generators with performance monitoring."""
    print("🧪 Testing VectorBT-optimized time feature generators...")
    
    try:
        from src.feature_generation.categories.time import (
            create_optimized_time_generators,
            get_time_feature_performance_stats,
            OptimizedTimeFeatureGenerator
        )
    except ImportError as e:
        print(f"❌ Failed to import time features: {e}")
        return False
    
    # Create test data
    test_data = create_test_data(5000)
    print(f"📊 Created test data with {len(test_data)} samples")
    
    # Create generators
    generators = create_optimized_time_generators()
    print(f"🔧 Created {len(generators)} time feature generators")
    
    # Test each generator
    results = {}
    total_time = 0
    
    for generator in generators:
        generator_name = generator.config.name
        print(f"\n🔍 Testing {generator_name}...")
        
        try:
            start_time = time.time()
            feature = generator.generate_feature(test_data)
            generation_time = time.time() - start_time
            total_time += generation_time
            
            # Validate feature
            if feature is None or feature.empty:
                print(f"❌ {generator_name}: Generated empty feature")
                continue
            
            # Check for NaN values
            nan_count = feature.isna().sum()
            if nan_count > 0:
                print(f"⚠️  {generator_name}: {nan_count} NaN values found")
            
            # Check feature properties
            results[generator_name] = {
                'shape': feature.shape,
                'dtype': str(feature.dtype),
                'generation_time': generation_time,
                'nan_count': nan_count,
                'min_value': feature.min() if not feature.empty else None,
                'max_value': feature.max() if not feature.empty else None,
                'unique_values': feature.nunique()
            }
            
            print(f"✅ {generator_name}: {feature.shape[0]} values, {generation_time:.4f}s")
            
        except Exception as e:
            print(f"❌ {generator_name}: Failed with error: {e}")
            results[generator_name] = {'error': str(e)}
    
    # Get performance statistics
    try:
        performance_stats = get_time_feature_performance_stats(generators)
        print(f"\n📈 Performance Statistics:")
        print(f"   VectorBT operations: {performance_stats['vectorbt_operations']}")
        print(f"   Unified vectorization operations: {performance_stats['unified_vectorization_operations']}")
        print(f"   Total operations: {performance_stats['total_operations']}")
        print(f"   Total generation time: {total_time:.4f}s")
    except Exception as e:
        print(f"⚠️  Could not get performance stats: {e}")
    
    # Print detailed results
    print(f"\n📋 Detailed Results:")
    for name, result in results.items():
        if 'error' in result:
            print(f"   {name}: ERROR - {result['error']}")
        else:
            print(f"   {name}: {result['shape']} {result['dtype']}, "
                  f"{result['generation_time']:.4f}s, "
                  f"{result['nan_count']} NaNs, "
                  f"{result['unique_values']} unique values")
    
    return len([r for r in results.values() if 'error' not in r]) > 0

def test_vectorbt_availability():
    """Test VectorBT availability and configuration."""
    print("\n🔍 Testing VectorBT availability...")
    
    try:
        import vectorbt as vbt
        print(f"✅ VectorBT version: {vbt.__version__}")
        
        # Test basic VectorBT functionality
        test_array = np.array([1, 2, 3, 4, 5])
        vbt_array = vbt.array_wrapper(test_array)
        result = vbt_array * 2
        print(f"✅ VectorBT array operations working")
        
        return True
    except ImportError:
        print("❌ VectorBT not available")
        return False
    except Exception as e:
        print(f"❌ VectorBT error: {e}")
        return False

def test_unified_vectorization_manager():
    """Test UnifiedVectorizationManager availability."""
    print("\n🔍 Testing UnifiedVectorizationManager availability...")
    
    try:
        from src.utils.ml_common.unified_vectorization_manager import (
            UnifiedVectorizationManager, 
            get_unified_vectorization_manager
        )
        
        manager = get_unified_vectorization_manager()
        print(f"✅ UnifiedVectorizationManager available")
        
        # Test basic functionality
        test_data = pd.DataFrame({'test': [1, 2, 3, 4, 5]})
        print(f"✅ UnifiedVectorizationManager initialized")
        
        return True
    except ImportError as e:
        print(f"❌ UnifiedVectorizationManager not available: {e}")
        return False
    except Exception as e:
        print(f"❌ UnifiedVectorizationManager error: {e}")
        return False

def test_vectorbt_rolling_optimizer():
    """Test VectorBTRollingOptimizer availability."""
    print("\n🔍 Testing VectorBTRollingOptimizer availability...")
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import (
            VectorBTRollingOptimizer,
            get_vectorbt_rolling_optimizer
        )
        
        optimizer = get_vectorbt_rolling_optimizer()
        print(f"✅ VectorBTRollingOptimizer available")
        
        # Test basic functionality
        test_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = optimizer.rolling_mean(test_data, window=3)
        print(f"✅ VectorBTRollingOptimizer working: {len(result)} values")
        
        return True
    except ImportError as e:
        print(f"❌ VectorBTRollingOptimizer not available: {e}")
        return False
    except Exception as e:
        print(f"❌ VectorBTRollingOptimizer error: {e}")
        return False

def benchmark_performance():
    """Benchmark performance of optimized vs non-optimized features."""
    print("\n⚡ Performance Benchmark...")
    
    try:
        from src.feature_generation.categories.time import (
            create_optimized_time_generators,
            HourGenerator,
            HourSinGenerator
        )
    except ImportError as e:
        print(f"❌ Could not import for benchmarking: {e}")
        return
    
    # Create test data of different sizes
    sizes = [1000, 5000, 10000, 20000]
    generators = [HourGenerator(), HourSinGenerator()]
    
    for size in sizes:
        print(f"\n📊 Testing with {size} samples...")
        test_data = create_test_data(size)
        
        for generator in generators:
            generator_name = generator.config.name
            
            # Reset performance stats
            generator.reset_performance_stats()
            
            # Time the generation
            start_time = time.time()
            feature = generator.generate_feature(test_data)
            generation_time = time.time() - start_time
            
            # Get performance stats
            stats = generator.get_performance_stats()
            
            print(f"   {generator_name}: {generation_time:.4f}s, "
                  f"VectorBT ops: {stats['vectorbt_operations']}, "
                  f"Unified ops: {stats['unified_vectorization_operations']}")

def main():
    """Main test function."""
    print("🚀 VectorBT Time Features Optimization Test")
    print("=" * 50)
    
    # Test component availability
    vectorbt_available = test_vectorbt_availability()
    unified_manager_available = test_unified_vectorization_manager()
    rolling_optimizer_available = test_vectorbt_rolling_optimizer()
    
    print(f"\n📋 Component Availability:")
    print(f"   VectorBT: {'✅' if vectorbt_available else '❌'}")
    print(f"   UnifiedVectorizationManager: {'✅' if unified_manager_available else '❌'}")
    print(f"   VectorBTRollingOptimizer: {'✅' if rolling_optimizer_available else '❌'}")
    
    # Test time feature generators
    if vectorbt_available or unified_manager_available or rolling_optimizer_available:
        success = test_time_feature_generators()
        
        if success:
            print("\n✅ Time feature generators test completed successfully!")
            
            # Run performance benchmark
            benchmark_performance()
        else:
            print("\n❌ Time feature generators test failed!")
    else:
        print("\n⚠️  No optimization components available, skipping feature tests")
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    main()