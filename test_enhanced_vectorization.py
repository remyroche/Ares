#!/usr/bin/env python3
"""
Test script for enhanced VectorBTRollingOptimizer and UnifiedVectorizationManager
with comprehensive logging and fast failing capabilities.
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_vectorbt_rolling_optimizer():
    """Test the enhanced VectorBTRollingOptimizer."""
    print("🧪 Testing Enhanced VectorBTRollingOptimizer")
    print("=" * 50)
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
        
        # Test 1: Basic initialization with logging
        print("\n1. Testing initialization with enhanced logging...")
        optimizer = VectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000,
            fast_fail=True,
            enable_logging=True
        )
        print("✅ Initialization successful")
        
        # Test 2: Create test data
        print("\n2. Creating test data...")
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
        test_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        print(f"✅ Test data created: {test_data.shape}")
        
        # Test 3: Rolling operations with logging
        print("\n3. Testing rolling operations with enhanced logging...")
        
        # Test rolling mean
        print("\n   Testing rolling mean...")
        rolling_mean = optimizer.rolling_mean(test_data['close'], window=20)
        print(f"✅ Rolling mean completed: {rolling_mean.shape}")
        
        # Test rolling std
        print("\n   Testing rolling std...")
        rolling_std = optimizer.rolling_std(test_data['close'], window=20)
        print(f"✅ Rolling std completed: {rolling_std.shape}")
        
        # Test rolling min
        print("\n   Testing rolling min...")
        rolling_min = optimizer.rolling_min(test_data['close'], window=20)
        print(f"✅ Rolling min completed: {rolling_min.shape}")
        
        # Test 4: Error handling with fast failing
        print("\n4. Testing error handling and fast failing...")
        
        try:
            # Test with invalid window size
            print("\n   Testing invalid window size...")
            optimizer.rolling_mean(test_data['close'], window=2000)  # Larger than data
        except Exception as e:
            print(f"✅ Fast fail caught invalid window: {type(e).__name__}")
        
        try:
            # Test with empty data
            print("\n   Testing empty data...")
            empty_data = pd.Series([], dtype=float)
            optimizer.rolling_mean(empty_data, window=10)
        except Exception as e:
            print(f"✅ Fast fail caught empty data: {type(e).__name__}")
        
        # Test 5: Performance statistics
        print("\n5. Testing performance statistics...")
        stats = optimizer.get_performance_stats()
        print(f"✅ Performance stats: {stats}")
        
        print("\n✅ VectorBTRollingOptimizer tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ VectorBTRollingOptimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unified_vectorization_manager():
    """Test the enhanced UnifiedVectorizationManager."""
    print("\n🧪 Testing Enhanced UnifiedVectorizationManager")
    print("=" * 50)
    
    try:
        from src.feature_generation.utils.unified_vectorization_manager import (
            UnifiedVectorizationManager, VectorizationConfig
        )
        
        # Test 1: Basic initialization with logging
        print("\n1. Testing initialization with enhanced logging...")
        config = VectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000,
            batch_size=10000,
            enable_monitoring=True
        )
        
        manager = UnifiedVectorizationManager(
            config=config,
            fast_fail=True,
            enable_logging=True
        )
        print("✅ Initialization successful")
        
        # Test 2: Create test data
        print("\n2. Creating test data...")
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
        test_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'volume': np.random.randint(1000, 10000, 1000),
            'high': 100 + np.cumsum(np.random.randn(1000) * 0.01) + np.abs(np.random.randn(1000) * 0.5),
            'low': 100 + np.cumsum(np.random.randn(1000) * 0.01) - np.abs(np.random.randn(1000) * 0.5)
        }, index=dates)
        print(f"✅ Test data created: {test_data.shape}")
        
        # Test 3: Rolling operations
        print("\n3. Testing rolling operations...")
        rolling_mean = manager.rolling_operation(test_data['close'], 'mean', window=20)
        print(f"✅ Rolling mean completed: {rolling_mean.shape}")
        
        # Test 4: Scaling operations
        print("\n4. Testing scaling operations...")
        scaled_close = manager.scale_data(test_data['close'], method='zscore')
        print(f"✅ Scaling completed: {scaled_close.shape}")
        
        # Test 5: Batch processing
        print("\n5. Testing batch feature processing...")
        feature_configs = [
            {'name': 'sma_20', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 20, 'column': 'close'}},
            {'name': 'sma_50', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 50, 'column': 'close'}},
            {'name': 'std_20', 'type': 'rolling', 'params': {'operation': 'std', 'window': 20, 'column': 'close'}},
            {'name': 'close_scaled', 'type': 'scaling', 'params': {'method': 'zscore', 'column': 'close'}},
            {'name': 'volume_scaled', 'type': 'scaling', 'params': {'method': 'minmax', 'column': 'volume'}}
        ]
        
        features = manager.batch_process_features(test_data, feature_configs)
        print(f"✅ Batch processing completed: {features.shape}")
        print(f"   Generated features: {list(features.columns)}")
        
        # Test 6: Error handling
        print("\n6. Testing error handling...")
        
        try:
            # Test with invalid operation
            print("\n   Testing invalid operation...")
            manager.rolling_operation(test_data['close'], 'invalid_op', window=20)
        except Exception as e:
            print(f"✅ Fast fail caught invalid operation: {type(e).__name__}")
        
        try:
            # Test with invalid scaling method
            print("\n   Testing invalid scaling method...")
            manager.scale_data(test_data['close'], method='invalid_method')
        except Exception as e:
            print(f"✅ Fast fail caught invalid scaling method: {type(e).__name__}")
        
        # Test 7: Performance statistics
        print("\n7. Testing performance statistics...")
        stats = manager.get_performance_stats()
        print(f"✅ Performance stats: {stats}")
        
        print("\n✅ UnifiedVectorizationManager tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ UnifiedVectorizationManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test backward compatibility of the enhanced classes."""
    print("\n🧪 Testing Backward Compatibility")
    print("=" * 50)
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
        from src.feature_generation.utils.unified_vectorization_manager import (
            UnifiedVectorizationManager, VectorizationConfig
        )
        
        # Test 1: Old-style initialization (should still work)
        print("\n1. Testing old-style initialization...")
        
        # VectorBTRollingOptimizer with old parameters
        old_optimizer = VectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000
        )
        print("✅ Old-style VectorBTRollingOptimizer initialization works")
        
        # UnifiedVectorizationManager with old parameters
        old_manager = UnifiedVectorizationManager()
        print("✅ Old-style UnifiedVectorizationManager initialization works")
        
        # Test 2: Old-style method calls (should still work)
        print("\n2. Testing old-style method calls...")
        
        # Create test data
        np.random.seed(42)
        test_data = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.01))
        
        # Test old-style rolling operations
        result1 = old_optimizer.rolling_mean(test_data, window=10)
        result2 = old_manager.rolling_operation(test_data, 'mean', window=10)
        
        print(f"✅ Old-style method calls work: {result1.shape}, {result2.shape}")
        
        # Test 3: New features are optional
        print("\n3. Testing that new features are optional...")
        
        # Test with fast_fail=False (old behavior)
        old_behavior_optimizer = VectorBTRollingOptimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000,
            fast_fail=False  # This should allow fallbacks
        )
        
        # This should not raise an exception even with invalid data
        try:
            empty_data = pd.Series([], dtype=float)
            result = old_behavior_optimizer.rolling_mean(empty_data, window=10)
            print("✅ Old behavior (fast_fail=False) allows fallbacks")
        except Exception as e:
            print(f"⚠️ Old behavior test: {e}")
        
        print("\n✅ Backward compatibility tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Enhanced Vectorization Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(test_vectorbt_rolling_optimizer())
    test_results.append(test_unified_vectorization_manager())
    test_results.append(test_backward_compatibility())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Enhanced vectorization is working correctly.")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)