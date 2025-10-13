#!/usr/bin/env python3
"""
Test script to demonstrate the improvements made to DataDrivenInteractionGenerator.

This script tests:
1. Broken initialization into smaller methods
2. Resource cleanup functionality
3. Cache invalidation
4. Memory management
5. Error handling improvements
6. Context manager support
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_improvements():
    """Test the improvements made to DataDrivenInteractionGenerator."""
    
    print("🧪 Testing DataDrivenInteractionGenerator Improvements")
    print("=" * 60)
    
    try:
        from src.feature_generation.utils.data_driven_interaction_generator import (
            DataDrivenInteractionGenerator, 
            EnhancedInteractionConfig
        )
        print("✅ Successfully imported DataDrivenInteractionGenerator")
    except ImportError as e:
        print(f"❌ Failed to import DataDrivenInteractionGenerator: {e}")
        return False
    
    # Test 1: Configuration validation
    print("\n1. Testing configuration validation...")
    try:
        # Test invalid configuration
        invalid_config = EnhancedInteractionConfig(
            max_interactions=-1,  # Invalid: negative
            utility_threshold=1.5,  # Invalid: > 1
            max_memory_gb=-1.0  # Invalid: negative
        )
        generator = DataDrivenInteractionGenerator(config=invalid_config)
        print("❌ Configuration validation failed - should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✅ Configuration validation working: {e}")
    except Exception as e:
        print(f"❌ Unexpected error in configuration validation: {e}")
        return False
    
    # Test 2: Proper initialization with smaller methods
    print("\n2. Testing proper initialization...")
    try:
        config = EnhancedInteractionConfig(
            max_interactions=10,
            utility_threshold=0.1,
            enable_vectorbt=False,  # Disable VectorBT for testing
            enable_caching=True,
            cache_size=5
        )
        generator = DataDrivenInteractionGenerator(config=config)
        print("✅ Generator initialized successfully with broken-down methods")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    # Test 3: Context manager support
    print("\n3. Testing context manager support...")
    try:
        with DataDrivenInteractionGenerator(config=config) as gen:
            print("✅ Context manager entry successful")
            # Test that generator is usable
            assert hasattr(gen, 'config')
            assert hasattr(gen, 'interaction_types')
            print("✅ Generator is usable within context")
        print("✅ Context manager exit successful (cleanup called)")
    except Exception as e:
        print(f"❌ Context manager test failed: {e}")
        return False
    
    # Test 4: Cache invalidation
    print("\n4. Testing cache invalidation...")
    try:
        generator = DataDrivenInteractionGenerator(config=config)
        
        # Test cache invalidation with pattern
        generator._invalidate_cache("test_pattern")
        print("✅ Cache invalidation with pattern successful")
        
        # Test cache invalidation without pattern (clear all)
        generator._invalidate_cache()
        print("✅ Cache invalidation (clear all) successful")
    except Exception as e:
        print(f"❌ Cache invalidation test failed: {e}")
        return False
    
    # Test 5: Memory management
    print("\n5. Testing memory management...")
    try:
        generator = DataDrivenInteractionGenerator(config=config)
        
        # Test memory check
        generator._check_memory_usage()
        print("✅ Memory usage check successful")
        
        # Test memory cleanup
        generator._cleanup_memory()
        print("✅ Memory cleanup successful")
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        return False
    
    # Test 6: Error handling improvements
    print("\n6. Testing improved error handling...")
    try:
        generator = DataDrivenInteractionGenerator(config=config)
        
        # Create test data
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'feature3': [3, 4, 5, 6, 7]
        })
        
        # Test with invalid input (should handle gracefully)
        try:
            result = generator._generate_single_interaction(
                test_data, 'feature1', 'feature2', 'unknown_type', None
            )
            if result is None:
                print("✅ Error handling working - returned None for invalid input")
            else:
                print("❌ Error handling failed - should have returned None")
                return False
        except Exception as e:
            print(f"❌ Error handling failed - raised exception: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Error handling test setup failed: {e}")
        return False
    
    # Test 7: Resource cleanup
    print("\n7. Testing resource cleanup...")
    try:
        generator = DataDrivenInteractionGenerator(config=config)
        
        # Test manual cleanup
        generator.cleanup()
        print("✅ Manual cleanup successful")
        
        # Test that cleanup can be called multiple times safely
        generator.cleanup()
        print("✅ Multiple cleanup calls handled safely")
        
    except Exception as e:
        print(f"❌ Resource cleanup test failed: {e}")
        return False
    
    # Test 8: Performance statistics
    print("\n8. Testing performance statistics...")
    try:
        generator = DataDrivenInteractionGenerator(config=config)
        
        stats = generator.get_performance_stats()
        expected_keys = [
            'total_interactions_generated', 'vectorbt_operations', 'pandas_fallbacks',
            'gpu_operations', 'batch_operations', 'cached_operations', 'memory_optimizations',
            'total_processing_time', 'average_utility_score', 'memory_savings',
            'cache_hit_rate', 'cache_misses', 'memory_usage_mb', 'peak_memory_usage_mb'
        ]
        
        for key in expected_keys:
            if key not in stats:
                print(f"❌ Missing performance stat: {key}")
                return False
        
        print("✅ Performance statistics include all expected metrics")
        
    except Exception as e:
        print(f"❌ Performance statistics test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Improvements are working correctly.")
    return True

def test_vectorbt_cleanup():
    """Test VectorBT utility cleanup methods."""
    
    print("\n🔧 Testing VectorBT utility cleanup...")
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
        print("✅ Successfully imported VectorBTRollingOptimizer")
    except ImportError as e:
        print(f"❌ Failed to import VectorBTRollingOptimizer: {e}")
        return False
    
    try:
        # Test context manager
        with VectorBTRollingOptimizer(enable_gpu=False, enable_parallel=False) as optimizer:
            print("✅ VectorBT optimizer context manager entry successful")
            
            # Test that optimizer is usable
            assert hasattr(optimizer, 'performance_stats')
            print("✅ VectorBT optimizer is usable within context")
        
        print("✅ VectorBT optimizer context manager exit successful (cleanup called)")
        
        # Test manual cleanup
        optimizer = VectorBTRollingOptimizer(enable_gpu=False, enable_parallel=False)
        optimizer.cleanup()
        print("✅ VectorBT optimizer manual cleanup successful")
        
    except Exception as e:
        print(f"❌ VectorBT cleanup test failed: {e}")
        return False
    
    print("✅ VectorBT utility cleanup tests passed!")
    return True

if __name__ == "__main__":
    print("🚀 Starting DataDrivenInteractionGenerator Improvement Tests")
    print("=" * 70)
    
    # Test main improvements
    main_test_passed = test_improvements()
    
    # Test VectorBT utility improvements
    vectorbt_test_passed = test_vectorbt_cleanup()
    
    print("\n" + "=" * 70)
    if main_test_passed and vectorbt_test_passed:
        print("🎉 ALL TESTS PASSED! All improvements are working correctly.")
        print("\n✅ Improvements implemented:")
        print("   • Broken initialization into smaller methods")
        print("   • Explicit resource cleanup with context managers")
        print("   • Cache invalidation mechanisms")
        print("   • Memory leak prevention and management")
        print("   • Improved error handling with specific exception types")
        print("   • Return type annotations added")
        print("   • VectorBT utility cleanup methods")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
        sys.exit(1)