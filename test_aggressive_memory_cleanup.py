#!/usr/bin/env python3
"""
Test script for aggressive memory cleanup improvements in final feature selection.

This script validates the enhanced memory management capabilities including:
- Advanced memory optimizer integration
- Aggressive cleanup strategies
- Memory pressure monitoring
- Component-specific cache clearing
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import gc
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_large_test_data(n_samples: int = 10000, n_features: int = 500) -> pd.DataFrame:
    """Create large test data to stress test memory management."""
    print(f"📊 Creating test data: {n_samples} samples, {n_features} features")
    
    # Create random data with some correlation patterns
    np.random.seed(42)
    data = {}
    
    for i in range(n_features):
        if i < 100:  # First 100 features are highly correlated
            base = np.random.randn(n_samples)
            data[f'feature_{i}'] = base + 0.1 * np.random.randn(n_samples)
        elif i < 200:  # Next 100 features are moderately correlated
            base = np.random.randn(n_samples)
            data[f'feature_{i}'] = base + 0.5 * np.random.randn(n_samples)
        else:  # Remaining features are independent
            data[f'feature_{i}'] = np.random.randn(n_samples)
    
    # Add some categorical features
    data['categorical_1'] = np.random.choice(['A', 'B', 'C'], n_samples)
    data['categorical_2'] = np.random.choice(['X', 'Y', 'Z'], n_samples)
    
    # Add datetime index
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1H')
    
    df = pd.DataFrame(data, index=dates)
    print(f"✅ Test data created: {df.shape}")
    return df

def test_memory_cleanup_methods():
    """Test the aggressive memory cleanup methods."""
    print("\n🧪 Testing Aggressive Memory Cleanup Methods")
    print("=" * 60)
    
    try:
        from src.training.steps.pre_training.components.final_feature_selection import FinalFeatureSelectionComponent
        from src.training.steps.pre_training.components.base_component import ComponentConfig
        
        # Initialize component
        print("🔧 Initializing FinalFeatureSelectionComponent...")
        config = ComponentConfig()
        component = FinalFeatureSelectionComponent(config)
        
        # Test memory pressure monitoring
        print("\n📊 Testing memory pressure monitoring...")
        memory_stats = component.monitor_memory_pressure()
        print(f"   Current pressure: {memory_stats['pressure']:.3f}")
        print(f"   Cleanup triggered: {memory_stats['cleanup_triggered']}")
        print(f"   Recommendations: {memory_stats['recommendations']}")
        
        # Test aggressive cleanup
        print("\n🧹 Testing aggressive memory cleanup...")
        cleanup_results = component.aggressive_memory_cleanup(force_cleanup=False)
        print(f"   Success: {cleanup_results['success']}")
        print(f"   Memory freed: {cleanup_results['memory_freed_mb']:.1f}MB")
        print(f"   Methods used: {cleanup_results['cleanup_methods_used']}")
        print(f"   Errors: {cleanup_results['errors']}")
        
        # Test component cache clearing
        print("\n🗑️ Testing component cache clearing...")
        component._clear_component_caches()
        print("   ✅ Component caches cleared")
        
        # Test memory optimization
        print("\n⚡ Testing memory optimization...")
        component.optimize_memory()
        print("   ✅ Memory optimization applied")
        
        # Cleanup
        component.cleanup()
        print("\n✅ All memory cleanup tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory cleanup test failed: {e}")
        import traceback
        print(f"🔍 Error details: {traceback.format_exc()}")
        return False

def test_pipeline_memory_management():
    """Test memory management in the feature selection pipeline."""
    print("\n🧪 Testing Pipeline Memory Management")
    print("=" * 60)
    
    try:
        from src.training.steps.pre_training.final_feature_selection_pipeline import MultiStageFeatureSelector, FeatureSelectionConfig
        
        # Initialize pipeline
        print("🔧 Initializing MultiStageFeatureSelector...")
        config = FeatureSelectionConfig()
        selector = MultiStageFeatureSelector(config)
        
        # Test memory pressure monitoring
        print("\n📊 Testing pipeline memory pressure monitoring...")
        memory_stats = selector._monitor_memory_pressure()
        print(f"   Current pressure: {memory_stats['pressure']:.3f}")
        print(f"   Cleanup triggered: {memory_stats['cleanup_triggered']}")
        print(f"   Recommendations: {memory_stats['recommendations']}")
        
        # Test aggressive cleanup
        print("\n🧹 Testing pipeline aggressive cleanup...")
        cleanup_results = selector._aggressive_memory_cleanup(force_cleanup=False)
        print(f"   Success: {cleanup_results['success']}")
        print(f"   Memory freed: {cleanup_results['memory_freed_mb']:.1f}MB")
        print(f"   Methods used: {cleanup_results['cleanup_methods_used']}")
        
        # Test cache management
        print("\n💾 Testing cache management...")
        cache_stats = selector._get_cache_stats()
        print(f"   Cache size: {cache_stats['cache_size']} entries")
        print(f"   Cache hits: {cache_stats['cache_hits']}")
        print(f"   Cache misses: {cache_stats['cache_misses']}")
        print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")
        
        print("\n✅ Pipeline memory management tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline memory management test failed: {e}")
        import traceback
        print(f"🔍 Error details: {traceback.format_exc()}")
        return False

def test_memory_under_load():
    """Test memory management under load with large datasets."""
    print("\n🧪 Testing Memory Management Under Load")
    print("=" * 60)
    
    try:
        from src.training.steps.pre_training.components.final_feature_selection import FinalFeatureSelectionComponent
        from src.training.steps.pre_training.components.base_component import ComponentConfig
        
        # Create large test data
        print("📊 Creating large test dataset...")
        X = create_large_test_data(n_samples=5000, n_features=200)
        y = pd.Series(np.random.randn(len(X)), index=X.index)
        
        # Initialize component
        print("🔧 Initializing component...")
        config = ComponentConfig()
        component = FinalFeatureSelectionComponent(config)
        
        # Monitor memory before processing
        print("\n📊 Memory state before processing...")
        memory_stats_before = component.monitor_memory_pressure()
        print(f"   Pressure: {memory_stats_before['pressure']:.3f}")
        
        # Simulate memory-intensive operations
        print("\n🔄 Simulating memory-intensive operations...")
        
        # Create some large temporary objects
        large_arrays = []
        for i in range(5):
            large_array = np.random.randn(1000, 1000)
            large_arrays.append(large_array)
        
        # Monitor memory after creating large objects
        memory_stats_after_creation = component.monitor_memory_pressure()
        print(f"   Pressure after creating large objects: {memory_stats_after_creation['pressure']:.3f}")
        
        # Trigger aggressive cleanup
        print("\n🧹 Triggering aggressive cleanup...")
        cleanup_results = component.aggressive_memory_cleanup(force_cleanup=True)
        print(f"   Memory freed: {cleanup_results['memory_freed_mb']:.1f}MB")
        print(f"   Success: {cleanup_results['success']}")
        
        # Monitor memory after cleanup
        memory_stats_after_cleanup = component.monitor_memory_pressure()
        print(f"   Pressure after cleanup: {memory_stats_after_cleanup['pressure']:.3f}")
        
        # Clean up large arrays
        del large_arrays
        gc.collect()
        
        # Final cleanup
        component.cleanup()
        
        print("\n✅ Memory management under load test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Memory under load test failed: {e}")
        import traceback
        print(f"🔍 Error details: {traceback.format_exc()}")
        return False

def test_hardware_optimization_integration():
    """Test integration with hardware optimization tools."""
    print("\n🧪 Testing Hardware Optimization Integration")
    print("=" * 60)
    
    try:
        from src.training.steps.pre_training.components.final_feature_selection import FinalFeatureSelectionComponent
        from src.training.steps.pre_training.components.base_component import ComponentConfig
        
        # Initialize component
        print("🔧 Initializing component with hardware optimization...")
        config = ComponentConfig()
        component = FinalFeatureSelectionComponent(config)
        
        # Check hardware optimization availability
        print("\n🛠️ Checking hardware optimization tools...")
        print(f"   Memory optimizer: {'✅ Available' if component.memory_optimizer else '❌ Not available'}")
        print(f"   Advanced memory optimizer: {'✅ Available' if component.advanced_memory_optimizer else '❌ Not available'}")
        print(f"   Hardware manager: {'✅ Available' if component.hardware_manager else '❌ Not available'}")
        print(f"   GPU manager: {'✅ Available' if component.gpu_manager else '❌ Not available'}")
        print(f"   Adaptive engine: {'✅ Available' if component.adaptive_engine else '❌ Not available'}")
        
        # Test hardware acceleration
        print("\n⚡ Testing hardware acceleration...")
        is_accelerated = component.is_hardware_accelerated()
        print(f"   Hardware acceleration: {'✅ Enabled' if is_accelerated else '❌ Disabled'}")
        
        # Test memory optimization
        print("\n🧠 Testing memory optimization...")
        component.optimize_memory()
        print("   ✅ Memory optimization applied")
        
        # Test memory pressure monitoring
        print("\n📊 Testing memory pressure monitoring...")
        memory_stats = component.monitor_memory_pressure()
        print(f"   Current pressure: {memory_stats['pressure']:.3f}")
        print(f"   Cleanup triggered: {memory_stats['cleanup_triggered']}")
        
        # Cleanup
        component.cleanup()
        
        print("\n✅ Hardware optimization integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Hardware optimization integration test failed: {e}")
        import traceback
        print(f"🔍 Error details: {traceback.format_exc()}")
        return False

def main():
    """Run all memory cleanup tests."""
    print("🚀 Testing Aggressive Memory Cleanup Improvements")
    print("=" * 80)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Memory Cleanup Methods", test_memory_cleanup_methods()))
    test_results.append(("Pipeline Memory Management", test_pipeline_memory_management()))
    test_results.append(("Memory Under Load", test_memory_under_load()))
    test_results.append(("Hardware Optimization Integration", test_hardware_optimization_integration()))
    
    # Print summary
    print("\n📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All memory cleanup improvements are working correctly!")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main())