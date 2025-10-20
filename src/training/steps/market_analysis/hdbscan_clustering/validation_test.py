#!/usr/bin/env python3
"""
Validation Test for HDBSCAN Clustering Performance Optimizations

This script validates that all optimizations work correctly and maintain functionality.
"""

import asyncio
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_timer, configure_tprint,
    TPrintConfig, LogLevel
)

# Configure tprint for validation
config = TPrintConfig(
    min_log_level=LogLevel.DEBUG,
    enable_memory_monitoring=True,
    enable_performance_tracking=True,
    use_colors=True
)
configure_tprint(config)

async def test_hdbscan_regime_discovery_step():
    """Test the HDBSCAN regime discovery step with optimizations."""
    tprint_info("🧪 Testing HDBSCAN Regime Discovery Step")
    
    try:
        from src.training.steps.market_analysis.hdbscan_clustering.hdbscan_regime_discovery_step import HDBSCANRegimeDiscoveryStep
        
        # Initialize step
        step = HDBSCANRegimeDiscoveryStep()
        
        # Test configuration
        test_config = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m',
            'data_dir': 'historical_data',
            'execution_mode': 'light',  # Use light mode for faster testing
            'live_mode': False
        }
        
        tprint_info("Running regime discovery test...")
        
        # Run the step
        result = await step.run(test_config)
        
        # Validate results
        if result['success']:
            tprint_success("✅ HDBSCAN regime discovery step test PASSED")
            tprint_debug(f"Processing time: {result.get('processing_time', 0):.2f}s")
            tprint_debug(f"Performance time: {result.get('performance_time', 0):.2f}s")
            tprint_debug(f"Memory usage: {result.get('memory_usage', {})}")
            return True
        else:
            tprint_error(f"❌ HDBSCAN regime discovery step test FAILED: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        tprint_error(f"❌ HDBSCAN regime discovery step test FAILED with exception: {e}")
        return False

async def test_optimized_components():
    """Test individual optimized components."""
    tprint_info("🧪 Testing Optimized Components")
    
    try:
        # Test optimized HDBSCAN clusterer
        from src.training.steps.market_analysis.hdbscan_clustering.optimization.optimized_hdbscan_clusterer import (
            OptimizedHDBSCANClusterer, HDBSCANConfig
        )
        
        config = HDBSCANConfig(
            min_cluster_size=10,
            enable_parameter_optimization=False,  # Disable for faster testing
            memory_efficient=True
        )
        
        clusterer = OptimizedHDBSCANClusterer(config)
        
        # Test with synthetic data
        np.random.seed(42)
        test_data = np.random.randn(100, 5)
        
        with tprint_timer("HDBSCAN clustering test"):
            labels = clusterer.fit_predict(test_data)
        
        tprint_success("✅ Optimized HDBSCAN clusterer test PASSED")
        tprint_debug(f"Clustered {len(test_data)} samples into {len(np.unique(labels))} clusters")
        
        # Test enhanced memory optimizer
        from src.training.steps.market_analysis.hdbscan_clustering.optimization.enhanced_memory_optimizer import (
            EnhancedMemoryOptimizer, MemoryOptimizationConfig
        )
        
        memory_config = MemoryOptimizationConfig(
            max_memory_gb=2.0,
            enable_memory_optimization=True
        )
        
        memory_optimizer = EnhancedMemoryOptimizer(memory_config)
        
        # Test memory optimization
        test_df = pd.DataFrame(np.random.randn(1000, 10))
        optimized_df = memory_optimizer.optimize_dataframe_memory(test_df)
        
        tprint_success("✅ Enhanced memory optimizer test PASSED")
        tprint_debug(f"Memory optimization applied to DataFrame: {test_df.shape}")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Optimized components test FAILED with exception: {e}")
        return False

def test_tprint_logging():
    """Test tprint logging functionality."""
    tprint_info("🧪 Testing tprint Logging Functionality")
    
    try:
        # Test different log levels
        tprint_debug("This is a debug message")
        tprint_info("This is an info message")
        tprint_success("This is a success message")
        tprint_warning("This is a warning message")
        tprint_error("This is an error message")
        
        # Test performance timing
        with tprint_timer("Test operation"):
            time.sleep(0.1)
        
        # Test memory monitoring
        from src.utils.common_operations import get_memory_usage
        memory = get_memory_usage()
        tprint_debug(f"Current memory usage: {memory:.2f}MB")
        
        tprint_success("✅ tprint logging functionality test PASSED")
        return True
        
    except Exception as e:
        tprint_error(f"❌ tprint logging test FAILED with exception: {e}")
        return False

def test_memory_optimizations():
    """Test memory optimization functions."""
    tprint_info("🧪 Testing Memory Optimizations")
    
    try:
        from src.utils.common_operations import optimize_dataframe_memory, get_memory_usage
        
        # Create test data
        np.random.seed(42)
        test_data = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),
            'float_col': np.random.randn(1000),
            'string_col': [f'string_{i}' for i in range(1000)],
            'bool_col': np.random.choice([True, False], 1000)
        })
        
        initial_memory = test_data.memory_usage(deep=True).sum() / 1024**2
        tprint_debug(f"Initial DataFrame memory: {initial_memory:.2f}MB")
        
        # Optimize memory
        optimized_data = optimize_dataframe_memory(test_data)
        final_memory = optimized_data.memory_usage(deep=True).sum() / 1024**2
        
        memory_saved = initial_memory - final_memory
        tprint_debug(f"Optimized DataFrame memory: {final_memory:.2f}MB (saved {memory_saved:.2f}MB)")
        
        # Test system memory monitoring
        system_memory = get_memory_usage()
        tprint_debug(f"System memory usage: {system_memory:.2f}MB")
        
        tprint_success("✅ Memory optimizations test PASSED")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Memory optimizations test FAILED with exception: {e}")
        return False

async def run_validation_tests():
    """Run all validation tests."""
    tprint_info("🚀 Starting HDBSCAN Clustering Performance Optimization Validation")
    
    tests = [
        ("tprint logging", test_tprint_logging),
        ("memory optimizations", test_memory_optimizations),
        ("optimized components", test_optimized_components),
        ("HDBSCAN regime discovery step", test_hdbscan_regime_discovery_step),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        tprint_info(f"Running {test_name} test...")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results.append((test_name, result))
            
            if result:
                tprint_success(f"✅ {test_name} test PASSED")
            else:
                tprint_error(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            tprint_error(f"❌ {test_name} test FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    tprint_info("📊 Validation Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        tprint_info(f"  {test_name}: {status}")
    
    tprint_info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        tprint_success("🎉 All validation tests PASSED! Optimizations are working correctly.")
        return True
    else:
        tprint_error(f"⚠️ {total - passed} validation tests FAILED. Please review the issues above.")
        return False

if __name__ == "__main__":
    # Run validation tests
    success = asyncio.run(run_validation_tests())
    sys.exit(0 if success else 1)