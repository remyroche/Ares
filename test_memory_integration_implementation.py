#!/usr/bin/env python3
"""
Test script to verify the memory integration implementation.
This script tests the actual memory management functionality.
"""

import sys
import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_memory_integration():
    """Test the memory integration implementation."""
    logger.info("🧪 Testing Memory Integration Implementation")
    
    try:
        # Import the memory integration module
        from src.utils.ml_common.utils.memory_integration import (
            auto_skim_memory,
            smart_memory_allocation,
            memory_skim_decorator,
            auto_memory_skim_context,
            smart_memory_context,
            MLMemoryManager,
            get_ml_memory_manager
        )
        
        logger.info("✅ Successfully imported memory integration modules")
        
        # Test 1: Basic memory skimming
        logger.info("🧪 Test 1: Basic memory skimming")
        skim_result = auto_skim_memory(100.0, "test_operation")
        logger.info(f"📊 Skim result: {skim_result}")
        
        assert skim_result['success'] == True, "Memory skimming should succeed"
        assert 'memory_freed_mb' in skim_result, "Result should contain memory_freed_mb"
        assert 'operation_type' in skim_result, "Result should contain operation_type"
        logger.info("✅ Basic memory skimming test passed")
        
        # Test 2: Smart memory allocation
        logger.info("🧪 Test 2: Smart memory allocation")
        allocation_result = smart_memory_allocation(200.0, "test_allocation")
        logger.info(f"📊 Allocation result: {allocation_result}")
        
        assert 'allocation_successful' in allocation_result, "Result should contain allocation_successful"
        assert 'operation_type' in allocation_result, "Result should contain operation_type"
        logger.info("✅ Smart memory allocation test passed")
        
        # Test 3: Memory manager
        logger.info("🧪 Test 3: Memory manager")
        manager = get_ml_memory_manager()
        assert manager is not None, "Memory manager should be created"
        
        # Test memory estimation
        estimated_memory = manager.estimate_ml_memory_requirements(
            'model_training',
            data_shape=(1000, 100),
            n_samples=1000,
            n_features=100
        )
        logger.info(f"📊 Estimated memory: {estimated_memory:.1f} MB")
        assert estimated_memory > 0, "Estimated memory should be positive"
        logger.info("✅ Memory manager test passed")
        
        # Test 4: Decorator functionality
        logger.info("🧪 Test 4: Memory skim decorator")
        
        @memory_skim_decorator("test_decorator")
        def test_function():
            # Simulate some memory usage
            data = np.random.rand(1000, 100)
            return data.sum()
        
        result = test_function()
        assert result is not None, "Decorated function should return a result"
        logger.info("✅ Memory skim decorator test passed")
        
        # Test 5: Context managers
        logger.info("🧪 Test 5: Context managers")
        
        # Test auto memory skim context
        with auto_memory_skim_context("test_context") as context:
            # Simulate some work
            data = np.random.rand(500, 50)
            result = data.mean()
            logger.info(f"📊 Context work result: {result}")
        
        logger.info("✅ Auto memory skim context test passed")
        
        # Test smart memory context
        with smart_memory_context("test_smart_context") as allocation_info:
            # Simulate some work
            data = np.random.rand(1000, 100)
            result = data.std()
            logger.info(f"📊 Smart context work result: {result}")
            logger.info(f"📊 Allocation info: {allocation_info}")
        
        logger.info("✅ Smart memory context test passed")
        
        # Test 6: Memory pressure handling
        logger.info("🧪 Test 6: Memory pressure handling")
        
        # Create a large array to increase memory pressure
        large_data = np.random.rand(10000, 1000)
        logger.info(f"📊 Created large array: {large_data.shape}")
        
        # Test memory skimming under pressure
        pressure_result = auto_skim_memory(500.0, "pressure_test")
        logger.info(f"📊 Pressure test result: {pressure_result}")
        
        # Clean up
        del large_data
        
        logger.info("✅ Memory pressure handling test passed")
        
        # Test 7: Integration with ML operations
        logger.info("🧪 Test 7: ML operations integration")
        
        # Test with ML memory manager
        ml_manager = MLMemoryManager()
        
        # Test memory estimation for different operations
        operations = [
            'hyperparameter_optimization',
            'cross_validation', 
            'model_training',
            'feature_engineering',
            'data_preprocessing',
            'model_inference'
        ]
        
        for operation in operations:
            estimated = ml_manager.estimate_ml_memory_requirements(operation)
            logger.info(f"📊 {operation}: {estimated:.1f} MB")
            assert estimated > 0, f"Estimated memory for {operation} should be positive"
        
        logger.info("✅ ML operations integration test passed")
        
        logger.info("🎉 All memory integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_optimization():
    """Test memory optimization functionality."""
    logger.info("🧪 Testing Memory Optimization")
    
    try:
        from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
        
        # Get memory optimizer
        optimizer = get_m1_memory_optimizer()
        
        # Test memory stats
        stats = optimizer.get_memory_stats()
        logger.info(f"📊 Memory stats: {stats}")
        
        # Test memory optimization
        optimization_result = optimizer.optimize_memory_usage(aggressive=False)
        logger.info(f"📊 Optimization result: {optimization_result}")
        
        # Test DataFrame optimization
        df = pd.DataFrame({
            'col1': np.random.rand(1000),
            'col2': np.random.randint(0, 100, 1000),
            'col3': ['category_' + str(i % 10) for i in range(1000)]
        })
        
        logger.info(f"📊 Original DataFrame memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        optimized_df = optimizer.optimize_dataframe_memory(df)
        logger.info(f"📊 Optimized DataFrame memory: {optimized_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        logger.info("✅ Memory optimization test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting Memory Integration Implementation Tests")
    
    # Test memory integration
    integration_success = test_memory_integration()
    
    # Test memory optimization
    optimization_success = test_memory_optimization()
    
    # Summary
    if integration_success and optimization_success:
        logger.info("🎉 All tests passed! Memory integration implementation is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())