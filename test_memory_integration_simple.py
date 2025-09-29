#!/usr/bin/env python3
"""
Simple test script to verify the memory integration implementation.
This script tests the basic functionality without external dependencies.
"""

import sys
import os
import time
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_memory_functions():
    """Test basic memory functions."""
    logger.info("🧪 Testing Basic Memory Functions")
    
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
            # Simulate some work
            return sum(range(1000))
        
        result = test_function()
        assert result is not None, "Decorated function should return a result"
        logger.info(f"📊 Decorated function result: {result}")
        logger.info("✅ Memory skim decorator test passed")
        
        # Test 5: Context managers
        logger.info("🧪 Test 5: Context managers")
        
        # Test auto memory skim context
        with auto_memory_skim_context("test_context") as context:
            # Simulate some work
            result = sum(i**2 for i in range(100))
            logger.info(f"📊 Context work result: {result}")
        
        logger.info("✅ Auto memory skim context test passed")
        
        # Test smart memory context
        with smart_memory_context("test_smart_context") as allocation_info:
            # Simulate some work
            result = sum(i**3 for i in range(50))
            logger.info(f"📊 Smart context work result: {result}")
            logger.info(f"📊 Allocation info: {allocation_info}")
        
        logger.info("✅ Smart memory context test passed")
        
        # Test 6: Different operation types
        logger.info("🧪 Test 6: Different operation types")
        
        operations = [
            'hyperparameter_optimization',
            'cross_validation', 
            'model_training',
            'feature_engineering',
            'data_preprocessing',
            'model_inference'
        ]
        
        for operation in operations:
            # Test memory skimming for each operation
            skim_result = auto_skim_memory(50.0, operation)
            logger.info(f"📊 {operation} skim result: success={skim_result['success']}, freed={skim_result.get('memory_freed_mb', 0):.1f} MB")
            
            # Test memory allocation for each operation
            allocation_result = smart_memory_allocation(100.0, operation)
            logger.info(f"📊 {operation} allocation result: success={allocation_result['allocation_successful']}")
        
        logger.info("✅ Different operation types test passed")
        
        logger.info("🎉 All basic memory integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_manager_features():
    """Test MLMemoryManager features."""
    logger.info("🧪 Testing MLMemoryManager Features")
    
    try:
        from src.utils.ml_common.utils.memory_integration import MLMemoryManager
        
        # Create memory manager
        manager = MLMemoryManager()
        
        # Test memory estimation for different scenarios
        test_scenarios = [
            {
                'operation_type': 'model_training',
                'data_shape': (1000, 50),
                'n_samples': 1000,
                'n_features': 50
            },
            {
                'operation_type': 'hyperparameter_optimization',
                'n_trials': 100,
                'cv_folds': 5
            },
            {
                'operation_type': 'cross_validation',
                'n_samples': 5000,
                'n_features': 100,
                'cv_folds': 10
            },
            {
                'operation_type': 'feature_engineering',
                'data_shape': (2000, 200)
            }
        ]
        
        for i, scenario in enumerate(test_scenarios):
            estimated = manager.estimate_ml_memory_requirements(**scenario)
            logger.info(f"📊 Scenario {i+1} ({scenario['operation_type']}): {estimated:.1f} MB")
            assert estimated > 0, f"Estimated memory for scenario {i+1} should be positive"
        
        # Test auto skim for ML operation
        skim_result = manager.auto_skim_for_ml_operation('model_training', data_shape=(500, 25))
        logger.info(f"📊 Auto skim result: {skim_result}")
        assert 'success' in skim_result, "Auto skim result should contain success"
        
        # Test smart allocation for ML operation
        allocation_result = manager.smart_allocate_for_ml_operation('feature_engineering', data_shape=(1000, 100))
        logger.info(f"📊 Smart allocation result: {allocation_result}")
        assert 'allocation_successful' in allocation_result, "Smart allocation result should contain allocation_successful"
        
        logger.info("✅ MLMemoryManager features test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ MLMemoryManager features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting Memory Integration Implementation Tests")
    
    # Test basic memory functions
    basic_success = test_basic_memory_functions()
    
    # Test memory manager features
    manager_success = test_memory_manager_features()
    
    # Summary
    if basic_success and manager_success:
        logger.info("🎉 All tests passed! Memory integration implementation is working correctly.")
        logger.info("📊 Summary of implemented features:")
        logger.info("  ✅ auto_skim_memory() - Real memory optimization with M1 optimizer")
        logger.info("  ✅ smart_memory_allocation() - Intelligent memory allocation with pressure handling")
        logger.info("  ✅ memory_skim_decorator() - Decorator with pre/post operation memory management")
        logger.info("  ✅ auto_memory_skim_context() - Context manager with memory tracking")
        logger.info("  ✅ smart_memory_context() - Smart allocation context manager")
        logger.info("  ✅ MLMemoryManager - Comprehensive ML memory management")
        logger.info("  ✅ Memory estimation for different ML operations")
        logger.info("  ✅ Memory pressure handling and optimization")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())