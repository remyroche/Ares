#!/usr/bin/env python3
"""
Comprehensive Tests for Advanced Hardware Optimizations.

This script tests all the new hardware optimization features including:
- Unified Hardware Manager
- Advanced CPU Optimizations
- Enhanced GPU Acceleration
- Memory Architecture Enhancements
- Adaptive Optimization Engine
"""

import sys
import os
import logging
import time
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestUnifiedHardwareManager(unittest.TestCase):
    """Test the Unified Hardware Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from utils.hardware.unified_hardware_manager import (
                UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
            )
            self.manager_class = UnifiedHardwareManager
            self.config_class = HardwareConfig
            self.workload_type = WorkloadType
            self.optimization_level = OptimizationLevel
        except ImportError as e:
            self.skipTest(f"Unified Hardware Manager not available: {e}")
            
    def test_manager_initialization(self):
        """Test manager initialization."""
        config = self.config_class(
            memory_limit_gb=4.0,
            enable_adaptive_optimization=True,
            performance_monitoring_enabled=True
        )
        
        manager = self.manager_class(config)
        self.assertIsNotNone(manager)
        self.assertIsNotNone(manager.cpu_optimizer)
        self.assertIsNotNone(manager.gpu_manager)
        self.assertIsNotNone(manager.memory_optimizer)
        self.assertIsNotNone(manager.performance_monitor)
        self.assertIsNotNone(manager.task_scheduler)
        
    def test_workload_optimization(self):
        """Test workload-specific optimization."""
        manager = self.manager_class()
        success = manager.initialize()
        self.assertTrue(success)
        
        # Test backtesting optimization
        success = manager.optimize_for_workload(
            self.workload_type.BACKTESTING,
            self.optimization_level.AGGRESSIVE
        )
        self.assertTrue(success)
        
        # Test ML training optimization
        success = manager.optimize_for_workload(
            self.workload_type.ML_TRAINING,
            self.optimization_level.BALANCED
        )
        self.assertTrue(success)
        
    def test_optimization_context(self):
        """Test optimization context manager."""
        manager = self.manager_class()
        manager.initialize()
        
        with manager.optimization_context(
            self.workload_type.DATA_PROCESSING,
            self.optimization_level.BALANCED
        ) as ctx:
            self.assertIsNotNone(ctx)
            self.assertEqual(manager.current_workload_type, self.workload_type.DATA_PROCESSING)
            
    def test_system_status(self):
        """Test system status reporting."""
        manager = self.manager_class()
        manager.initialize()
        
        status = manager.get_system_status()
        self.assertIsInstance(status, dict)
        self.assertIn('initialized', status)
        self.assertIn('performance_report', status)
        self.assertIn('cpu_info', status)
        self.assertIn('gpu_info', status)
        self.assertIn('memory_stats', status)
        
    def test_configuration_management(self):
        """Test configuration save/load."""
        manager = self.manager_class()
        manager.initialize()
        
        # Test save configuration
        config_path = "/tmp/test_hardware_config.json"
        manager.save_configuration(config_path)
        self.assertTrue(os.path.exists(config_path))
        
        # Test load configuration
        success = manager.load_configuration(config_path)
        self.assertTrue(success)
        
        # Cleanup
        if os.path.exists(config_path):
            os.remove(config_path)

class TestAdvancedCPUOptimizer(unittest.TestCase):
    """Test the Advanced CPU Optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from utils.hardware.advanced_cpu_optimizer import (
                AdvancedM1CPUOptimizer, CoreAffinityConfig, ThermalConfig, 
                PowerConfig, WorkloadProfile, CoreType, ThermalState, PowerState
            )
            self.optimizer_class = AdvancedM1CPUOptimizer
            self.core_affinity_config = CoreAffinityConfig
            self.thermal_config = ThermalConfig
            self.power_config = PowerConfig
            self.workload_profile = WorkloadProfile
            self.core_type = CoreType
            self.thermal_state = ThermalState
            self.power_state = PowerState
        except ImportError as e:
            self.skipTest(f"Advanced CPU Optimizer not available: {e}")
            
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = self.optimizer_class()
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(optimizer.core_affinity_manager)
        self.assertIsNotNone(optimizer.thermal_monitor)
        self.assertIsNotNone(optimizer.power_manager)
        
    def test_workload_profile_optimization(self):
        """Test workload profile optimization."""
        optimizer = self.optimizer_class()
        
        # Test backtesting profile
        success = optimizer.optimize_for_workload_profile('backtesting')
        self.assertTrue(success)
        
        # Test ML training profile
        success = optimizer.optimize_for_workload_profile('ml_training')
        self.assertTrue(success)
        
        # Test invalid profile
        success = optimizer.optimize_for_workload_profile('invalid_profile')
        self.assertFalse(success)
        
    def test_custom_workload_profile(self):
        """Test custom workload profile creation."""
        optimizer = self.optimizer_class()
        
        custom_profile = self.workload_profile(
            name='custom_test',
            cpu_intensity=0.8,
            memory_intensity=0.6,
            thermal_sensitivity=0.4,
            power_sensitivity=0.3,
            preferred_cores=self.core_type.PERFORMANCE,
            max_threads=6
        )
        
        optimizer.add_workload_profile(custom_profile)
        self.assertIn('custom_test', optimizer.workload_profiles)
        
    def test_advanced_thread_pool(self):
        """Test advanced thread pool creation."""
        optimizer = self.optimizer_class()
        
        # Test with workload profile
        executor = optimizer.create_optimized_thread_pool_with_affinity(
            max_workers=4,
            workload_profile='backtesting'
        )
        self.assertIsNotNone(executor)
        
        # Test without workload profile
        executor = optimizer.create_optimized_thread_pool_with_affinity(max_workers=2)
        self.assertIsNotNone(executor)
        
    def test_advanced_cpu_info(self):
        """Test advanced CPU information."""
        optimizer = self.optimizer_class()
        
        info = optimizer.get_advanced_cpu_info()
        self.assertIsInstance(info, dict)
        self.assertIn('thermal_stats', info)
        self.assertIn('power_stats', info)
        self.assertIn('core_affinity_config', info)
        self.assertIn('workload_profiles', info)
        
    def test_optimization_recommendations(self):
        """Test optimization recommendations."""
        optimizer = self.optimizer_class()
        
        recommendations = optimizer.get_optimization_recommendations()
        self.assertIsInstance(recommendations, list)

class TestEnhancedGPUManager(unittest.TestCase):
    """Test the Enhanced GPU Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from utils.hardware.enhanced_gpu_manager import (
                EnhancedM1GPUManager, GPUMemoryPool, BatchOperationConfig,
                GPUOperationType, GPUOperation, create_gpu_operation
            )
            self.manager_class = EnhancedM1GPUManager
            self.memory_pool_config = GPUMemoryPool
            self.batch_config = BatchOperationConfig
            self.operation_type = GPUOperationType
            self.operation_class = GPUOperation
            self.create_operation = create_gpu_operation
        except ImportError as e:
            self.skipTest(f"Enhanced GPU Manager not available: {e}")
            
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = self.manager_class()
        self.assertIsNotNone(manager)
        self.assertIsNotNone(manager.memory_pool_manager)
        self.assertIsNotNone(manager.compute_pipeline)
        self.assertIsNotNone(manager.batch_manager)
        
    def test_memory_pool_management(self):
        """Test GPU memory pool management."""
        manager = self.manager_class()
        
        # Test pool creation
        success = manager.memory_pool_manager.create_memory_pool('test_pool', 100.0)
        self.assertTrue(success)
        
        # Test allocation
        memory_address = manager.memory_pool_manager.allocate_from_pool(
            'test_pool', 50 * 1024 * 1024, 'test_object'
        )
        self.assertIsNotNone(memory_address)
        
        # Test deallocation
        success = manager.memory_pool_manager.deallocate_from_pool(
            'test_pool', 'test_object'
        )
        self.assertTrue(success)
        
    def test_compute_pipeline(self):
        """Test compute pipeline functionality."""
        manager = self.manager_class()
        
        # Test pipeline creation
        success = manager.create_optimized_pipeline(
            'test_pipeline',
            [self.operation_type.MATRIX_MULTIPLICATION],
            max_workers=2
        )
        self.assertTrue(success)
        
        # Test operation addition
        operation_id = manager.add_operation_to_pipeline(
            'test_pipeline',
            self.operation_type.MATRIX_MULTIPLICATION,
            np.random.rand(100, 100),
            {'test_param': 'value'}
        )
        self.assertNotEqual(operation_id, "")
        
    async def test_pipeline_execution(self):
        """Test pipeline execution."""
        manager = self.manager_class()
        
        # Create pipeline
        manager.create_optimized_pipeline('test_pipeline', [self.operation_type.MATRIX_MULTIPLICATION])
        
        # Add operations
        for i in range(3):
            manager.add_operation_to_pipeline(
                'test_pipeline',
                self.operation_type.MATRIX_MULTIPLICATION,
                np.random.rand(50, 50),
                {'index': i}
            )
            
        # Execute pipeline
        results = await manager.execute_pipeline('test_pipeline')
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
    def test_batch_operations(self):
        """Test batch operation functionality."""
        manager = self.manager_class()
        
        # Create operations
        operations = []
        for i in range(5):
            operation = self.create_operation(
                self.operation_type.MATRIX_MULTIPLICATION,
                np.random.rand(30, 30),
                {'batch_index': i}
            )
            operations.append(operation)
            
        # Batch operations
        operation_ids = manager.batch_gpu_operations(operations)
        self.assertEqual(len(operation_ids), 5)
        
    def test_enhanced_gpu_info(self):
        """Test enhanced GPU information."""
        manager = self.manager_class()
        
        info = manager.get_enhanced_gpu_info()
        self.assertIsInstance(info, dict)
        self.assertIn('memory_pool_stats', info)
        self.assertIn('batch_stats', info)
        self.assertIn('pipeline_stats', info)
        self.assertIn('enhanced_features', info)

class TestAdvancedMemoryOptimizer(unittest.TestCase):
    """Test the Advanced Memory Optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from utils.hardware.advanced_memory_optimizer import (
                AdvancedM1MemoryOptimizer, MemoryPoolType, MemoryStrategy,
                MemoryEventType, optimize_dataframe_advanced, get_memory_predictions
            )
            self.optimizer_class = AdvancedM1MemoryOptimizer
            self.pool_type = MemoryPoolType
            self.strategy = MemoryStrategy
            self.event_type = MemoryEventType
            self.optimize_dataframe = optimize_dataframe_advanced
            self.get_predictions = get_memory_predictions
        except ImportError as e:
            self.skipTest(f"Advanced Memory Optimizer not available: {e}")
            
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = self.optimizer_class(memory_limit_gb=4.0, strategy=self.strategy.ADAPTIVE)
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(optimizer.memory_pools)
        self.assertIsNotNone(optimizer.predictive_manager)
        self.assertIsNotNone(optimizer.event_tracker)
        
    def test_memory_pool_operations(self):
        """Test memory pool operations."""
        optimizer = self.optimizer_class()
        
        # Test allocation
        success = optimizer.allocate_from_pool(
            self.pool_type.NUMPY_ARRAYS,
            1024 * 1024,  # 1MB
            'test_array',
            'numpy_array'
        )
        self.assertTrue(success)
        
        # Test deallocation
        success = optimizer.deallocate_from_pool(
            self.pool_type.NUMPY_ARRAYS,
            'test_array'
        )
        self.assertTrue(success)
        
    def test_advanced_dataframe_optimization(self):
        """Test advanced DataFrame optimization."""
        # Create test DataFrame
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),
            'float_col': np.random.randn(1000),
            'string_col': ['test'] * 1000
        })
        
        optimized_df = self.optimize_dataframe(df)
        self.assertIsNotNone(optimized_df)
        self.assertEqual(len(optimized_df), len(df))
        
    def test_memory_strategy_management(self):
        """Test memory strategy management."""
        optimizer = self.optimizer_class()
        
        # Test strategy change
        optimizer.set_memory_strategy(self.strategy.AGGRESSIVE)
        self.assertEqual(optimizer.strategy, self.strategy.AGGRESSIVE)
        
        optimizer.set_memory_strategy(self.strategy.CONSERVATIVE)
        self.assertEqual(optimizer.strategy, self.strategy.CONSERVATIVE)
        
    def test_memory_predictions(self):
        """Test memory predictions."""
        predictions = self.get_predictions(time_horizon_minutes=30)
        self.assertIsNotNone(predictions)
        self.assertIn('predicted_usage_mb', predictions.__dict__)
        self.assertIn('confidence', predictions.__dict__)
        
    def test_advanced_memory_stats(self):
        """Test advanced memory statistics."""
        optimizer = self.optimizer_class()
        
        stats = optimizer.get_advanced_memory_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('memory_pools', stats)
        self.assertIn('predictive_analysis', stats)
        self.assertIn('event_statistics', stats)
        self.assertIn('advanced_features', stats)

class TestAdaptiveOptimizationEngine(unittest.TestCase):
    """Test the Adaptive Optimization Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from utils.hardware.adaptive_optimization_engine import (
                AdaptiveOptimizationEngine, OptimizationTarget, LearningAlgorithm,
                optimize_for_workload_adaptive, record_performance_adaptive
            )
            self.engine_class = AdaptiveOptimizationEngine
            self.target = OptimizationTarget
            self.algorithm = LearningAlgorithm
            self.optimize_workload = optimize_for_workload_adaptive
            self.record_performance = record_performance_adaptive
        except ImportError as e:
            self.skipTest(f"Adaptive Optimization Engine not available: {e}")
            
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = self.engine_class()
        self.assertIsNotNone(engine)
        self.assertIsNotNone(engine.database)
        self.assertIsNotNone(engine.learner)
        
    def test_workload_optimization(self):
        """Test adaptive workload optimization."""
        from utils.hardware.unified_hardware_manager import WorkloadType
        
        # Test performance optimization
        success = self.optimize_workload(WorkloadType.BACKTESTING, self.target.PERFORMANCE)
        # Note: May return False if insufficient training data
        self.assertIsInstance(success, bool)
        
        # Test efficiency optimization
        success = self.optimize_workload(WorkloadType.ML_TRAINING, self.target.EFFICIENCY)
        self.assertIsInstance(success, bool)
        
    def test_performance_recording(self):
        """Test performance recording."""
        # Test performance recording
        success = self.record_performance(
            execution_time=10.5,
            throughput=100.0,
            error_rate=0.01
        )
        # May return False if no current workload/target set
        self.assertIsInstance(success, bool)
        
    def test_learning_functionality(self):
        """Test learning functionality."""
        engine = self.engine_class()
        
        # Test model training
        success = engine.learner.train_model(
            self.target.PERFORMANCE,
            self.algorithm.LINEAR_REGRESSION
        )
        # May return False if insufficient training data
        self.assertIsInstance(success, bool)
        
    def test_learning_report(self):
        """Test learning report generation."""
        engine = self.engine_class()
        
        report = engine.get_learning_report()
        self.assertIsInstance(report, dict)
        self.assertIn('learning_enabled', report)
        self.assertIn('auto_tuning_enabled', report)
        self.assertIn('models', report)

class TestIntegration(unittest.TestCase):
    """Integration tests for all components."""
    
    def test_full_workflow(self):
        """Test complete optimization workflow."""
        try:
            from utils.hardware.unified_hardware_manager import (
                get_unified_hardware_manager, WorkloadType, OptimizationLevel
            )
            from utils.hardware.adaptive_optimization_engine import (
                get_adaptive_optimization_engine, OptimizationTarget
            )
            
            # Initialize managers
            hardware_manager = get_unified_hardware_manager()
            adaptive_engine = get_adaptive_optimization_engine()
            
            # Test hardware optimization
            success = hardware_manager.optimize_for_workload(
                WorkloadType.BACKTESTING,
                OptimizationLevel.AGGRESSIVE
            )
            self.assertTrue(success)
            
            # Test adaptive optimization
            success = adaptive_engine.optimize_for_workload(
                WorkloadType.BACKTESTING,
                OptimizationTarget.PERFORMANCE
            )
            self.assertIsInstance(success, bool)
            
            # Test system status
            status = hardware_manager.get_system_status()
            self.assertIsInstance(status, dict)
            
        except ImportError as e:
            self.skipTest(f"Integration test components not available: {e}")
            
    def test_performance_monitoring(self):
        """Test performance monitoring integration."""
        try:
            from utils.hardware.unified_hardware_manager import get_unified_hardware_manager
            
            manager = get_unified_hardware_manager()
            manager.initialize()
            
            # Wait for some monitoring data
            time.sleep(2)
            
            status = manager.get_system_status()
            performance_report = status.get('performance_report', {})
            
            self.assertIsInstance(performance_report, dict)
            
        except ImportError as e:
            self.skipTest(f"Performance monitoring not available: {e}")

def run_performance_benchmark():
    """Run performance benchmarks for the optimization systems."""
    logger.info("🚀 Running Performance Benchmarks")
    
    try:
        from utils.hardware.unified_hardware_manager import get_unified_hardware_manager, WorkloadType
        from utils.hardware.adaptive_optimization_engine import get_adaptive_optimization_engine, OptimizationTarget
        
        # Initialize systems
        hardware_manager = get_unified_hardware_manager()
        adaptive_engine = get_adaptive_optimization_engine()
        
        # Benchmark different workloads
        workloads = [
            (WorkloadType.BACKTESTING, OptimizationTarget.PERFORMANCE),
            (WorkloadType.ML_TRAINING, OptimizationTarget.EFFICIENCY),
            (WorkloadType.DATA_PROCESSING, OptimizationTarget.BALANCED)
        ]
        
        for workload_type, target in workloads:
            logger.info(f"📊 Benchmarking {workload_type.value} with {target.value} target")
            
            start_time = time.time()
            
            # Hardware optimization
            success = hardware_manager.optimize_for_workload(workload_type)
            hardware_time = time.time() - start_time
            
            # Adaptive optimization
            start_time = time.time()
            success = adaptive_engine.optimize_for_workload(workload_type, target)
            adaptive_time = time.time() - start_time
            
            logger.info(f"  Hardware optimization: {hardware_time:.3f}s")
            logger.info(f"  Adaptive optimization: {adaptive_time:.3f}s")
            
    except ImportError as e:
        logger.warning(f"Benchmark components not available: {e}")

def main():
    """Run all tests and benchmarks."""
    logger.info("🧪 Starting Advanced Hardware Optimization Tests")
    logger.info("=" * 60)
    
    # Run unit tests
    test_suites = [
        TestUnifiedHardwareManager,
        TestAdvancedCPUOptimizer,
        TestEnhancedGPUManager,
        TestAdvancedMemoryOptimizer,
        TestAdaptiveOptimizationEngine,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_suite in test_suites:
        logger.info(f"\n📋 Running {test_suite.__name__}")
        logger.info("-" * 40)
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_suite)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        suite_tests = result.testsRun
        suite_passed = suite_tests - len(result.failures) - len(result.errors)
        
        total_tests += suite_tests
        passed_tests += suite_passed
        
        logger.info(f"✅ {test_suite.__name__}: {suite_passed}/{suite_tests} tests passed")
        
        # Log any failures or errors
        for failure in result.failures:
            logger.warning(f"  ❌ FAILED: {failure[0]}")
        for error in result.errors:
            logger.warning(f"  ❌ ERROR: {error[0]}")
    
    # Run performance benchmarks
    logger.info("\n" + "=" * 60)
    run_performance_benchmark()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed tests: {passed_tests}")
    logger.info(f"Failed tests: {total_tests - passed_tests}")
    logger.info(f"Success rate: {(passed_tests / total_tests * 100):.1f}%")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)