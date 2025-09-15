#!/usr/bin/env python3
"""
Basic Functionality Tests for Hardware Optimizations.

This script tests the core functionality without external dependencies.
"""

import sys
import os
import logging
import time
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestBasicImports(unittest.TestCase):
    """Test basic imports and initialization."""
    
    def test_basic_imports(self):
        """Test that basic modules can be imported."""
        try:
            from utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
            from utils.hardware.m1_gpu_utils import M1GPUManager
            from utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
            logger.info("✅ Basic hardware modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import basic hardware modules: {e}")
    
    def test_advanced_imports(self):
        """Test that advanced modules can be imported."""
        try:
            from utils.hardware.unified_hardware_manager import UnifiedHardwareManager
            from utils.hardware.advanced_cpu_optimizer import AdvancedM1CPUOptimizer
            from utils.hardware.enhanced_gpu_manager import EnhancedM1GPUManager
            from utils.hardware.advanced_memory_optimizer import AdvancedM1MemoryOptimizer
            from utils.hardware.adaptive_optimization_engine import AdaptiveOptimizationEngine
            logger.info("✅ Advanced hardware modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import advanced hardware modules: {e}")
    
    def test_package_imports(self):
        """Test package-level imports."""
        try:
            from utils.hardware import (
                M1CPUOptimizer, M1GPUManager, M1MemoryOptimizer,
                get_unified_hardware_manager, get_advanced_cpu_optimizer,
                get_enhanced_gpu_manager, get_advanced_memory_optimizer,
                get_adaptive_optimization_engine
            )
            logger.info("✅ Package-level imports successful")
        except ImportError as e:
            self.fail(f"Failed to import from package: {e}")

class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality without external dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock external dependencies
        self.psutil_patcher = patch('utils.hardware.m1_cpu_optimizer.psutil')
        self.psutil_mock = self.psutil_patcher.start()
        self.psutil_mock.cpu_count.return_value = 8
        self.psutil_mock.virtual_memory.return_value = Mock(percent=50.0, total=8589934592)
        
        self.platform_patcher = patch('utils.hardware.m1_cpu_optimizer.platform')
        self.platform_mock = self.platform_patcher.start()
        self.platform_mock.system.return_value = 'Darwin'
        self.platform_mock.machine.return_value = 'arm64'
        
        self.subprocess_patcher = patch('utils.hardware.m1_cpu_optimizer.subprocess')
        self.subprocess_mock = self.subprocess_patcher.start()
        self.subprocess_mock.run.return_value = Mock(returncode=0, stdout='Apple M1')
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.psutil_patcher.stop()
        self.platform_patcher.stop()
        self.subprocess_patcher.stop()
    
    def test_m1_cpu_optimizer_creation(self):
        """Test M1 CPU optimizer creation."""
        try:
            from utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
            optimizer = M1CPUOptimizer()
            self.assertIsNotNone(optimizer)
            self.assertTrue(hasattr(optimizer, 'cpu_count'))
            self.assertTrue(hasattr(optimizer, 'is_m1'))
            logger.info("✅ M1 CPU Optimizer created successfully")
        except Exception as e:
            self.fail(f"Failed to create M1 CPU Optimizer: {e}")
    
    def test_m1_gpu_manager_creation(self):
        """Test M1 GPU manager creation."""
        try:
            from utils.hardware.m1_gpu_utils import M1GPUManager
            manager = M1GPUManager()
            self.assertIsNotNone(manager)
            self.assertTrue(hasattr(manager, 'is_m1'))
            self.assertTrue(hasattr(manager, 'mps_available'))
            logger.info("✅ M1 GPU Manager created successfully")
        except Exception as e:
            self.fail(f"Failed to create M1 GPU Manager: {e}")
    
    def test_m1_memory_optimizer_creation(self):
        """Test M1 memory optimizer creation."""
        try:
            from utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
            optimizer = M1MemoryOptimizer()
            self.assertIsNotNone(optimizer)
            self.assertTrue(hasattr(optimizer, 'memory_limit_gb'))
            self.assertTrue(hasattr(optimizer, 'thresholds'))
            logger.info("✅ M1 Memory Optimizer created successfully")
        except Exception as e:
            self.fail(f"Failed to create M1 Memory Optimizer: {e}")

class TestAdvancedFunctionality(unittest.TestCase):
    """Test advanced functionality with mocked dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock all external dependencies
        self.psutil_patcher = patch('utils.hardware.unified_hardware_manager.psutil')
        self.psutil_mock = self.psutil_patcher.start()
        self.psutil_mock.cpu_percent.return_value = 50.0
        self.psutil_mock.virtual_memory.return_value = Mock(percent=50.0, total=8589934592)
        
        self.platform_patcher = patch('utils.hardware.unified_hardware_manager.platform')
        self.platform_mock = self.platform_patcher.start()
        self.platform_mock.system.return_value = 'Darwin'
        
        self.subprocess_patcher = patch('utils.hardware.unified_hardware_manager.subprocess')
        self.subprocess_mock = self.subprocess_patcher.start()
        self.subprocess_mock.run.return_value = Mock(returncode=0, stdout='Apple M1')
        
        # Mock numpy and pandas
        self.numpy_patcher = patch('utils.hardware.enhanced_gpu_manager.numpy')
        self.numpy_mock = self.numpy_patcher.start()
        self.numpy_mock.random.rand.return_value = [[1, 2], [3, 4]]
        self.numpy_mock.random.randint.return_value = 100
        self.numpy_mock.random.uniform.return_value = 0.5
        self.numpy_mock.random.normal.return_value = 0.1
        
        self.pandas_patcher = patch('utils.hardware.enhanced_gpu_manager.pandas')
        self.pandas_mock = self.pandas_patcher.start()
        self.pandas_mock.DataFrame.return_value = Mock()
        
        # Mock torch
        self.torch_patcher = patch('utils.hardware.enhanced_gpu_manager.torch')
        self.torch_mock = self.torch_patcher.start()
        self.torch_mock.backends.mps.is_available.return_value = True
        self.torch_mock.cuda.is_available.return_value = False
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.psutil_patcher.stop()
        self.platform_patcher.stop()
        self.subprocess_patcher.stop()
        self.numpy_patcher.stop()
        self.pandas_patcher.stop()
        self.torch_patcher.stop()
    
    def test_unified_hardware_manager_creation(self):
        """Test unified hardware manager creation."""
        try:
            from utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig
            config = HardwareConfig(memory_limit_gb=4.0)
            manager = UnifiedHardwareManager(config)
            self.assertIsNotNone(manager)
            self.assertIsNotNone(manager.cpu_optimizer)
            self.assertIsNotNone(manager.gpu_manager)
            self.assertIsNotNone(manager.memory_optimizer)
            logger.info("✅ Unified Hardware Manager created successfully")
        except Exception as e:
            self.fail(f"Failed to create Unified Hardware Manager: {e}")
    
    def test_advanced_cpu_optimizer_creation(self):
        """Test advanced CPU optimizer creation."""
        try:
            from utils.hardware.advanced_cpu_optimizer import AdvancedM1CPUOptimizer
            optimizer = AdvancedM1CPUOptimizer()
            self.assertIsNotNone(optimizer)
            self.assertIsNotNone(optimizer.core_affinity_manager)
            self.assertIsNotNone(optimizer.thermal_monitor)
            self.assertIsNotNone(optimizer.power_manager)
            logger.info("✅ Advanced CPU Optimizer created successfully")
        except Exception as e:
            self.fail(f"Failed to create Advanced CPU Optimizer: {e}")
    
    def test_enhanced_gpu_manager_creation(self):
        """Test enhanced GPU manager creation."""
        try:
            from utils.hardware.enhanced_gpu_manager import EnhancedM1GPUManager
            manager = EnhancedM1GPUManager()
            self.assertIsNotNone(manager)
            self.assertIsNotNone(manager.memory_pool_manager)
            self.assertIsNotNone(manager.compute_pipeline)
            self.assertIsNotNone(manager.batch_manager)
            logger.info("✅ Enhanced GPU Manager created successfully")
        except Exception as e:
            self.fail(f"Failed to create Enhanced GPU Manager: {e}")
    
    def test_advanced_memory_optimizer_creation(self):
        """Test advanced memory optimizer creation."""
        try:
            from utils.hardware.advanced_memory_optimizer import AdvancedM1MemoryOptimizer, MemoryStrategy
            optimizer = AdvancedM1MemoryOptimizer(memory_limit_gb=4.0, strategy=MemoryStrategy.ADAPTIVE)
            self.assertIsNotNone(optimizer)
            self.assertIsNotNone(optimizer.memory_pools)
            self.assertIsNotNone(optimizer.predictive_manager)
            self.assertIsNotNone(optimizer.event_tracker)
            logger.info("✅ Advanced Memory Optimizer created successfully")
        except Exception as e:
            self.fail(f"Failed to create Advanced Memory Optimizer: {e}")
    
    def test_adaptive_optimization_engine_creation(self):
        """Test adaptive optimization engine creation."""
        try:
            from utils.hardware.adaptive_optimization_engine import AdaptiveOptimizationEngine
            engine = AdaptiveOptimizationEngine()
            self.assertIsNotNone(engine)
            self.assertIsNotNone(engine.database)
            self.assertIsNotNone(engine.learner)
            logger.info("✅ Adaptive Optimization Engine created successfully")
        except Exception as e:
            self.fail(f"Failed to create Adaptive Optimization Engine: {e}")

class TestConfigurationAndEnums(unittest.TestCase):
    """Test configuration classes and enums."""
    
    def test_workload_types(self):
        """Test workload type enums."""
        try:
            from utils.hardware.unified_hardware_manager import WorkloadType
            self.assertEqual(WorkloadType.BACKTESTING.value, "backtesting")
            self.assertEqual(WorkloadType.ML_TRAINING.value, "ml_training")
            self.assertEqual(WorkloadType.DATA_PROCESSING.value, "data_processing")
            logger.info("✅ Workload types defined correctly")
        except Exception as e:
            self.fail(f"Failed to test workload types: {e}")
    
    def test_optimization_levels(self):
        """Test optimization level enums."""
        try:
            from utils.hardware.unified_hardware_manager import OptimizationLevel
            self.assertEqual(OptimizationLevel.MINIMAL.value, "minimal")
            self.assertEqual(OptimizationLevel.BALANCED.value, "balanced")
            self.assertEqual(OptimizationLevel.AGGRESSIVE.value, "aggressive")
            logger.info("✅ Optimization levels defined correctly")
        except Exception as e:
            self.fail(f"Failed to test optimization levels: {e}")
    
    def test_optimization_targets(self):
        """Test optimization target enums."""
        try:
            from utils.hardware.adaptive_optimization_engine import OptimizationTarget
            self.assertEqual(OptimizationTarget.PERFORMANCE.value, "performance")
            self.assertEqual(OptimizationTarget.EFFICIENCY.value, "efficiency")
            self.assertEqual(OptimizationTarget.BALANCED.value, "balanced")
            logger.info("✅ Optimization targets defined correctly")
        except Exception as e:
            self.fail(f"Failed to test optimization targets: {e}")
    
    def test_hardware_config(self):
        """Test hardware configuration."""
        try:
            from utils.hardware.unified_hardware_manager import HardwareConfig, OptimizationLevel
            config = HardwareConfig(
                memory_limit_gb=8.0,
                cpu_optimization_level=OptimizationLevel.AGGRESSIVE,
                enable_adaptive_optimization=True
            )
            self.assertEqual(config.memory_limit_gb, 8.0)
            self.assertEqual(config.cpu_optimization_level, OptimizationLevel.AGGRESSIVE)
            self.assertTrue(config.enable_adaptive_optimization)
            logger.info("✅ Hardware configuration works correctly")
        except Exception as e:
            self.fail(f"Failed to test hardware configuration: {e}")

class TestGlobalFunctions(unittest.TestCase):
    """Test global convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock external dependencies
        self.psutil_patcher = patch('utils.hardware.unified_hardware_manager.psutil')
        self.psutil_mock = self.psutil_patcher.start()
        self.psutil_mock.cpu_percent.return_value = 50.0
        self.psutil_mock.virtual_memory.return_value = Mock(percent=50.0, total=8589934592)
        
        self.platform_patcher = patch('utils.hardware.unified_hardware_manager.platform')
        self.platform_mock = self.platform_patcher.start()
        self.platform_mock.system.return_value = 'Darwin'
        
        self.subprocess_patcher = patch('utils.hardware.unified_hardware_manager.subprocess')
        self.subprocess_mock = self.subprocess_patcher.start()
        self.subprocess_mock.run.return_value = Mock(returncode=0, stdout='Apple M1')
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.psutil_patcher.stop()
        self.platform_patcher.stop()
        self.subprocess_patcher.stop()
    
    def test_global_function_imports(self):
        """Test that global functions can be imported."""
        try:
            from utils.hardware import (
                get_unified_hardware_manager,
                get_advanced_cpu_optimizer,
                get_enhanced_gpu_manager,
                get_advanced_memory_optimizer,
                get_adaptive_optimization_engine
            )
            logger.info("✅ Global functions imported successfully")
        except Exception as e:
            self.fail(f"Failed to import global functions: {e}")
    
    def test_feature_availability(self):
        """Test feature availability functions."""
        try:
            from utils.hardware import get_feature_status, get_available_features, is_feature_available
            features = get_feature_status()
            self.assertIsInstance(features, dict)
            
            available = get_available_features()
            self.assertIsInstance(available, list)
            
            basic_available = is_feature_available('basic_cpu_optimization')
            self.assertTrue(basic_available)
            logger.info("✅ Feature availability functions work correctly")
        except Exception as e:
            self.fail(f"Failed to test feature availability: {e}")

def run_basic_tests():
    """Run basic functionality tests."""
    logger.info("🧪 Running Basic Hardware Optimization Tests")
    logger.info("=" * 60)
    
    # Create test suite
    test_suites = [
        TestBasicImports,
        TestBasicFunctionality,
        TestAdvancedFunctionality,
        TestConfigurationAndEnums,
        TestGlobalFunctions
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

def test_import_functionality():
    """Test that all modules can be imported without errors."""
    logger.info("🔍 Testing Import Functionality")
    logger.info("-" * 40)
    
    import_tests = [
        ("Basic CPU Optimizer", "utils.hardware.m1_cpu_optimizer", "M1CPUOptimizer"),
        ("Basic GPU Manager", "utils.hardware.m1_gpu_utils", "M1GPUManager"),
        ("Basic Memory Optimizer", "utils.hardware.m1_memory_optimizer", "M1MemoryOptimizer"),
        ("Unified Hardware Manager", "utils.hardware.unified_hardware_manager", "UnifiedHardwareManager"),
        ("Advanced CPU Optimizer", "utils.hardware.advanced_cpu_optimizer", "AdvancedM1CPUOptimizer"),
        ("Enhanced GPU Manager", "utils.hardware.enhanced_gpu_manager", "EnhancedM1GPUManager"),
        ("Advanced Memory Optimizer", "utils.hardware.advanced_memory_optimizer", "AdvancedM1MemoryOptimizer"),
        ("Adaptive Optimization Engine", "utils.hardware.adaptive_optimization_engine", "AdaptiveOptimizationEngine"),
    ]
    
    passed_imports = 0
    total_imports = len(import_tests)
    
    for test_name, module_name, class_name in import_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            logger.info(f"✅ {test_name}: Import successful")
            passed_imports += 1
        except Exception as e:
            logger.warning(f"❌ {test_name}: Import failed - {e}")
    
    logger.info(f"\nImport Test Results: {passed_imports}/{total_imports} successful")
    return passed_imports == total_imports

def main():
    """Run all tests."""
    logger.info("🚀 Starting Hardware Optimization Implementation Tests")
    logger.info("=" * 60)
    
    # Test imports first
    import_success = test_import_functionality()
    
    if not import_success:
        logger.error("❌ Import tests failed. Cannot proceed with functionality tests.")
        return False
    
    # Run functionality tests
    functionality_success = run_basic_tests()
    
    # Overall result
    overall_success = import_success and functionality_success
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Import tests: {'✅ PASSED' if import_success else '❌ FAILED'}")
    logger.info(f"Functionality tests: {'✅ PASSED' if functionality_success else '❌ FAILED'}")
    logger.info(f"Overall: {'🎉 SUCCESS' if overall_success else '❌ FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)