"""
Simple Test Script for Enhanced Step 2: Data Reading with Comprehensive Utility Integration

This script tests the enhanced step02 implementation without external dependencies
to ensure all utilities are properly integrated and functioning correctly.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleStep02Tester:
    """Simple test class for enhanced step02 implementation."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SimpleStep02Tester")
        self.test_results = {}
    
    def test_dependency_injection_imports(self) -> bool:
        """Test that dependency injection can be imported."""
        self.logger.info("🧪 Testing dependency injection imports...")
        
        try:
            # Test importing dependency injection
            from src.training.steps.data_collection.step02_dependency_injection import (
                DependencyInjectionContainer, UtilityManager, get_container, get_utility_manager
            )
            
            assert DependencyInjectionContainer is not None, "DependencyInjectionContainer should not be None"
            assert UtilityManager is not None, "UtilityManager should not be None"
            assert get_container is not None, "get_container function should not be None"
            assert get_utility_manager is not None, "get_utility_manager function should not be None"
            
            self.logger.info("✅ Dependency injection imports test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Dependency injection imports test failed: {e}")
            return False
    
    def test_enhanced_step02_imports(self) -> bool:
        """Test that enhanced step02 can be imported."""
        self.logger.info("🧪 Testing enhanced step02 imports...")
        
        try:
            # Test importing enhanced step02
            from src.training.steps.data_collection.step02_enhanced_with_utilities import (
                run_step_enhanced, EnhancedDataReadingStep, enhanced_monitor
            )
            
            assert run_step_enhanced is not None, "run_step_enhanced function should not be None"
            assert EnhancedDataReadingStep is not None, "EnhancedDataReadingStep class should not be None"
            assert enhanced_monitor is not None, "enhanced_monitor should not be None"
            
            self.logger.info("✅ Enhanced step02 imports test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced step02 imports test failed: {e}")
            return False
    
    def test_utility_module_imports(self) -> bool:
        """Test that utility modules can be imported."""
        self.logger.info("🧪 Testing utility module imports...")
        
        try:
            # Test importing utility modules
            from src.utils.common_operations import safe_mean, safe_std, get_current_datetime
            from src.utils.common_utilities import validate_dataframe_columns, calculate_data_quality_metrics
            from src.utils.math_validation import safe_divide, safe_log, validate_finite
            from src.utils.parquet_utils import ParquetUtils, get_parquet_utils
            from src.utils.serialization_utils import JSONSerializer, save_json, load_json
            from src.utils.data_processing_utils import DataFrameValidator, validate_dataframe
            from src.utils.m1_gpu_utils import M1GPUManager, get_m1_gpu_manager
            from src.utils.m1_memory_optimizer import M1MemoryOptimizer, get_m1_memory_optimizer
            from src.utils.m1_cpu_optimizer import M1CPUOptimizer, get_m1_cpu_optimizer
            
            # Test that functions are callable
            assert callable(safe_mean), "safe_mean should be callable"
            assert callable(safe_std), "safe_std should be callable"
            assert callable(get_current_datetime), "get_current_datetime should be callable"
            assert callable(validate_dataframe_columns), "validate_dataframe_columns should be callable"
            assert callable(calculate_data_quality_metrics), "calculate_data_quality_metrics should be callable"
            assert callable(safe_divide), "safe_divide should be callable"
            assert callable(safe_log), "safe_log should be callable"
            assert callable(validate_finite), "validate_finite should be callable"
            assert callable(get_parquet_utils), "get_parquet_utils should be callable"
            assert callable(save_json), "save_json should be callable"
            assert callable(load_json), "load_json should be callable"
            assert callable(validate_dataframe), "validate_dataframe should be callable"
            assert callable(get_m1_gpu_manager), "get_m1_gpu_manager should be callable"
            assert callable(get_m1_memory_optimizer), "get_m1_memory_optimizer should be callable"
            assert callable(get_m1_cpu_optimizer), "get_m1_cpu_optimizer should be callable"
            
            # Test that classes are classes
            assert isinstance(ParquetUtils, type), "ParquetUtils should be a class"
            assert isinstance(JSONSerializer, type), "JSONSerializer should be a class"
            assert isinstance(DataFrameValidator, type), "DataFrameValidator should be a class"
            assert isinstance(M1GPUManager, type), "M1GPUManager should be a class"
            assert isinstance(M1MemoryOptimizer, type), "M1MemoryOptimizer should be a class"
            assert isinstance(M1CPUOptimizer, type), "M1CPUOptimizer should be a class"
            
            self.logger.info("✅ Utility module imports test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Utility module imports test failed: {e}")
            return False
    
    def test_utility_functions_basic(self) -> bool:
        """Test basic utility functions without external dependencies."""
        self.logger.info("🧪 Testing basic utility functions...")
        
        try:
            from src.utils.common_operations import safe_mean, safe_std, get_current_datetime, format_datetime
            from src.utils.math_validation import safe_divide, safe_log, validate_finite
            
            # Test safe_mean
            test_data = [1, 2, 3, 4, 5]
            mean_result = safe_mean(test_data)
            assert mean_result == 3.0, f"Expected mean 3.0, got {mean_result}"
            
            # Test safe_std
            std_result = safe_std(test_data)
            assert std_result > 0, f"Expected positive std, got {std_result}"
            
            # Test safe_divide
            div_result = safe_divide(10, 2)
            assert div_result == 5.0, f"Expected 5.0, got {div_result}"
            
            # Test division by zero
            div_zero_result = safe_divide(10, 0)
            assert div_zero_result == 0.0, f"Expected 0.0 for division by zero, got {div_zero_result}"
            
            # Test safe_log
            log_result = safe_log(10)
            assert log_result > 0, f"Expected positive log result, got {log_result}"
            
            # Test log of zero
            log_zero_result = safe_log(0)
            assert log_zero_result == 0.0, f"Expected 0.0 for log of zero, got {log_zero_result}"
            
            # Test validate_finite
            finite_result = validate_finite(42.0, "test_value")
            assert finite_result == 42.0, f"Expected 42.0, got {finite_result}"
            
            # Test get_current_datetime
            current_time = get_current_datetime()
            assert current_time is not None, "Current datetime should not be None"
            
            # Test format_datetime
            formatted_time = format_datetime(current_time)
            assert isinstance(formatted_time, str), "Formatted datetime should be string"
            assert len(formatted_time) > 0, "Formatted datetime should not be empty"
            
            self.logger.info("✅ Basic utility functions test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Basic utility functions test failed: {e}")
            return False
    
    def test_dependency_injection_container(self) -> bool:
        """Test dependency injection container functionality."""
        self.logger.info("🧪 Testing dependency injection container...")
        
        try:
            from src.training.steps.data_collection.step02_dependency_injection import get_container
            
            # Test container creation
            container = get_container()
            assert container is not None, "Container should not be None"
            assert hasattr(container, 'get'), "Container should have get method"
            assert hasattr(container, 'register_singleton'), "Container should have register_singleton method"
            assert hasattr(container, 'register_transient'), "Container should have register_transient method"
            
            # Test utility manager retrieval
            utility_manager = container.get('utility_manager')
            assert utility_manager is not None, "Utility manager should not be None"
            assert hasattr(utility_manager, 'initialize'), "Utility manager should have initialize method"
            assert hasattr(utility_manager, 'common_ops'), "Utility manager should have common_ops property"
            assert hasattr(utility_manager, 'common_utils'), "Utility manager should have common_utils property"
            assert hasattr(utility_manager, 'math_validation'), "Utility manager should have math_validation property"
            assert hasattr(utility_manager, 'parquet_utils'), "Utility manager should have parquet_utils property"
            assert hasattr(utility_manager, 'serialization_utils'), "Utility manager should have serialization_utils property"
            assert hasattr(utility_manager, 'data_processing_utils'), "Utility manager should have data_processing_utils property"
            assert hasattr(utility_manager, 'm1_gpu_utils'), "Utility manager should have m1_gpu_utils property"
            assert hasattr(utility_manager, 'm1_memory_optimizer'), "Utility manager should have m1_memory_optimizer property"
            assert hasattr(utility_manager, 'm1_cpu_optimizer'), "Utility manager should have m1_cpu_optimizer property"
            
            self.logger.info("✅ Dependency injection container test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Dependency injection container test failed: {e}")
            return False
    
    def test_enhanced_step02_class_creation(self) -> bool:
        """Test enhanced step02 class creation."""
        self.logger.info("🧪 Testing enhanced step02 class creation...")
        
        try:
            from src.training.steps.data_collection.step02_enhanced_with_utilities import EnhancedDataReadingStep
            
            # Test class creation
            config = {
                'max_workers': 2,
                'chunk_size': 1000,
                'min_rows': 100,
                'max_duplicate_ratio': 0.01,
                'max_gap_seconds': 0.5
            }
            
            step = EnhancedDataReadingStep(config)
            assert step is not None, "EnhancedDataReadingStep should not be None"
            assert hasattr(step, 'utility_manager'), "Step should have utility_manager"
            assert hasattr(step, 'monitor'), "Step should have monitor"
            assert hasattr(step, 'execute'), "Step should have execute method"
            assert hasattr(step, 'read_unified_data_enhanced'), "Step should have read_unified_data_enhanced method"
            assert hasattr(step, 'validate_data_quality_enhanced'), "Step should have validate_data_quality_enhanced method"
            
            # Test configuration
            assert step.max_workers == 2, f"Expected max_workers 2, got {step.max_workers}"
            assert step.chunk_size == 1000, f"Expected chunk_size 1000, got {step.chunk_size}"
            assert step.min_rows == 100, f"Expected min_rows 100, got {step.min_rows}"
            
            self.logger.info("✅ Enhanced step02 class creation test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced step02 class creation test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        self.logger.info("🚀 Starting simple step02 utility integration tests...")
        
        test_methods = [
            ('dependency_injection_imports', self.test_dependency_injection_imports),
            ('enhanced_step02_imports', self.test_enhanced_step02_imports),
            ('utility_module_imports', self.test_utility_module_imports),
            ('utility_functions_basic', self.test_utility_functions_basic),
            ('dependency_injection_container', self.test_dependency_injection_container),
            ('enhanced_step02_class_creation', self.test_enhanced_step02_class_creation),
        ]
        
        results = {}
        
        for test_name, test_method in test_methods:
            try:
                self.logger.info(f"🧪 Running {test_name}...")
                result = test_method()
                results[test_name] = result
                if result:
                    self.logger.info(f"✅ {test_name} passed")
                else:
                    self.logger.error(f"❌ {test_name} failed")
            except Exception as e:
                self.logger.error(f"❌ {test_name} failed with exception: {e}")
                results[test_name] = False
        
        return results
    
    def print_test_summary(self, results: Dict[str, bool]):
        """Print test summary."""
        self.logger.info("📊 Test Summary:")
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        failed_tests = total_tests - passed_tests
        
        self.logger.info(f"   Total tests: {total_tests}")
        self.logger.info(f"   Passed: {passed_tests}")
        self.logger.info(f"   Failed: {failed_tests}")
        self.logger.info(f"   Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            self.logger.info("❌ Failed tests:")
            for test_name, result in results.items():
                if not result:
                    self.logger.info(f"   - {test_name}")
        else:
            self.logger.info("✅ All tests passed!")

def main():
    """Main test function."""
    logger.info("🚀 Starting Simple Step 2 Enhanced Utility Integration Tests")
    
    tester = SimpleStep02Tester()
    results = tester.run_all_tests()
    tester.print_test_summary(results)
    
    # Check if all tests passed
    all_passed = all(results.values())
    if all_passed:
        logger.info("🎉 All tests passed! Enhanced step02 utility integration is working correctly.")
        return 0
    else:
        logger.error("💥 Some tests failed. Please check the implementation.")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)