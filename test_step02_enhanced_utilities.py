"""
Test Script for Enhanced Step 2: Data Reading with Comprehensive Utility Integration

This script tests the enhanced step02 implementation to ensure all utilities
are properly integrated and functioning correctly.
"""

import asyncio
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import the enhanced step02 implementation
from src.training.steps.data_collection.step02_enhanced_with_utilities import (
    run_step_enhanced, EnhancedDataReadingStep, get_utility_manager, get_container
)

# Import dependency injection
from src.training.steps.data_collection.step02_dependency_injection import (
    DependencyInjectionContainer, UtilityManager
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Step02EnhancedTester:
    """Test class for enhanced step02 implementation."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Step02EnhancedTester")
        self.test_results = {}
        self.utility_manager = None
        self.container = None
    
    def setup_test_environment(self):
        """Setup test environment and initialize utilities."""
        self.logger.info("🔧 Setting up test environment...")
        
        try:
            # Initialize dependency injection container
            self.container = get_container()
            self.logger.info("✅ Dependency injection container initialized")
            
            # Initialize utility manager
            self.utility_manager = get_utility_manager()
            self.utility_manager.initialize()
            self.logger.info("✅ Utility manager initialized")
            
            # Test utility manager health
            health_status = self.utility_manager.get_health_status()
            self.logger.info(f"📊 Utility manager health: {health_status['status']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup test environment: {e}")
            return False
    
    def test_dependency_injection(self) -> bool:
        """Test dependency injection container functionality."""
        self.logger.info("🧪 Testing dependency injection container...")
        
        try:
            # Test container initialization
            if self.container is None:
                self.container = get_container()
            
            # Test utility manager retrieval
            utility_manager = self.container.get('utility_manager')
            assert utility_manager is not None, "Utility manager should not be None"
            
            # Test utility retrieval
            common_ops = utility_manager.common_ops
            assert common_ops is not None, "Common operations should not be None"
            assert len(common_ops) > 0, "Common operations should have functions"
            
            common_utils = utility_manager.common_utils
            assert common_utils is not None, "Common utilities should not be None"
            assert len(common_utils) > 0, "Common utilities should have functions"
            
            math_validation = utility_manager.math_validation
            assert math_validation is not None, "Math validation should not be None"
            assert len(math_validation) > 0, "Math validation should have functions"
            
            parquet_utils = utility_manager.parquet_utils
            assert parquet_utils is not None, "Parquet utils should not be None"
            
            serialization_utils = utility_manager.serialization_utils
            assert serialization_utils is not None, "Serialization utils should not be None"
            assert len(serialization_utils) > 0, "Serialization utils should have functions"
            
            data_processing_utils = utility_manager.data_processing_utils
            assert data_processing_utils is not None, "Data processing utils should not be None"
            assert len(data_processing_utils) > 0, "Data processing utils should have functions"
            
            m1_gpu_utils = utility_manager.m1_gpu_utils
            assert m1_gpu_utils is not None, "M1 GPU utils should not be None"
            assert len(m1_gpu_utils) > 0, "M1 GPU utils should have functions"
            
            m1_memory_optimizer = utility_manager.m1_memory_optimizer
            assert m1_memory_optimizer is not None, "M1 memory optimizer should not be None"
            assert len(m1_memory_optimizer) > 0, "M1 memory optimizer should have functions"
            
            m1_cpu_optimizer = utility_manager.m1_cpu_optimizer
            assert m1_cpu_optimizer is not None, "M1 CPU optimizer should not be None"
            assert len(m1_cpu_optimizer) > 0, "M1 CPU optimizer should have functions"
            
            self.logger.info("✅ Dependency injection container test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Dependency injection container test failed: {e}")
            return False
    
    def test_common_operations_integration(self) -> bool:
        """Test common operations utility integration."""
        self.logger.info("🧪 Testing common operations integration...")
        
        try:
            common_ops = self.utility_manager.common_ops
            
            # Test safe operations
            test_data = [1, 2, 3, 4, 5]
            mean_result = common_ops['safe_mean'](test_data)
            assert mean_result == 3.0, f"Expected mean 3.0, got {mean_result}"
            
            std_result = common_ops['safe_std'](test_data)
            assert std_result > 0, f"Expected positive std, got {std_result}"
            
            # Test safe file operations
            test_path = Path("/tmp/test_file.txt")
            exists_result = common_ops['safe_file_exists'](test_path)
            assert isinstance(exists_result, bool), "File exists check should return bool"
            
            # Test datetime operations
            current_time = common_ops['get_current_datetime']()
            assert current_time is not None, "Current datetime should not be None"
            
            formatted_time = common_ops['format_datetime'](current_time)
            assert isinstance(formatted_time, str), "Formatted datetime should be string"
            
            # Test safe data operations
            test_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            empty_df = common_ops['create_empty_dataframe'](['col1', 'col2'])
            assert len(empty_df) == 0, "Empty dataframe should have 0 rows"
            assert list(empty_df.columns) == ['col1', 'col2'], "Empty dataframe should have correct columns"
            
            self.logger.info("✅ Common operations integration test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Common operations integration test failed: {e}")
            return False
    
    def test_common_utilities_integration(self) -> bool:
        """Test common utilities integration."""
        self.logger.info("🧪 Testing common utilities integration...")
        
        try:
            common_utils = self.utility_manager.common_utils
            
            # Test DataFrame operations
            test_df = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [103, 104, 105],
                'volume': [1000, 1100, 1200],
                'timestamp': pd.date_range('2023-01-01', periods=3, freq='1min')
            })
            
            # Test column validation
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            is_valid, missing = common_utils['validate_dataframe_columns'](test_df, required_columns)
            assert is_valid, f"DataFrame should have required columns, missing: {missing}"
            
            # Test data quality metrics
            quality_metrics = common_utils['calculate_data_quality_metrics'](test_df)
            assert 'total_rows' in quality_metrics, "Quality metrics should include total_rows"
            assert quality_metrics['total_rows'] == 3, f"Expected 3 rows, got {quality_metrics['total_rows']}"
            
            # Test DataFrame info
            df_info = common_utils['get_dataframe_info'](test_df)
            assert 'shape' in df_info, "DataFrame info should include shape"
            assert df_info['shape'] == (3, 6), f"Expected shape (3, 6), got {df_info['shape']}"
            
            # Test safe operations
            result_df = common_utils['safe_dataframe_operation'](test_df, 'copy')
            assert len(result_df) == len(test_df), "Copied DataFrame should have same length"
            
            self.logger.info("✅ Common utilities integration test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Common utilities integration test failed: {e}")
            return False
    
    def test_math_validation_integration(self) -> bool:
        """Test math validation utility integration."""
        self.logger.info("🧪 Testing math validation integration...")
        
        try:
            math_validation = self.utility_manager.math_validation
            
            # Test safe division
            result = math_validation['safe_divide'](10, 2)
            assert result == 5.0, f"Expected 5.0, got {result}"
            
            # Test division by zero
            result = math_validation['safe_divide'](10, 0)
            assert result == 0.0, f"Expected 0.0 for division by zero, got {result}"
            
            # Test safe logarithm
            result = math_validation['safe_log'](10)
            assert result > 0, f"Expected positive log result, got {result}"
            
            # Test log of zero
            result = math_validation['safe_log'](0)
            assert result == 0.0, f"Expected 0.0 for log of zero, got {result}"
            
            # Test safe square root
            result = math_validation['safe_sqrt'](16)
            assert result == 4.0, f"Expected 4.0, got {result}"
            
            # Test sqrt of negative number
            result = math_validation['safe_sqrt'](-1)
            assert result == 0.0, f"Expected 0.0 for sqrt of negative, got {result}"
            
            # Test validation functions
            finite_result = math_validation['validate_finite'](42.0, "test_value")
            assert finite_result == 42.0, f"Expected 42.0, got {finite_result}"
            
            positive_result = math_validation['validate_positive'](5.0, "test_value")
            assert positive_result == 5.0, f"Expected 5.0, got {positive_result}"
            
            range_result = math_validation['validate_range'](5.0, 0.0, 10.0, "test_value")
            assert range_result == 5.0, f"Expected 5.0, got {range_result}"
            
            self.logger.info("✅ Math validation integration test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Math validation integration test failed: {e}")
            return False
    
    def test_parquet_utils_integration(self) -> bool:
        """Test parquet utilities integration."""
        self.logger.info("🧪 Testing parquet utilities integration...")
        
        try:
            parquet_utils = self.utility_manager.parquet_utils
            
            # Test ParquetUtils class
            parquet_utils_class = parquet_utils['ParquetUtils']
            assert parquet_utils_class is not None, "ParquetUtils class should not be None"
            
            # Test get_parquet_utils function
            get_parquet_utils_func = parquet_utils['get_parquet_utils']
            assert get_parquet_utils_func is not None, "get_parquet_utils function should not be None"
            
            # Test creating ParquetUtils instance
            parquet_utils_instance = get_parquet_utils_func()
            assert parquet_utils_instance is not None, "ParquetUtils instance should not be None"
            
            self.logger.info("✅ Parquet utilities integration test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Parquet utilities integration test failed: {e}")
            return False
    
    def test_serialization_utils_integration(self) -> bool:
        """Test serialization utilities integration."""
        self.logger.info("🧪 Testing serialization utilities integration...")
        
        try:
            serialization_utils = self.utility_manager.serialization_utils
            
            # Test serializer classes
            json_serializer = serialization_utils['JSONSerializer']
            assert json_serializer is not None, "JSONSerializer should not be None"
            
            pickle_serializer = serialization_utils['PickleSerializer']
            assert pickle_serializer is not None, "PickleSerializer should not be None"
            
            parquet_serializer = serialization_utils['ParquetSerializer']
            assert parquet_serializer is not None, "ParquetSerializer should not be None"
            
            universal_serializer = serialization_utils['UniversalSerializer']
            assert universal_serializer is not None, "UniversalSerializer should not be None"
            
            # Test convenience functions
            save_json_func = serialization_utils['save_json']
            assert save_json_func is not None, "save_json function should not be None"
            
            load_json_func = serialization_utils['load_json']
            assert load_json_func is not None, "load_json function should not be None"
            
            save_parquet_func = serialization_utils['save_parquet']
            assert save_parquet_func is not None, "save_parquet function should not be None"
            
            load_parquet_func = serialization_utils['load_parquet']
            assert load_parquet_func is not None, "load_parquet function should not be None"
            
            self.logger.info("✅ Serialization utilities integration test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Serialization utilities integration test failed: {e}")
            return False
    
    def test_data_processing_utils_integration(self) -> bool:
        """Test data processing utilities integration."""
        self.logger.info("🧪 Testing data processing utilities integration...")
        
        try:
            data_processing_utils = self.utility_manager.data_processing_utils
            
            # Test validator classes
            dataframe_validator = data_processing_utils['DataFrameValidator']
            assert dataframe_validator is not None, "DataFrameValidator should not be None"
            
            dataframe_cleaner = data_processing_utils['DataFrameCleaner']
            assert dataframe_cleaner is not None, "DataFrameCleaner should not be None"
            
            dataframe_transformer = data_processing_utils['DataFrameTransformer']
            assert dataframe_transformer is not None, "DataFrameTransformer should not be None"
            
            # Test convenience functions
            validate_dataframe_func = data_processing_utils['validate_dataframe']
            assert validate_dataframe_func is not None, "validate_dataframe function should not be None"
            
            clean_dataframe_func = data_processing_utils['clean_dataframe']
            assert clean_dataframe_func is not None, "clean_dataframe function should not be None"
            
            transform_dataframe_func = data_processing_utils['transform_dataframe']
            assert transform_dataframe_func is not None, "transform_dataframe function should not be None"
            
            get_dataframe_info_func = data_processing_utils['get_dataframe_info']
            assert get_dataframe_info_func is not None, "get_dataframe_info function should not be None"
            
            self.logger.info("✅ Data processing utilities integration test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data processing utilities integration test failed: {e}")
            return False
    
    def test_m1_optimizers_integration(self) -> bool:
        """Test M1 optimizers integration."""
        self.logger.info("🧪 Testing M1 optimizers integration...")
        
        try:
            # Test M1 GPU utils
            m1_gpu_utils = self.utility_manager.m1_gpu_utils
            
            m1_gpu_manager_class = m1_gpu_utils['M1GPUManager']
            assert m1_gpu_manager_class is not None, "M1GPUManager should not be None"
            
            get_m1_gpu_manager_func = m1_gpu_utils['get_m1_gpu_manager']
            assert get_m1_gpu_manager_func is not None, "get_m1_gpu_manager function should not be None"
            
            # Test M1 memory optimizer
            m1_memory_optimizer = self.utility_manager.m1_memory_optimizer
            
            m1_memory_optimizer_class = m1_memory_optimizer['M1MemoryOptimizer']
            assert m1_memory_optimizer_class is not None, "M1MemoryOptimizer should not be None"
            
            get_m1_memory_optimizer_func = m1_memory_optimizer['get_m1_memory_optimizer']
            assert get_m1_memory_optimizer_func is not None, "get_m1_memory_optimizer function should not be None"
            
            # Test M1 CPU optimizer
            m1_cpu_optimizer = self.utility_manager.m1_cpu_optimizer
            
            m1_cpu_optimizer_class = m1_cpu_optimizer['M1CPUOptimizer']
            assert m1_cpu_optimizer_class is not None, "M1CPUOptimizer should not be None"
            
            get_m1_cpu_optimizer_func = m1_cpu_optimizer['get_m1_cpu_optimizer']
            assert get_m1_cpu_optimizer_func is not None, "get_m1_cpu_optimizer function should not be None"
            
            self.logger.info("✅ M1 optimizers integration test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ M1 optimizers integration test failed: {e}")
            return False
    
    def test_enhanced_step02_class(self) -> bool:
        """Test enhanced step02 class initialization."""
        self.logger.info("🧪 Testing enhanced step02 class...")
        
        try:
            config = {
                'max_workers': 2,
                'chunk_size': 1000,
                'min_rows': 100,
                'max_duplicate_ratio': 0.01,
                'max_gap_seconds': 0.5
            }
            
            # Test class initialization
            step = EnhancedDataReadingStep(config)
            assert step is not None, "EnhancedDataReadingStep should not be None"
            assert step.utility_manager is not None, "Utility manager should be initialized"
            assert step.monitor is not None, "Monitor should be initialized"
            
            # Test configuration
            assert step.max_workers == 2, f"Expected max_workers 2, got {step.max_workers}"
            assert step.chunk_size == 1000, f"Expected chunk_size 1000, got {step.chunk_size}"
            assert step.min_rows == 100, f"Expected min_rows 100, got {step.min_rows}"
            
            self.logger.info("✅ Enhanced step02 class test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced step02 class test failed: {e}")
            return False
    
    async def test_enhanced_step02_execution(self) -> bool:
        """Test enhanced step02 execution (without actual data files)."""
        self.logger.info("🧪 Testing enhanced step02 execution...")
        
        try:
            # Test with non-existent data directory (should fail gracefully)
            result = await run_step_enhanced(
                symbol='TEST',
                exchange='TEST',
                timeframe='1m',
                data_dir='/non/existent/path'
            )
            
            # Should fail gracefully
            assert result['success'] == False, "Should fail with non-existent data directory"
            assert 'error' in result, "Error should be present in result"
            
            self.logger.info("✅ Enhanced step02 execution test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced step02 execution test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        self.logger.info("🚀 Starting comprehensive step02 utility integration tests...")
        
        test_methods = [
            ('setup_test_environment', self.setup_test_environment),
            ('dependency_injection', self.test_dependency_injection),
            ('common_operations_integration', self.test_common_operations_integration),
            ('common_utilities_integration', self.test_common_utilities_integration),
            ('math_validation_integration', self.test_math_validation_integration),
            ('parquet_utils_integration', self.test_parquet_utils_integration),
            ('serialization_utils_integration', self.test_serialization_utils_integration),
            ('data_processing_utils_integration', self.test_data_processing_utils_integration),
            ('m1_optimizers_integration', self.test_m1_optimizers_integration),
            ('enhanced_step02_class', self.test_enhanced_step02_class),
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
        
        # Run async test
        try:
            self.logger.info("🧪 Running enhanced_step02_execution...")
            result = asyncio.run(self.test_enhanced_step02_execution())
            results['enhanced_step02_execution'] = result
            if result:
                self.logger.info("✅ enhanced_step02_execution passed")
            else:
                self.logger.error("❌ enhanced_step02_execution failed")
        except Exception as e:
            self.logger.error(f"❌ enhanced_step02_execution failed with exception: {e}")
            results['enhanced_step02_execution'] = False
        
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

async def main():
    """Main test function."""
    logger.info("🚀 Starting Step 2 Enhanced Utility Integration Tests")
    
    tester = Step02EnhancedTester()
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
    exit_code = asyncio.run(main())
    sys.exit(exit_code)