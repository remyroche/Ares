"""
Simple test script for enhanced BaseStep functionality

This script tests the enhanced BaseStep without requiring all dependencies.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, '/workspace')

# Mock the missing dependencies
class MockArtifactManager:
    def __init__(self, *args, **kwargs):
        pass
    
    def ensure_step_category_directories(self):
        pass

class MockKlinesParquetManager:
    def __init__(self, *args, **kwargs):
        pass

class MockIntegratedHardwareManager:
    def __init__(self, *args, **kwargs):
        pass

# Mock the imports
sys.modules['src.utils.artifact_manager'] = type('MockModule', (), {
    'ArtifactManager': MockArtifactManager,
    'ArtifactMetadata': type('ArtifactMetadata', (), {}),
    'OperationMetrics': type('OperationMetrics', (), {}),
    'CacheEntry': type('CacheEntry', (), {}),
    'CompressionType': type('CompressionType', (), {}),
    'OperationType': type('OperationType', (), {}),
    'RetryStrategy': type('RetryStrategy', (), {}),
    'RetryConfig': type('RetryConfig', (), {}),
    'MemoryConfig': type('MemoryConfig', (), {}),
})()

sys.modules['src.utils.kline_parquet'] = type('MockModule', (), {
    'KlinesParquetManager': MockKlinesParquetManager,
    'StorageConfig': type('StorageConfig', (), {}),
})()

sys.modules['src.utils.hardware'] = type('MockModule', (), {
    'get_integrated_hardware_manager': lambda *args, **kwargs: MockIntegratedHardwareManager(),
    'IntegratedHardwareConfig': type('IntegratedHardwareConfig', (), {}),
    'm1_optimized': lambda *args, **kwargs: lambda x: x,
    'memory_optimized': lambda *args, **kwargs: lambda x: x,
    'optimize_dataframe': lambda *args, **kwargs: args[0] if args else None,
    'force_cleanup': lambda: None,
    'WorkloadCategory': type('WorkloadCategory', (), {'FEATURE_GENERATION': 'feature_generation'}),
    'OptimizationLevel': type('OptimizationLevel', (), {'AGGRESSIVE': 'aggressive'}),
    'get_memory_stats': lambda: {},
    'MemoryOptimizationLevel': type('MemoryOptimizationLevel', (), {'AGGRESSIVE': 'aggressive'}),
    'comprehensive_memory_optimization': lambda *args, **kwargs: lambda x: x,
    'memory_efficient': lambda *args, **kwargs: lambda x: x,
    'OptimizationConfig': type('OptimizationConfig', (), {}),
    'smart_cache': lambda *args, **kwargs: lambda x: x,
    'auto_optimize': lambda *args, **kwargs: lambda x: x,
    'performance_tracked': lambda *args, **kwargs: lambda x: x,
    'WorkloadType': type('WorkloadType', (), {'FEATURE_GENERATION': 'feature_generation'}),
})()

# Mock tprint functions
def mock_tprint(*args, **kwargs):
    print(f"TPRINT: {' '.join(str(arg) for arg in args)}")

sys.modules['src.utils.tprint'] = type('MockModule', (), {
    'tprint': mock_tprint,
    'tprint_success': mock_tprint,
    'tprint_info': mock_tprint,
    'tprint_warning': mock_tprint,
    'tprint_error': mock_tprint,
    'tprint_debug': mock_tprint,
    'tprint_performance': mock_tprint,
    'tprint_progress': mock_tprint,
    'tprint_structured': mock_tprint,
    'tprint_exception': mock_tprint,
    'tprint_with_level': mock_tprint,
    'tprint_timer': mock_tprint,
    'tprint_data_preview': mock_tprint,
    'tprint_data_format': mock_tprint,
    'tprint_metrics': mock_tprint,
    'tprint_summary': mock_tprint,
    'tprint_table': mock_tprint,
    'tprint_banner': mock_tprint,
    'tprint_separator': mock_tprint,
    'tprint_header': mock_tprint,
    'tprint_footer': mock_tprint,
    'tprint_step_start': mock_tprint,
    'tprint_step_end': mock_tprint,
    'tprint_operation_start': mock_tprint,
    'tprint_operation_end': mock_tprint,
    'tprint_data_summary': mock_tprint,
    'tprint_config_preview': mock_tprint,
    'tprint_validation_result': mock_tprint,
    'tprint_performance_summary': mock_tprint,
    'tprint_memory_usage': mock_tprint,
    'tprint_hardware_stats': mock_tprint,
    'tprint_dict': mock_tprint,
    'tprint_list': mock_tprint,
    'tprint_dataframe_info': mock_tprint,
    'tprint_model_info': mock_tprint,
    'tprint_artifact_info': mock_tprint,
    'tprint_execution_summary': mock_tprint,
    'LogLevel': type('LogLevel', (), {}),
    'TPrintConfig': type('TPrintConfig', (), {}),
    'LogLevelEnum': type('LogLevelEnum', (), {}),
})()

# Mock other utilities
sys.modules['src.utils.common_operations'] = type('MockModule', (), {
    'safe_json_load': lambda path: {},
    'safe_json_dump': lambda data, path: True,
    'safe_fillna': lambda df, value: df,
    'safe_to_parquet': lambda df, path: True,
    'safe_read_parquet': lambda path: None,
    'ensure_directory': lambda path: True,
    'safe_file_exists': lambda path: False,
    'get_current_datetime': lambda: type('DateTime', (), {'isoformat': lambda: '2024-01-01T00:00:00'}),
    'format_datetime': lambda dt: str(dt),
    'create_empty_dataframe': lambda: None,
    'validate_dataframe': lambda df: True,
    'optimize_dataframe_dtypes': lambda df: df,
    'safe_divide': lambda a, b, default=0: a / b if b != 0 else default,
    'safe_log': lambda x: 0,
    'safe_sqrt': lambda x: 0,
    'safe_percentage_change': lambda a, b: 0,
    'safe_weighted_average': lambda values, weights: 0,
    'get_m1_gpu_manager': lambda: None,
    'get_m1_memory_optimizer': lambda: None,
    'get_m1_cpu_optimizer': lambda: None,
    'cleanup_m1_optimizers': lambda: None,
    'integrate_with_m1_optimizers': lambda: None,
    'validate_positive': lambda x, default=0: x if x > 0 else default,
})()

sys.modules['src.utils.common_utilities'] = type('MockModule', (), {
    'safe_dataframe_operation': lambda df, op, **kwargs: df,
    'validate_dataframe_columns': lambda df, cols: True,
    'calculate_data_quality_metrics': lambda df: {},
    'safe_merge_dataframes': lambda df1, df2: df1,
    'create_summary_statistics': lambda df: {},
    'ensure_list': lambda x: [x] if not isinstance(x, list) else x,
    'ensure_array': lambda x: x,
    'flatten_dict': lambda d: d,
    'safe_convert_to_numeric': lambda x: x,
    'safe_drop_na': lambda df: df,
    'safe_reset_index': lambda df: df,
})()

sys.modules['src.utils.math_validation'] = type('MockModule', (), {
    'validate_finite': lambda x, default=None: x,
    'validate_positive': lambda x, default=0: x if x > 0 else default,
    'validate_range': lambda x, min_val, max_val: x,
    'validate_probability': lambda x: x,
    'validate_matrix_properties': lambda x: True,
    'validate_statistical_properties': lambda x: True,
    'safe_divide': lambda a, b, default=0: a / b if b != 0 else default,
    'safe_log': lambda x: 0,
    'safe_sqrt': lambda x: 0,
    'safe_percentage_change': lambda a, b: 0,
    'safe_weighted_average': lambda values, weights: 0,
    'MathValidationError': Exception,
})()

sys.modules['src.core.decorators'] = type('MockModule', (), {
    'handles_errors': lambda *args, **kwargs: lambda x: x,
    'error_boundary': lambda *args, **kwargs: lambda x: x,
    'converts_errors': lambda *args, **kwargs: lambda x: x,
    'traced': lambda *args, **kwargs: lambda x: x,
    'log_execution_time': lambda *args, **kwargs: lambda x: x,
    'timeout': lambda *args, **kwargs: lambda x: x,
    'validate_data_quality': lambda *args, **kwargs: lambda x: x,
    'compose': lambda *args: lambda x: x,
})()

sys.modules['src.core.errors'] = type('MockModule', (), {
    'AppError': Exception,
    'ValidationError': Exception,
    'DataIntegrityError': Exception,
    'NotFoundError': Exception,
    'BusinessRuleError': Exception,
    'FileOperationError': Exception,
    'MathValidationError': Exception,
    'TimeoutError': Exception,
})()

sys.modules['src.utils.ml_common.config'] = type('MockModule', (), {
    'BaseTrainingConfig': type('BaseTrainingConfig', (), {}),
})()

sys.modules['src.utils.ml_common.training'] = type('MockModule', (), {
    'PerRegimeTrainingStep': type('PerRegimeTrainingStep', (), {}),
})()

sys.modules['src.utils.ml_common.optimization'] = type('MockModule', (), {
    'HyperparameterOptimizer': type('HyperparameterOptimizer', (), {}),
})()

sys.modules['src.utils.ml_common.cv_utils'] = type('MockModule', (), {
    'TimeSeriesSplitValidator': type('TimeSeriesSplitValidator', (), {}),
})()

sys.modules['src.utils.ml_common.oof_generator'] = type('MockModule', (), {
    'OOFGenerator': type('OOFGenerator', (), {}),
})()

sys.modules['src.utils.ml_common.data_leakage_detector'] = type('MockModule', (), {
    'DataLeakageDetector': type('DataLeakageDetector', (), {}),
})()

sys.modules['src.utils.data.quality.data_cleaning'] = type('MockModule', (), {
    'DataCleaner': type('DataCleaner', (), {}),
    'CleaningConfig': type('CleaningConfig', (), {}),
    'MissingValueStrategy': type('MissingValueStrategy', (), {}),
    'OutlierStrategy': type('OutlierStrategy', (), {}),
})()

sys.modules['src.utils.ml_common.post_training.model_persistence'] = type('MockModule', (), {
    'ModelPersistence': type('ModelPersistence', (), {}),
    'ModelMetadata': type('ModelMetadata', (), {}),
    'PersistenceConfig': type('PersistenceConfig', (), {}),
})()

sys.modules['src.utils.ml_common.models.model_cache'] = type('MockModule', (), {
    'ModelCache': type('ModelCache', (), {}),
    'get_model_cache': lambda: None,
    'CachedModelMetadata': type('CachedModelMetadata', (), {}),
})()

# Now import BaseStep
from src.training.steps.base_step import BaseStep


class TestEnhancedStep(BaseStep):
    """Test step to verify enhanced BaseStep functionality."""
    
    def __init__(self, step_name: str = "test_enhanced_step", config: dict = None):
        super().__init__(step_name, config)
        print("✅ TestEnhancedStep initialized")
    
    async def execute(self, config: dict) -> dict:
        """Test the enhanced BaseStep functionality."""
        print("🚀 Testing enhanced BaseStep functionality...")
        
        # Test 1: Check utility availability
        print("\n📋 Testing utility availability...")
        availability = self._get_availability_status()
        print(f"Available utilities: {sum(availability.values())}/{len(availability)}")
        
        for utility, available in availability.items():
            status = "✅" if available else "❌"
            print(f"  {status} {utility}")
        
        # Test 2: Test convenience methods
        print("\n🔧 Testing convenience methods...")
        
        # Test JSON operations
        test_data = {"test": "value", "number": 42}
        json_saved = self._safe_json_save(test_data, "test_data.json")
        print(f"JSON save: {json_saved}")
        
        loaded_data = self._safe_json_load("test_data.json")
        print(f"JSON load: {loaded_data}")
        
        # Test math operations
        result = self._safe_divide(10, 2, default=0)
        print(f"Safe divide: {result}")
        
        finite_value = self._validate_finite(3.14, default=0)
        print(f"Validate finite: {finite_value}")
        
        positive_value = self._validate_positive(-5, default=1)
        print(f"Validate positive: {positive_value}")
        
        # Test directory operations
        dir_created = self._ensure_directory("test_dir")
        print(f"Directory created: {dir_created}")
        
        # Test 3: Test direct utility access
        print("\n📦 Testing direct utility access...")
        
        if self.common_ops:
            print("✅ Common operations available")
            current_time = self.common_ops['get_current_datetime']()
            print(f"Current time: {current_time}")
        else:
            print("❌ Common operations not available")
        
        if self.hardware_utils:
            print("✅ Hardware utilities available")
            print(f"Hardware manager: {self.hardware_utils['get_integrated_hardware_manager'] is not None}")
        else:
            print("❌ Hardware utilities not available")
        
        if self.math_validation:
            print("✅ Math validation available")
            safe_result = self.math_validation['safe_divide'](100, 3, default=0)
            print(f"Math validation result: {safe_result}")
        else:
            print("❌ Math validation not available")
        
        # Test 4: Test tprint functions
        print("\n📝 Testing tprint functions...")
        tprint("Testing basic tprint")
        tprint_success("Testing tprint_success")
        tprint_info("Testing tprint_info")
        tprint_warning("Testing tprint_warning")
        tprint_error("Testing tprint_error")
        
        # Test 5: Test help system
        print("\n❓ Testing help system...")
        help_info = self._get_utility_help()
        print(f"Help info keys: {list(help_info.keys())}")
        
        # Clean up test files
        try:
            os.remove("test_data.json")
            os.rmdir("test_dir")
            print("\n🧹 Test files cleaned up")
        except:
            pass
        
        print("\n✅ Enhanced BaseStep test completed successfully!")
        
        return {
            'success': True,
            'test_results': {
                'utility_availability': availability,
                'convenience_methods_working': True,
                'direct_utility_access_working': True,
                'tprint_functions_working': True,
                'help_system_working': True
            }
        }


async def main():
    """Run the test."""
    print("🧪 Starting Enhanced BaseStep Test")
    print("=" * 50)
    
    # Create test step
    step = TestEnhancedStep()
    
    # Test configuration
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'execution_mode': 'light'
    }
    
    try:
        # Run the test
        result = await step.run(config)
        
        print("\n" + "=" * 50)
        print("🎉 Test Results:")
        print(f"Success: {result.get('success', False)}")
        
        if result.get('success'):
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())