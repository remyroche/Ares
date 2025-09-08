"""
Step04 Utility Integration Examples and Documentation

This module demonstrates comprehensive usage of all utility modules in step04
with proper dependency injection patterns. It serves as both documentation
and examples for developers working with the step04 pipeline.

Author: AI Assistant
Date: 2024
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Import dependency injection for step04 utilities
from .step04_dependency_injection import (
    get_step04_utilities, get_step04_container, create_step04_config,
    get_common_ops, get_common_utils, get_math_validation, get_parquet_utils,
    get_serialization_utils, get_data_processing_utils, get_m1_gpu_utils,
    get_m1_memory_optimizer, get_m1_cpu_optimizer
)

class Step04UtilityIntegrationExamples:
    """
    Comprehensive examples of utility integration in step04 components.
    
    This class demonstrates:
    1. Proper dependency injection setup
    2. Extensive use of all utility modules
    3. Best practices for utility integration
    4. Error handling and validation patterns
    5. Performance optimization techniques
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with comprehensive utility integration."""
        self.config = config
        
        # Initialize dependency injection container
        self.utility_config = create_step04_config(
            enable_common_operations=True,
            enable_common_utilities=True,
            enable_math_validation=True,
            enable_parquet_utils=True,
            enable_serialization_utils=True,
            enable_data_processing_utils=True,
            enable_m1_gpu_utils=True,
            enable_m1_memory_optimizer=True,
            enable_m1_cpu_optimizer=True
        )
        self.container = get_step04_container(self.utility_config)
        self.utils = get_step04_utilities()
        
        # Get logger from utilities
        self.logger = self.utils.get_function('common_operations', 'get_logger')('Step04UtilityIntegrationExamples')
        
        # Initialize all utility components
        self._initialize_utilities()
        
    def _initialize_utilities(self):
        """Initialize all utility components with proper dependency injection."""
        try:
            # Common Operations Utilities
            self.common_ops = get_common_ops()
            self.logger.info('✅ Common Operations utilities initialized')
            
            # Common Utilities
            self.common_utils = get_common_utils()
            self.logger.info('✅ Common Utilities initialized')
            
            # Math Validation Utilities
            self.math_validation = get_math_validation()
            self.logger.info('✅ Math Validation utilities initialized')
            
            # Parquet Utilities
            self.parquet_utils = get_parquet_utils()
            self.logger.info('✅ Parquet utilities initialized')
            
            # Serialization Utilities
            self.serialization_utils = get_serialization_utils()
            self.logger.info('✅ Serialization utilities initialized')
            
            # Data Processing Utilities
            self.data_processing_utils = get_data_processing_utils()
            self.logger.info('✅ Data Processing utilities initialized')
            
            # M1 GPU Utilities
            self.m1_gpu_utils = get_m1_gpu_utils()
            self.logger.info('✅ M1 GPU utilities initialized')
            
            # M1 Memory Optimizer
            self.m1_memory_optimizer = get_m1_memory_optimizer()
            self.logger.info('✅ M1 Memory Optimizer initialized')
            
            # M1 CPU Optimizer
            self.m1_cpu_optimizer = get_m1_cpu_optimizer()
            self.logger.info('✅ M1 CPU Optimizer initialized')
            
        except Exception as e:
            self.logger.error(f'❌ Failed to initialize utilities: {e}')
            raise
    
    def demonstrate_common_operations_usage(self) -> Dict[str, Any]:
        """Demonstrate extensive use of common operations utilities."""
        self.logger.info('🔧 Demonstrating Common Operations utilities...')
        
        # Get utility functions
        safe_float = self.utils.get_function('common_operations', 'safe_float')
        safe_int = self.utils.get_function('common_operations', 'safe_int')
        safe_dict_get = self.utils.get_function('common_operations', 'safe_dict_get')
        ensure_directory = self.utils.get_function('common_operations', 'ensure_directory')
        safe_json_dump = self.utils.get_function('common_operations', 'safe_json_dump')
        safe_json_load = self.utils.get_function('common_operations', 'safe_json_load')
        safe_read_parquet = self.utils.get_function('common_operations', 'safe_read_parquet')
        safe_to_parquet = self.utils.get_function('common_operations', 'safe_to_parquet')
        get_logger = self.utils.get_function('common_operations', 'get_logger')
        format_bytes = self.utils.get_function('common_operations', 'format_bytes')
        chunked_iterable = self.utils.get_function('common_operations', 'chunked_iterable')
        parallel_map = self.utils.get_function('common_operations', 'parallel_map')
        optimize_dataframe_dtypes = self.utils.get_function('common_operations', 'optimize_dataframe_dtypes')
        validate_dataframe_schema = self.utils.get_function('common_operations', 'validate_dataframe_schema')
        validate_data_quality = self.utils.get_function('common_operations', 'validate_data_quality')
        
        results = {}
        
        # Safe type conversions
        results['safe_float'] = {
            'valid_input': safe_float(3.14, 0.0),
            'invalid_input': safe_float('invalid', 0.0),
            'none_input': safe_float(None, 0.0)
        }
        
        results['safe_int'] = {
            'valid_input': safe_int(42, 0),
            'invalid_input': safe_int('invalid', 0),
            'none_input': safe_int(None, 0)
        }
        
        # Safe dictionary access
        test_dict = {'key1': 'value1', 'key2': 42, 'nested': {'inner': 'value'}}
        results['safe_dict_get'] = {
            'existing_key': safe_dict_get(test_dict, 'key1', 'default'),
            'missing_key': safe_dict_get(test_dict, 'missing', 'default'),
            'nested_key': safe_dict_get(test_dict, 'nested.inner', 'default')
        }
        
        # Directory operations
        test_dir = Path('/tmp/step04_test')
        ensure_directory(test_dir)
        results['directory_created'] = test_dir.exists()
        
        # JSON operations
        test_data = {'test': 'data', 'number': 42, 'list': [1, 2, 3]}
        json_file = test_dir / 'test.json'
        safe_json_dump(test_data, json_file)
        loaded_data = safe_json_load(json_file)
        results['json_operations'] = loaded_data == test_data
        
        # DataFrame operations
        test_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'col3': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Optimize DataFrame dtypes
        optimized_df = optimize_dataframe_dtypes(test_df)
        results['dtype_optimization'] = {
            'original_memory': test_df.memory_usage(deep=True).sum(),
            'optimized_memory': optimized_df.memory_usage(deep=True).sum()
        }
        
        # Validate DataFrame schema
        schema_validation = validate_dataframe_schema(test_df, required_columns=['col1', 'col2'])
        results['schema_validation'] = schema_validation
        
        # Validate data quality
        quality_validation = validate_data_quality(test_df)
        results['quality_validation'] = quality_validation
        
        # Memory formatting
        results['memory_formatting'] = {
            'bytes': format_bytes(1024),
            'kilobytes': format_bytes(1024 * 1024),
            'megabytes': format_bytes(1024 * 1024 * 1024)
        }
        
        self.logger.info('✅ Common Operations utilities demonstration completed')
        return results
    
    def demonstrate_math_validation_usage(self) -> Dict[str, Any]:
        """Demonstrate extensive use of math validation utilities."""
        self.logger.info('🧮 Demonstrating Math Validation utilities...')
        
        # Get utility functions
        safe_divide = self.utils.get_function('math_validation', 'safe_divide')
        safe_log = self.utils.get_function('math_validation', 'safe_log')
        safe_sqrt = self.utils.get_function('math_validation', 'safe_sqrt')
        safe_power = self.utils.get_function('math_validation', 'safe_power')
        validate_positive = self.utils.get_function('math_validation', 'validate_positive')
        validate_range = self.utils.get_function('math_validation', 'validate_range')
        validate_finite = self.utils.get_function('math_validation', 'validate_finite')
        safe_kelly_calculation = self.utils.get_function('math_validation', 'safe_kelly_calculation')
        safe_weighted_average = self.utils.get_function('math_validation', 'safe_weighted_average')
        safe_percentage_change = self.utils.get_function('math_validation', 'safe_percentage_change')
        
        results = {}
        
        # Safe mathematical operations
        results['safe_divide'] = {
            'normal_division': safe_divide(10, 2, default=0.0),
            'division_by_zero': safe_divide(10, 0, default=0.0),
            'invalid_inputs': safe_divide('invalid', 'invalid', default=0.0)
        }
        
        results['safe_log'] = {
            'positive_number': safe_log(10, default=0.0),
            'zero': safe_log(0, default=0.0),
            'negative_number': safe_log(-5, default=0.0)
        }
        
        results['safe_sqrt'] = {
            'positive_number': safe_sqrt(16, default=0.0),
            'zero': safe_sqrt(0, default=0.0),
            'negative_number': safe_sqrt(-4, default=0.0)
        }
        
        results['safe_power'] = {
            'normal_power': safe_power(2, 3, default=0.0),
            'fractional_power': safe_power(4, 0.5, default=0.0),
            'negative_base': safe_power(-2, 2, default=0.0)
        }
        
        # Validation functions
        results['validate_positive'] = {
            'positive_number': validate_positive(5.0, "test_positive"),
            'zero': validate_positive(0.0, "test_zero"),
            'negative_number': validate_positive(-3.0, "test_negative")
        }
        
        results['validate_range'] = {
            'in_range': validate_range(5.0, 0.0, 10.0, "test_range"),
            'below_range': validate_range(-1.0, 0.0, 10.0, "test_below"),
            'above_range': validate_range(15.0, 0.0, 10.0, "test_above")
        }
        
        results['validate_finite'] = {
            'finite_number': validate_finite(3.14, "test_finite"),
            'infinity': validate_finite(float('inf'), "test_inf"),
            'nan': validate_finite(float('nan'), "test_nan")
        }
        
        # Financial calculations
        results['safe_kelly_calculation'] = {
            'win_probability': safe_kelly_calculation(0.6, 1.5, default=0.0),
            'edge_case': safe_kelly_calculation(0.5, 1.0, default=0.0),
            'invalid_inputs': safe_kelly_calculation('invalid', 'invalid', default=0.0)
        }
        
        results['safe_weighted_average'] = {
            'normal_calculation': safe_weighted_average([1, 2, 3], [0.3, 0.3, 0.4], default=0.0),
            'empty_arrays': safe_weighted_average([], [], default=0.0),
            'mismatched_lengths': safe_weighted_average([1, 2], [0.5], default=0.0)
        }
        
        results['safe_percentage_change'] = {
            'normal_change': safe_percentage_change(100, 120, default=0.0),
            'zero_initial': safe_percentage_change(0, 100, default=0.0),
            'negative_change': safe_percentage_change(100, 80, default=0.0)
        }
        
        self.logger.info('✅ Math Validation utilities demonstration completed')
        return results
    
    def demonstrate_parquet_utils_usage(self) -> Dict[str, Any]:
        """Demonstrate extensive use of parquet utilities."""
        self.logger.info('📊 Demonstrating Parquet utilities...')
        
        # Get utility functions
        validate_parquet_file = self.utils.get_function('parquet_utils', 'validate_parquet_file')
        safe_read_parquet = self.utils.get_function('parquet_utils', 'safe_read_parquet')
        repair_parquet_file = self.utils.get_function('parquet_utils', 'repair_parquet_file')
        
        results = {}
        
        # Create test DataFrame
        test_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
            'open': np.random.uniform(100, 200, 1000),
            'high': np.random.uniform(100, 200, 1000),
            'low': np.random.uniform(100, 200, 1000),
            'close': np.random.uniform(100, 200, 1000),
            'volume': np.random.uniform(1000, 10000, 1000)
        })
        
        # Test parquet file path
        test_parquet_file = Path('/tmp/step04_test/test.parquet')
        test_parquet_file.parent.mkdir(exist_ok=True)
        
        # Save test DataFrame
        test_df.to_parquet(test_parquet_file)
        
        # Validate parquet file
        validation_result = validate_parquet_file(test_parquet_file)
        results['parquet_validation'] = validation_result
        
        # Safe read parquet
        read_result = safe_read_parquet(test_parquet_file)
        results['safe_read'] = {
            'success': read_result is not None,
            'shape': read_result.shape if read_result is not None else None,
            'columns': list(read_result.columns) if read_result is not None else None
        }
        
        # Test with nrows parameter
        partial_read = safe_read_parquet(test_parquet_file, nrows=100)
        results['partial_read'] = {
            'success': partial_read is not None,
            'shape': partial_read.shape if partial_read is not None else None
        }
        
        self.logger.info('✅ Parquet utilities demonstration completed')
        return results
    
    def demonstrate_serialization_utils_usage(self) -> Dict[str, Any]:
        """Demonstrate extensive use of serialization utilities."""
        self.logger.info('💾 Demonstrating Serialization utilities...')
        
        # Get utility functions
        json_serializer = self.utils.get_function('serialization_utils', 'JSONSerializer')
        pickle_serializer = self.utils.get_function('serialization_utils', 'PickleSerializer')
        parquet_serializer = self.utils.get_function('serialization_utils', 'ParquetSerializer')
        universal_serializer = self.utils.get_function('serialization_utils', 'UniversalSerializer')
        
        results = {}
        
        # Test data
        test_data = {
            'numbers': [1, 2, 3, 4, 5],
            'strings': ['a', 'b', 'c'],
            'nested': {'inner': 'value', 'number': 42}
        }
        
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [1.1, 2.2, 3.3]
        })
        
        # JSON serialization
        json_serializer_instance = json_serializer()
        json_file = Path('/tmp/step04_test/test.json')
        json_serializer_instance.save(test_data, json_file)
        loaded_json = json_serializer_instance.load(json_file)
        results['json_serialization'] = loaded_json == test_data
        
        # Pickle serialization
        pickle_serializer_instance = pickle_serializer()
        pickle_file = Path('/tmp/step04_test/test.pkl')
        pickle_serializer_instance.save(test_data, pickle_file)
        loaded_pickle = pickle_serializer_instance.load(pickle_file)
        results['pickle_serialization'] = loaded_pickle == test_data
        
        # Parquet serialization
        parquet_serializer_instance = parquet_serializer()
        parquet_file = Path('/tmp/step04_test/test_serialization.parquet')
        parquet_serializer_instance.save(test_df, parquet_file)
        loaded_parquet = parquet_serializer_instance.load(parquet_file)
        results['parquet_serialization'] = loaded_parquet.equals(test_df)
        
        # Universal serializer
        universal_serializer_instance = universal_serializer()
        universal_file = Path('/tmp/step04_test/test_universal.json')
        universal_serializer_instance.save(test_data, universal_file)
        loaded_universal = universal_serializer_instance.load(universal_file)
        results['universal_serialization'] = loaded_universal == test_data
        
        self.logger.info('✅ Serialization utilities demonstration completed')
        return results
    
    def demonstrate_data_processing_utils_usage(self) -> Dict[str, Any]:
        """Demonstrate extensive use of data processing utilities."""
        self.logger.info('🔄 Demonstrating Data Processing utilities...')
        
        # Get utility functions
        create_data_quality_report = self.utils.get_function('data_processing_utils', 'create_data_quality_report')
        DataFrameValidator = self.utils.get_function('data_processing_utils', 'DataFrameValidator')
        DataFrameCleaner = self.utils.get_function('data_processing_utils', 'DataFrameCleaner')
        DataFrameTransformer = self.utils.get_function('data_processing_utils', 'DataFrameTransformer')
        
        results = {}
        
        # Create test DataFrame with various data quality issues
        test_df = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': [1.1, 2.2, 3.3, None, 5.5],
            'col3': ['a', 'b', 'c', 'd', 'e'],
            'col4': [1, 1, 1, 1, 1],  # Constant column
            'col5': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 100  # Large column
        })
        
        # Data quality report
        quality_report = create_data_quality_report(test_df)
        results['data_quality_report'] = quality_report
        
        # DataFrame validator
        validator = DataFrameValidator()
        validation_result = validator.validate(test_df)
        results['dataframe_validation'] = validation_result
        
        # DataFrame cleaner
        cleaner = DataFrameCleaner()
        cleaned_df = cleaner.clean(test_df)
        results['dataframe_cleaning'] = {
            'original_shape': test_df.shape,
            'cleaned_shape': cleaned_df.shape,
            'nulls_removed': test_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum()
        }
        
        # DataFrame transformer
        transformer = DataFrameTransformer()
        transformed_df = transformer.transform(test_df, operations=['normalize', 'standardize'])
        results['dataframe_transformation'] = {
            'original_shape': test_df.shape,
            'transformed_shape': transformed_df.shape
        }
        
        self.logger.info('✅ Data Processing utilities demonstration completed')
        return results
    
    def demonstrate_m1_optimization_usage(self) -> Dict[str, Any]:
        """Demonstrate extensive use of M1 optimization utilities."""
        self.logger.info('🍎 Demonstrating M1 Optimization utilities...')
        
        # Get utility functions
        M1GPUManager = self.utils.get_function('m1_gpu_utils', 'M1GPUManager')
        M1MemoryOptimizer = self.utils.get_function('m1_memory_optimizer', 'M1MemoryOptimizer')
        M1CPUOptimizer = self.utils.get_function('m1_cpu_optimizer', 'M1CPUOptimizer')
        
        results = {}
        
        # M1 GPU Manager
        try:
            gpu_manager = M1GPUManager()
            results['gpu_manager'] = {
                'device_available': gpu_manager.device_available,
                'should_use_gpu': gpu_manager.should_use_gpu(),
                'memory_info': gpu_manager.get_memory_info()
            }
        except Exception as e:
            results['gpu_manager'] = {'error': str(e)}
        
        # M1 Memory Optimizer
        try:
            memory_optimizer = M1MemoryOptimizer(
                memory_limit_gb=8.0,
                enable_gc_tuning=True,
                enable_memory_leak_detection=True
            )
            memory_report = memory_optimizer.get_memory_report()
            results['memory_optimizer'] = {
                'memory_report': memory_report,
                'should_chunk_data': memory_optimizer.should_chunk_data(1000000),
                'optimal_chunk_size': memory_optimizer.calculate_optimal_chunk_size(1000000)
            }
        except Exception as e:
            results['memory_optimizer'] = {'error': str(e)}
        
        # M1 CPU Optimizer
        try:
            cpu_optimizer = M1CPUOptimizer(
                max_workers=4,
                enable_hyperthreading=True
            )
            results['cpu_optimizer'] = {
                'optimal_workers': cpu_optimizer.calculate_optimal_workers(),
                'cpu_info': cpu_optimizer.get_cpu_info()
            }
        except Exception as e:
            results['cpu_optimizer'] = {'error': str(e)}
        
        self.logger.info('✅ M1 Optimization utilities demonstration completed')
        return results
    
    def run_comprehensive_demonstration(self) -> Dict[str, Any]:
        """Run comprehensive demonstration of all utility integrations."""
        self.logger.info('🚀 Starting comprehensive utility integration demonstration...')
        
        results = {
            'common_operations': self.demonstrate_common_operations_usage(),
            'math_validation': self.demonstrate_math_validation_usage(),
            'parquet_utils': self.demonstrate_parquet_utils_usage(),
            'serialization_utils': self.demonstrate_serialization_utils_usage(),
            'data_processing_utils': self.demonstrate_data_processing_utils_usage(),
            'm1_optimization': self.demonstrate_m1_optimization_usage()
        }
        
        self.logger.info('✅ Comprehensive utility integration demonstration completed')
        return results


def create_step04_utility_integration_guide():
    """
    Create a comprehensive guide for step04 utility integration.
    
    Returns:
        Dict containing integration patterns and best practices
    """
    guide = {
        'dependency_injection_setup': {
            'description': 'How to set up dependency injection for step04 utilities',
            'code_example': '''
# Initialize dependency injection container
utility_config = create_step04_config(
    enable_common_operations=True,
    enable_common_utilities=True,
    enable_math_validation=True,
    enable_parquet_utils=True,
    enable_serialization_utils=True,
    enable_data_processing_utils=True,
    enable_m1_gpu_utils=True,
    enable_m1_memory_optimizer=True,
    enable_m1_cpu_optimizer=True
)
container = get_step04_container(utility_config)
utils = get_step04_utilities()
            ''',
            'best_practices': [
                'Always initialize all utilities at class initialization',
                'Use utility functions through the utils object for consistency',
                'Handle utility initialization errors gracefully',
                'Log utility initialization status for debugging'
            ]
        },
        
        'common_operations_integration': {
            'description': 'Integration patterns for common operations utilities',
            'code_example': '''
# Get utility functions
safe_float = self.utils.get_function('common_operations', 'safe_float')
safe_int = self.utils.get_function('common_operations', 'safe_int')
safe_dict_get = self.utils.get_function('common_operations', 'safe_dict_get')
ensure_directory = self.utils.get_function('common_operations', 'ensure_directory')

# Use in your code
value = safe_float(config.get('threshold', 0.0), 0.0)
directory = ensure_directory(Path('/path/to/directory'))
            ''',
            'best_practices': [
                'Always use safe_float and safe_int for configuration values',
                'Use safe_dict_get for nested dictionary access',
                'Use ensure_directory before file operations',
                'Validate DataFrame schemas before processing'
            ]
        },
        
        'math_validation_integration': {
            'description': 'Integration patterns for math validation utilities',
            'code_example': '''
# Get utility functions
safe_divide = self.utils.get_function('math_validation', 'safe_divide')
validate_positive = self.utils.get_function('math_validation', 'validate_positive')
validate_range = self.utils.get_function('math_validation', 'validate_range')

# Use in calculations
result = safe_divide(numerator, denominator, default=0.0)
validated_value = validate_positive(value, "parameter_name")
ranged_value = validate_range(value, 0.0, 1.0, "parameter_name")
            ''',
            'best_practices': [
                'Always use safe_divide to prevent division by zero',
                'Validate all numeric inputs with validate_positive',
                'Use validate_range for bounded parameters',
                'Use safe_kelly_calculation for financial metrics'
            ]
        },
        
        'parquet_utils_integration': {
            'description': 'Integration patterns for parquet utilities',
            'code_example': '''
# Get utility functions
validate_parquet_file = self.utils.get_function('parquet_utils', 'validate_parquet_file')
safe_read_parquet = self.utils.get_function('parquet_utils', 'safe_read_parquet')

# Use in data loading
validation_result = validate_parquet_file(file_path)
if validation_result['is_valid']:
    data = safe_read_parquet(file_path, nrows=1000)
            ''',
            'best_practices': [
                'Always validate parquet files before reading',
                'Use safe_read_parquet with error handling',
                'Use nrows parameter for large files',
                'Handle parquet repair when needed'
            ]
        },
        
        'serialization_utils_integration': {
            'description': 'Integration patterns for serialization utilities',
            'code_example': '''
# Get utility functions
JSONSerializer = self.utils.get_function('serialization_utils', 'JSONSerializer')
UniversalSerializer = self.utils.get_function('serialization_utils', 'UniversalSerializer')

# Use for data persistence
json_serializer = JSONSerializer()
json_serializer.save(data, file_path)
loaded_data = json_serializer.load(file_path)
            ''',
            'best_practices': [
                'Use appropriate serializer for data type',
                'Use UniversalSerializer for auto-detection',
                'Handle serialization errors gracefully',
                'Use compression for large datasets'
            ]
        },
        
        'data_processing_utils_integration': {
            'description': 'Integration patterns for data processing utilities',
            'code_example': '''
# Get utility functions
create_data_quality_report = self.utils.get_function('data_processing_utils', 'create_data_quality_report')
DataFrameValidator = self.utils.get_function('data_processing_utils', 'DataFrameValidator')

# Use for data validation
quality_report = create_data_quality_report(dataframe)
validator = DataFrameValidator()
validation_result = validator.validate(dataframe)
            ''',
            'best_practices': [
                'Always create data quality reports',
                'Use DataFrameValidator for comprehensive validation',
                'Clean data before processing',
                'Transform data with appropriate operations'
            ]
        },
        
        'm1_optimization_integration': {
            'description': 'Integration patterns for M1 optimization utilities',
            'code_example': '''
# Get utility functions
M1MemoryOptimizer = self.utils.get_function('m1_memory_optimizer', 'M1MemoryOptimizer')
M1CPUOptimizer = self.utils.get_function('m1_cpu_optimizer', 'M1CPUOptimizer')

# Use for performance optimization
memory_optimizer = M1MemoryOptimizer(memory_limit_gb=8.0)
with memory_optimizer.memory_checkpoint("operation_name"):
    # Your memory-intensive operation here
    pass

cpu_optimizer = M1CPUOptimizer(max_workers=4)
results = cpu_optimizer.parallel_process(data, process_function)
            ''',
            'best_practices': [
                'Use memory checkpoints for large operations',
                'Use parallel processing for CPU-bound tasks',
                'Monitor memory usage with M1MemoryOptimizer',
                'Optimize batch sizes with M1CPUOptimizer'
            ]
        }
    }
    
    return guide


if __name__ == "__main__":
    # Example usage
    config = {
        'memory_limit_gb': 8.0,
        'max_parallel_workers': 4,
        'enable_gpu': True
    }
    
    # Create examples instance
    examples = Step04UtilityIntegrationExamples(config)
    
    # Run comprehensive demonstration
    results = examples.run_comprehensive_demonstration()
    
    # Print results
    print("Step04 Utility Integration Demonstration Results:")
    for category, result in results.items():
        print(f"\n{category.upper()}:")
        print(f"  {result}")
    
    # Create integration guide
    guide = create_step04_utility_integration_guide()
    print("\nIntegration Guide created successfully!")