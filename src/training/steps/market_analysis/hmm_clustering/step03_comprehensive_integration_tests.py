"""
Comprehensive Integration Tests for Step03 Utilities

This module provides comprehensive tests to verify that all specified utilities
are extensively used and working correctly in the Step03 pipeline.
"""

import asyncio
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import pytest
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import dependency injection and utilities
from .step03_dependency_injection import (
    Step03ServiceProvider, Step03Config, Step03UtilityMixin,
    get_step03_service_provider, inject_step03_utilities
)

# Import the enhanced step03 implementation
from .step03_enhanced_with_utilities import EnhancedHMMClusteringStep

class TestStep03UtilityIntegration:
    """Comprehensive test suite for Step03 utility integration."""
    
    @pytest.fixture
    def service_provider(self):
        """Create a test service provider."""
        config = Step03Config(
            enable_gpu_optimization=True,
            enable_memory_optimization=True,
            enable_cpu_optimization=True,
            enable_math_validation=True,
            enable_data_validation=True,
            enable_serialization=True,
            enable_parquet_operations=True,
            max_memory_usage_gb=4.0,
            max_workers=2,
            enable_extensive_logging=False  # Disable for tests
        )
        return get_step03_service_provider(config)
    
    @pytest.fixture
    def test_data(self):
        """Create test data for validation."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1min'),
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 105,
            'low': np.random.randn(1000).cumsum() + 95,
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 1000),
            'regime': np.random.choice(['bull', 'bear', 'sideways'], 1000)
        })
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_dependency_injection_initialization(self, service_provider):
        """Test that dependency injection container initializes correctly."""
        assert service_provider is not None
        assert service_provider._initialized is True
        
        # Test that all services are available
        utils = service_provider.get_all_utilities()
        assert 'common_operations' in utils
        assert 'common_utilities' in utils
        assert 'math_validation' in utils
        assert 'serialization' in utils
        assert 'm1_optimizers' in utils
        assert 'data_processing' in utils
        assert 'parquet' in utils

    def test_common_operations_integration(self, service_provider):
        """Test that common_operations.py utilities are extensively used."""
        utils = service_provider.get_common_operations()
        
        # Test datetime operations
        current_time = utils['datetime']['get_current_datetime']()
        assert current_time is not None
        
        formatted_time = utils['datetime']['format_datetime'](current_time)
        assert isinstance(formatted_time, str)
        
        # Test dataframe operations
        test_df = utils['dataframe']['create_empty_dataframe'](['col1', 'col2'])
        assert isinstance(test_df, pd.DataFrame)
        assert list(test_df.columns) == ['col1', 'col2']
        
        # Test file operations
        temp_file = Path('/tmp/test_file.json')
        test_data = {'test': 'data'}
        utils['file_operations']['safe_json_dump'](test_data, temp_file)
        assert temp_file.exists()
        
        loaded_data = utils['file_operations']['safe_json_load'](temp_file)
        assert loaded_data == test_data
        
        # Clean up
        temp_file.unlink()

    def test_common_utilities_integration(self, service_provider, test_data):
        """Test that common_utilities.py utilities are extensively used."""
        utils = service_provider.get_common_utilities()
        
        # Test dataframe operations
        result = utils['dataframe_operations']['safe_dataframe_operation'](
            test_data, 'dropna'
        )
        assert isinstance(result, pd.DataFrame)
        
        # Test data quality
        quality_metrics = utils['data_quality']['calculate_data_quality_metrics'](test_data)
        assert isinstance(quality_metrics, dict)
        assert 'total_rows' in quality_metrics
        assert 'total_columns' in quality_metrics
        
        # Test data quality report
        quality_report = utils['data_quality']['create_data_quality_report'](test_data)
        assert isinstance(quality_report, dict)
        assert 'status' in quality_report

    def test_math_validation_integration(self, service_provider):
        """Test that math_validation.py utilities are extensively used."""
        utils = service_provider.get_math_validation()
        
        # Test basic math operations
        result = utils['basic_math']['safe_divide'](10, 2)
        assert result == 5.0
        
        result = utils['basic_math']['safe_divide'](10, 0)
        assert result == 0.0  # Default value
        
        # Test validation
        finite_val = utils['validation']['validate_finite'](5.0)
        assert finite_val == 5.0
        
        with pytest.raises(Exception):  # Should raise for non-finite
            utils['validation']['validate_finite'](float('inf'))
        
        # Test financial math
        kelly_result = utils['financial_math']['safe_kelly_calculation'](0.6, 100, 50)
        assert isinstance(kelly_result, float)
        assert 0 <= kelly_result <= 1

    def test_serialization_utils_integration(self, service_provider, temp_dir):
        """Test that serialization_utils.py utilities are extensively used."""
        utils = service_provider.get_serialization_utils()
        
        # Test JSON serialization
        test_data = {'test': 'data', 'number': 42}
        json_file = Path(temp_dir) / 'test.json'
        
        utils['convenience_functions']['save_json'](test_data, json_file)
        assert json_file.exists()
        
        loaded_data = utils['convenience_functions']['load_json'](json_file)
        assert loaded_data == test_data
        
        # Test universal serializer
        universal_serializer = utils['serializers']['UniversalSerializer']
        universal_file = Path(temp_dir) / 'test_universal.json'
        
        universal_serializer.save(test_data, universal_file)
        assert universal_file.exists()
        
        loaded_universal = universal_serializer.load(universal_file)
        assert loaded_universal == test_data

    def test_parquet_utils_integration(self, service_provider, test_data, temp_dir):
        """Test that parquet_utils.py utilities are extensively used."""
        utils = service_provider.get_parquet_utils()
        
        parquet_handler = utils['ParquetUtils']
        
        # Test parquet file operations
        parquet_file = Path(temp_dir) / 'test.parquet'
        
        # Save test data
        test_data.to_parquet(parquet_file, index=False)
        
        # Validate parquet file
        validation_result = parquet_handler.validate_parquet_file(str(parquet_file))
        assert validation_result['valid'] is True
        assert validation_result['file_exists'] is True
        
        # Read parquet file
        loaded_data = parquet_handler.safe_read_parquet(str(parquet_file))
        assert loaded_data is not None
        assert len(loaded_data) == len(test_data)

    def test_data_processing_utils_integration(self, service_provider, test_data):
        """Test that data_processing_utils.py utilities are extensively used."""
        utils = service_provider.get_data_processing_utils()
        
        # Test DataFrame validator
        validator = utils['validators']['DataFrameValidator']
        validation_result = validator.validate_dataframe(test_data)
        assert isinstance(validation_result, utils['data_structures']['DataQualityReport'])
        assert validation_result.summary['total_rows'] == len(test_data)
        
        # Test DataFrame cleaner
        cleaner = utils['cleaners']['DataFrameCleaner']
        cleaned_data = cleaner.clean_dataframe(test_data)
        assert isinstance(cleaned_data, pd.DataFrame)
        
        # Test DataFrame transformer
        transformer = utils['transformers']['DataFrameTransformer']
        transformations = [{'type': 'rename_columns', 'params': {'mapping': {'open': 'price_open'}}}]
        transformed_data = transformer.transform_dataframe(test_data, transformations)
        assert 'price_open' in transformed_data.columns

    def test_m1_optimizers_integration(self, service_provider):
        """Test that M1 optimization utilities are extensively used."""
        utils = service_provider.get_m1_optimizers()
        
        # Test GPU manager
        gpu_manager = utils['gpu']['M1GPUManager']
        assert gpu_manager is not None
        assert gpu_manager.device is not None
        
        should_use_gpu = gpu_manager.should_use_gpu(1000, "general")
        assert isinstance(should_use_gpu, bool)
        
        # Test memory optimizer
        memory_optimizer = utils['memory']['M1MemoryOptimizer']
        memory_usage = memory_optimizer.get_memory_usage()
        assert isinstance(memory_usage, dict)
        assert 'rss_gb' in memory_usage
        
        # Test CPU optimizer
        cpu_optimizer = utils['cpu']['M1CPUOptimizer']
        cpu_usage = cpu_optimizer.get_cpu_usage_report()
        assert isinstance(cpu_usage, dict)

    def test_enhanced_hmm_clustering_step_initialization(self, service_provider):
        """Test that EnhancedHMMClusteringStep initializes with all utilities."""
        config = {
            'SYMBOL': 'ETHUSDT',
            'EXCHANGE': 'BINANCE',
            'TIMEFRAME': '1m',
            'DATA_DIR': '/tmp'
        }
        
        step = EnhancedHMMClusteringStep(config)
        
        # Verify all utilities are available
        assert step.utils is not None
        assert step.common_ops is not None
        assert step.common_utils is not None
        assert step.math_validation is not None
        assert step.serialization is not None
        assert step.m1_optimizers is not None
        assert step.data_processing is not None
        assert step.parquet_utils is not None
        
        # Verify M1 optimizers are initialized
        assert step.gpu_manager is not None
        assert step.memory_optimizer is not None
        assert step.cpu_optimizer is not None
        
        # Verify data processing utilities are initialized
        assert step.df_validator is not None
        assert step.df_cleaner is not None
        assert step.df_transformer is not None
        
        # Verify parquet utilities are initialized
        assert step.parquet_handler is not None

    @pytest.mark.asyncio
    async def test_utility_health_validation(self, service_provider):
        """Test that utility health validation works correctly."""
        config = {
            'SYMBOL': 'ETHUSDT',
            'EXCHANGE': 'BINANCE',
            'TIMEFRAME': '1m',
            'DATA_DIR': '/tmp'
        }
        
        step = EnhancedHMMClusteringStep(config)
        health_status = step._validate_utility_health()
        
        assert isinstance(health_status, dict)
        assert 'all_healthy' in health_status
        assert 'issues' in health_status
        assert 'utility_status' in health_status
        
        # Most utilities should be healthy in test environment
        assert health_status['all_healthy'] is True or len(health_status['issues']) == 0

    def test_dependency_injection_decorator(self, service_provider):
        """Test that the dependency injection decorator works correctly."""
        
        @inject_step03_utilities
        def test_function(utils=None, services=None, **kwargs):
            assert utils is not None
            assert services is not None
            assert 'common_operations' in utils
            return True
        
        result = test_function(test_param='value')
        assert result is True

    def test_step03_utility_mixin(self, service_provider):
        """Test that Step03UtilityMixin provides all utilities."""
        
        class TestClass(Step03UtilityMixin):
            def __init__(self):
                super().__init__()
        
        test_instance = TestClass()
        
        # Verify all utility categories are available
        assert test_instance.get_common_ops() is not None
        assert test_instance.get_common_utils() is not None
        assert test_instance.get_math_validation() is not None
        assert test_instance.get_serialization() is not None
        assert test_instance.get_m1_optimizers() is not None
        assert test_instance.get_data_processing() is not None
        assert test_instance.get_parquet_utils() is not None

    def test_comprehensive_utility_usage(self, service_provider, test_data, temp_dir):
        """Test comprehensive usage of all utilities in a realistic scenario."""
        utils = service_provider.get_all_utilities()
        
        # Simulate a realistic data processing pipeline
        
        # 1. Use common operations for file handling
        output_dir = Path(temp_dir) / 'output'
        utils['common_operations']['file_operations']['ensure_directory'](output_dir)
        
        # 2. Use data processing utilities for validation and cleaning
        df_validator = service_provider.get_service(utils['data_processing']['validators']['DataFrameValidator'])
        validation_result = df_validator.validate_dataframe(test_data)
        
        df_cleaner = service_provider.get_service(utils['data_processing']['cleaners']['DataFrameCleaner'])
        cleaned_data = df_cleaner.clean_dataframe(test_data)
        
        # 3. Use math validation for calculations
        mean_value = utils['math_validation']['basic_math']['safe_mean'](cleaned_data['close'].tolist())
        assert isinstance(mean_value, float)
        
        # 4. Use serialization utilities for saving results
        results_data = {
            'validation_result': validation_result.summary,
            'mean_close_price': mean_value,
            'data_shape': cleaned_data.shape
        }
        
        results_file = output_dir / 'results.json'
        utils['serialization']['convenience_functions']['save_json'](results_data, results_file)
        
        # 5. Use parquet utilities for data persistence
        data_file = output_dir / 'cleaned_data.parquet'
        parquet_handler = service_provider.get_service(utils['parquet']['ParquetUtils'])
        cleaned_data.to_parquet(data_file, index=False)
        
        validation_result = parquet_handler.validate_parquet_file(str(data_file))
        assert validation_result['valid'] is True
        
        # 6. Use M1 optimizers for performance monitoring
        memory_optimizer = service_provider.get_service(utils['m1_optimizers']['memory']['M1MemoryOptimizer'])
        memory_usage = memory_optimizer.get_memory_usage()
        assert isinstance(memory_usage, dict)
        
        # 7. Use common utilities for final analysis
        quality_report = utils['common_utilities']['data_quality']['create_data_quality_report'](cleaned_data)
        assert quality_report['status'] == 'success'
        
        # Verify all files were created
        assert results_file.exists()
        assert data_file.exists()
        
        # Verify data integrity
        loaded_data = parquet_handler.safe_read_parquet(str(data_file))
        assert len(loaded_data) == len(cleaned_data)

    def test_error_handling_and_fallbacks(self, service_provider):
        """Test that utilities handle errors gracefully with proper fallbacks."""
        utils = service_provider.get_all_utilities()
        
        # Test math validation with invalid inputs
        result = utils['math_validation']['basic_math']['safe_divide'](10, 0)
        assert result == 0.0  # Should return default value
        
        # Test file operations with non-existent files
        non_existent_file = Path('/non/existent/file.json')
        result = utils['common_operations']['file_operations']['safe_json_load'](non_existent_file, default={})
        assert result == {}
        
        # Test data processing with empty DataFrame
        empty_df = pd.DataFrame()
        quality_report = utils['common_utilities']['data_quality']['create_data_quality_report'](empty_df)
        assert quality_report['status'] == 'empty'

    def test_performance_optimization_integration(self, service_provider):
        """Test that M1 optimization utilities are properly integrated."""
        utils = service_provider.get_m1_optimizers()
        
        # Test GPU manager
        gpu_manager = utils['gpu']['M1GPUManager']
        
        # Test different operation types
        for operation_type in ['matrix_mult', 'neural_net', 'general']:
            should_use = gpu_manager.should_use_gpu(1000, operation_type)
            assert isinstance(should_use, bool)
        
        # Test memory optimizer
        memory_optimizer = utils['memory']['M1MemoryOptimizer']
        memory_usage = memory_optimizer.get_memory_usage()
        assert 'rss_gb' in memory_usage
        assert 'available_gb' in memory_usage
        
        # Test CPU optimizer
        cpu_optimizer = utils['cpu']['M1CPUOptimizer']
        optimal_workers = cpu_optimizer.get_optimal_workers_for_task('general')
        assert isinstance(optimal_workers, int)
        assert optimal_workers > 0

    def test_utility_integration_completeness(self, service_provider):
        """Test that all specified utilities are comprehensively integrated."""
        utils = service_provider.get_all_utilities()
        
        # Verify all required utility categories are present
        required_categories = [
            'common_operations',
            'common_utilities', 
            'math_validation',
            'serialization',
            'm1_optimizers',
            'data_processing',
            'parquet'
        ]
        
        for category in required_categories:
            assert category in utils, f"Missing utility category: {category}"
            assert isinstance(utils[category], dict), f"Utility category {category} should be a dict"
            assert len(utils[category]) > 0, f"Utility category {category} should not be empty"
        
        # Verify specific utility functions are available
        assert 'safe_divide' in utils['math_validation']['basic_math']
        assert 'validate_dataframe' in utils['data_processing']['validators']
        assert 'save_json' in utils['serialization']['convenience_functions']
        assert 'safe_read_parquet' in utils['parquet']['ParquetUtils'].__dict__ or hasattr(utils['parquet']['ParquetUtils'], 'safe_read_parquet')
        assert 'M1GPUManager' in utils['m1_optimizers']['gpu']
        assert 'M1MemoryOptimizer' in utils['m1_optimizers']['memory']
        assert 'M1CPUOptimizer' in utils['m1_optimizers']['cpu']

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])