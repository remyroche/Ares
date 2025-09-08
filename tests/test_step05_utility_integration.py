"""
Comprehensive test suite for Step05 utility integration.

This test suite validates that all utility modules are properly integrated
and functioning correctly within the Step05 optimized integrated pipeline.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import logging
from typing import Dict, Any

# Import the step05 components
from src.training.steps.step05_optimized_integrated import Step05OptimizedIntegrated
from src.training.steps.step05_dependency_injection import (
    Step05DependencyContainer, 
    UtilityConfig, 
    initialize_step05_utilities,
    get_step05_container
)

# Import utility modules for direct testing
from src.utils.common_operations import *
from src.utils.common_utilities import *
from src.utils.math_validation import *
from src.utils.parquet_utils import ParquetUtils
from src.utils.serialization_utils import *
from src.utils.data_processing_utils import *
from src.utils.m1_gpu_utils import M1GPUManager, M1PerformanceOptimizer
from src.utils.m1_memory_optimizer import M1MemoryOptimizer, M1DataManager
from src.utils.m1_cpu_optimizer import M1CPUOptimizer, M1BatchProcessor


class TestStep05UtilityIntegration:
    """Test suite for Step05 utility integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create a sample configuration for testing."""
        return {
            'SYMBOL': 'BTCUSDT',
            'EXCHANGE': 'binance',
            'TIMEFRAME': '1h',
            'DATA_DIR': str(temp_dir),
            'vectorized_labelling_orchestrator': {
                'auto_recalculate_hmm_barriers': True,
                'hmm_barrier_regime_column': 'hmm_regime',
                'time_barrier_minutes': 30,
                'max_lookahead': 100,
                'profit_take_multiplier': 0.002,
                'stop_loss_multiplier': 0.001
            },
            'transaction_costs': {
                'maker_fee': 0.001,
                'taker_fee': 0.001,
                'slippage_bps': 2.0,
                'funding_rate': 0.0001
            },
            'memory': {
                'thresholds': {
                    'warning_mb': 1000.0,
                    'critical_mb': 2000.0,
                    'max_memory_mb': 4000.0
                },
                'optimization_strategies': {
                    'dtype_optimization': True,
                    'categorical_optimization': True,
                    'sparse_optimization': True,
                    'chunk_processing': True,
                    'garbage_collection': True
                }
            },
            'streaming': {
                'chunk_size': 10000,
                'max_memory_mb': 1000.0,
                'overlap_rows': 100,
                'enable_compression': True,
                'enable_parallel_processing': False,
                'max_workers': 4,
                'progress_reporting_interval': 10
            },
            # Utility integration configuration
            'enable_gpu_optimization': True,
            'enable_memory_optimization': True,
            'enable_cpu_optimization': True,
            'enable_math_validation': True,
            'enable_data_validation': True,
            'enable_serialization': True,
            'memory_limit_gb': 8.0,
            'max_workers': 4,
            'gpu_memory_threshold': 0.8,
            'log_level': 'INFO'
        }
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1H'),
            'open': np.random.uniform(100, 200, 1000),
            'high': np.random.uniform(100, 200, 1000),
            'low': np.random.uniform(100, 200, 1000),
            'close': np.random.uniform(100, 200, 1000),
            'volume': np.random.uniform(1000, 10000, 1000),
            'hmm_regime': np.random.choice([0, 1, 2], 1000),
            'returns': np.random.normal(0, 0.01, 1000),
            'volatility': np.random.uniform(0.01, 0.05, 1000)
        }
        return pd.DataFrame(data)
    
    def test_dependency_injection_container_initialization(self, sample_config):
        """Test that the dependency injection container initializes correctly."""
        utility_config = UtilityConfig(
            enable_gpu_optimization=sample_config.get('enable_gpu_optimization', True),
            enable_memory_optimization=sample_config.get('enable_memory_optimization', True),
            enable_cpu_optimization=sample_config.get('enable_cpu_optimization', True),
            enable_math_validation=sample_config.get('enable_math_validation', True),
            enable_data_validation=sample_config.get('enable_data_validation', True),
            enable_serialization=sample_config.get('enable_serialization', True),
            memory_limit_gb=sample_config.get('memory_limit_gb', 8.0),
            max_workers=sample_config.get('max_workers', 4),
            gpu_memory_threshold=sample_config.get('gpu_memory_threshold', 0.8),
            log_level=sample_config.get('log_level', 'INFO')
        )
        
        container = initialize_step05_utilities(utility_config)
        
        # Test that container is properly initialized
        assert container is not None
        assert isinstance(container, Step05DependencyContainer)
        
        # Test that all utility categories are available
        categories = [
            'common_operations', 'common_utilities', 'math_validation',
            'parquet_utils', 'serialization_utils', 'data_processing_utils',
            'm1_gpu_utils', 'm1_memory_utils', 'm1_cpu_utils'
        ]
        
        for category in categories:
            assert container.has_category(category), f"Category {category} not found"
            category_utils = container.get_category(category)
            assert category_utils is not None, f"Category {category} is None"
            assert isinstance(category_utils, dict), f"Category {category} is not a dict"
    
    def test_utility_health_check(self, sample_config):
        """Test that utility health check works correctly."""
        utility_config = UtilityConfig()
        container = initialize_step05_utilities(utility_config)
        
        health_status = container.health_check()
        
        assert isinstance(health_status, dict)
        assert 'overall_health' in health_status
        assert 'categories' in health_status
        assert 'total_utilities' in health_status
        assert 'healthy_utilities' in health_status
        assert 'unhealthy_utilities' in health_status
    
    def test_utility_summary(self, sample_config):
        """Test that utility summary provides comprehensive information."""
        utility_config = UtilityConfig()
        container = initialize_step05_utilities(utility_config)
        
        summary = container.get_utility_summary()
        
        assert isinstance(summary, dict)
        assert len(summary) > 0
        
        # Check that all expected categories are in summary
        expected_categories = [
            'common_operations', 'common_utilities', 'math_validation',
            'parquet_utils', 'serialization_utils', 'data_processing_utils',
            'm1_gpu_utils', 'm1_memory_utils', 'm1_cpu_utils'
        ]
        
        for category in expected_categories:
            assert category in summary, f"Category {category} missing from summary"
            category_info = summary[category]
            assert 'type' in category_info
            assert 'total_utilities' in category_info
    
    def test_step05_initialization_with_utilities(self, sample_config):
        """Test that Step05 initializes with all utilities properly integrated."""
        step = Step05OptimizedIntegrated(sample_config)
        
        # Test that utility references are properly set
        assert hasattr(step, 'utils')
        assert hasattr(step, 'common_ops')
        assert hasattr(step, 'common_utils')
        assert hasattr(step, 'math_validation')
        assert hasattr(step, 'parquet_utils')
        assert hasattr(step, 'serialization_utils')
        assert hasattr(step, 'data_processing_utils')
        assert hasattr(step, 'm1_gpu_utils')
        assert hasattr(step, 'm1_memory_utils')
        assert hasattr(step, 'm1_cpu_utils')
        
        # Test that performance metrics include utility metrics
        assert 'gpu_operations' in step.performance_metrics
        assert 'cpu_parallel_operations' in step.performance_metrics
        assert 'math_validation_operations' in step.performance_metrics
        assert 'data_processing_operations' in step.performance_metrics
        assert 'serialization_operations' in step.performance_metrics
    
    @pytest.mark.asyncio
    async def test_step05_initialization_async(self, sample_config):
        """Test async initialization of Step05 with utilities."""
        step = Step05OptimizedIntegrated(sample_config)
        await step.initialize()
        
        # Test that initialization completed successfully
        assert step.start_time is not None
        assert step.step_timings is not None
        assert len(step.step_timings) > 0
    
    def test_common_operations_integration(self, sample_config):
        """Test that common_operations utilities are properly integrated."""
        step = Step05OptimizedIntegrated(sample_config)
        
        # Test datetime operations
        current_time = step.common_ops['datetime_ops']['get_current_datetime']()
        assert current_time is not None
        
        formatted_time = step.common_ops['datetime_ops']['format_datetime'](current_time)
        assert isinstance(formatted_time, str)
        assert len(formatted_time) > 0
        
        # Test string operations
        test_string = "Hello World"
        lower_string = step.common_ops['string_ops']['safe_lower'](test_string)
        assert lower_string == "hello world"
        
        upper_string = step.common_ops['string_ops']['safe_upper'](test_string)
        assert upper_string == "HELLO WORLD"
        
        # Test math operations
        safe_float = step.common_ops['math_ops']['safe_float']("123.45", 0.0)
        assert safe_float == 123.45
        
        safe_int = step.common_ops['math_ops']['safe_int']("123", 0)
        assert safe_int == 123
    
    def test_math_validation_integration(self, sample_config):
        """Test that math_validation utilities are properly integrated."""
        step = Step05OptimizedIntegrated(sample_config)
        
        # Test safe math operations
        safe_divide = step.math_validation['safe_math_ops']['safe_divide'](10, 2, 0.0)
        assert safe_divide == 5.0
        
        safe_divide_zero = step.math_validation['safe_math_ops']['safe_divide'](10, 0, 0.0)
        assert safe_divide_zero == 0.0
        
        # Test validation operations
        step.math_validation['validation_ops']['validate_positive'](5.0, "test_value")
        
        with pytest.raises(ValueError):
            step.math_validation['validation_ops']['validate_positive'](-1.0, "test_value")
        
        # Test range validation
        step.math_validation['validation_ops']['validate_range'](5.0, 0.0, 10.0, "test_value")
        
        with pytest.raises(ValueError):
            step.math_validation['validation_ops']['validate_range'](15.0, 0.0, 10.0, "test_value")
    
    def test_data_processing_utils_integration(self, sample_config, sample_dataframe):
        """Test that data_processing_utils are properly integrated."""
        step = Step05OptimizedIntegrated(sample_config)
        
        # Test DataFrame validator
        validator = step.data_processing_utils['validators']['DataFrameValidator']()
        quality_report = validator.validate_dataframe(sample_dataframe)
        
        assert quality_report is not None
        assert hasattr(quality_report, 'is_valid')
        assert hasattr(quality_report, 'issues')
        assert hasattr(quality_report, 'summary')
        
        # Test DataFrame cleaner
        cleaner = step.data_processing_utils['validators']['DataFrameCleaner']()
        cleaned_df = cleaner.clean_dataframe(sample_dataframe)
        
        assert cleaned_df is not None
        assert isinstance(cleaned_df, pd.DataFrame)
        assert len(cleaned_df) > 0
        
        # Test convenience functions
        data_info = step.data_processing_utils['convenience_functions']['get_dataframe_info'](sample_dataframe)
        assert isinstance(data_info, dict)
        assert 'shape' in data_info
        assert 'columns' in data_info
        assert 'dtypes' in data_info
    
    def test_parquet_utils_integration(self, sample_config, temp_dir, sample_dataframe):
        """Test that parquet_utils are properly integrated."""
        step = Step05OptimizedIntegrated(sample_config)
        
        # Test ParquetUtils
        parquet_utils = step.parquet_utils['parquet_utils']
        assert isinstance(parquet_utils, ParquetUtils)
        
        # Test file operations
        test_file = temp_dir / "test.parquet"
        sample_dataframe.to_parquet(test_file)
        
        # Test file validation
        validation_result = parquet_utils.validate_parquet_file(str(test_file))
        assert validation_result['valid'] is True
        assert 'file_size' in validation_result
        
        # Test safe read
        loaded_df = parquet_utils.safe_read_parquet(str(test_file))
        assert loaded_df is not None
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == len(sample_dataframe)
    
    def test_serialization_utils_integration(self, sample_config, temp_dir):
        """Test that serialization_utils are properly integrated."""
        step = Step05OptimizedIntegrated(sample_config)
        
        # Test JSON serializer
        json_serializer = step.serialization_utils['serializers']['JSONSerializer']
        test_data = {"key": "value", "number": 123, "list": [1, 2, 3]}
        
        test_file = temp_dir / "test.json"
        success = json_serializer.save(test_data, str(test_file))
        assert success is True
        
        loaded_data = json_serializer.load(str(test_file))
        assert loaded_data == test_data
        
        # Test Pickle serializer
        pickle_serializer = step.serialization_utils['serializers']['PickleSerializer']
        test_file_pkl = temp_dir / "test.pkl"
        success = pickle_serializer.save(test_data, str(test_file_pkl))
        assert success is True
        
        loaded_data_pkl = pickle_serializer.load(str(test_file_pkl))
        assert loaded_data_pkl == test_data
    
    def test_m1_optimization_utilities_integration(self, sample_config):
        """Test that M1 optimization utilities are properly integrated."""
        step = Step05OptimizedIntegrated(sample_config)
        
        # Test GPU manager
        gpu_manager = step.m1_gpu_utils['gpu_manager']
        assert isinstance(gpu_manager, M1GPUManager)
        
        # Test memory optimizer
        memory_optimizer = step.m1_memory_utils['memory_optimizer']
        assert isinstance(memory_optimizer, M1MemoryOptimizer)
        
        # Test CPU optimizer
        cpu_optimizer = step.m1_cpu_utils['cpu_optimizer']
        assert isinstance(cpu_optimizer, M1CPUOptimizer)
        
        # Test performance optimizer
        performance_optimizer = step.m1_gpu_utils['performance_optimizer']
        assert isinstance(performance_optimizer, M1PerformanceOptimizer)
        
        # Test batch processor
        batch_processor = step.m1_cpu_utils['batch_processor']
        assert isinstance(batch_processor, M1BatchProcessor)
        
        # Test data manager
        data_manager = step.m1_memory_utils['data_manager']
        assert isinstance(data_manager, M1DataManager)
    
    def test_performance_metrics_tracking(self, sample_config):
        """Test that performance metrics are properly tracked for utilities."""
        step = Step05OptimizedIntegrated(sample_config)
        
        # Test that utility metrics are initialized
        assert step.performance_metrics['gpu_operations'] == 0
        assert step.performance_metrics['cpu_parallel_operations'] == 0
        assert step.performance_metrics['math_validation_operations'] == 0
        assert step.performance_metrics['data_processing_operations'] == 0
        assert step.performance_metrics['serialization_operations'] == 0
        
        # Simulate some operations
        step.performance_metrics['gpu_operations'] += 1
        step.performance_metrics['math_validation_operations'] += 5
        step.performance_metrics['data_processing_operations'] += 3
        
        # Test that metrics are updated
        assert step.performance_metrics['gpu_operations'] == 1
        assert step.performance_metrics['math_validation_operations'] == 5
        assert step.performance_metrics['data_processing_operations'] == 3
    
    def test_utility_configuration_validation(self, sample_config):
        """Test that utility configuration is properly validated."""
        step = Step05OptimizedIntegrated(sample_config)
        
        # Test that configuration values are properly set
        assert step.config.get('enable_gpu_optimization', True) is True
        assert step.config.get('enable_memory_optimization', True) is True
        assert step.config.get('enable_cpu_optimization', True) is True
        assert step.config.get('enable_math_validation', True) is True
        assert step.config.get('enable_data_validation', True) is True
        assert step.config.get('enable_serialization', True) is True
        
        # Test memory and worker limits
        assert step.config.get('memory_limit_gb', 8.0) == 8.0
        assert step.config.get('max_workers', 4) == 4
        assert step.config.get('gpu_memory_threshold', 0.8) == 0.8
    
    def test_utility_error_handling(self, sample_config):
        """Test that utility error handling works correctly."""
        step = Step05OptimizedIntegrated(sample_config)
        
        # Test that error handler is properly initialized
        assert hasattr(step, 'error_handler')
        assert step.error_handler is not None
        
        # Test error summary
        error_summary = step.error_handler.get_error_summary()
        assert isinstance(error_summary, dict)
        assert 'total_errors' in error_summary
        assert 'error_types' in error_summary
    
    @pytest.mark.asyncio
    async def test_comprehensive_utility_integration(self, sample_config, temp_dir, sample_dataframe):
        """Test comprehensive utility integration in a realistic scenario."""
        # Create sample data files
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        
        # Save sample data
        sample_file = data_dir / "BTCUSDT_binance_1h_triple_barrier.parquet"
        sample_dataframe.to_parquet(sample_file)
        
        # Update config with test data directory
        sample_config['DATA_DIR'] = str(data_dir)
        
        # Initialize step
        step = Step05OptimizedIntegrated(sample_config)
        await step.initialize()
        
        # Test that all utilities are working together
        assert step.utils is not None
        health_status = step.utils.health_check()
        assert health_status['overall_health'] is True
        
        # Test utility summary
        utility_summary = step.utils.get_utility_summary()
        assert len(utility_summary) >= 9  # All utility categories
        
        # Test performance summary includes utility metrics
        performance_summary = step.get_performance_summary()
        assert 'utility_integration' in performance_summary
        assert 'm1_optimization' in performance_summary
        assert 'utility_usage_metrics' in performance_summary
        
        utility_metrics = performance_summary['utility_usage_metrics']
        assert 'gpu_operations' in utility_metrics
        assert 'cpu_parallel_operations' in utility_metrics
        assert 'math_validation_operations' in utility_metrics
        assert 'data_processing_operations' in utility_metrics
        assert 'serialization_operations' in utility_metrics


class TestUtilityModuleDirectIntegration:
    """Test direct integration of individual utility modules."""
    
    def test_common_operations_module(self):
        """Test common_operations module directly."""
        # Test datetime operations
        current_time = get_current_datetime()
        assert current_time is not None
        
        formatted_time = format_datetime(current_time)
        assert isinstance(formatted_time, str)
        
        # Test string operations
        assert safe_lower("HELLO") == "hello"
        assert safe_upper("hello") == "HELLO"
        
        # Test math operations
        assert safe_float("123.45", 0.0) == 123.45
        assert safe_int("123", 0) == 123
    
    def test_math_validation_module(self):
        """Test math_validation module directly."""
        # Test safe math operations
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_log(2.718) == pytest.approx(1.0, rel=1e-2)
        assert safe_sqrt(16) == 4.0
        
        # Test validation functions
        validate_positive(5.0, "test")
        validate_range(5.0, 0.0, 10.0, "test")
        
        with pytest.raises(ValueError):
            validate_positive(-1.0, "test")
        
        with pytest.raises(ValueError):
            validate_range(15.0, 0.0, 10.0, "test")
    
    def test_parquet_utils_module(self):
        """Test parquet_utils module directly."""
        parquet_utils = ParquetUtils()
        assert parquet_utils is not None
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            test_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            test_df.to_parquet(tmp.name)
            
            # Test validation
            result = parquet_utils.validate_parquet_file(tmp.name)
            assert result['valid'] is True
            
            # Test safe read
            loaded_df = parquet_utils.safe_read_parquet(tmp.name)
            assert loaded_df is not None
            assert len(loaded_df) == 3
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_serialization_utils_module(self):
        """Test serialization_utils module directly."""
        # Test JSON serializer
        json_serializer = JSONSerializer()
        test_data = {"key": "value", "number": 123}
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            success = json_serializer.save(test_data, tmp.name)
            assert success is True
            
            loaded_data = json_serializer.load(tmp.name)
            assert loaded_data == test_data
            
            # Clean up
            Path(tmp.name).unlink()
        
        # Test Pickle serializer
        pickle_serializer = PickleSerializer()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            success = pickle_serializer.save(test_data, tmp.name)
            assert success is True
            
            loaded_data = pickle_serializer.load(tmp.name)
            assert loaded_data == test_data
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_data_processing_utils_module(self):
        """Test data_processing_utils module directly."""
        # Create test DataFrame
        test_df = pd.DataFrame({
            'a': [1, 2, 3, None, 5],
            'b': [1.1, 2.2, 3.3, 4.4, 5.5],
            'c': ['x', 'y', 'z', 'w', 'v']
        })
        
        # Test DataFrameValidator
        validator = DataFrameValidator()
        quality_report = validator.validate_dataframe(test_df)
        assert quality_report is not None
        assert hasattr(quality_report, 'is_valid')
        
        # Test DataFrameCleaner
        cleaner = DataFrameCleaner()
        cleaned_df = cleaner.clean_dataframe(test_df)
        assert cleaned_df is not None
        assert isinstance(cleaned_df, pd.DataFrame)
        
        # Test convenience functions
        data_info = get_dataframe_info(test_df)
        assert isinstance(data_info, dict)
        assert 'shape' in data_info
    
    def test_m1_optimization_modules(self):
        """Test M1 optimization modules directly."""
        # Test GPU Manager
        gpu_manager = M1GPUManager()
        assert gpu_manager is not None
        assert hasattr(gpu_manager, 'device')
        assert hasattr(gpu_manager, 'memory_info')
        
        # Test Memory Optimizer
        memory_optimizer = M1MemoryOptimizer()
        assert memory_optimizer is not None
        assert hasattr(memory_optimizer, 'optimize_memory')
        
        # Test CPU Optimizer
        cpu_optimizer = M1CPUOptimizer()
        assert cpu_optimizer is not None
        assert hasattr(cpu_optimizer, 'max_workers')
        
        # Test Performance Optimizer
        performance_optimizer = M1PerformanceOptimizer()
        assert performance_optimizer is not None
        
        # Test Batch Processor
        batch_processor = M1BatchProcessor()
        assert batch_processor is not None
        
        # Test Data Manager
        data_manager = M1DataManager()
        assert data_manager is not None


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v', '--tb=short'])