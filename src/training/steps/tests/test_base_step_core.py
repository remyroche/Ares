"""
BaseStep Core Functionality Tests

This module provides comprehensive tests for the BaseStep core functionality,
including utility availability checking, convenience methods, error handling,
hardware optimization, data operations, and model persistence.
"""

import pytest
import asyncio
import logging
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List, Optional
import tempfile
import os
from pathlib import Path

# Import BaseStep and related classes
from src.training.steps.base_step import BaseStep
from src.training.steps.error_handling import (
    TrainingStepError, ValidationError, DataLoadError, ModelTrainingError
)
from src.core.decorators import handles_errors, traced, log_execution_time


class TestBaseStepCore:
    """Test suite for BaseStep core functionality."""

    @pytest.fixture
    def base_step(self):
        """Create a BaseStep instance for testing."""
        return BaseStep("test_step", {"test_param": "test_value"})

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock(spec=logging.Logger)

    def test_initialization(self, base_step):
        """Test BaseStep initialization."""
        assert base_step.name == "test_step"
        assert base_step.config == {"test_param": "test_value"}
        assert base_step.logger is not None
        assert base_step.artifacts == {}

    def test_initialization_with_logger(self, mock_logger):
        """Test BaseStep initialization with custom logger."""
        step = BaseStep("test_step", {"test_param": "test_value"}, mock_logger)
        assert step.logger == mock_logger

    def test_utility_availability_checking(self, base_step):
        """Test utility availability checking."""
        # Test available utilities
        assert base_step.has_utility('m1_gpu_manager') is True
        assert base_step.has_utility('m1_memory_optimizer') is True
        assert base_step.has_utility('m1_cpu_optimizer') is True
        assert base_step.has_utility('unified_matrix_operations') is True
        
        # Test non-existent utility
        assert base_step.has_utility('non_existent_utility') is False

    def test_utility_retrieval(self, base_step):
        """Test utility retrieval."""
        # Test getting available utility
        gpu_manager = base_step.get_utility('m1_gpu_manager')
        assert gpu_manager is not None
        
        # Test getting non-existent utility
        with pytest.raises(ValueError, match="Utility 'non_existent_utility' not available"):
            base_step.get_utility('non_existent_utility')

    def test_utility_retrieval_with_args(self, base_step):
        """Test utility retrieval with arguments."""
        # Test getting utility with arguments
        matrix_ops = base_step.get_utility('unified_matrix_operations', 'test_arg')
        assert matrix_ops is not None

    def test_convenience_methods(self, base_step):
        """Test convenience methods."""
        # Test get_name
        assert base_step.get_name() == "test_step"
        
        # Test get_config
        assert base_step.get_config() == {"test_param": "test_value"}
        
        # Test get_artifacts
        assert base_step.get_artifacts() == {}
        
        # Test set_artifact
        base_step.set_artifact('test_key', 'test_value')
        assert base_step.get_artifacts() == {'test_key': 'test_value'}
        
        # Test get_artifact
        assert base_step.get_artifact('test_key') == 'test_value'
        assert base_step.get_artifact('non_existent_key') is None

    def test_error_handling(self, base_step):
        """Test error handling functionality."""
        # Test error logging
        base_step.log_error("Test error message")
        base_step.logger.error.assert_called_with("Test error message")
        
        # Test warning logging
        base_step.log_warning("Test warning message")
        base_step.logger.warning.assert_called_with("Test warning message")
        
        # Test info logging
        base_step.log_info("Test info message")
        base_step.logger.info.assert_called_with("Test info message")

    def test_hardware_optimization_utilities(self, base_step):
        """Test hardware optimization utilities."""
        # Test M1 GPU manager
        gpu_manager = base_step.get_m1_gpu_manager()
        assert gpu_manager is not None
        
        # Test M1 memory optimizer
        memory_optimizer = base_step.get_m1_memory_optimizer()
        assert memory_optimizer is not None
        
        # Test M1 CPU optimizer
        cpu_optimizer = base_step.get_m1_cpu_optimizer()
        assert cpu_optimizer is not None

    def test_data_operations(self, base_step):
        """Test data operations utilities."""
        # Test dataframe validation
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = base_step.validate_dataframe(df)
        assert result is True
        
        # Test array validation
        arr = np.array([1, 2, 3, 4, 5])
        result = base_step.validate_array(arr)
        assert result is True
        
        # Test data conversion
        converted = base_step.convert_to_dataframe({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert isinstance(converted, pd.DataFrame)
        assert converted.shape == (3, 2)

    def test_model_persistence(self, base_step):
        """Test model persistence utilities."""
        # Test model saving
        model = {'type': 'test_model', 'params': {'test': 'value'}}
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'test_model.pkl')
            result = base_step.save_model(model, filepath)
            assert result is True
            assert os.path.exists(filepath)
        
        # Test model loading
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'test_model.pkl')
            base_step.save_model(model, filepath)
            loaded_model = base_step.load_model(filepath)
            assert loaded_model == model

    def test_async_operations(self, base_step):
        """Test async operations."""
        async def test_async_operation():
            return "async_result"
        
        # Test async execution
        result = asyncio.run(base_step.run_async(test_async_operation))
        assert result == "async_result"

    def test_validation_framework(self, base_step):
        """Test validation framework."""
        # Test input validation
        valid_data = {'features': pd.DataFrame({'A': [1, 2, 3]}), 'targets': pd.Series([1, 0, 1])}
        result = base_step.validate_inputs(valid_data)
        assert result is True
        
        # Test invalid input
        invalid_data = {'features': None, 'targets': None}
        with pytest.raises(ValidationError):
            base_step.validate_inputs(invalid_data)

    def test_performance_monitoring(self, base_step):
        """Test performance monitoring."""
        # Test performance tracking
        with base_step.performance_tracker("test_operation"):
            pass
        
        # Test memory monitoring
        memory_usage = base_step.get_memory_usage()
        assert isinstance(memory_usage, dict)
        assert 'used_memory' in memory_usage

    def test_error_recovery(self, base_step):
        """Test error recovery mechanisms."""
        # Test error recovery
        def failing_operation():
            raise ValueError("Test error")
        
        result = base_step.with_error_recovery(failing_operation, default_return="recovery_value")
        assert result == "recovery_value"

    def test_configuration_management(self, base_step):
        """Test configuration management."""
        # Test config update
        base_step.update_config({'new_param': 'new_value'})
        assert base_step.config['new_param'] == 'new_value'
        
        # Test config validation
        base_step.validate_config()
        # Should not raise any exceptions

    def test_artifact_management(self, base_step):
        """Test artifact management."""
        # Test artifact setting
        base_step.set_artifact('test_artifact', {'data': 'test'})
        assert base_step.get_artifact('test_artifact') == {'data': 'test'}
        
        # Test artifact validation
        result = base_step.validate_artifacts({'test_artifact': {'data': 'test'}})
        assert result is True
        
        # Test artifact clearing
        base_step.clear_artifacts()
        assert base_step.get_artifacts() == {}

    def test_logging_integration(self, base_step):
        """Test logging integration."""
        # Test structured logging
        base_step.log_structured("test_event", {"param1": "value1", "param2": "value2"})
        base_step.logger.info.assert_called()
        
        # Test performance logging
        base_step.log_performance("test_operation", 1.5, {"memory": "100MB"})
        base_step.logger.info.assert_called()

    def test_hardware_integration(self, base_step):
        """Test hardware integration."""
        # Test hardware optimization
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        optimized_data = base_step.optimize_for_hardware(data)
        assert isinstance(optimized_data, pd.DataFrame)
        
        # Test memory optimization
        memory_optimized = base_step.optimize_memory(data)
        assert isinstance(memory_optimized, pd.DataFrame)

    def test_error_handling_decorators(self, base_step):
        """Test error handling decorators."""
        # Test handles_errors decorator
        @handles_errors(exceptions=(ValueError,), default_return="error_handled")
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        assert result == "error_handled"

    def test_tracing_decorators(self, base_step):
        """Test tracing decorators."""
        # Test traced decorator
        @traced
        def traced_function():
            return "traced_result"
        
        result = traced_function()
        assert result == "traced_result"

    def test_execution_time_decorators(self, base_step):
        """Test execution time decorators."""
        # Test log_execution_time decorator
        @log_execution_time
        def timed_function():
            return "timed_result"
        
        result = timed_function()
        assert result == "timed_result"

    def test_utility_methods_comprehensive(self, base_step):
        """Test comprehensive utility methods."""
        # Test all utility methods exist
        utility_methods = [
            'get_m1_gpu_manager', 'get_m1_memory_optimizer', 'get_m1_cpu_optimizer',
            'get_unified_matrix_operations', 'get_vectorbt_rolling_optimizer',
            'get_hyperparameter_optimizer', 'get_data_leakage_detector',
            'get_time_series_split_validator', 'get_enhanced_oof_generator',
            'get_artifact_manager', 'get_version_manager'
        ]
        
        for method_name in utility_methods:
            assert hasattr(base_step, method_name)
            method = getattr(base_step, method_name)
            assert callable(method)

    def test_error_handling_comprehensive(self, base_step):
        """Test comprehensive error handling."""
        # Test custom exception handling
        with pytest.raises(TrainingStepError):
            base_step.raise_training_error("Test training error")
        
        with pytest.raises(ValidationError):
            base_step.raise_validation_error("Test validation error")
        
        with pytest.raises(DataLoadError):
            base_step.raise_data_load_error("Test data load error")
        
        with pytest.raises(ModelTrainingError):
            base_step.raise_model_training_error("Test model training error")

    def test_data_quality_checks(self, base_step):
        """Test data quality checks."""
        # Test valid data
        valid_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert base_step.check_data_quality(valid_df) is True
        
        # Test invalid data (with NaN)
        invalid_df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, 5, 6]})
        assert base_step.check_data_quality(invalid_df) is False

    def test_memory_management(self, base_step):
        """Test memory management."""
        # Test memory optimization
        data = pd.DataFrame({'A': range(1000), 'B': range(1000)})
        optimized = base_step.optimize_memory(data)
        assert isinstance(optimized, pd.DataFrame)
        
        # Test memory cleanup
        base_step.cleanup_memory()
        # Should not raise any exceptions

    def test_performance_optimization(self, base_step):
        """Test performance optimization."""
        # Test performance optimization
        data = pd.DataFrame({'A': range(1000), 'B': range(1000)})
        optimized = base_step.optimize_performance(data)
        assert isinstance(optimized, pd.DataFrame)

    def test_integration_with_existing_utilities(self, base_step):
        """Test integration with existing utilities."""
        # Test that BaseStep can work with existing utility functions
        from src.utils.common_operations import safe_divide
        result = base_step.safe_divide(10, 2)
        assert result == 5.0
        
        # Test that BaseStep can work with math validation
        from src.utils.math_validation import validate_finite
        result = base_step.validate_finite(42.0)
        assert result == 42.0

    def test_concurrent_operations(self, base_step):
        """Test concurrent operations."""
        async def test_concurrent():
            # Test concurrent utility access
            tasks = [
                base_step.run_async(lambda: base_step.get_m1_gpu_manager()),
                base_step.run_async(lambda: base_step.get_m1_memory_optimizer()),
                base_step.run_async(lambda: base_step.get_m1_cpu_optimizer())
            ]
            results = await asyncio.gather(*tasks)
            assert all(result is not None for result in results)
        
        asyncio.run(test_concurrent())

    def test_error_recovery_mechanisms(self, base_step):
        """Test error recovery mechanisms."""
        # Test automatic error recovery
        def unreliable_operation():
            if np.random.random() < 0.5:
                raise ValueError("Random error")
            return "success"
        
        result = base_step.with_retry(unreliable_operation, max_retries=3)
        assert result == "success"

    def test_utility_caching(self, base_step):
        """Test utility caching."""
        # Test that utilities are cached
        gpu_manager1 = base_step.get_m1_gpu_manager()
        gpu_manager2 = base_step.get_m1_gpu_manager()
        assert gpu_manager1 is gpu_manager2  # Same instance due to caching

    def test_configuration_validation(self, base_step):
        """Test configuration validation."""
        # Test valid configuration
        base_step.config = {'param1': 'value1', 'param2': 42}
        assert base_step.validate_config() is True
        
        # Test invalid configuration
        base_step.config = {'param1': None, 'param2': 'invalid'}
        with pytest.raises(ValidationError):
            base_step.validate_config()

    def test_artifact_serialization(self, base_step):
        """Test artifact serialization."""
        # Test artifact serialization
        artifact = {'data': [1, 2, 3], 'metadata': {'type': 'test'}}
        base_step.set_artifact('test_artifact', artifact)
        
        # Test artifact persistence
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'artifacts.json')
            base_step.save_artifacts(filepath)
            assert os.path.exists(filepath)
            
            # Test artifact loading
            loaded_artifacts = base_step.load_artifacts(filepath)
            assert loaded_artifacts == base_step.get_artifacts()


class TestBaseStepIntegration:
    """Test BaseStep integration with other components."""

    def test_integration_with_training_pipeline(self):
        """Test integration with training pipeline."""
        # This would test integration with actual training pipeline components
        # For now, we'll test the basic integration
        step = BaseStep("training_step", {"model_type": "lightgbm"})
        assert step.name == "training_step"
        assert step.config["model_type"] == "lightgbm"

    def test_integration_with_data_processing(self):
        """Test integration with data processing components."""
        step = BaseStep("data_processing_step", {"chunk_size": 1000})
        
        # Test data processing utilities
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        processed = step.process_data(data)
        assert isinstance(processed, pd.DataFrame)

    def test_integration_with_model_training(self):
        """Test integration with model training components."""
        step = BaseStep("model_training_step", {"epochs": 100})
        
        # Test model training utilities
        model = step.create_model("lightgbm")
        assert model is not None

    def test_integration_with_evaluation(self):
        """Test integration with evaluation components."""
        step = BaseStep("evaluation_step", {"metrics": ["accuracy", "f1"]})
        
        # Test evaluation utilities
        predictions = np.array([1, 0, 1, 0])
        targets = np.array([1, 0, 1, 0])
        metrics = step.evaluate_model(predictions, targets)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics


if __name__ == "__main__":
    pytest.main([__file__])