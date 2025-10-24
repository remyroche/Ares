"""
Error Handling Framework Tests

This module provides comprehensive tests for the error handling framework,
including decorator functionality, error recovery manager, custom exception handling,
and retry mechanisms.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List, Optional
import time

# Import error handling components
from src.training.steps.error_handling import (
    TrainingStepError, ValidationError, DataLoadError, ModelTrainingError,
    ErrorRecoveryManager, RetryManager, ErrorHandler
)
from src.core.decorators import handles_errors, traced, log_execution_time
from src.training.steps.base_step import BaseStep


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_training_step_error(self):
        """Test TrainingStepError exception."""
        error = TrainingStepError("Test training error")
        assert str(error) == "Test training error"
        assert isinstance(error, Exception)

    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationError("Test validation error")
        assert str(error) == "Test validation error"
        assert isinstance(error, TrainingStepError)

    def test_data_load_error(self):
        """Test DataLoadError exception."""
        error = DataLoadError("Test data load error")
        assert str(error) == "Test data load error"
        assert isinstance(error, TrainingStepError)

    def test_model_training_error(self):
        """Test ModelTrainingError exception."""
        error = ModelTrainingError("Test model training error")
        assert str(error) == "Test model training error"
        assert isinstance(error, TrainingStepError)

    def test_exception_inheritance(self):
        """Test exception inheritance hierarchy."""
        # Test that all custom exceptions inherit from TrainingStepError
        assert issubclass(ValidationError, TrainingStepError)
        assert issubclass(DataLoadError, TrainingStepError)
        assert issubclass(ModelTrainingError, TrainingStepError)
        
        # Test that TrainingStepError inherits from Exception
        assert issubclass(TrainingStepError, Exception)


class TestErrorRecoveryManager:
    """Test ErrorRecoveryManager functionality."""

    @pytest.fixture
    def recovery_manager(self):
        """Create ErrorRecoveryManager instance for testing."""
        return ErrorRecoveryManager()

    def test_initialization(self, recovery_manager):
        """Test ErrorRecoveryManager initialization."""
        assert recovery_manager.recovery_strategies == {}
        assert recovery_manager.error_counts == {}
        assert recovery_manager.max_retries == 3

    def test_register_recovery_strategy(self, recovery_manager):
        """Test registering recovery strategies."""
        def test_strategy(error):
            return "recovered"
        
        recovery_manager.register_strategy(ValueError, test_strategy)
        assert ValueError in recovery_manager.recovery_strategies
        assert recovery_manager.recovery_strategies[ValueError] == test_strategy

    def test_recovery_strategy_execution(self, recovery_manager):
        """Test recovery strategy execution."""
        def test_strategy(error):
            return "recovered"
        
        recovery_manager.register_strategy(ValueError, test_strategy)
        result = recovery_manager.execute_recovery(ValueError("Test error"))
        assert result == "recovered"

    def test_no_recovery_strategy(self, recovery_manager):
        """Test behavior when no recovery strategy is registered."""
        with pytest.raises(ValueError):
            recovery_manager.execute_recovery(ValueError("Test error"))

    def test_error_counting(self, recovery_manager):
        """Test error counting functionality."""
        error = ValueError("Test error")
        
        # Test initial count
        assert recovery_manager.get_error_count(error) == 0
        
        # Test incrementing count
        recovery_manager.increment_error_count(error)
        assert recovery_manager.get_error_count(error) == 1
        
        # Test resetting count
        recovery_manager.reset_error_count(error)
        assert recovery_manager.get_error_count(error) == 0

    def test_max_retries_exceeded(self, recovery_manager):
        """Test behavior when max retries exceeded."""
        error = ValueError("Test error")
        
        # Set max retries to 2
        recovery_manager.max_retries = 2
        
        # Increment error count beyond max retries
        recovery_manager.increment_error_count(error)
        recovery_manager.increment_error_count(error)
        recovery_manager.increment_error_count(error)
        
        # Should raise exception when max retries exceeded
        with pytest.raises(ValueError):
            recovery_manager.execute_recovery(error)


class TestRetryManager:
    """Test RetryManager functionality."""

    @pytest.fixture
    def retry_manager(self):
        """Create RetryManager instance for testing."""
        return RetryManager()

    def test_initialization(self, retry_manager):
        """Test RetryManager initialization."""
        assert retry_manager.max_retries == 3
        assert retry_manager.delay == 1.0
        assert retry_manager.backoff_factor == 2.0

    def test_successful_retry(self, retry_manager):
        """Test successful retry after initial failure."""
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = retry_manager.retry(failing_function)
        assert result == "success"
        assert call_count == 3

    def test_max_retries_exceeded(self, retry_manager):
        """Test behavior when max retries exceeded."""
        def always_failing_function():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            retry_manager.retry(always_failing_function)

    def test_retry_with_delay(self, retry_manager):
        """Test retry with delay."""
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return "success"
        
        start_time = time.time()
        result = retry_manager.retry(failing_function)
        end_time = time.time()
        
        assert result == "success"
        assert end_time - start_time >= retry_manager.delay

    def test_retry_with_backoff(self, retry_manager):
        """Test retry with exponential backoff."""
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        start_time = time.time()
        result = retry_manager.retry(failing_function)
        end_time = time.time()
        
        assert result == "success"
        # Should have exponential backoff delay
        expected_delay = retry_manager.delay + (retry_manager.delay * retry_manager.backoff_factor)
        assert end_time - start_time >= expected_delay

    def test_retry_with_custom_exceptions(self, retry_manager):
        """Test retry with custom exception types."""
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValidationError("Temporary validation error")
            return "success"
        
        result = retry_manager.retry(failing_function, exceptions=(ValidationError,))
        assert result == "success"

    def test_retry_with_success_callback(self, retry_manager):
        """Test retry with success callback."""
        success_called = False
        
        def success_callback(result):
            nonlocal success_called
            success_called = True
        
        def successful_function():
            return "success"
        
        result = retry_manager.retry(successful_function, success_callback=success_callback)
        assert result == "success"
        assert success_called

    def test_retry_with_failure_callback(self, retry_manager):
        """Test retry with failure callback."""
        failure_called = False
        
        def failure_callback(error):
            nonlocal failure_called
            failure_called = True
        
        def always_failing_function():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            retry_manager.retry(always_failing_function, failure_callback=failure_callback)
        
        assert failure_called


class TestErrorHandler:
    """Test ErrorHandler functionality."""

    @pytest.fixture
    def error_handler(self):
        """Create ErrorHandler instance for testing."""
        return ErrorHandler()

    def test_initialization(self, error_handler):
        """Test ErrorHandler initialization."""
        assert error_handler.handlers == {}
        assert error_handler.default_handler is None

    def test_register_handler(self, error_handler):
        """Test registering error handlers."""
        def test_handler(error):
            return "handled"
        
        error_handler.register_handler(ValueError, test_handler)
        assert ValueError in error_handler.handlers
        assert error_handler.handlers[ValueError] == test_handler

    def test_handle_error(self, error_handler):
        """Test handling errors."""
        def test_handler(error):
            return "handled"
        
        error_handler.register_handler(ValueError, test_handler)
        result = error_handler.handle_error(ValueError("Test error"))
        assert result == "handled"

    def test_handle_error_with_default(self, error_handler):
        """Test handling errors with default handler."""
        def default_handler(error):
            return "default_handled"
        
        error_handler.set_default_handler(default_handler)
        result = error_handler.handle_error(ValueError("Test error"))
        assert result == "default_handled"

    def test_handle_error_no_handler(self, error_handler):
        """Test handling errors when no handler is registered."""
        with pytest.raises(ValueError):
            error_handler.handle_error(ValueError("Test error"))


class TestErrorHandlingDecorators:
    """Test error handling decorators."""

    def test_handles_errors_decorator(self):
        """Test handles_errors decorator."""
        @handles_errors(exceptions=(ValueError,), default_return="error_handled")
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        assert result == "error_handled"

    def test_handles_errors_decorator_with_multiple_exceptions(self):
        """Test handles_errors decorator with multiple exceptions."""
        @handles_errors(exceptions=(ValueError, TypeError), default_return="error_handled")
        def failing_function():
            raise TypeError("Test error")
        
        result = failing_function()
        assert result == "error_handled"

    def test_handles_errors_decorator_with_custom_handler(self):
        """Test handles_errors decorator with custom handler."""
        def custom_handler(error):
            return f"custom_handled: {str(error)}"
        
        @handles_errors(exceptions=(ValueError,), handler=custom_handler)
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        assert result == "custom_handled: Test error"

    def test_handles_errors_decorator_with_context(self):
        """Test handles_errors decorator with context."""
        @handles_errors(exceptions=(ValueError,), default_return="error_handled", context="test_context")
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        assert result == "error_handled"

    def test_handles_errors_decorator_success_case(self):
        """Test handles_errors decorator with successful execution."""
        @handles_errors(exceptions=(ValueError,), default_return="error_handled")
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"

    def test_traced_decorator(self):
        """Test traced decorator."""
        @traced
        def traced_function():
            return "traced_result"
        
        result = traced_function()
        assert result == "traced_result"

    def test_log_execution_time_decorator(self):
        """Test log_execution_time decorator."""
        @log_execution_time
        def timed_function():
            return "timed_result"
        
        result = timed_function()
        assert result == "timed_result"


class TestBaseStepErrorHandling:
    """Test BaseStep error handling integration."""

    @pytest.fixture
    def base_step(self):
        """Create BaseStep instance for testing."""
        return BaseStep("test_step", {"test_param": "test_value"})

    def test_error_logging(self, base_step):
        """Test error logging in BaseStep."""
        base_step.log_error("Test error message")
        base_step.logger.error.assert_called_with("Test error message")

    def test_warning_logging(self, base_step):
        """Test warning logging in BaseStep."""
        base_step.log_warning("Test warning message")
        base_step.logger.warning.assert_called_with("Test warning message")

    def test_info_logging(self, base_step):
        """Test info logging in BaseStep."""
        base_step.log_info("Test info message")
        base_step.logger.info.assert_called_with("Test info message")

    def test_error_recovery(self, base_step):
        """Test error recovery in BaseStep."""
        def failing_operation():
            raise ValueError("Test error")
        
        result = base_step.with_error_recovery(failing_operation, default_return="recovery_value")
        assert result == "recovery_value"

    def test_retry_mechanism(self, base_step):
        """Test retry mechanism in BaseStep."""
        call_count = 0
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = base_step.with_retry(failing_operation, max_retries=3)
        assert result == "success"
        assert call_count == 3

    def test_validation_error_handling(self, base_step):
        """Test validation error handling in BaseStep."""
        with pytest.raises(ValidationError):
            base_step.raise_validation_error("Test validation error")

    def test_training_error_handling(self, base_step):
        """Test training error handling in BaseStep."""
        with pytest.raises(TrainingStepError):
            base_step.raise_training_error("Test training error")

    def test_data_load_error_handling(self, base_step):
        """Test data load error handling in BaseStep."""
        with pytest.raises(DataLoadError):
            base_step.raise_data_load_error("Test data load error")

    def test_model_training_error_handling(self, base_step):
        """Test model training error handling in BaseStep."""
        with pytest.raises(ModelTrainingError):
            base_step.raise_model_training_error("Test model training error")

    def test_error_context_management(self, base_step):
        """Test error context management in BaseStep."""
        with base_step.error_context("test_operation"):
            pass
        
        # Should not raise any exceptions

    def test_error_metrics_tracking(self, base_step):
        """Test error metrics tracking in BaseStep."""
        # Test error counting
        base_step.track_error("test_error")
        assert base_step.get_error_count("test_error") == 1
        
        # Test error rate calculation
        base_step.track_success("test_operation")
        base_step.track_error("test_operation")
        error_rate = base_step.get_error_rate("test_operation")
        assert error_rate == 0.5

    def test_error_recovery_strategies(self, base_step):
        """Test error recovery strategies in BaseStep."""
        def recovery_strategy(error):
            return "recovered"
        
        base_step.register_recovery_strategy(ValueError, recovery_strategy)
        
        def failing_operation():
            raise ValueError("Test error")
        
        result = base_step.with_recovery(failing_operation)
        assert result == "recovered"

    def test_error_handling_in_async_operations(self, base_step):
        """Test error handling in async operations."""
        async def failing_async_operation():
            raise ValueError("Async error")
        
        async def test_async_error_handling():
            result = await base_step.run_async_with_error_handling(
                failing_async_operation, 
                default_return="async_recovery"
            )
            assert result == "async_recovery"
        
        asyncio.run(test_async_error_handling())

    def test_error_handling_with_artifacts(self, base_step):
        """Test error handling with artifacts."""
        base_step.set_artifact("error_context", {"operation": "test"})
        
        def failing_operation():
            raise ValueError("Test error")
        
        try:
            base_step.with_error_handling(failing_operation)
        except ValueError:
            # Check that error context is preserved in artifacts
            error_context = base_step.get_artifact("error_context")
            assert error_context is not None

    def test_error_handling_with_performance_tracking(self, base_step):
        """Test error handling with performance tracking."""
        def failing_operation():
            raise ValueError("Test error")
        
        with base_step.performance_tracker("failing_operation"):
            try:
                failing_operation()
            except ValueError:
                pass
        
        # Check that performance metrics are tracked even when errors occur
        performance_metrics = base_step.get_performance_metrics()
        assert "failing_operation" in performance_metrics


class TestErrorHandlingIntegration:
    """Test error handling integration with other components."""

    def test_integration_with_training_pipeline(self):
        """Test error handling integration with training pipeline."""
        step = BaseStep("training_step", {"model_type": "lightgbm"})
        
        def training_operation():
            raise ModelTrainingError("Training failed")
        
        result = step.with_error_handling(training_operation, default_return="training_failed")
        assert result == "training_failed"

    def test_integration_with_data_processing(self):
        """Test error handling integration with data processing."""
        step = BaseStep("data_processing_step", {"chunk_size": 1000})
        
        def data_processing_operation():
            raise DataLoadError("Data load failed")
        
        result = step.with_error_handling(data_processing_operation, default_return="data_load_failed")
        assert result == "data_load_failed"

    def test_integration_with_validation(self):
        """Test error handling integration with validation."""
        step = BaseStep("validation_step", {"validation_rules": []})
        
        def validation_operation():
            raise ValidationError("Validation failed")
        
        result = step.with_error_handling(validation_operation, default_return="validation_failed")
        assert result == "validation_failed"

    def test_integration_with_model_evaluation(self):
        """Test error handling integration with model evaluation."""
        step = BaseStep("evaluation_step", {"metrics": ["accuracy"]})
        
        def evaluation_operation():
            raise TrainingStepError("Evaluation failed")
        
        result = step.with_error_handling(evaluation_operation, default_return="evaluation_failed")
        assert result == "evaluation_failed"


if __name__ == "__main__":
    pytest.main([__file__])