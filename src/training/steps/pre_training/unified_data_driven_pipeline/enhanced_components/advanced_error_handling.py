"""
Advanced Error Handling Framework for Unified Data-Driven Pipeline.

This module provides comprehensive error handling infrastructure similar to
FeatureLookbackOptimizationComponent but adapted for the unified pipeline.
"""

import logging
import traceback
from typing import Any, Optional, Dict, List, Callable, Type, Union
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from datetime import datetime

# Import utility modules
from src.utils.common_operations import (
    CommonUtilities, safe_dataframe_operation, safe_convert_dtypes,
    safe_merge_dataframes, safe_drop_columns, safe_rename_columns,
    safe_filter_dataframe, safe_groupby_operation, safe_apply_function,
    get_dataframe_info, create_summary_statistics, safe_log_metric,
    safe_log_params, safe_log_artifact, calculate_data_quality_metrics,
    validate_dataframe, validate_dataframe_columns, optimize_dataframe_dtypes,
    safe_fillna, safe_timestamp_conversion, guard_dataframe_nulls
)
from src.utils.serialization_utils import UniversalSerializer

try:
    from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_success, tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

import numpy as np
import pandas as pd

# Exception classes for fast failing
class PipelineError(Exception):
    """Base exception for pipeline-related errors."""

    def __init__(self, message: str, operation: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()

    def __str__(self):
        base_msg = f"PipelineError: {self.message}"
        if self.operation:
            base_msg += f" (Operation: {self.operation})"
        return base_msg

    def get_context(self) -> Dict[str, Any]:
        """Get error context information."""
        return {
            'message': self.message,
            'operation': self.operation,
            'context': self.context,
            'timestamp': self.timestamp
        }

class DataValidationError(PipelineError):
    """Exception raised when data validation fails."""

    def __init__(self, message: str, validation_type: str = None, data_info: Dict[str, Any] = None,
                 operation: str = None, context: Dict[str, Any] = None):
        super().__init__(message, operation, context)
        self.validation_type = validation_type
        self.data_info = data_info or {}

    def __str__(self):
        base_msg = f"DataValidationError: {self.message}"
        if self.validation_type:
            base_msg += f" (Type: {self.validation_type})"
        if self.operation:
            base_msg += f" (Operation: {self.operation})"
        return base_msg

    def get_validation_details(self) -> Dict[str, Any]:
        """Get detailed validation information."""
        details = self.get_context()
        details.update({
            'validation_type': self.validation_type,
            'data_info': self.data_info
        })
        return details

class FeatureGenerationError(PipelineError):
    """Exception raised when feature generation fails."""

    def __init__(self, message: str, feature_name: str = None, generation_step: str = None,
                 operation: str = None, context: Dict[str, Any] = None):
        super().__init__(message, operation, context)
        self.feature_name = feature_name
        self.generation_step = generation_step

    def __str__(self):
        base_msg = f"FeatureGenerationError: {self.message}"
        if self.feature_name:
            base_msg += f" (Feature: {self.feature_name})"
        if self.generation_step:
            base_msg += f" (Step: {self.generation_step})"
        if self.operation:
            base_msg += f" (Operation: {self.operation})"
        return base_msg

    def get_generation_details(self) -> Dict[str, Any]:
        """Get detailed feature generation information."""
        details = self.get_context()
        details.update({
            'feature_name': self.feature_name,
            'generation_step': self.generation_step
        })
        return details

class OptimizationError(PipelineError):
    """Exception raised when optimization fails."""

    def __init__(self, message: str, optimization_type: str = None, objective: str = None,
                 iteration: int = None, operation: str = None, context: Dict[str, Any] = None):
        super().__init__(message, operation, context)
        self.optimization_type = optimization_type
        self.objective = objective
        self.iteration = iteration

    def __str__(self):
        base_msg = f"OptimizationError: {self.message}"
        if self.optimization_type:
            base_msg += f" (Type: {self.optimization_type})"
        if self.objective:
            base_msg += f" (Objective: {self.objective})"
        if self.iteration is not None:
            base_msg += f" (Iteration: {self.iteration})"
        if self.operation:
            base_msg += f" (Operation: {self.operation})"
        return base_msg

    def get_optimization_details(self) -> Dict[str, Any]:
        """Get detailed optimization information."""
        details = self.get_context()
        details.update({
            'optimization_type': self.optimization_type,
            'objective': self.objective,
            'iteration': self.iteration
        })
        return details

class CacheError(PipelineError):
    """Exception raised when cache operations fail."""

    def __init__(self, message: str, cache_key: str = None, cache_operation: str = None,
                 operation: str = None, context: Dict[str, Any] = None):
        super().__init__(message, operation, context)
        self.cache_key = cache_key
        self.cache_operation = cache_operation

    def __str__(self):
        base_msg = f"CacheError: {self.message}"
        if self.cache_key:
            base_msg += f" (Key: {self.cache_key})"
        if self.cache_operation:
            base_msg += f" (Operation: {self.cache_operation})"
        if self.operation:
            base_msg += f" (Operation: {self.operation})"
        return base_msg

    def get_cache_details(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        details = self.get_context()
        details.update({
            'cache_key': self.cache_key,
            'cache_operation': self.cache_operation
        })
        return details

class MemoryError(PipelineError):
    """Exception raised when memory operations fail."""

    def __init__(self, message: str, memory_usage: float = None, memory_limit: float = None,
                 operation: str = None, context: Dict[str, Any] = None):
        super().__init__(message, operation, context)
        self.memory_usage = memory_usage
        self.memory_limit = memory_limit

    def __str__(self):
        base_msg = f"MemoryError: {self.message}"
        if self.memory_usage is not None:
            base_msg += f" (Usage: {self.memory_usage:.2f} MB)"
        if self.memory_limit is not None:
            base_msg += f" (Limit: {self.memory_limit:.2f} MB)"
        if self.operation:
            base_msg += f" (Operation: {self.operation})"
        return base_msg

    def get_memory_details(self) -> Dict[str, Any]:
        """Get detailed memory information."""
        details = self.get_context()
        details.update({
            'memory_usage': self.memory_usage,
            'memory_limit': self.memory_limit
        })
        return details

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories."""
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    DATA_PROCESSING = "data_processing"
    FEATURE_GENERATION = "feature_generation"
    FILE_IO = "file_io"
    MEMORY = "memory"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"

@dataclass
class ErrorDetails:
    """Detailed error information."""
    error: Exception
    severity: ErrorSeverity
    category: ErrorCategory
    operation: str
    timestamp: str
    context: Dict[str, Any]
    stack_trace: str
    recoverable: bool = False

class AdvancedErrorHandler:
    """
    Advanced error handler for unified pipeline.

    Provides comprehensive error handling with fast failing,
    detailed logging, and utility functions for safe operations.
    """

    def __init__(self, logger=None, component_name: str = "UnifiedDataDrivenPipeline"):
        """Initialize the advanced error handler."""
        tprint(f"🔧 Initializing AdvancedErrorHandler for {component_name}...")
        self.logger = logger or logging.getLogger(__name__)
        self.component_name = component_name
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()

        # Error tracking
        self.error_count = 0
        self.error_history: List[ErrorDetails] = []
        self.max_error_history = 1000

        tprint_success(f"✅ AdvancedErrorHandler initialized for {component_name}")

    def handle_error(self, error: Exception, operation: str,
                    return_value: Any = None, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Handle an error with comprehensive logging and tracking.

        Args:
            error: The exception that occurred
            operation: The operation that failed
            return_value: Value to return if error is handled
            context: Additional context information

        Returns:
            The return_value if provided, otherwise raises the error
        """
        self.error_count += 1

        # Determine error severity and category
        severity = self._classify_error_severity(error)
        category = self._classify_error_category(error)

        # Create error details
        error_details = ErrorDetails(
            error=error,
            severity=severity,
            category=category,
            operation=operation,
            timestamp=datetime.now().isoformat(),
            context=context or {},
            stack_trace=traceback.format_exc(),
            recoverable=self._is_recoverable(error)
        )

        # Add to error history
        self.error_history.append(error_details)
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]

        # Log error based on severity
        self._log_error(error_details)

        # Log error metrics safely
        safe_log_metric(f"error_{severity.value}", 1)
        safe_log_params({
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity.value,
            "category": category.value
        })

        # For critical errors, always raise
        if severity == ErrorSeverity.CRITICAL:
            tprint_error(f"❌ CRITICAL ERROR in {operation}: {str(error)}")
            safe_log_artifact("critical_error", f"critical_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            raise error

        # For high severity errors, raise unless return_value is provided
        if severity == ErrorSeverity.HIGH:
            if return_value is not None:
                tprint_warning(f"⚠️ HIGH SEVERITY ERROR in {operation}: {str(error)} - using return value")
                return return_value
            else:
                tprint_error(f"❌ HIGH SEVERITY ERROR in {operation}: {str(error)}")
                raise error

        # For medium and low severity, use return_value if provided
        if return_value is not None:
            tprint_warning(f"⚠️ {severity.value.upper()} ERROR in {operation}: {str(error)} - using return value")
            return return_value

        # Otherwise raise the error
        tprint_error(f"❌ {severity.value.upper()} ERROR in {operation}: {str(error)}")
        raise error

    def safe_execute(self, func: Callable, *args, return_value: Any = None,
                    operation: str = "unknown", context: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Safely execute a function with error handling.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            return_value: Value to return if function fails
            operation: Operation name for error reporting
            context: Additional context information
            **kwargs: Keyword arguments for the function

        Returns:
            Function result or return_value if error occurs
        """
        tprint_debug(f"🔧 Executing safe operation: {operation}")

        try:
            # Log function execution start
            tprint_debug(f"🚀 Starting {operation} execution")

            # Execute the function
            result = func(*args, **kwargs)

            # Log successful execution
            tprint_debug(f"✅ {operation} executed successfully")

            return result

        except Exception as e:
            tprint_error(f"❌ Error in {operation}: {str(e)}")
            tprint_error(f"❌ Error type: {type(e).__name__}")

            # Log context if available
            if context:
                tprint_debug(f"🔍 Error context: {context}")

            # Handle the error and return appropriate value
            return self.handle_error(e, operation, return_value, context)

    def safe_dataframe_operation(self, operation: str, data: pd.DataFrame,
                               func: Callable, *args, **kwargs) -> pd.DataFrame:
        """
        Safely execute a DataFrame operation using enhanced utilities.

        Args:
            operation: Operation name for error reporting
            data: DataFrame to operate on
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result DataFrame or original DataFrame if error occurs
        """
        tprint_debug(f"🔧 Executing safe DataFrame operation: {operation}")

        # Validate input data
        if data is None:
            tprint_error(f"❌ DataFrame is None for operation: {operation}")
            return pd.DataFrame()

        if data.empty:
            tprint_warning(f"⚠️ DataFrame is empty for operation: {operation}")
            return data

        tprint_debug(f"📊 DataFrame shape: {data.shape}, columns: {list(data.columns)}")

        try:
            # Use utility function for safe operation
            tprint_debug(f"🚀 Starting DataFrame operation: {operation}")
            result = safe_dataframe_operation(data, func, *args, **kwargs)

            # Validate result
            if result is None:
                tprint_warning(f"⚠️ Operation {operation} returned None, returning original data")
                return data

            tprint_success(f"✅ DataFrame operation {operation} completed successfully")
            tprint_debug(f"📊 Result shape: {result.shape}")

            return result

        except Exception as e:
            tprint_error(f"❌ DataFrame operation {operation} failed: {str(e)}")
            tprint_error(f"❌ Error type: {type(e).__name__}")
            tprint_warning(f"⚠️ Returning original DataFrame for operation: {operation}")
            return data

    def safe_numpy_operation(self, operation: str, data: np.ndarray,
                           func: Callable, *args, **kwargs) -> np.ndarray:
        """
        Safely execute a NumPy operation.

        Args:
            operation: Operation name for error reporting
            data: NumPy array to operate on
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result array or original array if error occurs
        """
        tprint_debug(f"🔧 Executing safe NumPy operation: {operation}")

        # Validate input data
        if data is None:
            tprint_error(f"❌ NumPy array is None for operation: {operation}")
            return np.array([])

        if data.size == 0:
            tprint_warning(f"⚠️ NumPy array is empty for operation: {operation}")
            return data

        tprint_debug(f"📊 Array shape: {data.shape}, dtype: {data.dtype}")

        try:
            # Execute the function
            tprint_debug(f"🚀 Starting NumPy operation: {operation}")
            result = func(data, *args, **kwargs)

            # Validate result
            if result is None:
                tprint_warning(f"⚠️ Operation {operation} returned None, returning original array")
                return data

            tprint_success(f"✅ NumPy operation {operation} completed successfully")
            tprint_debug(f"📊 Result shape: {result.shape}")

            return result

        except Exception as e:
            tprint_error(f"❌ NumPy operation {operation} failed: {str(e)}")
            tprint_error(f"❌ Error type: {type(e).__name__}")
            tprint_warning(f"⚠️ Returning original array for operation: {operation}")
            return data

    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on error type and message."""
        error_type = type(error).__name__
        error_message = str(error).lower()

        # Critical errors
        if any(keyword in error_message for keyword in ['critical', 'fatal', 'cannot proceed', 'abort']):
            return ErrorSeverity.CRITICAL

        if error_type in ['MemoryError', 'SystemError', 'KeyboardInterrupt']:
            return ErrorSeverity.CRITICAL

        # High severity errors
        if any(keyword in error_message for keyword in ['failed', 'error', 'exception', 'invalid']):
            return ErrorSeverity.HIGH

        if error_type in ['ValueError', 'TypeError', 'KeyError', 'AttributeError']:
            return ErrorSeverity.HIGH

        # Medium severity errors
        if any(keyword in error_message for keyword in ['warning', 'caution', 'unexpected']):
            return ErrorSeverity.MEDIUM

        if error_type in ['UserWarning', 'DeprecationWarning', 'FutureWarning']:
            return ErrorSeverity.MEDIUM

        # Low severity errors
        return ErrorSeverity.LOW

    def _classify_error_category(self, error: Exception) -> ErrorCategory:
        """Classify error category based on error type and message."""
        error_type = type(error).__name__
        error_message = str(error).lower()

        if 'validation' in error_message or error_type in ['DataValidationError', 'ValidationError']:
            return ErrorCategory.VALIDATION

        if 'optimization' in error_message or error_type in ['OptimizationError']:
            return ErrorCategory.OPTIMIZATION

        if 'feature' in error_message or error_type in ['FeatureGenerationError']:
            return ErrorCategory.FEATURE_GENERATION

        if 'data' in error_message or error_type in ['DataError', 'DataFrameError']:
            return ErrorCategory.DATA_PROCESSING

        if 'cache' in error_message or error_type in ['CacheError']:
            return ErrorCategory.FILE_IO

        if 'memory' in error_message or error_type in ['MemoryError']:
            return ErrorCategory.MEMORY

        if 'network' in error_message or error_type in ['ConnectionError', 'TimeoutError']:
            return ErrorCategory.NETWORK

        if 'config' in error_message or error_type in ['ConfigurationError']:
            return ErrorCategory.CONFIGURATION

        return ErrorCategory.UNKNOWN

    def _is_recoverable(self, error: Exception) -> bool:
        """Determine if an error is recoverable."""
        error_type = type(error).__name__
        error_message = str(error).lower()

        # Non-recoverable errors
        if error_type in ['MemoryError', 'SystemError', 'KeyboardInterrupt']:
            return False

        if any(keyword in error_message for keyword in ['critical', 'fatal', 'cannot proceed']):
            return False

        # Recoverable errors
        if error_type in ['ValueError', 'TypeError', 'KeyError', 'AttributeError']:
            return True

        if any(keyword in error_message for keyword in ['warning', 'caution', 'unexpected']):
            return True

        return True

    def _log_error(self, error_details: ErrorDetails):
        """Log error based on severity."""
        log_message = f"[{error_details.severity.value.upper()}] {error_details.operation}: {str(error_details.error)}"

        if error_details.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra={'error_details': error_details.__dict__})
            tprint_error(f"❌ CRITICAL: {log_message}")
        elif error_details.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, extra={'error_details': error_details.__dict__})
            tprint_error(f"❌ ERROR: {log_message}")
        elif error_details.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, extra={'error_details': error_details.__dict__})
            tprint_warning(f"⚠️ WARNING: {log_message}")
        else:
            self.logger.info(log_message, extra={'error_details': error_details.__dict__})
            tprint_debug(f"ℹ️ INFO: {log_message}")

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {
                'total_errors': 0,
                'error_count': self.error_count,
                'severity_distribution': {},
                'category_distribution': {},
                'recent_errors': []
            }

        # Calculate severity distribution
        severity_counts = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Calculate category distribution
        category_counts = {}
        for error in self.error_history:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1

        # Get recent errors (last 10)
        recent_errors = [
            {
                'operation': error.operation,
                'severity': error.severity.value,
                'category': error.category.value,
                'message': str(error.error),
                'timestamp': error.timestamp
            }
            for error in self.error_history[-10:]
        ]

        return {
            'total_errors': len(self.error_history),
            'error_count': self.error_count,
            'severity_distribution': severity_counts,
            'category_distribution': category_counts,
            'recent_errors': recent_errors
        }

    def reset_error_stats(self):
        """Reset error statistics."""
        self.error_count = 0
        self.error_history = []
        tprint_success("✅ Error statistics reset")

    def safe_dataframe_conversion(self, data: pd.DataFrame, dtype_mapping: Dict[str, str]) -> pd.DataFrame:
        """Safely convert DataFrame column dtypes using utilities."""
        return self.safe_execute(
            safe_convert_dtypes, data, dtype_mapping,
            operation="dataframe_dtype_conversion",
            return_value=data
        )

    def safe_dataframe_quality_assessment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Safely assess DataFrame quality using utilities."""
        return self.safe_execute(
            calculate_data_quality_metrics, data,
            operation="dataframe_quality_assessment",
            return_value={}
        )

    def safe_dataframe_optimization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Safely optimize DataFrame using utilities."""
        return self.safe_execute(
            optimize_dataframe_dtypes, data,
            operation="dataframe_optimization",
            return_value=data
        )

    def safe_dataframe_null_handling(self, data: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Safely handle null values in DataFrame using utilities."""
        return self.safe_execute(
            guard_dataframe_nulls, data, threshold,
            operation="dataframe_null_handling",
            return_value=data
        )

    def safe_dataframe_merge(self, df1: pd.DataFrame, df2: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Safely merge DataFrames using utilities."""
        return self.safe_execute(
            safe_merge_dataframes, df1, df2, **kwargs,
            operation="dataframe_merge",
            return_value=df1
        )

    def safe_dataframe_drop_columns(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Safely drop columns using utilities."""
        return self.safe_execute(
            safe_drop_columns, data, columns,
            operation="dataframe_drop_columns",
            return_value=data
        )

    def safe_dataframe_rename_columns(self, data: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Safely rename columns using utilities."""
        return self.safe_execute(
            safe_rename_columns, data, column_mapping,
            operation="dataframe_rename_columns",
            return_value=data
        )

    def safe_dataframe_filter(self, data: pd.DataFrame, condition: str) -> pd.DataFrame:
        """Safely filter DataFrame using utilities."""
        return self.safe_execute(
            safe_filter_dataframe, data, condition,
            operation="dataframe_filter",
            return_value=data
        )

    def safe_dataframe_groupby(self, data: pd.DataFrame, group_cols: List[str], agg_dict: Dict[str, str]) -> pd.DataFrame:
        """Safely perform groupby operation using utilities."""
        return self.safe_execute(
            safe_groupby_operation, data, group_cols, agg_dict,
            operation="dataframe_groupby",
            return_value=data
        )

    def safe_dataframe_apply(self, data: pd.DataFrame, func: Callable, axis: int = 0) -> pd.DataFrame:
        """Safely apply function to DataFrame using utilities."""
        return self.safe_execute(
            safe_apply_function, data, func, axis,
            operation="dataframe_apply",
            return_value=data
        )

    def get_dataframe_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive DataFrame summary using utilities."""
        return self.safe_execute(
            get_dataframe_info, data,
            operation="dataframe_summary",
            return_value={}
        )

    def get_dataframe_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get DataFrame statistics using utilities."""
        return self.safe_execute(
            create_summary_statistics, data,
            operation="dataframe_statistics",
            return_value={}
        )

def error_handler_decorator(operation: str, return_value: Any = None,
                          context: Optional[Dict[str, Any]] = None):
    """
    Decorator for automatic error handling.

    Args:
        operation: Operation name for error reporting
        return_value: Value to return if error occurs
        context: Additional context information
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get error handler from self if available
            error_handler = None
            if args and hasattr(args[0], 'error_handler'):
                error_handler = args[0].error_handler

            if error_handler:
                return error_handler.safe_execute(func, *args, return_value=return_value,
                                                operation=operation, context=context, **kwargs)
            else:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tprint_error(f"❌ Unhandled error in {operation}: {str(e)}")
                    if return_value is not None:
                        return return_value
                    raise

        return wrapper
    return decorator
