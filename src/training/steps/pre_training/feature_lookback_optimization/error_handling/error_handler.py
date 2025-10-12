"""
Error Handling Framework for Feature Lookback Optimization.

This module provides standardized error handling with fast failing,
detailed logging, and utility functions for safe operations.
"""

import logging
import traceback
from typing import Any, Optional, Dict, List, Callable, Type, Union
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import numpy as np
import pandas as pd

# Import utility modules
from src.utils.common_utilities import CommonUtilities
from src.utils.serialization_utils import UniversalSerializer
from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_success, tprint_debug


# Exception classes for fast failing
class OptimizationError(Exception):
    """Base exception for optimization-related errors."""
    pass


class DataValidationError(OptimizationError):
    """Exception raised when data validation fails."""
    pass


class ScoringError(OptimizationError):
    """Exception raised when scoring calculations fail."""
    pass


class CacheError(OptimizationError):
    """Exception raised when cache operations fail."""
    pass


class MemoryError(OptimizationError):
    """Exception raised when memory operations fail."""
    pass


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


# Removed ErrorRecoveryResult - not needed for fast failing


class StandardizedErrorHandler:
    """
    Standardized error handler with fast failing.

    Provides consistent error handling across the feature lookback optimization
    component with immediate failure propagation and detailed logging.
    """

    def __init__(self, logger=None, component_name: str = "FeatureLookbackOptimization"):
        """Initialize the error handler."""
        self.logger = logger or logging.getLogger(__name__)
        self.component_name = component_name
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()

        # Error tracking
        self.error_counts = {}
        self.recent_errors = []

    def handle_error(
        self,
        error: Exception,
        operation: str,
        return_value: Any = None,
        reraise: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Handle errors with configurable behavior.

        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            return_value: Value to return if reraise=False
            reraise: Whether to re-raise the exception (default: True for fast failing)
            context: Additional context information

        Returns:
            return_value if reraise=False, otherwise raises the error
        """
        try:
            # Create error details
            error_details = self._create_error_details(error, operation, context)

            # Log the error with tprint
            self._log_error(error_details)
            tprint_error(f"❌ Error in {operation}: {str(error_details.error)}")
            if error_details.severity == ErrorSeverity.CRITICAL:
                tprint_error(f"🚨 Critical error - failing immediately: {operation}")

            # Track error statistics
            self._track_error(error_details)

            # Update recent errors
            self.recent_errors.append(error_details)
            if len(self.recent_errors) > 100:  # Keep only recent errors
                self.recent_errors = self.recent_errors[-100:]

            # Respect the reraise parameter
            if reraise:
                raise error
            else:
                tprint_warning(f"⚠️ Error suppressed in {operation}, returning fallback value")
                return return_value

        except Exception as e:
            self.logger.critical(f"Error handler failed: {e}")
            tprint_error(f"🚨 Error handler itself failed: {e}")
            # Always re-raise the original error if handler fails
            raise error

    def handle_warning(self, warning_msg: str, operation: str, context: Optional[Dict[str, Any]] = None):
        """Handle warnings in a standardized way."""
        try:
            self.logger.warning(f"[{self.component_name}] {operation}: {warning_msg}")
            if context:
                self.logger.debug(f"Warning context: {context}")
        except Exception as e:
            self.logger.error(f"Failed to handle warning: {e}")

    def handle_info(self, info_msg: str, operation: str, context: Optional[Dict[str, Any]] = None):
        """Handle info messages in a standardized way."""
        try:
            self.logger.info(f"[{self.component_name}] {operation}: {info_msg}")
            if context:
                self.logger.debug(f"Info context: {context}")
        except Exception as e:
            self.logger.error(f"Failed to handle info message: {e}")

    def _create_error_details(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorDetails:
        """Create detailed error information."""
        import datetime

        # Determine severity and category
        severity = self._classify_error_severity(error, operation)
        category = self._classify_error_category(error, operation)

        # Determine if error is recoverable
        recoverable = self._is_error_recoverable(error, operation, category)

        return ErrorDetails(
            error=error,
            severity=severity,
            category=category,
            operation=operation,
            timestamp=datetime.datetime.now().isoformat(),
            context=context or {},
            stack_trace=traceback.format_exc(),
            recoverable=recoverable
        )

    def _classify_error_severity(self, error: Exception, operation: str) -> ErrorSeverity:
        """Classify error severity based on error type and operation."""
        error_type = type(error).__name__

        # Critical errors
        critical_operations = ['execute', 'optimize', 'validate_data']
        if operation in critical_operations:
            return ErrorSeverity.CRITICAL

        # High severity errors
        high_severity_types = ['ValueError', 'TypeError', 'KeyError']
        if error_type in high_severity_types:
            return ErrorSeverity.HIGH

        # Medium severity errors
        medium_severity_types = ['AttributeError', 'IndexError']
        if error_type in medium_severity_types:
            return ErrorSeverity.MEDIUM

        # Default to low severity
        return ErrorSeverity.LOW

    def _classify_error_category(self, error: Exception, operation: str) -> ErrorCategory:
        """Classify error category based on error type and operation."""
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # Validation errors
        if 'validation' in operation.lower() or 'validate' in operation.lower():
            return ErrorCategory.VALIDATION

        # Data processing errors
        if any(term in error_msg for term in ['data', 'column', 'dataframe', 'series']):
            return ErrorCategory.DATA_PROCESSING

        # Memory errors
        if any(term in error_msg for term in ['memory', 'out of memory', 'allocation']):
            return ErrorCategory.MEMORY

        # File I/O errors
        if any(term in error_msg for term in ['file', 'io', 'read', 'write', 'path']):
            return ErrorCategory.FILE_IO

        # Configuration errors
        if any(term in error_msg for term in ['config', 'parameter', 'setting']):
            return ErrorCategory.CONFIGURATION

        # Network errors
        if any(term in error_msg for term in ['network', 'connection', 'timeout', 'http']):
            return ErrorCategory.NETWORK

        # Optimization errors
        if any(term in operation.lower() for term in ['optimize', 'lookback', 'feature']):
            return ErrorCategory.OPTIMIZATION

        # Default to unknown
        return ErrorCategory.UNKNOWN

    def _is_error_recoverable(self, error: Exception, operation: str, category: ErrorCategory) -> bool:
        """Determine if an error is recoverable - always False for fast failing."""
        # Fast failing: all errors are non-recoverable
        return False

    def _log_error(self, error_details: ErrorDetails):
        """Log error with appropriate level."""
        log_message = f"[{self.component_name}] {error_details.operation}: {error_details.error}"

        if error_details.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            self.logger.critical(f"Stack trace: {error_details.stack_trace}")
        elif error_details.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
            self.logger.debug(f"Stack trace: {error_details.stack_trace}")
        elif error_details.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
            self.logger.debug(f"Context: {error_details.context}")
        else:
            self.logger.info(log_message)

    def _track_error(self, error_details: ErrorDetails):
        """Track error statistics."""
        # Track by category
        category_key = f"category_{error_details.category.value}"
        self.error_counts[category_key] = self.error_counts.get(category_key, 0) + 1

        # Track by severity
        severity_key = f"severity_{error_details.severity.value}"
        self.error_counts[severity_key] = self.error_counts.get(severity_key, 0) + 1

        # Track by operation
        operation_key = f"operation_{error_details.operation}"
        self.error_counts[operation_key] = self.error_counts.get(operation_key, 0) + 1

    # Removed all recovery methods - fast failing doesn't need them

    def get_error_statistics(self) -> Dict[str, int]:
        """Get error statistics."""
        return self.error_counts.copy()

    def get_recent_errors(self, limit: int = 10) -> List[ErrorDetails]:
        """Get recent errors."""
        return self.recent_errors[-limit:].copy()

    def reset_error_tracking(self):
        """Reset error tracking statistics."""
        self.error_counts.clear()
        self.recent_errors.clear()


# Utility functions for safe operations
def safe_operation(
    operation_name: str,
    default_value: Any = None,
    log_errors: bool = True,
    reraise: bool = True,  # Changed to True for fast failing
    expected_exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for safe operation execution with standardized error handling.
    
    Args:
        operation_name: Name of the operation for logging
        default_value: Value to return on error (only if reraise=False)
        log_errors: Whether to log errors
        reraise: Whether to reraise exceptions (default: True for fast failing)
        expected_exceptions: Tuple of exception types to catch
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except expected_exceptions as e:
                if log_errors:
                    tprint_error(f"❌ {operation_name} failed: {e}")
                    tprint_debug(f"   → Function: {func.__name__}")
                    tprint_debug(f"   → Args: {len(args)} positional, {len(kwargs)} keyword")
                    tprint_debug(f"   → Traceback: {traceback.format_exc()}")
                
                if reraise:
                    raise
                
                return default_value
            except Exception as e:
                if log_errors:
                    tprint_error(f"❌ Unexpected error in {operation_name}: {e}")
                    tprint_debug(f"   → Function: {func.__name__}")
                    tprint_debug(f"   → Traceback: {traceback.format_exc()}")
                
                if reraise:
                    raise OptimizationError(f"Unexpected error in {operation_name}: {e}") from e
                
                return default_value
        
        return wrapper
    return decorator


def safe_mi_calculation(
    feature_values: np.ndarray, 
    target_values: np.ndarray,
    default_value: float = 0.0
) -> float:
    """
    Safely calculate mutual information with standardized error handling.
    
    Args:
        feature_values: Feature values array
        target_values: Target values array
        default_value: Default value to return on error
        
    Returns:
        Mutual information score or default value
    """
    try:
        # Validate inputs
        if not isinstance(feature_values, np.ndarray) or not isinstance(target_values, np.ndarray):
            raise DataValidationError("Inputs must be numpy arrays")
        
        if len(feature_values) != len(target_values):
            raise DataValidationError("Feature and target arrays must have same length")
        
        if len(feature_values) < 2:
            raise DataValidationError("Need at least 2 data points for MI calculation")
        
        # Remove NaN values
        valid_mask = ~(np.isnan(feature_values) | np.isnan(target_values))
        if not np.any(valid_mask):
            raise DataValidationError("No valid data points after NaN removal")
        
        feature_clean = feature_values[valid_mask]
        target_clean = target_values[valid_mask]
        
        if len(feature_clean) < 2:
            raise DataValidationError("Insufficient valid data points")
        
        # Calculate mutual information
        from sklearn.feature_selection import mutual_info_regression
        mi_scores = mutual_info_regression(
            feature_clean.reshape(-1, 1), 
            target_clean,
            random_state=42
        )
        
        if len(mi_scores) == 0:
            raise ScoringError("MI calculation returned empty result")
        
        mi_score = float(mi_scores[0])
        
        # Validate result
        if not np.isfinite(mi_score):
            raise ScoringError(f"MI score is not finite: {mi_score}")
        
        return max(0.0, mi_score)  # Ensure non-negative
        
    except (DataValidationError, ScoringError) as e:
        tprint_warning(f"⚠️ Error in MI calculation: {e}")
        if default_value is not None:
            return default_value
        else:
            raise  # Fast fail
    except Exception as e:
        tprint_error(f"❌ Unexpected error in MI calculation: {e}")
        raise OptimizationError(f"Unexpected error in MI calculation: {e}") from e


def safe_correlation_calculation(
    x: np.ndarray, 
    y: np.ndarray,
    default_value: float = 0.0
) -> float:
    """
    Safely calculate correlation with standardized error handling.
    
    Args:
        x: First array
        y: Second array
        default_value: Default value to return on error
        
    Returns:
        Correlation coefficient or default value
    """
    try:
        # Validate inputs
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise DataValidationError("Inputs must be numpy arrays")
        
        if len(x) != len(y):
            raise DataValidationError("Arrays must have same length")
        
        if len(x) < 2:
            raise DataValidationError("Need at least 2 data points for correlation")
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if not np.any(valid_mask):
            raise DataValidationError("No valid data points after NaN removal")
        
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) < 2:
            raise DataValidationError("Insufficient valid data points")
        
        # Calculate correlation
        correlation = np.corrcoef(x_clean, y_clean)[0, 1]
        
        # Validate result
        if not np.isfinite(correlation):
            raise ScoringError(f"Correlation is not finite: {correlation}")
        
        return float(correlation)
        
    except (DataValidationError, ScoringError) as e:
        tprint_warning(f"⚠️ Error in correlation calculation: {e}")
        if default_value is not None:
            return default_value
        else:
            raise  # Fast fail
    except Exception as e:
        tprint_error(f"❌ Unexpected error in correlation calculation: {e}")
        raise OptimizationError(f"Unexpected error in correlation calculation: {e}") from e


def safe_dataframe_operation(
    operation_name: str,
    df: pd.DataFrame,
    operation: Callable[[pd.DataFrame], Any],
    default_value: Any = None
) -> Any:
    """
    Safely perform DataFrame operations with standardized error handling.
    
    Args:
        operation_name: Name of the operation for logging
        df: DataFrame to operate on
        operation: Function to apply to DataFrame
        default_value: Default value to return on error
        
    Returns:
        Operation result or default value
    """
    try:
        # Validate DataFrame
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise DataValidationError("DataFrame is empty")
        
        # Perform operation
        result = operation(df)
        
        return result
        
    except (DataValidationError, ScoringError) as e:
        tprint_warning(f"⚠️ Error in {operation_name}: {e}")
        if default_value is not None:
            return default_value
        else:
            raise  # Fast fail
    except Exception as e:
        tprint_error(f"❌ Error in {operation_name}: {e}")
        raise OptimizationError(f"Error in {operation_name}: {e}") from e


def safe_numpy_operation(
    operation_name: str,
    arrays: list,
    operation: Callable,
    default_value: Any = None
) -> Any:
    """
    Safely perform NumPy operations with standardized error handling.
    
    Args:
        operation_name: Name of the operation for logging
        arrays: List of arrays to operate on
        operation: Function to apply to arrays
        default_value: Default value to return on error
        
    Returns:
        Operation result or default value
    """
    try:
        # Validate arrays
        for i, arr in enumerate(arrays):
            if not isinstance(arr, np.ndarray):
                raise DataValidationError(f"Array {i} must be a numpy array")
            
            if arr.size == 0:
                raise DataValidationError(f"Array {i} is empty")
        
        # Perform operation
        result = operation(*arrays)
        
        # Validate result
        if isinstance(result, np.ndarray) and not np.any(np.isfinite(result)):
            raise ScoringError(f"Operation {operation_name} produced non-finite result")
        
        return result
        
    except (DataValidationError, ScoringError) as e:
        tprint_warning(f"⚠️ Error in {operation_name}: {e}")
        if default_value is not None:
            return default_value
        else:
            raise  # Fast fail
    except Exception as e:
        tprint_error(f"❌ Error in {operation_name}: {e}")
        raise OptimizationError(f"Error in {operation_name}: {e}") from e


# Global error handler instance
_global_error_handler = None


def get_error_handler() -> 'StandardizedErrorHandler':
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = StandardizedErrorHandler()
    return _global_error_handler
