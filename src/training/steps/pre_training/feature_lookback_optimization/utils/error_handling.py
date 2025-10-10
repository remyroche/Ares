"""
Standardized Error Handling Utilities for Feature Lookback Optimization.

This module provides consistent error handling patterns across the optimization system.
"""

import logging
from typing import Any, Callable, Optional, Type, Union
from functools import wraps
import traceback
import numpy as np
import pandas as pd

# Import tprint for enhanced logging
try:
    from src.utils.tprint import tprint_error, tprint_warning, tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)


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


def safe_operation(
    operation_name: str,
    default_value: Any = None,
    log_errors: bool = True,
    reraise: bool = False,
    expected_exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for safe operation execution with standardized error handling.
    
    Args:
        operation_name: Name of the operation for logging
        default_value: Value to return on error
        log_errors: Whether to log errors
        reraise: Whether to reraise exceptions
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
        
    except DataValidationError as e:
        tprint_warning(f"⚠️ Data validation failed in MI calculation: {e}")
        return default_value
    except ScoringError as e:
        tprint_warning(f"⚠️ Scoring error in MI calculation: {e}")
        return default_value
    except Exception as e:
        tprint_error(f"❌ Unexpected error in MI calculation: {e}")
        return default_value


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
        
    except DataValidationError as e:
        tprint_warning(f"⚠️ Data validation failed in correlation calculation: {e}")
        return default_value
    except ScoringError as e:
        tprint_warning(f"⚠️ Scoring error in correlation calculation: {e}")
        return default_value
    except Exception as e:
        tprint_error(f"❌ Unexpected error in correlation calculation: {e}")
        return default_value


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
        
    except DataValidationError as e:
        tprint_warning(f"⚠️ Data validation failed in {operation_name}: {e}")
        return default_value
    except Exception as e:
        tprint_error(f"❌ Error in {operation_name}: {e}")
        return default_value


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
        
    except DataValidationError as e:
        tprint_warning(f"⚠️ Data validation failed in {operation_name}: {e}")
        return default_value
    except ScoringError as e:
        tprint_warning(f"⚠️ Scoring error in {operation_name}: {e}")
        return default_value
    except Exception as e:
        tprint_error(f"❌ Error in {operation_name}: {e}")
        return default_value


class ErrorHandler:
    """Centralized error handler for optimization operations."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
        self.error_history = []
    
    def handle_error(
        self, 
        error: Exception, 
        context: str, 
        operation: str,
        reraise: bool = False
    ) -> Any:
        """
        Handle errors with standardized logging and tracking.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
            operation: Operation that failed
            reraise: Whether to reraise the exception
            
        Returns:
            None or reraised exception
        """
        # Track error
        error_key = f"{context}:{operation}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log error
        self.logger.error(f"❌ Error in {context} during {operation}: {error}")
        self.logger.debug(f"   → Error type: {type(error).__name__}")
        self.logger.debug(f"   → Traceback: {traceback.format_exc()}")
        
        # Store in history
        self.error_history.append({
            'timestamp': pd.Timestamp.now(),
            'context': context,
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error)
        })
        
        # Keep only recent errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]
        
        if reraise:
            raise
        
        return None
    
    def get_error_summary(self) -> dict:
        """Get summary of errors encountered."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'unique_error_types': len(self.error_counts),
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
    
    def reset_error_tracking(self):
        """Reset error tracking statistics."""
        self.error_counts.clear()
        self.error_history.clear()


# Global error handler instance
_global_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler