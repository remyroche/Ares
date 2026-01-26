"""
SR Error Handlers Module

Provides comprehensive error handling, validation, and monitoring functionality
for the SR (Sharpe Ratio) clustering system.

This module includes:
- Error handling decorators for SR detection operations
- Data validation functions and decorators
- Performance monitoring utilities
- Custom exception classes for SR operations
- Error recovery and fallback mechanisms
"""

import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np


class SRErrorSeverity(Enum):
    """Severity levels for SR-related errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SRErrorCategory(Enum):
    """Categories for SR-related errors."""
    DATA_VALIDATION = "data_validation"
    DETECTION = "detection"
    OPTIMIZATION = "optimization"
    CONFIGURATION = "configuration"
    PERFORMANCE = "performance"
    SYSTEM = "system"


# Custom exception classes for SR operations
class SRError(Exception):
    """Base exception class for SR-related errors."""
    
    def __init__(self, message: str, severity: SRErrorSeverity = SRErrorSeverity.MEDIUM,
                 category: SRErrorCategory = SRErrorCategory.SYSTEM,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.details = details or {}
        self.timestamp = time.time()
    
    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.category.value}: {self.message}"


class SRDataError(SRError):
    """Exception raised for SR data-related issues."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            severity=SRErrorSeverity.HIGH,
            category=SRErrorCategory.DATA_VALIDATION,
            details=details
        )


class SROptimizationError(SRError):
    """Exception raised for SR optimization-related issues."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            severity=SRErrorSeverity.MEDIUM,
            category=SRErrorCategory.OPTIMIZATION,
            details=details
        )


class SRConfigurationError(SRError):
    """Exception raised for SR configuration-related issues."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            severity=SRErrorSeverity.HIGH,
            category=SRErrorCategory.CONFIGURATION,
            details=details
        )


class SRPerformanceError(SRError):
    """Exception raised for SR performance-related issues."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            severity=SRErrorSeverity.MEDIUM,
            category=SRErrorCategory.PERFORMANCE,
            details=details
        )


@dataclass
class SRErrorContext:
    """Context information for SR errors."""
    operation: str
    data_shape: Optional[tuple] = None
    parameters: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None


@dataclass
class SRValidationResult:
    """Result of SR data validation."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    quality_score: float
    recommendations: List[str]


# Global logger for SR error handling
sr_logger = logging.getLogger('sr_error_handlers')


def handles_sr_detection_errors(
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = False
) -> Callable:
    """
    Decorator for handling SR detection errors with comprehensive logging and recovery.
    
    Args:
        default_return: Default value to return on error
        log_errors: Whether to log errors
        reraise: Whether to re-raise exceptions after handling
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            context = SRErrorContext(
                operation=func.__name__,
                parameters=kwargs
            )
            
            try:
                result = func(*args, **kwargs)
                context.execution_time = time.time() - start_time
                return result
                
            except Exception as e:
                context.execution_time = time.time() - start_time
                
                # Create appropriate error based on exception type
                if isinstance(e, SRError):
                    error = e
                else:
                    error = SROptimizationError(
                        message=f"SR detection error in {func.__name__}: {str(e)}",
                        details={
                            'original_exception': str(e),
                            'traceback': traceback.format_exc(),
                            'context': context.__dict__
                        }
                    )
                
                if log_errors:
                    sr_logger.error(
                        f"SR detection error in {func.__name__}: {error}",
                        extra={'context': context.__dict__, 'error_details': error.details}
                    )
                
                if reraise:
                    raise error
                    
                return default_return
                
        return wrapper
    return decorator


def handles_sr_data_validation(
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = False
) -> Callable:
    """
    Decorator for handling SR data validation errors.
    
    Args:
        default_return: Default value to return on error
        log_errors: Whether to log errors
        reraise: Whether to re-raise exceptions after handling
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            context = SRErrorContext(
                operation=func.__name__,
                parameters=kwargs
            )
            
            try:
                result = func(*args, **kwargs)
                context.execution_time = time.time() - start_time
                return result
                
            except Exception as e:
                context.execution_time = time.time() - start_time
                
                # Create appropriate error based on exception type
                if isinstance(e, SRError):
                    error = e
                else:
                    error = SRDataError(
                        message=f"SR data validation error in {func.__name__}: {str(e)}",
                        details={
                            'original_exception': str(e),
                            'traceback': traceback.format_exc(),
                            'context': context.__dict__
                        }
                    )
                
                if log_errors:
                    sr_logger.error(
                        f"SR data validation error in {func.__name__}: {error}",
                        extra={'context': context.__dict__, 'error_details': error.details}
                    )
                
                if reraise:
                    raise error
                    
                return default_return
                
        return wrapper
    return decorator


def monitors_sr_performance(
    log_slow_operations: bool = True,
    performance_threshold: float = 1.0
) -> Callable:
    """
    Decorator for monitoring SR operation performance.
    
    Args:
        log_slow_operations: Whether to log slow operations
        performance_threshold: Threshold in seconds for considering operation slow
        
    Returns:
        Decorated function with performance monitoring
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            start_memory = _get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                end_memory = _get_memory_usage()
                memory_delta = end_memory - start_memory
                
                # Log performance metrics
                if log_slow_operations and execution_time > performance_threshold:
                    sr_logger.warning(
                        f"Slow SR operation detected: {func.__name__} took {execution_time:.3f}s "
                        f"(threshold: {performance_threshold:.3f}s)"
                    )
                
                sr_logger.debug(
                    f"SR performance: {func.__name__} - Time: {execution_time:.3f}s, "
                    f"Memory delta: {memory_delta:.2f}MB"
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                sr_logger.error(
                    f"SR operation failed: {func.__name__} after {execution_time:.3f}s - {str(e)}"
                )
                raise
                
        return wrapper
    return decorator


def validates_sr_output(
    validate_types: bool = True,
    validate_values: bool = True,
    log_validation: bool = True
) -> Callable:
    """
    Decorator for validating SR operation outputs.
    
    Args:
        validate_types: Whether to validate output types
        validate_values: Whether to validate output values
        log_validation: Whether to log validation results
        
    Returns:
        Decorated function with output validation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)
                
                if validate_types:
                    _validate_output_types(result, func.__name__)
                
                if validate_values:
                    _validate_output_values(result, func.__name__)
                
                if log_validation:
                    sr_logger.debug(f"SR output validation passed for {func.__name__}")
                
                return result
                
            except Exception as e:
                sr_logger.error(f"SR output validation failed for {func.__name__}: {str(e)}")
                raise SRDataError(f"Output validation failed: {str(e)}")
                
        return wrapper
    return decorator


def validate_sr_data(
    data: pd.DataFrame,
    min_rows: int = 10,
    required_columns: Optional[List[str]] = None,
    check_nulls: bool = True,
    check_outliers: bool = True,
    outlier_threshold: float = 3.0
) -> SRValidationResult:
    """
    Validate SR input data for quality and consistency.
    
    Args:
        data: DataFrame to validate
        min_rows: Minimum number of rows required
        required_columns: List of required column names
        check_nulls: Whether to check for null values
        check_outliers: Whether to check for outliers
        outlier_threshold: Standard deviation threshold for outlier detection
        
    Returns:
        SRValidationResult with validation details
    """
    issues = []
    warnings = []
    quality_score = 1.0
    recommendations = []
    
    try:
        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            issues.append("Input data is not a pandas DataFrame")
            quality_score *= 0.1
            return SRValidationResult(False, issues, warnings, quality_score, recommendations)
        
        # Check minimum rows
        if len(data) < min_rows:
            issues.append(f"Insufficient data: {len(data)} rows (minimum: {min_rows})")
            quality_score *= 0.3
        
        # Check required columns
        if required_columns:
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")
                quality_score *= 0.5
        
        # Check for null values
        if check_nulls:
            null_counts = data.isnull().sum()
            high_null_cols = [col for col, count in null_counts.items() if count > len(data) * 0.05]
            if high_null_cols:
                warnings.append(f"High null values in columns: {high_null_cols}")
                quality_score *= 0.9
        
        # Check for outliers
        if check_outliers and len(data) > 0:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in data.columns:
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    if std_val > 0:
                        outliers = data[np.abs(data[col] - mean_val) > outlier_threshold * std_val]
                        if len(outliers) > 0:
                            outlier_pct = len(outliers) / len(data) * 100
                            if outlier_pct > 5:  # More than 5% outliers
                                warnings.append(f"High outlier percentage in {col}: {outlier_pct:.1f}%")
                                quality_score *= 0.95
        
        # Generate recommendations
        if quality_score < 0.7:
            recommendations.append("Consider data cleaning and preprocessing")
        if len(data) < min_rows * 2:
            recommendations.append("Consider collecting more data for better results")
        
        is_valid = len(issues) == 0
        
        sr_logger.debug(f"SR data validation completed - Valid: {is_valid}, Score: {quality_score:.3f}")
        
        return SRValidationResult(is_valid, issues, warnings, quality_score, recommendations)
        
    except Exception as e:
        error_msg = f"SR data validation failed: {str(e)}"
        sr_logger.error(error_msg)
        return SRValidationResult(
            False, [error_msg], [], 0.0, ["Fix validation logic"]
        )


def validate_sr_parameters(
    params: Dict[str, Any],
    required_params: Optional[List[str]] = None,
    param_ranges: Optional[Dict[str, tuple]] = None
) -> SRValidationResult:
    """
    Validate SR operation parameters.
    
    Args:
        params: Parameter dictionary to validate
        required_params: List of required parameter names
        param_ranges: Dictionary of parameter ranges (min, max)
        
    Returns:
        SRValidationResult with validation details
    """
    issues = []
    warnings = []
    quality_score = 1.0
    recommendations = []
    
    try:
        # Check required parameters
        if required_params:
            missing_params = [p for p in required_params if p not in params]
            if missing_params:
                issues.append(f"Missing required parameters: {missing_params}")
                quality_score *= 0.3
        
        # Check parameter ranges
        if param_ranges:
            for param_name, (min_val, max_val) in param_ranges.items():
                if param_name in params:
                    value = params[param_name]
                    if value is not None and not isinstance(value, (int, float)):
                        issues.append(f"Parameter {param_name} must be numeric")
                        quality_score *= 0.5
                    elif value is not None and (value < min_val or value > max_val):
                        issues.append(f"Parameter {param_name} out of range: {value} (expected: {min_val}-{max_val})")
                        quality_score *= 0.7
        
        # Check for common parameter issues
        for param_name, value in params.items():
            if value is None:
                warnings.append(f"Parameter {param_name} is None")
                quality_score *= 0.95
        
        is_valid = len(issues) == 0
        
        sr_logger.debug(f"SR parameter validation completed - Valid: {is_valid}, Score: {quality_score:.3f}")
        
        return SRValidationResult(is_valid, issues, warnings, quality_score, recommendations)
        
    except Exception as e:
        error_msg = f"SR parameter validation failed: {str(e)}"
        sr_logger.error(error_msg)
        return SRValidationResult(
            False, [error_msg], [], 0.0, ["Fix parameter validation logic"]
        )


def handle_sr_error(
    error: Exception,
    context: Optional[SRErrorContext] = None,
    fallback_strategy: str = "default",
    recovery_attempts: int = 3
) -> Any:
    """
    Handle SR errors with appropriate recovery strategies.
    
    Args:
        error: The exception that occurred
        context: Error context information
        fallback_strategy: Recovery strategy to use
        recovery_attempts: Number of recovery attempts
        
    Returns:
        Appropriate fallback result or re-raises exception
    """
    try:
        sr_logger.error(f"Handling SR error: {str(error)}")
        
        # Log error details
        if isinstance(error, SRError):
            sr_logger.error(f"SR Error Details: {error.details}")
        
        # Attempt recovery based on strategy
        if fallback_strategy == "ignore":
            sr_logger.warning("Using ignore strategy for SR error")
            return None
            
        elif fallback_strategy == "retry":
            sr_logger.info(f"Using retry strategy (attempts: {recovery_attempts})")
            # Note: Actual retry logic would need to be implemented by caller
            return None
            
        elif fallback_strategy == "fallback":
            sr_logger.warning("Using fallback strategy for SR error")
            return _get_fallback_result(error, context)
            
        else:  # default strategy
            sr_logger.error("No recovery strategy specified, re-raising error")
            raise error
            
    except Exception as recovery_error:
        sr_logger.error(f"Error recovery failed: {str(recovery_error)}")
        raise error  # Re-raise original error


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def _validate_output_types(result: Any, operation_name: str) -> None:
    """Validate the types of SR operation outputs."""
    if result is None:
        raise ValueError(f"Operation {operation_name} returned None")
    
    # Add more specific type validations based on expected outputs
    # This is a basic implementation that can be extended
    pass


def _validate_output_values(result: Any, operation_name: str) -> None:
    """Validate the values of SR operation outputs."""
    if isinstance(result, dict):
        # Check for common SR output structure
        if 'levels' in result and result['levels'] is not None:
            if not isinstance(result['levels'], (list, dict)):
                raise ValueError(f"Operation {operation_name} returned invalid levels type")
    # Add more specific value validations based on expected outputs
    # This is a basic implementation that can be extended
    pass


def _get_fallback_result(error: Exception, context: Optional[SRErrorContext]) -> Any:
    """Get appropriate fallback result based on error type."""
    if isinstance(error, SRDataError):
        return {'levels': [], 'error': 'data_validation_failed'}
    elif isinstance(error, SROptimizationError):
        return {'levels': [], 'error': 'optimization_failed'}
    elif isinstance(error, SRConfigurationError):
        return {'levels': [], 'error': 'configuration_failed'}
    else:
        return {'levels': [], 'error': 'unknown_error'}


# Utility functions for error reporting and analysis
def create_error_report(
    errors: List[Exception],
    context: Optional[SRErrorContext] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive error report from a list of errors.
    
    Args:
        errors: List of exceptions to analyze
        context: Error context information
        
    Returns:
        Dictionary with error analysis and recommendations
    """
    error_types = {}
    severity_counts = {}
    category_counts = {}
    
    for error in errors:
        # Count error types
        error_type = type(error).__name__
        error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Count severities
        if isinstance(error, SRError):
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
    
    return {
        'total_errors': len(errors),
        'error_types': error_types,
        'severity_distribution': severity_counts,
        'category_distribution': category_counts,
        'context': context.__dict__ if context else {},
        'recommendations': _generate_error_recommendations(error_types, severity_counts)
    }


def _generate_error_recommendations(
    error_types: Dict[str, int],
    severity_counts: Dict[str, int]
) -> List[str]:
    """Generate recommendations based on error analysis."""
    recommendations = []
    
    # High severity errors recommendations
    if severity_counts.get('high', 0) > 0 or severity_counts.get('critical', 0) > 0:
        recommendations.append("Review and fix high-priority errors immediately")
    
    # Data validation errors recommendations
    if error_types.get('SRDataError', 0) > 0:
        recommendations.append("Implement robust data validation and cleaning")
    
    # Configuration errors recommendations
    if error_types.get('SRConfigurationError', 0) > 0:
        recommendations.append("Review and validate configuration parameters")
    
    # Optimization errors recommendations
    if error_types.get('SROptimizationError', 0) > 0:
        recommendations.append("Check optimization parameters and constraints")
    
    return recommendations


# Performance monitoring utilities
class SRPerformanceMonitor:
    """Monitor and track SR operation performance."""
    
    def __init__(self):
        self.operations = {}
        self.start_times = {}
    
    def start_operation(self, operation_name: str) -> None:
        """Start monitoring an operation."""
        self.start_times[operation_name] = time.time()
        sr_logger.debug(f"Started monitoring SR operation: {operation_name}")
    
    def end_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """End monitoring an operation and record performance."""
        if operation_name in self.start_times:
            execution_time = time.time() - self.start_times[operation_name]
            
            if operation_name not in self.operations:
                self.operations[operation_name] = []
            
            self.operations[operation_name].append({
                'execution_time': execution_time,
                'timestamp': time.time(),
                'metadata': metadata or {}
            })
            
            del self.start_times[operation_name]
            
            sr_logger.debug(f"SR operation {operation_name} completed in {execution_time:.3f}s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all monitored operations."""
        summary = {}
        
        for operation_name, records in self.operations.items():
            if records:
                times = [r['execution_time'] for r in records]
                summary[operation_name] = {
                    'count': len(records),
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_time': np.sum(times)
                }
        
        return summary


# Global performance monitor instance
sr_performance_monitor = SRPerformanceMonitor()


# Convenience functions for common operations
def log_sr_operation_start(operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Log the start of an SR operation."""
    sr_performance_monitor.start_operation(operation_name)
    if metadata:
        sr_logger.debug(f"SR operation {operation_name} started with metadata: {metadata}")


def log_sr_operation_end(operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Log the end of an SR operation."""
    sr_performance_monitor.end_operation(operation_name, metadata)


def get_sr_performance_stats() -> Dict[str, Any]:
    """Get current SR performance statistics."""
    return sr_performance_monitor.get_performance_summary()


# Module initialization (guarded)
from src.utils.initialization_guard import init_guard

if init_guard.mark_initialized("training.market_analysis.sr_error_handlers"):
    sr_logger.info("SR Error Handlers module initialized")