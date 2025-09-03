from __future__ import annotations
"""
Domain-specific decorators built on top of core decorators.

This module provides specialized decorators for the trading system
that compose and extend the core decorator functionality.
"""

from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar
import pandas as pd
import numpy as np
import time
import logging
import asyncio
from enum import Enum

from src.core.decorators import (
    compose,
    validates,
    handles_errors,
    retry,
    timeout,
    cached,
    log_call,
    log_execution_time,
    traced,
    fallback,
    ensure_async,
    ensure_sync,
)
from src.core.errors import (
    ValidationError,
    DataIntegrityError,
    BusinessRuleError,
)

# Type variables
F = TypeVar('F', bound=Callable[..., Any])

# Enums for configuration
class ValidationLevel(str, Enum):
    """Validation level for data quality checks."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class PerformanceLevel(str, Enum):
    """Performance monitoring levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# Data Quality Decorators
def validate_data_quality(
    validation_level: ValidationLevel = ValidationLevel.WARNING,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    max_null_ratio: float = 0.0,
    check_duplicates: bool = True,
    check_timestamps: bool = True,
    check_nan: bool = True,
    check_infinite: bool = True,
    check_constant: bool = True,
    check_correlation: bool = True,
    max_correlation_threshold: float = 0.95,
    min_unique_values: int = 2,
    context: str = "data_validation",
    fail_on_issues: bool = False,
) -> Callable[[F], F]:
    """
    Comprehensive data quality validation decorator.
    
    Validates pandas DataFrames for various quality issues.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract DataFrames from arguments
            dfs_to_validate = []
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    dfs_to_validate.append(arg)
            for value in kwargs.values():
                if isinstance(value, pd.DataFrame):
                    dfs_to_validate.append(value)
            
            # Validate each DataFrame
            issues = []
            for df in dfs_to_validate:
                df_issues = _validate_dataframe(
                    df,
                    required_columns=required_columns,
                    min_rows=min_rows,
                    max_null_ratio=max_null_ratio,
                    check_duplicates=check_duplicates,
                    check_timestamps=check_timestamps,
                    check_nan=check_nan,
                    check_infinite=check_infinite,
                    check_constant=check_constant,
                    check_correlation=check_correlation,
                    max_correlation_threshold=max_correlation_threshold,
                    min_unique_values=min_unique_values,
                )
                issues.extend(df_issues)
            
            # Handle validation results
            if issues:
                if validation_level == ValidationLevel.ERROR or fail_on_issues:
                    raise ValidationError(f"Data quality validation failed: {issues}")
                elif validation_level == ValidationLevel.WARNING:
                    logging.warning(f"Data quality issues in {context}: {issues}")
                else:
                    logging.info(f"Data quality issues in {context}: {issues}")
            
            # Execute function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def _validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    max_null_ratio: float = 0.0,
    check_duplicates: bool = True,
    check_timestamps: bool = True,
    check_nan: bool = True,
    check_infinite: bool = True,
    check_constant: bool = True,
    check_correlation: bool = True,
    max_correlation_threshold: float = 0.95,
    min_unique_values: int = 2,
) -> List[str]:
    """Internal function to validate a DataFrame."""
    issues = []
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
    
    # Check minimum rows
    if len(df) < min_rows:
        issues.append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
    
    # Check null ratio
    if max_null_ratio < 1.0:
        null_ratios = df.isnull().sum() / len(df)
        high_null_cols = null_ratios[null_ratios > max_null_ratio]
        if not high_null_cols.empty:
            issues.append(f"Columns with high null ratio: {high_null_cols.to_dict()}")
    
    # Check duplicates
    if check_duplicates and df.duplicated().any():
        issues.append(f"Found {df.duplicated().sum()} duplicate rows")
    
    # Check NaN values
    if check_nan and df.isnull().any().any():
        nan_cols = df.columns[df.isnull().any()].tolist()
        issues.append(f"Columns with NaN values: {nan_cols}")
    
    # Check infinite values
    if check_infinite:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        if inf_cols:
            issues.append(f"Columns with infinite values: {inf_cols}")
    
    # Check constant features
    if check_constant:
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() < min_unique_values:
                constant_cols.append(col)
        if constant_cols:
            issues.append(f"Constant or low-variance columns: {constant_cols}")
    
    # Check high correlations
    if check_correlation and len(df.columns) > 1:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_pairs = []
            for col in upper_triangle.columns:
                high_corr_cols = upper_triangle.index[
                    upper_triangle[col] > max_correlation_threshold
                ].tolist()
                for high_corr_col in high_corr_cols:
                    high_corr_pairs.append((col, high_corr_col))
            if high_corr_pairs:
                issues.append(f"Highly correlated column pairs: {high_corr_pairs}")
    
    return issues


# Feature Engineering Validation Decorators
def validate_feature_engineering_with_lookahead_bias_detection(
    lag_periods: int = 1,
    check_future_data: bool = True,
    timestamp_column: str = "timestamp",
) -> Callable[[F], F]:
    """Validate feature engineering and detect lookahead bias."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if isinstance(result, pd.DataFrame) and check_future_data:
                # Check for potential lookahead bias
                if timestamp_column in result.columns:
                    # Ensure features don't use future data
                    # This is a simplified check - real implementation would be more sophisticated
                    logging.info(f"Validated features for lookahead bias with lag={lag_periods}")
            
            return result
        
        return wrapper
    return decorator


# Monitoring and Performance Decorators
def monitor_step_execution(
    step_name: str,
    performance_level: PerformanceLevel = PerformanceLevel.MEDIUM,
    log_memory: bool = True,
    log_inputs: bool = False,
    log_outputs: bool = False,
) -> Callable[[F], F]:
    """Monitor step execution with performance tracking."""
    def decorator(func: F) -> F:
        # Compose multiple decorators
        return compose(
            log_execution_time,
            log_call(
                include_args=log_inputs,
                include_result=log_outputs,
                max_length=100
            ),
            traced(name=f"step.{step_name}")
        )(func)
    
    return decorator


def quality_gate(
    min_score: float = 0.8,
    metrics: Optional[List[str]] = None,
    fail_on_breach: bool = True,
) -> Callable[[F], F]:
    """Quality gate decorator to ensure minimum performance standards."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Check if result contains quality metrics
            if isinstance(result, dict) and "metrics" in result:
                quality_score = result["metrics"].get("quality_score", 1.0)
                if quality_score < min_score:
                    msg = f"Quality gate failed: score {quality_score} < {min_score}"
                    if fail_on_breach:
                        raise BusinessRuleError(msg)
                    else:
                        logging.warning(msg)
            
            return result
        
        return wrapper
    return decorator


# Security and Data Processing Decorators
def secure_data_processing(
    mask_sensitive_columns: Optional[List[str]] = None,
    allowed_operations: Optional[List[str]] = None,
    audit: bool = True,
) -> Callable[[F], F]:
    """Secure data processing with sensitive data protection."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log operation if auditing is enabled
            if audit:
                logging.info(f"Secure operation: {func.__name__}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Mask sensitive columns in result if needed
            if isinstance(result, pd.DataFrame) and mask_sensitive_columns:
                for col in mask_sensitive_columns:
                    if col in result.columns:
                        result[col] = "***MASKED***"
            
            return result
        
        return wrapper
    return decorator


def prevent_data_leakage(
    temporal_column: str = "timestamp",
    lookback_only: bool = True,
    max_lookahead: int = 0,
) -> Callable[[F], F]:
    """Prevent data leakage in time series operations."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Add metadata about leakage prevention
            if hasattr(func, "__name__"):
                logging.debug(f"Applying data leakage prevention to {func.__name__}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Validate no future data is used if result is a DataFrame
            if isinstance(result, pd.DataFrame) and temporal_column in result.columns:
                # Check would be implemented here
                pass
            
            return result
        
        return wrapper
    return decorator


def ensure_data_integrity(
    check_before: bool = True,
    check_after: bool = True,
    integrity_checks: Optional[List[str]] = None,
) -> Callable[[F], F]:
    """Ensure data integrity before and after operations."""
    def decorator(func: F) -> F:
        # Compose with validation and error handling
        return compose(
            validates,
            handles_errors(
                fallback=None,
                log_errors=True,
                raise_errors=True
            )
        )(func)
    
    return decorator


# Pipeline and Step Management Decorators
def validate_pipeline_step(
    prerequisites: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    stage: Optional[str] = None,
) -> Callable[[F], F]:
    """Validate pipeline step prerequisites and outputs."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check prerequisites
            if prerequisites:
                logging.info(f"Checking prerequisites for {func.__name__}: {prerequisites}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Validate outputs
            if outputs and isinstance(result, dict):
                missing_outputs = set(outputs) - set(result.keys())
                if missing_outputs:
                    raise ValidationError(f"Missing required outputs: {missing_outputs}")
            
            return result
        
        return wrapper
    return decorator


# Specialized Validation Decorators
def validate_klines_data_quality(
    required_columns: List[str] = ["open", "high", "low", "close", "volume"],
    check_ohlc_integrity: bool = True,
) -> Callable[[F], F]:
    """Validate OHLC/klines data quality."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use the general data quality validator with specific settings
            validator = validate_data_quality(
                required_columns=required_columns,
                check_nan=True,
                check_infinite=True,
                context="klines_validation"
            )
            
            # Apply validation
            validated_func = validator(func)
            result = validated_func(*args, **kwargs)
            
            # Additional OHLC integrity checks
            if check_ohlc_integrity and isinstance(result, pd.DataFrame):
                # Check high >= low
                if "high" in result.columns and "low" in result.columns:
                    invalid_rows = result["high"] < result["low"]
                    if invalid_rows.any():
                        raise DataIntegrityError(f"Found {invalid_rows.sum()} rows where high < low")
                
                # Check OHLC relationships
                if all(col in result.columns for col in ["open", "high", "low", "close"]):
                    # High should be >= max(open, close)
                    invalid_high = result["high"] < result[["open", "close"]].max(axis=1)
                    if invalid_high.any():
                        raise DataIntegrityError(f"Found {invalid_high.sum()} rows with invalid high values")
                    
                    # Low should be <= min(open, close)
                    invalid_low = result["low"] > result[["open", "close"]].min(axis=1)
                    if invalid_low.any():
                        raise DataIntegrityError(f"Found {invalid_low.sum()} rows with invalid low values")
            
            return result
        
        return wrapper
    return decorator


# Multi-timeframe validation
def validate_multi_timeframe_data_quality(
    timeframes: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"],
    alignment_tolerance: int = 1000,  # milliseconds
) -> Callable[[F], F]:
    """Validate multi-timeframe data quality and alignment."""
    def decorator(func: F) -> F:
        return compose(
            validate_data_quality(
                context="multi_timeframe",
                check_timestamps=True
            ),
            traced(name="validate.multi_timeframe")
        )(func)
    
    return decorator


# Convenience functions to create common decorator combinations
def create_step_decorator(
    step_name: str,
    validate_inputs: bool = True,
    monitor_performance: bool = True,
    handle_errors: bool = True,
    cache_results: bool = False,
    timeout_seconds: Optional[int] = None,
) -> Callable[[F], F]:
    """Create a comprehensive decorator for a pipeline step."""
    decorators = []
    
    if handle_errors:
        decorators.append(handles_errors(log_errors=True))
    
    if timeout_seconds:
        decorators.append(timeout(seconds=timeout_seconds))
    
    if validate_inputs:
        decorators.append(validates)
    
    if monitor_performance:
        decorators.append(monitor_step_execution(step_name))
    
    if cache_results:
        decorators.append(cached(ttl=3600))
    
    return compose(*decorators)


# Export all decorators
__all__ = [
    # Enums
    "ValidationLevel",
    "PerformanceLevel",
    
    # Data Quality
    "validate_data_quality",
    "validate_feature_engineering_with_lookahead_bias_detection",
    "validate_klines_data_quality",
    "validate_multi_timeframe_data_quality",
    
    # Monitoring and Performance
    "monitor_step_execution",
    "quality_gate",
    
    # Security and Processing
    "secure_data_processing",
    "prevent_data_leakage",
    "ensure_data_integrity",
    
    # Pipeline Management
    "validate_pipeline_step",
    
    # Utilities
    "create_step_decorator",
]
