"""Data quality and formatting decorators for step1 and step1_5.

This module provides specialized decorators for data collection and processing
steps to ensure data quality, proper formatting, and robust error handling.
"""

import logging
import pandas as pd
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
import asyncio

from src.utils.error_handler import (
    handle_errors,
    ErrorSeverity,
)
from src.utils.logger import system_logger

# Type variables
F = Callable[..., Any]


def validate_data_quality(
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    max_null_ratio: float = 0.5,
    check_duplicates: bool = True,
    check_timestamps: bool = True,
    context: str = "",
) -> Callable[[F], F]:
    """Decorator to validate data quality before processing.

    Args:
        required_columns: List of required columns in the dataframe
        min_rows: Minimum number of rows required
        max_null_ratio: Maximum allowed ratio of null values per column
        check_duplicates: Whether to check for duplicate rows
        check_timestamps: Whether to validate timestamp columns
        context: Additional context for error logging

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild("DataQualityValidator")
            try:
                # Execute the function
                result = await func(*args, **kwargs)

                # Validate result if it's a DataFrame
                if isinstance(result, pd.DataFrame):
                    validation_result = await _validate_dataframe_quality(
                        result, required_columns, min_rows, max_null_ratio,
                        check_duplicates, check_timestamps, context
                    )

                    if not validation_result["valid"]:
                        logger.warning(f"⚠️ Data quality issues detected in {context}:")
                        for issue in validation_result["issues"]:
                            logger.warning(f"   - {issue}")

                    # Add quality metrics to result
                    result.attrs["quality_metrics"] = validation_result

                return result

            except Exception as e:
                logger.error(f"❌ Data quality validation failed in {context}: {e}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild("DataQualityValidator")
            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Validate result if it's a DataFrame
                if isinstance(result, pd.DataFrame):
                    validation_result = _validate_dataframe_quality_sync(
                        result, required_columns, min_rows, max_null_ratio,
                        check_duplicates, check_timestamps, context
                    )

                    if not validation_result["valid"]:
                        logger.warning(f"⚠️ Data quality issues detected in {context}:")
                        for issue in validation_result["issues"]:
                            logger.warning(f"   - {issue}")

                    # Add quality metrics to result
                    result.attrs["quality_metrics"] = validation_result

                return result

            except Exception as e:
                logger.error(f"❌ Data quality validation failed in {context}: {e}")
                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def ensure_data_format(
    expected_schema: Optional[Dict[str, str]] = None,
    timestamp_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    context: str = "",
) -> Callable[[F], F]:
    """Decorator to ensure data is in the expected format.

    Args:
        expected_schema: Expected column names and types
        timestamp_columns: Columns that should be datetime
        numeric_columns: Columns that should be numeric
        categorical_columns: Columns that should be categorical
        context: Additional context for error logging

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild("DataFormatter")
            try:
                # Execute the function
                result = await func(*args, **kwargs)

                # Format result if it's a DataFrame
                if isinstance(result, pd.DataFrame):
                    formatted_result = await _format_dataframe(
                        result, expected_schema, timestamp_columns,
                        numeric_columns, categorical_columns, context
                    )
                    return formatted_result

                return result

            except Exception as e:
                logger.error(f"❌ Data formatting failed in {context}: {e}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild("DataFormatter")
            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Format result if it's a DataFrame
                if isinstance(result, pd.DataFrame):
                    formatted_result = _format_dataframe_sync(
                        result, expected_schema, timestamp_columns,
                        numeric_columns, categorical_columns, context
                    )
                    return formatted_result

                return result

            except Exception as e:
                logger.error(f"❌ Data formatting failed in {context}: {e}")
                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def handle_data_collection_errors(
    context: str = "",
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Callable[[F], F]:
    """Decorator for handling data collection specific errors.

    Args:
        context: Additional context for error logging
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds

    Returns:
        Decorated function
    """
    return handle_errors(
        exceptions=(
            ConnectionError,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError
        ),
        default_return=None,
        context=f"data_collection_{context}",
        severity=ErrorSeverity.MEDIUM,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


def handle_data_processing_errors(
    context: str = "",
) -> Callable[[F], F]:
    """Decorator for handling data processing specific errors.

    Args:
        context: Additional context for error logging

    Returns:
        Decorated function
    """
    return handle_errors(
        exceptions=(
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            MemoryError
        ),
        default_return=None,
        context=f"data_processing_{context}",
        severity=ErrorSeverity.MEDIUM,
    )


def log_data_metrics(
    context: str = "",
    log_memory_usage: bool = True,
    log_shape: bool = True,
    log_dtypes: bool = False,
) -> Callable[[F], F]:
    """Decorator to log data metrics after processing.

    Args:
        context: Additional context for logging
        log_memory_usage: Whether to log memory usage
        log_shape: Whether to log data shape
        log_dtypes: Whether to log data types

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild("DataMetricsLogger")

            # Execute the function
            result = await func(*args, **kwargs)

            # Log metrics if result is a DataFrame
            if isinstance(result, pd.DataFrame):
                await _log_dataframe_metrics(
                    result, context, log_memory_usage, log_shape, log_dtypes, logger
                )

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild("DataMetricsLogger")

            # Execute the function
            result = func(*args, **kwargs)

            # Log metrics if result is a DataFrame
            if isinstance(result, pd.DataFrame):
                _log_dataframe_metrics_sync(
                    result, context, log_memory_usage, log_shape, log_dtypes, logger
                )

            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


async def _validate_dataframe_quality(
    df: pd.DataFrame,
    required_columns: Optional[List[str]],
    min_rows: int,
    max_null_ratio: float,
    check_duplicates: bool,
    check_timestamps: bool,
    context: str,
) -> Dict[str, Any]:
    """Validate DataFrame quality asynchronously."""
    issues = []
    valid = True

    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            valid = False

    # Check minimum rows
    if len(df) < min_rows:
        issues.append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
        valid = False

    # Check null ratios
    for col in df.columns:
        null_ratio = df[col].isnull().sum() / len(df)
        if null_ratio > max_null_ratio:
            issues.append(f"⚠️ Column '{col}' has {null_ratio:.2%} null values (max: {max_null_ratio:.2%})")

    # Check duplicates
    if check_duplicates and df.duplicated().any():
        duplicate_count = df.duplicated().sum()
        issues.append(f"⚠️ Found {duplicate_count} duplicate rows")

    # Check timestamps
    if check_timestamps:
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        for col in timestamp_cols:
            if col in df.columns:
                try:
                    pd.to_datetime(df[col])
                except Exception:
                    issues.append(f"⚠️ Column '{col}' contains invalid timestamp data")

    return {
        "valid": valid,
        "issues": issues,
        "row_count": len(df),
        "column_count": len(df.columns),
        "null_ratios": {col: df[col].isnull().sum() / len(df) for col in df.columns}
    }


def _validate_dataframe_quality_sync(
    df: pd.DataFrame,
    required_columns: Optional[List[str]],
    min_rows: int,
    max_null_ratio: float,
    check_duplicates: bool,
    check_timestamps: bool,
    context: str,
) -> Dict[str, Any]:
    """Validate DataFrame quality synchronously."""
    issues = []
    valid = True

    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            valid = False

    # Check minimum rows
    if len(df) < min_rows:
        issues.append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
        valid = False

    # Check null ratios
    for col in df.columns:
        null_ratio = df[col].isnull().sum() / len(df)
        if null_ratio > max_null_ratio:
            issues.append(f"⚠️ Column '{col}' has {null_ratio:.2%} null values (max: {max_null_ratio:.2%})")

    # Check duplicates
    if check_duplicates and df.duplicated().any():
        duplicate_count = df.duplicated().sum()
        issues.append(f"⚠️ Found {duplicate_count} duplicate rows")

    # Check timestamps
    if check_timestamps:
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        for col in timestamp_cols:
            if col in df.columns:
                try:
                    pd.to_datetime(df[col])
                except Exception:
                    issues.append(f"⚠️ Column '{col}' contains invalid timestamp data")

    return {
        "valid": valid,
        "issues": issues,
        "row_count": len(df),
        "column_count": len(df.columns),
        "null_ratios": {col: df[col].isnull().sum() / len(df) for col in df.columns}
    }


async def _format_dataframe(
    df: pd.DataFrame,
    expected_schema: Optional[Dict[str, str]],
    timestamp_columns: Optional[List[str]],
    numeric_columns: Optional[List[str]],
    categorical_columns: Optional[List[str]],
    context: str,
) -> pd.DataFrame:
    """Format DataFrame asynchronously."""
    formatted_df = df.copy()

    # Convert timestamp columns
    if timestamp_columns:
        for col in timestamp_columns:
            if col in formatted_df.columns:
                try:
                    formatted_df[col] = pd.to_datetime(formatted_df[col])
                except Exception as e:
                    system_logger.warning(f"⚠️ Failed to convert column '{col}' to datetime: {e}")

    # Convert numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in formatted_df.columns:
                try:
                    formatted_df[col] = pd.to_numeric(formatted_df[col], errors='coerce')
                except Exception as e:
                    system_logger.warning(f"⚠️ Failed to convert column '{col}' to numeric: {e}")

    # Convert categorical columns
    if categorical_columns:
        for col in categorical_columns:
            if col in formatted_df.columns:
                try:
                    formatted_df[col] = formatted_df[col].astype('category')
                except Exception as e:
                    system_logger.warning(f"⚠️ Failed to convert column '{col}' to categorical: {e}")

    return formatted_df


def _format_dataframe_sync(
    df: pd.DataFrame,
    expected_schema: Optional[Dict[str, str]],
    timestamp_columns: Optional[List[str]],
    numeric_columns: Optional[List[str]],
    categorical_columns: Optional[List[str]],
    context: str,
) -> pd.DataFrame:
    """Format DataFrame synchronously."""
    formatted_df = df.copy()

    # Convert timestamp columns
    if timestamp_columns:
        for col in timestamp_columns:
            if col in formatted_df.columns:
                try:
                    formatted_df[col] = pd.to_datetime(formatted_df[col])
                except Exception as e:
                    system_logger.warning(f"⚠️ Failed to convert column '{col}' to datetime: {e}")

    # Convert numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in formatted_df.columns:
                try:
                    formatted_df[col] = pd.to_numeric(formatted_df[col], errors='coerce')
                except Exception as e:
                    system_logger.warning(f"⚠️ Failed to convert column '{col}' to numeric: {e}")

    # Convert categorical columns
    if categorical_columns:
        for col in categorical_columns:
            if col in formatted_df.columns:
                try:
                    formatted_df[col] = formatted_df[col].astype('category')
                except Exception as e:
                    system_logger.warning(f"⚠️ Failed to convert column '{col}' to categorical: {e}")

    return formatted_df


async def _log_dataframe_metrics(
    df: pd.DataFrame,
    context: str,
    log_memory_usage: bool,
    log_shape: bool,
    log_dtypes: bool,
    logger: logging.Logger,
) -> None:
    """Log DataFrame metrics asynchronously."""
    metrics = []

    if log_shape:
        metrics.append(f"Shape: {df.shape}")

    if log_memory_usage:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        metrics.append(f"Memory: {memory_mb:.2f} MB")

    if log_dtypes:
        dtypes_str = ", ".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])
        metrics.append(f"Dtypes: {dtypes_str}")

    if metrics:
        logger.info(f"📊 {context} metrics: {' | '.join(metrics)}")


def _log_dataframe_metrics_sync(
    df: pd.DataFrame,
    context: str,
    log_memory_usage: bool,
    log_shape: bool,
    log_dtypes: bool,
    logger: logging.Logger,
) -> None:
    """Log DataFrame metrics synchronously."""
    try:
        metrics = []

        if log_shape:
            metrics.append(f"Shape: {df.shape}")

        if log_memory_usage:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            metrics.append(f"Memory: {memory_mb:.2f} MB")

        if log_dtypes:
            dtypes_str = ", ".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])
            metrics.append(f"Dtypes: {dtypes_str}")

        if metrics:
            logger.info(f"📊 {context} metrics: {' | '.join(metrics)}")

    except Exception as e:
        logger.warning(f"⚠️ Failed to log metrics for {context}: {e}")


# Specialized decorators for step1 and step1_5
def validate_klines_data(context: str = "") -> Callable[[F], F]:
    """Decorator to validate klines data quality."""
    return validate_data_quality(
        required_columns=["timestamp", "open", "high", "low", "close", "volume"],
        min_rows=1,
        max_null_ratio=0.1,
        check_duplicates=True,
        check_timestamps=True,
        context=f"klines_{context}",
    )


def validate_aggtrades_data(context: str = "") -> Callable[[F], F]:
    """Decorator to validate aggtrades data quality."""
    return validate_data_quality(
        required_columns=["timestamp", "price", "quantity"],
        min_rows=1,
        max_null_ratio=0.1,
        check_duplicates=True,
        check_timestamps=True,
        context=f"aggtrades_{context}",
    )


def validate_futures_data(context: str = "") -> Callable[[F], F]:
    """Decorator to validate futures data quality."""
    return validate_data_quality(
        required_columns=["timestamp", "fundingRate"],
        min_rows=1,
        max_null_ratio=0.3,  # Higher tolerance for futures data
        check_duplicates=True,
        check_timestamps=True,
        context=f"futures_{context}",
    )


def format_klines_data(context: str = "") -> Callable[[F], F]:
    """Decorator to format klines data."""
    return ensure_data_format(
        expected_schema={
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
        timestamp_columns=["timestamp"],
        numeric_columns=["open", "high", "low", "close", "volume"],
        context=f"klines_{context}",
    )


def format_aggtrades_data(context: str = "") -> Callable[[F], F]:
    """Decorator to format aggtrades data."""
    return ensure_data_format(
        expected_schema={
            "timestamp": "int64",
            "price": "float64",
            "quantity": "float64",
        },
        timestamp_columns=["timestamp"],
        numeric_columns=["price", "quantity"],
        context=f"aggtrades_{context}",
    )


def format_futures_data(context: str = "") -> Callable[[F], F]:
    """Decorator to format futures data."""
    return ensure_data_format(
        expected_schema={
            "timestamp": "int64",
            "fundingRate": "float64",
        },
        timestamp_columns=["timestamp"],
        numeric_columns=["fundingRate"],
        context=f"futures_{context}",
    )


def log_step_metrics(context: str = "") -> Callable[[F], F]:
    """Decorator to log step execution metrics."""
    return log_data_metrics(
        context=f"step_{context}",
        log_memory_usage=True,
        log_shape=True,
        log_dtypes=False,
    )
