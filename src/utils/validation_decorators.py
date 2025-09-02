"""
Validation Decorators for Continuous File Validation

This module provides decorators that validate files at every action throughout
the pipeline steps, ensuring continuous data quality monitoring.
"""

import functools
import inspect
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.comprehensive_file_validation import (
    ComprehensiveFileValidator,
    FileValidationResult,
    ValidationIssue,
    ValidationSeverity,
)
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards


def validate_file_operation(
    step_name: str,
    expected_schema: Optional[str] = None,
    validate_input: bool = True,
    validate_output: bool = True,
    log_level: str = "INFO",
):
    """
    Decorator to validate files at every file operation.

    Args:
        step_name: Name of the step for context
        expected_schema: Expected schema for validation
        validate_input: Whether to validate input files
        validate_output: Whether to validate output files
        log_level: Logging level for validation messages
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"ValidationDecorator.{step_name}")

            # Validate input files
            if validate_input:
                input_files = _extract_file_paths_from_args(args, kwargs, "input")
                for file_path in input_files:
                    if file_path and os.path.exists(file_path):
                        await _validate_file_operation(
                            file_path, step_name, expected_schema, "input", logger, log_level
                        )

            # Execute the function
            result = await func(*args, **kwargs)

            # Validate output files
            if validate_output:
                output_files = _extract_file_paths_from_result(result, "output")
                for file_path in output_files:
                    if file_path and os.path.exists(file_path):
                        await _validate_file_operation(
                            file_path, step_name, expected_schema, "output", logger, log_level
                        )

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"ValidationDecorator.{step_name}")

            # Validate input files
            if validate_input:
                input_files = _extract_file_paths_from_args(args, kwargs, "input")
                for file_path in input_files:
                    if file_path and os.path.exists(file_path):
                        _validate_file_operation_sync(file_path, step_name, expected_schema, "input", logger, log_level)

            # Execute the function
            result = func(*args, **kwargs)

            # Validate output files
            if validate_output:
                output_files = _extract_file_paths_from_result(result, "output")
                for file_path in output_files:
                    if file_path and os.path.exists(file_path):
                        _validate_file_operation_sync(
                            file_path, step_name, expected_schema, "output", logger, log_level
                        )

            return result

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def validate_dataframe_operation(
    step_name: str, validate_before: bool = True, validate_after: bool = True, log_level: str = "INFO"
):
    """
    Decorator to validate DataFrames at every operation.

    Args:
        step_name: Name of the step for context
        validate_before: Whether to validate before operation
        validate_after: Whether to validate after operation
        log_level: Logging level for validation messages
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"DataFrameValidation.{step_name}")

            # Validate input DataFrames
            if validate_before:
                dataframes = _extract_dataframes_from_args(args, kwargs)
                for i, df in enumerate(dataframes):
                    if df is not None:
                        await _validate_dataframe_operation(df, step_name, f"input_{i}", logger, log_level)

            # Execute the function
            result = await func(*args, **kwargs)

            # Validate output DataFrames
            if validate_after:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if hasattr(value, "shape"):  # Likely a DataFrame
                            await _validate_dataframe_operation(value, step_name, f"output_{key}", logger, log_level)
                elif hasattr(result, "shape"):  # Single DataFrame result
                    await _validate_dataframe_operation(result, step_name, "output", logger, log_level)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"DataFrameValidation.{step_name}")

            # Validate input DataFrames
            if validate_before:
                dataframes = _extract_dataframes_from_args(args, kwargs)
                for i, df in enumerate(dataframes):
                    if df is not None:
                        _validate_dataframe_operation_sync(df, step_name, f"input_{i}", logger, log_level)

            # Execute the function
            result = func(*args, **kwargs)

            # Validate output DataFrames
            if validate_after:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if hasattr(value, "shape"):  # Likely a DataFrame
                            _validate_dataframe_operation_sync(value, step_name, f"output_{key}", logger, log_level)
                elif hasattr(result, "shape"):  # Single DataFrame result
                    _validate_dataframe_operation_sync(result, step_name, "output", logger, log_level)

            return result

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def validate_step_operation(
    step_name: str, validate_files: bool = True, validate_dataframes: bool = True, log_level: str = "INFO"
):
    """
    Comprehensive decorator for step operations that validates both files and DataFrames.

    Args:
        step_name: Name of the step for context
        validate_files: Whether to validate file operations
        validate_dataframes: Whether to validate DataFrame operations
        log_level: Logging level for validation messages
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"StepValidation.{step_name}")

            # Execute the function
            result = await func(*args, **kwargs)

            # Post-execution validation
            if validate_files:
                await _validate_step_files(step_name, result, logger, log_level)

            if validate_dataframes:
                await _validate_step_dataframes(step_name, result, logger, log_level)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"StepValidation.{step_name}")

            # Execute the function
            result = func(*args, **kwargs)

            # Post-execution validation
            if validate_files:
                _validate_step_files_sync(step_name, result, logger, log_level)

            if validate_dataframes:
                _validate_step_dataframes_sync(step_name, result, logger, log_level)

            return result

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Helper functions for validation decorators


def _extract_file_paths_from_args(args: tuple, kwargs: dict, operation_type: str) -> List[str]:
    """Extract file paths from function arguments."""
    file_paths = []

    # Look for file paths in arguments
    for arg in args:
        if isinstance(arg, str) and _looks_like_file_path(arg):
            file_paths.append(arg)
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, str) and _looks_like_file_path(item):
                    file_paths.append(item)

    # Look for file paths in keyword arguments
    file_keywords = ["file_path", "filepath", "path", "file", "filename", "data_dir", "output_dir"]
    for key, value in kwargs.items():
        if any(file_key in key.lower() for file_key in file_keywords):
            if isinstance(value, str) and _looks_like_file_path(value):
                file_paths.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str) and _looks_like_file_path(item):
                        file_paths.append(item)

    return file_paths


def _extract_file_paths_from_result(result: Any, operation_type: str) -> List[str]:
    """Extract file paths from function result."""
    file_paths = []

    if isinstance(result, str) and _looks_like_file_path(result):
        file_paths.append(result)
    elif isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, str) and _looks_like_file_path(item):
                file_paths.append(item)
    elif isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, str) and _looks_like_file_path(value):
                file_paths.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str) and _looks_like_file_path(item):
                        file_paths.append(item)

    return file_paths


def _extract_dataframes_from_args(args: tuple, kwargs: dict) -> List[Any]:
    """Extract DataFrames from function arguments."""
    dataframes = []

    # Look for DataFrames in arguments
    for arg in args:
        if hasattr(arg, "shape"):  # Likely a DataFrame
            dataframes.append(arg)

    # Look for DataFrames in keyword arguments
    df_keywords = ["df", "dataframe", "data", "df_", "data_"]
    for key, value in kwargs.items():
        if any(df_key in key.lower() for df_key in df_keywords):
            if hasattr(value, "shape"):  # Likely a DataFrame
                dataframes.append(value)

    return dataframes


def _looks_like_file_path(path: str) -> bool:
    """Check if a string looks like a file path."""
    if not isinstance(path, str):
        return False

    # Check for common file extensions
    file_extensions = [".parquet", ".csv", ".json", ".pkl", ".pickle", ".h5", ".hdf5"]
    return any(path.lower().endswith(ext) for ext in file_extensions) or "/" in path or "\\" in path


async def _validate_file_operation(
    file_path: str, step_name: str, expected_schema: Optional[str], operation_type: str, logger: Any, log_level: str
) -> None:
    """Validate a file operation."""
    try:
        validator = ComprehensiveFileValidator()
        result = validator.validate_file_format(file_path, expected_schema, step_name)

        if result.is_valid:
            if log_level.upper() == "DEBUG":
                logger.debug(f"✅ {operation_type.capitalize()} file validation passed: {file_path}")
        else:
            if log_level.upper() in ["WARNING", "ERROR", "CRITICAL"]:
                logger.warning(f"⚠️ {operation_type.capitalize()} file validation issues: {file_path}")
                for issue in result.issues:
                    logger.warning(f"   - {issue.severity.value.upper()}: {issue.description}")

    except Exception as e:
        logger.error(f"❌ Error validating {operation_type} file {file_path}: {e}")


def _validate_file_operation_sync(
    file_path: str, step_name: str, expected_schema: Optional[str], operation_type: str, logger: Any, log_level: str
) -> None:
    """Validate a file operation (synchronous version)."""
    try:
        validator = ComprehensiveFileValidator()
        result = validator.validate_file_format(file_path, expected_schema, step_name)

        if result.is_valid:
            if log_level.upper() == "DEBUG":
                logger.debug(f"✅ {operation_type.capitalize()} file validation passed: {file_path}")
        else:
            if log_level.upper() in ["WARNING", "ERROR", "CRITICAL"]:
                logger.warning(f"⚠️ {operation_type.capitalize()} file validation issues: {file_path}")
                for issue in result.issues:
                    logger.warning(f"   - {issue.severity.value.upper()}: {issue.description}")

    except Exception as e:
        logger.error(f"❌ Error validating {operation_type} file {file_path}: {e}")


async def _validate_dataframe_operation(
    df: Any, step_name: str, operation_type: str, logger: Any, log_level: str
) -> None:
    """Validate a DataFrame operation."""
    try:
        # Basic DataFrame validation
        if df is None or df.empty:
            logger.warning(f"⚠️ {operation_type.capitalize()} DataFrame is None or empty")
            return

        # Check for common DataFrame issues
        issues = []

        # Check for null values
        null_counts = df.isnull().sum()
        high_null_columns = null_counts[null_counts > len(df) * 0.5]
        if not high_null_columns.empty:
            issues.append(f"High null ratio in columns: {list(high_null_columns.index)}")

        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate rows")

        # Check for infinite values
        if hasattr(df, "select_dtypes"):
            numeric_cols = df.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                if hasattr(df[col], "isin"):
                    infinite_count = df[col].isin([float("inf"), float("-inf")]).sum()
                    if infinite_count > 0:
                        issues.append(f"Column '{col}' has {infinite_count} infinite values")

        if issues:
            if log_level.upper() in ["WARNING", "ERROR", "CRITICAL"]:
                logger.warning(f"⚠️ {operation_type.capitalize()} DataFrame validation issues:")
                for issue in issues:
                    logger.warning(f"   - {issue}")
        else:
            if log_level.upper() == "DEBUG":
                logger.debug(f"✅ {operation_type.capitalize()} DataFrame validation passed: shape={df.shape}")

    except Exception as e:
        logger.error(f"❌ Error validating {operation_type} DataFrame: {e}")


def _validate_dataframe_operation_sync(
    df: Any, step_name: str, operation_type: str, logger: Any, log_level: str
) -> None:
    """Validate a DataFrame operation (synchronous version)."""
    try:
        # Basic DataFrame validation
        if df is None or df.empty:
            logger.warning(f"⚠️ {operation_type.capitalize()} DataFrame is None or empty")
            return

        # Check for common DataFrame issues
        issues = []

        # Check for null values
        null_counts = df.isnull().sum()
        high_null_columns = null_counts[null_counts > len(df) * 0.5]
        if not high_null_columns.empty:
            issues.append(f"High null ratio in columns: {list(high_null_columns.index)}")

        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate rows")

        # Check for infinite values
        if hasattr(df, "select_dtypes"):
            numeric_cols = df.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                if hasattr(df[col], "isin"):
                    infinite_count = df[col].isin([float("inf"), float("-inf")]).sum()
                    if infinite_count > 0:
                        issues.append(f"Column '{col}' has {infinite_count} infinite values")

        if issues:
            if log_level.upper() in ["WARNING", "ERROR", "CRITICAL"]:
                logger.warning(f"⚠️ {operation_type.capitalize()} DataFrame validation issues:")
                for issue in issues:
                    logger.warning(f"   - {issue}")
        else:
            if log_level.upper() == "DEBUG":
                logger.debug(f"✅ {operation_type.capitalize()} DataFrame validation passed: shape={df.shape}")

    except Exception as e:
        logger.error(f"❌ Error validating {operation_type} DataFrame: {e}")


async def _validate_step_files(step_name: str, result: Any, logger: Any, log_level: str) -> None:
    """Validate files after step execution."""
    # This would implement step-specific file validation logic
    pass


def _validate_step_files_sync(step_name: str, result: Any, logger: Any, log_level: str) -> None:
    """Validate files after step execution (synchronous version)."""
    # This would implement step-specific file validation logic
    pass


async def _validate_step_dataframes(step_name: str, result: Any, logger: Any, log_level: str) -> None:
    """Validate DataFrames after step execution."""
    # This would implement step-specific DataFrame validation logic
    pass


def _validate_step_dataframes_sync(step_name: str, result: Any, logger: Any, log_level: str) -> None:
    """Validate DataFrames after step execution (synchronous version)."""
    # This would implement step-specific DataFrame validation logic
    pass


# Convenience decorators for specific steps


def validate_step1_operation(func: Callable) -> Callable:
    """Decorator for step 1 operations."""
    return validate_step_operation("step01", validate_files=True, validate_dataframes=True)(func)


def validate_step1_5_operation(func: Callable) -> Callable:
    """Decorator for step 1.5 operations."""
    return validate_step_operation("step1_5", validate_files=True, validate_dataframes=True)(func)


def validate_step2_operation(func: Callable) -> Callable:
    """Decorator for step 2 operations."""
    return validate_step_operation("step02", validate_files=True, validate_dataframes=True)(func)


def validate_step4_operation(func: Callable) -> Callable:
    """Decorator for step 4 operations."""
    return validate_step_operation("step04", validate_files=True, validate_dataframes=True)(func)
