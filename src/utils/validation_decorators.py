"""
Validation Decorators for Continuous File Validation

This module provides decorators that validate files at every action throughout
the pipeline steps, ensuring continuous data quality monitoring.
"""

import os
import sys
import functools
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import system_logger
from src.utils.comprehensive_file_validation import (
    ComprehensiveFileValidator,
    FileValidationResult,
    ValidationIssue,
    ValidationSeverity
)

def validate_file_operation(
    step_name: str,
    expected_schema: Optional[Dict] = None,
    validate_input: bool = True,
    validate_output: bool = True,
    log_level: str = "INFO"
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
                input_files, _ = _extract_file_paths_from_args(args, kwargs, "input")
                for file_path in input_files:
                    if file_path and os.path.exists(file_path):
                        await _validate_file_operation(
                            file_path, step_name, expected_schema, "input", logger, log_level
                        )

            # Execute the function
            result = await func(*args, **kwargs)

            # Validate output files
            if validate_output:
                output_files, _ = _extract_file_paths_from_result(result, "output")
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
                input_files, _ = _extract_file_paths_from_args(args, kwargs, "input")
                for file_path in input_files:
                    if file_path and os.path.exists(file_path):
                        _validate_file_operation_sync(
                            file_path, step_name, expected_schema, "input", logger, log_level
                        )

            # Execute the function
            result = func(*args, **kwargs)

            # Validate output files
            if validate_output:
                output_files, _ = _extract_file_paths_from_result(result, "output")
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
    step_name: str,
    validate_before: bool = True,
    validate_after: bool = True,
    log_level: str = "INFO"
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
                dataframes, _ = _extract_dataframes_from_args(args, kwargs)
                for i, df in enumerate(dataframes):
                    if df is not None:
                        await _validate_dataframe_operation(
                            df, step_name, f"input_{i}", logger, log_level
                        )

            # Execute the function
            result = await func(*args, **kwargs)

            # Validate output DataFrames
            if validate_after:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if hasattr(value, 'shape'):  # Likely a DataFrame
                            await _validate_dataframe_operation(
                                value, step_name, f"output_{key}", logger, log_level
                            )
                elif hasattr(result, 'shape'):  # Single DataFrame result
                    await _validate_dataframe_operation(
                        result, step_name, "output", logger, log_level
                    )

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"DataFrameValidation.{step_name}")

            # Validate input DataFrames
            if validate_before:
                dataframes, _ = _extract_dataframes_from_args(args, kwargs)
                for i, df in enumerate(dataframes):
                    if df is not None:
                        _validate_dataframe_operation_sync(
                            df, step_name, f"input_{i}", logger, log_level
                        )

            # Execute the function
            result = func(*args, **kwargs)

            # Validate output DataFrames
            if validate_after:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if hasattr(value, 'shape'):  # Likely a DataFrame
                            _validate_dataframe_operation_sync(
                                value, step_name, f"output_{key}", logger, log_level
                            )
                elif hasattr(result, 'shape'):  # Single DataFrame result
                    _validate_dataframe_operation_sync(
                        result, step_name, "output", logger, log_level
                    )

            return result

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

def validate_step_operation(
    step_name: str,
    expected_schema: Optional[Dict] = None,
    validate_files: bool = True,
    validate_dataframes: bool = True,
    log_level: str = "INFO"
):
    """
    Comprehensive decorator for step operations that validates both files and DataFrames.

    Args:
        step_name: Name of the step for context
        expected_schema: Expected schema for validation
        validate_files: Whether to validate file operations
        validate_dataframes: Whether to validate DataFrame operations
        log_level: Logging level for validation messages
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"StepValidation.{step_name}")

            # Validate input files
            if validate_files:
                input_files, _ = _extract_file_paths_from_args(args, kwargs, "input")
                for file_path in input_files:
                    if file_path and os.path.exists(file_path):
                        await _validate_file_operation(
                            file_path, step_name, expected_schema, "input", logger, log_level
                        )

            # Validate input DataFrames
            if validate_dataframes:
                dataframes, _ = _extract_dataframes_from_args(args, kwargs)
                for i, df in enumerate(dataframes):
                    if df is not None:
                        await _validate_dataframe_operation(
                            df, step_name, f"input_{i}", logger, log_level
                        )

            # Execute the function
            result = await func(*args, **kwargs)

            # Validate output files
            if validate_files:
                output_files, _ = _extract_file_paths_from_result(result, "output")
                for file_path in output_files:
                    if file_path and os.path.exists(file_path):
                        await _validate_file_operation(
                            file_path, step_name, expected_schema, "output", logger, log_level
                        )

            # Validate output DataFrames
            if validate_dataframes:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if hasattr(value, 'shape'):  # Likely a DataFrame
                            await _validate_dataframe_operation(
                                value, step_name, f"output_{key}", logger, log_level
                            )
                elif hasattr(result, 'shape'):  # Single DataFrame result
                    await _validate_dataframe_operation(
                        result, step_name, "output", logger, log_level
                    )

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"StepValidation.{step_name}")

            # Validate input files
            if validate_files:
                input_files, _ = _extract_file_paths_from_args(args, kwargs, "input")
                for file_path in input_files:
                    if file_path and os.path.exists(file_path):
                        _validate_file_operation_sync(
                            file_path, step_name, expected_schema, "input", logger, log_level
                        )

            # Validate input DataFrames
            if validate_dataframes:
                dataframes, _ = _extract_dataframes_from_args(args, kwargs)
                for i, df in enumerate(dataframes):
                    if df is not None:
                        _validate_dataframe_operation_sync(
                            df, step_name, f"input_{i}", logger, log_level
                        )

            # Execute the function
            result = func(*args, **kwargs)

            # Validate output files
            if validate_files:
                output_files, _ = _extract_file_paths_from_result(result, "output")
                for file_path in output_files:
                    if file_path and os.path.exists(file_path):
                        _validate_file_operation_sync(
                            file_path, step_name, expected_schema, "output", logger, log_level
                        )

            # Validate output DataFrames
            if validate_dataframes:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if hasattr(value, 'shape'):  # Likely a DataFrame
                            _validate_dataframe_operation_sync(
                                value, step_name, f"output_{key}", logger, log_level
                            )
                elif hasattr(result, 'shape'):  # Single DataFrame result
                    _validate_dataframe_operation_sync(
                        result, step_name, "output", logger, log_level
                    )

            return result

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

# Helper functions
def _extract_file_paths_from_args(args: tuple, kwargs: dict, file_type: str) -> tuple:
    """
    Extract file paths from function arguments.
    
    Args:
        args: Function arguments
        kwargs: Function keyword arguments
        file_type: Type of files to extract ("input" or "output")
        
    Returns:
        Tuple of (file_paths, metadata)
    """
    file_paths = []
    metadata = {}
    
    # Look for file paths in arguments
    for arg in args:
        if isinstance(arg, str) and os.path.exists(arg):
            file_paths.append(arg)
        elif isinstance(arg, Path):
            file_paths.append(str(arg))
    
    # Look for file paths in keyword arguments
    for key, value in kwargs.items():
        if isinstance(value, str) and os.path.exists(value):
            file_paths.append(value)
        elif isinstance(value, Path):
            file_paths.append(str(value))
    
    return file_paths, metadata

def _extract_file_paths_from_result(result: Any, file_type: str) -> tuple:
    """
    Extract file paths from function result.
    
    Args:
        result: Function result
        file_type: Type of files to extract ("input" or "output")
        
    Returns:
        Tuple of (file_paths, metadata)
    """
    file_paths = []
    metadata = {}
    
    if isinstance(result, str) and os.path.exists(result):
        file_paths.append(result)
    elif isinstance(result, Path):
        file_paths.append(str(result))
    elif isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, str) and os.path.exists(value):
                file_paths.append(value)
            elif isinstance(value, Path):
                file_paths.append(str(value))
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, str) and os.path.exists(item):
                file_paths.append(item)
            elif isinstance(item, Path):
                file_paths.append(str(item))
    
    return file_paths, metadata

def _extract_dataframes_from_args(args: tuple, kwargs: dict) -> tuple:
    """
    Extract DataFrames from function arguments.
    
    Args:
        args: Function arguments
        kwargs: Function keyword arguments
        
    Returns:
        Tuple of (dataframes, metadata)
    """
    dataframes = []
    metadata = {}
    
    # Look for DataFrames in arguments
    for arg in args:
        if hasattr(arg, 'shape') and hasattr(arg, 'columns'):
            dataframes.append(arg)
    
    # Look for DataFrames in keyword arguments
    for key, value in kwargs.items():
        if hasattr(value, 'shape') and hasattr(value, 'columns'):
            dataframes.append(value)
    
    return dataframes, metadata

async def _validate_file_operation(
    file_path: str,
    step_name: str,
    expected_schema: Optional[Dict],
    operation_type: str,
    logger,
    log_level: str
) -> None:
    """
    Validate a file operation asynchronously.
    
    Args:
        file_path: Path to the file to validate
        step_name: Name of the step
        expected_schema: Expected schema for validation
        operation_type: Type of operation ("input" or "output")
        logger: Logger instance
        log_level: Logging level
    """
    try:
        validator = ComprehensiveFileValidator()
        result = await validator.validate_file(file_path, expected_schema)
        
        if result.is_valid:
            logger.log(getattr(logging, log_level.upper(), logging.INFO),
                      f"File validation passed: {file_path} ({operation_type})")
        else:
            logger.warning(f"File validation failed: {file_path} ({operation_type})")
            for issue in result.issues:
                logger.warning(f"  - {issue.message} (Severity: {issue.severity})")
    except Exception as e:
        logger.error(f"Error during file validation: {e}")

def _validate_file_operation_sync(
    file_path: str,
    step_name: str,
    expected_schema: Optional[Dict],
    operation_type: str,
    logger,
    log_level: str
) -> None:
    """
    Validate a file operation synchronously.
    
    Args:
        file_path: Path to the file to validate
        step_name: Name of the step
        expected_schema: Expected schema for validation
        operation_type: Type of operation ("input" or "output")
        logger: Logger instance
        log_level: Logging level
    """
    try:
        validator = ComprehensiveFileValidator()
        result = validator.validate_file_sync(file_path, expected_schema)
        
        if result.is_valid:
            logger.log(getattr(logging, log_level.upper(), logging.INFO),
                      f"File validation passed: {file_path} ({operation_type})")
        else:
            logger.warning(f"File validation failed: {file_path} ({operation_type})")
            for issue in result.issues:
                logger.warning(f"  - {issue.message} (Severity: {issue.severity})")
    except Exception as e:
        logger.error(f"Error during file validation: {e}")

async def _validate_dataframe_operation(
    df,
    step_name: str,
    operation_type: str,
    logger,
    log_level: str
) -> None:
    """
    Validate a DataFrame operation asynchronously.
    
    Args:
        df: DataFrame to validate
        step_name: Name of the step
        operation_type: Type of operation
        logger: Logger instance
        log_level: Logging level
    """
    try:
        # Basic DataFrame validation
        if df is None:
            logger.warning(f"DataFrame is None: {operation_type}")
            return
        
        if df.empty:
            logger.warning(f"DataFrame is empty: {operation_type}")
            return
        
        # Check for common data quality issues
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.info(f"DataFrame has {null_counts.sum()} null values: {operation_type}")
        
        logger.log(getattr(logging, log_level.upper(), logging.INFO),
                  f"DataFrame validation passed: {operation_type} (shape: {df.shape})")
    except Exception as e:
        logger.error(f"Error during DataFrame validation: {e}")

def _validate_dataframe_operation_sync(
    df,
    step_name: str,
    operation_type: str,
    logger,
    log_level: str
) -> None:
    """
    Validate a DataFrame operation synchronously.
    
    Args:
        df: DataFrame to validate
        step_name: Name of the step
        operation_type: Type of operation
        logger: Logger instance
        log_level: Logging level
    """
    try:
        # Basic DataFrame validation
        if df is None:
            logger.warning(f"DataFrame is None: {operation_type}")
            return
        
        if df.empty:
            logger.warning(f"DataFrame is empty: {operation_type}")
            return
        
        # Check for common data quality issues
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.info(f"DataFrame has {null_counts.sum()} null values: {operation_type}")
        
        logger.log(getattr(logging, log_level.upper(), logging.INFO),
                  f"DataFrame validation passed: {operation_type} (shape: {df.shape})")
    except Exception as e:
        logger.error(f"Error during DataFrame validation: {e}")

# Add missing import
import logging