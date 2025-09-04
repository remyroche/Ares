#!/usr/bin/env python3
"""
Operation Protection Decorators

This module provides decorators for protecting data operations, formatting, analysis,
and access throughout the pipeline with comprehensive validation and error handling.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union, Type, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
    safe_file_exists,
    ensure_directory,
)
from src.utils.logger import system_logger
from src.utils.pipeline_validation_framework import (
    validation_orchestrator,
    ValidationLevel,
    ValidationResult,
)


class OperationProtectionError(Exception):
    """Custom exception for operation protection failures."""
    pass


class DataFormattingError(OperationProtectionError):
    """Exception for data formatting failures."""
    pass


class DataAnalysisError(OperationProtectionError):
    """Exception for data analysis failures."""
    pass


class DataAccessError(OperationProtectionError):
    """Exception for data access failures."""
    pass


class ModelTrainingError(OperationProtectionError):
    """Exception for model training failures."""
    pass


def validate_data_format(
    required_columns: Optional[List[str]] = None,
    required_dtypes: Optional[Dict[str, Type]] = None,
    allow_empty: bool = False,
    validation_level: ValidationLevel = ValidationLevel.STANDARD
):
    """
    Decorator to validate data formatting operations.
    
    Args:
        required_columns: List of required column names
        required_dtypes: Dictionary mapping column names to required data types
        allow_empty: Whether to allow empty DataFrames
        validation_level: Validation level to use
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            logger = system_logger.getChild(f"DataFormatValidator.{func.__name__}")
            start_time = time.time()
            
            try:
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Validate the result
                if isinstance(result, pd.DataFrame):
                    # Check if DataFrame is empty
                    if result.empty and not allow_empty:
                        raise DataFormattingError(f"Function {func.__name__} returned empty DataFrame")
                    
                    # Check required columns
                    if required_columns:
                        missing_columns = [col for col in required_columns if col not in result.columns]
                        if missing_columns:
                            raise DataFormattingError(
                                f"Function {func.__name__} missing required columns: {missing_columns}"
                            )
                    
                    # Check data types
                    if required_dtypes:
                        for col, expected_dtype in required_dtypes.items():
                            if col in result.columns:
                                actual_dtype = result[col].dtype
                                if not np.issubdtype(actual_dtype, expected_dtype):
                                    raise DataFormattingError(
                                        f"Function {func.__name__} column {col} has wrong dtype: "
                                        f"expected {expected_dtype}, got {actual_dtype}"
                                    )
                    
                    # Validate data quality
                    validation_context = {
                        'function_name': func.__name__,
                        'validation_level': validation_level.value,
                        'required_columns': required_columns,
                        'required_dtypes': required_dtypes,
                        'allow_empty': allow_empty
                    }
                    
                    validation_results = await validation_orchestrator.validate_step(
                        step_name=f"data_format_{func.__name__}",
                        data=result,
                        context=validation_context,
                        validation_types=['data_format']
                    )
                    
                    # Check validation results
                    for validation_type, report in validation_results.items():
                        if report.result == ValidationResult.FAILED:
                            raise DataFormattingError(
                                f"Data format validation failed: {report.errors}"
                            )
                        elif report.result == ValidationResult.WARNING:
                            logger.warning(f"Data format validation warnings: {report.warnings}")
                
                duration = time.time() - start_time
                logger.info(f"✅ Data format validation passed for {func.__name__} in {duration:.3f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ Data format validation failed for {func.__name__} in {duration:.3f}s: {e}")
                raise DataFormattingError(f"Data format validation failed: {e}") from e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            logger = system_logger.getChild(f"DataFormatValidator.{func.__name__}")
            start_time = time.time()
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Validate the result
                if isinstance(result, pd.DataFrame):
                    # Check if DataFrame is empty
                    if result.empty and not allow_empty:
                        raise DataFormattingError(f"Function {func.__name__} returned empty DataFrame")
                    
                    # Check required columns
                    if required_columns:
                        missing_columns = [col for col in required_columns if col not in result.columns]
                        if missing_columns:
                            raise DataFormattingError(
                                f"Function {func.__name__} missing required columns: {missing_columns}"
                            )
                    
                    # Check data types
                    if required_dtypes:
                        for col, expected_dtype in required_dtypes.items():
                            if col in result.columns:
                                actual_dtype = result[col].dtype
                                if not np.issubdtype(actual_dtype, expected_dtype):
                                    raise DataFormattingError(
                                        f"Function {func.__name__} column {col} has wrong dtype: "
                                        f"expected {expected_dtype}, got {actual_dtype}"
                                    )
                
                duration = time.time() - start_time
                logger.info(f"✅ Data format validation passed for {func.__name__} in {duration:.3f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ Data format validation failed for {func.__name__} in {duration:.3f}s: {e}")
                raise DataFormattingError(f"Data format validation failed: {e}") from e
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def validate_data_analysis(
    required_outputs: Optional[List[str]] = None,
    validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE
):
    """
    Decorator to validate data analysis operations.
    
    Args:
        required_outputs: List of required output keys in analysis results
        validation_level: Validation level to use
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            logger = system_logger.getChild(f"DataAnalysisValidator.{func.__name__}")
            start_time = time.time()
            
            try:
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Validate the result
                if isinstance(result, dict):
                    # Check required outputs
                    if required_outputs:
                        missing_outputs = [output for output in required_outputs if output not in result]
                        if missing_outputs:
                            raise DataAnalysisError(
                                f"Function {func.__name__} missing required outputs: {missing_outputs}"
                            )
                    
                    # Validate analysis results
                    validation_context = {
                        'function_name': func.__name__,
                        'validation_level': validation_level.value,
                        'required_outputs': required_outputs
                    }
                    
                    validation_results = await validation_orchestrator.validate_step(
                        step_name=f"data_analysis_{func.__name__}",
                        data=result,
                        context=validation_context,
                        validation_types=['data_analysis']
                    )
                    
                    # Check validation results
                    for validation_type, report in validation_results.items():
                        if report.result == ValidationResult.FAILED:
                            raise DataAnalysisError(
                                f"Data analysis validation failed: {report.errors}"
                            )
                        elif report.result == ValidationResult.WARNING:
                            logger.warning(f"Data analysis validation warnings: {report.warnings}")
                
                duration = time.time() - start_time
                logger.info(f"✅ Data analysis validation passed for {func.__name__} in {duration:.3f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ Data analysis validation failed for {func.__name__} in {duration:.3f}s: {e}")
                raise DataAnalysisError(f"Data analysis validation failed: {e}") from e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            logger = system_logger.getChild(f"DataAnalysisValidator.{func.__name__}")
            start_time = time.time()
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Validate the result
                if isinstance(result, dict):
                    # Check required outputs
                    if required_outputs:
                        missing_outputs = [output for output in required_outputs if output not in result]
                        if missing_outputs:
                            raise DataAnalysisError(
                                f"Function {func.__name__} missing required outputs: {missing_outputs}"
                            )
                
                duration = time.time() - start_time
                logger.info(f"✅ Data analysis validation passed for {func.__name__} in {duration:.3f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ Data analysis validation failed for {func.__name__} in {duration:.3f}s: {e}")
                raise DataAnalysisError(f"Data analysis validation failed: {e}") from e
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def validate_data_access(
    required_files: Optional[List[str]] = None,
    required_directories: Optional[List[str]] = None,
    validation_level: ValidationLevel = ValidationLevel.CRITICAL
):
    """
    Decorator to validate data access operations.
    
    Args:
        required_files: List of required file paths
        required_directories: List of required directory paths
        validation_level: Validation level to use
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            logger = system_logger.getChild(f"DataAccessValidator.{func.__name__}")
            start_time = time.time()
            
            try:
                # Validate required files before execution
                if required_files:
                    missing_files = [f for f in required_files if not safe_file_exists(f)]
                    if missing_files:
                        raise DataAccessError(
                            f"Function {func.__name__} missing required files: {missing_files}"
                        )
                
                # Validate required directories before execution
                if required_directories:
                    missing_dirs = [d for d in required_directories if not safe_file_exists(d)]
                    if missing_dirs:
                        # Try to create missing directories
                        for dir_path in missing_dirs:
                            try:
                                ensure_directory(dir_path)
                                logger.info(f"Created missing directory: {dir_path}")
                            except Exception as e:
                                raise DataAccessError(
                                    f"Function {func.__name__} cannot create required directory {dir_path}: {e}"
                                )
                
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Validate the result
                validation_context = {
                    'function_name': func.__name__,
                    'validation_level': validation_level.value,
                    'required_files': required_files,
                    'required_directories': required_directories,
                    'file_paths': required_files or [],
                    'data_dir': required_directories[0] if required_directories else None
                }
                
                validation_results = await validation_orchestrator.validate_step(
                    step_name=f"data_access_{func.__name__}",
                    data=result,
                    context=validation_context,
                    validation_types=['data_access']
                )
                
                # Check validation results
                for validation_type, report in validation_results.items():
                    if report.result == ValidationResult.FAILED:
                        raise DataAccessError(
                            f"Data access validation failed: {report.errors}"
                        )
                    elif report.result == ValidationResult.WARNING:
                        logger.warning(f"Data access validation warnings: {report.warnings}")
                
                duration = time.time() - start_time
                logger.info(f"✅ Data access validation passed for {func.__name__} in {duration:.3f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ Data access validation failed for {func.__name__} in {duration:.3f}s: {e}")
                raise DataAccessError(f"Data access validation failed: {e}") from e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            logger = system_logger.getChild(f"DataAccessValidator.{func.__name__}")
            start_time = time.time()
            
            try:
                # Validate required files before execution
                if required_files:
                    missing_files = [f for f in required_files if not safe_file_exists(f)]
                    if missing_files:
                        raise DataAccessError(
                            f"Function {func.__name__} missing required files: {missing_files}"
                        )
                
                # Validate required directories before execution
                if required_directories:
                    missing_dirs = [d for d in required_directories if not safe_file_exists(d)]
                    if missing_dirs:
                        # Try to create missing directories
                        for dir_path in missing_dirs:
                            try:
                                ensure_directory(dir_path)
                                logger.info(f"Created missing directory: {dir_path}")
                            except Exception as e:
                                raise DataAccessError(
                                    f"Function {func.__name__} cannot create required directory {dir_path}: {e}"
                                )
                
                # Execute the function
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.info(f"✅ Data access validation passed for {func.__name__} in {duration:.3f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ Data access validation failed for {func.__name__} in {duration:.3f}s: {e}")
                raise DataAccessError(f"Data access validation failed: {e}") from e
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def validate_model_training(
    required_metrics: Optional[List[str]] = None,
    validation_level: ValidationLevel = ValidationLevel.CRITICAL
):
    """
    Decorator to validate model training operations.
    
    Args:
        required_metrics: List of required training metrics
        validation_level: Validation level to use
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            logger = system_logger.getChild(f"ModelTrainingValidator.{func.__name__}")
            start_time = time.time()
            
            try:
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Validate the result
                if isinstance(result, dict):
                    # Check required metrics
                    if required_metrics:
                        missing_metrics = [metric for metric in required_metrics if metric not in result]
                        if missing_metrics:
                            raise ModelTrainingError(
                                f"Function {func.__name__} missing required metrics: {missing_metrics}"
                            )
                    
                    # Validate training results
                    validation_context = {
                        'function_name': func.__name__,
                        'validation_level': validation_level.value,
                        'required_metrics': required_metrics
                    }
                    
                    validation_results = await validation_orchestrator.validate_step(
                        step_name=f"model_training_{func.__name__}",
                        data=result,
                        context=validation_context,
                        validation_types=['model_training']
                    )
                    
                    # Check validation results
                    for validation_type, report in validation_results.items():
                        if report.result == ValidationResult.FAILED:
                            raise ModelTrainingError(
                                f"Model training validation failed: {report.errors}"
                            )
                        elif report.result == ValidationResult.WARNING:
                            logger.warning(f"Model training validation warnings: {report.warnings}")
                
                duration = time.time() - start_time
                logger.info(f"✅ Model training validation passed for {func.__name__} in {duration:.3f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ Model training validation failed for {func.__name__} in {duration:.3f}s: {e}")
                raise ModelTrainingError(f"Model training validation failed: {e}") from e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            logger = system_logger.getChild(f"ModelTrainingValidator.{func.__name__}")
            start_time = time.time()
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Validate the result
                if isinstance(result, dict):
                    # Check required metrics
                    if required_metrics:
                        missing_metrics = [metric for metric in required_metrics if metric not in result]
                        if missing_metrics:
                            raise ModelTrainingError(
                                f"Function {func.__name__} missing required metrics: {missing_metrics}"
                            )
                
                duration = time.time() - start_time
                logger.info(f"✅ Model training validation passed for {func.__name__} in {duration:.3f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ Model training validation failed for {func.__name__} in {duration:.3f}s: {e}")
                raise ModelTrainingError(f"Model training validation failed: {e}") from e
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def safe_operation(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    fallback_value: Any = None,
    log_errors: bool = True
):
    """
    Decorator to make operations safe with retry logic and error handling.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        fallback_value: Value to return if all retries fail
        log_errors: Whether to log errors
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            logger = system_logger.getChild(f"SafeOperation.{func.__name__}")
            
            for attempt in range(max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"✅ Function {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                    
                except Exception as e:
                    if attempt < max_retries:
                        if log_errors:
                            logger.warning(f"⚠️ Function {func.__name__} failed on attempt {attempt + 1}: {e}")
                        await asyncio.sleep(retry_delay)
                    else:
                        if log_errors:
                            logger.error(f"❌ Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        if fallback_value is not None:
                            logger.info(f"🔄 Returning fallback value for {func.__name__}")
                            return fallback_value
                        raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            logger = system_logger.getChild(f"SafeOperation.{func.__name__}")
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"✅ Function {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                    
                except Exception as e:
                    if attempt < max_retries:
                        if log_errors:
                            logger.warning(f"⚠️ Function {func.__name__} failed on attempt {attempt + 1}: {e}")
                        time.sleep(retry_delay)
                    else:
                        if log_errors:
                            logger.error(f"❌ Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        if fallback_value is not None:
                            logger.info(f"🔄 Returning fallback value for {func.__name__}")
                            return fallback_value
                        raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def performance_monitor(
    log_performance: bool = True,
    performance_threshold: float = 1.0
):
    """
    Decorator to monitor performance of operations.
    
    Args:
        log_performance: Whether to log performance metrics
        performance_threshold: Threshold in seconds for performance warnings
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            logger = system_logger.getChild(f"PerformanceMonitor.{func.__name__}")
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                if log_performance:
                    if duration > performance_threshold:
                        logger.warning(f"⚠️ Function {func.__name__} took {duration:.3f}s (threshold: {performance_threshold}s)")
                    else:
                        logger.info(f"✅ Function {func.__name__} completed in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                if log_performance:
                    logger.error(f"❌ Function {func.__name__} failed after {duration:.3f}s: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            logger = system_logger.getChild(f"PerformanceMonitor.{func.__name__}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if log_performance:
                    if duration > performance_threshold:
                        logger.warning(f"⚠️ Function {func.__name__} took {duration:.3f}s (threshold: {performance_threshold}s)")
                    else:
                        logger.info(f"✅ Function {func.__name__} completed in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                if log_performance:
                    logger.error(f"❌ Function {func.__name__} failed after {duration:.3f}s: {e}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator