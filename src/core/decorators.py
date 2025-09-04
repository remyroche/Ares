#!/usr/bin/env python3
"""
Core Decorators for Pipeline Protection and Monitoring

This module provides comprehensive decorators for:
1. Data protection and access control
2. Error handling and recovery
3. Operation monitoring and logging
4. Performance tracking
5. Security validation
"""

import asyncio
import functools
import time
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Type
from datetime import datetime
import logging
import traceback
import hashlib
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
    safe_json_dump,
    safe_json_load,
)
from src.utils.logger import system_logger
from src.utils.security_framework import SecurityFramework
from src.utils.data_quality_framework import DataQualityFramework

logger = system_logger.getChild("CoreDecorators")


class DataProtectionError(Exception):
    """Exception raised when data protection validation fails."""
    pass


class OperationMonitoringError(Exception):
    """Exception raised when operation monitoring fails."""
    pass


def handles_errors(
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    fallback: Any = None,
    context: str = "unknown",
    log_errors: bool = True,
    reraise: bool = False
):
    """
    Decorator for comprehensive error handling with fallback mechanisms.
    
    Args:
        exceptions: Exception types to catch
        fallback: Fallback value to return on error
        context: Context description for logging
        log_errors: Whether to log errors
        reraise: Whether to reraise exceptions after handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = f"{func.__module__}.{func.__name__}"
            
            try:
                logger.debug(f"🚀 Starting {context}: {func_name}")
                result = await func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.debug(f"✅ Completed {context}: {func_name} in {duration:.3f}s")
                return result
                
            except exceptions as e:
                duration = time.time() - start_time
                error_msg = f"❌ Error in {context}: {func_name} - {str(e)}"
                
                if log_errors:
                    logger.error(error_msg, exc_info=True)
                
                # Log error details
                error_details = {
                    'function': func_name,
                    'context': context,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'duration': duration,
                    'timestamp': get_current_datetime().isoformat(),
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()) if kwargs else []
                }
                
                # Save error details to file
                try:
                    error_log_file = Path("log") / f"error_log_{get_current_datetime().strftime('%Y%m%d')}.json"
                    error_log_file.parent.mkdir(exist_ok=True)
                    
                    # Load existing errors
                    existing_errors = []
                    if error_log_file.exists():
                        try:
                            existing_errors = safe_json_load(error_log_file)
                        except:
                            existing_errors = []
                    
                    # Add new error
                    existing_errors.append(error_details)
                    
                    # Keep only last 1000 errors
                    if len(existing_errors) > 1000:
                        existing_errors = existing_errors[-1000:]
                    
                    # Save updated errors
                    safe_json_dump(existing_errors, error_log_file)
                    
                except Exception as log_error:
                    logger.warning(f"⚠️ Failed to log error details: {log_error}")
                
                if reraise:
                    raise
                
                return fallback
            
            except Exception as e:
                duration = time.time() - start_time
                error_msg = f"💥 Unexpected error in {context}: {func_name} - {str(e)}"
                
                if log_errors:
                    logger.critical(error_msg, exc_info=True)
                
                if reraise:
                    raise
                
                return fallback
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = f"{func.__module__}.{func.__name__}"
            
            try:
                logger.debug(f"🚀 Starting {context}: {func_name}")
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.debug(f"✅ Completed {context}: {func_name} in {duration:.3f}s")
                return result
                
            except exceptions as e:
                duration = time.time() - start_time
                error_msg = f"❌ Error in {context}: {func_name} - {str(e)}"
                
                if log_errors:
                    logger.error(error_msg, exc_info=True)
                
                if reraise:
                    raise
                
                return fallback
            
            except Exception as e:
                duration = time.time() - start_time
                error_msg = f"💥 Unexpected error in {context}: {func_name} - {str(e)}"
                
                if log_errors:
                    logger.critical(error_msg, exc_info=True)
                
                if reraise:
                    raise
                
                return fallback
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def data_protection(
    operation_type: str = "data_operation",
    validate_input: bool = True,
    validate_output: bool = True,
    encrypt_sensitive: bool = False,
    log_access: bool = True
):
    """
    Decorator for data protection and access control.
    
    Args:
        operation_type: Type of data operation
        validate_input: Whether to validate input data
        validate_output: Whether to validate output data
        encrypt_sensitive: Whether to encrypt sensitive data
        log_access: Whether to log data access
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = f"{func.__module__}.{func.__name__}"
            
            try:
                # Initialize security framework
                security = SecurityFramework()
                await security.initialize()
                
                # Log data access
                if log_access:
                    access_log = {
                        'function': func_name,
                        'operation_type': operation_type,
                        'timestamp': get_current_datetime().isoformat(),
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys()) if kwargs else []
                    }
                    logger.info(f"🔒 Data access: {func_name} - {operation_type}")
                
                # Validate input data
                if validate_input:
                    input_validation = await security.validate_data_access(
                        operation_type=operation_type,
                        data_context="input",
                        function_name=func_name
                    )
                    if not input_validation.get('allowed', True):
                        raise DataProtectionError(f"Data access denied: {input_validation.get('reason')}")
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Validate output data
                if validate_output and result is not None:
                    output_validation = await security.validate_data_access(
                        operation_type=operation_type,
                        data_context="output",
                        function_name=func_name
                    )
                    if not output_validation.get('allowed', True):
                        raise DataProtectionError(f"Data output denied: {output_validation.get('reason')}")
                
                # Encrypt sensitive data if requested
                if encrypt_sensitive and result is not None:
                    result = await security.encrypt_sensitive_data(result, operation_type)
                
                duration = time.time() - start_time
                logger.debug(f"🔒 Data protection completed: {func_name} in {duration:.3f}s")
                
                return result
                
            except DataProtectionError:
                raise
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ Data protection error in {func_name}: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = f"{func.__module__}.{func.__name__}"
            
            try:
                # Log data access
                if log_access:
                    logger.info(f"🔒 Data access: {func_name} - {operation_type}")
                
                # Execute function
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.debug(f"🔒 Data protection completed: {func_name} in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ Data protection error in {func_name}: {e}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def operation_monitoring(
    operation_name: str = "unknown_operation",
    track_performance: bool = True,
    track_memory: bool = True,
    track_errors: bool = True,
    save_metrics: bool = True
):
    """
    Decorator for comprehensive operation monitoring.
    
    Args:
        operation_name: Name of the operation for monitoring
        track_performance: Whether to track performance metrics
        track_memory: Whether to track memory usage
        track_errors: Whether to track errors
        save_metrics: Whether to save metrics to file
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = 0
            func_name = f"{func.__module__}.{func.__name__}"
            
            try:
                # Track memory usage
                if track_memory:
                    import psutil
                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                logger.info(f"📊 Monitoring operation: {operation_name} - {func_name}")
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Calculate metrics
                duration = time.time() - start_time
                end_memory = 0
                memory_delta = 0
                
                if track_memory:
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_delta = end_memory - start_memory
                
                # Create metrics
                metrics = {
                    'operation_name': operation_name,
                    'function_name': func_name,
                    'duration': duration,
                    'start_memory_mb': start_memory,
                    'end_memory_mb': end_memory,
                    'memory_delta_mb': memory_delta,
                    'success': True,
                    'timestamp': get_current_datetime().isoformat(),
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()) if kwargs else []
                }
                
                # Save metrics
                if save_metrics:
                    await _save_operation_metrics(metrics)
                
                logger.info(f"📊 Operation completed: {operation_name} - {duration:.3f}s, {memory_delta:+.1f}MB")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                end_memory = 0
                memory_delta = 0
                
                if track_memory:
                    try:
                        import psutil
                        process = psutil.Process()
                        end_memory = process.memory_info().rss / 1024 / 1024  # MB
                        memory_delta = end_memory - start_memory
                    except:
                        pass
                
                # Create error metrics
                error_metrics = {
                    'operation_name': operation_name,
                    'function_name': func_name,
                    'duration': duration,
                    'start_memory_mb': start_memory,
                    'end_memory_mb': end_memory,
                    'memory_delta_mb': memory_delta,
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'timestamp': get_current_datetime().isoformat(),
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()) if kwargs else []
                }
                
                # Save error metrics
                if save_metrics and track_errors:
                    await _save_operation_metrics(error_metrics)
                
                logger.error(f"📊 Operation failed: {operation_name} - {duration:.3f}s, {memory_delta:+.1f}MB - {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = 0
            func_name = f"{func.__module__}.{func.__name__}"
            
            try:
                # Track memory usage
                if track_memory:
                    try:
                        import psutil
                        process = psutil.Process()
                        start_memory = process.memory_info().rss / 1024 / 1024  # MB
                    except:
                        pass
                
                logger.info(f"📊 Monitoring operation: {operation_name} - {func_name}")
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate metrics
                duration = time.time() - start_time
                end_memory = 0
                memory_delta = 0
                
                if track_memory:
                    try:
                        end_memory = process.memory_info().rss / 1024 / 1024  # MB
                        memory_delta = end_memory - start_memory
                    except:
                        pass
                
                # Create metrics
                metrics = {
                    'operation_name': operation_name,
                    'function_name': func_name,
                    'duration': duration,
                    'start_memory_mb': start_memory,
                    'end_memory_mb': end_memory,
                    'memory_delta_mb': memory_delta,
                    'success': True,
                    'timestamp': get_current_datetime().isoformat(),
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()) if kwargs else []
                }
                
                # Save metrics
                if save_metrics:
                    asyncio.create_task(_save_operation_metrics(metrics))
                
                logger.info(f"📊 Operation completed: {operation_name} - {duration:.3f}s, {memory_delta:+.1f}MB")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                end_memory = 0
                memory_delta = 0
                
                if track_memory:
                    try:
                        end_memory = process.memory_info().rss / 1024 / 1024  # MB
                        memory_delta = end_memory - start_memory
                    except:
                        pass
                
                # Create error metrics
                error_metrics = {
                    'operation_name': operation_name,
                    'function_name': func_name,
                    'duration': duration,
                    'start_memory_mb': start_memory,
                    'end_memory_mb': end_memory,
                    'memory_delta_mb': memory_delta,
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'timestamp': get_current_datetime().isoformat(),
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()) if kwargs else []
                }
                
                # Save error metrics
                if save_metrics and track_errors:
                    asyncio.create_task(_save_operation_metrics(error_metrics))
                
                logger.error(f"📊 Operation failed: {operation_name} - {duration:.3f}s, {memory_delta:+.1f}MB - {e}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def validate_data_format(
    expected_format: str = "dataframe",
    required_columns: Optional[List[str]] = None,
    data_types: Optional[Dict[str, str]] = None
):
    """
    Decorator for data format validation.
    
    Args:
        expected_format: Expected data format (dataframe, dict, list, etc.)
        required_columns: Required columns for DataFrame
        data_types: Expected data types for columns
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Validate input data format
            for arg in args:
                if hasattr(arg, 'shape'):  # Likely a DataFrame
                    if expected_format == "dataframe":
                        await _validate_dataframe_format(arg, required_columns, data_types)
                elif isinstance(arg, dict):
                    if expected_format == "dict":
                        await _validate_dict_format(arg, required_columns)
                elif isinstance(arg, list):
                    if expected_format == "list":
                        await _validate_list_format(arg)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Validate output data format
            if result is not None:
                if hasattr(result, 'shape'):  # Likely a DataFrame
                    if expected_format == "dataframe":
                        await _validate_dataframe_format(result, required_columns, data_types)
                elif isinstance(result, dict):
                    if expected_format == "dict":
                        await _validate_dict_format(result, required_columns)
                elif isinstance(result, list):
                    if expected_format == "list":
                        await _validate_list_format(result)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Validate input data format
            for arg in args:
                if hasattr(arg, 'shape'):  # Likely a DataFrame
                    if expected_format == "dataframe":
                        _validate_dataframe_format_sync(arg, required_columns, data_types)
                elif isinstance(arg, dict):
                    if expected_format == "dict":
                        _validate_dict_format_sync(arg, required_columns)
                elif isinstance(arg, list):
                    if expected_format == "list":
                        _validate_list_format_sync(arg)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Validate output data format
            if result is not None:
                if hasattr(result, 'shape'):  # Likely a DataFrame
                    if expected_format == "dataframe":
                        _validate_dataframe_format_sync(result, required_columns, data_types)
                elif isinstance(result, dict):
                    if expected_format == "dict":
                        _validate_dict_format_sync(result, required_columns)
                elif isinstance(result, list):
                    if expected_format == "list":
                        _validate_list_format_sync(result)
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


async def _save_operation_metrics(metrics: Dict[str, Any]) -> None:
    """Save operation metrics to file."""
    try:
        metrics_file = Path("log") / f"operation_metrics_{get_current_datetime().strftime('%Y%m%d')}.json"
        metrics_file.parent.mkdir(exist_ok=True)
        
        # Load existing metrics
        existing_metrics = []
        if metrics_file.exists():
            try:
                existing_metrics = safe_json_load(metrics_file)
            except:
                existing_metrics = []
        
        # Add new metrics
        existing_metrics.append(metrics)
        
        # Keep only last 10000 metrics
        if len(existing_metrics) > 10000:
            existing_metrics = existing_metrics[-10000:]
        
        # Save updated metrics
        safe_json_dump(existing_metrics, metrics_file)
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to save operation metrics: {e}")


async def _validate_dataframe_format(data, required_columns: Optional[List[str]], data_types: Optional[Dict[str, str]]) -> None:
    """Validate DataFrame format."""
    import pandas as pd
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Expected DataFrame")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if data_types:
        for col, expected_type in data_types.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if expected_type not in actual_type:
                    raise ValueError(f"Column {col} has type {actual_type}, expected {expected_type}")


def _validate_dataframe_format_sync(data, required_columns: Optional[List[str]], data_types: Optional[Dict[str, str]]) -> None:
    """Validate DataFrame format (sync version)."""
    import pandas as pd
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Expected DataFrame")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if data_types:
        for col, expected_type in data_types.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if expected_type not in actual_type:
                    raise ValueError(f"Column {col} has type {actual_type}, expected {expected_type}")


async def _validate_dict_format(data: Dict[str, Any], required_keys: Optional[List[str]]) -> None:
    """Validate dictionary format."""
    if required_keys:
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")


def _validate_dict_format_sync(data: Dict[str, Any], required_keys: Optional[List[str]]) -> None:
    """Validate dictionary format (sync version)."""
    if required_keys:
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")


async def _validate_list_format(data: List[Any]) -> None:
    """Validate list format."""
    if not isinstance(data, list):
        raise ValueError("Expected list")


def _validate_list_format_sync(data: List[Any]) -> None:
    """Validate list format (sync version)."""
    if not isinstance(data, list):
        raise ValueError("Expected list")


# Composite decorator for comprehensive protection
def comprehensive_protection(
    operation_name: str = "unknown_operation",
    operation_type: str = "data_operation",
    context: str = "unknown",
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    fallback: Any = None
):
    """
    Composite decorator that combines all protection mechanisms.
    
    Args:
        operation_name: Name of the operation for monitoring
        operation_type: Type of data operation
        context: Context description for logging
        exceptions: Exception types to catch
        fallback: Fallback value to return on error
    """
    def decorator(func: Callable) -> Callable:
        # Apply all decorators in sequence
        func = handles_errors(exceptions, fallback, context)(func)
        func = data_protection(operation_type)(func)
        func = operation_monitoring(operation_name)(func)
        
        return func
    
    return decorator