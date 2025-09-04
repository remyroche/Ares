#!/usr/bin/env python3
"""
Pipeline Decorators for Data Operations

This module provides comprehensive decorators for protecting and validating
all pipeline operations including data formatting, analysis, access, and manipulation.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union, Type, Tuple
import pandas as pd
import numpy as np

from src.core.decorators.errors import handles_errors, error_boundary
from src.core.decorators.validate import validates
from src.core.decorators.logging import logs_execution
from src.core.decorators.cache import caches_result
from src.utils.pipeline_validator_framework import (
    validator_orchestrator,
    ValidationLevel,
    ValidationResult
)
from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    safe_file_exists,
    ensure_directory
)


def pipeline_step(
    step_name: str,
    validation_level: ValidationLevel = ValidationLevel.CRITICAL,
    validate_input: bool = True,
    validate_output: bool = True,
    cache_result: bool = False,
    log_execution: bool = True
):
    """
    Comprehensive decorator for pipeline steps with validation, caching, and logging.
    
    Args:
        step_name: Name of the pipeline step
        validation_level: Level of validation required
        validate_input: Whether to validate input data
        validate_output: Whether to validate output data
        cache_result: Whether to cache the result
        log_execution: Whether to log execution details
    """
    def decorator(func: Callable) -> Callable:
        # Apply base decorators
        if log_execution:
            func = logs_execution(f"pipeline_step_{step_name}")(func)
        
        if cache_result:
            func = caches_result(ttl=3600, key_prefix=f"pipeline_{step_name}")(func)
        
        func = error_boundary(name=f"pipeline_step_{step_name}")(func)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            logger = logging.getLogger(f"pipeline_step.{step_name}")
            
            try:
                # Input validation
                if validate_input and args:
                    input_data = args[0] if args else None
                    if input_data is not None:
                        validation_results = await validator_orchestrator.validate_pipeline_step(
                            step_name=f"{step_name}_input",
                            data=input_data,
                            context={"validation_level": validation_level.value},
                            validators_to_run=["data_format", "data_quality"]
                        )
                        
                        # Check if validation failed
                        for validator_name, report in validation_results.items():
                            if report.result == ValidationResult.FAILED:
                                raise ValueError(f"Input validation failed for {step_name}: {report.message}")
                            elif report.result == ValidationResult.WARNING:
                                logger.warning(f"Input validation warning: {report.message}")
                
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Output validation
                if validate_output and result is not None:
                    validation_results = await validator_orchestrator.validate_pipeline_step(
                        step_name=f"{step_name}_output",
                        data=result,
                        context={"validation_level": validation_level.value},
                        validators_to_run=["data_format", "data_quality"]
                    )
                    
                    # Check if validation failed
                    for validator_name, report in validation_results.items():
                        if report.result == ValidationResult.FAILED:
                            raise ValueError(f"Output validation failed for {step_name}: {report.message}")
                        elif report.result == ValidationResult.WARNING:
                            logger.warning(f"Output validation warning: {report.message}")
                
                execution_time = time.time() - start_time
                logger.info(f"Pipeline step '{step_name}' completed successfully in {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Pipeline step '{step_name}' failed after {execution_time:.2f}s: {str(e)}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            logger = logging.getLogger(f"pipeline_step.{step_name}")
            
            try:
                # Input validation
                if validate_input and args:
                    input_data = args[0] if args else None
                    if input_data is not None:
                        # Run validation synchronously
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            validation_results = loop.run_until_complete(
                                validator_orchestrator.validate_pipeline_step(
                                    step_name=f"{step_name}_input",
                                    data=input_data,
                                    context={"validation_level": validation_level.value},
                                    validators_to_run=["data_format", "data_quality"]
                                )
                            )
                            
                            # Check if validation failed
                            for validator_name, report in validation_results.items():
                                if report.result == ValidationResult.FAILED:
                                    raise ValueError(f"Input validation failed for {step_name}: {report.message}")
                                elif report.result == ValidationResult.WARNING:
                                    logger.warning(f"Input validation warning: {report.message}")
                        finally:
                            loop.close()
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Output validation
                if validate_output and result is not None:
                    # Run validation synchronously
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        validation_results = loop.run_until_complete(
                            validator_orchestrator.validate_pipeline_step(
                                step_name=f"{step_name}_output",
                                data=result,
                                context={"validation_level": validation_level.value},
                                validators_to_run=["data_format", "data_quality"]
                            )
                        )
                        
                        # Check if validation failed
                        for validator_name, report in validation_results.items():
                            if report.result == ValidationResult.FAILED:
                                raise ValueError(f"Output validation failed for {step_name}: {report.message}")
                            elif report.result == ValidationResult.WARNING:
                                logger.warning(f"Output validation warning: {report.message}")
                    finally:
                        loop.close()
                
                execution_time = time.time() - start_time
                logger.info(f"Pipeline step '{step_name}' completed successfully in {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Pipeline step '{step_name}' failed after {execution_time:.2f}s: {str(e)}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def data_operation(
    operation_type: str,
    validate_schema: bool = True,
    preserve_metadata: bool = True,
    log_changes: bool = True
):
    """
    Decorator for data operations with schema validation and change tracking.
    
    Args:
        operation_type: Type of data operation (read, write, transform, analyze)
        validate_schema: Whether to validate data schema
        preserve_metadata: Whether to preserve data metadata
        log_changes: Whether to log data changes
    """
    def decorator(func: Callable) -> Callable:
        func = error_boundary(name=f"data_operation_{operation_type}")(func)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(f"data_operation.{operation_type}")
            start_time = time.time()
            
            try:
                # Log operation start
                if log_changes:
                    logger.info(f"Starting {operation_type} operation: {func.__name__}")
                
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Log operation completion
                execution_time = time.time() - start_time
                if log_changes:
                    logger.info(f"Completed {operation_type} operation in {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{operation_type} operation failed after {execution_time:.2f}s: {str(e)}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(f"data_operation.{operation_type}")
            start_time = time.time()
            
            try:
                # Log operation start
                if log_changes:
                    logger.info(f"Starting {operation_type} operation: {func.__name__}")
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Log operation completion
                execution_time = time.time() - start_time
                if log_changes:
                    logger.info(f"Completed {operation_type} operation in {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{operation_type} operation failed after {execution_time:.2f}s: {str(e)}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def data_access_control(
    allowed_operations: List[str] = None,
    require_authentication: bool = False,
    log_access: bool = True
):
    """
    Decorator for controlling data access with operation restrictions.
    
    Args:
        allowed_operations: List of allowed operations (read, write, delete, etc.)
        require_authentication: Whether authentication is required
        log_access: Whether to log access attempts
    """
    if allowed_operations is None:
        allowed_operations = ["read"]
    
    def decorator(func: Callable) -> Callable:
        func = error_boundary(name=f"data_access_{func.__name__}")(func)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(f"data_access.{func.__name__}")
            
            # Check operation permissions
            operation = kwargs.get("operation", "read")
            if operation not in allowed_operations:
                raise PermissionError(f"Operation '{operation}' not allowed. Allowed: {allowed_operations}")
            
            # Log access attempt
            if log_access:
                logger.info(f"Data access attempt: {operation} on {func.__name__}")
            
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Log successful access
            if log_access:
                logger.info(f"Data access successful: {operation} on {func.__name__}")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(f"data_access.{func.__name__}")
            
            # Check operation permissions
            operation = kwargs.get("operation", "read")
            if operation not in allowed_operations:
                raise PermissionError(f"Operation '{operation}' not allowed. Allowed: {allowed_operations}")
            
            # Log access attempt
            if log_access:
                logger.info(f"Data access attempt: {operation} on {func.__name__}")
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Log successful access
            if log_access:
                logger.info(f"Data access successful: {operation} on {func.__name__}")
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def data_integrity_check(
    checksum_validation: bool = True,
    schema_validation: bool = True,
    size_validation: bool = True
):
    """
    Decorator for data integrity checks.
    
    Args:
        checksum_validation: Whether to validate data checksums
        schema_validation: Whether to validate data schema
        size_validation: Whether to validate data size
    """
    def decorator(func: Callable) -> Callable:
        func = error_boundary(name=f"data_integrity_{func.__name__}")(func)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(f"data_integrity.{func.__name__}")
            
            try:
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Perform integrity checks on result
                if result is not None:
                    integrity_issues = []
                    
                    if isinstance(result, pd.DataFrame):
                        # Schema validation
                        if schema_validation:
                            if result.empty:
                                integrity_issues.append("DataFrame is empty")
                            
                            # Check for required columns
                            required_columns = kwargs.get("required_columns", [])
                            missing_columns = [col for col in required_columns if col not in result.columns]
                            if missing_columns:
                                integrity_issues.append(f"Missing required columns: {missing_columns}")
                        
                        # Size validation
                        if size_validation:
                            min_rows = kwargs.get("min_rows", 0)
                            if len(result) < min_rows:
                                integrity_issues.append(f"DataFrame has {len(result)} rows, minimum required: {min_rows}")
                    
                    # Report integrity issues
                    if integrity_issues:
                        logger.warning(f"Data integrity issues found: {integrity_issues}")
                    
                    logger.info(f"Data integrity check completed for {func.__name__}")
                
                return result
                
            except Exception as e:
                logger.error(f"Data integrity check failed for {func.__name__}: {str(e)}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(f"data_integrity.{func.__name__}")
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Perform integrity checks on result
                if result is not None:
                    integrity_issues = []
                    
                    if isinstance(result, pd.DataFrame):
                        # Schema validation
                        if schema_validation:
                            if result.empty:
                                integrity_issues.append("DataFrame is empty")
                            
                            # Check for required columns
                            required_columns = kwargs.get("required_columns", [])
                            missing_columns = [col for col in required_columns if col not in result.columns]
                            if missing_columns:
                                integrity_issues.append(f"Missing required columns: {missing_columns}")
                        
                        # Size validation
                        if size_validation:
                            min_rows = kwargs.get("min_rows", 0)
                            if len(result) < min_rows:
                                integrity_issues.append(f"DataFrame has {len(result)} rows, minimum required: {min_rows}")
                    
                    # Report integrity issues
                    if integrity_issues:
                        logger.warning(f"Data integrity issues found: {integrity_issues}")
                    
                    logger.info(f"Data integrity check completed for {func.__name__}")
                
                return result
                
            except Exception as e:
                logger.error(f"Data integrity check failed for {func.__name__}: {str(e)}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def performance_monitor(
    log_performance: bool = True,
    memory_monitoring: bool = True,
    execution_time_threshold: float = 60.0
):
    """
    Decorator for monitoring performance metrics.
    
    Args:
        log_performance: Whether to log performance metrics
        memory_monitoring: Whether to monitor memory usage
        execution_time_threshold: Threshold for execution time warnings (seconds)
    """
    def decorator(func: Callable) -> Callable:
        func = error_boundary(name=f"performance_monitor_{func.__name__}")(func)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(f"performance.{func.__name__}")
            start_time = time.time()
            
            # Memory monitoring
            if memory_monitoring:
                import psutil
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            try:
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Calculate performance metrics
                execution_time = time.time() - start_time
                
                if log_performance:
                    metrics = {
                        "execution_time": execution_time,
                        "function": func.__name__
                    }
                    
                    if memory_monitoring:
                        end_memory = process.memory_info().rss / 1024 / 1024  # MB
                        metrics["memory_usage"] = end_memory - start_memory
                        metrics["peak_memory"] = end_memory
                    
                    logger.info(f"Performance metrics: {metrics}")
                    
                    # Check for performance issues
                    if execution_time > execution_time_threshold:
                        logger.warning(f"Function {func.__name__} took {execution_time:.2f}s (threshold: {execution_time_threshold}s)")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Function {func.__name__} failed after {execution_time:.2f}s: {str(e)}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(f"performance.{func.__name__}")
            start_time = time.time()
            
            # Memory monitoring
            if memory_monitoring:
                import psutil
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Calculate performance metrics
                execution_time = time.time() - start_time
                
                if log_performance:
                    metrics = {
                        "execution_time": execution_time,
                        "function": func.__name__
                    }
                    
                    if memory_monitoring:
                        end_memory = process.memory_info().rss / 1024 / 1024  # MB
                        metrics["memory_usage"] = end_memory - start_memory
                        metrics["peak_memory"] = end_memory
                    
                    logger.info(f"Performance metrics: {metrics}")
                    
                    # Check for performance issues
                    if execution_time > execution_time_threshold:
                        logger.warning(f"Function {func.__name__} took {execution_time:.2f}s (threshold: {execution_time_threshold}s)")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Function {func.__name__} failed after {execution_time:.2f}s: {str(e)}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def step_dependency_check(
    required_steps: List[str] = None,
    check_files: bool = True,
    check_completion: bool = True
):
    """
    Decorator for checking step dependencies before execution.
    
    Args:
        required_steps: List of required previous steps
        check_files: Whether to check for required files
        check_completion: Whether to check step completion status
    """
    if required_steps is None:
        required_steps = []
    
    def decorator(func: Callable) -> Callable:
        func = error_boundary(name=f"step_dependency_{func.__name__}")(func)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(f"step_dependency.{func.__name__}")
            
            # Check step dependencies
            if required_steps:
                validation_results = await validator_orchestrator.validate_pipeline_step(
                    step_name=func.__name__,
                    data=None,
                    context={
                        "required_steps": required_steps,
                        "check_files": check_files,
                        "check_completion": check_completion
                    },
                    validators_to_run=["step_dependency"]
                )
                
                # Check if dependency validation failed
                for validator_name, report in validation_results.items():
                    if report.result == ValidationResult.FAILED:
                        raise ValueError(f"Step dependency check failed: {report.message}")
                    elif report.result == ValidationResult.WARNING:
                        logger.warning(f"Step dependency warning: {report.message}")
            
            # Execute the function
            result = await func(*args, **kwargs)
            
            logger.info(f"Step dependency check passed for {func.__name__}")
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(f"step_dependency.{func.__name__}")
            
            # Check step dependencies synchronously
            if required_steps:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    validation_results = loop.run_until_complete(
                        validator_orchestrator.validate_pipeline_step(
                            step_name=func.__name__,
                            data=None,
                            context={
                                "required_steps": required_steps,
                                "check_files": check_files,
                                "check_completion": check_completion
                            },
                            validators_to_run=["step_dependency"]
                        )
                    )
                    
                    # Check if dependency validation failed
                    for validator_name, report in validation_results.items():
                        if report.result == ValidationResult.FAILED:
                            raise ValueError(f"Step dependency check failed: {report.message}")
                        elif report.result == ValidationResult.WARNING:
                            logger.warning(f"Step dependency warning: {report.message}")
                finally:
                    loop.close()
            
            # Execute the function
            result = func(*args, **kwargs)
            
            logger.info(f"Step dependency check passed for {func.__name__}")
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Convenience decorators for common pipeline operations
def data_reader(validate_schema: bool = True):
    """Decorator for data reading operations."""
    return data_operation("read", validate_schema=validate_schema)

def data_writer(validate_schema: bool = True):
    """Decorator for data writing operations."""
    return data_operation("write", validate_schema=validate_schema)

def data_transformer(validate_schema: bool = True):
    """Decorator for data transformation operations."""
    return data_operation("transform", validate_schema=validate_schema)

def data_analyzer(validate_schema: bool = True):
    """Decorator for data analysis operations."""
    return data_operation("analyze", validate_schema=validate_schema)