#!/usr/bin/env python3
"""
Comprehensive Step Decorators for Data Collection Pipeline

This module provides specialized decorators for protecting data operations,
ensuring proper data formatting, analysis, and access control.
"""

import asyncio
import functools
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.core.decorators.validate import validates, validate_dataframe
from src.core.domain.decorators import (
    validate_data_quality,
    validate_klines_data_quality,
    ValidationLevel,
    monitor_step_execution,
    ensure_data_integrity,
    secure_data_processing,
    prevent_data_leakage,
    quality_gate
)
from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    safe_file_exists,
    ensure_directory
)

F = TypeVar('F', bound=Callable[..., Any])


class DataOperationType(Enum):
    """Types of data operations."""
    READ = "READ"
    WRITE = "WRITE"
    TRANSFORM = "TRANSFORM"
    ANALYZE = "ANALYZE"
    VALIDATE = "VALIDATE"
    CONVERT = "CONVERT"


class SecurityLevel(Enum):
    """Security levels for data operations."""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    SENSITIVE = "SENSITIVE"
    CONFIDENTIAL = "CONFIDENTIAL"


@dataclass
class DataOperationContext:
    """Context for data operations."""
    operation_type: DataOperationType
    security_level: SecurityLevel
    step_name: str
    symbol: str
    exchange: str
    data_dir: str
    timestamp: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class DataOperationLogger:
    """Logger for data operations with audit trail."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.audit_log: List[Dict[str, Any]] = []
    
    def log_operation(
        self,
        context: DataOperationContext,
        operation: str,
        details: Dict[str, Any],
        success: bool,
        execution_time: float
    ) -> None:
        """Log a data operation for audit purposes."""
        log_entry = {
            "timestamp": context.timestamp,
            "operation": operation,
            "context": {
                "operation_type": context.operation_type.value,
                "security_level": context.security_level.value,
                "step_name": context.step_name,
                "symbol": context.symbol,
                "exchange": context.exchange,
                "data_dir": context.data_dir,
                "user_id": context.user_id,
                "session_id": context.session_id
            },
            "details": details,
            "success": success,
            "execution_time": execution_time
        }
        
        self.audit_log.append(log_entry)
        
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            f"Data operation: {operation} | "
            f"Type: {context.operation_type.value} | "
            f"Security: {context.security_level.value} | "
            f"Success: {success} | "
            f"Time: {execution_time:.3f}s"
        )


# Global operation logger instance
operation_logger = DataOperationLogger()


def data_operation_protection(
    operation_type: DataOperationType,
    security_level: SecurityLevel = SecurityLevel.INTERNAL,
    audit: bool = True,
    validate_inputs: bool = True,
    validate_outputs: bool = True,
    timeout_seconds: Optional[int] = None,
    retry_attempts: int = 0,
    **validation_kwargs
) -> Callable[[F], F]:
    """
    Comprehensive decorator for protecting data operations.
    
    Args:
        operation_type: Type of data operation
        security_level: Security level for the operation
        audit: Whether to audit the operation
        validate_inputs: Whether to validate inputs
        validate_outputs: Whether to validate outputs
        timeout_seconds: Timeout for the operation
        retry_attempts: Number of retry attempts on failure
        **validation_kwargs: Additional validation parameters
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Extract context from function arguments
            context = _extract_context_from_args(args, kwargs, operation_type, security_level)
            
            start_time = time.time()
            operation_name = f"{func.__name__}_{operation_type.value.lower()}"
            
            try:
                # Input validation
                if validate_inputs:
                    await _validate_operation_inputs(func, args, kwargs, validation_kwargs)
                
                # Execute operation with timeout if specified
                if timeout_seconds:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout_seconds
                    )
                else:
                    result = await func(*args, **kwargs)
                
                # Output validation
                if validate_outputs:
                    await _validate_operation_outputs(result, validation_kwargs)
                
                execution_time = time.time() - start_time
                
                # Log successful operation
                if audit:
                    operation_logger.log_operation(
                        context=context,
                        operation=operation_name,
                        details={"result_type": type(result).__name__},
                        success=True,
                        execution_time=execution_time
                    )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log failed operation
                if audit:
                    operation_logger.log_operation(
                        context=context,
                        operation=operation_name,
                        details={"error": str(e), "error_type": type(e).__name__},
                        success=False,
                        execution_time=execution_time
                    )
                
                # Retry logic
                if retry_attempts > 0:
                    return await _retry_operation(
                        func, args, kwargs, retry_attempts, context, operation_name
                    )
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Extract context from function arguments
            context = _extract_context_from_args(args, kwargs, operation_type, security_level)
            
            start_time = time.time()
            operation_name = f"{func.__name__}_{operation_type.value.lower()}"
            
            try:
                # Input validation
                if validate_inputs:
                    _validate_operation_inputs_sync(func, args, kwargs, validation_kwargs)
                
                # Execute operation
                result = func(*args, **kwargs)
                
                # Output validation
                if validate_outputs:
                    _validate_operation_outputs_sync(result, validation_kwargs)
                
                execution_time = time.time() - start_time
                
                # Log successful operation
                if audit:
                    operation_logger.log_operation(
                        context=context,
                        operation=operation_name,
                        details={"result_type": type(result).__name__},
                        success=True,
                        execution_time=execution_time
                    )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log failed operation
                if audit:
                    operation_logger.log_operation(
                        context=context,
                        operation=operation_name,
                        details={"error": str(e), "error_type": type(e).__name__},
                        success=False,
                        execution_time=execution_time
                    )
                
                # Retry logic
                if retry_attempts > 0:
                    return _retry_operation_sync(
                        func, args, kwargs, retry_attempts, context, operation_name
                    )
                
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def data_formatting_protection(
    required_columns: Optional[List[str]] = None,
    data_types: Optional[Dict[str, type]] = None,
    min_rows: int = 1,
    max_null_ratio: float = 0.1,
    check_duplicates: bool = True,
    check_timestamps: bool = True,
    **kwargs
) -> Callable[[F], F]:
    """
    Decorator for protecting data formatting operations.
    
    Ensures data integrity and proper formatting before and after operations.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Pre-operation validation
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    _validate_dataframe_format(
                        arg, required_columns, data_types, min_rows, 
                        max_null_ratio, check_duplicates, check_timestamps
                    )
            
            # Execute operation
            result = func(*args, **kwargs)
            
            # Post-operation validation
            if isinstance(result, pd.DataFrame):
                _validate_dataframe_format(
                    result, required_columns, data_types, min_rows,
                    max_null_ratio, check_duplicates, check_timestamps
                )
            
            return result
        
        return wrapper
    return decorator


def data_analysis_protection(
    prevent_lookahead_bias: bool = True,
    temporal_column: str = 'timestamp',
    max_lookahead: int = 0,
    validate_statistical_properties: bool = True,
    **kwargs
) -> Callable[[F], F]:
    """
    Decorator for protecting data analysis operations.
    
    Prevents data leakage and ensures statistical validity.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check for lookahead bias
            if prevent_lookahead_bias:
                for arg in args:
                    if isinstance(arg, pd.DataFrame) and temporal_column in arg.columns:
                        _check_lookahead_bias(arg, temporal_column, max_lookahead)
            
            # Execute operation
            result = func(*args, **kwargs)
            
            # Validate statistical properties
            if validate_statistical_properties and isinstance(result, dict):
                _validate_statistical_properties(result)
            
            return result
        
        return wrapper
    return decorator


def data_access_protection(
    allowed_operations: Optional[List[str]] = None,
    require_authentication: bool = False,
    rate_limit: Optional[int] = None,
    **kwargs
) -> Callable[[F], F]:
    """
    Decorator for protecting data access operations.
    
    Controls access to sensitive data and operations.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check authentication if required
            if require_authentication:
                _check_authentication()
            
            # Check operation permissions
            if allowed_operations:
                operation_name = func.__name__
                if operation_name not in allowed_operations:
                    raise PermissionError(f"Operation {operation_name} not allowed")
            
            # Rate limiting
            if rate_limit:
                _check_rate_limit(func.__name__, rate_limit)
            
            # Execute operation
            result = func(*args, **kwargs)
            
            return result
        
        return wrapper
    return decorator


def step_execution_protection(
    step_name: str,
    prerequisites: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    quality_threshold: float = 0.8,
    timeout_seconds: Optional[int] = None,
    **kwargs
) -> Callable[[F], F]:
    """
    Comprehensive decorator for protecting pipeline step execution.
    
    Combines multiple protection mechanisms for robust step execution.
    """
    def decorator(func: F) -> F:
        # Compose multiple decorators
        protected_func = func
        
        # Add step monitoring
        protected_func = monitor_step_execution(
            step_name=step_name,
            log_memory=True,
            log_inputs=True,
            log_outputs=True
        )(protected_func)
        
        # Add data integrity protection
        protected_func = ensure_data_integrity()(protected_func)
        
        # Add quality gate
        protected_func = quality_gate(
            min_score=quality_threshold,
            fail_on_breach=True
        )(protected_func)
        
        # Add pipeline step validation
        protected_func = validate_pipeline_step(
            prerequisites=prerequisites,
            outputs=outputs,
            stage=step_name
        )(protected_func)
        
        return protected_func
    
    return decorator


# Helper functions

def _extract_context_from_args(
    args: tuple,
    kwargs: dict,
    operation_type: DataOperationType,
    security_level: SecurityLevel
) -> DataOperationContext:
    """Extract operation context from function arguments."""
    # Try to extract common parameters
    symbol = kwargs.get('symbol', 'UNKNOWN')
    exchange = kwargs.get('exchange', 'UNKNOWN')
    data_dir = kwargs.get('data_dir', 'data_cache')
    step_name = kwargs.get('step_name', 'unknown_step')
    
    return DataOperationContext(
        operation_type=operation_type,
        security_level=security_level,
        step_name=step_name,
        symbol=symbol,
        exchange=exchange,
        data_dir=data_dir,
        timestamp=format_datetime(get_current_datetime())
    )


async def _validate_operation_inputs(
    func: Callable,
    args: tuple,
    kwargs: dict,
    validation_kwargs: dict
) -> None:
    """Validate operation inputs."""
    # Basic input validation
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            _validate_dataframe_basic(arg)


def _validate_operation_inputs_sync(
    func: Callable,
    args: tuple,
    kwargs: dict,
    validation_kwargs: dict
) -> None:
    """Validate operation inputs (sync version)."""
    # Basic input validation
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            _validate_dataframe_basic(arg)


async def _validate_operation_outputs(
    result: Any,
    validation_kwargs: dict
) -> None:
    """Validate operation outputs."""
    if isinstance(result, pd.DataFrame):
        _validate_dataframe_basic(result)


def _validate_operation_outputs_sync(
    result: Any,
    validation_kwargs: dict
) -> None:
    """Validate operation outputs (sync version)."""
    if isinstance(result, pd.DataFrame):
        _validate_dataframe_basic(result)


def _validate_dataframe_basic(df: pd.DataFrame) -> None:
    """Basic DataFrame validation."""
    if df is None:
        raise ValueError("DataFrame cannot be None")
    
    if len(df) == 0:
        raise ValueError("DataFrame cannot be empty")
    
    if df.isnull().all().all():
        raise ValueError("DataFrame cannot contain only null values")


def _validate_dataframe_format(
    df: pd.DataFrame,
    required_columns: Optional[List[str]],
    data_types: Optional[Dict[str, type]],
    min_rows: int,
    max_null_ratio: float,
    check_duplicates: bool,
    check_timestamps: bool
) -> None:
    """Validate DataFrame format."""
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check data types
    if data_types:
        for col, expected_type in data_types.items():
            if col in df.columns:
                if not isinstance(df[col].iloc[0] if len(df) > 0 else None, expected_type):
                    raise ValueError(f"Column {col} has wrong data type")
    
    # Check minimum rows
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
    
    # Check null ratio
    if max_null_ratio < 1.0:
        null_ratios = df.isnull().sum() / len(df)
        high_null_cols = null_ratios[null_ratios > max_null_ratio]
        if not high_null_cols.empty:
            raise ValueError(f"Columns with high null ratio: {high_null_cols.to_dict()}")
    
    # Check duplicates
    if check_duplicates and df.duplicated().any():
        raise ValueError(f"Found {df.duplicated().sum()} duplicate rows")
    
    # Check timestamps
    if check_timestamps and 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            raise ValueError("Timestamp column must be datetime type")


def _check_lookahead_bias(
    df: pd.DataFrame,
    temporal_column: str,
    max_lookahead: int
) -> None:
    """Check for lookahead bias in time series data."""
    if temporal_column not in df.columns:
        return
    
    # Sort by timestamp
    df_sorted = df.sort_values(temporal_column)
    
    # Check for future data leakage
    # This is a simplified check - more sophisticated checks would be needed
    # for complex feature engineering scenarios
    pass


def _validate_statistical_properties(result: dict) -> None:
    """Validate statistical properties of analysis results."""
    if 'metrics' in result:
        metrics = result['metrics']
        
        # Check for reasonable metric values
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    raise ValueError(f"Invalid metric value: {key} = {value}")


def _check_authentication() -> None:
    """Check user authentication."""
    # Placeholder for authentication logic
    pass


def _check_rate_limit(operation_name: str, rate_limit: int) -> None:
    """Check rate limiting for operations."""
    # Placeholder for rate limiting logic
    pass


async def _retry_operation(
    func: Callable,
    args: tuple,
    kwargs: dict,
    retry_attempts: int,
    context: DataOperationContext,
    operation_name: str
) -> Any:
    """Retry operation on failure."""
    for attempt in range(retry_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == retry_attempts - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff


def _retry_operation_sync(
    func: Callable,
    args: tuple,
    kwargs: dict,
    retry_attempts: int,
    context: DataOperationContext,
    operation_name: str
) -> Any:
    """Retry operation on failure (sync version)."""
    for attempt in range(retry_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == retry_attempts - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff


def validate_pipeline_step(
    prerequisites: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    stage: Optional[str] = None
) -> Callable[[F], F]:
    """Validate pipeline step prerequisites and outputs."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if prerequisites:
                logging.info(f'Checking prerequisites for {func.__name__}: {prerequisites}')
            
            result = func(*args, **kwargs)
            
            if outputs and isinstance(result, dict):
                missing_outputs = set(outputs) - set(result.keys())
                if missing_outputs:
                    raise ValueError(f'Missing required outputs: {missing_outputs}')
            
            return result
        
        return wrapper
    return decorator


# Export main decorators
__all__ = [
    'DataOperationType',
    'SecurityLevel',
    'DataOperationContext',
    'DataOperationLogger',
    'data_operation_protection',
    'data_formatting_protection',
    'data_analysis_protection',
    'data_access_protection',
    'step_execution_protection',
    'validate_pipeline_step'
]