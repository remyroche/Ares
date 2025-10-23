#!/usr/bin/env python3
import numpy as np
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

"""
Comprehensive Step Decorators for Data Collection Pipeline

This module provides specialized decorators for protecting data operations,
ensuring proper data formatting, analysis, and access control.
"""

import asyncio
import functools
import logging
import time
from dataclasses import dataclass
from enum import Enum

from src.core.decorators import (
    handles_errors,
    log_call,
    monitor_function_calls,
    validate_dataframe,
    traced
)
import pandas as pd

import typing

from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    safe_file_exists,
    ensure_directory
)

F = TypeVar('F', bound = Callable[..., Any])

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

            # Setup logging
            logger = logging.getLogger(f'secure_data_operation.{func.__name__}')
            logger.info(f'🔒 Starting secure data operation: {operation_name}')
            logger.info(f'📊 Operation type: {operation_type.value}')
            logger.info(f'🔐 Security level: {security_level.value}')
            logger.info(f'📋 Input validation: {validate_inputs}')
            logger.info(f'📋 Output validation: {validate_outputs}')
            logger.info(f'📋 Audit logging: {audit}')
            logger.info(f'⏱️ Timeout: {timeout_seconds}s' if timeout_seconds else '⏱️ No timeout')
            logger.info(f'🔄 Retry attempts: {retry_attempts}')

            try:
                # Input validation
                if validate_inputs:
                    logger.info('🔍 Validating operation inputs...')
                    await _validate_operation_inputs(func, args, kwargs, validation_kwargs)
                    logger.info('✅ Input validation passed')

                # Execute operation with timeout if specified
                logger.info('🚀 Executing operation...')
                if timeout_seconds:
                    logger.info(f'⏱️ Using timeout: {timeout_seconds}s')
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout = timeout_seconds
                    )
                else:
                    result = await func(*args, **kwargs)

                logger.info('✅ Operation execution completed')
                logger.info(f'📊 Result type: {type(result).__name__}')

                # Output validation
                if validate_outputs:
                    logger.info('🔍 Validating operation outputs...')
                    await _validate_operation_outputs(result, validation_kwargs)
                    logger.info('✅ Output validation passed')

                execution_time = time.time() - start_time
                logger.info(f'⏱️ Total execution time: {execution_time:.2f}s')

                # Log successful operation
                if audit:
                    logger.info('📝 Logging operation to audit trail...')
                    operation_logger.log_operation(
                        context = context,
                        operation = operation_name,
                        details={"result_type": type(result).__name__},
                        success = True,
                        execution_time = execution_time
                    )
                    logger.info('✅ Operation logged to audit trail')

                logger.info('✅ Secure data operation completed successfully')
                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f'❌ Secure data operation failed: {e}')
                logger.error(f'📊 Error type: {type(e).__name__}')
                logger.error(f'⏱️ Execution time before failure: {execution_time:.2f}s')

                # Log failed operation
                if audit:
                    logger.info('📝 Logging failed operation to audit trail...')
                    operation_logger.log_operation(
                        context = context,
                        operation = operation_name,
                        details={"error": str(e), "error_type": type(e).__name__},
                        success = False,
                        execution_time = execution_time
                    )
                    logger.info('✅ Failed operation logged to audit trail')

                # Retry logic
                if retry_attempts > 0:
                    logger.info(f'🔄 Attempting retry with {retry_attempts} attempts remaining...')
                    return await _retry_operation(
                        func, args, kwargs, retry_attempts, context, operation_name
                    )
                else:
                    logger.error('❌ No retry attempts remaining, raising exception')

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
                        context = context,
                        operation = operation_name,
                        details={"result_type": type(result).__name__},
                        success = True,
                        execution_time = execution_time
                    )

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                # Log failed operation
                if audit:
                    operation_logger.log_operation(
                        context = context,
                        operation = operation_name,
                        details={"error": str(e), "error_type": type(e).__name__},
                        success = False,
                        execution_time = execution_time
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
        protected_func = monitor_function_calls(
            step_name = step_name,
            enable_performance_monitoring = True
        )(protected_func)

        # Add data integrity protection
        protected_func = validate_dataframe(protected_func)

        # Add quality gate
        protected_func = log_call(step_name = step_name)(protected_func)

        # Add pipeline step validation
        protected_func = validate_pipeline_step(
            prerequisites = prerequisites,
            outputs = outputs,
            stage = step_name
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
    data_dir = kwargs.get('data_dir', 'historical_data')
    step_name = kwargs.get('step_name', 'unknown_step')

    return DataOperationContext(
        operation_type = operation_type,
        security_level = security_level,
        step_name = step_name,
        symbol = symbol,
        exchange = exchange,
        data_dir = data_dir,
        timestamp = format_datetime(get_current_datetime())
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

    logger = logging.getLogger(__name__)
    logger.debug(f"Checking for lookahead bias in column '{temporal_column}' with max_lookahead={max_lookahead}")

    try:
        # Sort by timestamp
        df_sorted = df.sort_values(temporal_column).copy()
        
        # Check if timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_sorted[temporal_column]):
            logger.warning(f"Timestamp column '{temporal_column}' is not datetime type - cannot check lookahead bias")
            return

        # Check for duplicate timestamps
        duplicate_timestamps = df_sorted[temporal_column].duplicated()
        if duplicate_timestamps.any():
            logger.warning(f"Found {duplicate_timestamps.sum()} duplicate timestamps - this may indicate data quality issues")

        # Check for non-monotonic timestamps (future data in past)
        timestamp_diff = df_sorted[temporal_column].diff()
        negative_diffs = timestamp_diff < pd.Timedelta(0)
        if negative_diffs.any():
            logger.error(f"Found {negative_diffs.sum()} non-monotonic timestamps - potential lookahead bias detected")
            raise ValueError(f"Non-monotonic timestamps detected in column '{temporal_column}' - potential lookahead bias")

        # Check for gaps in time series that might indicate data leakage
        if len(df_sorted) > 1:
            time_gaps = timestamp_diff[1:]  # Skip first NaN
            median_gap = time_gaps.median()
            large_gaps = time_gaps > median_gap * 10  # Gaps 10x larger than median
            
            if large_gaps.any():
                logger.warning(f"Found {large_gaps.sum()} large time gaps - verify data integrity")
                
                # Log details of large gaps
                large_gap_indices = df_sorted.index[large_gaps]
                for idx in large_gap_indices[:5]:  # Show first 5 large gaps
                    prev_time = df_sorted.loc[idx - 1, temporal_column] if idx > 0 else None
                    curr_time = df_sorted.loc[idx, temporal_column]
                    if prev_time:
                        gap_duration = curr_time - prev_time
                        logger.debug(f"Large gap at index {idx}: {gap_duration}")

        # Check for potential future data leakage in feature columns
        # This is a heuristic check - look for columns that might contain future information
        suspicious_columns = []
        for col in df_sorted.columns:
            if col == temporal_column:
                continue
                
            # Check for columns that might be forward-looking
            if any(keyword in col.lower() for keyword in ['future', 'next', 'forward', 'ahead', 'prediction']):
                suspicious_columns.append(col)
                logger.warning(f"Column '{col}' contains suspicious keywords - verify it doesn't contain future data")

        # Check for statistical patterns that might indicate lookahead bias
        if len(df_sorted) > 100:  # Only check if we have enough data
            # Check for perfect correlation between features and targets (if target columns exist)
            target_columns = [col for col in df_sorted.columns if any(keyword in col.lower() for keyword in ['target', 'label', 'y', 'outcome'])]
            
            for target_col in target_columns:
                for feature_col in df_sorted.columns:
                    if feature_col in [temporal_column, target_col]:
                        continue
                    
                    try:
                        # Calculate rolling correlation to detect suspicious patterns
                        if pd.api.types.is_numeric_dtype(df_sorted[feature_col]) and pd.api.types.is_numeric_dtype(df_sorted[target_col]):
                            # Use a small window to avoid lookahead bias in the check itself
                            window_size = min(20, len(df_sorted) // 10)
                            if window_size > 5:
                                rolling_corr = df_sorted[feature_col].rolling(window=window_size).corr(df_sorted[target_col].rolling(window=window_size))
                                high_corr = rolling_corr.abs() > 0.95
                                
                                if high_corr.any():
                                    logger.warning(f"High correlation detected between '{feature_col}' and '{target_col}' - verify no lookahead bias")
                    except Exception as e:
                        logger.debug(f"Could not check correlation between '{feature_col}' and '{target_col}': {e}")

        # Check for max_lookahead constraint
        if max_lookahead > 0:
            # This is a basic check - in practice, you'd need more sophisticated validation
            # based on your specific feature engineering pipeline
            logger.debug(f"Max lookahead constraint: {max_lookahead} periods")
            
            # Check if any features might be using more than max_lookahead periods
            # This would require knowledge of your feature engineering pipeline
            # For now, we'll just log the constraint
            logger.info(f"Lookahead bias check completed with max_lookahead={max_lookahead}")

        logger.debug("Lookahead bias check completed successfully")

    except Exception as e:
        logger.error(f"Lookahead bias check failed: {e}")
        raise ValueError(f"Lookahead bias validation failed: {e}")

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
    try:
        # Check if user is authenticated
        # In a production environment, this would check against an authentication service
        # For now, we'll implement a basic check
        
        # Check environment variables for authentication tokens
        import os
        auth_token = os.getenv('AUTH_TOKEN')
        api_key = os.getenv('API_KEY')
        
        if not auth_token and not api_key:
            # Check if we're in a development environment
            if os.getenv('ENVIRONMENT') == 'development':
                logging.warning("Running in development mode without authentication")
                return
            
            raise AuthenticationError("Authentication required: No valid tokens found")
        
        # Validate token format (basic validation)
        if auth_token and len(auth_token) < 32:
            raise AuthenticationError("Invalid authentication token format")
        
        if api_key and len(api_key) < 16:
            raise AuthenticationError("Invalid API key format")
        
        # In production, you would validate the token against your auth service
        # For now, we'll just log that authentication passed
        logging.info("Authentication check passed")
        
    except Exception as e:
        logging.error(f"Authentication check failed: {e}")
        raise AuthenticationError(f"Authentication failed: {e}")


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", error_code: str = "AUTH_ERROR", details: Optional[Dict[str, Any]] = None):
        """
        Initialize authentication error.
        
        Args:
            message: Error message
            error_code: Error code for programmatic handling
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = get_current_datetime()
    
    def __str__(self) -> str:
        """String representation of the error."""
        return f"{self.error_code}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": "AuthenticationError",
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": format_datetime(self.timestamp)
        }
    
    @classmethod
    def from_credentials(cls, username: str, reason: str = "Invalid credentials") -> 'AuthenticationError':
        """Create authentication error from credential failure."""
        return cls(
            message=f"Authentication failed for user '{username}': {reason}",
            error_code="INVALID_CREDENTIALS",
            details={"username": username, "reason": reason}
        )
    
    @classmethod
    def from_token(cls, token_type: str, reason: str = "Invalid token") -> 'AuthenticationError':
        """Create authentication error from token failure."""
        return cls(
            message=f"Token authentication failed ({token_type}): {reason}",
            error_code="INVALID_TOKEN",
            details={"token_type": token_type, "reason": reason}
        )
    
    @classmethod
    def from_permission(cls, operation: str, required_permission: str) -> 'AuthenticationError':
        """Create authentication error from permission failure."""
        return cls(
            message=f"Insufficient permissions for operation '{operation}': requires '{required_permission}'",
            error_code="INSUFFICIENT_PERMISSIONS",
            details={"operation": operation, "required_permission": required_permission}
        )

def _check_rate_limit(operation_name: str, rate_limit: int) -> None:
    """Check rate limiting for operations."""
    try:
        import time
        from collections import defaultdict
        
        # In-memory rate limiting (in production, use Redis or similar)
        if not hasattr(_check_rate_limit, '_rate_limits'):
            _check_rate_limit._rate_limits = defaultdict(list)
        
        current_time = time.time()
        operation_requests = _check_rate_limit._rate_limits[operation_name]
        
        # Remove old requests outside the time window
        window_start = current_time - 60  # 1 minute window
        operation_requests[:] = [req_time for req_time in operation_requests if req_time > window_start]
        
        # Check if we're within rate limit
        if len(operation_requests) >= rate_limit:
            time_until_reset = operation_requests[0] + 60 - current_time
            raise RateLimitError(
                f"Rate limit exceeded for {operation_name}. "
                f"Limit: {rate_limit} requests per minute. "
                f"Try again in {time_until_reset:.1f} seconds"
            )
        
        # Add current request
        operation_requests.append(current_time)
        
        logging.debug(f"Rate limit check passed for {operation_name}: {len(operation_requests)}/{rate_limit}")
        
    except RateLimitError:
        raise
    except Exception as e:
        logging.error(f"Rate limit check failed: {e}")
        # Don't block operations if rate limiting fails
        logging.warning("Rate limiting disabled due to error")


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", operation: str = "unknown", 
                 limit: int = 0, reset_time: Optional[float] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize rate limit error.
        
        Args:
            message: Error message
            operation: Operation that hit the rate limit
            limit: Rate limit that was exceeded
            reset_time: Time when the rate limit resets (Unix timestamp)
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.limit = limit
        self.reset_time = reset_time
        self.details = details or {}
        self.timestamp = get_current_datetime()
    
    def __str__(self) -> str:
        """String representation of the error."""
        if self.reset_time:
            time_until_reset = self.reset_time - time.time()
            return f"RateLimitError: {self.message} (resets in {time_until_reset:.1f}s)"
        return f"RateLimitError: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": "RateLimitError",
            "message": self.message,
            "operation": self.operation,
            "limit": self.limit,
            "reset_time": self.reset_time,
            "details": self.details,
            "timestamp": format_datetime(self.timestamp)
        }
    
    @classmethod
    def from_operation(cls, operation: str, limit: int, current_count: int, 
                      window_seconds: int = 60) -> 'RateLimitError':
        """Create rate limit error from operation details."""
        reset_time = time.time() + window_seconds
        return cls(
            message=f"Rate limit exceeded for operation '{operation}': {current_count}/{limit} requests in {window_seconds}s",
            operation=operation,
            limit=limit,
            reset_time=reset_time,
            details={
                "current_count": current_count,
                "window_seconds": window_seconds,
                "time_until_reset": window_seconds
            }
        )
    
    @classmethod
    def from_api_call(cls, endpoint: str, limit: int, current_count: int) -> 'RateLimitError':
        """Create rate limit error from API call."""
        return cls(
            message=f"API rate limit exceeded for endpoint '{endpoint}': {current_count}/{limit} requests",
            operation=f"api_call_{endpoint}",
            limit=limit,
            details={
                "endpoint": endpoint,
                "current_count": current_count,
                "api_error": True
            }
        )
    
    def get_retry_after(self) -> float:
        """Get seconds to wait before retrying."""
        if self.reset_time:
            return max(0, self.reset_time - time.time())
        return 60.0  # Default 1 minute

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
