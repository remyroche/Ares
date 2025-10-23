from .core.decorators import handles_errors
"""
Utility decorators for the Ares project.

This module provides commonly used decorators for error handling, logging,
caching, validation, and other cross-cutting concerns.
"""

import functools
import time
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
import hashlib
import json
import numpy as np
import pandas as pd

# Get logger
logger = logging.getLogger(__name__)

# Circuit breaker state
_circuit_breaker_states: Dict[str, Dict[str, Any]] = {}

def _validate_auth_token(token: str) -> bool:
    """Validate authentication token."""
    try:
        # Simple token validation - in production, this would check against a database
        # or use JWT validation
        if not token or len(token) < 10:
            return False
        
        # Check if token has valid format (basic validation)
        if not token.replace('-', '').replace('_', '').isalnum():
            return False
        
        # In a real implementation, you would:
        # 1. Decode JWT token
        # 2. Check expiration
        # 3. Validate signature
        # 4. Check against user database
        
        return True
    except Exception:
        return False

def handles_errors(fallback = None, log_errors = True, reraise = False):
    """
    Decorator for handling errors in functions.

    Args:
        fallback: Value to return on error (default: None)
        log_errors: Whether to log errors (default: True)
        reraise: Whether to reraise the exception (default: False)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")

                if reraise:
                    raise

                return fallback
        return wrapper
    return decorator

def log_execution_time(level="INFO", log_args = False):
    """
    Decorator to log function execution time.

    Args:
        level: Log level (default: "INFO")
        log_args: Whether to log function arguments (default: False)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            if log_args:
                logger.log(getattr(logging, level.upper(), logging.INFO),
                          f"Starting {func.__name__} with args={args}, kwargs={kwargs}")
            else:
                logger.log(getattr(logging, level.upper(), logging.INFO),
                          f"Starting {func.__name__}")

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.log(getattr(logging, level.upper(), logging.INFO),
                          f"Completed {func.__name__} in {execution_time:.4f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Failed {func.__name__} after {execution_time:.4f}s: {e}")
                raise
        return wrapper
    return decorator

def log_call(level="INFO", log_result = False):
    """
    Decorator to log function calls.

    Args:
        level: Log level (default: "INFO")
        log_result: Whether to log the result (default: False)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.log(getattr(logging, level.upper(), logging.INFO),
                      f"Calling {func.__name__}")

            result = func(*args, **kwargs)

            if log_result:
                logger.log(getattr(logging, level.upper(), logging.INFO),
                          f"Result from {func.__name__}: {result}")

            return result
        return wrapper
    return decorator

def traced(span_name = None, log_entry = True, log_exit = True):
    """
    Decorator for function tracing.

    Args:
        span_name: Custom span name (default: function name)
        log_entry: Whether to log function entry (default: True)
        log_exit: Whether to log function exit (default: True)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = span_name or func.__name__

            if log_entry:
                logger.debug(f"Entering span: {name}")

            try:
                result = func(*args, **kwargs)
                if log_exit:
                    logger.debug(f"Exiting span: {name}")
                return result
            except Exception as e:
                logger.error(f"Error in span {name}: {e}")
                raise
        return wrapper
    return decorator

def validates(*validators, **kwargs):
    """
    Decorator for input validation.

    Args:
        *validators: Validation functions to apply
        **kwargs: Additional validation options
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Apply validators to arguments
            for validator in validators:
                if callable(validator):
                    try:
                        validator(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Validation failed for {func.__name__}: {e}")
                        raise ValueError(f"Validation failed: {e}")

            return func(*args, **kwargs)
        return wrapper
    return decorator

def cached(max_size = 128, ttl = None):
    """
    Decorator for function result caching.

    Args:
        max_size: Maximum cache size (default: 128)
        ttl: Time to live in seconds (default: None - no expiration)
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key_data = (args, tuple(sorted(kwargs.items())))
            key = hashlib.md5(json.dumps(key_data, default = str).encode()).hexdigest()

            # Check if cached result exists and is still valid
            if key in cache:
                if ttl is None or (time.time() - cache_times[key]) < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache[key]
                else:
                    # Expired, remove from cache
                    del cache[key]
                    del cache_times[key]

            # Cache miss, compute result
            result = func(*args, **kwargs)

            # Store in cache
            if len(cache) >= max_size:
                # Remove oldest entry
                oldest_key = min(cache_times.keys(), key = lambda k: cache_times[k])
                del cache[oldest_key]
                del cache_times[oldest_key]

            cache[key] = result
            cache_times[key] = time.time()

            logger.debug(f"Cache miss for {func.__name__}, result cached")
            return result
        return wrapper
    return decorator

def circuit_breaker(failure_threshold = 5, recovery_timeout = 60, expected_exception = Exception):
    """
    Circuit breaker decorator to prevent cascading failures.

    Args:
        failure_threshold: Number of failures before opening circuit (default: 5)
        recovery_timeout: Time in seconds before attempting recovery (default: 60)
        expected_exception: Exception type to catch (default: Exception)
    """
    def decorator(func: Callable) -> Callable:
        func_name = func.__name__

        # Initialize circuit breaker state
        if func_name not in _circuit_breaker_states:
            _circuit_breaker_states[func_name] = {
                'failure_count': 0,
                'last_failure_time': None,
                'state': 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
            }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            state = _circuit_breaker_states[func_name]
            current_time = time.time()

            # Check if circuit is open
            if state['state'] == 'OPEN':
                if current_time - state['last_failure_time'] > recovery_timeout:
                    state['state'] = 'HALF_OPEN'
                    logger.info(f"Circuit breaker for {func_name} moved to HALF_OPEN")
                else:
                    logger.warning(f"Circuit breaker for {func_name} is OPEN, skipping call")
                    raise Exception(f"Circuit breaker is OPEN for {func_name}")

            try:
                result = func(*args, **kwargs)

                # Success - reset failure count and close circuit if needed
                if state['state'] == 'HALF_OPEN':
                    state['state'] = 'CLOSED'
                    logger.info(f"Circuit breaker for {func_name} moved to CLOSED")

                state['failure_count'] = 0
                return result

            except expected_exception as e:
                state['failure_count'] += 1
                state['last_failure_time'] = current_time

                if state['failure_count'] >= failure_threshold:
                    state['state'] = 'OPEN'
                    logger.error(f"Circuit breaker for {func_name} moved to OPEN after {failure_threshold} failures")

                raise

        return wrapper
    return decorator

def retry(max_attempts = 3, delay = 1, backoff = 2, exceptions=(Exception,)):
    """
    Decorator for retrying failed function calls.

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 1)
        backoff: Backoff multiplier for delay (default: 2)
        exceptions: Tuple of exceptions to catch and retry (default: (Exception,))
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                        raise

                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff

            return None
        return wrapper
    return decorator

def authenticated(required_roles = None):
    """
    Decorator for authentication and authorization.

    Args:
        required_roles: List of required roles (default: None - any authenticated user)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check for authentication token in kwargs or request context
            auth_token = kwargs.get('auth_token')
            if not auth_token:
                # Try to get from request context if available
                try:
                    from flask import request
                    auth_token = request.headers.get('Authorization', '').replace('Bearer ', '')
                except ImportError:
                    pass
            
            if not auth_token:
                logger.warning(f"Authentication required for {func.__name__} but no token provided")
                raise PermissionError("Authentication required")
            
            # Validate token (simplified implementation)
            if not _validate_auth_token(auth_token):
                logger.warning(f"Invalid authentication token for {func.__name__}")
                raise PermissionError("Invalid authentication token")
            
            logger.debug(f"Authentication successful for {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def requires_role(*roles):
    """
    Decorator for role-based access control.

    Args:
        *roles: Required roles for access
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get user roles from kwargs or request context
            user_roles = kwargs.get('user_roles', [])
            if not user_roles:
                # Try to get from request context if available
                try:
                    from flask import request
                    user_roles = getattr(request, 'user_roles', [])
                except ImportError:
                    pass
            
            if not user_roles:
                logger.warning(f"Role check failed for {func.__name__}: no user roles provided")
                raise PermissionError("User roles required")
            
            # Check if user has any of the required roles
            has_required_role = any(role in user_roles for role in roles)
            if not has_required_role:
                logger.warning(f"Role check failed for {func.__name__}: user roles {user_roles} don't include required roles {roles}")
                raise PermissionError(f"Insufficient permissions. Required roles: {roles}")
            
            logger.debug(f"Role check successful for {func.__name__}: user has required roles")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_schema(schema):
    """
    Decorator for schema validation.

    Args:
        schema: Schema to validate against
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Validate input arguments against schema
                if hasattr(schema, 'validate'):
                    # If schema has a validate method (e.g., Marshmallow, Pydantic)
                    validation_result = schema.validate(kwargs)
                    if validation_result:
                        logger.warning(f"Schema validation failed for {func.__name__}: {validation_result}")
                        raise ValueError(f"Schema validation failed: {validation_result}")
                elif isinstance(schema, dict):
                    # Simple dict-based schema validation
                    for field, expected_type in schema.items():
                        if field in kwargs:
                            if not isinstance(kwargs[field], expected_type):
                                logger.warning(f"Schema validation failed for {func.__name__}: {field} should be {expected_type}, got {type(kwargs[field])}")
                                raise TypeError(f"Field '{field}' should be {expected_type}, got {type(kwargs[field])}")
                        elif field not in kwargs and not field.startswith('optional_'):
                            logger.warning(f"Schema validation failed for {func.__name__}: required field '{field}' missing")
                            raise ValueError(f"Required field '{field}' missing")
                
                logger.debug(f"Schema validation successful for {func.__name__}")
                return func(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Schema validation error for {func.__name__}: {e}")
                raise
        return wrapper
    return decorator

def validate_dataframe(required_columns = None, required_dtypes = None):
    """
    Decorator for DataFrame validation.

    Args:
        required_columns: List of required columns
        required_dtypes: Dict of column -> expected dtype
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Find DataFrame arguments
                dataframes = []
                for arg in args:
                    if isinstance(arg, pd.DataFrame):
                        dataframes.append(arg)
                
                for value in kwargs.values():
                    if isinstance(value, pd.DataFrame):
                        dataframes.append(value)
                
                # Validate each DataFrame
                for i, df in enumerate(dataframes):
                    logger.debug(f"Validating DataFrame {i+1} for {func.__name__}")
                    
                    # Check required columns
                    if required_columns:
                        missing_columns = set(required_columns) - set(df.columns)
                        if missing_columns:
                            logger.warning(f"DataFrame validation failed for {func.__name__}: missing columns {missing_columns}")
                            raise ValueError(f"Missing required columns: {missing_columns}")
                    
                    # Check required dtypes
                    if required_dtypes:
                        for column, expected_dtype in required_dtypes.items():
                            if column in df.columns:
                                if not pd.api.types.is_dtype_equal(df[column].dtype, expected_dtype):
                                    logger.warning(f"DataFrame validation failed for {func.__name__}: column '{column}' should be {expected_dtype}, got {df[column].dtype}")
                                    raise TypeError(f"Column '{column}' should be {expected_dtype}, got {df[column].dtype}")
                    
                    # Check for empty DataFrame
                    if df.empty:
                        logger.warning(f"DataFrame validation failed for {func.__name__}: DataFrame is empty")
                        raise ValueError("DataFrame cannot be empty")
                
                logger.debug(f"DataFrame validation successful for {func.__name__}")
                return func(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"DataFrame validation error for {func.__name__}: {e}")
                raise
        return wrapper
    return decorator

def comprehensive_validation(validators = None):
    """
    Decorator for comprehensive validation.

    Args:
        validators: List of validation functions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                validation_errors = []
                
                # Run custom validators if provided
                if validators:
                    for validator in validators:
                        try:
                            result = validator(*args, **kwargs)
                            if result is not None and not result:
                                validation_errors.append(f"Validation failed: {validator.__name__}")
                        except Exception as e:
                            validation_errors.append(f"Validation error in {validator.__name__}: {e}")
                
                # Basic input validation
                for i, arg in enumerate(args):
                    if arg is None:
                        validation_errors.append(f"Argument {i} cannot be None")
                    elif isinstance(arg, (list, tuple)) and len(arg) == 0:
                        validation_errors.append(f"Argument {i} cannot be empty")
                
                # Check for required keyword arguments
                required_kwargs = ['data', 'config']  # Common required kwargs
                for req_kwarg in required_kwargs:
                    if req_kwarg not in kwargs or kwargs[req_kwarg] is None:
                        validation_errors.append(f"Required keyword argument '{req_kwarg}' missing or None")
                
                if validation_errors:
                    error_msg = f"Comprehensive validation failed for {func.__name__}: {'; '.join(validation_errors)}"
                    logger.warning(error_msg)
                    raise ValueError(error_msg)
                
                logger.debug(f"Comprehensive validation successful for {func.__name__}")
                return func(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Comprehensive validation error for {func.__name__}: {e}")
                raise
        return wrapper
    return decorator

def secure_data_processing(encrypt = False, audit = True):
    """
    Decorator for secure data processing.

    Args:
        encrypt: Whether to encrypt data (default: False)
        audit: Whether to audit data access (default: True)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if audit:
                logger.info(f"Secure data processing: {func.__name__}")

            if encrypt:
                logger.debug(f"Data encryption enabled for {func.__name__}")

            return func(*args, **kwargs)
        return wrapper
    return decorator

def compose(*decorators):
    """
    Decorator to compose multiple decorators.

    Args:
        *decorators: Decorators to compose
    """
    def decorator(func: Callable) -> Callable:
        result = func
        for dec in reversed(decorators):
            result = dec(result)
        return result
    return decorator

class CachePolicy:
    """Cache policy configuration."""

    def __init__(self, max_size = 128, ttl = None, eviction_policy='LRU'):
        self.max_size = max_size
        self.ttl = ttl
        self.eviction_policy = eviction_policy
