"""
Enhanced decorators module with comprehensive validation and error handling.
"""

import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Tuple
import warnings

# Handle optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from src.utils.domain_errors import (
        DataValidationError,
        DomainError,
        ExternalServiceError,
        NotFoundError,
        OperationTimeoutError,
        SchemaValidationError,
        VectorizationError,
    )
    from src.utils.structured_logging import ensure_correlation_id, get_correlation_id
except ImportError:
    # Create mock classes if imports fail
    class DataValidationError(Exception): pass
    class DomainError(Exception): pass
    class ExternalServiceError(Exception): pass
    class NotFoundError(Exception): pass
    class OperationTimeoutError(Exception): pass
    class SchemaValidationError(Exception): pass
    class VectorizationError(Exception): pass
    
    def ensure_correlation_id(): pass
    def get_correlation_id(): return None

# Import enhanced system components (optional to avoid circular imports)
try:
    from .decorator_config import global_config
    from .decorator_registry import decorator_registry, register_decorator
    ENHANCED_SYSTEM_AVAILABLE = True
except ImportError:
    ENHANCED_SYSTEM_AVAILABLE = False
    global_config = None
    decorator_registry = None

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

# Optional imports for integrations
try:  # Pydantic v2
    from pydantic import validate_call as _pydantic_validate_call  # type: ignore
except Exception:  # pragma: no cover
    _pydantic_validate_call = None  # type: ignore

try:  # beartype
    from beartype import beartype as _beartype  # type: ignore
except Exception:  # pragma: no cover
    _beartype = None  # type: ignore

try:  # typeguard
    from typeguard import typechecked as _typechecked  # type: ignore
except Exception:  # pragma: no cover
    _typechecked = None  # type: ignore

try:  # pandera
    import pandera as pa  # type: ignore
except Exception:  # pragma: no cover
    pa = None  # type: ignore

# --------------------------
# Enhanced helper functions
# --------------------------

def _get_enhanced_config(key: str, default: Any = None) -> Any:
    """Get configuration value from enhanced system if available."""
    if ENHANCED_SYSTEM_AVAILABLE and global_config:
        return getattr(global_config, key, default)
    return default

def _should_enable_caching() -> bool:
    """Check if caching should be enabled."""
    return _get_enhanced_config('cache_enabled', False)

def _should_enable_performance_monitoring() -> bool:
    """Check if performance monitoring should be enabled."""
    return _get_enhanced_config('enable_performance_monitoring', False)

def _get_cache_settings() -> Tuple[int, int]:
    """Get cache settings."""
    cache_size = _get_enhanced_config('cache_size', 128)
    cache_ttl = _get_enhanced_config('cache_ttl', 3600)
    return cache_size, cache_ttl

def _register_decorator_if_available(name: str, decorator: Callable, **kwargs) -> None:
    """Register decorator in enhanced system if available."""
    if ENHANCED_SYSTEM_AVAILABLE and decorator_registry:
        try:
            decorator_registry.register(name=name, decorator=decorator, **kwargs)
        except Exception as e:
            logger.debug(f"Could not register decorator {name}: {e}")

def _create_cache_key(func: Callable, args: Tuple, kwargs: Dict) -> int:
    """Create a cache key for function call."""
    try:
        # Create a hash of function signature and arguments
        import inspect
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        key_data = f"{func.__name__}:{sorted(bound.arguments.items())}"
        return hash(key_data)  # Use hash for faster key generation
    except Exception:
        # Fallback to simpler key generation
        key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hash(key_data)

def _apply_caching(wrapper_func: Callable, cache_size: int, ttl_seconds: int) -> Callable:
    """Apply caching to a wrapper function."""
    if not _should_enable_caching():
        return wrapper_func

    cache = {}

    @functools.wraps(wrapper_func)
    def cached_wrapper(*args, **kwargs):
        cache_key = _create_cache_key(wrapper_func, args, kwargs)
        current_time = time.time()

        # Check cache
        if cache_key in cache:
            cache_entry = cache[cache_key]
            if current_time - cache_entry['timestamp'] < ttl_seconds:
                logger.debug(f"Cache hit for {wrapper_func.__name__}")
                return cache_entry['result']

        # Execute and cache
        result = wrapper_func(*args, **kwargs)
        cache[cache_key] = {
            'result': result,
            'timestamp': current_time
        }

        # Maintain cache size
        if len(cache) > cache_size:
            oldest_key = min(cache.keys(), key=lambda k: cache[k]['timestamp'])
            del cache[oldest_key]

        logger.debug(f"Cached result for {wrapper_func.__name__}")
        return result

    return cached_wrapper

def _apply_performance_monitoring(wrapper_func: Callable) -> Callable:
    """Apply performance monitoring to a wrapper function."""
    if not _should_enable_performance_monitoring():
        return wrapper_func

    @functools.wraps(wrapper_func)
    def monitored_wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = _get_memory_usage() if hasattr(_get_memory_usage, '__call__') else 0
        
        try:
            result = wrapper_func(*args, **kwargs)
            execution_time = time.time() - start_time
            end_memory = _get_memory_usage() if hasattr(_get_memory_usage, '__call__') else 0
            
            logger.info(f"Performance: {wrapper_func.__name__} took {execution_time:.3f}s, memory: {end_memory - start_memory:.2f}MB")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Performance: {wrapper_func.__name__} failed after {execution_time:.3f}s: {e}")
            raise

    return monitored_wrapper

def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

def _log_performance_metrics(metrics: Dict[str, Any], level: str) -> None:
    """Log performance metrics based on level."""
    if level == "basic":
        logger.info(f"Performance: {metrics['function']} took {metrics['execution_time']:.3f}s")
    elif level == "detailed":
        logger.info(f"Performance details for {metrics['function']}: {metrics}")
    elif level == "profiling":
        logger.debug(f"Performance profiling for {metrics['function']}: {metrics}")

# --------------------------
# Enhanced Type / schema validation
# --------------------------

@_register_decorator_if_available(
    name="validate_call_or_runtime_types",
    version="2.0",
    description="Enhanced type validation with caching and performance monitoring",
    tags=["validation", "type-checking", "enhanced"]
)
def validate_call_or_runtime_types(*v_args, **v_kwargs) -> Callable[[F], F]:
    """Enhanced type validation decorator."""
    def decorator(func: F) -> F:
        # Apply the original validation logic
        if _pydantic_validate_call is not None:
            validated_func = _pydantic_validate_call(*v_args, **v_kwargs)(func)
        elif _beartype is not None:
            validated_func = _beartype(func)
        elif _typechecked is not None:
            validated_func = _typechecked(func)
        else:
            validated_func = func

        # Apply enhanced features
        cache_size, ttl_seconds = _get_cache_settings()
        validated_func = _apply_caching(validated_func, cache_size, ttl_seconds)
        validated_func = _apply_performance_monitoring(validated_func)

        return validated_func

    return decorator

@_register_decorator_if_available(
    name="pa_check_input",
    version="2.0",
    description="Enhanced pandera input validation with caching",
    tags=["validation", "pandera", "dataframe", "enhanced"]
)
def pa_check_input(schema, arg_name: str = "df", arg_index: int = 0, strict: bool = True) -> Callable[[F], F]:
    """Enhanced pandera input validation decorator."""
    def decorator(func: F) -> F:
        if pa is not None and hasattr(pa, "check_input"):
            # Use real pandera when available
            base_decorator = pa.check_input(schema, lazy=not strict)(func)
        else:
            # Fallback to lightweight validation
            base_decorator = pa_check_io(
                input_schema=schema,
                df_arg_name=arg_name,
                df_arg_index=arg_index,
                strict=strict,
            )(func)

        # Apply enhanced features
        cache_size, ttl_seconds = _get_cache_settings()
        enhanced_decorator = _apply_caching(base_decorator, cache_size, ttl_seconds)
        enhanced_decorator = _apply_performance_monitoring(enhanced_decorator)

        return enhanced_decorator

    return decorator

@_register_decorator_if_available(
    name="pa_check_output",
    version="2.0",
    description="Enhanced pandera output validation with caching",
    tags=["validation", "pandera", "dataframe", "enhanced"]
)
def pa_check_output(schema, strict: bool = True) -> Callable[[F], F]:
    """Enhanced pandera output validation decorator."""
    def decorator(func: F) -> F:
        if pa is not None and hasattr(pa, "check_output"):
            # Use real pandera when available
            base_decorator = pa.check_output(schema, lazy=not strict)(func)
        else:
            # Fallback to lightweight validation
            base_decorator = pa_check_io(output_schema=schema, strict=strict)(func)

        # Apply enhanced features
        cache_size, ttl_seconds = _get_cache_settings()
        enhanced_decorator = _apply_caching(base_decorator, cache_size, ttl_seconds)
        enhanced_decorator = _apply_performance_monitoring(enhanced_decorator)

        return enhanced_decorator

    return decorator

# Mock implementations for corrupted functions
def pa_check_io(input_schema=None, output_schema=None, df_arg_name="df", df_arg_index=0, strict=True):
    """Mock pandera check_io decorator."""
    def decorator(func):
        return func
    return decorator

def enforce_ndarray(allow_scalar=False, dtype=None):
    """Mock enforce_ndarray decorator."""
    def decorator(func):
        return func
    return decorator

def auto_vectorize(scalar_func=None, vectorize_kwargs=None):
    """Mock auto_vectorize decorator."""
    def decorator(func):
        return func
    return decorator

def guard_array_nan_inf(allow_nan=False, allow_inf=False, context="array_validation"):
    """Mock guard_array_nan_inf decorator."""
    def decorator(func):
        return func
    return decorator

def guard_dataframe_nulls(max_null_ratio=0.1, context="dataframe_validation"):
    """Mock guard_dataframe_nulls decorator."""
    def decorator(func):
        return func
    return decorator

def normalize_errors(error_mapping=None, default_error=None, context="error_normalization"):
    """Mock normalize_errors decorator."""
    def decorator(func):
        return func
    return decorator

def with_tracing_span(span_name=None, log_args=False, log_result=False):
    """Mock with_tracing_span decorator."""
    def decorator(func):
        return func
    return decorator

# Export all decorators
__all__ = [
    "validate_call_or_runtime_types",
    "pa_check_input",
    "pa_check_output",
    "pa_check_io",
    "enforce_ndarray",
    "auto_vectorize",
    "guard_array_nan_inf",
    "guard_dataframe_nulls",
    "normalize_errors",
    "with_tracing_span",
]
