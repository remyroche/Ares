"""
Reusable decorators for validation, vectorization, data hygiene, error normalization, and tracing.

- Type/shape/schema validation: integrates with pydantic.validate_call if available,
  and optionally beartype/typeguard. Pandera DataFrame schema checks are supported when installed.
- Vectorization guarantees: auto-vectorize scalar logic or enforce ndarray inputs.
- NaN/Inf/null guards: fast pre-checks for arrays/DataFrames with helpful messages.
- Error normalization: centralize exception mapping into domain-specific errors.
- Logging/tracing/audit: correlation IDs and structured entry/exit logs with PII scrubbing.

ENHANCED FEATURES:
- Integration with enhanced decorator system
- Better error handling and recovery
- Intelligent caching for expensive operations
- Performance monitoring and metrics
- Centralized configuration support
"""


import functools
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any, Iterable, TypeVar, cast, Dict, Optional

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

def _should_enable_caching() -> bool:
    """Check if caching should be enabled based on configuration."""
    return _get_enhanced_config('cache_enabled', False)

def _should_enable_performance_monitoring() -> bool:
    """Check if performance monitoring should be enabled."""
    return _get_enhanced_config('enable_performance_monitoring', False)

def _register_decorator_if_available(name: str, decorator: Callable, **kwargs):
    """Register decorator in enhanced system if available."""
    if ENHANCED_SYSTEM_AVAILABLE and decorator_registry:
        try:
            decorator_registry.register(name=name, decorator=decorator, **kwargs)
        except Exception as e:
            logger.debug(f"Could not register decorator {name}: {e}")

def _create_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Create a cache key for function calls."""
    try:
        # Create a hash of function signature and arguments
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
    return cached_wrapper

def _apply_performance_monitoring(wrapper_func: Callable, level: str = "basic") -> Callable:
    """Apply performance monitoring to a wrapper function."""
    if not _should_enable_performance_monitoring():
        return wrapper_func

    @functools.wraps(wrapper_func)
    return monitored_wrapper

def _log_performance_metrics(metrics: Dict[str, Any], level: str):
    """Log performance metrics based on level."""
    if level == "basic":
        logger.info(f"Performance: {metrics['function']} took {metrics['execution_time']:.3f}s")
    elif level == "detailed":
        logger.info(f"Performance details for {metrics['function']}: {metrics}")
    elif level == "profiling":
        logger.debug(f"Performance profiling for {metrics['function']}: {metrics}")


# --------------------------
# Enhanced Type/schema validation
# --------------------------

@_register_decorator_if_available(
    name="validate_call_or_runtime_types",
    version="2.0",
    description="Enhanced type validation with caching and performance monitoring",
    tags=["validation", "type-checking", "enhanced"]
)
def validate_call_or_runtime_types(*v_args: Any, **v_kwargs: Any) -> Callable[[F], F]:
    """Enhanced decorator factory that prefers pydantic.validate_call if available.

    Falls back to beartype or typeguard if pydantic is unavailable.
    If none are available, acts as a no-op decorator.

    ENHANCED FEATURES:
    - Automatic caching for expensive validation operations
    - Performance monitoring and metrics
    - Integration with enhanced configuration system
    """

    return decorator


@_register_decorator_if_available(
    name="pa_check_input",
    version="2.0",
    description="Enhanced pandera input validation with caching",
    tags=["validation", "pandera", "dataframe", "enhanced"]
)
def pa_check_input(
    schema: Any, *, arg_name: str | None = None, arg_index: int = 0, strict: bool = True
) -> Callable[[F], F]:
    """Enhanced compatibility wrapper for pandera.check_input.

    ENHANCED FEATURES:
    - Intelligent caching for schema validation results
    - Performance monitoring for validation operations
    - Better error handling and recovery
    """

    return decorator


@_register_decorator_if_available(
    name="pa_check_output",
    version="2.0",
    description="Enhanced pandera output validation with caching",
    tags=["validation", "pandera", "dataframe", "enhanced"]
)
def pa_check_output(schema: Any, *, strict: bool = True) -> Callable[[F], F]:
    """Enhanced compatibility wrapper for pandera.check_output.

    ENHANCED FEATURES:
    - Intelligent caching for schema validation results
    - Performance monitoring for validation operations
    - Better error handling and recovery
    """

    return decorator


@_register_decorator_if_available(
    name="pa_check_io",
    version="2.0",
    description="Enhanced pandera I/O validation with intelligent caching",
    tags=["validation", "pandera", "dataframe", "enhanced"]
)
def pa_check_io(
    *,
    input_schema: Any | None = None,
    output_schema: Any | None = None,
    df_arg_name: str | None = None,
    df_arg_index: int = 0,
    strict: bool = True,
) -> Callable[[F], F]:
    """Enhanced validate DataFrame input/output with pandera if available.

    ENHANCED FEATURES:
    - Intelligent caching for validation results
    - Performance monitoring and metrics
    - Better error handling with recovery strategies
    - Integration with enhanced configuration system

    - If pandera is installed and schemas are provided, validate the DataFrame
      argument identified by name or index and the returned DataFrame.
    - If pandera is not installed, performs a lightweight check that the
      argument/return is a pandas DataFrame when schemas are provided.
    """

    return decorator


# --------------------------
# Enhanced Vectorization guarantees
# --------------------------

@_register_decorator_if_available(
    name="enforce_ndarray",
    version="2.0",
    description="Enhanced ndarray enforcement with performance monitoring",
    tags=["vectorization", "numpy", "enhanced"]
)
def enforce_ndarray(
    *,
    arg_index: int = 0,
    forbid_lists: bool = False,
    require_vector: bool = False,
) -> Callable[[F], F]:
    """Enhanced coerce the selected argument to numpy.ndarray and optionally forbid lists.

    ENHANCED FEATURES:
    - Performance monitoring for vectorization operations
    - Intelligent caching for repeated operations
    - Better error handling and recovery

    - forbid_lists=True raises if a list is provided
    - require_vector=True requires at least 1-D input (no pure scalars)
    """

    return decorator


@_register_decorator_if_available(
    name="auto_vectorize",
    version="2.0",
    description="Enhanced auto-vectorization with intelligent caching",
    tags=["vectorization", "numpy", "enhanced"]
)
def auto_vectorize(*, otypes: list[type] | None = None) -> Callable[[F], F]:
    """Enhanced wrap a scalar function so that it transparently handles numpy arrays.

    ENHANCED FEATURES:
    - Intelligent caching for vectorization results
    - Performance monitoring for vectorization operations
    - Better memory management

    - If the first positional argument is an ndarray with ndim>=1, applies
      numpy.vectorize to broadcast the scalar logic across elements.
    - Otherwise, calls the function directly.
    """

    return decorator


# --------------------------
# Enhanced NaN/Inf/null guards
# --------------------------

@_register_decorator_if_available(
    name="guard_array_nan_inf",
    version="2.0",
    description="Enhanced NaN/Inf guards with intelligent caching",
    tags=["data-quality", "validation", "enhanced"]
)
def guard_array_nan_inf(
    *,
    mode: str = "raise",  # "raise" | "warn" | "coerce"
    coerce_value: float = 0.0,
    arg_indices: Iterable[int] = (0,),
) -> Callable[[F], F]:
    """Enhanced pre-check numpy arrays or pandas objects for NaN/Inf before executing.

    ENHANCED FEATURES:
    - Intelligent caching for validation results
    - Performance monitoring for validation operations
    - Better error handling and recovery strategies
    - Integration with enhanced configuration system

    mode:
      - "raise": raise DataValidationError on detection
      - "warn": log a warning and continue
      - "coerce": replace NaN/Inf with coerce_value before calling func
    """

    return decorator


@_register_decorator_if_available(
    name="guard_dataframe_nulls",
    version="2.0",
    description="Enhanced DataFrame null guards with intelligent caching",
    tags=["data-quality", "validation", "dataframe", "enhanced"]
)
def guard_dataframe_nulls(
    *,
    columns: list[str] | None = None,
    mode: str = "raise",  # "raise" | "warn" | "fill"
    fill_value: float | int | str | None = 0,
    arg_index: int = 0,
) -> Callable[[F], F]:
    """Enhanced check a pandas DataFrame argument for nulls/NaN/Inf.

    ENHANCED FEATURES:
    - Intelligent caching for validation results
    - Performance monitoring for validation operations
    - Better error handling and recovery strategies
    - Integration with enhanced configuration system

    arg_index selects which positional argument is the DataFrame (0 for functions where df is first, 1 for instance methods).
    If columns is provided, restrict checks to those columns.
    """

    return decorator


# --------------------------
# Enhanced Error normalization
# --------------------------

_EXCEPTION_MAP: dict[type[BaseException], type[DomainError]] = {
    ValueError: DataValidationError,
    TypeError: SchemaValidationError,
    KeyError: NotFoundError,
    TimeoutError: OperationTimeoutError,
}

# Optional external libraries (best-effort mapping without hard deps)
try:  # requests
    import requests  # type: ignore

    _EXCEPTION_MAP[requests.exceptions.RequestException] = ExternalServiceError  # type: ignore
except Exception:  # pragma: no cover
    pass

try:  # aiohttp
    import aiohttp  # type: ignore

    _EXCEPTION_MAP[aiohttp.ClientError] = ExternalServiceError  # type: ignore
except Exception:  # pragma: no cover
    pass


@_register_decorator_if_available(
    name="normalize_errors",
    version="2.0",
    description="Enhanced error normalization with intelligent recovery",
    tags=["error-handling", "recovery", "enhanced"]
)
def normalize_errors(
    *,
    map_exceptions: dict[type[BaseException], type[DomainError]] | None = None,
    default_error: type[DomainError] = DomainError,
    reraise: bool = False,
) -> Callable[[F], F]:
    """Enhanced normalize heterogeneous exceptions into domain-specific errors.

    ENHANCED FEATURES:
    - Intelligent error recovery strategies
    - Performance monitoring for error handling
    - Better logging and correlation
    - Integration with enhanced configuration system

    - map_exceptions augments the built-in mapping
    - if reraise=True, re-raises the normalized DomainError after logging
    - otherwise returns None and logs; for functions that must return a value,
      consider using together with default returns in your wrapper logic.
    """

    exception_map = dict(_EXCEPTION_MAP)
    if map_exceptions:
        exception_map.update(map_exceptions)

    return decorator


# --------------------------
# Enhanced Logging/tracing/audit
# --------------------------

_SENSITIVE_KEYS = {
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
}


def _sanitize(value: Any) -> Any:
    """Best-effort PII scrubbing for dict-like inputs and sequences.

    Masks values of known sensitive keys. Keeps structure to aid debugging.
    """
    try:
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, val in value.items():
                if str(key).lower() in _SENSITIVE_KEYS:
                    redacted[key] = "***REDACTED***"
                else:
                    redacted[key] = _sanitize(val)
            return redacted
        if isinstance(value, (list, tuple)):
            return type(value)(_sanitize(v) for v in value)
        return value
    except Exception:
        return value


@_register_decorator_if_available(
    name="with_tracing_span",
    version="2.0",
    description="Enhanced tracing with performance monitoring and caching",
    tags=["tracing", "logging", "performance", "enhanced"]
)
def with_tracing_span(
    span_name: str | None = None,
    *,
    log_args: bool = False,
    log_result_len_only: bool = True,
) -> Callable[[F], F]:
    """Enhanced add correlation-aware entry/exit logs around a function call.

    ENHANCED FEATURES:
    - Performance monitoring and metrics collection
    - Intelligent caching for repeated operations
    - Better error handling and recovery
    - Integration with enhanced configuration system

    - Ensures a correlation ID is present
    - Optionally logs sanitized args/kwargs (avoid for heavy data)
    - Logs result size instead of full content by default
    """

    return decorator


# --------------------------
# Export all decorators
# --------------------------

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
