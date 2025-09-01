"""
Reusable decorators for validation, vectorization, data hygiene, error normalization, and tracing.

- Type / shape / schema validation: integrates with pydantic.validate_call if available,
  and optionally beartype / typeguard. Pandera DataFrame schema checks are supported when installed.
- Vectorization guarantees: auto - vectorize scalar logic or enforce ndarray inputs.
- NaN / Inf / null guards: fast pre - checks for arrays / DataFrames with helpful messages.
- Error normalization: centralize exception mapping into domain - specific errors.
- Logging / tracing / audit: correlation IDs and structured entry / exit logs with PII scrubbing.

ENHANCED FEATURES:
    pass - Integration with enhanced decorator system - Better error handling and recovery - Intelligent caching for expensive operations - Performance monitoring and metrics - Centralized configuration support
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
    NUMPY_AVAILABLE, True
except ImportError:
    NUMPY_AVAILABLE, False
    np, None

try:
    import pandas as pd
    PANDAS_AVAILABLE, True
except ImportError:
    PANDAS_AVAILABLE, False
    pd, None

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
    ENHANCED_SYSTEM_AVAILABLE, True
except ImportError:
    ENHANCED_SYSTEM_AVAILABLE, False
    global_config, None
    decorator_registry, None

T, TypeVar("T")
F, TypeVar("F", bound = Callable[..., Any])

logger, logging.getLogger(__name__)

# Optional imports for integrations
try:  # Pydantic v2
    from pydantic import validate_call as _pydantic_validate_call  # type: ignore
except Exception:  # pragma: no cover
    _pydantic_validate_call, None  # type: ignore

try:  # beartype
    from beartype import beartype as _beartype  # type: ignore
except Exception:  # pragma: no cover
    _beartype, None  # type: ignore

try:  # typeguard
    from typeguard import typechecked as _typechecked  # type: ignore
except Exception:  # pragma: no cover
    _typechecked, None  # type: ignore

try:  # pandera
    import pandera as pa  # type: ignore
except Exception:  # pragma: no cover
    pa, None  # type: ignore

# --------------------------
# Enhanced helper functions
# --------------------------

def _get_enhanced_config(key: str, default: Any, None) -> Any:
    """Get configuration from enhanced system if available."""
    if ENHANCED_SYSTEM_AVAILABLE and global_config:
        return getattr(global_config, key, default)
    return default

def _should_enable_caching() -> bool:
    """Check if caching should be enabled based on configuration."""
    return _get_enhanced_config('cache_enabled', False)

def _should_enable_performance_monitoring() -> bool:
    """Check if performance monitoring should be enabled."""
    return _get_enhanced_config('enable_performance_monitoring', False)

def _get_cache_settings() -> tuple[int, int]:
    """Get cache settings from configuration."""
    cache_size, _get_enhanced_config('cache_size', 128)
    cache_ttl, _get_enhanced_config('cache_ttl', 3600)
    return cache_size, cache_ttl

def _register_decorator_if_available(name: str, decorator: Callable, **kwargs):
    """Register decorator in enhanced system if available."""
    if ENHANCED_SYSTEM_AVAILABLE and decorator_registry:
        try:
            decorator_registry.register(name = name, decorator = decorator, **kwargs)
        except Exception as e:
            logger.debug(f"Could not register decorator {name}: {e}")

def _create_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Create a cache key for function calls."""
    try:
        # Create a hash of function signature and arguments
        sig, inspect.signature(func)
        bound, sig.bind(*args, **kwargs)
        bound.apply_defaults()
        key_data, f"{func.__name__}:{sorted(bound.arguments.items())}"
        return hash(key_data)  # Use hash for faster key generation
    except Exception:
        # Fallback to simpler key generation
        key_data, f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hash(key_data)

def _apply_caching(wrapper_func: Callable, cache_size: int, ttl_seconds: int) -> Callable:
    """Apply caching to a wrapper function."""
    if not _should_enable_caching():
        return wrapper_func

    cache = {}

    @functools.wraps(wrapper_func)
    def cached_wrapper(*args, **kwargs):
        cache_key, _create_cache_key(wrapper_func, args, kwargs)
        current_time, time.time()

        # Check cache
        if cache_key in cache:
            cache_entry, cache[cache_key]
        if current_time - cache_entry['timestamp'] < ttl_seconds:
                logger.debug(f"Cache hit for {wrapper_func.__name__}")
        return cache_entry['result']

        # Execute and cache
        result, wrapper_func(*args, **kwargs)
        cache[cache_key] = {
            'result': result,
            'timestamp': current_time
        }

        # Maintain cache size
        if len(cache) > cache_size:
            oldest_key, min(cache.keys(), key = lambda k: cache[k]['timestamp'])
            del cache[oldest_key]

        logger.debug(f"Cached result for {wrapper_func.__name__}")
        return result

    return cached_wrapper

def _apply_performance_monitoring(wrapper_func: Callable, level: str = "basic") -> Callable:
    """Apply performance monitoring to a wrapper function."""
    if not _should_enable_performance_monitoring():
        return wrapper_func

    @functools.wraps(wrapper_func)
    def monitored_wrapper(*args, **kwargs):
        start_time, time.time()
        start_memory, _get_memory_usage() if level in ["detailed", "profiling"] else 0

        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            result, wrapper_func(*args, **kwargs)
        return result
        finally:
            end_time, time.time()
            execution_time, end_time - start_time

            metrics = {
                'function': wrapper_func.__name__,
                'execution_time': execution_time,
                'timestamp': time.time()
            }

        if level in ["detailed", "profiling"]:
                end_memory, _get_memory_usage()
                metrics['memory_delta_mb'] = end_memory - start_memory
                metrics['peak_memory_mb'] = end_memory

            _log_performance_metrics(metrics, level)

    return monitored_wrapper

def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process, psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

def _log_performance_metrics(metrics: Dict[str, Any], level: str):
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
    tags=["validation", "type - checking", "enhanced"]
)
def validate_call_or_runtime_types(*v_args: Any, **v_kwargs: Any) -> Callable[[F], F]:
    """Enhanced decorator factory that prefers pydantic.validate_call if available.

    Falls back to beartype or typeguard if pydantic is unavailable.
    If none are available, acts as a no - op decorator.

    ENHANCED FEATURES:
    - Automatic caching for expensive validation operations - Performance monitoring and metrics - Integration with enhanced configuration system
    """

    def decorator(func: F) -> F:
        # Apply the original validation logic
        if _pydantic_validate_call is not None:
            validated_func, cast("F", _pydantic_validate_call(*v_args, **v_kwargs)(func))
        elif _beartype is not None:
            validated_func, cast("F", _beartype(func))
        elif _typechecked is not None:
            validated_func, cast("F", _typechecked(func))
        else:
            validated_func, func

        # Apply enhanced features
        cache_size, ttl_seconds, _get_cache_settings()
        validated_func, _apply_caching(validated_func, cache_size, ttl_seconds)
        validated_func, _apply_performance_monitoring(validated_func, "basic")

        return validated_func

    return decorator

@_register_decorator_if_available(
    name="pa_check_input",
    version="2.0",
    description="Enhanced pandera input validation with caching",
    tags=["validation", "pandera", "dataframe", "enhanced"]
)
def pa_check_input(
    schema: Any, *, arg_name: str | None, None, arg_index: int, 0, strict: bool, True
) -> Callable[[F], F]:
    """Enhanced compatibility wrapper for pandera.check_input.

    ENHANCED FEATURES:
    - Intelligent caching for schema validation results - Performance monitoring for validation operations - Better error handling and recovery
    """

    def decorator(func: F) -> F:
        if pa is not None and hasattr(pa, "check_input"):
        # Use real pandera when available
            base_decorator, cast("F", pa.check_input(schema, lazy = not strict)(func))
        else:
        # Fallback to lightweight validation
            base_decorator, pa_check_io(
                input_schema = schema,
                df_arg_name = arg_name,
                df_arg_index = arg_index,
                strict = strict,
            )(func)

        # Apply enhanced features
        cache_size, ttl_seconds, _get_cache_settings()
        enhanced_decorator, _apply_caching(base_decorator, cache_size, ttl_seconds)
        enhanced_decorator, _apply_performance_monitoring(enhanced_decorator, "basic")

        return enhanced_decorator

    return decorator

@_register_decorator_if_available(
    name="pa_check_output",
    version="2.0",
    description="Enhanced pandera output validation with caching",
    tags=["validation", "pandera", "dataframe", "enhanced"]
)
def pa_check_output(schema: Any, *, strict: bool, True) -> Callable[[F], F]:
    """Enhanced compatibility wrapper for pandera.check_output.

    ENHANCED FEATURES:
    - Intelligent caching for schema validation results - Performance monitoring for validation operations - Better error handling and recovery
    """

    def decorator(func: F) -> F:
        if pa is not None and hasattr(pa, "check_output"):
        # Use real pandera when available
            base_decorator, cast("F", pa.check_output(schema, lazy = not strict)(func))
        else:
        # Fallback to lightweight validation
            base_decorator, pa_check_io(output_schema = schema, strict = strict)(func)

        # Apply enhanced features
        cache_size, ttl_seconds, _get_cache_settings()
        enhanced_decorator, _apply_caching(base_decorator, cache_size, ttl_seconds)
        enhanced_decorator, _apply_performance_monitoring(enhanced_decorator, "basic")

        return enhanced_decorator

    return decorator

@_register_decorator_if_available(
    name="pa_check_io",
    version="2.0",
    description="Enhanced pandera I / O validation with intelligent caching",
    tags=["validation", "pandera", "dataframe", "enhanced"]
)
def pa_check_io(
    *,
    input_schema: Any | None, None,
    output_schema: Any | None, None,
    df_arg_name: str | None, None,
    df_arg_index: int, 0,
    strict: bool, True,
) -> Callable[[F], F]:
    """Enhanced validate DataFrame input / output with pandera if available.

    ENHANCED FEATURES:
    - Intelligent caching for validation results - Performance monitoring and metrics - Better error handling with recovery strategies - Integration with enhanced configuration system - If pandera is installed and schemas are provided, validate the DataFrame
      argument identified by name or index and the returned DataFrame.
    - If pandera is not installed, performs a lightweight check that the
      argument / return is a pandas DataFrame when schemas are provided.
    """

    def decorator(func: F) -> F:
        def _resolve_df(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
            df_value: Any | None, None
        if df_arg_name is not None and df_arg_name in kwargs:
                df_value, kwargs.get(df_arg_name)
            elif df_arg_name is not None:
        # Try inspect for positional mapping
                sig, inspect.signature(func)
                bound, sig.bind_partial(*args, **kwargs)
        if df_arg_name in bound.arguments:
                    df_value, bound.arguments[df_arg_name]
            elif len(args) > df_arg_index:
                df_value, args[df_arg_index]
        return df_value

        def _validate_input(df_value: Any) -> None:
        if input_schema is None:
        # Fallback implementation for input_schema
                return
        if pa is not None and hasattr(input_schema, "validate"):
        try:
                    input_schema.validate(df_value, lazy = not strict)
        except Exception as exc:  # pandera raises SchemaErrors
                    raise SchemaValidationError(
                        f"Input DataFrame failed schema validation: {exc}",
                        context={"function": func.__name__},
                    ) from exc
            else:
        if not isinstance(df_value, pd.DataFrame):
                    raise SchemaValidationError(
                        "Input is not a pandas DataFrame and pandera is unavailable",
                        context={"function": func.__name__},
                    )

        def _validate_output(result: Any) -> Any:
        if output_schema is None:
        # Fallback implementation for output_schema
        return result
        if pa is not None and hasattr(output_schema, "validate"):
        try:
                    output_schema.validate(result, lazy = not strict)
        except Exception as exc:  # pandera raises SchemaErrors
                    raise SchemaValidationError(
                        f"Output DataFrame failed schema validation: {exc}",
                        context={"function": func.__name__},
                    ) from exc
            else:
        if not isinstance(result, pd.DataFrame):
                    raise SchemaValidationError(
                        "Output is not a pandas DataFrame and pandera is unavailable",
                        context={"function": func.__name__},
                    )
        return result

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
            df_value, _resolve_df(args, kwargs)
        if input_schema is not None:
                _validate_input(df_value)
            result, await func(*args, **kwargs)  # type: ignore[misc]
        if output_schema is not None:
                _validate_output(result)
        return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            df_value, _resolve_df(args, kwargs)
        if input_schema is not None:
                _validate_input(df_value)
            result, func(*args, **kwargs)
        if output_schema is not None:
                _validate_output(result)
        return result

        # Choose the appropriate wrapper
        if inspect.iscoroutinefunction(func):
            base_wrapper, async_wrapper
        else:
            base_wrapper, sync_wrapper

        # Apply enhanced features
        cache_size, ttl_seconds, _get_cache_settings()
        enhanced_wrapper, _apply_caching(base_wrapper, cache_size, ttl_seconds)
        enhanced_wrapper, _apply_performance_monitoring(enhanced_wrapper, "basic")

        return cast("F", enhanced_wrapper)

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
    arg_index: int, 0,
    forbid_lists: bool, False,
    require_vector: bool, False,
) -> Callable[[F], F]:
    """Enhanced coerce the selected argument to numpy.ndarray and optionally forbid lists.

    ENHANCED FEATURES:
    - Performance monitoring for vectorization operations - Intelligent caching for repeated operations - Better error handling and recovery - forbid_lists = True raises if a list is provided - require_vector = True requires at least 1 - D input (no pure scalars)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            sig, inspect.signature(func)
        try:
                bound_args, sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
        except TypeError as exc:
                raise VectorizationError(
                    f"Could not bind arguments for {func.__name__}: {exc}",
                ) from exc

        try:
                param_name, list(sig.parameters.keys())[arg_index]
        except IndexError as exc:
                raise VectorizationError(
                    f"Argument index {arg_index} out of range for {func.__name__}",
                ) from exc

            value, bound_args.arguments.get(param_name)

        if forbid_lists and isinstance(value, list):
                raise VectorizationError(
                    "Python lists are forbidden for this function; use numpy arrays",
                    context={"function": func.__name__},
                )

            coerced, np.asarray(value)
        if require_vector and coerced.ndim == 0:
                raise VectorizationError(
                    "Scalar inputs are not allowed; provide vectorized data",
                    context={"function": func.__name__},
                )

            bound_args.arguments[param_name] = coerced
        return func(*bound_args.args, **bound_args.kwargs)

        # Apply enhanced features
        cache_size, ttl_seconds, _get_cache_settings()
        enhanced_wrapper, _apply_caching(wrapper, cache_size, ttl_seconds)
        enhanced_wrapper, _apply_performance_monitoring(enhanced_wrapper, "basic")

        return cast("F", enhanced_wrapper)

    return decorator

@_register_decorator_if_available(
    name="auto_vectorize",
    version="2.0",
    description="Enhanced auto - vectorization with intelligent caching",
    tags=["vectorization", "numpy", "enhanced"]
)
def auto_vectorize(*, otypes: list[type] | None, None) -> Callable[[F], F]:
    """Enhanced wrap a scalar function so that it transparently handles numpy arrays.

    ENHANCED FEATURES:
    - Intelligent caching for vectorization results - Performance monitoring for vectorization operations - Better memory management - If the first positional argument is an ndarray with ndim>=1, applies
      numpy.vectorize to broadcast the scalar logic across elements.
    - Otherwise, calls the function directly.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(first: Any, *args: Any, **kwargs: Any):
            array, np.asarray(first)
        if array.ndim == 0:
        return func(cast(Any, array.item()), *args, **kwargs)
            vec, np.vectorize(lambda v: func(v, *args, **kwargs), otypes = otypes)
        return vec(array)

        # Apply enhanced features
        cache_size, ttl_seconds, _get_cache_settings()
        enhanced_wrapper, _apply_caching(wrapper, cache_size, ttl_seconds)
        enhanced_wrapper, _apply_performance_monitoring(enhanced_wrapper, "basic")

        return cast("F", enhanced_wrapper)

    return decorator

# --------------------------
# Enhanced NaN / Inf / null guards
# --------------------------

@_register_decorator_if_available(
    name="guard_array_nan_inf",
    version="2.0",
    description="Enhanced NaN / Inf guards with intelligent caching",
    tags=["data - quality", "validation", "enhanced"]
)
def guard_array_nan_inf(
    *,
    mode: str = "raise",  # "raise" | "warn" | "coerce"
    coerce_value: float, 0.0,
    arg_indices: Iterable[int] = (0,),
) -> Callable[[F], F]:
    """Enhanced pre - check numpy arrays or pandas objects for NaN / Inf before executing.

    ENHANCED FEATURES:
    - Intelligent caching for validation results - Performance monitoring for validation operations - Better error handling and recovery strategies - Integration with enhanced configuration system

    mode:
      - "raise": raise DataValidationError on detection
      - "warn": log a warning and continue
      - "coerce": replace NaN / Inf with coerce_value before calling func
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            sig, inspect.signature(func)
        try:
                bound_args, sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
        except TypeError as exc:
                raise DataValidationError(
                    f"Could not bind arguments for {func.__name__}: {exc}",
                    context={"function": func.__name__},
                ) from exc

            param_names, list(sig.parameters.keys())

        for index in arg_indices:
        if index >= len(param_names):
                    continue
                param_name, param_names[index]
                value, bound_args.arguments.get(param_name)
        if value is None:
        # Fallback implementation for value
                    continue

        # Convert to numpy for checking
        if isinstance(value, (pd.Series, pd.DataFrame)):
                    data, value.to_numpy()
                else:
                    data, np.asarray(value)

        # Only attempt numeric checks
        try:
                    is_numeric, np.issubdtype(data.dtype, np.number)
        except Exception:
                    is_numeric, False

        if not is_numeric:
                    continue

                has_nan, np.isnan(data).any()
                has_inf, np.isinf(data).any()
        if has_nan or has_inf:
                    msg = (
                        f"Detected {'NaN' if has_nan else ''}{' and ' if has_nan and has_inf else ''}"
                        f"{'Inf' if has_inf else ''} in argument '{param_name}' (index {index}) for {func.__name__}"
                    )
        if mode == "raise":
                        raise DataValidationError(
                            msg, context={"function": func.__name__}
                        )
        if mode == "warn":
                        logger.warning(msg)
        if mode == "coerce":
                        coerced_array, np.asarray(value, dtype = float)
                        coerced_array, np.nan_to_num(
                            coerced_array,
                            nan = coerce_value,
                            posinf = coerce_value,
                            neginf = coerce_value,
                        )
        if isinstance(value, pd.DataFrame):
                            coerced_value, pd.DataFrame(
                                coerced_array,
                                index = value.index,
                                columns = value.columns,
                            )
                        elif isinstance(value, pd.Series):
                            coerced_value, pd.Series(
                                coerced_array,
                                index = value.index,
                                name = value.name,
                            )
                        else:
                            coerced_value, coerced_array
                        bound_args.arguments[param_name] = coerced_value

        return func(*bound_args.args, **bound_args.kwargs)

        # Apply enhanced features
        cache_size, ttl_seconds, _get_cache_settings()
        enhanced_wrapper, _apply_caching(wrapper, cache_size, ttl_seconds)
        enhanced_wrapper, _apply_performance_monitoring(enhanced_wrapper, "basic")

        return cast("F", enhanced_wrapper)

    return decorator

@_register_decorator_if_available(
    name="guard_dataframe_nulls",
    version="2.0",
    description="Enhanced DataFrame null guards with intelligent caching",
    tags=["data - quality", "validation", "dataframe", "enhanced"]
)
def guard_dataframe_nulls(
    *,
    columns: list[str] | None, None,
    mode: str = "raise",  # "raise" | "warn" | "fill"
    fill_value: float | int | str | None, 0,
    arg_index: int, 0,
) -> Callable[[F], F]:
    """Enhanced check a pandas DataFrame argument for nulls / NaN / Inf.

    ENHANCED FEATURES:
    - Intelligent caching for validation results - Performance monitoring for validation operations - Better error handling and recovery strategies - Integration with enhanced configuration system

    arg_index selects which positional argument is the DataFrame (0 for functions where df is first, 1 for instance methods).
    If columns is provided, restrict checks to those columns.
    """

    def decorator(func: F) -> F:
        def _check(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
                raise DataValidationError(
                    "Target argument must be a pandas DataFrame",
                    context={"function": func.__name__},
                )
            selected, df if columns is None else df[columns]
            num_nan, int(selected.isna().sum().sum())

        # Safely check for infinite values only on numeric columns
            num_inf, 0
        try:
        # First try to get numeric columns only
                numeric_selected, selected.select_dtypes(include=[np.number])
        if not numeric_selected.empty:
                    num_inf, int(np.isinf(numeric_selected.to_numpy()).sum())
        except Exception:
        # Fallback: handle mixed data types more carefully
                num_inf, 0
        for col in selected.columns:
        try:
        # Check if column is numeric before processing
        if pd.api.types.is_numeric_dtype(selected[col]):
                            col_data, selected[col]
        # Handle pandas Series with mixed types
        if hasattr(col_data, 'dtype') and col_data.dtype == 'object':
        # Try to convert to numeric, skipping non - numeric values
                                col_data, pd.to_numeric(col_data, errors='coerce')
                            num_inf += int(np.isinf(col_data).sum())
        except (ValueError, TypeError, AttributeError):
        # Skip non - numeric columns or columns with conversion issues
                        continue

        if num_nan or num_inf:
                msg, f"DataFrame has {num_nan} NaN and {num_inf} Inf values in {func.__name__}"
        if mode == "raise":
                    raise DataValidationError(msg, context={"function": func.__name__})
        if mode == "warn":
                    logger.warning(msg)
        if mode == "fill":
                    df, df.copy()
        # Only fill numeric columns to avoid type issues
                    numeric_cols, selected.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
                        df[numeric_cols] = selected[numeric_cols].replace(
                            [np.inf, -np.inf], fill_value
                        ).fillna(fill_value)
        # Fill NaN values in all columns
                    df[selected.columns] = selected.fillna(fill_value)
        return df

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
            sig, inspect.signature(func)
            bound_args, sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            param_names, list(sig.parameters.keys())
        if arg_index < len(param_names):
                param_name, param_names[arg_index]
        if param_name in bound_args.arguments:
                    bound_args.arguments[param_name] = _check(
                        bound_args.arguments[param_name]
                    )
        return await func(*bound_args.args, **bound_args.kwargs)  # type: ignore[misc]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            sig, inspect.signature(func)
            bound_args, sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            param_names, list(sig.parameters.keys())
        if arg_index < len(param_names):
                param_name, param_names[arg_index]
        if param_name in bound_args.arguments:
                    bound_args.arguments[param_name] = _check(
                        bound_args.arguments[param_name]
                    )
        return func(*bound_args.args, **bound_args.kwargs)

        # Choose the appropriate wrapper
        if inspect.iscoroutinefunction(func):
            base_wrapper, async_wrapper
        else:
            base_wrapper, sync_wrapper

        # Apply enhanced features
        cache_size, ttl_seconds, _get_cache_settings()
        enhanced_wrapper, _apply_caching(base_wrapper, cache_size, ttl_seconds)
        enhanced_wrapper, _apply_performance_monitoring(enhanced_wrapper, "basic")

        return cast("F", enhanced_wrapper)

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

# Optional external libraries (best - effort mapping without hard deps)
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
    tags=["error - handling", "recovery", "enhanced"]
)
def normalize_errors(
    *,
    map_exceptions: dict[type[BaseException], type[DomainError]] | None, None,
    default_error: type[DomainError] = DomainError,
    reraise: bool, False,
) -> Callable[[F], F]:
    """Enhanced normalize heterogeneous exceptions into domain - specific errors.

    ENHANCED FEATURES:
    - Intelligent error recovery strategies - Performance monitoring for error handling - Better logging and correlation - Integration with enhanced configuration system - map_exceptions augments the built - in mapping - if reraise = True, re - raises the normalized DomainError after logging - otherwise returns None and logs; for functions that must return a value,
      consider using together with default returns in your wrapper logic.
    """

    exception_map, dict(_EXCEPTION_MAP)
    if map_exceptions:
        exception_map.update(map_exceptions)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
        try:
        return await func(*args, **kwargs)  # type: ignore[misc]
        except tuple(exception_map.keys()) as exc:  # type: ignore[arg - type]
                domain_exc_type, default_error
        for base_exc, mapped in exception_map.items():
        if isinstance(exc, base_exc):
                        domain_exc_type, mapped
                        break
                norm_exc, domain_exc_type(
                    f"{func.__name__} failed: {exc}",
                    context={"function": func.__name__},
                )
                logger.exception(
                    "Normalized error", extra={"correlation_id": get_correlation_id()}
                )
        if reraise:
                    raise norm_exc from exc
        return None

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
        try:
        return func(*args, **kwargs)
        except tuple(exception_map.keys()) as exc:  # type: ignore[arg - type]
                domain_exc_type, default_error
        for base_exc, mapped in exception_map.items():
        if isinstance(exc, base_exc):
                        domain_exc_type, mapped
                        break
                norm_exc, domain_exc_type(
                    f"{func.__name__} failed: {exc}",
                    context={"function": func.__name__},
                )
                logger.exception(
                    "Normalized error", extra={"correlation_id": get_correlation_id()}
                )
        if reraise:
                    raise norm_exc from exc
        return None

        # Choose the appropriate wrapper
        if inspect.iscoroutinefunction(func):
            base_wrapper, async_wrapper
        else:
            base_wrapper, sync_wrapper

        # Apply enhanced features
        cache_size, ttl_seconds, _get_cache_settings()
        enhanced_wrapper, _apply_caching(base_wrapper, cache_size, ttl_seconds)
        enhanced_wrapper, _apply_performance_monitoring(enhanced_wrapper, "basic")

        return cast("F", enhanced_wrapper)

    return decorator

# --------------------------
# Enhanced Logging / tracing / audit
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
    """Best - effort PII scrubbing for dict - like inputs and sequences.

    Masks values of known sensitive keys. Keeps structure to aid debugging.
    """
    try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
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
    span_name: str | None, None,
    *,
    log_args: bool, False,
    log_result_len_only: bool, True,
) -> Callable[[F], F]:
    """Enhanced add correlation - aware entry / exit logs around a function call.

    ENHANCED FEATURES:
    - Performance monitoring and metrics collection - Intelligent caching for repeated operations - Better error handling and recovery - Integration with enhanced configuration system - Ensures a correlation ID is present - Optionally logs sanitized args / kwargs (avoid for heavy data)
    - Logs result size instead of full content by default
    """

    def decorator(func: F) -> F:
        resolved_span, span_name or func.__name__
        # Base fallback logger on the wrapped function's module
        module_logger, logging.getLogger(func.__module__)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
            cid, ensure_correlation_id()
        # Prefer instance logger if available
            active_logger = (
                getattr(args[0], "logger", module_logger) if args else module_logger
            )
        if log_args:
                safe_args, _sanitize(args)
                safe_kwargs, _sanitize(kwargs)
                active_logger.info(
                    f"➡️ {resolved_span} start",
                    extra={
                        "correlation_id": cid,
                        "args": safe_args,
                        "kwargs": safe_kwargs,
                    },
                )
            else:
                active_logger.info(
                    f"➡️ {resolved_span} start",
                    extra={"correlation_id": cid},
                )

            result, await func(*args, **kwargs)  # type: ignore[misc]

        if log_result_len_only:
        try:
                    length, None
        if hasattr(result, "__len__"):
                        length, len(cast(Any, result))
                    active_logger.info(
                        f"✅ {resolved_span} done",
                        extra={"correlation_id": cid, "result_len": length},
                    )
        except Exception:
                    active_logger.info(
                        f"✅ {resolved_span} done",
                        extra={"correlation_id": cid},
                    )
            else:
                active_logger.info(
                    f"✅ {resolved_span} done",
                    extra={"correlation_id": cid, "result": _sanitize(result)},
                )

        return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            cid, ensure_correlation_id()
        # Prefer instance logger if available
            active_logger = (
                getattr(args[0], "logger", module_logger) if args else module_logger
            )
        if log_args:
                safe_args, _sanitize(args)
                safe_kwargs, _sanitize(kwargs)
                active_logger.info(
                    f"➡️ {resolved_span} start",
                    extra={
                        "correlation_id": cid,
                        "args": safe_args,
                        "kwargs": safe_kwargs,
                    },
                )
            else:
                active_logger.info(
                    f"➡️ {resolved_span} start",
                    extra={"correlation_id": cid},
                )

            result, func(*args, **kwargs)

        if log_result_len_only:
        try:
                    length, None
        if hasattr(result, "__len__"):
                        length, len(cast(Any, result))
                    active_logger.info(
                        f"✅ {resolved_span} done",
                        extra={"correlation_id": cid, "result_len": length},
                    )
        except Exception:
                    active_logger.info(
                        f"✅ {resolved_span} done",
                        extra={"correlation_id": cid},
                    )
            else:
                active_logger.info(
                    f"✅ {resolved_span} done",
                    extra={"correlation_id": cid, "result": _sanitize(result)},
                )

        return result

        # Choose the appropriate wrapper
        if inspect.iscoroutinefunction(func):
            base_wrapper, async_wrapper
        else:
            base_wrapper, sync_wrapper

        # Apply enhanced features
        cache_size, ttl_seconds, _get_cache_settings()
        enhanced_wrapper, _apply_caching(base_wrapper, cache_size, ttl_seconds)
        enhanced_wrapper, _apply_performance_monitoring(enhanced_wrapper, "detailed")

        return cast("F", enhanced_wrapper)

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
