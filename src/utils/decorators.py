"""
Reusable decorators for validation, vectorization, data hygiene, error normalization, and tracing.

- Type / shape / schema validation: integrates with pydantic.validate_call if available,
and optionally beartype / typeguard. Pandera DataFrame schema checks are supported when installed.
- Vectorization guarantees: auto - vectorize scalar logic or enforce ndarray inputs.
- NaN / Inf / null guards: fast pre - checks for arrays / DataFrames with helpful messages.
- Error normalization: centralize exception mapping into domain - specific errors.
- Logging / tracing / audit: correlation IDs and structured entry / exit logs with PII scrubbing.

ENHANCED FEATURES:
    passpass - Integration with enhanced decorator system - Better error handling and recovery - Intelligent caching for expensive operations - Performance monitoring and metrics - Centralized configuration support
"""

import functools
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any, Iterable, TypeVar, cast, Dict, Optional

# Handle optional dependencies
try:
    passpasspassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
import numpy as np
NUMPY_AVAILABLE, True
except ImportError:
    passpassNUMPY_AVAILABLE, False
np, None

try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
import pandas as pd
PANDAS_AVAILABLE, True
except ImportError:
    passpassPANDAS_AVAILABLE, False
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
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
from .decorator_config import global_config
from .decorator_registry import decorator_registry, register_decorator
ENHANCED_SYSTEM_AVAILABLE, True
except ImportError:
    passpassENHANCED_SYSTEM_AVAILABLE, False
global_config, None
decorator_registry, None

T, TypeVar("T")
F, TypeVar("F", bound = Callable[..., Any])

logger, logging.getLogger(__name__)

# Optional imports for integrations
try:  # Pydantic v2
    self.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
from pydantic import validate_call as _pydantic_validate_call  # type: ignore
except Exception:  # pragma: no cover
_pydantic_validate_call, None  # type: ignore

try:  # beartype
    self.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
from beartype import beartype as _beartype  # type: ignore
except Exception:  # pragma: no cover
_beartype, None  # type: ignore

try:  # typeguard
    self.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
from typeguard import typechecked as _typechecked  # type: ignore
except Exception:  # pragma: no cover
_typechecked, None  # type: ignore

try:  # pandera
    self.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
import pandera as pa  # type: ignore
except Exception:  # pragma: no cover
pa, None  # type: ignore

# --------------------------
# Enhanced helper functions
# --------------------------

def _get_enhanced_config(...) -> ...:
    """..."""
    passif ENHANCED_SYSTEM_AVAILABLE and global_config:
    passreturn getattr(global_config, key, default)
return default

def _should_enable_caching(...) -> ...:
    """..."""
    passreturn _get_enhanced_config('cache_enabled', False)

def _should_enable_performance_monitoring(...) -> ...:
    """..."""
    passreturn _get_enhanced_config('enable_performance_monitoring', False)

def _get_cache_settings(...) -> ...:
    """..."""
    passcache_size, _get_enhanced_config('cache_size', 128)
cache_ttl, _get_enhanced_config('cache_ttl', 3600)
return cache_size, cache_ttl

def _register_decorator_if_available(...):
    passdef _register_decorator_if_available(...):
    passdef _register_decorator_if_available(...):
    passdef _register_decorator_if_available(...):
    pass"""Register decorator in enhanced system if available."""
if ENHANCED_SYSTEM_AVAILABLE and decorator_registry:
    passpasstry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
decorator_registry.register(name = name, decorator = decorator, **kwargs)
except Exception as e:
    passpasspasspasspasspasspasslogger.debug(f"Could not register decorator {name}: {e}")

def _create_cache_key(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Create a hash of function signature and arguments
sig, inspect.signature(func)
bound, sig.bind(*args, **kwargs)
bound.apply_defaults()
key_data, f"{func.__name__}:{sorted(bound.arguments.items())}"
return hash(key_data)  # Use hash for faster key generation
except Exception:
    passpasspass# Fallback to simpler key generation
key_data, f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
return hash(key_data)

def _apply_caching(...) -> ...:
    """..."""
    passif not _should_enable_caching():
    passreturn wrapper_func

cache = {}

@functools.wraps(wrapper_func)
def cached_wrapper(...):
    passdef cached_wrapper(...):
    passdef cached_wrapper(...):
    passdef cached_wrapper(...):
    passcache_key, _create_cache_key(wrapper_func, args, kwargs)
current_time, time.time()

# Check cache
if cache_key in cache:
    passcache_entry, cache[cache_key]
if current_time - cache_entry['timestamp'] < ttl_seconds:
    passlogger.debug(f"Cache hit for {wrapper_func.__name__}")
return cache_entry['result']

# Execute and cache
result, wrapper_func(*args, **kwargs)
cache[cache_key] = {
'result': result,
'timestamp': current_time
}

# Maintain cache size
if len(cache) > cache_size:
    passoldest_key, min(cache.keys(), key = lambda k: cache[k]['timestamp'])
del cache[oldest_key]

logger.debug(f"Cached result for {wrapper_func.__name__}")
return result

return cached_wrapper

def _apply_performance_monitoring(...) -> ...:
    pass"""..."""
    passif not _should_enable_performance_monitoring():
    passreturn wrapper_func

@functools.wraps(wrapper_func)
def monitored_wrapper(...):
    passdef monitored_wrapper(...):
    passdef monitored_wrapper(...):
    passdef monitored_wrapper(...):
    passstart_time, time.time()
start_memory, _get_memory_usage() if level in ["detailed", "profiling"] else 0

try:
    passpassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
result, wrapper_func(*args, **kwargs)
return result
finally:
    passend_time, time.time()
execution_time, end_time - start_time

metrics = {
'function': wrapper_func.__name__,
'execution_time': execution_time,
'timestamp': time.time()
}

if level in ["detailed", "profiling"]:
    passend_memory, _get_memory_usage()
metrics['memory_delta_mb'] = end_memory - start_memory
metrics['peak_memory_mb'] = end_memory

_log_performance_metrics(metrics, level)

return monitored_wrapper

def _get_memory_usage(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
import psutil
process, psutil.Process()
return process.memory_info().rss / 1024 / 1024
except ImportError:
    passpassreturn 0.0

def _log_performance_metrics(...):
    passdef _log_performance_metrics(...):
    passdef _log_performance_metrics(...):
    passdef _log_performance_metrics(...):
    pass"""Log performance metrics based on level."""
if level == "basic":
    passlogger.info(f"Performance: {metrics['function']} took {metrics['execution_time']:.3f}s")
elif level == "detailed":
    passpasslogger.info(f"Performance details for {metrics['function']}: {metrics}")
elif level == "profiling":
    passpasslogger.debug(f"Performance profiling for {metrics['function']}: {metrics}")

# --------------------------
# Enhanced Type / schema validation
# --------------------------

@_register_decorator_if_available(
name="validate_call_or_runtime_types",
version="2.0",
description="Enhanced type validation with caching and performance monitoring",
tags=["validation", "type - checking", "enhanced"]
)
def validate_call_or_runtime_types(...) -> ...:
    pass"""..."""
    passdef decorator(func: F) -> F:
        # Apply the original validation logic
if _pydantic_validate_call is not None:
    passvalidated_func, cast("F", _pydantic_validate_call(*v_args, **v_kwargs)(func))
elif _beartype is not None:
    passpassvalidated_func, cast("F", _beartype(func))
elif _typechecked is not None:
    passpassvalidated_func, cast("F", _typechecked(func))
else:
    passvalidated_func, func

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
def pa_check_input(...) -> ...:
    pass"""..."""
    passdef decorator(func: F) -> F:
        if pa is not None and hasattr(pa, "check_input"):
    pass# Use real pandera when available
base_decorator, cast("F", pa.check_input(schema, lazy = not strict)(func))
else:
    pass# Fallback to lightweight validation
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
def pa_check_output(...) -> ...:
    pass"""..."""
    passdef decorator(func: F) -> F:
        if pa is not None and hasattr(pa, "check_output"):
    pass# Use real pandera when available
base_decorator, cast("F", pa.check_output(schema, lazy = not strict)(func))
else:
    pass# Fallback to lightweight validation
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
def pa_check_io(...) -> ...:
    pass"""..."""
    passdef decorator(func: F) -> F:
        def _resolve_df(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
            df_value: Any | None, None
if df_arg_name is not None and df_arg_name in kwargs:
    passdf_value, kwargs.get(df_arg_name)
elif df_arg_name is not None:
    passpass# Try inspect for positional mapping
sig, inspect.signature(func)
bound, sig.bind_partial(*args, **kwargs)
if df_arg_name in bound.arguments:
    passpassdf_value, bound.arguments[df_arg_name]
elif len(args) > df_arg_index:
    passpassdf_value, args[df_arg_index]
return df_value

def _validate_input(df_value: Any) -> None:
        if input_schema is None:
    pass# Fallback implementation for input_schema
return
if pa is not None and hasattr(input_schema, "validate"):
    passpasstry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
input_schema.validate(df_value, lazy = not strict)
except Exception as exc:  # pandera raises SchemaErrors
raise SchemaValidationError(
f"Input DataFrame failed schema validation: {exc}",
context={"function": func.__name__},
) from exc
else:
    passif not isinstance(df_value, pd.DataFrame):
    passraise SchemaValidationError(
"Input is not a pandas DataFrame and pandera is unavailable",
context={"function": func.__name__},
)

def _validate_output(result: Any) -> Any:
        if output_schema is None:
    pass# Fallback implementation for output_schema
return result
if pa is not None and hasattr(output_schema, "validate"):
    passpasstry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
output_schema.validate(result, lazy = not strict)
except Exception as exc:  # pandera raises SchemaErrors
raise SchemaValidationError(
f"Output DataFrame failed schema validation: {exc}",
context={"function": func.__name__},
) from exc
else:
    passif not isinstance(result, pd.DataFrame):
    passraise SchemaValidationError(
"Output is not a pandas DataFrame and pandera is unavailable",
context={"function": func.__name__},
)
return result

@functools.wraps(func)
async def async_wrapper(...):
    passself.logger.info("Implementation placeholder - needs specific logic")
async def async_wrapper(...):
    passself.logger.info("Implementation placeholder - needs specific logic")
async def async_wrapper(...):
    passdf_value, _resolve_df(args, kwargs)
if input_schema is not None:
    pass_validate_input(df_value)
result, await func(*args, **kwargs)  # type: ignore[misc]
if output_schema is not None:
    pass_validate_output(result)
return result

@functools.wraps(func)
def sync_wrapper(...):
    passdef sync_wrapper(...):
    passdef sync_wrapper(...):
    passdef sync_wrapper(...):
    passdf_value, _resolve_df(args, kwargs)
if input_schema is not None:
    pass_validate_input(df_value)
result, func(*args, **kwargs)
if output_schema is not None:
    pass_validate_output(result)
return result

# Choose the appropriate wrapper
if inspect.iscoroutinefunction(func):
    passbase_wrapper, async_wrapper
else:
    passbase_wrapper, sync_wrapper

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
def enforce_ndarray(...) -> ...:
    pass"""..."""
    passdef decorator(func: F) -> F:
        @functools.wraps(func)
def wrapper(...):
    passdef wrapper(...):
    passdef wrapper(...):
    passdef wrapper(...):
    passsig, inspect.signature(func)
try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
bound_args, sig.bind(*args, **kwargs)
bound_args.apply_defaults()
except TypeError as exc:
    passpasspasspasspasspasspassraise VectorizationError(
f"Could not bind arguments for {func.__name__}: {exc}",
) from exc

try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
param_name, list(sig.parameters.keys())[arg_index]
except IndexError as exc:
    passpasspasspasspasspasspassraise VectorizationError(
f"Argument index {arg_index} out of range for {func.__name__}",
) from exc

value, bound_args.arguments.get(param_name)

if forbid_lists and isinstance(value, list):
    passpassraise VectorizationError(
"Python lists are forbidden for this function; use numpy arrays",
context={"function": func.__name__},
)

coerced, np.asarray(value)
if require_vector and coerced.ndim == 0:
    passraise VectorizationError(
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
def auto_vectorize(...) -> ...:
    pass"""..."""
    passdef decorator(func: F) -> F:
        @functools.wraps(func)
def wrapper(...):
    passdef wrapper(...):
    passdef wrapper(...):
    passdef wrapper(...):
    passarray, np.asarray(first)
if array.ndim == 0:
    passreturn func(cast(Any, array.item()), *args, **kwargs)
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
    pass- Intelligent caching for validation results - Performance monitoring for validation operations - Better error handling and recovery strategies - Integration with enhanced configuration system

mode:
    passpass- "raise": raise DataValidationError on detection
- "warn": log a warning and continue
- "coerce": replace NaN / Inf with coerce_value before calling func
"""

def decorator(func: F) -> F:
        @functools.wraps(func)
def wrapper(...):
    passdef wrapper(...):
    passdef wrapper(...):
    passdef wrapper(...):
    passsig, inspect.signature(func)
try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
bound_args, sig.bind(*args, **kwargs)
bound_args.apply_defaults()
except TypeError as exc:
    passpasspasspasspasspasspassraise DataValidationError(
f"Could not bind arguments for {func.__name__}: {exc}",
context={"function": func.__name__},
) from exc

param_names, list(sig.parameters.keys())

for index in arg_indices:
    passif index >= len(param_names):
    passcontinue
param_name, param_names[index]
value, bound_args.arguments.get(param_name)
if value is None:
    pass# Fallback implementation for value
continue

# Convert to numpy for checking
if isinstance(value, (pd.Series, pd.DataFrame)):
    passpassdata, value.to_numpy()
else:
    passdata, np.asarray(value)

# Only attempt numeric checks
try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
is_numeric, np.issubdtype(data.dtype, np.number)
except Exception:
    passpassis_numeric, False

if not is_numeric:
    passcontinue

has_nan, np.isnan(data).any()
has_inf, np.isinf(data).any()
if has_nan or has_inf:
    passmsg = (
f"Detected {'NaN' if has_nan else ''}{' and ' if has_nan and has_inf else ''}"
f"{'Inf' if has_inf else ''} in argument '{param_name}' (index {index}) for {func.__name__}"
)
if mode == "raise":
    passpassraise DataValidationError(
msg, context={"function": func.__name__}
)
if mode == "warn":
    passlogger.warning(msg)
if mode == "coerce":
    passcoerced_array, np.asarray(value, dtype = float)
coerced_array, np.nan_to_num(
coerced_array,
nan = coerce_value,
posinf = coerce_value,
neginf = coerce_value,
)
if isinstance(value, pd.DataFrame):
    passcoerced_value, pd.DataFrame(
coerced_array,
index = value.index,
columns = value.columns,
)
elif isinstance(value, pd.Series):
    passpasscoerced_value, pd.Series(
coerced_array,
index = value.index,
name = value.name,
)
else:
    passcoerced_value, coerced_array
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
def guard_dataframe_nulls(...) -> ...:
    pass"""..."""
    passdef decorator(func: F) -> F:
        def _check(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
    passraise DataValidationError(
"Target argument must be a pandas DataFrame",
context={"function": func.__name__},
)
selected, df if columns is None else df[columns]
num_nan, int(selected.isna().sum().sum())

# Safely check for infinite values only on numeric columns
num_inf, 0
try:
    passpasspassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# First try to get numeric columns only
numeric_selected, selected.select_dtypes(include=[np.number])
if not numeric_selected.empty:
    passnum_inf, int(np.isinf(numeric_selected.to_numpy()).sum())
except Exception:
    passpass# Fallback: handle mixed data types more carefully
num_inf, 0
for col in selected.columns:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Check if column is numeric before processing
if pd.api.types.is_numeric_dtype(selected[col]):
    passcol_data, selected[col]
# Handle pandas Series with mixed types
if hasattr(col_data, 'dtype') and col_data.dtype == 'object':
    passpass# Try to convert to numeric, skipping non - numeric values
col_data, pd.to_numeric(col_data, errors='coerce')
num_inf += int(np.isinf(col_data).sum())
except (ValueError, TypeError, AttributeError):
    passpass# Skip non - numeric columns or columns with conversion issues
continue

if num_nan or num_inf:
    passpassmsg, f"DataFrame has {num_nan} NaN and {num_inf} Inf values in {func.__name__}"
if mode == "raise":
    passraise DataValidationError(msg, context={"function": func.__name__})
if mode == "warn":
    passlogger.warning(msg)
if mode == "fill":
    passdf, df.copy()
# Only fill numeric columns to avoid type issues
numeric_cols, selected.select_dtypes(include=[np.number]).columns
if not numeric_cols.empty:
    passdf[numeric_cols] = selected[numeric_cols].replace(
[np.inf, -np.inf], fill_value
).fillna(fill_value)
# Fill NaN values in all columns
df[selected.columns] = selected.fillna(fill_value)
return df

@functools.wraps(func)
async def async_wrapper(...):
    passself.logger.info("Implementation placeholder - needs specific logic")
async def async_wrapper(...):
    passself.logger.info("Implementation placeholder - needs specific logic")
async def async_wrapper(...):
    passsig, inspect.signature(func)
bound_args, sig.bind(*args, **kwargs)
bound_args.apply_defaults()
param_names, list(sig.parameters.keys())
if arg_index < len(param_names):
    passparam_name, param_names[arg_index]
if param_name in bound_args.arguments:
    passbound_args.arguments[param_name] = _check(
bound_args.arguments[param_name]
)
return await func(*bound_args.args, **bound_args.kwargs)  # type: ignore[misc]

@functools.wraps(func)
def sync_wrapper(...):
    passdef sync_wrapper(...):
    passdef sync_wrapper(...):
    passdef sync_wrapper(...):
    passsig, inspect.signature(func)
bound_args, sig.bind(*args, **kwargs)
bound_args.apply_defaults()
param_names, list(sig.parameters.keys())
if arg_index < len(param_names):
    passparam_name, param_names[arg_index]
if param_name in bound_args.arguments:
    passbound_args.arguments[param_name] = _check(
bound_args.arguments[param_name]
)
return func(*bound_args.args, **bound_args.kwargs)

# Choose the appropriate wrapper
if inspect.iscoroutinefunction(func):
    passbase_wrapper, async_wrapper
else:
    passbase_wrapper, sync_wrapper

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
    self.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
import requests  # type: ignore

_EXCEPTION_MAP[requests.exceptions.RequestException] = ExternalServiceError  # type: ignore
except Exception:  # pragma: no cover
pass

try:  # aiohttp
    self.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
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
def normalize_errors(...) -> ...:
    pass"""..."""
    passexception_map, dict(_EXCEPTION_MAP)
if map_exceptions:
    passexception_map.update(map_exceptions)

def decorator(func: F) -> F:
        @functools.wraps(func)
async def async_wrapper(...):
    passself.logger.info("Implementation placeholder - needs specific logic")
async def async_wrapper(...):
    passself.logger.info("Implementation placeholder - needs specific logic")
async def async_wrapper(...):
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
return await func(*args, **kwargs)  # type: ignore[misc]
except tuple(exception_map.keys()) as exc:  # type: ignore[arg - type]
domain_exc_type, default_error
for base_exc, mapped in exception_map.items():
    passif isinstance(exc, base_exc):
    passdomain_exc_type, mapped
break
norm_exc, domain_exc_type(
f"{func.__name__} failed: {exc}",
context={"function": func.__name__},
)
logger.exception(
"Normalized error", extra={"correlation_id": get_correlation_id()}
)
if reraise:
    passraise norm_exc from exc
return None

@functools.wraps(func)
def sync_wrapper(...):
    passdef sync_wrapper(...):
    passdef sync_wrapper(...):
    passdef sync_wrapper(...):
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
return func(*args, **kwargs)
except tuple(exception_map.keys()) as exc:  # type: ignore[arg - type]
domain_exc_type, default_error
for base_exc, mapped in exception_map.items():
    passif isinstance(exc, base_exc):
    passdomain_exc_type, mapped
break
norm_exc, domain_exc_type(
f"{func.__name__} failed: {exc}",
context={"function": func.__name__},
)
logger.exception(
"Normalized error", extra={"correlation_id": get_correlation_id()}
)
if reraise:
    passraise norm_exc from exc
return None

# Choose the appropriate wrapper
if inspect.iscoroutinefunction(func):
    passbase_wrapper, async_wrapper
else:
    passbase_wrapper, sync_wrapper

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

def _sanitize(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
if isinstance(value, dict):
    passredacted: dict[str, Any] = {}
for key, val in value.items():
    passif str(key).lower() in _SENSITIVE_KEYS:
    passredacted[key] = "***REDACTED***"
else:
    passredacted[key] = _sanitize(val)
return redacted
if isinstance(value, (list, tuple)):
    passreturn type(value)(_sanitize(v) for v in value)
return value
except Exception:
    passpasspassreturn value

@_register_decorator_if_available(
name="with_tracing_span",
version="2.0",
description="Enhanced tracing with performance monitoring and caching",
tags=["tracing", "logging", "performance", "enhanced"]
)
def with_tracing_span(...) -> ...:
    pass"""..."""
    passdef decorator(func: F) -> F:
        resolved_span, span_name or func.__name__
# Base fallback logger on the wrapped function's module
module_logger, logging.getLogger(func.__module__)

@functools.wraps(func)
async def async_wrapper(...):
    passself.logger.info("Implementation placeholder - needs specific logic")
async def async_wrapper(...):
    passself.logger.info("Implementation placeholder - needs specific logic")
async def async_wrapper(...):
    passcid, ensure_correlation_id()
# Prefer instance logger if available
active_logger = (
getattr(args[0], "logger", module_logger) if args else module_logger
)
if log_args:
    passsafe_args, _sanitize(args)
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
    passactive_logger.info(
f"➡️ {resolved_span} start",
extra={"correlation_id": cid},
)

result, await func(*args, **kwargs)  # type: ignore[misc]

if log_result_len_only:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
length, None
if hasattr(result, "__len__"):
    passlength, len(cast(Any, result))
active_logger.info(
f"✅ {resolved_span} done",
extra={"correlation_id": cid, "result_len": length},
)
except Exception:
    passpassactive_logger.info(
f"✅ {resolved_span} done",
extra={"correlation_id": cid},
)
else:
    passactive_logger.info(
f"✅ {resolved_span} done",
extra={"correlation_id": cid, "result": _sanitize(result)},
)

return result

@functools.wraps(func)
def sync_wrapper(...):
    passdef sync_wrapper(...):
    passdef sync_wrapper(...):
    passdef sync_wrapper(...):
    passcid, ensure_correlation_id()
# Prefer instance logger if available
active_logger = (
getattr(args[0], "logger", module_logger) if args else module_logger
)
if log_args:
    passsafe_args, _sanitize(args)
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
    passactive_logger.info(
f"➡️ {resolved_span} start",
extra={"correlation_id": cid},
)

result, func(*args, **kwargs)

if log_result_len_only:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
length, None
if hasattr(result, "__len__"):
    passlength, len(cast(Any, result))
active_logger.info(
f"✅ {resolved_span} done",
extra={"correlation_id": cid, "result_len": length},
)
except Exception:
    passpassactive_logger.info(
f"✅ {resolved_span} done",
extra={"correlation_id": cid},
)
else:
    passactive_logger.info(
f"✅ {resolved_span} done",
extra={"correlation_id": cid, "result": _sanitize(result)},
)

return result

# Choose the appropriate wrapper
if inspect.iscoroutinefunction(func):
    passbase_wrapper, async_wrapper
else:
    passbase_wrapper, sync_wrapper

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
