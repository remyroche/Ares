"""
Reusable decorators for validation, vectorization, data hygiene, error normalization, and tracing.

- Type/shape/schema validation: integrates with pydantic.validate_call if available,
  and optionally beartype/typeguard. Pandera DataFrame schema checks are supported when installed.
- Vectorization guarantees: auto-vectorize scalar logic or enforce ndarray inputs.
- NaN/Inf/null guards: fast pre-checks for arrays/DataFrames with helpful messages.
- Error normalization: centralize exception mapping into domain-specific errors.
- Logging/tracing/audit: correlation IDs and structured entry/exit logs with PII scrubbing.
"""
from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any, Iterable, TypeVar, cast

import numpy as np
import pandas as pd

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
# Type/schema validation
# --------------------------

def validate_call_or_runtime_types(*v_args: Any, **v_kwargs: Any) -> Callable[[F], F]:
    """Decorator factory that prefers pydantic.validate_call if available.

    Falls back to beartype or typeguard if pydantic is unavailable.
    If none are available, acts as a no-op decorator.
    """

    def decorator(func: F) -> F:  # type: ignore[override]
        if _pydantic_validate_call is not None:
            return cast("F", _pydantic_validate_call(*v_args, **v_kwargs)(func))
        if _beartype is not None:
            return cast("F", _beartype(func))
        if _typechecked is not None:
            return cast("F", _typechecked(func))
        return func

    return decorator


def pa_check_input(schema: Any, *, arg_name: str | None = None, arg_index: int = 0, strict: bool = True) -> Callable[[F], F]:
    """Compatibility wrapper for pandera.check_input.

    Uses real pandera when available; otherwise performs minimal DataFrame/type checks.
    """

    def decorator(func: F) -> F:  # type: ignore[override]
        if pa is not None and hasattr(pa, "check_input"):
            return cast("F", pa.check_input(schema, lazy=not strict)(func))  # type: ignore[attr-defined]

        # Fallback to lightweight validation
        return pa_check_io(input_schema=schema, df_arg_name=arg_name, df_arg_index=arg_index, strict=strict)(func)

    return decorator


def pa_check_output(schema: Any, *, strict: bool = True) -> Callable[[F], F]:
    """Compatibility wrapper for pandera.check_output.

    Uses real pandera when available; otherwise performs minimal DataFrame/type checks.
    """

    def decorator(func: F) -> F:  # type: ignore[override]
        if pa is not None and hasattr(pa, "check_output"):
            return cast("F", pa.check_output(schema, lazy=not strict)(func))  # type: ignore[attr-defined]

        # Fallback to lightweight validation
        return pa_check_io(output_schema=schema, strict=strict)(func)

    return decorator


def pa_check_io(
    *,
    input_schema: Any | None = None,
    output_schema: Any | None = None,
    df_arg_name: str | None = None,
    df_arg_index: int = 0,
    strict: bool = True,
) -> Callable[[F], F]:
    """Validate DataFrame input/output with pandera if available.

    - If pandera is installed and schemas are provided, validate the DataFrame
      argument identified by name or index and the returned DataFrame.
    - If pandera is not installed, performs a lightweight check that the
      argument/return is a pandas DataFrame when schemas are provided.
    """

    def decorator(func: F) -> F:  # type: ignore[override]
        def _resolve_df(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
            df_value: Any | None = None
            if df_arg_name is not None and df_arg_name in kwargs:
                df_value = kwargs.get(df_arg_name)
            elif df_arg_name is not None:
                # Try inspect for positional mapping
                sig = inspect.signature(func)
                bound = sig.bind_partial(*args, **kwargs)
                if df_arg_name in bound.arguments:
                    df_value = bound.arguments[df_arg_name]
            elif len(args) > df_arg_index:
                df_value = args[df_arg_index]
            return df_value

        def _validate_input(df_value: Any) -> None:
            if input_schema is None:
                return
            if pa is not None and hasattr(input_schema, "validate"):
                try:
                    input_schema.validate(df_value, lazy=not strict)
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
                return result
            if pa is not None and hasattr(output_schema, "validate"):
                try:
                    output_schema.validate(result, lazy=not strict)
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
            df_value = _resolve_df(args, kwargs)
            if input_schema is not None:
                _validate_input(df_value)
            result = await func(*args, **kwargs)  # type: ignore[misc]
            if output_schema is not None:
                _validate_output(result)
            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            df_value = _resolve_df(args, kwargs)
            if input_schema is not None:
                _validate_input(df_value)
            result = func(*args, **kwargs)
            if output_schema is not None:
                _validate_output(result)
            return result

        if inspect.iscoroutinefunction(func):
            return cast("F", async_wrapper)
        return cast("F", sync_wrapper)

    return decorator


# --------------------------
# Vectorization guarantees
# --------------------------

def enforce_ndarray(
    *,
    arg_index: int = 0,
    forbid_lists: bool = False,
    require_vector: bool = False,
) -> Callable[[F], F]:
    """Coerce the selected argument to numpy.ndarray and optionally forbid lists.

    - forbid_lists=True raises if a list is provided
    - require_vector=True requires at least 1-D input (no pure scalars)
    """

    def decorator(func: F) -> F:  # type: ignore[override]
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            args_list = list(args)
            try:
                value = args_list[arg_index]
            except IndexError as exc:
                raise VectorizationError(
                    f"Argument index {arg_index} out of range for {func.__name__}",
                ) from exc

            if forbid_lists and isinstance(value, list):
                raise VectorizationError(
                    "Python lists are forbidden for this function; use numpy arrays",
                    context={"function": func.__name__},
                )

            coerced = np.asarray(value)
            if require_vector and coerced.ndim == 0:
                raise VectorizationError(
                    "Scalar inputs are not allowed; provide vectorized data",
                    context={"function": func.__name__},
                )

            args_list[arg_index] = coerced
            return func(*tuple(args_list), **kwargs)

        return cast("F", wrapper)

    return decorator


def auto_vectorize(*, otypes: list[type] | None = None) -> Callable[[F], F]:
    """Wrap a scalar function so that it transparently handles numpy arrays.

    - If the first positional argument is an ndarray with ndim>=1, applies
      numpy.vectorize to broadcast the scalar logic across elements.
    - Otherwise, calls the function directly.
    """

    def decorator(func: F) -> F:  # type: ignore[override]
        @functools.wraps(func)
        def wrapper(first: Any, *args: Any, **kwargs: Any):
            array = np.asarray(first)
            if array.ndim == 0:
                return func(cast(Any, array.item()), *args, **kwargs)
            vec = np.vectorize(lambda v: func(v, *args, **kwargs), otypes=otypes)
            return vec(array)

        return cast("F", wrapper)

    return decorator


# --------------------------
# NaN/Inf/null guards
# --------------------------

def guard_array_nan_inf(
    *,
    mode: str = "raise",  # "raise" | "warn" | "coerce"
    coerce_value: float = 0.0,
    arg_indices: Iterable[int] = (0,),
) -> Callable[[F], F]:
    """Pre-check numpy arrays or pandas objects for NaN/Inf before executing.

    mode:
      - "raise": raise DataValidationError on detection
      - "warn": log a warning and continue
      - "coerce": replace NaN/Inf with coerce_value before calling func
    """

    def decorator(func: F) -> F:  # type: ignore[override]
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            args_list = list(args)
            for index in arg_indices:
                if index >= len(args_list):
                    continue
                value = args_list[index]
                if isinstance(value, pd.Series) or isinstance(value, pd.DataFrame):
                    data = value.to_numpy()
                else:
                    data = np.asarray(value)

                has_nan = np.isnan(data).any()
                has_inf = np.isinf(data).any()
                if has_nan or has_inf:
                    msg = (
                        f"Detected {'NaN' if has_nan else ''}{' and ' if has_nan and has_inf else ''}"
                        f"{'Inf' if has_inf else ''} in argument {index} for {func.__name__}"
                    )
                    if mode == "raise":
                        raise DataValidationError(msg, context={"function": func.__name__})
                    if mode == "warn":
                        logger.warning(msg)
                    if mode == "coerce":
                        coerced = np.asarray(value, dtype=float)
                        coerced = np.nan_to_num(coerced, nan=coerce_value, posinf=coerce_value, neginf=coerce_value)
                        args_list[index] = coerced

            return func(*tuple(args_list), **kwargs)

        return cast("F", wrapper)

    return decorator


def guard_dataframe_nulls(
    *,
    columns: list[str] | None = None,
    mode: str = "raise",  # "raise" | "warn" | "fill"
    fill_value: float | int | str | None = 0,
    arg_index: int = 0,
) -> Callable[[F], F]:
    """Check a pandas DataFrame argument for nulls/NaN/Inf.

    arg_index selects which positional argument is the DataFrame (0 for functions where df is first, 1 for instance methods).
    If columns is provided, restrict checks to those columns.
    """

    def decorator(func: F) -> F:  # type: ignore[override]
        def _check(df: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(df, pd.DataFrame):
                raise DataValidationError(
                    "Target argument must be a pandas DataFrame",
                    context={"function": func.__name__},
                )
            selected = df if columns is None else df[columns]
            num_nan = int(selected.isna().sum().sum())
            num_inf = int(np.isinf(selected.to_numpy()).sum())
            if num_nan or num_inf:
                msg = (
                    f"DataFrame has {num_nan} NaN and {num_inf} Inf values in {func.__name__}"
                )
                if mode == "raise":
                    raise DataValidationError(msg, context={"function": func.__name__})
                if mode == "warn":
                    logger.warning(msg)
                if mode == "fill":
                    df = df.copy()
                    df[selected.columns] = (
                        selected.replace([np.inf, -np.inf], fill_value).fillna(fill_value)
                    )
            return df

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
            args_list = list(args)
            if len(args_list) > arg_index:
                args_list[arg_index] = _check(args_list[arg_index])
            return await func(*tuple(args_list), **kwargs)  # type: ignore[misc]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            args_list = list(args)
            if len(args_list) > arg_index:
                args_list[arg_index] = _check(args_list[arg_index])
            return func(*tuple(args_list), **kwargs)

        if inspect.iscoroutinefunction(func):
            return cast("F", async_wrapper)
        return cast("F", sync_wrapper)

    return decorator


# --------------------------
# Error normalization
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


def normalize_errors(
    *,
    map_exceptions: dict[type[BaseException], type[DomainError]] | None = None,
    default_error: type[DomainError] = DomainError,
    reraise: bool = False,
) -> Callable[[F], F]:
    """Normalize heterogeneous exceptions into domain-specific errors.

    - map_exceptions augments the built-in mapping
    - if reraise=True, re-raises the normalized DomainError after logging
    - otherwise returns None and logs; for functions that must return a value,
      consider using together with default returns in your wrapper logic.
    """

    exception_map = dict(_EXCEPTION_MAP)
    if map_exceptions:
        exception_map.update(map_exceptions)

    def decorator(func: F) -> F:  # type: ignore[override]
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
            try:
                return await func(*args, **kwargs)  # type: ignore[misc]
            except tuple(exception_map.keys()) as exc:  # type: ignore[arg-type]
                domain_exc_type = default_error
                for base_exc, mapped in exception_map.items():
                    if isinstance(exc, base_exc):
                        domain_exc_type = mapped
                        break
                norm_exc = domain_exc_type(
                    f"{func.__name__} failed: {exc}",
                    context={"function": func.__name__},
                )
                logger.exception("Normalized error", extra={"correlation_id": get_correlation_id()})
                if reraise:
                    raise norm_exc from exc
                return None

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            try:
                return func(*args, **kwargs)
            except tuple(exception_map.keys()) as exc:  # type: ignore[arg-type]
                domain_exc_type = default_error
                for base_exc, mapped in exception_map.items():
                    if isinstance(exc, base_exc):
                        domain_exc_type = mapped
                        break
                norm_exc = domain_exc_type(
                    f"{func.__name__} failed: {exc}",
                    context={"function": func.__name__},
                )
                logger.exception("Normalized error", extra={"correlation_id": get_correlation_id()})
                if reraise:
                    raise norm_exc from exc
                return None

        if inspect.iscoroutinefunction(func):
            return cast("F", async_wrapper)
        return cast("F", sync_wrapper)

    return decorator


# --------------------------
# Logging/tracing/audit
# --------------------------

_SENSITIVE_KEYS = {"password", "secret", "token", "api_key", "apikey", "access_key", "private_key"}


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


def with_tracing_span(
    span_name: str | None = None,
    *,
    log_args: bool = False,
    log_result_len_only: bool = True,
) -> Callable[[F], F]:
    """Add correlation-aware entry/exit logs around a function call.

    - Ensures a correlation ID is present
    - Optionally logs sanitized args/kwargs (avoid for heavy data)
    - Logs result size instead of full content by default
    """

    def decorator(func: F) -> F:  # type: ignore[override]
        resolved_span = span_name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
            cid = ensure_correlation_id()
            if log_args:
                safe_args = _sanitize(args)
                safe_kwargs = _sanitize(kwargs)
                logger.info(
                    f"➡️ {resolved_span} start",
                    extra={"correlation_id": cid, "args": safe_args, "kwargs": safe_kwargs},
                )
            else:
                logger.info(
                    f"➡️ {resolved_span} start",
                    extra={"correlation_id": cid},
                )

            result = await func(*args, **kwargs)  # type: ignore[misc]

            if log_result_len_only:
                try:
                    length = None
                    if hasattr(result, "__len__"):
                        length = len(cast(Any, result))
                    logger.info(
                        f"✅ {resolved_span} done",
                        extra={"correlation_id": cid, "result_len": length},
                    )
                except Exception:
                    logger.info(
                        f"✅ {resolved_span} done",
                        extra={"correlation_id": cid},
                    )
            else:
                logger.info(
                    f"✅ {resolved_span} done",
                    extra={"correlation_id": cid, "result": _sanitize(result)},
                )

            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            cid = ensure_correlation_id()
            if log_args:
                safe_args = _sanitize(args)
                safe_kwargs = _sanitize(kwargs)
                logger.info(
                    f"➡️ {resolved_span} start",
                    extra={"correlation_id": cid, "args": safe_args, "kwargs": safe_kwargs},
                )
            else:
                logger.info(
                    f"➡️ {resolved_span} start",
                    extra={"correlation_id": cid},
                )

            result = func(*args, **kwargs)

            if log_result_len_only:
                try:
                    length = None
                    if hasattr(result, "__len__"):
                        length = len(cast(Any, result))
                    logger.info(
                        f"✅ {resolved_span} done",
                        extra={"correlation_id": cid, "result_len": length},
                    )
                except Exception:
                    logger.info(
                        f"✅ {resolved_span} done",
                        extra={"correlation_id": cid},
                    )
            else:
                logger.info(
                    f"✅ {resolved_span} done",
                    extra={"correlation_id": cid, "result": _sanitize(result)},
                )

            return result

        if inspect.iscoroutinefunction(func):
            return cast("F", async_wrapper)
        return cast("F", sync_wrapper)

    return decorator


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