"""Centralized, typed decorators used across the project.

Exposes minimal, safe implementations for commonly referenced decorators:
- validate_data_quality
- validate_wavelet_data_quality
- guard_dataframe_nulls
- with_tracing_span
"""
from __future__ import annotations

import time
from typing import Any, Callable, TypeVar, cast

from .logger import get_logger

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

F = TypeVar("F", bound=Callable[..., Any])


def _is_dataframe(obj: Any) -> bool:
    return pd is not None and obj is not None and obj.__class__.__name__ in {"DataFrame", "Series"}


def with_tracing_span(func: F) -> F:
    logger = get_logger("Tracing")

    def _format_duration(seconds: float) -> str:
        ms = int(seconds * 1000)
        if ms < 1000:
            return f"{ms}ms"
        if seconds < 60:
            return f"{seconds:.2f}s"
        minutes, sec = divmod(int(seconds), 60)
        return f"{minutes}m {sec}s"

    def _wrap(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        logger.info(f"▶️  {func.__name__} start")
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.info(f"⏹️  {func.__name__} done in {_format_duration(elapsed)}")

    _wrap.__name__ = func.__name__
    _wrap.__doc__ = func.__doc__
    return cast(F, _wrap)


def _validate_df(logger_name: str, df: Any, *, context: str) -> None:
    logger = get_logger(logger_name)
    if pd is None or not _is_dataframe(df):
        return
    if getattr(df, "empty", False):
        logger.warning(f"⚠️ [{context}] DataFrame is empty")
        return
    try:
        null_counts = df.isnull().sum()
        all_null_cols = [c for c, v in null_counts.items() if int(v) >= len(df)]
        if all_null_cols:
            logger.warning(
                f"⚠️ [{context}] Columns with all NaNs: {sorted(all_null_cols)[:10]}"
            )
    except Exception:
        logger.debug(f"[{context}] Skipped null checks due to error")


def validate_data_quality(func: F) -> F:
    logger = get_logger("DataQuality")

    def _wrap(*args: Any, **kwargs: Any) -> Any:
        if args:
            _validate_df("DataQuality", args[0], context=f"{func.__name__}:input")
        result = func(*args, **kwargs)
        _validate_df("DataQuality", result, context=f"{func.__name__}:output")
        logger.debug(f"{func.__name__} quality checks complete")
        return result

    _wrap.__name__ = func.__name__
    _wrap.__doc__ = func.__doc__
    return cast(F, _wrap)


def validate_wavelet_data_quality(func: F) -> F:
    return validate_data_quality(func)


def guard_dataframe_nulls(func: F) -> F:
    def _wrap(*args: Any, **kwargs: Any) -> Any:
        if args:
            _validate_df("NullGuard", args[0], context=f"{func.__name__}:input")
        result = func(*args, **kwargs)
        _validate_df("NullGuard", result, context=f"{func.__name__}:output")
        return result

    _wrap.__name__ = func.__name__
    _wrap.__doc__ = func.__doc__
    return cast(F, _wrap)
