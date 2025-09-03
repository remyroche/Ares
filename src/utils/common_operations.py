from __future__ import annotations

"""
Common Operations Utility Module

This module provides commonly used operations that were identified as undefined
in the codebase analysis. It serves as a central location for these utilities.
"""

import argparse
import asyncio
import datetime
import glob
import hashlib
import json
import logging
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import (
    A,
    Callableny,
)

import numpy as np
import pandas as pd


# DateTime utilities
def get_current_datetime() -> datetime.datetime:
    """Get current datetime."""
    return datetime.datetime.now()


def get_today() -> datetime.date:
    """Get today's date."""
    return datetime.date.today()


def format_datetime(dt: datetime.datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime to string."""
    return dt.strftime(fmt)


def parse_datetime(
    date_string: str, fmt: str = "%Y-%m-%d %H:%M:%S"
) -> datetime.datetime:
    """Parse string to datetime."""
    return datetime.datetime.strptime(date_string, fmt)


# DataFrame operations
def create_empty_dataframe(columns: list[str]) -> pd.DataFrame:
    """Create an empty DataFrame with specified columns."""
    return pd.DataFrame(columns=columns)


def safe_fillna(df: pd.DataFrame, value: Any = 0) -> pd.DataFrame:
    """Safely fill NaN values in a DataFrame."""
    return df.fillna(value)


def safe_rolling(
    df: pd.DataFrame, window: int, min_periods: int = 1
) -> pd.core.window.Rolling:
    """Create a rolling window object safely."""
    return df.rolling(window=window, min_periods=min_periods)


# Numeric operations
def safe_mean(values: list | np.ndarray | pd.Series) -> float:
    """Calculate mean safely, handling empty inputs."""
    if isinstance(values, list | tuple):
        values = np.array(values)
    if len(values) == 0:
        return np.nan
    return np.nanmean(values)


def safe_std(values: list | np.ndarray | pd.Series) -> float:
    """Calculate standard deviation safely."""
    if isinstance(values, list | tuple):
        values = np.array(values)
    if len(values) == 0:
        return np.nan
    return np.nanstd(values)


# File operations
def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_file_exists(path: str | Path) -> bool:
    """Check if a file exists safely."""
    try:
        return Path(path).exists()
    except Exception:
        return False


def safe_json_dump(data: Any, file_path: str | Path, **kwargs) -> None:
    """Safely dump data to JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, **kwargs)


def safe_json_load(file_path: str | Path) -> Any:
    """Safely load data from JSON file."""
    with open(file_path) as f:
        return json.load(f)


# Async utilities
async def safe_sleep(seconds: float) -> None:
    """Async sleep wrapper."""
    await asyncio.sleep(seconds)


async def safe_gather(*coroutines, return_exceptions: bool = True):
    """Safely gather multiple coroutines."""
    return await asyncio.gather(*coroutines, return_exceptions=return_exceptions)


def create_async_task(coroutine) -> asyncio.Task:
    """Create an async task safely."""
    loop = asyncio.get_event_loop()
    return loop.create_task(coroutine)


# Collection utilities
def safe_append(lst: list[Any], item: Any) -> list[Any]:
    """Safely append to a list."""
    if lst is None:
        lst = []
    lst.append(item)
    return lst


def safe_extend(lst: list[Any], items: list[Any]) -> list[Any]:
    """Safely extend a list."""
    if lst is None:
        lst = []
    lst.extend(items)
    return lst


def safe_dict_get(d: dict[Any, Any], key: Any, default: Any = None) -> Any:
    """Safely get value from dictionary."""
    if d is None:
        return default
    return d.get(key, default)


def safe_dict_items(d: dict[Any, Any]) -> list[tuple]:
    """Safely get items from dictionary."""
    if d is None:
        return []
    return list(d.items())


# String operations
def safe_lower(s: str) -> str:
    """Safely convert string to lowercase."""
    if s is None:
        return ""
    return str(s).lower()


def safe_upper(s: str) -> str:
    """Safely convert string to uppercase."""
    if s is None:
        return ""
    return str(s).upper()


def safe_join(separator: str, items: list[Any]) -> str:
    """Safely join items into a string."""
    if items is None:
        return ""
    return separator.join(str(item) for item in items)


# Logging utilities
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def setup_basic_logging(level: int = logging.INFO) -> None:
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


# Argument parsing utilities
def create_argument_parser(description: str) -> argparse.ArgumentParser:
    """Create an argument parser."""
    return argparse.ArgumentParser(description=description)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to parser."""
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="Configuration file path"
    )
    parser.add_argument("--output", "-o", type=str, help="Output file path")


# Exception handling utilities
def safe_exception_handler(func: Callable) -> Callable:
    """Decorator for safe exception handling."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = get_logger(func.__module__)
            logger.exception(f"Error in {func.__name__}: {e}")
            return None

    return wrapper


# Type conversion utilities
def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# Optuna-specific utilities (for hyperparameter optimization)
def suggest_float_uniform(trial: Any, name: str, low: float, high: float) -> float:
    """Wrapper for Optuna's suggest_float."""
    if hasattr(trial, "suggest_float"):
        return trial.suggest_float(name, low, high)
    # Fallback for non-Optuna contexts
    import random

    return random.uniform(low, high)


def suggest_int_uniform(trial: Any, name: str, low: int, high: int) -> int:
    """Wrapper for Optuna's suggest_int."""
    if hasattr(trial, "suggest_int"):
        return trial.suggest_int(name, low, high)
    # Fallback for non-Optuna contexts
    import random

    return random.randint(low, high)


# Validation utilities
def validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> bool:
    """Validate that a DataFrame has required columns."""
    if df is None or df.empty:
        return False
    return all(col in df.columns for col in required_columns)


def validate_numeric_range(value: float, min_val: float, max_val: float) -> bool:
    """Validate that a value is within a numeric range."""
    return min_val <= value <= max_val


# Memory optimization utilities
def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting types."""
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != "object":
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif (
                c_min > np.finfo(np.float16).min
                and c_max < np.finfo(np.float16).max
                or c_min > np.finfo(np.float32).min
                and c_max < np.finfo(np.float32).max
            ):
                df[col] = df[col].astype(np.float32)

    return df


# Parquet operations
def safe_read_parquet(
    file_path: str | Path, columns: list[str] | None = None
) -> pd.DataFrame:
    """Safely read parquet file with error handling."""
    try:
        return pd.read_parquet(file_path, columns=columns)
    except Exception as e:
        logger = get_logger(__name__)
        logger.exception(f"Failed to read parquet file {file_path}: {e}")
        return pd.DataFrame()


def safe_to_parquet(df: pd.DataFrame, file_path: str | Path, **kwargs) -> bool:
    """Safely write DataFrame to parquet with error handling."""
    try:
        df.to_parquet(file_path, **kwargs)
        return True
    except Exception as e:
        logger = get_logger(__name__)
        logger.exception(f"Failed to write parquet file {file_path}: {e}")
        return False


def list_parquet_files(directory: str | Path, recursive: bool = True) -> list[Path]:
    """List all parquet files in a directory."""
    directory = Path(directory)
    if recursive:
        return list(directory.rglob("*.parquet"))
    return list(directory.glob("*.parquet"))


# Hashing and cache operations
def generate_hash(data: str | bytes | pd.DataFrame, algorithm: str = "md5") -> str:
    """Generate hash for data with support for different types."""

    if isinstance(data, pd.DataFrame):
        data = pd.util.hash_pandas_object(data).values.tobytes()
    elif isinstance(data, str):
        data = data.encode()

    if algorithm == "md5":
        return hashlib.md5(data).hexdigest()
    if algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    msg = f"Unsupported algorithm: {algorithm}"
    raise ValueError(msg)


def generate_cache_key(prefix: str, *args, max_length: int = 16) -> str:
    """Generate a cache key from multiple inputs."""
    combined = f"{prefix}_" + "_".join(str(arg) for arg in args)
    hash_val = generate_hash(combined, "sha256")
    return hash_val[:max_length]


# Enhanced DataFrame operations
def safe_copy(df: pd.DataFrame, deep: bool = True) -> pd.DataFrame:
    """Safely copy a DataFrame with error handling."""
    try:
        return df.copy(deep=deep)
    except Exception:
        return df


def safe_deepcopy(obj: Any) -> Any:
    """Safely deep copy an object."""
    try:
        return deepcopy(obj)
    except Exception:
        return obj


# Enhanced file system operations
def safe_glob(pattern: str, recursive: bool = False) -> list[Path]:
    """Safely glob for files with error handling."""
    try:
        files = glob.glob(pattern, recursive=recursive)
        return [Path(f) for f in files]
    except Exception:
        return []


def list_files(
    directory: str | Path, pattern: str = "*", suffix: str | None = None
) -> list[Path]:
    """List files in directory with optional pattern/suffix filter."""
    directory = Path(directory)
    if not directory.exists():
        return []

    if suffix:
        return [f for f in directory.iterdir() if f.is_file() and f.suffix == suffix]

    return [f for f in directory.glob(pattern) if f.is_file()]


def get_latest_file(directory: str | Path, pattern: str = "*") -> Path | None:
    """Get the most recently modified file matching pattern."""
    files = list_files(directory, pattern)
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


# Enhanced data validation
def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: list[str],
    column_types: dict[str, type] | None = None,
) -> tuple[bool, list[str]]:
    """Validate DataFrame schema including column types."""
    errors = []

    # Check required columns
    missing = set(required_columns) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Check column types if specified
    if column_types:
        for col, expected_type in column_types.items():
            if col in df.columns:
                actual_type = df[col].dtype
                if not np.issubdtype(actual_type, expected_type):
                    errors.append(
                        f"Column {col} has type {actual_type}, expected {expected_type}"
                    )

    return len(errors) == 0, errors


def validate_data_quality(
    df: pd.DataFrame, max_nan_ratio: float = 0.1, check_duplicates: bool = True
) -> dict[str, Any]:
    """Comprehensive data quality validation."""
    quality_report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "issues": [],
    }

    # Check NaN ratio
    nan_ratios = df.isna().sum() / len(df)
    high_nan_cols = nan_ratios[nan_ratios > max_nan_ratio]
    if not high_nan_cols.empty:
        quality_report["issues"].append(
            {
                "type": "high_nan_ratio",
                "columns": high_nan_cols.to_dict(),
            }
        )

    # Check duplicates
    if check_duplicates:
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            quality_report["issues"].append(
                {
                    "type": "duplicates",
                    "count": duplicates,
                }
            )

    quality_report["is_valid"] = len(quality_report["issues"]) == 0
    return quality_report


# Time series operations
def safe_resample(
    df: pd.DataFrame, rule: str, agg_dict: dict[str, str] | None = None
) -> pd.DataFrame:
    """Safely resample time series data."""
    if not isinstance(df.index, pd.DatetimeIndex):
        msg = "DataFrame must have DatetimeIndex"
        raise ValueError(msg)

    if agg_dict is None:
        # Default aggregations for common columns
        agg_dict = {
            "close": "last",
            "open": "first",
            "high": "max",
            "low": "min",
            "volume": "sum",
        }
        # Only use columns that exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    return df.resample(rule).agg(agg_dict)


def align_dataframes(*dfs: pd.DataFrame, method: str = "inner") -> list[pd.DataFrame]:
    """Align multiple DataFrames by index."""
    if len(dfs) < 2:
        return list(dfs)

    # Find common index range
    if method == "inner":
        start = max(df.index.min() for df in dfs)
        end = min(df.index.max() for df in dfs)
        aligned = [df.loc[start:end] for df in dfs]
    else:  # outer
        aligned = list(dfs)

    return aligned


# Enhanced collection utilities
def safe_defaultdict(default_factory: Callable) -> defaultdict:
    """Create a defaultdict safely."""
    return defaultdict(default_factory)


def safe_counter(items: list[Any] | None = None) -> Counter:
    """Create a Counter safely."""
    return Counter(items or [])


def safe_deque(items: list[Any] | None = None, maxlen: int | None = None) -> deque:
    """Create a deque safely."""
    return deque(items or [], maxlen=maxlen)


# Progress and timing utilities
def timed_operation(operation_name: str):
    """Decorator to time operations."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            logger = get_logger(func.__module__)
            logger.info(f"Starting {operation_name}...")
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                logger.info(f"Completed {operation_name} in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.exception(f"Failed {operation_name} after {elapsed:.2f}s: {e}")
                raise

        return wrapper

    return decorator


def format_bytes(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


# Batch processing utilities
def chunked_iterable(iterable: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split an iterable into chunks."""
    chunks = []
    for i in range(0, len(iterable), chunk_size):
        chunks.append(iterable[i : i + chunk_size])
    return chunks


def parallel_map(
    func: Callable, items: list[Any], max_workers: int | None = None
) -> list[Any]:
    """Apply function to items in parallel."""

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))


# MLflow integration helpers
def safe_log_metric(key: str, value: float, step: int | None = None) -> None:
    """Safely log metric to MLflow if available."""
    try:
        import mlflow

        if mlflow.active_run():
            mlflow.log_metric(key, value, step)
    except Exception:
        pass


def safe_log_params(params: dict[str, Any]) -> None:
    """Safely log parameters to MLflow if available."""
    try:
        import mlflow

        if mlflow.active_run():
            mlflow.log_params(params)
    except Exception:
        pass


def safe_log_artifact(file_path: str | Path) -> None:
    """Safely log artifact to MLflow if available."""
    try:
        import mlflow

        if mlflow.active_run():
            mlflow.log_artifact(str(file_path))
    except Exception:
        pass


# Export commonly used items for easier imports
__all__ = [
    # DateTime utilities
    "get_current_datetime",
    "get_today",
    "format_datetime",
    "parse_datetime",
    # DataFrame operations
    "create_empty_dataframe",
    "safe_fillna",
    "safe_rolling",
    "safe_copy",
    "safe_deepcopy",
    "safe_resample",
    "align_dataframes",
    # Numeric operations
    "safe_mean",
    "safe_std",
    # File operations
    "ensure_directory",
    "safe_file_exists",
    "safe_json_dump",
    "safe_json_load",
    "safe_glob",
    "list_files",
    "get_latest_file",
    # Parquet operations
    "safe_read_parquet",
    "safe_to_parquet",
    "list_parquet_files",
    # Hashing and cache operations
    "generate_hash",
    "generate_cache_key",
    # Async utilities
    "safe_sleep",
    "safe_gather",
    "create_async_task",
    # Collection utilities
    "safe_append",
    "safe_extend",
    "safe_dict_get",
    "safe_dict_items",
    "safe_defaultdict",
    "safe_counter",
    "safe_deque",
    # String operations
    "safe_lower",
    "safe_upper",
    "safe_join",
    # Logging utilities
    "get_logger",
    "setup_basic_logging",
    # Argument parsing utilities
    "create_argument_parser",
    "add_common_arguments",
    # Exception handling utilities
    "safe_exception_handler",
    # Type conversion utilities
    "safe_float",
    "safe_int",
    # Optuna utilities
    "suggest_float_uniform",
    "suggest_int_uniform",
    # Validation utilities
    "validate_dataframe",
    "validate_numeric_range",
    "validate_dataframe_schema",
    "validate_data_quality",
    # Memory optimization utilities
    "optimize_dataframe_dtypes",
    # Progress and timing utilities
    "timed_operation",
    "format_bytes",
    # Batch processing utilities
    "chunked_iterable",
    "parallel_map",
    # MLflow integration helpers
    "safe_log_metric",
    "safe_log_params",
    "safe_log_artifact",
]
