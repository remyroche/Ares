"""
Common Operations Utility Module

This module provides commonly used operations that were identified as undefined
in the codebase analysis. It serves as a central location for these utilities.
"""

import datetime
import pandas as pd
import numpy as np
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from collections import defaultdict, Counter, deque
import logging
from copy import copy, deepcopy
import argparse


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


def parse_datetime(date_string: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime.datetime:
    """Parse string to datetime."""
    return datetime.datetime.strptime(date_string, fmt)


# DataFrame operations
def create_empty_dataframe(columns: List[str]) -> pd.DataFrame:
    """Create an empty DataFrame with specified columns."""
    return pd.DataFrame(columns=columns)


def safe_fillna(df: pd.DataFrame, value: Any = 0) -> pd.DataFrame:
    """Safely fill NaN values in a DataFrame."""
    return df.fillna(value)


def safe_rolling(df: pd.DataFrame, window: int, min_periods: int = 1) -> pd.core.window.Rolling:
    """Create a rolling window object safely."""
    return df.rolling(window=window, min_periods=min_periods)


# Numeric operations
def safe_mean(values: Union[List, np.ndarray, pd.Series]) -> float:
    """Calculate mean safely, handling empty inputs."""
    if isinstance(values, (list, tuple)):
        values = np.array(values)
    if len(values) == 0:
        return np.nan
    return np.nanmean(values)


def safe_std(values: Union[List, np.ndarray, pd.Series]) -> float:
    """Calculate standard deviation safely."""
    if isinstance(values, (list, tuple)):
        values = np.array(values)
    if len(values) == 0:
        return np.nan
    return np.nanstd(values)


# File operations
def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_file_exists(path: Union[str, Path]) -> bool:
    """Check if a file exists safely."""
    try:
        return Path(path).exists()
    except Exception:
        return False


def safe_json_dump(data: Any, file_path: Union[str, Path], **kwargs) -> None:
    """Safely dump data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, **kwargs)


def safe_json_load(file_path: Union[str, Path]) -> Any:
    """Safely load data from JSON file."""
    with open(file_path, 'r') as f:
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
def safe_append(lst: List[Any], item: Any) -> List[Any]:
    """Safely append to a list."""
    if lst is None:
        lst = []
    lst.append(item)
    return lst


def safe_extend(lst: List[Any], items: List[Any]) -> List[Any]:
    """Safely extend a list."""
    if lst is None:
        lst = []
    lst.extend(items)
    return lst


def safe_dict_get(d: Dict[Any, Any], key: Any, default: Any = None) -> Any:
    """Safely get value from dictionary."""
    if d is None:
        return default
    return d.get(key, default)


def safe_dict_items(d: Dict[Any, Any]) -> List[tuple]:
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


def safe_join(separator: str, items: List[Any]) -> str:
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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Argument parsing utilities
def create_argument_parser(description: str) -> argparse.ArgumentParser:
    """Create an argument parser."""
    return argparse.ArgumentParser(description=description)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to parser."""
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path')


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
    if hasattr(trial, 'suggest_float'):
        return trial.suggest_float(name, low, high)
    else:
        # Fallback for non-Optuna contexts
        import random
        return random.uniform(low, high)


def suggest_int_uniform(trial: Any, name: str, low: int, high: int) -> int:
    """Wrapper for Optuna's suggest_int."""
    if hasattr(trial, 'suggest_int'):
        return trial.suggest_int(name, low, high)
    else:
        # Fallback for non-Optuna contexts
        import random
        return random.randint(low, high)


# Validation utilities
def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
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
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    return df


# Export commonly used items for easier imports
__all__ = [
    'get_current_datetime', 'get_today', 'format_datetime', 'parse_datetime',
    'create_empty_dataframe', 'safe_fillna', 'safe_rolling',
    'safe_mean', 'safe_std',
    'ensure_directory', 'safe_file_exists', 'safe_json_dump', 'safe_json_load',
    'safe_sleep', 'safe_gather', 'create_async_task',
    'safe_append', 'safe_extend', 'safe_dict_get', 'safe_dict_items',
    'safe_lower', 'safe_upper', 'safe_join',
    'get_logger', 'setup_basic_logging',
    'create_argument_parser', 'add_common_arguments',
    'safe_exception_handler',
    'safe_float', 'safe_int',
    'suggest_float_uniform', 'suggest_int_uniform',
    'validate_dataframe', 'validate_numeric_range',
    'optimize_dataframe_dtypes'
]