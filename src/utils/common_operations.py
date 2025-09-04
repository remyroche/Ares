from __future__ import annotations
from typing import Dict, List, Optional, Union, Any, Tuple
'\nCommon Operations Utility Module with Comprehensive Error Handling\n\nThis module provides commonly used operations that were identified as undefined\nin the codebase analysis. It serves as a central location for these utilities\nwith enhanced error handling and emoji-based logging.\n'
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
import numpy as np
import pandas as pd

# Import enhanced logging functions
try:
    from .logger import log_error_with_context, log_performance_metrics, log_data_quality_check
    from .warning_symbols import error, warning, info, success
except ImportError:
    # Fallback if imports fail
    def log_error_with_context(logger, error, context=None, operation="", recovery_attempted=False):
        logger.error(f"Error in {operation}: {error}")
    
    def log_performance_metrics(logger, operation_name, duration, memory_usage=None, additional_metrics=None):
        logger.info(f"Performance | {operation_name} | Duration: {duration:.3f}s")
    
    def log_data_quality_check(logger, check_name, status, details="", stats=None):
        logger.info(f"Data Quality Check | {check_name} | {status.upper()}")
    
    def error(msg): return f"❌ {msg}"
    def warning(msg): return f"⚠️ {msg}"
    def info(msg): return f"ℹ️ {msg}"
    def success(msg): return f"✅ {msg}"

logger = logging.getLogger(__name__)

def get_current_datetime() -> datetime.datetime:
    """Get current datetime with comprehensive error handling."""
    try:
        logger.debug("🕐 Getting current datetime")
        result = datetime.datetime.now()
        logger.debug(f"✅ Current datetime: {result}")
        return result
    except Exception as e:
        logger.error(f"❌ Failed to get current datetime: {e}")
        log_error_with_context(
            logger, e,
            operation="get_current_datetime"
        )
        # Return a fallback datetime
        return datetime.datetime(1970, 1, 1)

def get_today() -> datetime.date:
    """Get today's date with comprehensive error handling."""
    try:
        logger.debug("📅 Getting today's date")
        result = datetime.date.today()
        logger.debug(f"✅ Today's date: {result}")
        return result
    except Exception as e:
        logger.error(f"❌ Failed to get today's date: {e}")
        log_error_with_context(
            logger, e,
            operation="get_today"
        )
        # Return a fallback date
        return datetime.date(1970, 1, 1)

def format_datetime(dt: datetime.datetime, fmt: str='%Y-%m-%d %H:%M:%S') -> str:
    """Format datetime to string with comprehensive error handling."""
    try:
        logger.debug(f"🔧 Formatting datetime with format: {fmt}")
        
        if not isinstance(dt, datetime.datetime):
            raise ValueError(f"Expected datetime.datetime, got {type(dt)}")
        
        if not fmt or not isinstance(fmt, str):
            raise ValueError(f"Invalid format string: {fmt}")
        
        result = dt.strftime(fmt)
        logger.debug(f"✅ Formatted datetime: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to format datetime: {e}")
        log_error_with_context(
            logger, e,
            context={"datetime": str(dt), "format": fmt},
            operation="format_datetime"
        )
        # Return a fallback formatted string
        return "1970-01-01 00:00:00"

def parse_datetime(date_string: str, fmt: str='%Y-%m-%d %H:%M:%S') -> datetime.datetime:
    """Parse string to datetime with comprehensive error handling."""
    try:
        logger.debug(f"🔍 Parsing datetime string: {date_string} with format: {fmt}")
        
        if not date_string or not isinstance(date_string, str):
            raise ValueError(f"Invalid date string: {date_string}")
        
        if not fmt or not isinstance(fmt, str):
            raise ValueError(f"Invalid format string: {fmt}")
        
        result = datetime.datetime.strptime(date_string, fmt)
        logger.debug(f"✅ Parsed datetime: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to parse datetime: {e}")
        log_error_with_context(
            logger, e,
            context={"date_string": date_string, "format": fmt},
            operation="parse_datetime"
        )
        # Return a fallback datetime
        return datetime.datetime(1970, 1, 1)

def create_empty_dataframe(columns: list[str]) -> pd.DataFrame:
    """Create an empty DataFrame with specified columns and comprehensive error handling."""
    try:
        logger.debug(f"📊 Creating empty DataFrame with {len(columns)} columns")
        
        if not isinstance(columns, (list, tuple)):
            raise ValueError(f"Columns must be a list or tuple, got {type(columns)}")
        
        if not columns:
            logger.warning("⚠️ Creating DataFrame with no columns")
        
        # Validate column names
        for i, col in enumerate(columns):
            if not isinstance(col, str):
                logger.warning(f"⚠️ Column {i} is not a string: {col}")
        
        result = pd.DataFrame(columns=columns)
        # Only log if there are issues or debugging is needed
        if not columns:
            logger.warning("⚠️ Created DataFrame with no columns")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to create empty DataFrame: {e}")
        log_error_with_context(
            logger, e,
            context={"columns": columns, "column_count": len(columns) if columns else 0},
            operation="create_empty_dataframe"
        )
        # Return a fallback empty DataFrame
        return pd.DataFrame()

def safe_fillna(df: pd.DataFrame, value: Any=0) -> pd.DataFrame:
    """Safely fill NaN values in a DataFrame with comprehensive error handling."""
    try:
        logger.debug(f"🔧 Filling NaN values in DataFrame with value: {value}")
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected pandas.DataFrame, got {type(df)}")
        
        if df.empty:
            logger.warning("⚠️ DataFrame is empty, returning as-is")
            return df
        
        # Count NaN values before filling
        nan_count = df.isnull().sum().sum()
        logger.debug(f"📊 Found {nan_count} NaN values to fill")
        
        result = df.fillna(value)
        
        # Verify the operation
        remaining_nans = result.isnull().sum().sum()
        if remaining_nans > 0:
            logger.warning(f"⚠️ {remaining_nans} NaN values remain after filling")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to fill NaN values: {e}")
        log_error_with_context(
            logger, e,
            context={"value": str(value), "dataframe_shape": df.shape if hasattr(df, 'shape') else 'unknown'},
            operation="safe_fillna"
        )
        # Return original DataFrame as fallback
        return df

def safe_rolling(df: pd.DataFrame, window: int, min_periods: int=1) -> pd.core.window.Rolling:
    """Create a rolling window object safely with comprehensive error handling."""
    try:
        logger.debug(f"🔄 Creating rolling window with window={window}, min_periods={min_periods}")
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected pandas.DataFrame, got {type(df)}")
        
        if df.empty:
            raise ValueError("Cannot create rolling window on empty DataFrame")
        
        if window <= 0:
            raise ValueError(f"Window size must be positive, got {window}")
        
        if min_periods < 0:
            raise ValueError(f"min_periods must be non-negative, got {min_periods}")
        
        if window > len(df):
            logger.warning(f"⚠️ Window size ({window}) is larger than DataFrame length ({len(df)})")
        
        result = df.rolling(window=window, min_periods=min_periods)
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to create rolling window: {e}")
        log_error_with_context(
            logger, e,
            context={"window": window, "min_periods": min_periods, "dataframe_shape": df.shape if hasattr(df, 'shape') else 'unknown'},
            operation="safe_rolling"
        )
        raise

def safe_mean(values: list | np.ndarray | pd.Series) -> float:
    """Calculate mean safely, handling empty inputs with comprehensive error handling."""
    try:
        logger.debug(f"📊 Calculating mean for {type(values).__name__}")
        
        if values is None:
            raise ValueError("Values cannot be None")
        
        # Convert to numpy array if needed
        if isinstance(values, (list, tuple)):
            if not values:
                logger.warning("⚠️ Empty list provided, returning 0.0")
                return 0.0
            values = np.array(values)
        elif isinstance(values, pd.Series):
            if values.empty:
                logger.warning("⚠️ Empty Series provided, returning 0.0")
                return 0.0
            values = values.values
        
        if not isinstance(values, np.ndarray):
            raise ValueError(f"Unsupported type for mean calculation: {type(values)}")
        
        if values.size == 0:
            logger.warning("⚠️ Empty array provided, returning 0.0")
            return 0.0
        
        # Check for all NaN values
        if np.all(np.isnan(values)):
            logger.warning("⚠️ All values are NaN, returning 0.0")
            return 0.0
        
        result = np.nanmean(values)
        logger.debug(f"✅ Calculated mean: {result}")
        return float(result)
        
    except Exception as e:
        logger.error(f"❌ Failed to calculate mean: {e}")
        log_error_with_context(
            logger, e,
            context={"values_type": type(values).__name__, "values_length": len(values) if hasattr(values, '__len__') else 'unknown'},
            operation="safe_mean"
        )
        # Return 0.0 as fallback
        return 0.0

def safe_std(values: list | np.ndarray | pd.Series) -> float:
    """Calculate standard deviation safely with comprehensive error handling."""
    try:
        logger.debug(f"📊 Calculating standard deviation for {type(values).__name__}")
        
        if values is None:
            raise ValueError("Values cannot be None")
        
        # Convert to numpy array if needed
        if isinstance(values, (list, tuple)):
            if not values:
                logger.warning("⚠️ Empty list provided, returning 0.0")
                return 0.0
            values = np.array(values)
        elif isinstance(values, pd.Series):
            if values.empty:
                logger.warning("⚠️ Empty Series provided, returning 0.0")
                return 0.0
            values = values.values
        
        if not isinstance(values, np.ndarray):
            raise ValueError(f"Unsupported type for std calculation: {type(values)}")
        
        if values.size == 0:
            logger.warning("⚠️ Empty array provided, returning 0.0")
            return 0.0
        
        # Check for all NaN values
        if np.all(np.isnan(values)):
            logger.warning("⚠️ All values are NaN, returning 0.0")
            return 0.0
        
        result = np.nanstd(values)
        logger.debug(f"✅ Calculated standard deviation: {result}")
        return float(result)
        
    except Exception as e:
        logger.error(f"❌ Failed to calculate standard deviation: {e}")
        log_error_with_context(
            logger, e,
            context={"values_type": type(values).__name__, "values_length": len(values) if hasattr(values, '__len__') else 'unknown'},
            operation="safe_std"
        )
        # Return 0.0 as fallback
        return 0.0

def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary with comprehensive error handling."""
    try:
        logger.debug(f"📁 Ensuring directory exists: {path}")
        
        if not path:
            raise ValueError("Path cannot be empty")
        
        path_obj = Path(path)
        
        # Check if it already exists
        if path_obj.exists():
            if path_obj.is_dir():
                logger.debug(f"✅ Directory already exists: {path_obj}")
                return path_obj
            else:
                raise ValueError(f"Path exists but is not a directory: {path_obj}")
        
        # Create the directory
        path_obj.mkdir(parents=True, exist_ok=True)
        # Only log if there are issues - directory creation is normal
        return path_obj
        
    except Exception as e:
        logger.error(f"❌ Failed to ensure directory exists: {e}")
        log_error_with_context(
            logger, e,
            context={"path": str(path)},
            operation="ensure_directory"
        )
        raise

def safe_file_exists(path: str | Path) -> bool:
    """Check if a file exists safely with comprehensive error handling."""
    try:
        logger.debug(f"🔍 Checking if file exists: {path}")
        
        if not path:
            logger.warning("⚠️ Empty path provided")
            return False
        
        path_obj = Path(path)
        exists = path_obj.exists()
        
        if exists:
            if path_obj.is_file():
                logger.debug(f"✅ File exists: {path_obj}")
            else:
                logger.debug(f"📁 Path exists but is not a file: {path_obj}")
        else:
            logger.debug(f"❌ File does not exist: {path_obj}")
        
        return exists
        
    except Exception as e:
        logger.error(f"❌ Error checking file existence: {e}")
        log_error_with_context(
            logger, e,
            context={"path": str(path)},
            operation="safe_file_exists"
        )
        return False

def safe_json_dump(data: Any, file_path: str | Path, **kwargs) -> None:
    """Safely dump data to JSON file with comprehensive error handling."""
    try:
        logger.debug(f"💾 Saving data to JSON file: {file_path}")
        
        if not file_path:
            raise ValueError("File path cannot be empty")
        
        path_obj = Path(file_path)
        
        # Ensure parent directory exists
        if path_obj.parent:
            ensure_directory(path_obj.parent)
        
        # Write the JSON file
        with open(path_obj, 'w') as f:
            json.dump(data, f, **kwargs)
        
        # Verify the file was created
        if path_obj.exists():
            file_size = path_obj.stat().st_size
            # Only log if there are issues - file saving is normal
        else:
            logger.error(f"❌ JSON file was not created: {path_obj}")
            raise RuntimeError("File was not created")
        
    except Exception as e:
        logger.error(f"❌ Failed to save JSON file: {e}")
        log_error_with_context(
            logger, e,
            context={"file_path": str(file_path), "data_type": type(data).__name__},
            operation="safe_json_dump"
        )
        raise

def safe_json_load(file_path: str | Path) -> Any:
    """Safely load data from JSON file with comprehensive error handling."""
    try:
        logger.debug(f"📂 Loading data from JSON file: {file_path}")
        
        if not file_path:
            raise ValueError("File path cannot be empty")
        
        path_obj = Path(file_path)
        
        # Check if file exists
        if not path_obj.exists():
            raise FileNotFoundError(f"JSON file not found: {path_obj}")
        
        if not path_obj.is_file():
            raise ValueError(f"Path is not a file: {path_obj}")
        
        # Check file size
        file_size = path_obj.stat().st_size
        if file_size == 0:
            logger.warning(f"⚠️ JSON file is empty: {path_obj}")
            return {}
        
        # Load the JSON file
        with open(path_obj, 'r') as f:
            data = json.load(f)
        
        # Only log if there are issues - file loading is normal
        return data
        
    except Exception as e:
        logger.error(f"❌ Failed to load JSON file: {e}")
        log_error_with_context(
            logger, e,
            context={"file_path": str(file_path)},
            operation="safe_json_load"
        )
        raise

async def safe_sleep(seconds: float) -> None:
    """Async sleep wrapper with comprehensive error handling."""
    try:
        logger.debug(f"⏰ Sleeping for {seconds} seconds")
        
        if seconds < 0:
            raise ValueError(f"Sleep duration cannot be negative: {seconds}")
        
        if seconds > 3600:  # 1 hour
            logger.warning(f"⚠️ Long sleep duration: {seconds} seconds")
        
        await asyncio.sleep(seconds)
        logger.debug(f"✅ Sleep completed: {seconds} seconds")
        
    except Exception as e:
        logger.error(f"❌ Error during sleep: {e}")
        log_error_with_context(
            logger, e,
            context={"seconds": seconds},
            operation="safe_sleep"
        )
        raise

async def safe_gather(*coroutines, return_exceptions: bool=True) -> list:
    """Safely gather multiple coroutines with comprehensive error handling."""
    try:
        logger.debug(f"🔄 Gathering {len(coroutines)} coroutines")
        
        if not coroutines:
            logger.warning("⚠️ No coroutines provided to gather")
            return []
        
        # Validate coroutines
        for i, coro in enumerate(coroutines):
            if not asyncio.iscoroutine(coro):
                logger.warning(f"⚠️ Item {i} is not a coroutine: {type(coro)}")
        
        results = await asyncio.gather(*coroutines, return_exceptions=return_exceptions)
        
        # Check for exceptions in results
        exception_count = sum(1 for r in results if isinstance(r, Exception))
        if exception_count > 0:
            logger.warning(f"⚠️ {exception_count} coroutines raised exceptions")
        
        logger.info(f"✅ Gathered {len(coroutines)} coroutines successfully")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error gathering coroutines: {e}")
        log_error_with_context(
            logger, e,
            context={"coroutine_count": len(coroutines), "return_exceptions": return_exceptions},
            operation="safe_gather"
        )
        raise

def create_async_task(coroutine: Any) -> asyncio.Task:
    """Create an async task safely with comprehensive error handling."""
    try:
        logger.debug(f"🎯 Creating async task for coroutine: {type(coroutine).__name__}")
        
        if not asyncio.iscoroutine(coroutine):
            raise ValueError(f"Expected coroutine, got {type(coroutine)}")
        
        loop = asyncio.get_event_loop()
        task = loop.create_task(coroutine)
        
        logger.info(f"✅ Created async task: {task.get_name()}")
        return task
        
    except Exception as e:
        logger.error(f"❌ Failed to create async task: {e}")
        log_error_with_context(
            logger, e,
            context={"coroutine_type": type(coroutine).__name__},
            operation="create_async_task"
        )
        raise

def safe_append(lst: list[Any], item: Any) -> list[Any]:
    """Safely append to a list with comprehensive error handling."""
    try:
        logger.debug(f"📝 Appending item to list: {type(item).__name__}")
        
        if lst is None:
            logger.debug("🔄 Creating new list (input was None)")
            lst = []
        
        if not isinstance(lst, list):
            logger.warning(f"⚠️ Expected list, got {type(lst)}, converting")
            lst = list(lst)
        
        lst.append(item)
        logger.debug(f"✅ Appended item, list length: {len(lst)}")
        return lst
        
    except Exception as e:
        logger.error(f"❌ Failed to append to list: {e}")
        log_error_with_context(
            logger, e,
            context={"list_type": type(lst).__name__, "item_type": type(item).__name__},
            operation="safe_append"
        )
        # Return a new list with the item as fallback
        return [item]

def safe_extend(lst: list[Any], items: list[Any]) -> list[Any]:
    """Safely extend a list with comprehensive error handling."""
    try:
        logger.debug(f"📝 Extending list with {len(items) if items else 0} items")
        
        if lst is None:
            logger.debug("🔄 Creating new list (input was None)")
            lst = []
        
        if not isinstance(lst, list):
            logger.warning(f"⚠️ Expected list, got {type(lst)}, converting")
            lst = list(lst)
        
        if items is None:
            logger.warning("⚠️ Items to extend with is None, skipping")
            return lst
        
        if not isinstance(items, (list, tuple, set)):
            logger.warning(f"⚠️ Expected iterable, got {type(items)}, converting to list")
            items = [items]
        
        lst.extend(items)
        logger.debug(f"✅ Extended list, new length: {len(lst)}")
        return lst
        
    except Exception as e:
        logger.error(f"❌ Failed to extend list: {e}")
        log_error_with_context(
            logger, e,
            context={"list_type": type(lst).__name__, "items_type": type(items).__name__},
            operation="safe_extend"
        )
        # Return original list as fallback
        return lst if lst is not None else []

def safe_dict_get(d: dict[Any, Any], key: Any, default: Any=None) -> Any:
    """Safely get value from dictionary with comprehensive error handling."""
    try:
        logger.debug(f"🔍 Getting value from dictionary for key: {key}")
        
        if d is None:
            logger.debug("⚠️ Dictionary is None, returning default")
            return default
        
        if not isinstance(d, dict):
            logger.warning(f"⚠️ Expected dict, got {type(d)}")
            return default
        
        result = d.get(key, default)
        logger.debug(f"✅ Retrieved value: {type(result).__name__}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to get value from dictionary: {e}")
        log_error_with_context(
            logger, e,
            context={"dict_type": type(d).__name__, "key": str(key)},
            operation="safe_dict_get"
        )
        return default

def safe_dict_items(d: dict[Any, Any]) -> list[tuple]:
    """Safely get items from dictionary with comprehensive error handling."""
    try:
        logger.debug(f"📋 Getting items from dictionary")
        
        if d is None:
            logger.debug("⚠️ Dictionary is None, returning empty list")
            return []
        
        if not isinstance(d, dict):
            logger.warning(f"⚠️ Expected dict, got {type(d)}")
            return []
        
        items = list(d.items())
        logger.debug(f"✅ Retrieved {len(items)} items from dictionary")
        return items
        
    except Exception as e:
        logger.error(f"❌ Failed to get items from dictionary: {e}")
        log_error_with_context(
            logger, e,
            context={"dict_type": type(d).__name__},
            operation="safe_dict_items"
        )
        return []

def safe_lower(s: str) -> str:
    """Safely convert string to lowercase with comprehensive error handling."""
    try:
        logger.debug(f"🔤 Converting string to lowercase")
        
        if s is None:
            logger.debug("⚠️ String is None, returning empty string")
            return ''
        
        result = str(s).lower()
        logger.debug(f"✅ Converted to lowercase: {result[:50]}{'...' if len(result) > 50 else ''}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to convert string to lowercase: {e}")
        log_error_with_context(
            logger, e,
            context={"string_type": type(s).__name__},
            operation="safe_lower"
        )
        return ''

def safe_upper(s: str) -> str:
    """Safely convert string to uppercase with comprehensive error handling."""
    try:
        logger.debug(f"🔤 Converting string to uppercase")
        
        if s is None:
            logger.debug("⚠️ String is None, returning empty string")
            return ''
        
        result = str(s).upper()
        logger.debug(f"✅ Converted to uppercase: {result[:50]}{'...' if len(result) > 50 else ''}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to convert string to uppercase: {e}")
        log_error_with_context(
            logger, e,
            context={"string_type": type(s).__name__},
            operation="safe_upper"
        )
        return ''

def safe_join(separator: str, items: list[Any]) -> str:
    """Safely join items into a string with comprehensive error handling."""
    try:
        logger.debug(f"🔗 Joining {len(items) if items else 0} items with separator: '{separator}'")
        
        if separator is None:
            logger.warning("⚠️ Separator is None, using empty string")
            separator = ''
        
        if items is None:
            logger.debug("⚠️ Items is None, returning empty string")
            return ''
        
        if not isinstance(items, (list, tuple, set)):
            logger.warning(f"⚠️ Expected iterable, got {type(items)}, converting to list")
            items = [items]
        
        # Convert all items to strings
        str_items = [str(item) for item in items]
        result = separator.join(str_items)
        
        logger.debug(f"✅ Joined items, result length: {len(result)}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to join items: {e}")
        log_error_with_context(
            logger, e,
            context={"separator": str(separator), "items_type": type(items).__name__},
            operation="safe_join"
        )
        return ''

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with comprehensive error handling."""
    try:
        logger.debug(f"🔧 Getting logger instance: {name}")
        
        if not name or not isinstance(name, str):
            raise ValueError(f"Logger name must be a non-empty string, got: {name}")
        
        result = logging.getLogger(name)
        logger.debug(f"✅ Retrieved logger: {name}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to get logger: {e}")
        log_error_with_context(
            logger, e,
            context={"name": str(name)},
            operation="get_logger"
        )
        # Return root logger as fallback
        return logging.getLogger()

def setup_basic_logging(level: int=logging.INFO) -> None:
    """Setup basic logging configuration with comprehensive error handling."""
    try:
        logger.info(f"🔧 Setting up basic logging with level: {level}")
        
        if not isinstance(level, int):
            raise ValueError(f"Log level must be an integer, got: {type(level)}")
        
        if level < 0 or level > 50:
            logger.warning(f"⚠️ Unusual log level: {level}")
        
        # Configure basic logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        logger.info(f"✅ Basic logging configured with level: {level}")
        
    except Exception as e:
        logger.error(f"❌ Failed to setup basic logging: {e}")
        log_error_with_context(
            logger, e,
            context={"level": level},
            operation="setup_basic_logging"
        )
        raise


def get_common_operations_health_status() -> Dict[str, Any]:
    """
    Get comprehensive health status of all common operations functions.
    
    Returns:
        Dict[str, Any]: Health status information
    """
    try:
        logger.info("🏥 Getting CommonOperations health status")
        
        # Test key functions
        health_tests = {
            "datetime_operations": {
                "get_current_datetime": get_current_datetime(),
                "get_today": get_today(),
                "format_datetime": format_datetime(datetime.datetime.now()),
                "parse_datetime": parse_datetime("2023-01-01 00:00:00")
            },
            "dataframe_operations": {
                "create_empty_dataframe": create_empty_dataframe(["test"]),
                "safe_mean": safe_mean([1, 2, 3, 4, 5]),
                "safe_std": safe_std([1, 2, 3, 4, 5])
            },
            "file_operations": {
                "safe_file_exists": safe_file_exists("/tmp"),
                "safe_lower": safe_lower("TEST"),
                "safe_upper": safe_upper("test"),
                "safe_join": safe_join(",", ["a", "b", "c"])
            },
            "list_operations": {
                "safe_append": safe_append([], "test"),
                "safe_extend": safe_extend([], ["a", "b"]),
                "safe_dict_get": safe_dict_get({"key": "value"}, "key"),
                "safe_dict_items": safe_dict_items({"a": 1, "b": 2})
            }
        }
        
        # Count successful operations
        total_operations = 0
        successful_operations = 0
        
        for category, operations in health_tests.items():
            for operation_name, result in operations.items():
                total_operations += 1
                if result is not None:
                    successful_operations += 1
        
        success_rate = (successful_operations / total_operations) * 100 if total_operations > 0 else 0
        
        # Determine overall health status
        if success_rate >= 95:
            status = "excellent"
        elif success_rate >= 85:
            status = "good"
        elif success_rate >= 70:
            status = "fair"
        else:
            status = "poor"
        
        health_info = {
            "status": status,
            "success_rate": success_rate,
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": total_operations - successful_operations,
            "test_results": health_tests,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        logger.info(f"✅ CommonOperations health check completed: {status} ({success_rate:.1f}%)")
        return health_info
        
    except Exception as e:
        logger.error(f"❌ Error getting CommonOperations health status: {e}")
        log_error_with_context(
            logger, e,
            operation="get_common_operations_health_status"
        )
        return {
            "status": "error",
            "success_rate": 0,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }

def create_argument_parser(description: str) -> argparse.ArgumentParser:
    """Create an argument parser with comprehensive error handling."""
    try:
        logger.debug(f"🔧 Creating argument parser: {description}")
        
        if not description or not isinstance(description, str):
            raise ValueError(f"Description must be a non-empty string, got: {description}")
        
        parser = argparse.ArgumentParser(description=description)
        logger.info(f"✅ Created argument parser: {description}")
        return parser
        
    except Exception as e:
        logger.error(f"❌ Failed to create argument parser: {e}")
        log_error_with_context(
            logger, e,
            context={"description": str(description)},
            operation="create_argument_parser"
        )
        raise

def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to parser with comprehensive error handling."""
    try:
        logger.debug("🔧 Adding common arguments to parser")
        
        if not isinstance(parser, argparse.ArgumentParser):
            raise ValueError(f"Expected ArgumentParser, got {type(parser)}")
        
        parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
        parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
        parser.add_argument('--output', '-o', type=str, help='Output file path')
        
        logger.info("✅ Added common arguments to parser")
        
    except Exception as e:
        logger.error(f"❌ Failed to add common arguments: {e}")
        log_error_with_context(
            logger, e,
            context={"parser_type": type(parser).__name__},
            operation="add_common_arguments"
        )
        raise

def safe_exception_handler(func: Callable) -> Callable:
    """Decorator for safe exception handling with comprehensive error handling."""

    def wrapper(*args, **kwargs) -> Any:
        try:
            logger.debug(f"🛡️ Executing function with exception handler: {func.__name__}")
            result = func(*args, **kwargs)
            logger.debug(f"✅ Function executed successfully: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"❌ Error in {func.__name__}: {e}")
            log_error_with_context(
                logger, e,
                context={"function": func.__name__, "module": func.__module__},
                operation="safe_exception_handler"
            )
            return None
    return wrapper

def safe_float(value: Any, default: float=0.0) -> float:
    """Safely convert to float with comprehensive error handling."""
    try:
        logger.debug(f"🔢 Converting to float: {value}")
        
        if value is None:
            logger.debug("⚠️ Value is None, returning default")
            return default
        
        result = float(value)
        logger.debug(f"✅ Converted to float: {result}")
        return result
        
    except (TypeError, ValueError) as e:
        logger.warning(f"⚠️ Failed to convert to float: {e}, using default: {default}")
        return default
    except Exception as e:
        logger.error(f"❌ Unexpected error converting to float: {e}")
        log_error_with_context(
            logger, e,
            context={"value": str(value), "default": default},
            operation="safe_float"
        )
        return default

def safe_int(value: Any, default: int=0) -> int:
    """Safely convert to int with comprehensive error handling."""
    try:
        logger.debug(f"🔢 Converting to int: {value}")
        
        if value is None:
            logger.debug("⚠️ Value is None, returning default")
            return default
        
        result = int(value)
        logger.debug(f"✅ Converted to int: {result}")
        return result
        
    except (TypeError, ValueError) as e:
        logger.warning(f"⚠️ Failed to convert to int: {e}, using default: {default}")
        return default
    except Exception as e:
        logger.error(f"❌ Unexpected error converting to int: {e}")
        log_error_with_context(
            logger, e,
            context={"value": str(value), "default": default},
            operation="safe_int"
        )
        return default

def suggest_float_uniform(trial: Any, name: str, low: float, high: float) -> float:
    """Wrapper for Optuna's suggest_float with comprehensive error handling."""
    try:
        logger.debug(f"🎯 Suggesting float uniform: {name} in range [{low}, {high}]")
        
        if not isinstance(name, str):
            raise ValueError(f"Name must be a string, got: {type(name)}")
        
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ValueError(f"Low and high must be numbers, got: {type(low)}, {type(high)}")
        
        if low >= high:
            raise ValueError(f"Low must be less than high, got: {low} >= {high}")
        
        if hasattr(trial, 'suggest_float'):
            result = trial.suggest_float(name, low, high)
            logger.debug(f"✅ Optuna suggested float: {result}")
            return result
        else:
            import random
            result = random.uniform(low, high)
            logger.debug(f"✅ Random suggested float: {result}")
            return result
        
    except Exception as e:
        logger.error(f"❌ Failed to suggest float uniform: {e}")
        log_error_with_context(
            logger, e,
            context={"name": name, "low": low, "high": high},
            operation="suggest_float_uniform"
        )
        # Return midpoint as fallback
        return (low + high) / 2

def suggest_int_uniform(trial: Any, name: str, low: int, high: int) -> int:
    """Wrapper for Optuna's suggest_int."""
    if hasattr(trial, 'suggest_int'):
        return trial.suggest_int(name, low, high)
    import random
    return random.randint(low, high)

def validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> bool:
    """Validate that a DataFrame has required columns."""
    if df is None or df.empty:
        return False
    return all((col in df.columns for col in required_columns))

def validate_numeric_range(value: float, min_val: float, max_val: float) -> bool:
    """Validate that a value is within a numeric range."""
    return min_val <= value <= max_val

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
            elif c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max or (c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max):
                df[col] = df[col].astype(np.float32)
    return df

def safe_read_parquet(file_path: str | Path, columns: list[str] | None=None) -> pd.DataFrame:
    """Safely read parquet file with error handling."""
    try:
        return pd.read_parquet(file_path, columns=columns)
    except Exception as e:
        logger = get_logger(__name__)
        logger.exception(f'Failed to read parquet file {file_path}: {e}')
        return pd.DataFrame()

def safe_to_parquet(df: pd.DataFrame, file_path: str | Path, **kwargs) -> bool:
    """Safely write DataFrame to parquet with error handling."""
    try:
        df.to_parquet(file_path, **kwargs)
        return True
    except Exception as e:
        logger = get_logger(__name__)
        logger.exception(f'Failed to write parquet file {file_path}: {e}')
        return False

def list_parquet_files(directory: str | Path, recursive: bool=True) -> list[Path]:
    """List all parquet files in a directory."""
    directory = Path(directory)
    if recursive:
        return list(directory.rglob('*.parquet'))
    return list(directory.glob('*.parquet'))

def generate_hash(data: str | bytes | pd.DataFrame, algorithm: str='md5') -> str:
    """Generate hash for data with support for different types."""
    if isinstance(data, pd.DataFrame):
        data = pd.util.hash_pandas_object(data).values.tobytes()
    elif isinstance(data, str):
        data = data.encode()
    if algorithm == 'md5':
        return hashlib.md5(data).hexdigest()
    if algorithm == 'sha256':
        return hashlib.sha256(data).hexdigest()
    msg = f'Unsupported algorithm: {algorithm}'
    raise ValueError(msg)

def generate_cache_key(prefix: str, *args, max_length: int=16) -> str:
    """Generate a cache key from multiple inputs."""
    combined = f'{prefix}_' + '_'.join((str(arg) for arg in args))
    hash_val = generate_hash(combined, 'sha256')
    return hash_val[:max_length]

def safe_copy(df: pd.DataFrame, deep: bool=True) -> pd.DataFrame:
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

def safe_glob(pattern: str, recursive: bool=False) -> list[Path]:
    """Safely glob for files with error handling."""
    try:
        files = glob.glob(pattern, recursive=recursive)
        return [Path(f) for f in files]
    except Exception:
        return []

def list_files(directory: str | Path, pattern: str='*', suffix: str | None=None) -> list[Path]:
    """List files in directory with optional pattern/suffix filter."""
    directory = Path(directory)
    if not directory.exists():
        return []
    if suffix:
        return [f for f in directory.iterdir() if f.is_file() and f.suffix == suffix]
    return [f for f in directory.glob(pattern) if f.is_file()]

def get_latest_file(directory: str | Path, pattern: str='*') -> Path | None:
    """Get the most recently modified file matching pattern."""
    files = list_files(directory, pattern)
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)

def validate_dataframe_schema(df: pd.DataFrame, required_columns: list[str], column_types: dict[str, type] | None=None) -> tuple[bool, list[str]]:
    """Validate DataFrame schema including column types."""
    errors = []
    missing = set(required_columns) - set(df.columns)
    if missing:
        errors.append(f'Missing columns: {missing}')
    if column_types:
        for col, expected_type in column_types.items():
            if col in df.columns:
                actual_type = df[col].dtype
                if not np.issubdtype(actual_type, expected_type):
                    errors.append(f'Column {col} has type {actual_type}, expected {expected_type}')
    return (len(errors) == 0, errors)

def validate_data_quality(df: pd.DataFrame, max_nan_ratio: float=0.1, check_duplicates: bool=True) -> dict[str, Any]:
    """Comprehensive data quality validation."""
    quality_report = {'total_rows': len(df), 'total_columns': len(df.columns), 'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024, 'issues': []}
    nan_ratios = df.isna().sum() / len(df)
    high_nan_cols = nan_ratios[nan_ratios > max_nan_ratio]
    if not high_nan_cols.empty:
        quality_report['issues'].append({'type': 'high_nan_ratio', 'columns': high_nan_cols.to_dict()})
    if check_duplicates:
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            quality_report['issues'].append({'type': 'duplicates', 'count': duplicates})
    quality_report['is_valid'] = len(quality_report['issues']) == 0
    return quality_report

def safe_resample(df: pd.DataFrame, rule: str, agg_dict: dict[str, str] | None=None) -> pd.DataFrame:
    """Safely resample time series data."""
    if not isinstance(df.index, pd.DatetimeIndex):
        msg = 'DataFrame must have DatetimeIndex'
        raise ValueError(msg)
    if agg_dict is None:
        agg_dict = {'close': 'last', 'open': 'first', 'high': 'max', 'low': 'min', 'volume': 'sum'}
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    return df.resample(rule).agg(agg_dict)

def align_dataframes(*dfs: pd.DataFrame, method: str='inner') -> list[pd.DataFrame]:
    """Align multiple DataFrames by index."""
    if len(dfs) < 2:
        return list(dfs)
    if method == 'inner':
        start = max((df.index.min() for df in dfs))
        end = min((df.index.max() for df in dfs))
        aligned = [df.loc[start:end] for df in dfs]
    else:
        aligned = list(dfs)
    return aligned

def safe_defaultdict(default_factory: Callable) -> defaultdict:
    """Create a defaultdict safely."""
    return defaultdict(default_factory)

def safe_counter(items: list[Any] | None=None) -> Counter:
    """Create a Counter safely."""
    return Counter(items or [])

def safe_deque(items: list[Any] | None=None, maxlen: int | None=None) -> deque:
    """Create a deque safely."""
    return deque(items or [], maxlen=maxlen)

def timed_operation(operation_name: str) -> None:
    """Decorator to time operations."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            start = time.time()
            logger = get_logger(func.__module__)
            logger.info(f'Starting {operation_name}...')
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                logger.info(f'Completed {operation_name} in {elapsed:.2f}s')
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.exception(f'Failed {operation_name} after {elapsed:.2f}s: {e}')
                raise
        return wrapper
    return decorator

def format_bytes(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f'{size_bytes:.2f} {unit}'
        size_bytes /= 1024.0
    return f'{size_bytes:.2f} PB'

def chunked_iterable(iterable: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split an iterable into chunks."""
    chunks = []
    for i in range(0, len(iterable), chunk_size):
        chunks.append(iterable[i:i + chunk_size])
    return chunks

def parallel_map(func: Callable, items: list[Any], max_workers: int | None=None) -> list[Any]:
    """Apply function to items in parallel."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))

def safe_log_metric(key: str, value: float, step: int | None=None) -> None:
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
__all__ = ['get_current_datetime', 'get_today', 'format_datetime', 'parse_datetime', 'create_empty_dataframe', 'safe_fillna', 'safe_rolling', 'safe_copy', 'safe_deepcopy', 'safe_resample', 'align_dataframes', 'safe_mean', 'safe_std', 'ensure_directory', 'safe_file_exists', 'safe_json_dump', 'safe_json_load', 'safe_glob', 'list_files', 'get_latest_file', 'safe_read_parquet', 'safe_to_parquet', 'list_parquet_files', 'generate_hash', 'generate_cache_key', 'safe_sleep', 'safe_gather', 'create_async_task', 'safe_append', 'safe_extend', 'safe_dict_get', 'safe_dict_items', 'safe_defaultdict', 'safe_counter', 'safe_deque', 'safe_lower', 'safe_upper', 'safe_join', 'get_logger', 'setup_basic_logging', 'create_argument_parser', 'add_common_arguments', 'safe_exception_handler', 'safe_float', 'safe_int', 'suggest_float_uniform', 'suggest_int_uniform', 'validate_dataframe', 'validate_numeric_range', 'validate_dataframe_schema', 'validate_data_quality', 'optimize_dataframe_dtypes', 'timed_operation', 'format_bytes', 'chunked_iterable', 'parallel_map', 'safe_log_metric', 'safe_log_params', 'safe_log_artifact']

def standardize_price_action_probabilities(probabilities: dict) -> dict:
    """Standardize various model probability outputs to the unified schema.

    Ensures keys exist and values are clamped to [0, 1]. Missing keys are filled with 0.5.
    """
    if probabilities is None:
        probabilities = {}
    out = {}
    for key in ['triple_barrier_probability', 'direction_probability', 'magnitude_probability', 'barrier_avoidance_probability']:
        val = probabilities.get(key, 0.5)
        try:
            val_f = float(val)
        except Exception:
            val_f = 0.5
        if val_f < 0.0:
            val_f = 0.0
        if val_f > 1.0:
            val_f = 1.0
        out[key] = val_f
    return out