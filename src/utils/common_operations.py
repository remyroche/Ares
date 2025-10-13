"""
Unified Common Operations - Enhanced Utility Functions

This module provides comprehensive utility functions for data operations,
DataFrame processing, validation, and common data processing utilities.
Consolidates functionality from common_operations.py and common_utilities.py.
"""

import json
import os
import logging
import asyncio
import time
import functools
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager
import pandas as pd
import numpy as np
from datetime import datetime, date
import concurrent.futures

# Import core utilities
from .core.common import create_fallback_logger, create_fallback_decorator

# Import M1 utilities
try:
    from .hardware.m1_gpu_utils import is_m1_available, is_mps_available
except ImportError:
    def is_m1_available():
        return False
    def is_mps_available():
        return False

# Setup logging early to avoid undefined logger errors
logger = logging.getLogger(__name__)

def get_m1_gpu_manager():
    """Get the M1 GPU manager instance."""
    try:
        from .hardware.m1_gpu_utils import get_m1_gpu_manager as _get_m1_gpu_manager
        return _get_m1_gpu_manager()
    except ImportError:
        logger.warning("⚠️ M1 GPU utilities not available")
        return None


def get_m1_memory_optimizer():
    """Get the M1 memory optimizer instance."""
    try:
        from .hardware.m1_memory_optimizer import get_m1_memory_optimizer as _get_m1_memory_optimizer
        return _get_m1_memory_optimizer()
    except ImportError:
        logger.warning("⚠️ M1 memory optimizer not available")
        return None


def get_m1_cpu_optimizer():
    """Get the M1 CPU optimizer instance."""
    try:
        from .hardware.m1_cpu_optimizer import get_m1_cpu_optimizer as _get_m1_cpu_optimizer
        return _get_m1_cpu_optimizer()
    except ImportError:
        logger.warning("⚠️ M1 CPU optimizer not available")
        return None


def cleanup_m1_optimizers():
    """Clean up M1 optimizers and release resources."""
    try:
        # Import optimizers
        from .hardware.m1_gpu_utils import get_m1_gpu_manager
        from .hardware.m1_memory_optimizer import get_m1_memory_optimizer
        from .hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

        # Get instances
        gpu_manager = get_m1_gpu_manager()
        memory_optimizer = get_m1_memory_optimizer()
        cpu_optimizer = get_m1_cpu_optimizer()

        # Clean up resources
        if memory_optimizer and hasattr(memory_optimizer, 'stop_monitoring'):
            memory_optimizer.stop_monitoring()

        # Log cleanup
        logger.info("🧠 M1 optimizers cleaned up successfully")

        return True

    except ImportError:
        logger.warning("⚠️ M1 optimizers not available for cleanup")
        return False
    except Exception as e:
        logger.error(f"❌ Error during M1 optimizer cleanup: {e}")
        return False


def integrate_with_m1_optimizers() -> dict:
    """Integrate with M1 GPU and CPU optimizers.

    Returns:
        Dictionary with integration status and component information
    """
    try:
        # Import M1 utilities

        # Initialize components
        gpu_manager = get_m1_gpu_manager()
        memory_optimizer = get_m1_memory_optimizer()
        cpu_optimizer = get_m1_cpu_optimizer()

        # Start memory monitoring
        memory_optimizer.start_monitoring()

        # Optimize numpy for M1
        cpu_optimizer.optimize_numpy_operations()

        # Log integration status
        gpu_info = gpu_manager.get_gpu_info()
        cpu_info = cpu_optimizer.get_cpu_info()

        logger.info("🧠 M1 Integration Status:")
        logger.info(f"   - M1 Hardware: {'✅ Available' if is_m1_available() else '❌ Not available'}")
        logger.info(f"   - MPS (GPU): {'✅ Available' if is_mps_available() else '❌ Not available'}")
        logger.info(f"   - Performance Cores: {cpu_info.get('performance_cores', 'Unknown')}")
        logger.info(f"   - Memory Monitoring: ✅ Active")

        return {
            'integration_status': 'success',
            'gpu_manager': is_mps_available(),
            'memory_optimizer': True,
            'cpu_optimizer': True,
            'gpu_info': gpu_info,
            'cpu_info': cpu_info,
            'success': True
        }

    except ImportError as e:
        logger.warning(f"⚠️ M1 utilities not available: {e}")
        return {
            'integration_status': 'failed',
            'error': str(e),
            'gpu_manager': False,
            'memory_optimizer': False,
            'cpu_optimizer': False,
            'success': False
        }
    except Exception as e:
        logger.error(f"❌ M1 integration failed: {e}")
        return {
            'integration_status': 'failed',
            'error': str(e),
            'gpu_manager': False,
            'memory_optimizer': False,
            'cpu_optimizer': False,
            'success': False
        }

# Logging setup moved to top of file to avoid undefined logger errors

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name or __name__)

def setup_basic_logging(level: int = logging.INFO) -> None:
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )

def safe_log_metric(name: str, value: float) -> None:
    """Safely log metric with proper error handling."""
    try:
        logger.info(f"📊 Metric {name}: {value}")
    except (AttributeError, TypeError) as e:
        logger.debug(f"Failed to log metric {name}: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error logging metric {name}: {e}")

def safe_log_params(params: Dict[str, Any]) -> None:
    """Safely log parameters with proper error handling."""
    try:
        logger.info(f"⚙️ Parameters: {params}")
    except (AttributeError, TypeError) as e:
        logger.debug(f"Failed to log parameters: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error logging parameters: {e}")

def safe_log_artifact(name: str, path: str) -> None:
    """Safely log artifact with proper error handling."""
    try:
        logger.info(f"📁 Artifact {name} saved to {path}")
    except (AttributeError, TypeError) as e:
        logger.debug(f"Failed to log artifact {name}: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error logging artifact {name}: {e}")

# =============================================================================
# DATETIME UTILITIES
# =============================================================================

def get_current_datetime() -> datetime:
    """Get current datetime."""
    return datetime.now()

def get_today() -> date:
    """Get today's date."""
    return date.today()

def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime to string."""
    return dt.strftime(format_str)

def parse_datetime(dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """Parse string to datetime."""
    return datetime.strptime(dt_str, format_str)

# =============================================================================
# FILE AND DIRECTORY UTILITIES
# =============================================================================

def ensure_directory(path: Union[str, Path]) -> bool:
    """Ensure directory exists with proper error handling."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except PermissionError as e:
        logger.error(f"❌ Permission denied creating directory {path}: {e}")
        return False
    except OSError as e:
        logger.error(f"❌ OS error creating directory {path}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error creating directory {path}: {e}")
        return False

def safe_file_exists(path: Union[str, Path]) -> bool:
    """Safely check if file exists with proper error handling."""
    try:
        return Path(path).exists()
    except (OSError, PermissionError) as e:
        logger.debug(f"Error checking file existence for {path}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error checking file existence for {path}: {e}")
        return False

def safe_json_dump(data: Any, file_path: Union[str, Path], **kwargs) -> bool:
    """Safely dump data to JSON file with proper error handling."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, **kwargs)
        return True
    except PermissionError as e:
        logger.error(f"❌ Permission denied writing JSON to {file_path}: {e}")
        return False
    except OSError as e:
        logger.error(f"❌ OS error writing JSON to {file_path}: {e}")
        return False
    except (TypeError, ValueError) as e:
        logger.error(f"❌ Data serialization error writing JSON to {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error saving JSON to {file_path}: {e}")
        return False

def safe_json_load(file_path: Union[str, Path], default: Any = None) -> Any:
    """Safely load JSON data from file with proper error handling."""
    try:
        if not safe_file_exists(file_path):
            logger.debug(f"JSON file not found: {file_path}")
            return default
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except PermissionError as e:
        logger.error(f"❌ Permission denied reading JSON from {file_path}: {e}")
        return default
    except OSError as e:
        logger.error(f"❌ OS error reading JSON from {file_path}: {e}")
        return default
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"❌ JSON parsing error in {file_path}: {e}")
        return default
    except Exception as e:
        logger.error(f"❌ Unexpected error loading JSON from {file_path}: {e}")
        return default

# =============================================================================
# DATAFRAME UTILITIES
# =============================================================================

def create_empty_dataframe(columns: List[str] = None) -> pd.DataFrame:
    """Create an empty DataFrame with specified columns."""
    return pd.DataFrame(columns=columns or [])

def validate_dataframe(df: Any) -> bool:
    """Validate DataFrame with proper type checking and error handling.

    Args:
        df: Object to validate as DataFrame

    Returns:
        bool: True if valid DataFrame, False otherwise
    """
    try:
        if df is None:
            logger.debug("DataFrame validation failed: df is None")
            return False
        if not isinstance(df, pd.DataFrame):
            logger.debug(f"DataFrame validation failed: not a DataFrame, got {type(df)}")
            return False
        if df.empty:
            logger.debug("DataFrame validation failed: DataFrame is empty")
            return False
        return True
    except AttributeError as e:
        logger.debug(f"DataFrame validation failed - attribute error: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error during DataFrame validation: {e}")
        return False

def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns with enhanced error handling.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        bool: True if all required columns present, False otherwise
    """
    try:
        if not validate_dataframe(df):
            logger.warning("DataFrame validation failed before column validation")
            return False

        if not required_columns:
            logger.debug("No required columns specified for validation")
            return True

        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.warning(f"⚠️ Missing required columns: {sorted(missing_columns)}")
            return False

        logger.debug(f"✅ All required columns present: {sorted(required_columns)}")
        return True
    except AttributeError as e:
        logger.error(f"❌ Attribute error validating DataFrame columns: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error validating DataFrame columns: {e}")
        return False

def safe_dataframe_operation(df: pd.DataFrame, operation: Callable[..., pd.DataFrame], *args, **kwargs) -> pd.DataFrame:
    """Safely perform operation on DataFrame with comprehensive error handling.

    Args:
        df: DataFrame to operate on
        operation: Function to apply to DataFrame
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation

    Returns:
        pd.DataFrame: Result of operation or original DataFrame on error
    """
    try:
        if not validate_dataframe(df):
            logger.warning("Cannot perform operation on invalid DataFrame")
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

        result = operation(df, *args, **kwargs)

        # Validate result is still a DataFrame
        if not isinstance(result, pd.DataFrame):
            logger.warning(f"Operation {operation.__name__} did not return DataFrame, got {type(result)}")
            return df

        return result
    except (AttributeError, TypeError) as e:
        logger.warning(f"⚠️ Error in DataFrame operation {operation.__name__}: {e}")
        return df
    except Exception as e:
        logger.error(f"❌ Unexpected error in DataFrame operation {operation.__name__}: {e}")
        return df

def safe_fillna(df: pd.DataFrame, value: Any = None, method: str = None) -> pd.DataFrame:
    """Safely fill NaN values in DataFrame."""
    try:
        if method:
            # Handle deprecated fillna methods
            if method == 'forward':
                method = 'ffill'
            elif method == 'backward':
                method = 'bfill'
            return df.fillna(method=method)
        return df.fillna(value)
    except Exception as e:
        logger.warning(f"⚠️ Error filling NaN values: {e}")
        return df

def safe_convert_dtypes(df: pd.DataFrame, dtype_mapping: Dict[str, str]) -> pd.DataFrame:
    """Safely convert DataFrame column dtypes."""
    try:
        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        return df
    except Exception as e:
        logger.warning(f"⚠️ Error converting dtypes: {e}")
        return df

def safe_merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Safely merge two DataFrames."""
    try:
        return pd.merge(df1, df2, **kwargs)
    except Exception as e:
        logger.warning(f"⚠️ Error merging DataFrames: {e}")
        return df1

def safe_drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Safely drop columns from DataFrame."""
    try:
        existing_columns = [col for col in columns if col in df.columns]
        return df.drop(columns=existing_columns)
    except Exception as e:
        logger.warning(f"⚠️ Error dropping columns: {e}")
        return df

def safe_rename_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """Safely rename DataFrame columns."""
    try:
        return df.rename(columns=column_mapping)
    except Exception as e:
        logger.warning(f"⚠️ Error renaming columns: {e}")
        return df

def validate_timestamp_column(df: pd.DataFrame, column: str) -> bool:
    """Validate that column contains valid timestamps."""
    try:
        if column not in df.columns:
            return False
        pd.to_datetime(df[column])
        return True
    except Exception:
        return False

def safe_timestamp_conversion(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Safely convert column to timestamp."""
    try:
        df[column] = pd.to_datetime(df[column])
        return df
    except Exception as e:
        logger.warning(f"⚠️ Error converting timestamp column {column}: {e}")
        return df

def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame data types for memory efficiency."""
    try:
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                except Exception as e:
                    logger.debug(f"⚠️ Could not convert column '{col}' to integer: {e}")
                    try:
                        df[col] = pd.to_numeric(df[col], downcast='float')
                    except Exception as e:
                        logger.debug(f"⚠️ Could not convert column '{col}' to float: {e}")
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
        return df
    except Exception as e:
        logger.warning(f"⚠️ Error optimizing DataFrame dtypes: {e}")
        return df

# =============================================================================
# DATA QUALITY UTILITIES
# =============================================================================

def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate data quality metrics for DataFrame."""
    try:
        metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
        }
        return metrics
    except Exception as e:
        logger.error(f"❌ Error calculating data quality metrics: {e}")
        return {}

def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive DataFrame information."""
    try:
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'index_type': type(df.index).__name__,
            'has_duplicates': df.duplicated().any(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns)
        }
        return info
    except Exception as e:
        logger.error(f"❌ Error getting DataFrame info: {e}")
        return {}

def create_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Create comprehensive data quality report."""
    try:
        report = {
            'basic_info': get_dataframe_info(df),
            'quality_metrics': calculate_data_quality_metrics(df),
            'issues': []
        }
        
        # Check for common data quality issues
        if report['quality_metrics']['missing_percentage'] > 50:
            report['issues'].append("High percentage of missing values")
        
        if report['quality_metrics']['duplicate_percentage'] > 10:
            report['issues'].append("High percentage of duplicate rows")
        
        if len(report['basic_info']['numeric_columns']) == 0:
            report['issues'].append("No numeric columns found")
        
        return report
    except Exception as e:
        logger.error(f"❌ Error creating data quality report: {e}")
        return {}

# =============================================================================
# MATH UTILITIES
# =============================================================================

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    try:
        return a / b if b != 0 else default
    except Exception:
        return default

def safe_log(x: float, default: float = 0.0) -> float:
    """Safely calculate logarithm."""
    try:
        return np.log(x) if x > 0 else default
    except Exception:
        return default

def safe_sqrt(x: float, default: float = 0.0) -> float:
    """Safely calculate square root."""
    try:
        return np.sqrt(x) if x >= 0 else default
    except Exception:
        return default

def safe_power(x: float, y: float, default: float = 0.0) -> float:
    """Safely calculate power."""
    try:
        return x ** y
    except Exception:
        return default

def safe_mean(series: pd.Series) -> float:
    """Safely calculate mean."""
    try:
        return float(series.mean())
    except Exception:
        return 0.0

def safe_std(series: pd.Series) -> float:
    """Safely calculate standard deviation."""
    try:
        return float(series.std())
    except Exception:
        return 0.0

def safe_correlation(x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray], default: float = 0.0) -> float:
    """Safely compute correlation between two vectors."""
    try:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if x_arr.ndim > 1:
            x_arr = x_arr.reshape(-1)
        if y_arr.ndim > 1:
            y_arr = y_arr.reshape(-1)

        valid_len = min(x_arr.size, y_arr.size)
        if valid_len < 2:
            return default

        x_arr = x_arr[:valid_len]
        y_arr = y_arr[:valid_len]

        if not np.isfinite(x_arr).all() or not np.isfinite(y_arr).all():
            return default

        corr_matrix = np.corrcoef(x_arr, y_arr)
        corr_value = corr_matrix[0, 1]
        if np.isfinite(corr_value):
            return float(corr_value)
    except Exception:
        return default

    return default

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except Exception:
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    try:
        return int(value)
    except Exception:
        return default

def validate_finite(value: Any, name: str = "value") -> float:
    """Validate that a value is finite."""
    try:
        val = float(value)
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val}")
        return val
    except Exception as e:
        raise ValueError(f"Invalid {name}: {e}")

def validate_positive(value: float, name: str = "value") -> float:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value

def validate_range(value: float, min_val: float = None, max_val: float = None, name: str = "value") -> float:
    """Validate that a value is in range."""
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")
    return value

def safe_kelly_calculation(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Safely calculate Kelly criterion."""
    try:
        if avg_loss <= 0:
            return 0.0
        return (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
    except Exception:
        return 0.0

def safe_weighted_average(values: List[float], weights: List[float]) -> float:
    """Safely calculate weighted average."""
    try:
        if not values or not weights or len(values) != len(weights):
            return 0.0
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        return sum(v * w for v, w in zip(values, weights)) / total_weight
    except Exception:
        return 0.0

def safe_percentage_change(old_value: float, new_value: float) -> float:
    """Safely calculate percentage change."""
    try:
        if old_value == 0:
            return 0.0
        return ((new_value - old_value) / old_value) * 100
    except Exception:
        return 0.0

def optimize_memory_usage() -> Dict[str, Any]:
    """
    Optimize memory usage by leveraging matrix operations manager.
    
    Returns:
        Dictionary containing memory optimization statistics
    """
    try:
        from .matrix_operations.convenience import optimize_memory_usage as matrix_optimize
        return matrix_optimize()
    except ImportError as e:
        logger.warning(f"⚠️ Matrix operations not available for memory optimization: {e}")
        # Return a fallback dictionary
        return {
            'status': 'unavailable',
            'message': 'Matrix operations module not available',
            'memory_freed_mb': 0.0,
            'success': False
        }
    except Exception as e:
        logger.error(f"❌ Memory optimization failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'memory_freed_mb': 0.0,
            'success': False
        }

def parallel_processing_optimizer(data: Any, operation: Callable, num_workers: int = None) -> Any:
    """
    Optimize parallel processing operations.
    
    Args:
        data: Data to process
        operation: Operation to apply
        num_workers: Number of parallel workers (None for auto-detection)
        
    Returns:
        Processed data
    """
    try:
        import multiprocessing
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        
        # Use concurrent processing for large datasets
        if hasattr(data, '__len__') and len(data) > 1000:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(operation, data))
            return results
        else:
            # For small datasets, direct processing is faster
            return [operation(item) for item in data]
    except Exception as e:
        logger.warning(f"⚠️ Parallel processing failed, falling back to sequential: {e}")
        # Fallback to sequential processing
        return [operation(item) for item in data]

# =============================================================================
# STRING UTILITIES
# =============================================================================

def safe_lower(s: str) -> str:
    """Safely convert string to lowercase."""
    try:
        return s.lower()
    except Exception:
        return s

def safe_upper(s: str) -> str:
    """Safely convert string to uppercase."""
    try:
        return s.upper()
    except Exception:
        return s

def safe_join(iterable: List[str], separator: str = " ") -> str:
    """Safely join strings."""
    try:
        return separator.join(str(item) for item in iterable)
    except Exception:
        return ""

# =============================================================================
# COLLECTION UTILITIES
# =============================================================================

def safe_append(lst: List[Any], item: Any) -> bool:
    """Safely append item to list."""
    try:
        lst.append(item)
        return True
    except Exception as e:
        logger.warning(f"⚠️ Error appending to list: {e}")
        return False

def safe_extend(lst: List[Any], items: List[Any]) -> bool:
    """Safely extend list with items."""
    try:
        lst.extend(items)
        return True
    except Exception as e:
        logger.warning(f"⚠️ Error extending list: {e}")
        return False

def safe_dict_get(d: Dict[Any, Any], key: Any, default: Any = None) -> Any:
    """Safely get value from dictionary."""
    try:
        return d.get(key, default)
    except Exception:
        return default

def safe_dict_items(d: Dict[Any, Any]) -> List[tuple]:
    """Safely get dictionary items."""
    try:
        return list(d.items())
    except Exception:
        return []

# =============================================================================
# ASYNC UTILITIES
# =============================================================================

def safe_sleep(seconds: float) -> None:
    """Safely sleep for specified seconds."""
    try:
        time.sleep(seconds)
    except Exception as e:
        logger.warning(f"⚠️ Error during sleep: {e}")

async def safe_gather(*coros) -> List[Any]:
    """Safely gather async coroutines."""
    try:
        return await asyncio.gather(*coros)
    except Exception as e:
        logger.error(f"❌ Error in async gather: {e}")
        return []

def create_async_task(coro) -> asyncio.Task:
    """Create async task."""
    return asyncio.create_task(coro)

# =============================================================================
# PERFORMANCE UTILITIES
# =============================================================================

def timed_operation(func: Callable) -> Callable:
    """Decorator to time operations."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"⏱️ Operation {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def chunked_iterable(iterable: List[Any], chunk_size: int):
    """Yield chunks of iterable."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]

def parallel_map(func: Callable, iterable: List[Any], max_workers: int = None) -> List[Any]:
    """Apply function to iterable in parallel."""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(func, iterable))
    except Exception as e:
        logger.error(f"❌ Error in parallel map: {e}")
        return [func(item) for item in iterable]

# =============================================================================
# MATRIX UTILITIES
# =============================================================================

def validate_correlation_matrix(corr_matrix: np.ndarray) -> bool:
    """Validate correlation matrix."""
    try:
        if not isinstance(corr_matrix, np.ndarray):
            return False
        if corr_matrix.ndim != 2:
            return False
        if corr_matrix.shape[0] != corr_matrix.shape[1]:
            return False
        # Check if all values are between -1 and 1
        return np.all((corr_matrix >= -1) & (corr_matrix <= 1))
    except Exception:
        return False

def safe_matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    """Safely calculate matrix inverse."""
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if regular inverse fails
        return np.linalg.pinv(matrix)
    except Exception:
        return np.eye(matrix.shape[0])

def math_safe(func: Callable, *args, default: Any = 0.0, **kwargs) -> Any:
    """Safely execute math function."""
    try:
        return func(*args, **kwargs)
    except Exception:
        return default

# =============================================================================
# EXCEPTIONS
# =============================================================================

class MathValidationError(Exception):
    """Exception raised for math validation errors."""
    pass

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def safe_rolling(series: pd.Series, window: int, **kwargs) -> pd.Series:
    """Safely apply rolling operation."""
    try:
        return series.rolling(window=window, **kwargs)
    except Exception as e:
        logger.warning(f"⚠️ Error in rolling operation: {e}")
        return series

def safe_groupby_operation(df: pd.DataFrame, group_cols: List[str], agg_dict: Dict[str, str]) -> pd.DataFrame:
    """Safely perform groupby operation."""
    try:
        return df.groupby(group_cols).agg(agg_dict)
    except Exception as e:
        logger.warning(f"⚠️ Error in groupby operation: {e}")
        return df

def safe_apply_function(df: pd.DataFrame, func: Callable, axis: int = 0) -> pd.DataFrame:
    """Safely apply function to DataFrame."""
    try:
        return df.apply(func, axis=axis)
    except Exception as e:
        logger.warning(f"⚠️ Error applying function: {e}")
        return df

def safe_filter_dataframe(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Safely filter DataFrame using query condition."""
    try:
        return df.query(condition)
    except Exception as e:
        logger.warning(f"⚠️ Error filtering DataFrame: {e}")
        return df

def create_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Create summary statistics for DataFrame."""
    try:
        summary = {
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'missing_values': df.isnull().sum().to_dict(),
            'unique_values': df.nunique().to_dict()
        }
        return summary
    except Exception as e:
        logger.error(f"❌ Error creating summary statistics: {e}")
        return {}

def safe_to_parquet(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> bool:
    """Safely save DataFrame to parquet format."""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        df.to_parquet(file_path, **kwargs)
        logger.info(f"✅ Successfully saved DataFrame to {file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving DataFrame to parquet {file_path}: {e}")
        return False

def safe_read_parquet(file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
    """Safely read DataFrame from parquet format."""
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not file_path.exists():
            logger.warning(f"⚠️ Parquet file does not exist: {file_path}")
            return None

        df = pd.read_parquet(file_path, **kwargs)
        logger.info(f"✅ Successfully read DataFrame from {file_path}")
        return df
    except Exception as e:
        logger.error(f"❌ Error reading DataFrame from parquet {file_path}: {e}")
        return None

def list_parquet_files(directory: Union[str, Path]) -> List[Path]:
    """List all parquet files in a directory."""
    try:
        if isinstance(directory, str):
            directory = Path(directory)

        if not directory.exists():
            logger.warning(f"⚠️ Directory does not exist: {directory}")
            return []

        parquet_files = list(directory.glob("**/*.parquet"))
        logger.info(f"✅ Found {len(parquet_files)} parquet files in {directory}")
        return parquet_files
    except Exception as e:
        logger.error(f"❌ Error listing parquet files in {directory}: {e}")
        return []


def get_latest_outcome_file(pattern: str = "market_analysis_optimal_regime_clustering_outcome_*.json") -> Optional[Path]:
    """Get the latest outcome file matching the given pattern from outcomes/ directory.

    Args:
        pattern: File pattern to search for (default: optimal regime clustering outcomes)

    Returns:
        Path to the latest file matching the pattern, or None if no files found
    """
    try:
        outcomes_dir = Path("outcomes")
        if not outcomes_dir.exists():
            logger.warning(f"⚠️ Outcomes directory does not exist: {outcomes_dir}")
            return None

        # Find files matching the pattern
        matching_files = list(outcomes_dir.glob(pattern))

        if not matching_files:
            logger.warning(f"⚠️ No files found matching pattern: {pattern}")
            return None

        # Sort by modification time (latest first)
        matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        latest_file = matching_files[0]
        logger.info(f"✅ Found latest outcome file: {latest_file}")
        return latest_file

    except Exception as e:
        logger.error(f"❌ Error finding latest outcome file with pattern {pattern}: {e}")
        return None


def load_latest_optimal_regime_clustering_outcome() -> Optional[Dict[str, Any]]:
    """Load the latest optimal regime clustering outcome file.

    Returns:
        Dictionary containing the outcome data, or None if loading fails
    """
    try:
        latest_file = get_latest_outcome_file("market_analysis_optimal_regime_clustering_outcome_*.json")

        if not latest_file:
            logger.warning("⚠️ No optimal regime clustering outcome file found")
            return None

        outcome_data = safe_json_load(latest_file)
        if outcome_data:
            logger.info(f"✅ Loaded optimal regime clustering outcome from {latest_file}")
            return outcome_data
        else:
            logger.warning(f"⚠️ Failed to load outcome data from {latest_file}")
            return None

    except Exception as e:
        logger.error(f"❌ Error loading latest optimal regime clustering outcome: {e}")
        return None

def safe_copy(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """Safely copy a file from source to destination."""
    try:
        import shutil

        if isinstance(src, str):
            src = Path(src)
        if isinstance(dst, str):
            dst = Path(dst)

        if not src.exists():
            logger.warning(f"⚠️ Source file does not exist: {src}")
            return False

        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, dst)
        logger.info(f"✅ Successfully copied {src} to {dst}")
        return True
    except Exception as e:
        logger.error(f"❌ Error copying {src} to {dst}: {e}")
        return False

def safe_deepcopy(obj: Any) -> Any:
    """Safely create a deep copy of an object."""
    try:
        import copy
        return copy.deepcopy(obj)
    except Exception as e:
        logger.warning(f"⚠️ Deep copy failed: {e}, returning original object")
        return obj

def safe_resample(df: pd.DataFrame, rule: str, agg_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Safely resample a DataFrame with error handling."""
    try:
        if agg_dict is None:
            # Default aggregation for time series data
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }

        resampled = df.resample(rule).agg(agg_dict)

        # Remove any columns that are all NaN
        resampled = resampled.dropna(axis=1, how='all')

        logger.info(f"✅ Successfully resampled DataFrame from {len(df)} to {len(resampled)} rows")
        return resampled

    except Exception as e:
        logger.error(f"❌ Error resampling DataFrame: {e}")
        return df

def align_dataframes(*dfs: pd.DataFrame, method: str = "inner") -> List[pd.DataFrame]:
    """Align multiple DataFrames by index using specified join method."""
    try:
        if not dfs:
            return []

        if len(dfs) == 1:
            return list(dfs)

        # Use the first DataFrame as the reference
        reference_df = dfs[0]

        aligned_dfs = [reference_df]

        for df in dfs[1:]:
            if method == "inner":
                aligned = reference_df.join(df, how="inner")
            elif method == "outer":
                aligned = reference_df.join(df, how="outer")
            elif method == "left":
                aligned = reference_df.join(df, how="left")
            elif method == "right":
                aligned = reference_df.join(df, how="right")
            else:
                logger.warning(f"⚠️ Unknown join method: {method}, using inner")
                aligned = reference_df.join(df, how="inner")

            aligned_dfs.append(aligned)

        logger.info(f"✅ Successfully aligned {len(dfs)} DataFrames using {method} join")
        return aligned_dfs

    except Exception as e:
        logger.error(f"❌ Error aligning DataFrames: {e}")
        return list(dfs)

def validate_dataframe_schema(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns."""
    try:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"❌ Missing required columns: {missing_columns}")
            return False

        logger.info(f"✅ DataFrame schema validation passed for {len(required_columns)} required columns")
        return True
    except Exception as e:
        logger.error(f"❌ Error validating DataFrame schema: {e}")
        return False

def validate_file_size(max_size_mb: int = 100):
    """Decorator to validate file size."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get file_path from kwargs if available
            file_path = kwargs.get('file_path')
            if file_path:
                if isinstance(file_path, str):
                    file_path = Path(file_path)

                if not file_path.exists():
                    logger.warning(f"⚠️ File does not exist: {file_path}")
                    raise ValueError(f"File does not exist: {file_path}")

                file_size_mb_actual = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb_actual > max_size_mb:
                    logger.warning(f"⚠️ File too large: {file_size_mb_actual:.2f}MB (max: {max_size_mb}MB)")
                    raise ValueError(f"File too large: {file_size_mb_actual:.2f}MB (max: {max_size_mb}MB)")

                logger.info(f"✅ File size validation passed: {file_size_mb_actual:.2f}MB")

            return func(*args, **kwargs)
        return wrapper
    return decorator

def guard_dataframe_nulls(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Guard against excessive null values in DataFrame."""
    try:
        if df is None:
            logger.warning("⚠️ DataFrame is None")
            return df

        null_ratio = df.isnull().mean().mean()
        if null_ratio > threshold:
            logger.warning(f"⚠️ High null ratio: {null_ratio:.2%} (threshold: {threshold:.2%})")
            # Fill with appropriate defaults
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median() if not df[col].median() != df[col].median() else 0)
                else:
                    df[col] = df[col].fillna('')

        logger.info(f"✅ DataFrame null guard passed with ratio: {null_ratio:.2%}")
        return df
    except Exception as e:
        logger.error(f"❌ Error in null guard: {e}")
        return df

def secure_file_path(allowed_dirs: List[str] = None):
    """Decorator to secure file paths."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Basic path security check
            if 'file_path' in kwargs:
                file_path = kwargs['file_path']
                if isinstance(file_path, str):
                    file_path = Path(file_path)
                # Basic security - prevent access to parent directories
                if '..' in str(file_path):
                    logger.warning(f"⚠️ Potential path traversal attempt: {file_path}")
                    raise ValueError("Path traversal not allowed")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def with_tracing_span(span_name: str = None):
    """Decorator for tracing spans."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Basic tracing - just log the function call
            logger.info(f"🔍 Tracing span: {span_name or func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"✅ Tracing span completed: {span_name or func.__name__}")
                return result
            except Exception as e:
                logger.error(f"❌ Tracing span failed: {span_name or func.__name__}: {e}")
                raise
        return wrapper
    return decorator

def sanitize_string(s: str, max_length: int = 255) -> str:
    """Sanitize string input."""
    try:
        if not isinstance(s, str):
            s = str(s)

        # Remove potentially dangerous characters
        import re
        s = re.sub(r'[^\w\s\-_.]', '', s)

        # Truncate if too long
        if len(s) > max_length:
            s = s[:max_length]

        return s.strip()
    except Exception as e:
        logger.error(f"❌ Error sanitizing string: {e}")
        return ""


# =============================================================================
# M1 OPTIMIZATION UTILITIES
# =============================================================================

def memory_checkpoint(name: str):
    """Create a memory checkpoint context manager.

    Args:
        name: Name of the checkpoint for logging

    Returns:
        Context manager for memory checkpointing
    """
    from contextlib import contextmanager

    @contextmanager
    def _memory_checkpoint():
        """Enhanced memory checkpoint with proper error handling and logging."""
        # Try to get M1 memory optimizer with specific error handling
        memory_optimizer = None
        try:
            memory_optimizer = get_m1_memory_optimizer()
        except ImportError as e:
            logger.debug(f"M1 memory optimizer not available: {e}")
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"M1 memory optimizer initialization failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting M1 memory optimizer: {e}")

        # Use memory checkpointing if available, otherwise just yield
        if memory_optimizer and hasattr(memory_optimizer, 'memory_checkpoint'):
            try:
                with memory_optimizer.memory_checkpoint(name):
                    yield
            except AttributeError as e:
                logger.warning(f"Memory checkpoint method not available: {e}")
                yield  # Fallback: just yield without checkpointing
            except Exception as e:
                logger.error(f"Error during memory checkpointing: {e}")
                yield  # Fallback: just yield without checkpointing
        else:
            # Fallback: just yield without checkpointing
            logger.debug(f"Memory checkpointing not available for {name}, using fallback")
            yield

    return _memory_checkpoint()


def gpu_context(name: str):
    """Create a GPU context manager.

    Args:
        name: Name of the context for logging

    Returns:
        Context manager for GPU operations
    """

    @contextmanager
    def _gpu_context():
        try:
            # Try to get M1 GPU manager
            gpu_manager = get_m1_gpu_manager()
            if gpu_manager and hasattr(gpu_manager, 'gpu_context'):
                with gpu_manager.gpu_context(name):
                    yield
            else:
                # Fallback: just yield without GPU context
                yield
        except Exception:
            # If anything fails, just yield without GPU context
            yield

    return _gpu_context()


def optimize_memory() -> Dict[str, Any]:
    """Optimize memory usage across the system.

    Returns:
        Dictionary with memory optimization results
    """
    try:
        # Try to get M1 memory optimizer
        memory_optimizer = get_m1_memory_optimizer()
        if memory_optimizer and hasattr(memory_optimizer, 'optimize_memory'):
            return memory_optimizer.optimize_memory()
        else:
            # Fallback: basic garbage collection
            import gc
            collected = gc.collect()
            return {
                'objects_collected': collected,
                'method': 'fallback_gc',
                'success': True
            }
    except Exception as e:
        logger.warning(f"⚠️ Memory optimization failed: {e}")
        return {
            'error': str(e),
            'method': 'failed',
            'success': False
        }


def get_memory_usage() -> float:
    """Get current memory usage in bytes.

    Returns:
        Current memory usage in bytes
    """
    try:
        import psutil
        return psutil.virtual_memory().used
    except ImportError:
        return 0

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    try:
        warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")
    except NameError:
        # warnings not available in this context
        pass

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

def get_memory_usage() -> float:
    """Get current memory usage in bytes."""
    try:
        return psutil.Process().memory_info().rss
    except ImportError:
        logger.warning("⚠️ psutil not available for memory monitoring")
        return 0.0


def validate_file_path(file_path: Union[str, Path]) -> bool:
    """Validate if a file path exists and is accessible.

    Args:
        file_path: Path to validate

    Returns:
        True if file exists and is accessible, False otherwise
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get the size of a file in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes, or 0 if file doesn't exist or can't be accessed
    """
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            return path.stat().st_size
        return 0
    except Exception:
        return 0


def check_disk_space(path: Union[str, Path], required_gb: float = 1.0) -> Dict[str, Any]:
    """Check if there's sufficient disk space available.

    Args:
        path: Path to check disk space for
        required_gb: Required disk space in GB

    Returns:
        Dictionary with disk space information and availability status
    """
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            path_obj = path_obj.parent if path_obj.parent.exists() else Path.home()

        stat = shutil.disk_usage(str(path_obj))
        total_gb = stat.total / (1024 ** 3)
        free_gb = stat.free / (1024 ** 3)
        used_gb = stat.used / (1024 ** 3)

        sufficient = free_gb >= required_gb

        return {
            'total_gb': round(total_gb, 2),
            'free_gb': round(free_gb, 2),
            'used_gb': round(used_gb, 2),
            'required_gb': required_gb,
            'sufficient': sufficient,
            'available_percentage': round((free_gb / total_gb) * 100, 2)
        }
    except Exception as e:
        logger.warning(f"⚠️ Failed to check disk space: {e}")
        return {
            'error': str(e),
            'sufficient': False,
            'total_gb': 0.0,
            'free_gb': 0.0,
            'used_gb': 0.0,
            'required_gb': required_gb,
            'available_percentage': 0.0
        }


class CommonUtilities:
    """Common utilities class for unified operations."""
    
    def __init__(self):
        """Initialize common utilities."""
        self.logger = logging.getLogger(__name__)
        self.m1_available = is_m1_available()
        self.mps_available = is_mps_available()
    
    def get_m1_status(self):
        """Get M1 status information."""
        return {
            'm1_available': self.m1_available,
            'mps_available': self.mps_available
        }
    
    def optimize_for_m1(self, data):
        """Optimize data processing for M1."""
        if self.m1_available:
            # M1-specific optimizations
            if hasattr(data, 'values'):
                return data.values
        return data
    
    def get_system_info(self):
        """Get system information."""
        return {
            'm1_available': self.m1_available,
            'mps_available': self.mps_available,
            'platform': os.name,
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
