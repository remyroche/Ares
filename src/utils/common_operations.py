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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime, date
import concurrent.futures

# Setup logging
logger = logging.getLogger(__name__)

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
    """Safely log metric."""
    try:
        logger.info(f"📊 Metric {name}: {value}")
    except Exception:
        pass

def safe_log_params(params: Dict[str, Any]) -> None:
    """Safely log parameters."""
    try:
        logger.info(f"⚙️ Parameters: {params}")
    except Exception:
        pass

def safe_log_artifact(name: str, path: str) -> None:
    """Safely log artifact."""
    try:
        logger.info(f"📁 Artifact {name} saved to {path}")
    except Exception:
        pass

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
    """Ensure directory exists."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"❌ Error creating directory {path}: {e}")
        return False

def safe_file_exists(path: Union[str, Path]) -> bool:
    """Safely check if file exists."""
    try:
        return Path(path).exists()
    except Exception:
        return False

def safe_json_dump(data: Any, file_path: Union[str, Path], **kwargs) -> bool:
    """Safely dump data to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, **kwargs)
        return True
    except Exception as e:
        logger.error(f"❌ Error saving JSON to {file_path}: {e}")
        return False

def safe_json_load(file_path: Union[str, Path], default: Any = None) -> Any:
    """Safely load JSON data from file."""
    try:
        if not safe_file_exists(file_path):
            return default
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading JSON from {file_path}: {e}")
        return default

# =============================================================================
# DATAFRAME UTILITIES
# =============================================================================

def create_empty_dataframe(columns: List[str] = None) -> pd.DataFrame:
    """Create an empty DataFrame with specified columns."""
    return pd.DataFrame(columns=columns or [])

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate DataFrame."""
    try:
        return isinstance(df, pd.DataFrame) and not df.empty
    except Exception:
        return False

def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns."""
    try:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.warning(f"⚠️ Missing required columns: {missing_columns}")
            return False
        return True
    except Exception as e:
        logger.error(f"❌ Error validating DataFrame columns: {e}")
        return False

def safe_dataframe_operation(df: pd.DataFrame, operation: Callable, *args, **kwargs) -> pd.DataFrame:
    """Safely perform operation on DataFrame."""
    try:
        return operation(df, *args, **kwargs)
    except Exception as e:
        logger.warning(f"⚠️ Error in DataFrame operation {operation.__name__}: {e}")
        return df

def safe_fillna(df: pd.DataFrame, value: Any = None, method: str = None) -> pd.DataFrame:
    """Safely fill NaN values in DataFrame."""
    try:
        if method:
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
                except:
                    try:
                        df[col] = pd.to_numeric(df[col], downcast='float')
                    except:
                        pass
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