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



# Logging setup moved to top of file to avoid undefined logger errors

# =============================================================================
# LOGGING UTILITIES
# =============================================================================


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




# =============================================================================
# DATETIME UTILITIES
# =============================================================================


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




# =============================================================================
# STRING UTILITIES
# =============================================================================


# =============================================================================
# COLLECTION UTILITIES
# =============================================================================


# =============================================================================
# ASYNC UTILITIES
# =============================================================================


# =============================================================================
# PERFORMANCE UTILITIES
# =============================================================================




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










# =============================================================================
# M1 OPTIMIZATION UTILITIES
# =============================================================================





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

except ImportError:
    pass



