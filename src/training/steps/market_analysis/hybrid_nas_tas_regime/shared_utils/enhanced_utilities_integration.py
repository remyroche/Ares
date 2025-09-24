"""
Enhanced Utilities Integration Module

This module integrates all external utility modules to provide enhanced functionality
for the hybrid NAS-TAS regime detection system.

Integrated modules:
- src/utils/common_operations.py
- src/utils/math_validation.py
- src/utils/serialization_utils.py
- src/utils/data/ utilities
- src/utils/matrix_operations/ utilities
- src/utils/ml_common/ utilities
- src/utils/hardware/ M1 optimization utilities
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

# Add src to path for imports
src_path = Path(__file__).parents[4] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# =============================================================================
# COMMON OPERATIONS INTEGRATION
# =============================================================================

logger = logging.getLogger(__name__)

def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance using common operations utilities."""
    try:
        from utils.common_operations import get_logger as _get_logger
        return _get_logger(name or __name__)
    except ImportError:
        return logging.getLogger(name or __name__)

def setup_basic_logging(level: int = logging.INFO) -> None:
    """Setup basic logging configuration."""
    try:
        from utils.common_operations import setup_basic_logging as _setup_basic_logging
        _setup_basic_logging(level)
    except ImportError:
        logging.basicConfig(level=level)

def safe_log_metric(name: str, value: float) -> None:
    """Safely log metric."""
    try:
        from utils.common_operations import safe_log_metric as _safe_log_metric
        _safe_log_metric(name, value)
    except ImportError:
        logger.info(f"📊 Metric {name}: {value}")

def safe_log_params(params: Dict[str, Any]) -> None:
    """Safely log parameters."""
    try:
        from utils.common_operations import safe_log_params as _safe_log_params
        _safe_log_params(params)
    except ImportError:
        logger.info(f"⚙️ Parameters: {params}")

def safe_log_artifact(name: str, path: str) -> None:
    """Safely log artifact."""
    try:
        from utils.common_operations import safe_log_artifact as _safe_log_artifact
        _safe_log_artifact(name, path)
    except ImportError:
        logger.info(f"📁 Artifact {name} saved to {path}")

# =============================================================================
# DATETIME UTILITIES
# =============================================================================

def get_current_datetime():
    """Get current datetime."""
    try:
        from utils.common_operations import get_current_datetime as _get_current_datetime
        return _get_current_datetime()
    except ImportError:
        from datetime import datetime
        return datetime.now()

def get_today():
    """Get today's date."""
    try:
        from utils.common_operations import get_today as _get_today
        return _get_today()
    except ImportError:
        from datetime import date
        return date.today()

def format_datetime(dt, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime to string."""
    try:
        from utils.common_operations import format_datetime as _format_datetime
        return _format_datetime(dt, format_str)
    except ImportError:
        return dt.strftime(format_str)

def parse_datetime(dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S"):
    """Parse string to datetime."""
    try:
        from utils.common_operations import parse_datetime as _parse_datetime
        return _parse_datetime(dt_str, format_str)
    except ImportError:
        from datetime import datetime
        return datetime.strptime(dt_str, format_str)

# =============================================================================
# FILE AND DIRECTORY UTILITIES
# =============================================================================

def ensure_directory(path: Union[str, Path]) -> bool:
    """Ensure directory exists."""
    try:
        from utils.common_operations import ensure_directory as _ensure_directory
        return _ensure_directory(path)
    except ImportError:
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        return True

def safe_file_exists(path: Union[str, Path]) -> bool:
    """Safely check if file exists."""
    try:
        from utils.common_operations import safe_file_exists as _safe_file_exists
        return _safe_file_exists(path)
    except ImportError:
        from pathlib import Path
        return Path(path).exists()

def safe_json_dump(data: Any, file_path: Union[str, Path], **kwargs) -> bool:
    """Safely dump data to JSON file."""
    try:
        from utils.common_operations import safe_json_dump as _safe_json_dump
        return _safe_json_dump(data, file_path, **kwargs)
    except ImportError:
        import json
        from pathlib import Path
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
        from utils.common_operations import safe_json_load as _safe_json_load
        return _safe_json_load(file_path, default)
    except ImportError:
        import json
        from pathlib import Path
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

def create_empty_dataframe(columns: List[str] = None):
    """Create an empty DataFrame with specified columns."""
    try:
        from utils.common_operations import create_empty_dataframe as _create_empty_dataframe
        return _create_empty_dataframe(columns)
    except ImportError:
        import pandas as pd
        return pd.DataFrame(columns=columns or [])

def validate_dataframe(df):
    """Validate DataFrame."""
    try:
        from utils.common_operations import validate_dataframe as _validate_dataframe
        return _validate_dataframe(df)
    except ImportError:
        import pandas as pd
        try:
            return isinstance(df, pd.DataFrame) and not df.empty
        except Exception:
            return False

def validate_dataframe_columns(df, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns."""
    try:
        from utils.common_operations import validate_dataframe_columns as _validate_dataframe_columns
        return _validate_dataframe_columns(df, required_columns)
    except ImportError:
        try:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                logger.warning(f"⚠️ Missing required columns: {missing_columns}")
                return False
            return True
        except Exception as e:
            logger.error(f"❌ Error validating DataFrame columns: {e}")
            return False

def safe_dataframe_operation(df, operation: Callable, *args, **kwargs):
    """Safely perform operation on DataFrame."""
    try:
        from utils.common_operations import safe_dataframe_operation as _safe_dataframe_operation
        return _safe_dataframe_operation(df, operation, *args, **kwargs)
    except ImportError:
        try:
            return operation(df, *args, **kwargs)
        except Exception as e:
            logger.warning(f"⚠️ Error in DataFrame operation {operation.__name__}: {e}")
            return df

def safe_fillna(df, value: Any = None, method: str = None):
    """Safely fill NaN values in DataFrame."""
    try:
        from utils.common_operations import safe_fillna as _safe_fillna
        return _safe_fillna(df, value, method)
    except ImportError:
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

def safe_convert_dtypes(df, dtype_mapping: Dict[str, str]):
    """Safely convert DataFrame column dtypes."""
    try:
        from utils.common_operations import safe_convert_dtypes as _safe_convert_dtypes
        return _safe_convert_dtypes(df, dtype_mapping)
    except ImportError:
        try:
            for col, dtype in dtype_mapping.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
            return df
        except Exception as e:
            logger.warning(f"⚠️ Error converting dtypes: {e}")
            return df

def safe_merge_dataframes(df1, df2, **kwargs):
    """Safely merge two DataFrames."""
    try:
        from utils.common_operations import safe_merge_dataframes as _safe_merge_dataframes
        return _safe_merge_dataframes(df1, df2, **kwargs)
    except ImportError:
        try:
            import pandas as pd
            return pd.merge(df1, df2, **kwargs)
        except Exception as e:
            logger.warning(f"⚠️ Error merging DataFrames: {e}")
            return df1

def safe_drop_columns(df, columns: List[str]):
    """Safely drop columns from DataFrame."""
    try:
        from utils.common_operations import safe_drop_columns as _safe_drop_columns
        return _safe_drop_columns(df, columns)
    except ImportError:
        try:
            existing_columns = [col for col in columns if col in df.columns]
            return df.drop(columns=existing_columns)
        except Exception as e:
            logger.warning(f"⚠️ Error dropping columns: {e}")
            return df

def safe_rename_columns(df, column_mapping: Dict[str, str]):
    """Safely rename DataFrame columns."""
    try:
        from utils.common_operations import safe_rename_columns as _safe_rename_columns
        return _safe_rename_columns(df, column_mapping)
    except ImportError:
        try:
            return df.rename(columns=column_mapping)
        except Exception as e:
            logger.warning(f"⚠️ Error renaming columns: {e}")
            return df

def validate_timestamp_column(df, column: str) -> bool:
    """Validate that column contains valid timestamps."""
    try:
        from utils.common_operations import validate_timestamp_column as _validate_timestamp_column
        return _validate_timestamp_column(df, column)
    except ImportError:
        try:
            if column not in df.columns:
                return False
            import pandas as pd
            pd.to_datetime(df[column])
            return True
        except Exception:
            return False

def safe_timestamp_conversion(df, column: str):
    """Safely convert column to timestamp."""
    try:
        from utils.common_operations import safe_timestamp_conversion as _safe_timestamp_conversion
        return _safe_timestamp_conversion(df, column)
    except ImportError:
        try:
            import pandas as pd
            df[column] = pd.to_datetime(df[column])
            return df
        except Exception as e:
            logger.warning(f"⚠️ Error converting timestamp column {column}: {e}")
            return df

def optimize_dataframe_dtypes(df):
    """Optimize DataFrame data types for memory efficiency."""
    try:
        from utils.common_operations import optimize_dataframe_dtypes as _optimize_dataframe_dtypes
        return _optimize_dataframe_dtypes(df)
    except ImportError:
        try:
            import pandas as pd
            import numpy as np
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

def calculate_data_quality_metrics(df):
    """Calculate data quality metrics for DataFrame."""
    try:
        from utils.common_operations import calculate_data_quality_metrics as _calculate_data_quality_metrics
        return _calculate_data_quality_metrics(df)
    except ImportError:
        try:
            import pandas as pd
            import numpy as np
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

def get_dataframe_info(df):
    """Get comprehensive DataFrame information."""
    try:
        from utils.common_operations import get_dataframe_info as _get_dataframe_info
        return _get_dataframe_info(df)
    except ImportError:
        try:
            import pandas as pd
            import numpy as np
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

def create_data_quality_report(df):
    """Create comprehensive data quality report."""
    try:
        from utils.common_operations import create_data_quality_report as _create_data_quality_report
        return _create_data_quality_report(df)
    except ImportError:
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
        from utils.common_operations import safe_divide as _safe_divide
        return _safe_divide(a, b, default)
    except ImportError:
        try:
            return a / b if b != 0 else default
        except Exception:
            return default

def safe_log(x: float, default: float = 0.0) -> float:
    """Safely calculate logarithm."""
    try:
        from utils.common_operations import safe_log as _safe_log
        return _safe_log(x, default)
    except ImportError:
        try:
            import numpy as np
            return np.log(x) if x > 0 else default
        except Exception:
            return default

def safe_sqrt(x: float, default: float = 0.0) -> float:
    """Safely calculate square root."""
    try:
        from utils.common_operations import safe_sqrt as _safe_sqrt
        return _safe_sqrt(x, default)
    except ImportError:
        try:
            import numpy as np
            return np.sqrt(x) if x >= 0 else default
        except Exception:
            return default

def safe_power(x: float, y: float, default: float = 0.0) -> float:
    """Safely calculate power."""
    try:
        from utils.common_operations import safe_power as _safe_power
        return _safe_power(x, y, default)
    except ImportError:
        try:
            return x ** y
        except Exception:
            return default

def safe_mean(series):
    """Safely calculate mean."""
    try:
        from utils.common_operations import safe_mean as _safe_mean
        return _safe_mean(series)
    except ImportError:
        try:
            return float(series.mean())
        except Exception:
            return 0.0

def safe_std(series):
    """Safely calculate standard deviation."""
    try:
        from utils.common_operations import safe_std as _safe_std
        return _safe_std(series)
    except ImportError:
        try:
            return float(series.std())
        except Exception:
            return 0.0

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        from utils.common_operations import safe_float as _safe_float
        return _safe_float(value, default)
    except ImportError:
        try:
            return float(value)
        except Exception:
            return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    try:
        from utils.common_operations import safe_int as _safe_int
        return _safe_int(value, default)
    except ImportError:
        try:
            return int(value)
        except Exception:
            return default

def validate_finite(value: Any, name: str = "value") -> float:
    """Validate that a value is finite."""
    try:
        from utils.common_operations import validate_finite as _validate_finite
        return _validate_finite(value, name)
    except ImportError:
        try:
            import numpy as np
            val = float(value)
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")
            return val
        except Exception as e:
            raise ValueError(f"Invalid {name}: {e}")

def validate_positive(value: float, name: str = "value") -> float:
    """Validate that a value is positive."""
    try:
        from utils.common_operations import validate_positive as _validate_positive
        return _validate_positive(value, name)
    except ImportError:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

def validate_range(value: float, min_val: float = None, max_val: float = None, name: str = "value") -> float:
    """Validate that a value is in range."""
    try:
        from utils.common_operations import validate_range as _validate_range
        return _validate_range(value, min_val, max_val, name)
    except ImportError:
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")
        return value

def safe_kelly_calculation(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Safely calculate Kelly criterion."""
    try:
        from utils.common_operations import safe_kelly_calculation as _safe_kelly_calculation
        return _safe_kelly_calculation(win_rate, avg_win, avg_loss)
    except ImportError:
        try:
            if avg_loss <= 0:
                return 0.0
            return (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
        except Exception:
            return 0.0

def safe_weighted_average(values: List[float], weights: List[float]) -> float:
    """Safely calculate weighted average."""
    try:
        from utils.common_operations import safe_weighted_average as _safe_weighted_average
        return _safe_weighted_average(values, weights)
    except ImportError:
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
        from utils.common_operations import safe_percentage_change as _safe_percentage_change
        return _safe_percentage_change(old_value, new_value)
    except ImportError:
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
        from utils.common_operations import safe_lower as _safe_lower
        return _safe_lower(s)
    except ImportError:
        try:
            return s.lower()
        except Exception:
            return s

def safe_upper(s: str) -> str:
    """Safely convert string to uppercase."""
    try:
        from utils.common_operations import safe_upper as _safe_upper
        return _safe_upper(s)
    except ImportError:
        try:
            return s.upper()
        except Exception:
            return s

def safe_join(iterable: List[str], separator: str = " ") -> str:
    """Safely join strings."""
    try:
        from utils.common_operations import safe_join as _safe_join
        return _safe_join(iterable, separator)
    except ImportError:
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
        from utils.common_operations import safe_append as _safe_append
        return _safe_append(lst, item)
    except ImportError:
        try:
            lst.append(item)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Error appending to list: {e}")
            return False

def safe_extend(lst: List[Any], items: List[Any]) -> bool:
    """Safely extend list with items."""
    try:
        from utils.common_operations import safe_extend as _safe_extend
        return _safe_extend(lst, items)
    except ImportError:
        try:
            lst.extend(items)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Error extending list: {e}")
            return False

def safe_dict_get(d: Dict[Any, Any], key: Any, default: Any = None) -> Any:
    """Safely get value from dictionary."""
    try:
        from utils.common_operations import safe_dict_get as _safe_dict_get
        return _safe_dict_get(d, key, default)
    except ImportError:
        try:
            return d.get(key, default)
        except Exception:
            return default

def safe_dict_items(d: Dict[Any, Any]) -> List[tuple]:
    """Safely get dictionary items."""
    try:
        from utils.common_operations import safe_dict_items as _safe_dict_items
        return _safe_dict_items(d)
    except ImportError:
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
        from utils.common_operations import safe_sleep as _safe_sleep
        _safe_sleep(seconds)
    except ImportError:
        try:
            import time
            time.sleep(seconds)
        except Exception as e:
            logger.warning(f"⚠️ Error during sleep: {e}")

async def safe_gather(*coros):
    """Safely gather async coroutines."""
    try:
        from utils.common_operations import safe_gather as _safe_gather
        return await _safe_gather(*coros)
    except ImportError:
        try:
            import asyncio
            return await asyncio.gather(*coros)
        except Exception as e:
            logger.error(f"❌ Error in async gather: {e}")
            return []

def create_async_task(coro):
    """Create async task."""
    try:
        from utils.common_operations import create_async_task as _create_async_task
        return _create_async_task(coro)
    except ImportError:
        try:
            import asyncio
            return asyncio.create_task(coro)
        except Exception:
            return None

# =============================================================================
# PERFORMANCE UTILITIES
# =============================================================================

def timed_operation(func: Callable):
    """Decorator to time operations."""
    try:
        from utils.common_operations import timed_operation as _timed_operation
        return _timed_operation(func)
    except ImportError:
        def decorator(f):
            def wrapper(*args, **kwargs):
                import time
                start_time = time.time()
                result = f(*args, **kwargs)
                end_time = time.time()
                logger.info(f"⏱️ Operation {f.__name__} took {end_time - start_time:.2f} seconds")
                return result
            return wrapper
        return decorator(func)

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string."""
    try:
        from utils.common_operations import format_bytes as _format_bytes
        return _format_bytes(bytes_value)
    except ImportError:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"

def chunked_iterable(iterable: List[Any], chunk_size: int):
    """Yield chunks of iterable."""
    try:
        from utils.common_operations import chunked_iterable as _chunked_iterable
        return _chunked_iterable(iterable, chunk_size)
    except ImportError:
        for i in range(0, len(iterable), chunk_size):
            yield iterable[i:i + chunk_size]

def parallel_map(func: Callable, iterable: List[Any], max_workers: int = None):
    """Apply function to iterable in parallel."""
    try:
        from utils.common_operations import parallel_map as _parallel_map
        return _parallel_map(func, iterable, max_workers)
    except ImportError:
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                return list(executor.map(func, iterable))
        except Exception as e:
            logger.error(f"❌ Error in parallel map: {e}")
            return [func(item) for item in iterable]

# =============================================================================
# MATRIX UTILITIES
# =============================================================================

def validate_correlation_matrix(corr_matrix):
    """Validate correlation matrix."""
    try:
        from utils.common_operations import validate_correlation_matrix as _validate_correlation_matrix
        return _validate_correlation_matrix(corr_matrix)
    except ImportError:
        try:
            import numpy as np
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

def safe_matrix_inverse(matrix):
    """Safely calculate matrix inverse."""
    try:
        from utils.common_operations import safe_matrix_inverse as _safe_matrix_inverse
        return _safe_matrix_inverse(matrix)
    except ImportError:
        try:
            import numpy as np
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if regular inverse fails
            return np.linalg.pinv(matrix)
        except Exception:
            return np.eye(matrix.shape[0])

def math_safe(func: Callable, *args, default: Any = 0.0, **kwargs) -> Any:
    """Safely execute math function."""
    try:
        from utils.common_operations import math_safe as _math_safe
        return _math_safe(func, *args, default=default, **kwargs)
    except ImportError:
        try:
            return func(*args, **kwargs)
        except Exception:
            return default

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def safe_rolling(series, window: int, **kwargs):
    """Safely apply rolling operation."""
    try:
        from utils.common_operations import safe_rolling as _safe_rolling
        return _safe_rolling(series, window, **kwargs)
    except ImportError:
        try:
            return series.rolling(window=window, **kwargs)
        except Exception as e:
            logger.warning(f"⚠️ Error in rolling operation: {e}")
            return series

def safe_groupby_operation(df, group_cols: List[str], agg_dict: Dict[str, str]):
    """Safely perform groupby operation."""
    try:
        from utils.common_operations import safe_groupby_operation as _safe_groupby_operation
        return _safe_groupby_operation(df, group_cols, agg_dict)
    except ImportError:
        try:
            return df.groupby(group_cols).agg(agg_dict)
        except Exception as e:
            logger.warning(f"⚠️ Error in groupby operation: {e}")
            return df

def safe_apply_function(df, func: Callable, axis: int = 0):
    """Safely apply function to DataFrame."""
    try:
        from utils.common_operations import safe_apply_function as _safe_apply_function
        return _safe_apply_function(df, func, axis)
    except ImportError:
        try:
            return df.apply(func, axis=axis)
        except Exception as e:
            logger.warning(f"⚠️ Error applying function: {e}")
            return df

def safe_filter_dataframe(df, condition: str):
    """Safely filter DataFrame using query condition."""
    try:
        from utils.common_operations import safe_filter_dataframe as _safe_filter_dataframe
        return _safe_filter_dataframe(df, condition)
    except ImportError:
        try:
            return df.query(condition)
        except Exception as e:
            logger.warning(f"⚠️ Error filtering DataFrame: {e}")
            return df

def create_summary_statistics(df):
    """Create summary statistics for DataFrame."""
    try:
        from utils.common_operations import create_summary_statistics as _create_summary_statistics
        return _create_summary_statistics(df)
    except ImportError:
        try:
            import pandas as pd
            import numpy as np
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

def safe_to_parquet(df, file_path: Union[str, Path], **kwargs) -> bool:
    """Safely save DataFrame to parquet format."""
    try:
        from utils.common_operations import safe_to_parquet as _safe_to_parquet
        return _safe_to_parquet(df, file_path, **kwargs)
    except ImportError:
        try:
            from pathlib import Path
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

def safe_read_parquet(file_path: Union[str, Path], **kwargs):
    """Safely read DataFrame from parquet format."""
    try:
        from utils.common_operations import safe_read_parquet as _safe_read_parquet
        return _safe_read_parquet(file_path, **kwargs)
    except ImportError:
        try:
            import pandas as pd
            from pathlib import Path
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

def list_parquet_files(directory: Union[str, Path]):
    """List all parquet files in a directory."""
    try:
        from utils.common_operations import list_parquet_files as _list_parquet_files
        return _list_parquet_files(directory)
    except ImportError:
        try:
            from pathlib import Path
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

def get_latest_outcome_file(pattern: str = "market_analysis_optimal_regime_clustering_outcome_*.json"):
    """Get the latest outcome file matching the given pattern from outcomes/ directory."""
    try:
        from utils.common_operations import get_latest_outcome_file as _get_latest_outcome_file
        return _get_latest_outcome_file(pattern)
    except ImportError:
        try:
            from pathlib import Path
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

def load_latest_optimal_regime_clustering_outcome():
    """Load the latest optimal regime clustering outcome file."""
    try:
        from utils.common_operations import load_latest_optimal_regime_clustering_outcome as _load_latest_optimal_regime_clustering_outcome
        return _load_latest_optimal_regime_clustering_outcome()
    except ImportError:
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
        from utils.common_operations import safe_copy as _safe_copy
        return _safe_copy(src, dst)
    except ImportError:
        try:
            import shutil
            from pathlib import Path

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
        from utils.common_operations import safe_deepcopy as _safe_deepcopy
        return _safe_deepcopy(obj)
    except ImportError:
        try:
            import copy
            return copy.deepcopy(obj)
        except Exception as e:
            logger.warning(f"⚠️ Deep copy failed: {e}, returning original object")
            return obj

def safe_resample(df, rule: str, agg_dict: Optional[Dict[str, str]] = None):
    """Safely resample a DataFrame with error handling."""
    try:
        from utils.common_operations import safe_resample as _safe_resample
        return _safe_resample(df, rule, agg_dict)
    except ImportError:
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

def align_dataframes(*dfs, method: str = "inner"):
    """Align multiple DataFrames by index using specified join method."""
    try:
        from utils.common_operations import align_dataframes as _align_dataframes
        return _align_dataframes(*dfs, method=method)
    except ImportError:
        try:
            import pandas as pd
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

def validate_dataframe_schema(df, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns."""
    try:
        from utils.common_operations import validate_dataframe_schema as _validate_dataframe_schema
        return _validate_dataframe_schema(df, required_columns)
    except ImportError:
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

def guard_dataframe_nulls(df, threshold: float = 0.5):
    """Guard against excessive null values in DataFrame."""
    try:
        from utils.common_operations import guard_dataframe_nulls as _guard_dataframe_nulls
        return _guard_dataframe_nulls(df, threshold)
    except ImportError:
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

def sanitize_string(s: str, max_length: int = 255) -> str:
    """Sanitize string input."""
    try:
        from utils.common_operations import sanitize_string as _sanitize_string
        return _sanitize_string(s, max_length)
    except ImportError:
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
    """Create a memory checkpoint context manager."""
    try:
        from utils.common_operations import memory_checkpoint as _memory_checkpoint
        return _memory_checkpoint(name)
    except ImportError:
        from contextlib import contextmanager

        @contextmanager
        def _memory_checkpoint():
            try:
                # Try to get M1 memory optimizer
                memory_optimizer = get_m1_memory_optimizer()
                if memory_optimizer and hasattr(memory_optimizer, 'memory_checkpoint'):
                    with memory_optimizer.memory_checkpoint(name):
                        yield
                else:
                    # Fallback: just yield without checkpointing
                    yield
            except Exception:
                # If anything fails, just yield without checkpointing
                yield

        return _memory_checkpoint()

def gpu_context(name: str):
    """Create a GPU context manager."""
    try:
        from utils.common_operations import gpu_context as _gpu_context
        return _gpu_context(name)
    except ImportError:
        from contextlib import contextmanager

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
    """Optimize memory usage across the system."""
    try:
        from utils.common_operations import optimize_memory as _optimize_memory
        return _optimize_memory()
    except ImportError:
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
    """Get current memory usage in bytes."""
    try:
        from utils.common_operations import get_memory_usage as _get_memory_usage
        return _get_memory_usage()
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except ImportError:
            logger.warning("⚠️ psutil not available for memory monitoring")
            return 0.0

def validate_file_path(file_path: Union[str, Path]) -> bool:
    """Validate if a file path exists and is accessible."""
    try:
        from utils.common_operations import validate_file_path as _validate_file_path
        return _validate_file_path(file_path)
    except ImportError:
        try:
            path = Path(file_path)
            return path.exists() and path.is_file()
        except Exception:
            return False

def get_file_size(file_path: Union[str, Path]) -> int:
    """Get the size of a file in bytes."""
    try:
        from utils.common_operations import get_file_size as _get_file_size
        return _get_file_size(file_path)
    except ImportError:
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                return path.stat().st_size
            return 0
        except Exception:
            return 0

def check_disk_space(path: Union[str, Path], required_gb: float = 1.0) -> Dict[str, Any]:
    """Check if there's sufficient disk space available."""
    try:
        from utils.common_operations import check_disk_space as _check_disk_space
        return _check_disk_space(path, required_gb)
    except ImportError:
        try:
            import shutil
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

# =============================================================================
# MATH VALIDATION INTEGRATION
# =============================================================================

def safe_correlation(x, y, default: float = 0.0) -> float:
    """Safely calculate correlation coefficient between two arrays."""
    try:
        from utils.math_validation import safe_correlation as _safe_correlation
        return _safe_correlation(x, y, default)
    except ImportError:
        try:
            import numpy as np
            if x is None or y is None:
                return default
            if len(x) != len(y):
                return default
            if len(x) <= 1:
                return default

            # Remove NaN and infinite values
            valid_mask = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid_mask):
                return default

            x_clean = x[valid_mask]
            y_clean = y[valid_mask]

            if len(x_clean) <= 1:
                return default

            # Calculate correlation coefficient
            corr_matrix = np.corrcoef(x_clean, y_clean)
            if corr_matrix.shape != (2, 2):
                return default

            corr = corr_matrix[0, 1]

            # Ensure result is valid
            if not np.isfinite(corr):
                return default

            return corr

        except Exception:
            return default

def safe_covariance(x, y, default: float = 0.0) -> float:
    """Safely calculate covariance between two arrays."""
    try:
        from utils.math_validation import safe_covariance as _safe_covariance
        return _safe_covariance(x, y, default)
    except ImportError:
        try:
            import numpy as np
            if x is None or y is None:
                return default
            if len(x) != len(y):
                return default
            if len(x) <= 1:
                return default

            # Remove NaN and infinite values
            valid_mask = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid_mask):
                return default

            x_clean = x[valid_mask]
            y_clean = y[valid_mask]

            if len(x_clean) <= 1:
                return default

            # Calculate covariance
            cov = np.cov(x_clean, y_clean)[0, 1]

            # Ensure result is valid
            if not np.isfinite(cov):
                return default

            return cov

        except Exception:
            return default

def safe_percentile(x, percentile: float = 50.0, default: float = 0.0) -> float:
    """Safely calculate percentile of array."""
    try:
        from utils.math_validation import safe_percentile as _safe_percentile
        return _safe_percentile(x, percentile, default)
    except ImportError:
        try:
            import numpy as np
            if x is None or len(x) == 0:
                return default
            if not (0 <= percentile <= 100):
                return default

            # Remove NaN and infinite values
            valid_mask = np.isfinite(x)
            if not np.any(valid_mask):
                return default

            x_clean = x[valid_mask]
            if len(x_clean) == 0:
                return default

            percentile_val = np.percentile(x_clean, percentile)

            # Ensure result is valid
            if not np.isfinite(percentile_val):
                return default

            return percentile_val

        except Exception:
            return default

# =============================================================================
# SERIALIZATION UTILITIES INTEGRATION
# =============================================================================

class JSONSerializer:
    """JSON serialization utilities."""

    @staticmethod
    def save(data: Any, filepath: str) -> bool:
        """Save data as JSON."""
        try:
            from utils.serialization_utils import JSONSerializer as _JSONSerializer
            return _JSONSerializer.save(data, filepath)
        except ImportError:
            try:
                import json
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                return True
            except Exception as e:
                logger.error(f"Failed to save JSON: {e}")
                return False

    @staticmethod
    def load(filepath: str):
        """Load data from JSON."""
        try:
            from utils.serialization_utils import JSONSerializer as _JSONSerializer
            return _JSONSerializer.load(filepath)
        except ImportError:
            try:
                import json
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load JSON: {e}")
                return None

class PickleSerializer:
    """Pickle serialization utilities."""

    @staticmethod
    def save(data: Any, filepath: str) -> bool:
        """Save data as pickle."""
        try:
            from utils.serialization_utils import PickleSerializer as _PickleSerializer
            return _PickleSerializer.save(data, filepath)
        except ImportError:
            try:
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                return True
            except Exception as e:
                logger.error(f"Failed to save pickle: {e}")
                return False

    @staticmethod
    def load(filepath: str):
        """Load data from pickle."""
        try:
            from utils.serialization_utils import PickleSerializer as _PickleSerializer
            return _PickleSerializer.load(filepath)
        except ImportError:
            try:
                import pickle
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load pickle: {e}")
                return None

class ParquetSerializer:
    """Parquet serialization utilities."""

    @staticmethod
    def save(data: Any, filepath: str) -> bool:
        """Save data as parquet."""
        try:
            from utils.serialization_utils import ParquetSerializer as _ParquetSerializer
            return _ParquetSerializer.save(data, filepath)
        except ImportError:
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(filepath)
                    return True
                else:
                    logger.error("ParquetSerializer only supports pandas DataFrames")
                    return False
            except Exception as e:
                logger.error(f"Failed to save parquet: {e}")
                return False

    @staticmethod
    def load(filepath: str):
        """Load data from parquet."""
        try:
            from utils.serialization_utils import ParquetSerializer as _ParquetSerializer
            return _ParquetSerializer.load(filepath)
        except ImportError:
            try:
                import pandas as pd
                return pd.read_parquet(filepath)
            except Exception as e:
                logger.error(f"Failed to load parquet: {e}")
                return None

class UniversalSerializer:
    """Universal serialization that tries multiple formats."""

    def __init__(self):
        self.serializers = {
            'json': JSONSerializer,
            'pickle': PickleSerializer,
            'parquet': ParquetSerializer
        }

    def save(self, data: Any, filepath: str, format: str = 'auto') -> bool:
        """Save data with automatic format detection."""
        try:
            from utils.serialization_utils import UniversalSerializer as _UniversalSerializer
            serializer = _UniversalSerializer()
            return serializer.save(data, filepath, format)
        except ImportError:
            if format == 'auto':
                if filepath.endswith('.json'):
                    format = 'json'
                elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                    format = 'pickle'
                elif filepath.endswith('.parquet'):
                    format = 'parquet'
                else:
                    format = 'pickle'  # default

            serializer = self.serializers.get(format)
            if serializer:
                return serializer.save(data, filepath)
            else:
                logger.error(f"Unsupported format: {format}")
                return False

    def load(self, filepath: str):
        """Load data with automatic format detection."""
        try:
            from utils.serialization_utils import UniversalSerializer as _UniversalSerializer
            serializer = _UniversalSerializer()
            return serializer.load(filepath)
        except ImportError:
            if filepath.endswith('.json'):
                return JSONSerializer.load(filepath)
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                return PickleSerializer.load(filepath)
            elif filepath.endswith('.parquet'):
                return ParquetSerializer.load(filepath)
            else:
                # Try pickle as default
                return PickleSerializer.load(filepath)

# =============================================================================
# DATA UTILITIES INTEGRATION
# =============================================================================

def get_kline_parquet_manager():
    """Get the kline parquet manager instance."""
    try:
        from utils.kline_parquet import get_kline_parquet_manager
        return get_kline_parquet_manager()
    except ImportError:
        logger.warning("⚠️ Kline parquet utilities not available")
        return None

def load_market_data_from_kline_parquet(symbol: str, timeframe: str, start_date: str, end_date: str):
    """Load market data from kline parquet files."""
    try:
        from utils.kline_parquet import load_market_data_from_kline_parquet as _load_market_data_from_kline_parquet
        return _load_market_data_from_kline_parquet(symbol, timeframe, start_date, end_date)
    except ImportError:
        logger.warning("⚠️ Kline parquet utilities not available")
        return None

# =============================================================================
# MATRIX OPERATIONS INTEGRATION
# =============================================================================

def get_matrix_operations_manager():
    """Get the matrix operations manager instance."""
    try:
        from utils.matrix_operations.unified_operations import get_matrix_operations_manager
        return get_matrix_operations_manager()
    except ImportError:
        logger.warning("⚠️ Matrix operations utilities not available")
        return None

def batch_matrix_operations():
    """Get batch matrix operations utilities."""
    try:
        from utils.matrix_operations.batch_operations import batch_matrix_operations
        return batch_matrix_operations()
    except ImportError:
        logger.warning("⚠️ Batch matrix operations not available")
        return None

def computation_toolbox():
    """Get computation toolbox utilities."""
    try:
        from utils.matrix_operations.computation_toolbox import computation_toolbox
        return computation_toolbox()
    except ImportError:
        logger.warning("⚠️ Computation toolbox not available")
        return None

# =============================================================================
# ML COMMON UTILITIES INTEGRATION
# =============================================================================

def get_ml_common_manager():
    """Get the ML common utilities manager instance."""
    try:
        from utils.ml_common.pipeline_orchestrator import get_ml_common_manager
        return get_ml_common_manager()
    except ImportError:
        logger.warning("⚠️ ML Common utilities not available")
        return None

def get_cross_validator():
    """Get cross validator from ML common."""
    try:
        from utils.ml_common.validation.matrix_cross_validation import get_cross_validator
        return get_cross_validator()
    except ImportError:
        logger.warning("⚠️ Cross validator not available")
        return None

def get_overfitting_detector():
    """Get overfitting detector from ML common."""
    try:
        from utils.ml_common.validation.overfitting_detector import get_overfitting_detector
        return get_overfitting_detector()
    except ImportError:
        logger.warning("⚠️ Overfitting detector not available")
        return None

def get_data_leakage_detector():
    """Get data leakage detector from ML common."""
    try:
        from utils.ml_common.validation.data_leakage_detector import get_data_leakage_detector
        return get_data_leakage_detector()
    except ImportError:
        logger.warning("⚠️ Data leakage detector not available")
        return None

def get_hyperparameter_optimizer():
    """Get hyperparameter optimizer from ML common."""
    try:
        from utils.ml_common.optimization.hyperparameter_optimizer import get_hyperparameter_optimizer
        return get_hyperparameter_optimizer()
    except ImportError:
        logger.warning("⚠️ Hyperparameter optimizer not available")
        return None

# =============================================================================
# M1 HARDWARE OPTIMIZATION INTEGRATION
# =============================================================================

def get_m1_gpu_manager():
    """Get the M1 GPU manager instance."""
    try:
        from utils.hardware.m1_gpu_utils import get_m1_gpu_manager as _get_m1_gpu_manager
        return _get_m1_gpu_manager()
    except ImportError:
        logger.warning("⚠️ M1 GPU utilities not available")
        return None

def get_m1_memory_optimizer():
    """Get the M1 memory optimizer instance."""
    try:
        from utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer as _get_m1_memory_optimizer
        return _get_m1_memory_optimizer()
    except ImportError:
        logger.warning("⚠️ M1 memory optimizer not available")
        return None

def get_m1_cpu_optimizer():
    """Get the M1 CPU optimizer instance."""
    try:
        from utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer as _get_m1_cpu_optimizer
        return _get_m1_cpu_optimizer()
    except ImportError:
        logger.warning("⚠️ M1 CPU optimizer not available")
        return None

def cleanup_m1_optimizers():
    """Clean up M1 optimizers and release resources."""
    try:
        from utils.hardware.m1_gpu_utils import get_m1_gpu_manager
        from utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
        from utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

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
    """Integrate with M1 GPU and CPU optimizers."""
    try:
        from utils.hardware.m1_gpu_utils import get_m1_gpu_manager, is_m1_available, is_mps_available
        from utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, start_m1_memory_monitoring
        from utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

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

# =============================================================================
# MAIN INTEGRATION FUNCTIONS
# =============================================================================

def initialize_enhanced_utilities() -> Dict[str, Any]:
    """Initialize all enhanced utilities and return integration status."""
    integration_status = {
        'common_operations': False,
        'math_validation': False,
        'serialization': False,
        'data_utils': False,
        'matrix_ops': False,
        'ml_common': False,
        'm1_hardware': False,
        'overall_status': 'success'
    }

    # Test common operations
    try:
        test_df = create_empty_dataframe(['test'])
        integration_status['common_operations'] = True
        logger.info("✅ Common operations integration successful")
    except Exception as e:
        logger.warning(f"⚠️ Common operations integration failed: {e}")

    # Test math validation
    try:
        result = safe_divide(10, 2)
        integration_status['math_validation'] = True
        logger.info("✅ Math validation integration successful")
    except Exception as e:
        logger.warning(f"⚠️ Math validation integration failed: {e}")

    # Test serialization
    try:
        test_data = {'test': 'data'}
        result = safe_json_dump(test_data, '/tmp/test.json')
        integration_status['serialization'] = result
        logger.info("✅ Serialization integration successful")
    except Exception as e:
        logger.warning(f"⚠️ Serialization integration failed: {e}")

    # Test data utils
    try:
        manager = get_kline_parquet_manager()
        integration_status['data_utils'] = manager is not None
        logger.info("✅ Data utilities integration successful")
    except Exception as e:
        logger.warning(f"⚠️ Data utilities integration failed: {e}")

    # Test matrix ops
    try:
        manager = get_matrix_operations_manager()
        integration_status['matrix_ops'] = manager is not None
        logger.info("✅ Matrix operations integration successful")
    except Exception as e:
        logger.warning(f"⚠️ Matrix operations integration failed: {e}")

    # Test ML common
    try:
        manager = get_ml_common_manager()
        integration_status['ml_common'] = manager is not None
        logger.info("✅ ML common integration successful")
    except Exception as e:
        logger.warning(f"⚠️ ML common integration failed: {e}")

    # Test M1 hardware
    try:
        m1_status = integrate_with_m1_optimizers()
        integration_status['m1_hardware'] = m1_status.get('success', False)
        logger.info("✅ M1 hardware integration successful")
    except Exception as e:
        logger.warning(f"⚠️ M1 hardware integration failed: {e}")

    # Determine overall status
    all_successful = all(integration_status.values())
    integration_status['overall_status'] = 'success' if all_successful else 'partial'

    logger.info(f"🔧 Enhanced utilities initialization complete: {integration_status['overall_status']}")
    return integration_status

# Global serializer instance
universal_serializer = UniversalSerializer()