"""
Common utility functions for DataFrame operations and data processing.

This module provides shared utility functions for DataFrame operations,
data validation, and common data processing utilities.
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

def safe_dataframe_operation(df: pd.DataFrame, operation: Callable, *args, **kwargs) -> pd.DataFrame:
    """Safely perform operation on DataFrame."""
    try:
        return operation(df, *args, **kwargs)
    except Exception as e:
        logger.warning(f"Error in DataFrame operation {operation.__name__}: {e}")
        return df

def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns."""
    try:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating DataFrame columns: {e}")
        return False

def safe_convert_dtypes(df: pd.DataFrame, dtype_mapping: Dict[str, str]) -> pd.DataFrame:
    """Safely convert DataFrame column dtypes."""
    try:
        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        return df
    except Exception as e:
        logger.warning(f"Error converting dtypes: {e}")
        return df

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
        logger.error(f"Error calculating data quality metrics: {e}")
        return {}

def safe_merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Safely merge two DataFrames."""
    try:
        return pd.merge(df1, df2, **kwargs)
    except Exception as e:
        logger.warning(f"Error merging DataFrames: {e}")
        return df1

def safe_groupby_operation(df: pd.DataFrame, group_cols: List[str], agg_dict: Dict[str, str]) -> pd.DataFrame:
    """Safely perform groupby operation."""
    try:
        return df.groupby(group_cols).agg(agg_dict)
    except Exception as e:
        logger.warning(f"Error in groupby operation: {e}")
        return df

def safe_apply_function(df: pd.DataFrame, func: Callable, axis: int = 0) -> pd.DataFrame:
    """Safely apply function to DataFrame."""
    try:
        return df.apply(func, axis=axis)
    except Exception as e:
        logger.warning(f"Error applying function: {e}")
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
        logger.error(f"Error creating summary statistics: {e}")
        return {}

def safe_drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Safely drop columns from DataFrame."""
    try:
        existing_columns = [col for col in columns if col in df.columns]
        return df.drop(columns=existing_columns)
    except Exception as e:
        logger.warning(f"Error dropping columns: {e}")
        return df

def safe_rename_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """Safely rename DataFrame columns."""
    try:
        return df.rename(columns=column_mapping)
    except Exception as e:
        logger.warning(f"Error renaming columns: {e}")
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
        logger.warning(f"Error converting timestamp column {column}: {e}")
        return df

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
        logger.error(f"Error getting DataFrame info: {e}")
        return {}

def safe_filter_dataframe(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Safely filter DataFrame using query condition."""
    try:
        return df.query(condition)
    except Exception as e:
        logger.warning(f"Error filtering DataFrame: {e}")
        return df

def create_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Create comprehensive data quality report."""
    try:
        report = {
            'basic_info': get_dataframe_info(df),
            'quality_metrics': calculate_data_quality_metrics(df),
            'summary_stats': create_summary_statistics(df),
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
        logger.error(f"Error creating data quality report: {e}")
        return {}


class CommonUtilities:
    """Common utilities class for shared functionality."""

    def __init__(self):
        """Initialize CommonUtilities."""
        self.logger = logging.getLogger(__name__)

    def safe_dataframe_operation(self, df: pd.DataFrame, operation: Callable, *args, **kwargs) -> pd.DataFrame:
        """Safely perform operation on DataFrame."""
        try:
            return operation(df, *args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Error in DataFrame operation {operation.__name__}: {e}")
            return df

    def validate_dataframe_columns(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that DataFrame has required columns."""
        try:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                self.logger.warning(f"Missing required columns: {missing_columns}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error validating DataFrame columns: {e}")
            return False

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic summary statistics for DataFrame."""
        try:
            return {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'null_counts': df.isnull().sum().to_dict()
            }
        except Exception as e:
            self.logger.error(f"Error getting data summary: {e}")
            return {}

    def safe_convert_dtypes(self, df: pd.DataFrame, dtype_mapping: Dict[str, str]) -> pd.DataFrame:
        """Safely convert DataFrame column dtypes."""
        try:
            for col, dtype in dtype_mapping.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
            return df
        except Exception as e:
            self.logger.warning(f"Error converting dtypes: {e}")
            return df

    def join_paths(self, *paths):
        """Join paths using os.path.join."""
        import os
        return os.path.join(*paths)
    
    def file_exists(self, path):
        """Check if file exists."""
        import os
        return os.path.isfile(path)
    
    def directory_exists(self, path):
        """Check if directory exists."""
        import os
        return os.path.isdir(path)
    
    def glob_files(self, pattern):
        """Glob files matching pattern."""
        from pathlib import Path
        return list(Path().glob(pattern))
    
    def get_file_size(self, path):
        """Get file size in bytes."""
        import os
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    
    def get_file_extension(self, path):
        """Get file extension."""
        from pathlib import Path
        return Path(path).suffix
