"""
Common Utilities Module

This module provides commonly used utility functions that complement the common_operations module.
It focuses on data processing, validation, and transformation utilities.
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)


def safe_dataframe_operation(df: pd.DataFrame, operation: str, **kwargs) -> Any:
    """
    Safely perform operations on a DataFrame with error handling.
    
    Args:
        df: DataFrame to operate on
        operation: Name of the operation to perform
        **kwargs: Additional arguments for the operation
        
    Returns:
        Result of the operation or None if failed
    """
    try:
        if operation == 'dropna':
            return df.dropna(**kwargs)
        elif operation == 'fillna':
            return df.fillna(**kwargs)
        elif operation == 'sort_values':
            return df.sort_values(**kwargs)
        elif operation == 'reset_index':
            return df.reset_index(**kwargs)
        elif operation == 'copy':
            return df.copy(**kwargs)
        else:
            logger.warning(f"Unknown operation: {operation}")
            return df
    except Exception as e:
        logger.error(f"Error performing {operation} on DataFrame: {e}")
        return df


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that a DataFrame has all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    if df is None or df.empty:
        return False, required_columns
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns


def safe_convert_dtypes(df: pd.DataFrame, dtype_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Safely convert DataFrame column dtypes.
    
    Args:
        df: DataFrame to convert
        dtype_mapping: Dictionary mapping column names to target dtypes
        
    Returns:
        DataFrame with converted dtypes
    """
    try:
        result_df = df.copy()
        for column, dtype in dtype_mapping.items():
            if column in result_df.columns:
                result_df[column] = result_df[column].astype(dtype)
        return result_df
    except Exception as e:
        logger.error(f"Error converting dtypes: {e}")
        return df


def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive data quality metrics.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing quality metrics
    """
    try:
        metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df) * 100) if len(df) > 0 else 0,
            'dtypes': df.dtypes.to_dict()
        }
        
        # Calculate numeric column statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            metrics['numeric_stats'] = df[numeric_columns].describe().to_dict()
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating data quality metrics: {e}")
        return {}


def safe_merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Safely merge two DataFrames with error handling.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        **kwargs: Additional arguments for merge operation
        
    Returns:
        Merged DataFrame or empty DataFrame if failed
    """
    try:
        if df1 is None or df1.empty:
            return df2 if df2 is not None else pd.DataFrame()
        if df2 is None or df2.empty:
            return df1
        
        return pd.merge(df1, df2, **kwargs)
    except Exception as e:
        logger.error(f"Error merging DataFrames: {e}")
        return pd.DataFrame()


def safe_groupby_operation(df: pd.DataFrame, groupby_columns: List[str], agg_dict: Dict[str, str]) -> pd.DataFrame:
    """
    Safely perform groupby operations on a DataFrame.
    
    Args:
        df: DataFrame to group
        groupby_columns: Columns to group by
        agg_dict: Dictionary of column:aggregation_function mappings
        
    Returns:
        Grouped DataFrame
    """
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Check if all groupby columns exist
        missing_cols = [col for col in groupby_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing groupby columns: {missing_cols}")
            return pd.DataFrame()
        
        return df.groupby(groupby_columns).agg(agg_dict).reset_index()
    except Exception as e:
        logger.error(f"Error in groupby operation: {e}")
        return pd.DataFrame()


def safe_apply_function(df: pd.DataFrame, func: callable, axis: int = 0, **kwargs) -> pd.DataFrame:
    """
    Safely apply a function to a DataFrame.
    
    Args:
        df: DataFrame to apply function to
        func: Function to apply
        axis: Axis to apply function along
        **kwargs: Additional arguments for the function
        
    Returns:
        DataFrame with function applied
    """
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        
        return df.apply(func, axis=axis, **kwargs)
    except Exception as e:
        logger.error(f"Error applying function to DataFrame: {e}")
        return df


def create_summary_statistics(df: pd.DataFrame, numeric_only: bool = True) -> Dict[str, Any]:
    """
    Create summary statistics for a DataFrame.
    
    Args:
        df: DataFrame to summarize
        numeric_only: Whether to include only numeric columns
        
    Returns:
        Dictionary containing summary statistics
    """
    try:
        if df is None or df.empty:
            return {}
        
        if numeric_only:
            summary_df = df.describe(include=[np.number])
        else:
            summary_df = df.describe(include='all')
        
        return summary_df.to_dict()
    except Exception as e:
        logger.error(f"Error creating summary statistics: {e}")
        return {}


def safe_drop_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Safely drop columns from a DataFrame.
    
    Args:
        df: DataFrame to modify
        columns_to_drop: List of column names to drop
        
    Returns:
        DataFrame with columns dropped
    """
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Only drop columns that actually exist
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        if existing_columns:
            return df.drop(columns=existing_columns)
        return df
    except Exception as e:
        logger.error(f"Error dropping columns: {e}")
        return df


def safe_rename_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Safely rename columns in a DataFrame.
    
    Args:
        df: DataFrame to modify
        column_mapping: Dictionary mapping old names to new names
        
    Returns:
        DataFrame with renamed columns
    """
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        
        return df.rename(columns=column_mapping)
    except Exception as e:
        logger.error(f"Error renaming columns: {e}")
        return df


def validate_timestamp_column(df: pd.DataFrame, timestamp_column: str = 'timestamp') -> Tuple[bool, str]:
    """
    Validate that a timestamp column is properly formatted.
    
    Args:
        df: DataFrame to validate
        timestamp_column: Name of the timestamp column
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if df is None or df.empty:
            return False, "DataFrame is empty"
        
        if timestamp_column not in df.columns:
            return False, f"Timestamp column '{timestamp_column}' not found"
        
        # Check if column can be converted to datetime
        try:
            pd.to_datetime(df[timestamp_column])
            return True, ""
        except Exception as e:
            return False, f"Cannot convert timestamp column to datetime: {e}"
            
    except Exception as e:
        return False, f"Error validating timestamp column: {e}"


def safe_timestamp_conversion(df: pd.DataFrame, timestamp_column: str = 'timestamp') -> pd.DataFrame:
    """
    Safely convert a timestamp column to datetime.
    
    Args:
        df: DataFrame to modify
        timestamp_column: Name of the timestamp column
        
    Returns:
        DataFrame with converted timestamp column
    """
    try:
        if df is None or df.empty or timestamp_column not in df.columns:
            return df
        
        df_copy = df.copy()
        df_copy[timestamp_column] = pd.to_datetime(df_copy[timestamp_column])
        return df_copy
    except Exception as e:
        logger.error(f"Error converting timestamp column: {e}")
        return df


def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing DataFrame information
    """
    try:
        if df is None:
            return {"error": "DataFrame is None"}
        
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "null_counts": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "index_type": str(type(df.index)),
            "is_empty": df.empty
        }
        
        # Add timestamp info if available
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if timestamp_cols:
            info["timestamp_columns"] = timestamp_cols
            for col in timestamp_cols:
                try:
                    info[f"{col}_range"] = {
                        "min": str(df[col].min()),
                        "max": str(df[col].max())
                    }
                except Exception:
                    pass
        
        return info
    except Exception as e:
        logger.error(f"Error getting DataFrame info: {e}")
        return {"error": str(e)}


def safe_filter_dataframe(df: pd.DataFrame, filter_condition: str) -> pd.DataFrame:
    """
    Safely filter a DataFrame using a query condition.
    
    Args:
        df: DataFrame to filter
        filter_condition: Query condition string
        
    Returns:
        Filtered DataFrame
    """
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        
        return df.query(filter_condition)
    except Exception as e:
        logger.error(f"Error filtering DataFrame: {e}")
        return df


def create_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a comprehensive data quality report.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing quality report
    """
    try:
        if df is None or df.empty:
            return {"status": "empty", "message": "DataFrame is empty or None"}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "basic_info": get_dataframe_info(df),
            "quality_metrics": calculate_data_quality_metrics(df),
            "issues": [],
            "recommendations": []
        }
        
        # Check for common issues
        null_percentages = report["quality_metrics"].get("null_percentages", {})
        high_null_cols = [col for col, pct in null_percentages.items() if pct > 50]
        if high_null_cols:
            report["issues"].append(f"High null percentage in columns: {high_null_cols}")
            report["recommendations"].append("Consider dropping or imputing high-null columns")
        
        duplicate_pct = report["quality_metrics"].get("duplicate_percentage", 0)
        if duplicate_pct > 10:
            report["issues"].append(f"High duplicate percentage: {duplicate_pct:.2f}%")
            report["recommendations"].append("Consider removing duplicate rows")
        
        # Check for timestamp issues
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        for col in timestamp_cols:
            is_valid, error = validate_timestamp_column(df, col)
            if not is_valid:
                report["issues"].append(f"Timestamp column '{col}' issue: {error}")
                report["recommendations"].append(f"Fix timestamp column '{col}' format")
        
        report["status"] = "success"
        return report
        
    except Exception as e:
        logger.error(f"Error creating data quality report: {e}")
        return {"status": "error", "message": str(e)}


# Export commonly used functions
__all__ = [
    'safe_dataframe_operation',
    'validate_dataframe_columns', 
    'safe_convert_dtypes',
    'calculate_data_quality_metrics',
    'safe_merge_dataframes',
    'safe_groupby_operation',
    'safe_apply_function',
    'create_summary_statistics',
    'safe_drop_columns',
    'safe_rename_columns',
    'validate_timestamp_column',
    'safe_timestamp_conversion',
    'get_dataframe_info',
    'safe_filter_dataframe',
    'create_data_quality_report'
]