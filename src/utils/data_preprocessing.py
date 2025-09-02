"""
Data Preprocessing Utilities for Ares Trading System
Provides functions for regularizing timestamps, handling data quality issues,
and preparing data for feature engineering.
"""

from datetime import timedelta
import warnings
from typing import Any, Optional, Union
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import system_logger

# Try to import pandas and numpy
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    NUMPY_AVAILABLE = False
    pd = None
    np = None

warnings.filterwarnings("ignore")

def regularize_timestamps(
    data,
    expected_interval: Optional[timedelta] = None,
    tolerance_seconds: float = 1.0,
    method: str = "forward_fill"
):
    """
    Regularize timestamps in a DataFrame to ensure consistent intervals.
    
    Args:
        data: Input DataFrame with timestamp index or timestamp column
        expected_interval: Expected time interval between rows
        tolerance_seconds: Tolerance for irregular intervals in seconds
        method: Method for handling missing timestamps ('forward_fill', 'interpolate', 'drop')
        
    Returns:
        DataFrame with regularized timestamps
    """
    if not PANDAS_AVAILABLE:
        logger = system_logger.getChild("DataPreprocessing")
        logger.error("pandas not available, cannot regularize timestamps")
        return data
    
    logger = system_logger.getChild("DataPreprocessing")
    
    try:
        if data is None or data.empty:
            return data

        # Make a copy to avoid modifying original data
        processed_data = data.copy()

        # Ensure timestamp is the index
        if "timestamp" in processed_data.columns:
            processed_data = processed_data.set_index("timestamp")
        elif not isinstance(processed_data.index, pd.DatetimeIndex):
            logger.warning("⚠️ No timestamp column found, cannot regularize intervals")
            return data

        # Sort by timestamp
        processed_data = processed_data.sort_index()

        # Check for irregular intervals
        time_diffs = processed_data.index.to_series().diff().dropna()
        if len(time_diffs) == 0:
            return data

        # Calculate expected interval if not provided
        if expected_interval is None:
            # Fallback implementation for expected_interval
            expected_interval = (
                time_diffs.mode().iloc[0]
                if len(time_diffs.mode()) > 0
                else time_diffs.median()
            )

        # Identify irregular intervals
        irregular_mask = abs(time_diffs - expected_interval) > timedelta(
            seconds=tolerance_seconds
        )
        irregular_ratio = irregular_mask.sum() / len(time_diffs)

        if irregular_ratio > 0.0001:  # If more than 0.01% irregular intervals (more sensitive)
            logger.info(
                f"🔄 Regularizing timestamps (irregular ratio: {irregular_ratio:.3f})",
            )

            # Create a regular timestamp index
            start_time = processed_data.index.min()
            end_time = processed_data.index.max()

            # Determine the frequency string based on expected interval
            freq = _get_frequency_string(expected_interval)

            # Create regular timestamp index
            regular_index = pd.date_range(start=start_time, end=end_time, freq=freq)

            # Reindex data to regular intervals
            if method == "forward_fill":
                processed_data = processed_data.reindex(regular_index, method="ffill")
            elif method == "interpolate":
                processed_data = processed_data.reindex(regular_index).interpolate(
                    method="time",
                )
            elif method == "drop":
                processed_data = processed_data.reindex(regular_index)
            else:
                processed_data = processed_data.reindex(regular_index, method="ffill")

            # Drop rows that are completely NaN (before the first valid data point)
            processed_data = processed_data.dropna(how="all")

            logger.info(
                f"✅ Regularized timestamps: {len(processed_data)} rows with {freq} intervals",
            )

        return processed_data

    except Exception as e:
        logger.exception(f"🚨 Error regularizing timestamps: {e}")
        return data

def _get_frequency_string(interval: timedelta) -> str:
    """
    Convert timedelta to pandas frequency string.
    
    Args:
        interval: Time interval as timedelta
        
    Returns:
        Pandas frequency string
    """
    total_seconds = interval.total_seconds()
    
    if total_seconds <= 1:
        return "S"  # 1 second
    elif total_seconds <= 60:
        return f"{int(total_seconds)}S"
    elif total_seconds <= 3600:
        minutes = int(total_seconds / 60)
        return f"{minutes}T"
    elif total_seconds <= 86400:
        hours = int(total_seconds / 3600)
        return f"{hours}H"
    else:
        days = int(total_seconds / 86400)
        return f"{days}D"

def handle_missing_values(
    data,
    method: str = "interpolate",
    limit: Optional[int] = None,
    fill_value: Optional[Union[str, float, int]] = None
):
    """
    Handle missing values in a DataFrame.
    
    Args:
        data: Input DataFrame
        method: Method for handling missing values ('interpolate', 'forward_fill', 'backward_fill', 'drop')
        limit: Maximum number of consecutive missing values to fill
        fill_value: Value to use for filling missing values
        
    Returns:
        DataFrame with handled missing values
    """
    if not PANDAS_AVAILABLE:
        logger = system_logger.getChild("DataPreprocessing")
        logger.error("pandas not available, cannot handle missing values")
        return data
    
    logger = system_logger.getChild("DataPreprocessing")
    
    try:
        if data is None or data.empty:
            return data
            
        processed_data = data.copy()
        
        # Count missing values before processing
        missing_before = processed_data.isnull().sum().sum()
        
        if missing_before == 0:
            logger.info("✅ No missing values found")
            return processed_data
            
        logger.info(f"🔍 Found {missing_before} missing values, applying {method} method")
        
        if method == "interpolate":
            processed_data = processed_data.interpolate(limit=limit)
        elif method == "forward_fill":
            processed_data = processed_data.fillna(method='ffill', limit=limit)
        elif method == "backward_fill":
            processed_data = processed_data.fillna(method='bfill', limit=limit)
        elif method == "drop":
            processed_data = processed_data.dropna()
        elif fill_value is not None:
            processed_data = processed_data.fillna(fill_value)
        else:
            logger.warning(f"⚠️ Unknown method '{method}', using forward fill")
            processed_data = processed_data.fillna(method='ffill')
            
        # Count missing values after processing
        missing_after = processed_data.isnull().sum().sum()
        logger.info(f"✅ Reduced missing values from {missing_before} to {missing_after}")
        
        return processed_data
        
    except Exception as e:
        logger.exception(f"🚨 Error handling missing values: {e}")
        return data

def remove_outliers(
    data,
    columns: Optional[list] = None,
    method: str = "iqr",
    threshold: float = 1.5
):
    """
    Remove outliers from DataFrame using specified method.
    
    Args:
        data: Input DataFrame
        columns: Columns to process (None for all numeric columns)
        method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    if not PANDAS_AVAILABLE or not NUMPY_AVAILABLE:
        logger = system_logger.getChild("DataPreprocessing")
        logger.error("pandas or numpy not available, cannot remove outliers")
        return data
    
    logger = system_logger.getChild("DataPreprocessing")
    
    try:
        if data is None or data.empty:
            return data
            
        processed_data = data.copy()
        
        # Select columns to process
        if columns is None:
            columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            logger.warning("⚠️ No numeric columns found for outlier removal")
            return processed_data
            
        initial_rows = len(processed_data)
        
        for column in columns:
            if column not in processed_data.columns:
                continue
                
            if method == "iqr":
                Q1 = processed_data[column].quantile(0.25)
                Q3 = processed_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers_mask = (processed_data[column] < lower_bound) | (processed_data[column] > upper_bound)
                
            elif method == "zscore":
                z_scores = np.abs((processed_data[column] - processed_data[column].mean()) / processed_data[column].std())
                outliers_mask = z_scores > threshold
                
            else:
                logger.warning(f"⚠️ Unknown method '{method}', using IQR")
                Q1 = processed_data[column].quantile(0.25)
                Q3 = processed_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers_mask = (processed_data[column] < lower_bound) | (processed_data[column] > upper_bound)
            
            outliers_count = outliers_mask.sum()
            if outliers_count > 0:
                processed_data = processed_data[~outliers_mask]
                logger.info(f"🔍 Removed {outliers_count} outliers from column '{column}' using {method}")
        
        final_rows = len(processed_data)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            logger.info(f"✅ Removed {removed_rows} outlier rows ({removed_rows/initial_rows*100:.1f}%)")
        else:
            logger.info("✅ No outliers found")
            
        return processed_data
        
    except Exception as e:
        logger.exception(f"🚨 Error removing outliers: {e}")
        return data

def normalize_features(
    data,
    columns: Optional[list] = None,
    method: str = "standard",
    return_scaler: bool = False
):
    """
    Normalize features in a DataFrame.
    
    Args:
        data: Input DataFrame
        columns: Columns to normalize (None for all numeric columns)
        method: Normalization method ('standard', 'minmax', 'robust')
        return_scaler: Whether to return the scaler object
        
    Returns:
        Normalized DataFrame and optionally the scaler
    """
    if not PANDAS_AVAILABLE or not NUMPY_AVAILABLE:
        logger = system_logger.getChild("DataPreprocessing")
        logger.error("pandas or numpy not available, cannot normalize features")
        return data if not return_scaler else (data, None)
    
    logger = system_logger.getChild("DataPreprocessing")
    
    try:
        if data is None or data.empty:
            return data if not return_scaler else (data, None)
            
        processed_data = data.copy()
        
        # Select columns to normalize
        if columns is None:
            columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            logger.warning("⚠️ No numeric columns found for normalization")
            return processed_data if not return_scaler else (processed_data, None)
            
        logger.info(f"🔧 Normalizing {len(columns)} columns using {method} method")
        
        # Apply normalization
        if method == "standard":
            # Z-score normalization
            for column in columns:
                if column in processed_data.columns:
                    mean_val = processed_data[column].mean()
                    std_val = processed_data[column].std()
                    if std_val > 0:
                        processed_data[column] = (processed_data[column] - mean_val) / std_val
                        
        elif method == "minmax":
            # Min-max normalization to [0, 1]
            for column in columns:
                if column in processed_data.columns:
                    min_val = processed_data[column].min()
                    max_val = processed_data[column].max()
                    if max_val > min_val:
                        processed_data[column] = (processed_data[column] - min_val) / (max_val - min_val)
                        
        elif method == "robust":
            # Robust normalization using median and IQR
            for column in columns:
                if column in processed_data.columns:
                    median_val = processed_data[column].median()
                    Q1 = processed_data[column].quantile(0.25)
                    Q3 = processed_data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        processed_data[column] = (processed_data[column] - median_val) / IQR
                        
        else:
            logger.warning(f"⚠️ Unknown method '{method}', using standard normalization")
            for column in columns:
                if column in processed_data.columns:
                    mean_val = processed_data[column].mean()
                    std_val = processed_data[column].std()
                    if std_val > 0:
                        processed_data[column] = (processed_data[column] - mean_val) / std_val
        
        logger.info("✅ Feature normalization completed")
        
        if return_scaler:
            # Return a simple scaler info dict since we're not using sklearn
            scaler_info = {
                "method": method,
                "columns": columns,
                "applied": True
            }
            return processed_data, scaler_info
        else:
            return processed_data
            
    except Exception as e:
        logger.exception(f"🚨 Error normalizing features: {e}")
        return data if not return_scaler else (data, None)

def validate_data_quality(
    data,
    check_missing: bool = True,
    check_duplicates: bool = True,
    check_types: bool = True,
    check_ranges: bool = True
):
    """
    Validate data quality and return a report.
    
    Args:
        data: Input DataFrame
        check_missing: Whether to check for missing values
        check_duplicates: Whether to check for duplicate rows
        check_types: Whether to check data types
        check_ranges: Whether to check value ranges
        
    Returns:
        Dictionary containing data quality report
    """
    if not PANDAS_AVAILABLE:
        logger = system_logger.getChild("DataPreprocessing")
        logger.error("pandas not available, cannot validate data quality")
        return {"error": "pandas not available"}
    
    logger = system_logger.getChild("DataPreprocessing")
    
    try:
        if data is None or data.empty:
            return {"error": "Data is None or empty"}
            
        report = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check missing values
        if check_missing:
            missing_counts = data.isnull().sum()
            total_missing = missing_counts.sum()
            if total_missing > 0:
                report["issues"].append(f"Found {total_missing} missing values")
                high_missing_cols = missing_counts[missing_counts > len(data) * 0.1]
                if not high_missing_cols.empty:
                    report["warnings"].append(f"High missing ratio in columns: {list(high_missing_cols.index)}")
            else:
                report["recommendations"].append("No missing values found")
                
        # Check duplicates
        if check_duplicates:
            duplicate_count = data.duplicated().sum()
            if duplicate_count > 0:
                report["issues"].append(f"Found {duplicate_count} duplicate rows")
            else:
                report["recommendations"].append("No duplicate rows found")
                
        # Check data types
        if check_types:
            numeric_cols = data.select_dtypes(include=[np.number]).columns if NUMPY_AVAILABLE else []
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            
            report["data_types"] = {
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols),
                "other": len(data.columns) - len(numeric_cols) - len(categorical_cols)
            }
            
        # Check value ranges for numeric columns
        if check_ranges and len(numeric_cols) > 0:
            for col in numeric_cols[:5]:  # Check first 5 numeric columns
                if col in data.columns:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        report["warnings"].append(f"Column '{col}': min={col_data.min():.4f}, max={col_data.max():.4f}")
                        
        # Overall quality score
        total_issues = len(report["issues"])
        total_warnings = len(report["warnings"])
        
        if total_issues == 0 and total_warnings == 0:
            report["quality_score"] = "EXCELLENT"
        elif total_issues == 0:
            report["quality_score"] = "GOOD"
        elif total_issues <= 2:
            report["quality_score"] = "FAIR"
        else:
            report["quality_score"] = "POOR"
            
        logger.info(f"✅ Data quality validation completed: {report['quality_score']}")
        return report
        
    except Exception as e:
        logger.exception(f"🚨 Error validating data quality: {e}")
        return {"error": str(e)}
