"""
Common utility functions for DataFrame operations and data processing.

This module provides shared utility functions for DataFrame operations,
data validation, and common data processing utilities with comprehensive
input validation and memory management.
"""

import logging
import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

# Import input validation
from .input_validation import (
    validate_dataframe, validate_array, validate_numeric, validate_string,
    validate_list, validate_path, validate_function, ValidationResult
)

# Import memory management
from .memory_management import (
    memory_managed, MemoryStrategy, get_memory_manager, force_cleanup
)

# Enhanced hardware optimization imports
try:
    from src.utils.hardware.optimization_decorators import (
        smart_cache, auto_optimize, memory_efficient, performance_tracked
    )
    from src.utils.hardware.memory_optimized_decorators import (
        memory_optimized, comprehensive_memory_optimization, MemoryOptimizationLevel
    )
    from src.utils.hardware.integrated_hardware_manager import (
        get_integrated_hardware_manager, WorkloadType
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

@memory_managed(MemoryStrategy.MODERATE)
def safe_dataframe_operation(df: pd.DataFrame, operation: Callable, *args, **kwargs) -> pd.DataFrame:
    """
    Safely perform operation on DataFrame with comprehensive validation.
    
    Args:
        df: DataFrame to operate on
        operation: Function to apply to DataFrame
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation
        
    Returns:
        DataFrame after operation or original DataFrame if error occurs
        
    Raises:
        ValueError: If validation fails
    """
    # Validate inputs
    df_result = validate_dataframe(df, "dataframe", allow_empty=False)
    if not df_result.is_valid:
        raise ValueError(f"DataFrame validation failed: {', '.join(df_result.errors)}")
    
    func_result = validate_function(operation, "operation")
    if not func_result.is_valid:
        raise ValueError(f"Function validation failed: {', '.join(func_result.errors)}")
    
    try:
        return operation(df, *args, **kwargs)
    except Exception as e:
        logger.warning(f"Error in DataFrame operation {operation.__name__}: {e}")
        return df

@memory_managed(MemoryStrategy.CONSERVATIVE)
def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame has required columns with comprehensive validation.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if all required columns are present, False otherwise
        
    Raises:
        ValueError: If validation fails
    """
    # Validate inputs
    df_result = validate_dataframe(df, "dataframe", allow_empty=False)
    if not df_result.is_valid:
        raise ValueError(f"DataFrame validation failed: {', '.join(df_result.errors)}")
    
    cols_result = validate_list(required_columns, "required_columns", 
                               min_length=1, element_type=str)
    if not cols_result.is_valid:
        raise ValueError(f"Required columns validation failed: {', '.join(cols_result.errors)}")
    
    try:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating DataFrame columns: {e}")
        return False

@memory_managed(MemoryStrategy.MODERATE)
def safe_convert_dtypes(df: pd.DataFrame, dtype_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Safely convert DataFrame column dtypes with comprehensive validation.
    
    Args:
        df: DataFrame to convert
        dtype_mapping: Dictionary mapping column names to dtypes
        
    Returns:
        DataFrame with converted dtypes or original DataFrame if error occurs
        
    Raises:
        ValueError: If validation fails
    """
    # Validate inputs
    df_result = validate_dataframe(df, "dataframe", allow_empty=False)
    if not df_result.is_valid:
        raise ValueError(f"DataFrame validation failed: {', '.join(df_result.errors)}")
    
    if not isinstance(dtype_mapping, dict):
        raise ValueError("dtype_mapping must be a dictionary")
    
    try:
        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        return df
    except Exception as e:
        logger.warning(f"Error converting dtypes: {e}")
        return df

def analyze_nan_values_detailed(data: Union[pd.DataFrame, np.ndarray], feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of NaN values in the dataset.

    Args:
        data: DataFrame or numpy array to analyze
        feature_names: Optional list of feature names for numpy arrays

    Returns:
        Dictionary with detailed NaN analysis results
    """
    try:
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=feature_names)
        else:
            df = data.copy()

        # Basic NaN statistics
        total_cells = df.size
        total_nans = df.isnull().sum().sum()
        nan_percentage = (total_nans / total_cells) * 100

        # Feature-wise NaN analysis
        feature_nan_counts = df.isnull().sum()
        feature_nan_percentages = (feature_nan_counts / len(df)) * 100

        # Features with most NaN values
        features_with_nans = feature_nan_counts[feature_nan_counts > 0].sort_values(ascending=False)
        top_nan_features = features_with_nans.head(10).to_dict()

        # Row-wise NaN analysis
        row_nan_counts = df.isnull().sum(axis=1)
        rows_with_nans = (row_nan_counts > 0).sum()
        rows_with_all_nans = (row_nan_counts == df.shape[1]).sum()

        # Rows with most NaN values
        rows_with_most_nans = row_nan_counts[row_nan_counts > 0].sort_values(ascending=False)
        top_nan_rows = rows_with_most_nans.head(10).to_dict()

        # Complete rows (no NaN values)
        complete_rows = (row_nan_counts == 0).sum()
        complete_row_percentage = (complete_rows / len(df)) * 100

        # Complete features (no NaN values)
        complete_features = (feature_nan_counts == 0).sum()
        complete_feature_percentage = (complete_features / df.shape[1]) * 100

        # NaN patterns analysis
        nan_patterns = {}
        for threshold in [0.1, 0.25, 0.5, 0.75, 0.9]:
            features_above_threshold = (feature_nan_percentages >= threshold * 100).sum()
            nan_patterns[f"features_with_{int(threshold*100)}%_or_more_nans"] = features_above_threshold

        # Correlation between NaN patterns
        nan_matrix = df.isnull().astype(int)
        if nan_matrix.shape[1] > 1:
            nan_correlation = nan_matrix.corr()
            # Find features with similar NaN patterns (high correlation)
            similar_nan_patterns = []
            for i in range(len(nan_correlation.columns)):
                for j in range(i+1, len(nan_correlation.columns)):
                    corr_val = nan_correlation.iloc[i, j]
                    if corr_val > 0.7:  # High correlation threshold
                        similar_nan_patterns.append({
                            'feature1': nan_correlation.columns[i],
                            'feature2': nan_correlation.columns[j],
                            'correlation': corr_val
                        })
        else:
            similar_nan_patterns = []

        return {
            'total_cells': total_cells,
            'total_nans': int(total_nans),
            'nan_percentage': round(nan_percentage, 2),
            'total_features': df.shape[1],
            'total_rows': len(df),
            'features_with_nans': int(features_with_nans.count()),
            'rows_with_nans': int(rows_with_nans),
            'rows_with_all_nans': int(rows_with_all_nans),
            'complete_rows': int(complete_rows),
            'complete_row_percentage': round(complete_row_percentage, 2),
            'complete_features': int(complete_features),
            'complete_feature_percentage': round(complete_feature_percentage, 2),
            'top_nan_features': top_nan_features,
            'top_nan_rows': top_nan_rows,
            'nan_patterns': nan_patterns,
            'similar_nan_patterns': similar_nan_patterns[:10],  # Top 10 similar patterns
            'feature_nan_percentages': feature_nan_percentages.to_dict(),
            'row_nan_counts': row_nan_counts.to_dict()
        }

    except Exception as e:
        logger.error(f"Error analyzing NaN values: {e}")
        return {
            'error': str(e),
            'total_nans': 0,
            'nan_percentage': 0
        }

def format_nan_analysis_report(analysis_results: Dict[str, Any], prefix: str = "") -> str:
    """
    Format detailed NaN analysis results into a readable report.

    Args:
        analysis_results: Results from analyze_nan_values_detailed
        prefix: Optional prefix for log messages

    Returns:
        Formatted report string
    """
    try:
        if 'error' in analysis_results:
            return f"{prefix}❌ Error in NaN analysis: {analysis_results['error']}"

        report_lines = []
        report_lines.append(f"{prefix}📊 NaN Analysis Summary:")
        report_lines.append(f"{prefix}  • Total cells: {analysis_results['total_cells']:,}")
        report_lines.append(f"{prefix}  • Total NaN values: {analysis_results['total_nans']:,}")
        report_lines.append(f"{prefix}  • NaN percentage: {analysis_results['nan_percentage']:.2f}%")
        report_lines.append(f"{prefix}  • Dataset shape: {analysis_results['total_rows']} rows × {analysis_results['total_features']} features")

        report_lines.append(f"\n{prefix}📈 Row Analysis:")
        report_lines.append(f"{prefix}  • Rows with NaN values: {analysis_results['rows_with_nans']:,} ({analysis_results['rows_with_nans']/analysis_results['total_rows']*100:.1f}%)")
        report_lines.append(f"{prefix}  • Complete rows (no NaN): {analysis_results['complete_rows']:,} ({analysis_results['complete_row_percentage']:.1f}%)")
        report_lines.append(f"{prefix}  • Rows with all NaN values: {analysis_results['rows_with_all_nans']:,}")

        report_lines.append(f"\n{prefix}🔍 Feature Analysis:")
        report_lines.append(f"{prefix}  • Features with NaN values: {analysis_results['features_with_nans']:,} ({analysis_results['features_with_nans']/analysis_results['total_features']*100:.1f}%)")
        report_lines.append(f"{prefix}  • Complete features (no NaN): {analysis_results['complete_features']:,} ({analysis_results['complete_feature_percentage']:.1f}%)")

        # Top problematic features
        if analysis_results['top_nan_features']:
            report_lines.append(f"\n{prefix}⚠️ Top Features with Most NaN Values:")
            for feature, count in list(analysis_results['top_nan_features'].items())[:5]:
                percentage = (count / analysis_results['total_rows']) * 100
                report_lines.append(f"{prefix}  • {feature}: {count:,} NaN values ({percentage:.1f}%)")

        # Top problematic rows
        if analysis_results['top_nan_rows']:
            report_lines.append(f"\n{prefix}⚠️ Rows with Most NaN Values:")
            for row_idx, count in list(analysis_results['top_nan_rows'].items())[:5]:
                percentage = (count / analysis_results['total_features']) * 100
                report_lines.append(f"{prefix}  • Row {row_idx}: {count:,} NaN values ({percentage:.1f}%)")

        # NaN patterns
        if analysis_results['nan_patterns']:
            report_lines.append(f"\n{prefix}📊 NaN Distribution Patterns:")
            for pattern, count in analysis_results['nan_patterns'].items():
                if count > 0:
                    report_lines.append(f"{prefix}  • {pattern}: {count} features")

        # Similar NaN patterns
        if analysis_results['similar_nan_patterns']:
            report_lines.append(f"\n{prefix}🔗 Features with Similar NaN Patterns:")
            for pattern in analysis_results['similar_nan_patterns'][:3]:
                report_lines.append(f"{prefix}  • {pattern['feature1']} ↔ {pattern['feature2']}: {pattern['correlation']:.3f} correlation")

        return "\n".join(report_lines)

    except Exception as e:
        logger.error(f"Error formatting NaN analysis report: {e}")
        return f"{prefix}❌ Error formatting NaN analysis report: {e}"

@memory_managed(MemoryStrategy.AGGRESSIVE)
def calculate_data_quality_metrics(df: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
    """
    Calculate data quality metrics for DataFrame or numpy array with comprehensive validation.
    
    Args:
        df: DataFrame or numpy array to analyze
        
    Returns:
        Dictionary with data quality metrics
        
    Raises:
        ValueError: If validation fails
    """
    # Validate input
    if isinstance(df, np.ndarray):
        arr_result = validate_array(df, "data", allow_empty=False)
        if not arr_result.is_valid:
            raise ValueError(f"Array validation failed: {', '.join(arr_result.errors)}")
    else:
        df_result = validate_dataframe(df, "dataframe", allow_empty=False)
        if not df_result.is_valid:
            raise ValueError(f"DataFrame validation failed: {', '.join(df_result.errors)}")
    
    try:
        # Convert numpy array to DataFrame if needed
        if isinstance(df, np.ndarray):
            # If it's a 2D array, assume first row contains column names or create generic names
            if df.ndim == 2:
                if df.shape[1] <= 50:  # Reasonable number of features
                    # Try to infer column names from the array
                    df = pd.DataFrame(df)
                else:
                    logger.warning("Large numpy array detected, using generic column names")
                    df = pd.DataFrame(df)
            else:
                logger.error("1D numpy array not supported for data quality metrics")
                return {'error': '1D numpy array not supported'}

        # Ensure we have a DataFrame at this point
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Unsupported data type: {type(df)}")
            return {'error': f'Unsupported data type: {type(df)}'}

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

@memory_efficient(memory_threshold_mb=50.0, auto_cleanup=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
@performance_tracked(log_performance=True, track_memory=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
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
        return os.path.isfile(path)

    def directory_exists(self, path):
        """Check if directory exists."""
        return os.path.isdir(path)

    def glob_files(self, pattern):
        """Glob files matching pattern."""
        return list(Path().glob(pattern))

    def get_file_size(self, path):
        """Get file size in bytes."""
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    def get_file_extension(self, path):
        """Get file extension."""
        return Path(path).suffix
