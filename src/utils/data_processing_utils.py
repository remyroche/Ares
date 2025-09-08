"""
Enhanced Data Processing Utilities

This module provides comprehensive data processing utilities for DataFrames,
data validation, cleaning, and transformation operations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class DataQualityLevel(Enum):
    """Data quality levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

@dataclass
class DataQualityIssue:
    """Data quality issue representation."""
    issue_type: str
    level: DataQualityLevel
    description: str
    affected_rows: Optional[List[int]] = None
    affected_columns: Optional[List[str]] = None
    suggested_fix: Optional[str] = None

@dataclass
class DataQualityReport:
    """Data quality report."""
    issues: List[DataQualityIssue]
    summary: Dict[str, Any]
    recommendations: List[str]

class DataFrameValidator:
    """Comprehensive DataFrame validation utility."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataFrame validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or {}
        self.default_thresholds = {
            'max_null_ratio': 0.5,
            'min_rows': 1,
            'max_duplicate_ratio': 0.1,
            'max_outlier_ratio': 0.05
        }
        self.thresholds = {**self.default_thresholds, **self.config.get('thresholds', {})}
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          schema: Optional[Dict[str, Any]] = None) -> DataQualityReport:
        """
        Validate DataFrame for data quality issues.
        
        Args:
            df: DataFrame to validate
            schema: Expected schema definition
            
        Returns:
            Data quality report
        """
        issues = []
        
        # Basic structure validation
        issues.extend(self._validate_structure(df))
        
        # Data type validation
        if schema:
            issues.extend(self._validate_schema(df, schema))
        
        # Data quality validation
        issues.extend(self._validate_data_quality(df))
        
        # Statistical validation
        issues.extend(self._validate_statistics(df))
        
        # Generate summary and recommendations
        summary = self._generate_summary(df, issues)
        recommendations = self._generate_recommendations(issues)
        
        return DataQualityReport(
            issues=issues,
            summary=summary,
            recommendations=recommendations
        )
    
    def _validate_structure(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Validate DataFrame structure."""
        issues = []
        
        # Check if DataFrame is empty
        if df.empty:
            issues.append(DataQualityIssue(
                issue_type="empty_dataframe",
                level=DataQualityLevel.CRITICAL,
                description="DataFrame is empty",
                suggested_fix="Check data source and loading process"
            ))
        
        # Check minimum rows
        if len(df) < self.thresholds['min_rows']:
            issues.append(DataQualityIssue(
                issue_type="insufficient_rows",
                level=DataQualityLevel.WARNING,
                description=f"DataFrame has only {len(df)} rows, minimum recommended: {self.thresholds['min_rows']}",
                suggested_fix="Collect more data or adjust minimum threshold"
            ))
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            issues.append(DataQualityIssue(
                issue_type="duplicate_columns",
                level=DataQualityLevel.CRITICAL,
                description=f"Duplicate columns found: {duplicate_cols}",
                affected_columns=duplicate_cols,
                suggested_fix="Remove or rename duplicate columns"
            ))
        
        return issues
    
    def _validate_schema(self, df: pd.DataFrame, schema: Dict[str, Any]) -> List[DataQualityIssue]:
        """Validate DataFrame against schema."""
        issues = []
        
        required_columns = schema.get('required_columns', [])
        optional_columns = schema.get('optional_columns', [])
        data_types = schema.get('data_types', {})
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(DataQualityIssue(
                issue_type="missing_columns",
                level=DataQualityLevel.CRITICAL,
                description=f"Missing required columns: {missing_columns}",
                affected_columns=missing_columns,
                suggested_fix="Add missing columns or update schema"
            ))
        
        # Check data types
        for col, expected_type in data_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not self._is_compatible_type(actual_type, expected_type):
                    issues.append(DataQualityIssue(
                        issue_type="type_mismatch",
                        level=DataQualityLevel.WARNING,
                        description=f"Column '{col}' has type {actual_type}, expected {expected_type}",
                        affected_columns=[col],
                        suggested_fix=f"Convert column '{col}' to {expected_type}"
                    ))
        
        return issues
    
    def _validate_data_quality(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Validate data quality."""
        issues = []
        
        # Check for null values
        null_ratios = df.isnull().sum() / len(df)
        high_null_cols = null_ratios[null_ratios > self.thresholds['max_null_ratio']].index.tolist()
        
        if high_null_cols:
            issues.append(DataQualityIssue(
                issue_type="high_null_ratio",
                level=DataQualityLevel.WARNING,
                description=f"Columns with high null ratio: {high_null_cols}",
                affected_columns=high_null_cols,
                suggested_fix="Consider imputation or removal of high-null columns"
            ))
        
        # Check for duplicates
        duplicate_ratio = df.duplicated().sum() / len(df)
        if duplicate_ratio > self.thresholds['max_duplicate_ratio']:
            issues.append(DataQualityIssue(
                issue_type="high_duplicate_ratio",
                level=DataQualityLevel.WARNING,
                description=f"High duplicate ratio: {duplicate_ratio:.2%}",
                suggested_fix="Remove duplicates or investigate data source"
            ))
        
        # Check for infinite values
        inf_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            issues.append(DataQualityIssue(
                issue_type="infinite_values",
                level=DataQualityLevel.WARNING,
                description=f"Columns with infinite values: {inf_cols}",
                affected_columns=inf_cols,
                suggested_fix="Replace infinite values with NaN or appropriate values"
            ))
        
        return issues
    
    def _validate_statistics(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Validate statistical properties."""
        issues = []
        
        # Check for constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            issues.append(DataQualityIssue(
                issue_type="constant_columns",
                level=DataQualityLevel.INFO,
                description=f"Constant columns: {constant_cols}",
                affected_columns=constant_cols,
                suggested_fix="Consider removing constant columns"
            ))
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_cols = []
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_ratio = outliers / len(df)
            
            if outlier_ratio > self.thresholds['max_outlier_ratio']:
                outlier_cols.append(col)
        
        if outlier_cols:
            issues.append(DataQualityIssue(
                issue_type="high_outlier_ratio",
                level=DataQualityLevel.INFO,
                description=f"Columns with high outlier ratio: {outlier_cols}",
                affected_columns=outlier_cols,
                suggested_fix="Investigate outliers or apply outlier treatment"
            ))
        
        return issues
    
    def _is_compatible_type(self, actual_type: str, expected_type: str) -> bool:
        """Check if actual type is compatible with expected type."""
        type_mapping = {
            'int64': ['int', 'integer', 'int64', 'int32', 'int16', 'int8'],
            'float64': ['float', 'float64', 'float32', 'float16'],
            'object': ['str', 'string', 'object'],
            'bool': ['bool', 'boolean'],
            'datetime64[ns]': ['datetime', 'datetime64', 'timestamp']
        }
        
        for compatible_types in type_mapping.values():
            if actual_type in compatible_types and expected_type in compatible_types:
                return True
        
        return actual_type == expected_type
    
    def _generate_summary(self, df: pd.DataFrame, issues: List[DataQualityIssue]) -> Dict[str, Any]:
        """Generate validation summary."""
        critical_issues = [i for i in issues if i.level == DataQualityLevel.CRITICAL]
        warning_issues = [i for i in issues if i.level == DataQualityLevel.WARNING]
        info_issues = [i for i in issues if i.level == DataQualityLevel.INFO]
        
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_issues': len(issues),
            'critical_issues': len(critical_issues),
            'warning_issues': len(warning_issues),
            'info_issues': len(info_issues),
            'data_quality_score': max(0, 100 - len(critical_issues) * 20 - len(warning_issues) * 5),
            'null_ratio': df.isnull().sum().sum() / (len(df) * len(df.columns)),
            'duplicate_ratio': df.duplicated().sum() / len(df)
        }
    
    def _generate_recommendations(self, issues: List[DataQualityIssue]) -> List[str]:
        """Generate recommendations based on issues."""
        recommendations = []
        
        issue_types = [issue.issue_type for issue in issues]
        
        if 'empty_dataframe' in issue_types:
            recommendations.append("Investigate data source and loading process")
        
        if 'missing_columns' in issue_types:
            recommendations.append("Update data collection to include all required columns")
        
        if 'high_null_ratio' in issue_types:
            recommendations.append("Implement data imputation strategy")
        
        if 'high_duplicate_ratio' in issue_types:
            recommendations.append("Add deduplication step to data pipeline")
        
        if 'type_mismatch' in issue_types:
            recommendations.append("Add data type conversion to preprocessing pipeline")
        
        return recommendations

class DataFrameCleaner:
    """DataFrame cleaning and preprocessing utility."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataFrame cleaner.
        
        Args:
            config: Cleaning configuration
        """
        self.config = config or {}
    
    def clean_dataframe(self, df: pd.DataFrame, 
                       cleaning_steps: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Clean DataFrame using specified steps.
        
        Args:
            df: DataFrame to clean
            cleaning_steps: List of cleaning steps to apply
            
        Returns:
            Cleaned DataFrame
        """
        if cleaning_steps is None:
            cleaning_steps = ['remove_duplicates', 'handle_nulls', 'fix_types', 'remove_constant_columns']
        
        cleaned_df = df.copy()
        
        for step in cleaning_steps:
            if hasattr(self, f'_{step}'):
                cleaned_df = getattr(self, f'_{step}')(cleaned_df)
                logger.debug(f"Applied cleaning step: {step}")
            else:
                logger.warning(f"Unknown cleaning step: {step}")
        
        return cleaned_df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_rows = len(df)
        cleaned_df = df.drop_duplicates()
        removed_rows = initial_rows - len(cleaned_df)
        
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} duplicate rows")
        
        return cleaned_df
    
    def _handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values."""
        strategy = self.config.get('null_strategy', 'drop')
        
        if strategy == 'drop':
            cleaned_df = df.dropna()
        elif strategy == 'fill':
            method = self.config.get('fill_method', 'forward')
            if method == 'forward':
                cleaned_df = df.fillna(method='ffill')
            elif method == 'backward':
                cleaned_df = df.fillna(method='bfill')
            elif method == 'mean':
                cleaned_df = df.fillna(df.select_dtypes(include=[np.number]).mean())
            elif method == 'median':
                cleaned_df = df.fillna(df.select_dtypes(include=[np.number]).median())
            else:
                cleaned_df = df.fillna(method=method)
        elif strategy == 'interpolate':
            cleaned_df = df.interpolate()
        else:
            cleaned_df = df
        
        return cleaned_df
    
    def _fix_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types."""
        cleaned_df = df.copy()
        
        # Convert object columns to appropriate types
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            # Try to convert to numeric
            try:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore')
            except:
                pass
            
            # Try to convert to datetime
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='ignore')
            except:
                pass
        
        return cleaned_df
    
    def _remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove constant columns."""
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            logger.info(f"Removing constant columns: {constant_cols}")
            return df.drop(columns=constant_cols)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cleaned_df = df.copy()
        
        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
            if outliers.any():
                logger.info(f"Removing {outliers.sum()} outliers from column '{col}'")
                cleaned_df = cleaned_df[~outliers]
        
        return cleaned_df

class DataFrameTransformer:
    """DataFrame transformation utility."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataFrame transformer.
        
        Args:
            config: Transformation configuration
        """
        self.config = config or {}
    
    def transform_dataframe(self, df: pd.DataFrame, 
                           transformations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply transformations to DataFrame.
        
        Args:
            df: DataFrame to transform
            transformations: List of transformation specifications
            
        Returns:
            Transformed DataFrame
        """
        transformed_df = df.copy()
        
        for transformation in transformations:
            transform_type = transformation.get('type')
            params = transformation.get('params', {})
            
            if hasattr(self, f'_transform_{transform_type}'):
                transformed_df = getattr(self, f'_transform_{transform_type}')(transformed_df, params)
                logger.debug(f"Applied transformation: {transform_type}")
            else:
                logger.warning(f"Unknown transformation type: {transform_type}")
        
        return transformed_df
    
    def _transform_rename_columns(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Rename columns."""
        column_mapping = params.get('mapping', {})
        return df.rename(columns=column_mapping)
    
    def _transform_drop_columns(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Drop columns."""
        columns_to_drop = params.get('columns', [])
        return df.drop(columns=columns_to_drop)
    
    def _transform_select_columns(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Select specific columns."""
        columns_to_select = params.get('columns', [])
        return df[columns_to_select]
    
    def _transform_add_column(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Add new column."""
        column_name = params.get('name')
        column_value = params.get('value')
        column_function = params.get('function')
        
        if column_function:
            df[column_name] = column_function(df)
        else:
            df[column_name] = column_value
        
        return df
    
    def _transform_apply_function(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply function to DataFrame."""
        function = params.get('function')
        axis = params.get('axis', 0)
        
        if function:
            return df.apply(function, axis=axis)
        
        return df

# Convenience functions
def validate_dataframe(df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> DataQualityReport:
    """Convenience function to validate DataFrame."""
    validator = DataFrameValidator()
    return validator.validate_dataframe(df, schema)

def clean_dataframe(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Convenience function to clean DataFrame."""
    cleaner = DataFrameCleaner(kwargs)
    return cleaner.clean_dataframe(df)

def transform_dataframe(df: pd.DataFrame, transformations: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convenience function to transform DataFrame."""
    transformer = DataFrameTransformer()
    return transformer.transform_dataframe(df, transformations)

def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive DataFrame information."""
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'null_ratios': (df.isnull().sum() / len(df)).to_dict(),
        'memory_usage': df.memory_usage(deep=True).to_dict(),
        'total_memory': df.memory_usage(deep=True).sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
    }

__all__ = [
    'DataQualityLevel',
    'DataQualityIssue',
    'DataQualityReport',
    'DataFrameValidator',
    'DataFrameCleaner',
    'DataFrameTransformer',
    'validate_dataframe',
    'clean_dataframe',
    'transform_dataframe',
    'get_dataframe_info'
]