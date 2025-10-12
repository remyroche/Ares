"""
Comprehensive Data Validation Utilities for Feature Lookback Optimization.

This module provides robust data validation and quality assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

from ..constants import VALIDATION_CONSTANTS, QUALITY_CONSTANTS
from ..error_handling.error_handler import safe_operation, DataValidationError


class ValidationLevel(Enum):
    """Validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CUSTOM = "custom"


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    score: float  # 0.0 to 1.0, higher is better
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


class DataValidator:
    """Comprehensive data validator for optimization operations."""
    
    def __init__(self, 
                 validation_level: ValidationLevel = ValidationLevel.STANDARD,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize data validator.
        
        Args:
            validation_level: Level of validation to perform
            logger: Logger instance
        """
        self.validation_level = validation_level
        self.logger = logger or logging.getLogger(__name__)
        self.constants = VALIDATION_CONSTANTS
        self.error_constants = None  # Using fast failing, no error constants needed
    
    @safe_operation("data validation", default_value=ValidationResult(False, 0.0, [], [], [], {}))
    def validate_dataframe(self, 
                          df: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None,
                          min_rows: Optional[int] = None) -> ValidationResult:
        """
        Validate a pandas DataFrame for optimization use.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            
        Returns:
            ValidationResult with validation details
        """
        issues = []
        warnings = []
        recommendations = []
        metadata = {}
        
        # Basic DataFrame validation
        if not isinstance(df, pd.DataFrame):
            issues.append("Input is not a pandas DataFrame")
            return ValidationResult(False, 0.0, issues, warnings, recommendations, metadata)
        
        if df.empty:
            issues.append("DataFrame is empty")
            return ValidationResult(False, 0.0, issues, warnings, recommendations, metadata)
        
        # Check minimum rows
        min_rows = min_rows or self.constants.MIN_DATA_POINTS
        if len(df) < min_rows:
            issues.append(f"Insufficient data: {len(df)} rows < {min_rows} required")
        
        # Check required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                issues.append(f"Missing required columns: {missing_columns}")
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            warnings.append(f"Completely empty columns: {empty_columns}")
        
        # Check data types
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_columns) > 0:
            warnings.append(f"Non-numeric columns found: {non_numeric_columns.tolist()}")
        
        # Check for infinite values
        inf_columns = []
        for col in numeric_columns:
            if np.isinf(df[col]).any():
                inf_columns.append(col)
        
        if inf_columns:
            issues.append(f"Columns with infinite values: {inf_columns}")
        
        # Check for excessive missing values
        missing_ratio = df.isnull().sum() / len(df)
        high_missing_columns = missing_ratio[missing_ratio > self.constants.MAX_MISSING_RATIO].index.tolist()
        
        if high_missing_columns:
            warnings.append(f"Columns with high missing ratio: {high_missing_columns}")
        
        # Calculate validation score
        score = self._calculate_validation_score(issues, warnings, len(df), len(df.columns))
        
        # Generate recommendations
        if issues:
            recommendations.append("Fix critical issues before proceeding")
        if warnings:
            recommendations.append("Consider addressing warnings for better performance")
        if len(df) < self.constants.MIN_DATA_POINTS * 2:
            recommendations.append("Consider collecting more data for better optimization results")
        
        metadata = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_columns),
            'missing_ratio': float(missing_ratio.mean()),
            'empty_columns': len(empty_columns),
            'inf_columns': len(inf_columns)
        }
        
        is_valid = len(issues) == 0
        
        return ValidationResult(is_valid, score, issues, warnings, recommendations, metadata)
    
    @safe_operation("feature validation", default_value=ValidationResult(False, 0.0, [], [], [], {}))
    def validate_feature_values(self, 
                               feature_values: np.ndarray,
                               feature_name: str = "unknown") -> ValidationResult:
        """
        Validate feature values for optimization.
        
        Args:
            feature_values: Array of feature values
            feature_name: Name of the feature for logging
            
        Returns:
            ValidationResult with validation details
        """
        issues = []
        warnings = []
        recommendations = []
        metadata = {}
        
        # Basic array validation
        if not isinstance(feature_values, np.ndarray):
            issues.append(f"Feature {feature_name}: Input is not a numpy array")
            return ValidationResult(False, 0.0, issues, warnings, recommendations, metadata)
        
        if feature_values.size == 0:
            issues.append(f"Feature {feature_name}: Array is empty")
            return ValidationResult(False, 0.0, issues, warnings, recommendations, metadata)
        
        # Check array size
        if len(feature_values) < self.constants.MIN_FEATURE_VALUES:
            issues.append(f"Feature {feature_name}: Insufficient values ({len(feature_values)} < {self.constants.MIN_FEATURE_VALUES})")
        
        if len(feature_values) > self.constants.MAX_FEATURE_VALUES:
            warnings.append(f"Feature {feature_name}: Very large array ({len(feature_values)} > {self.constants.MAX_FEATURE_VALUES})")
        
        # Check for NaN values
        nan_count = np.isnan(feature_values).sum()
        nan_ratio = nan_count / len(feature_values)
        
        if nan_ratio > self.constants.MAX_MISSING_RATIO:
            issues.append(f"Feature {feature_name}: Too many NaN values ({nan_ratio:.2%})")
        elif nan_count > 0:
            warnings.append(f"Feature {feature_name}: Contains {nan_count} NaN values")
        
        # Check for infinite values
        inf_count = np.isinf(feature_values).sum()
        if inf_count > 0:
            issues.append(f"Feature {feature_name}: Contains {inf_count} infinite values")
        
        # Check for constant values
        if len(feature_values) > 1:
            std_value = np.std(feature_values[~np.isnan(feature_values)])
            if std_value < self.constants.MIN_FEATURE_STD:
                warnings.append(f"Feature {feature_name}: Very low standard deviation ({std_value:.2e})")
                recommendations.append("Consider if this feature provides useful information")
        
        # Check for extreme values
        finite_values = feature_values[np.isfinite(feature_values)]
        if len(finite_values) > 0:
            q99 = np.percentile(finite_values, 99)
            q01 = np.percentile(finite_values, 1)
            if q99 / q01 > 1000:  # Very large range
                warnings.append(f"Feature {feature_name}: Very large value range")
        
        # Calculate validation score
        score = self._calculate_validation_score(issues, warnings, len(feature_values), 1)
        
        metadata = {
            'array_length': len(feature_values),
            'nan_count': int(nan_count),
            'nan_ratio': float(nan_ratio),
            'inf_count': int(inf_count),
            'std_value': float(std_value) if len(feature_values) > 1 else 0.0,
            'min_value': float(np.min(finite_values)) if len(finite_values) > 0 else np.nan,
            'max_value': float(np.max(finite_values)) if len(finite_values) > 0 else np.nan
        }
        
        is_valid = len(issues) == 0
        
        return ValidationResult(is_valid, score, issues, warnings, recommendations, metadata)
    
    @safe_operation("target validation", default_value=ValidationResult(False, 0.0, [], [], [], {}))
    def validate_target_values(self, 
                              target_values: np.ndarray,
                              target_name: str = "unknown") -> ValidationResult:
        """
        Validate target values for optimization.
        
        Args:
            target_values: Array of target values
            target_name: Name of the target for logging
            
        Returns:
            ValidationResult with validation details
        """
        issues = []
        warnings = []
        recommendations = []
        metadata = {}
        
        # Basic array validation
        if not isinstance(target_values, np.ndarray):
            issues.append(f"Target {target_name}: Input is not a numpy array")
            return ValidationResult(False, 0.0, issues, warnings, recommendations, metadata)
        
        if target_values.size == 0:
            issues.append(f"Target {target_name}: Array is empty")
            return ValidationResult(False, 0.0, issues, warnings, recommendations, metadata)
        
        # Check array size
        if len(target_values) < self.constants.MIN_TARGET_VALUES:
            issues.append(f"Target {target_name}: Insufficient values ({len(target_values)} < {self.constants.MIN_TARGET_VALUES})")
        
        if len(target_values) > self.constants.MAX_TARGET_VALUES:
            warnings.append(f"Target {target_name}: Very large array ({len(target_values)} > {self.constants.MAX_TARGET_VALUES})")
        
        # Check for NaN values
        nan_count = np.isnan(target_values).sum()
        nan_ratio = nan_count / len(target_values)
        
        if nan_ratio > self.constants.MAX_MISSING_RATIO:
            issues.append(f"Target {target_name}: Too many NaN values ({nan_ratio:.2%})")
        elif nan_count > 0:
            warnings.append(f"Target {target_name}: Contains {nan_count} NaN values")
        
        # Check for infinite values
        inf_count = np.isinf(target_values).sum()
        if inf_count > 0:
            issues.append(f"Target {target_name}: Contains {inf_count} infinite values")
        
        # Check for constant values
        if len(target_values) > 1:
            std_value = np.std(target_values[~np.isnan(target_values)])
            if std_value < self.constants.MIN_TARGET_STD:
                warnings.append(f"Target {target_name}: Very low standard deviation ({std_value:.2e})")
                recommendations.append("Target appears to be constant - optimization may not be meaningful")
        
        # Check for extreme values
        finite_values = target_values[np.isfinite(target_values)]
        if len(finite_values) > 0:
            q99 = np.percentile(finite_values, 99)
            q01 = np.percentile(finite_values, 1)
            if q99 / q01 > 1000:  # Very large range
                warnings.append(f"Target {target_name}: Very large value range")
        
        # Calculate validation score
        score = self._calculate_validation_score(issues, warnings, len(target_values), 1)
        
        metadata = {
            'array_length': len(target_values),
            'nan_count': int(nan_count),
            'nan_ratio': float(nan_ratio),
            'inf_count': int(inf_count),
            'std_value': float(std_value) if len(target_values) > 1 else 0.0,
            'min_value': float(np.min(finite_values)) if len(finite_values) > 0 else np.nan,
            'max_value': float(np.max(finite_values)) if len(finite_values) > 0 else np.nan
        }
        
        is_valid = len(issues) == 0
        
        return ValidationResult(is_valid, score, issues, warnings, recommendations, metadata)
    
    @safe_operation("lookback validation", default_value=ValidationResult(False, 0.0, [], [], [], {}))
    def validate_lookback_range(self, 
                               lookback_range: List[int],
                               min_lookback: int = 5,
                               max_lookback: int = 300) -> ValidationResult:
        """
        Validate lookback range for optimization.
        
        Args:
            lookback_range: List of lookback values
            min_lookback: Minimum allowed lookback
            max_lookback: Maximum allowed lookback
            
        Returns:
            ValidationResult with validation details
        """
        issues = []
        warnings = []
        recommendations = []
        metadata = {}
        
        # Basic validation
        if not isinstance(lookback_range, (list, tuple, range)):
            issues.append("Lookback range must be a list, tuple, or range")
            return ValidationResult(False, 0.0, issues, warnings, recommendations, metadata)
        
        if len(lookback_range) == 0:
            issues.append("Lookback range is empty")
            return ValidationResult(False, 0.0, issues, warnings, recommendations, metadata)
        
        # Check for valid values
        invalid_values = [x for x in lookback_range if not isinstance(x, (int, np.integer)) or x <= 0]
        if invalid_values:
            issues.append(f"Invalid lookback values: {invalid_values}")
        
        # Check range bounds
        min_val = min(lookback_range)
        max_val = max(lookback_range)
        
        if min_val < min_lookback:
            issues.append(f"Minimum lookback {min_val} < {min_lookback}")
        
        if max_val > max_lookback:
            issues.append(f"Maximum lookback {max_val} > {max_lookback}")
        
        # Check for reasonable range
        if max_val - min_val < 5:
            warnings.append("Lookback range is very narrow")
            recommendations.append("Consider expanding the lookback range for better optimization")
        
        # Check for duplicates
        if len(set(lookback_range)) != len(lookback_range):
            warnings.append("Lookback range contains duplicate values")
        
        # Check for reasonable step size
        if len(lookback_range) > 1:
            steps = [lookback_range[i+1] - lookback_range[i] for i in range(len(lookback_range)-1)]
            if len(set(steps)) == 1 and steps[0] > 20:
                warnings.append("Large step size in lookback range")
                recommendations.append("Consider smaller steps for better granularity")
        
        # Calculate validation score
        score = self._calculate_validation_score(issues, warnings, len(lookback_range), 1)
        
        metadata = {
            'range_length': len(lookback_range),
            'min_value': int(min_val),
            'max_value': int(max_val),
            'unique_values': len(set(lookback_range)),
            'step_size': int(steps[0]) if len(lookback_range) > 1 and len(set(steps)) == 1 else None
        }
        
        is_valid = len(issues) == 0
        
        return ValidationResult(is_valid, score, issues, warnings, recommendations, metadata)
    
    def _calculate_validation_score(self, 
                                  issues: List[str], 
                                  warnings: List[str], 
                                  data_size: int,
                                  complexity: int) -> float:
        """
        Calculate validation score based on issues and warnings.
        
        Args:
            issues: List of critical issues
            warnings: List of warnings
            data_size: Size of the data
            complexity: Complexity factor
            
        Returns:
            Validation score (0.0 to 1.0)
        """
        # Start with perfect score
        score = 1.0
        
        # Deduct for issues (critical)
        score -= len(issues) * 0.3
        
        # Deduct for warnings (less critical)
        score -= len(warnings) * 0.1
        
        # Bonus for good data size
        if data_size >= self.constants.MIN_DATA_POINTS * 2:
            score += 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    @safe_operation("comprehensive validation", default_value=ValidationResult(False, 0.0, [], [], [], {}))
    def validate_optimization_inputs(self, 
                                   data: pd.DataFrame,
                                   feature_names: List[str],
                                   target_column: str,
                                   lookback_range: List[int]) -> ValidationResult:
        """
        Perform comprehensive validation of all optimization inputs.
        
        Args:
            data: Input DataFrame
            feature_names: List of feature names to optimize
            target_column: Target column name
            lookback_range: List of lookback values
            
        Returns:
            Comprehensive validation result
        """
        all_issues = []
        all_warnings = []
        all_recommendations = []
        all_metadata = {}
        
        # Validate DataFrame
        df_result = self.validate_dataframe(data, required_columns=[target_column])
        all_issues.extend(df_result.issues)
        all_warnings.extend(df_result.warnings)
        all_recommendations.extend(df_result.recommendations)
        all_metadata['dataframe'] = df_result.metadata
        
        # Validate target column
        if target_column in data.columns:
            target_result = self.validate_target_values(data[target_column].values, target_column)
            all_issues.extend(target_result.issues)
            all_warnings.extend(target_result.warnings)
            all_recommendations.extend(target_result.recommendations)
            all_metadata['target'] = target_result.metadata
        else:
            all_issues.append(f"Target column '{target_column}' not found in data")
        
        # Validate feature names
        missing_features = [name for name in feature_names if name not in data.columns]
        if missing_features:
            all_issues.append(f"Missing feature columns: {missing_features}")
        
        # Validate lookback range
        lookback_result = self.validate_lookback_range(lookback_range)
        all_issues.extend(lookback_result.issues)
        all_warnings.extend(lookback_result.warnings)
        all_recommendations.extend(lookback_result.recommendations)
        all_metadata['lookback_range'] = lookback_result.metadata
        
        # Calculate overall score
        overall_score = self._calculate_validation_score(all_issues, all_warnings, len(data), len(feature_names))
        
        is_valid = len(all_issues) == 0
        
        return ValidationResult(is_valid, overall_score, all_issues, all_warnings, all_recommendations, all_metadata)


# Global validator instance
_global_validator: Optional[DataValidator] = None


def get_data_validator(validation_level: ValidationLevel = ValidationLevel.STANDARD) -> DataValidator:
    """Get the global data validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = DataValidator(validation_level)
    return _global_validator


def validate_optimization_data(data: pd.DataFrame,
                              feature_names: List[str],
                              target_column: str,
                              lookback_range: List[int],
                              validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
    """
    Convenience function for comprehensive data validation.
    
    Args:
        data: Input DataFrame
        feature_names: List of feature names to optimize
        target_column: Target column name
        lookback_range: List of lookback values
        validation_level: Level of validation to perform
        
    Returns:
        Validation result
    """
    validator = get_data_validator(validation_level)
    return validator.validate_optimization_inputs(data, feature_names, target_column, lookback_range)