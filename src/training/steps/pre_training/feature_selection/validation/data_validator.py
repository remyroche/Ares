"""
Data validation utilities for feature selection.

This module provides comprehensive data validation capabilities
for feature selection operations including data quality checks,
schema validation, and temporal alignment verification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import warnings

from src.utils.tprint import tprint_debug, tprint_info, tprint_warning, tprint_success


@dataclass
class DataValidationResult:
    """Result of data validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_metrics: Dict[str, Any]
    validated_data: Optional[pd.DataFrame] = None


@dataclass
class DataQualityMetrics:
    """Data quality metrics for validation."""
    n_samples: int
    n_features: int
    missing_ratio: float
    duplicate_ratio: float
    variance_ratio: float
    correlation_ratio: float
    temporal_consistency: float
    data_quality_score: float


class DataValidator:
    """Data validator for feature selection operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("DataValidator")
        
        # Validation thresholds
        self.thresholds = {
            'max_missing_ratio': self.config.get('max_missing_ratio', 0.1),
            'max_duplicate_ratio': self.config.get('max_duplicate_ratio', 0.05),
            'min_variance_ratio': self.config.get('min_variance_ratio', 0.01),
            'max_correlation_ratio': self.config.get('max_correlation_ratio', 0.95),
            'min_temporal_consistency': self.config.get('min_temporal_consistency', 0.8)
        }
    
    def validate_data(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> DataValidationResult:
        """
        Validate data for feature selection with fast fail on critical errors.
        
        Args:
            data: Feature matrix to validate
            target: Optional target variable
            
        Returns:
            DataValidationResult with validation results
        """
        tprint_info(f"🔍 Validating data: {data.shape}")
        tprint_debug(f"   📊 Data type: {type(data)}")
        tprint_debug(f"   📊 Target type: {type(target) if target is not None else 'None'}")
        
        errors = []
        warnings = []
        quality_metrics = {}
        
        try:
            # FAST FAIL: Basic data structure validation - fail immediately on critical errors
            tprint_debug("🔍 Performing fast fail data structure validation")
            structure_errors = self._validate_data_structure(data)
            errors.extend(structure_errors)
            
            if structure_errors:
                tprint_error(f"❌ FAST FAIL: Critical data structure errors detected: {len(structure_errors)}")
                for i, error in enumerate(structure_errors, 1):
                    tprint_error(f"   {i}. {error}")
                return DataValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    quality_metrics=quality_metrics
                )
            
            tprint_success("   ✅ Data structure validation passed")
            
            # FAST FAIL: Target validation if provided - fail immediately on critical errors
            if target is not None:
                tprint_debug("🔍 Performing fast fail target validation")
                target_errors, target_warnings = self._validate_target(target, data)
                errors.extend(target_errors)
                warnings.extend(target_warnings)
                
                if target_errors:
                    tprint_error(f"❌ FAST FAIL: Critical target validation errors detected: {len(target_errors)}")
                    for i, error in enumerate(target_errors, 1):
                        tprint_error(f"   {i}. {error}")
                    return DataValidationResult(
                        is_valid=False,
                        errors=errors,
                        warnings=warnings,
                        quality_metrics=quality_metrics
                    )
                
                tprint_success("   ✅ Target validation passed")
                if target_warnings:
                    tprint_warning(f"   ⚠️ Target warnings: {len(target_warnings)}")
                    for i, warning in enumerate(target_warnings, 1):
                        tprint_warning(f"   {i}. {warning}")
            
            # Calculate quality metrics with extensive logging
            tprint_debug("📊 Calculating data quality metrics")
            quality_metrics = self._calculate_quality_metrics(data, target)
            tprint_debug(f"   📊 Quality score: {quality_metrics.get('data_quality_score', 0.0):.3f}")
            
            # Validate data quality with detailed logging
            tprint_debug("🔍 Validating data quality thresholds")
            quality_errors, quality_warnings = self._validate_data_quality(data, quality_metrics)
            errors.extend(quality_errors)
            warnings.extend(quality_warnings)
            
            if quality_errors:
                tprint_error(f"❌ Data quality validation failed: {len(quality_errors)} errors")
                for i, error in enumerate(quality_errors, 1):
                    tprint_error(f"   {i}. {error}")
            else:
                tprint_success("   ✅ Data quality validation passed")
            
            if quality_warnings:
                tprint_warning(f"   ⚠️ Data quality warnings: {len(quality_warnings)}")
                for i, warning in enumerate(quality_warnings, 1):
                    tprint_warning(f"   {i}. {warning}")
            
            # Validate temporal consistency with detailed logging
            tprint_debug("🔍 Validating temporal consistency")
            temporal_errors, temporal_warnings = self._validate_temporal_consistency(data)
            errors.extend(temporal_errors)
            warnings.extend(temporal_warnings)
            
            if temporal_errors:
                tprint_error(f"❌ Temporal consistency validation failed: {len(temporal_errors)} errors")
                for i, error in enumerate(temporal_errors, 1):
                    tprint_error(f"   {i}. {error}")
            else:
                tprint_success("   ✅ Temporal consistency validation passed")
            
            if temporal_warnings:
                tprint_warning(f"   ⚠️ Temporal consistency warnings: {len(temporal_warnings)}")
                for i, warning in enumerate(temporal_warnings, 1):
                    tprint_warning(f"   {i}. {warning}")
            
            # Clean data if validation passes
            cleaned_data = None
            if not errors:
                tprint_debug("🧹 Cleaning validated data")
                cleaned_data = self._clean_data(data)
                tprint_success("   ✅ Data cleaning completed")
            else:
                tprint_warning("   ⚠️ Skipping data cleaning due to validation errors")
            
            is_valid = len(errors) == 0
            
            if is_valid:
                tprint_success(f"✅ Data validation PASSED - {len(warnings)} warnings generated")
                if warnings:
                    tprint_warning(f"   ⚠️ Warning summary:")
                    for i, warning in enumerate(warnings, 1):
                        tprint_warning(f"   {i}. {warning}")
            else:
                tprint_error(f"❌ Data validation FAILED - {len(errors)} errors, {len(warnings)} warnings")
                tprint_error(f"   ❌ Error summary:")
                for i, error in enumerate(errors, 1):
                    tprint_error(f"   {i}. {error}")
            
            return DataValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                quality_metrics=quality_metrics,
                validated_data=cleaned_data
            )
            
        except Exception as e:
            error_msg = f"Data validation failed with exception: {e}"
            tprint_error(f"❌ {error_msg}")
            tprint_debug(f"   🔍 Exception type: {type(e).__name__}")
            tprint_debug(f"   🔍 Exception details: {str(e)}")
            return DataValidationResult(
                is_valid=False,
                errors=[error_msg],
                warnings=warnings,
                quality_metrics=quality_metrics
            )
    
    def _validate_data_structure(self, data: pd.DataFrame) -> List[str]:
        """Validate basic data structure with detailed logging."""
        errors = []
        
        tprint_debug("   🔍 Checking data emptiness")
        # Check if data is empty
        if data.empty:
            error_msg = "Data is empty"
            tprint_error(f"   ❌ {error_msg}")
            errors.append(error_msg)
            return errors
        tprint_debug("   ✅ Data is not empty")
        
        tprint_debug("   🔍 Checking data dimensions")
        # Check if data has features
        if data.shape[1] == 0:
            error_msg = "Data has no features"
            tprint_error(f"   ❌ {error_msg}")
            errors.append(error_msg)
        else:
            tprint_debug(f"   ✅ Data has {data.shape[1]} features")
        
        # Check if data has samples
        if data.shape[0] == 0:
            error_msg = "Data has no samples"
            tprint_error(f"   ❌ {error_msg}")
            errors.append(error_msg)
        else:
            tprint_debug(f"   ✅ Data has {data.shape[0]} samples")
        
        tprint_debug("   🔍 Checking data types")
        # Check for numeric data
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        
        tprint_debug(f"   📊 Numeric columns: {len(numeric_cols)}")
        tprint_debug(f"   📊 Non-numeric columns: {len(non_numeric_cols)}")
        
        if len(numeric_cols) == 0:
            error_msg = "Data contains no numeric features"
            tprint_error(f"   ❌ {error_msg}")
            errors.append(error_msg)
        elif len(numeric_cols) < data.shape[1] * 0.8:  # Less than 80% numeric
            error_msg = f"Data contains too many non-numeric features ({len(non_numeric_cols)}/{data.shape[1]} = {len(non_numeric_cols)/data.shape[1]:.1%})"
            tprint_error(f"   ❌ {error_msg}")
            errors.append(error_msg)
        else:
            tprint_debug(f"   ✅ Data has sufficient numeric features ({len(numeric_cols)}/{data.shape[1]} = {len(numeric_cols)/data.shape[1]:.1%})")
        
        tprint_debug("   🔍 Checking for infinite values")
        # Check for infinite values
        if not numeric_cols.empty:
            inf_mask = np.isinf(data[numeric_cols])
            inf_count = inf_mask.sum().sum()
            if inf_count > 0:
                error_msg = f"Data contains {inf_count} infinite values"
                tprint_error(f"   ❌ {error_msg}")
                errors.append(error_msg)
            else:
                tprint_debug("   ✅ No infinite values found")
        else:
            tprint_debug("   ⚠️ Skipping infinite value check (no numeric columns)")
        
        return errors
    
    def _calculate_quality_metrics(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        tprint_debug("📊 Calculating data quality metrics")
        
        # Basic metrics
        n_samples, n_features = data.shape
        missing_ratio = data.isnull().sum().sum() / (n_samples * n_features)
        duplicate_ratio = data.duplicated().sum() / n_samples
        
        # Variance metrics
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            variances = numeric_data.var()
            variance_ratio = variances.mean() / (variances.std() + 1e-10)
        else:
            variance_ratio = 0.0
        
        # Correlation metrics
        if not numeric_data.empty and numeric_data.shape[1] > 1:
            corr_matrix = numeric_data.corr().abs()
            # Remove diagonal and get upper triangle
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_ratio = (upper_triangle > 0.95).sum().sum() / (n_features * (n_features - 1) / 2)
        else:
            high_corr_ratio = 0.0
        
        # Temporal consistency (simplified)
        temporal_consistency = 1.0  # Default to perfect if no temporal validation needed
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            missing_ratio, duplicate_ratio, variance_ratio, high_corr_ratio, temporal_consistency
        )
        
        metrics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'missing_ratio': missing_ratio,
            'duplicate_ratio': duplicate_ratio,
            'variance_ratio': variance_ratio,
            'correlation_ratio': high_corr_ratio,
            'temporal_consistency': temporal_consistency,
            'data_quality_score': quality_score
        }
        
        tprint_debug(f"   📊 Quality score: {quality_score:.3f}")
        return metrics
    
    def _calculate_quality_score(
        self, 
        missing_ratio: float, 
        duplicate_ratio: float, 
        variance_ratio: float, 
        correlation_ratio: float, 
        temporal_consistency: float
    ) -> float:
        """Calculate overall data quality score."""
        # Weighted combination of quality metrics
        weights = {
            'missing': 0.3,
            'duplicate': 0.2,
            'variance': 0.2,
            'correlation': 0.2,
            'temporal': 0.1
        }
        
        # Convert ratios to scores (higher is better)
        missing_score = max(0, 1 - missing_ratio / self.thresholds['max_missing_ratio'])
        duplicate_score = max(0, 1 - duplicate_ratio / self.thresholds['max_duplicate_ratio'])
        variance_score = min(1, variance_ratio / self.thresholds['min_variance_ratio'])
        correlation_score = max(0, 1 - correlation_ratio / self.thresholds['max_correlation_ratio'])
        temporal_score = temporal_consistency
        
        # Calculate weighted average
        quality_score = (
            weights['missing'] * missing_score +
            weights['duplicate'] * duplicate_score +
            weights['variance'] * variance_score +
            weights['correlation'] * correlation_score +
            weights['temporal'] * temporal_score
        )
        
        return quality_score
    
    def _validate_data_quality(self, data: pd.DataFrame, metrics: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate data quality based on metrics."""
        errors = []
        warnings = []
        
        # Check missing data
        if metrics['missing_ratio'] > self.thresholds['max_missing_ratio']:
            errors.append(f"Missing data ratio ({metrics['missing_ratio']:.3f}) exceeds threshold ({self.thresholds['max_missing_ratio']})")
        elif metrics['missing_ratio'] > self.thresholds['max_missing_ratio'] * 0.5:
            warnings.append(f"Missing data ratio ({metrics['missing_ratio']:.3f}) is high")
        
        # Check duplicate data
        if metrics['duplicate_ratio'] > self.thresholds['max_duplicate_ratio']:
            errors.append(f"Duplicate data ratio ({metrics['duplicate_ratio']:.3f}) exceeds threshold ({self.thresholds['max_duplicate_ratio']})")
        elif metrics['duplicate_ratio'] > self.thresholds['max_duplicate_ratio'] * 0.5:
            warnings.append(f"Duplicate data ratio ({metrics['duplicate_ratio']:.3f}) is high")
        
        # Check variance
        if metrics['variance_ratio'] < self.thresholds['min_variance_ratio']:
            errors.append(f"Variance ratio ({metrics['variance_ratio']:.3f}) below threshold ({self.thresholds['min_variance_ratio']})")
        elif metrics['variance_ratio'] < self.thresholds['min_variance_ratio'] * 2:
            warnings.append(f"Variance ratio ({metrics['variance_ratio']:.3f}) is low")
        
        # Check correlation
        if metrics['correlation_ratio'] > self.thresholds['max_correlation_ratio']:
            errors.append(f"High correlation ratio ({metrics['correlation_ratio']:.3f}) exceeds threshold ({self.thresholds['max_correlation_ratio']})")
        elif metrics['correlation_ratio'] > self.thresholds['max_correlation_ratio'] * 0.8:
            warnings.append(f"Correlation ratio ({metrics['correlation_ratio']:.3f}) is high")
        
        # Check overall quality score
        if metrics['data_quality_score'] < 0.5:
            errors.append(f"Overall data quality score ({metrics['data_quality_score']:.3f}) is too low")
        elif metrics['data_quality_score'] < 0.7:
            warnings.append(f"Overall data quality score ({metrics['data_quality_score']:.3f}) is moderate")
        
        return errors, warnings
    
    def _validate_target(self, target: pd.Series, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate target variable with detailed logging."""
        errors = []
        warnings = []
        
        tprint_debug("   🔍 Validating target variable")
        tprint_debug(f"   📊 Target shape: {target.shape}")
        tprint_debug(f"   📊 Target dtype: {target.dtype}")
        tprint_debug(f"   📊 Target name: {target.name}")
        
        # Check length match
        tprint_debug("   🔍 Checking target-data length match")
        if len(target) != len(data):
            error_msg = f"Target length ({len(target)}) doesn't match data length ({len(data)})"
            tprint_error(f"   ❌ {error_msg}")
            errors.append(error_msg)
            return errors, warnings
        tprint_debug(f"   ✅ Target and data lengths match: {len(target)}")
        
        # Check for missing values
        tprint_debug("   🔍 Checking for missing values in target")
        if target.isnull().any():
            missing_count = target.isnull().sum()
            missing_ratio = missing_count / len(target)
            error_msg = f"Target contains {missing_count} missing values ({missing_ratio:.1%})"
            tprint_error(f"   ❌ {error_msg}")
            errors.append(error_msg)
        else:
            tprint_debug("   ✅ No missing values in target")
        
        # Check for infinite values
        tprint_debug("   🔍 Checking for infinite values in target")
        if np.isinf(target).any():
            inf_count = np.isinf(target).sum()
            error_msg = f"Target contains {inf_count} infinite values"
            tprint_error(f"   ❌ {error_msg}")
            errors.append(error_msg)
        else:
            tprint_debug("   ✅ No infinite values in target")
        
        # Check data type
        tprint_debug("   🔍 Checking target data type")
        if not pd.api.types.is_numeric_dtype(target):
            warning_msg = "Target is not numeric, may need encoding"
            tprint_warning(f"   ⚠️ {warning_msg}")
            warnings.append(warning_msg)
        else:
            tprint_debug("   ✅ Target is numeric")
        
        # Check variance
        tprint_debug("   🔍 Checking target variance")
        target_var = target.var()
        tprint_debug(f"   📊 Target variance: {target_var:.6f}")
        
        if target_var == 0:
            error_msg = "Target has zero variance"
            tprint_error(f"   ❌ {error_msg}")
            errors.append(error_msg)
        elif target_var < 1e-10:
            warning_msg = f"Target has very low variance ({target_var:.2e})"
            tprint_warning(f"   ⚠️ {warning_msg}")
            warnings.append(warning_msg)
        else:
            tprint_debug(f"   ✅ Target has sufficient variance: {target_var:.6f}")
        
        # Additional target statistics
        tprint_debug("   📊 Target statistics:")
        tprint_debug(f"   📊 Min: {target.min():.6f}")
        tprint_debug(f"   📊 Max: {target.max():.6f}")
        tprint_debug(f"   📊 Mean: {target.mean():.6f}")
        tprint_debug(f"   📊 Std: {target.std():.6f}")
        
        return errors, warnings
    
    def _validate_temporal_consistency(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate temporal consistency if applicable."""
        errors = []
        warnings = []
        
        # Check if data has datetime index
        if isinstance(data.index, pd.DatetimeIndex):
            # Check for gaps in time series
            time_diffs = data.index.to_series().diff()
            if time_diffs.isnull().sum() > 1:  # More than one gap
                warnings.append("Time series has gaps")
            
            # Check for duplicate timestamps
            if data.index.duplicated().any():
                errors.append("Time series has duplicate timestamps")
            
            # Check for time order
            if not data.index.is_monotonic_increasing:
                warnings.append("Time series is not in chronological order")
        
        return errors, warnings
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data based on validation results."""
        tprint_debug("🧹 Cleaning data")
        
        cleaned_data = data.copy()
        
        # Remove rows with all NaN values
        cleaned_data = cleaned_data.dropna(how='all')
        
        # Remove columns with all NaN values
        cleaned_data = cleaned_data.dropna(axis=1, how='all')
        
        # Fill remaining NaN values with median for numeric columns
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_data[col].isnull().any():
                median_val = cleaned_data[col].median()
                cleaned_data[col] = cleaned_data[col].fillna(median_val)
        
        # Remove duplicate rows
        cleaned_data = cleaned_data.drop_duplicates()
        
        tprint_debug(f"   🧹 Cleaned data: {data.shape} -> {cleaned_data.shape}")
        return cleaned_data
    
    def validate_feature_selection_input(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        feature_names: Optional[List[str]] = None
    ) -> DataValidationResult:
        """Validate input for feature selection operations."""
        tprint_info("🔍 Validating feature selection input")
        
        # Validate feature matrix
        X_result = self.validate_data(X)
        if not X_result.is_valid:
            return X_result
        
        # Validate target
        y_result = self.validate_data(pd.DataFrame({'target': y}))
        if not y_result.is_valid:
            return DataValidationResult(
                is_valid=False,
                errors=y_result.errors,
                warnings=y_result.warnings,
                quality_metrics={}
            )
        
        # Validate feature names if provided
        if feature_names is not None:
            if len(feature_names) != X.shape[1]:
                return DataValidationResult(
                    is_valid=False,
                    errors=[f"Feature names length ({len(feature_names)}) doesn't match data columns ({X.shape[1]})"],
                    warnings=[],
                    quality_metrics={}
                )
        
        # Combine results
        combined_errors = X_result.errors + y_result.errors
        combined_warnings = X_result.warnings + y_result.warnings
        combined_metrics = {**X_result.quality_metrics, **y_result.quality_metrics}
        
        return DataValidationResult(
            is_valid=len(combined_errors) == 0,
            errors=combined_errors,
            warnings=combined_warnings,
            quality_metrics=combined_metrics,
            validated_data=X_result.validated_data
        )
    
    def get_validation_summary(self, result: DataValidationResult) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            'is_valid': result.is_valid,
            'error_count': len(result.errors),
            'warning_count': len(result.warnings),
            'quality_score': result.quality_metrics.get('data_quality_score', 0.0),
            'data_shape': result.validated_data.shape if result.validated_data is not None else None,
            'has_cleaned_data': result.validated_data is not None
        }